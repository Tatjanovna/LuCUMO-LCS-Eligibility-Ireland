import csv,hashlib,json,math,subprocess,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"export_for_cea"
BUILD=ROOT/"scripts"/"build_smooth_smoking_history_outputs.py"; EXPORT=ROOT/"scripts"/"export_cea_population_inputs_v4.py"
GROUPS={"50-54","55-59","60-64","65-69","70-74","75-79"}
MODEL_FILES={"plco_smoking_history_parameters.csv","plco_smoking_history_correlations.csv","plco_smoking_history_validation_targets.csv","plco_smoking_history_synthetic_pool.csv","smoking_history_model_diagnostics.csv","smoking_history_cleaning_audit.csv","smoking_history_residual_donor_audit.csv","smoking_history_model_specification.json"}

def run(path): subprocess.run([sys.executable,str(path)],cwd=ROOT,check=True,capture_output=True,text=True)
def rows(name,directory=OUT):
    with (directory/name).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def setup_module(): run(BUILD); run(EXPORT)

def test_census_population_and_status_are_preserved():
    pop=rows("irish_population_age_sex_2022.csv"); assert {int(r["age"]) for r in pop}==set(range(50,80))
    by_age=defaultdict(set)
    for r in pop: by_age[int(r["age"])].add(r["sex"]); assert int(r["population_count"])>=0
    assert all(v=={"female","male"} for v in by_age.values())
    status=rows("irish_smoking_status_age_sex_2022.csv"); g=defaultdict(list)
    for r in status: g[(r["age_group"],r["sex"])].append(r); assert r["source_dataset"]=="data_processed/cleaned_smoking_data_ag.csv"
    assert {k[0] for k in g}==GROUPS
    for v in g.values():
        assert {r["smoking_status"] for r in v}=={"current","former","never"}
        assert abs(sum(float(r["assignment_probability"]) for r in v)-1)<2e-11

def test_model_specification_has_intended_source_flow():
    s=json.loads((OUT/"smoking_history_model_specification.json").read_text())
    assert s["model_family"]=="non_bayesian_robust_penalised_spline_location_scale"
    assert s["population_role"]["smoking_status_prevalence"].startswith("Irish Census")
    assert s["population_role"]["conditional_smoking_histories"]=="Eurobarometer 2017"
    assert s["cleaning"]["global_mean_imputation"] is False
    assert s["joint_generation"]["method"]=="resample complete standardised residual vectors"
    assert s["synthetic_pool"]["draws_per_exact_age_sex_status_cell"]==250

def test_cleaning_and_donor_audits_are_aggregate():
    a=rows("smoking_history_cleaning_audit.csv"); assert a and "retained_for_modelling" in {r["reason"] for r in a}
    d=rows("smoking_history_residual_donor_audit.csv")
    assert {(r["sex"],r["smoking_status"]) for r in d}=={(s,t) for s in ("female","male") for t in ("current","former")}
    assert all(int(r["n_complete_residual_vectors"])>0 for r in d)
    forbidden={"uniqid","respondent_id","source_row","name","email","address","phone"}
    for f in ("smoking_history_cleaning_audit.csv","smoking_history_residual_donor_audit.csv"):
        with (OUT/f).open(newline="",encoding="utf-8") as h: assert forbidden.isdisjoint(csv.DictReader(h).fieldnames or [])

def test_synthetic_pool_exact_coverage_and_constraints():
    data=rows("plco_smoking_history_synthetic_pool.csv"); counts=defaultdict(int)
    for r in data:
        age=int(r["attained_age"]); sex=r["sex"]; status=r["smoking_status"]; counts[(age,sex,status)]+=1
        start=float(r["age_at_initiation"]); cigs=float(r["cigarettes_per_day"]); dur=float(r["smoking_duration"]); quit=float(r["years_since_quitting"]); py=float(r["standard_pack_years"]); adj=float(r["adjusted_pack_years"])
        assert 50<=age<=79 and sex in {"female","male"} and status in {"current","former"}
        assert 5<=start<age and 0<cigs<=80+1e-9 and dur>0 and py<=120+1e-8 and adj<=60+1e-8
        assert math.isclose(py,dur*cigs/20,rel_tol=1e-9,abs_tol=1e-9)
        if status=="current": assert r["age_at_stopping"]=="" and math.isclose(dur,age-start,abs_tol=1e-9) and quit==0
        else:
            stop=float(r["age_at_stopping"]); assert start<stop<=age
            assert math.isclose(dur,stop-start,abs_tol=1e-9) and math.isclose(quit,age-stop,abs_tol=1e-9)
    assert set(counts)=={(a,s,t) for a in range(50,80) for s in ("female","male") for t in ("current","former")}
    assert set(counts.values())=={250}

def test_predictive_parameters_replace_sparse_cell_empirical_summaries():
    data=rows("plco_smoking_history_parameters.csv"); cells=defaultdict(set)
    for r in data:
        cells[(r["age_group"],r["sex"],r["smoking_status"])].add(r["history_variable"])
        assert int(r["n"])==1250 and int(r["synthetic_n"])==1250 and int(r["source_n_sex_status_model"])>0
        q=[float(r[x]) for x in ("minimum","p01","p05","p25","p50","p75","p95","p99","maximum")]; assert q==sorted(q)
        assert r["source_processing"]=="robust_penalised_spline_with_complete_residual_vector_resampling"
        if not (r["smoking_status"]=="current" and r["history_variable"]=="years_since_quitting"): assert float(r["sd"])>0
        if r["history_variable"]=="standard_pack_years": assert float(r["maximum"])<=120+1e-8
    assert set(cells)=={(g,s,t) for g in GROUPS for s in ("female","male") for t in ("current","former")}

def test_correlations_and_model_diagnostics():
    for r in rows("plco_smoking_history_correlations.csv"):
        assert int(r["n_complete"])==1250 and r["joint_generation_method"]=="complete_standardised_residual_vector_resampling"
        if r["correlation_estimable"]=="true": assert -1<=float(r["correlation"])<=1
    diag=rows("smoking_history_model_diagnostics.csv")
    assert {(r["smoking_status"],r["history_variable"]) for r in diag}=={("current","start_t"),("current","cigs_t"),("former","start_t"),("former","cigs_t"),("former","quit_t")}
    assert all(int(r["source_n"])>0 and float(r["mean_penalty_lambda"])>0 for r in diag)

def test_model_outputs_are_copied_and_reproducible():
    for f in MODEL_FILES: assert (ROOT/"data_processed"/f).read_bytes()==(OUT/f).read_bytes()
    before_p={f:(ROOT/"data_processed"/f).read_bytes() for f in MODEL_FILES}; before_e={p.name:p.read_bytes() for p in OUT.iterdir() if p.is_file()}
    run(BUILD); run(EXPORT)
    assert before_p=={f:(ROOT/"data_processed"/f).read_bytes() for f in MODEL_FILES}
    assert before_e=={p.name:p.read_bytes() for p in OUT.iterdir() if p.is_file()}

def test_metadata_and_no_identifiers():
    forbidden={"uniqid","respondent_id","source_row","survey","name","email","address","phone"}
    for f in MODEL_FILES:
        p=OUT/f
        if p.suffix==".csv":
            with p.open(newline="",encoding="utf-8") as h: assert forbidden.isdisjoint(csv.DictReader(h).fieldnames or [])
    m=json.loads((OUT/"source_metadata.json").read_text()); assert m["package_version"]=="4.0.0" and m["preferred_history_input"]=="plco_smoking_history_synthetic_pool.csv"
    assert m["source_separation"]["smoking_status_by_age_sex"].startswith("Irish Census")
    commit=m["source_commit_sha"]; assert len(commit)==40; subprocess.run(["git","cat-file","-e",f"{commit}^{{commit}}"],cwd=ROOT,check=True)
    for f,d in m["exports"].items(): assert d["sha256"]==hashlib.sha256((OUT/f).read_bytes()).hexdigest()
