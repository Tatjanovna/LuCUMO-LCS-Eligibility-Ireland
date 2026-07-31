#!/usr/bin/env python3
"""Export Census status inputs and smooth Eurobarometer smoking histories for the CEA."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,subprocess
from pathlib import Path
import export_cea_population_inputs as old

ROOT,RAW,PROCESSED=old.ROOT,old.RAW,old.PROCESSED
OUT=old.OUT; VERSION="4.0.0"
MODEL_FILES=("plco_smoking_history_parameters.csv","plco_smoking_history_correlations.csv","plco_smoking_history_validation_targets.csv","plco_smoking_history_synthetic_pool.csv","smoking_history_model_diagnostics.csv","smoking_history_cleaning_audit.csv","smoking_history_residual_donor_audit.csv","smoking_history_model_specification.json")

def require_sources():
    files=(RAW/"projections2057_raw.csv",PROCESSED/"projections2057.csv",PROCESSED/"cleaned_smoking_data_ag.csv",PROCESSED/"pack_year_dist_cleaned.csv",*(PROCESSED/f for f in MODEL_FILES))
    missing=[str(x.relative_to(ROOT)) for x in files if not x.exists()]
    if missing: raise FileNotFoundError("Missing sources: "+", ".join(missing)+". Run scripts/build_smooth_smoking_history_outputs.py")

def validate():
    required={
      "plco_smoking_history_parameters.csv":{"age_group","sex","smoking_status","history_variable","n","mean","sd","p01","p05","p25","p50","p75","p95","p99","source_n_exact_age_group","source_n_sex_status_model","synthetic_n"},
      "plco_smoking_history_correlations.csv":{"age_group","sex","smoking_status","variable_1","variable_2","n_complete","correlation","joint_generation_method"},
      "plco_smoking_history_synthetic_pool.csv":{"attained_age","age_group","sex","smoking_status","age_at_initiation","age_at_stopping","cigarettes_per_day","smoking_duration","years_since_quitting","standard_pack_years"},
      "smoking_history_model_diagnostics.csv":{"smoking_status","history_variable","source_n","mean_penalty_lambda"},
      "smoking_history_cleaning_audit.csv":{"sex","smoking_status","reason","n_records"},
      "smoking_history_residual_donor_audit.csv":{"sex","smoking_status","donor_source","n_complete_residual_vectors"}}
    counts={}
    for name,cols in required.items():
      with (PROCESSED/name).open(newline="",encoding="utf-8") as h:
        r=csv.DictReader(h); miss=cols-set(r.fieldnames or []); rows=list(r)
      if miss: raise ValueError(f"{name} missing {sorted(miss)}")
      if not rows: raise ValueError(f"{name} is empty")
      counts[name]=len(rows)
    spec=json.loads((PROCESSED/"smoking_history_model_specification.json").read_text())
    if spec.get("model_family")!="non_bayesian_robust_penalised_spline_location_scale": raise ValueError("Unexpected model specification")
    counts["smoking_history_model_specification.json"]=None
    return counts

def generation_parameters():
    return {"model_version":VERSION,"population_year":2022,"population_age_range":{"minimum":50,"maximum":79,"inclusive":True},"smoking_statuses":["current","former","never"],"status_assignment":{"source":"data_processed/cleaned_smoking_data_ag.csv","role":"Irish Census determines current/former/never prevalence by age and sex","not_stated_handling":"renormalise known statuses"},"conditional_history_assignment":{"preferred_source":"data_processed/plco_smoking_history_synthetic_pool.csv","matching_keys":["exact_age","sex","smoking_status"],"method":"sample one complete synthetic predictive row","model":"robust penalised continuous-age spline location-scale regressions","joint_method":"complete standardised residual-vector resampling","chronology":"bounded initiation and quitting-fraction transformations","respondent_rows_exported":False},"compatibility_outputs":{"parameters":"plco_smoking_history_parameters.csv","correlations":"plco_smoking_history_correlations.csv","note":"derived from the smooth pool; complete-row sampling is preferred"},"constraints":["5 <= initiation age < attained age","former stopping age is between initiation and attained age","0 < cigarettes/day <= 80","standard pack-years <= 120","adjusted pack-years <= 60"]}

def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def metadata(details,pop_validation):
    commit=git("log","-1","--format=%H","--","data_raw/projections2057_raw.csv","data_raw/eurobarometer.dta","data_processed/cleaned_smoking_data_ag.csv","scripts/build_smooth_smoking_history_outputs.py")
    return {"package_version":VERSION,"extraction_date":git("show","-s","--format=%cs",commit),"source_repository":"Tatjanovna/LuCUMO-LCS-Eligibility-Ireland","source_commit_sha":commit,"population_year":2022,"population_age_range":{"minimum":50,"maximum":79,"inclusive":True},"smoking_statuses":["current","former","never"],"source_separation":{"age_sex_population":"Irish Census-based population inputs","smoking_status_by_age_sex":"Irish Census smoking-status outputs","smoking_histories_conditional_on_age_sex_status":"Eurobarometer 2017 smooth model"},"preferred_history_input":"plco_smoking_history_synthetic_pool.csv","transformations":["Census smoking probabilities retained and renormalised within age-sex cells","Eurobarometer histories cleaned without global mean imputation","continuous-age robust penalised spline location and scale models fitted","complete residual vectors resampled to preserve dependence","chronology constrained during prediction","no respondent rows or identifiers exported"],"population_grouped_validation":pop_validation,"exports":details}

def readme():
    return """# CEA population and smoking inputs, version 4.0.0

Irish Census-based files determine age, sex, and current/former/never smoking status. Eurobarometer 2017 is used only to estimate smoking histories conditional on exact age, sex, and current/former status.

Run:

```bash
python scripts/build_smooth_smoking_history_outputs.py
python scripts/export_cea_population_inputs_v4.py
```

The history model removes impossible, incomplete, duplicate, and robustly detected outlying histories; fits robust penalised spline location-scale models over continuous age; and resamples complete standardised residual vectors. The preferred CEA input is `plco_smoking_history_synthetic_pool.csv`: match exact age, sex, and smoking status, then sample one complete row. The parameter and correlation files are compatibility summaries derived from the same pool.

No Eurobarometer respondent rows or identifiers are exported.
"""

def main():
    global OUT
    p=argparse.ArgumentParser(); p.add_argument("--output-directory",type=Path,default=Path("export_for_cea")); a=p.parse_args()
    if a.output_directory.is_absolute(): raise ValueError("output directory must be repository-relative")
    OUT=(ROOT/a.output_directory).resolve()
    if ROOT.resolve() not in OUT.parents: raise ValueError("output must remain inside repository")
    require_sources(); OUT.mkdir(exist_ok=True)
    for x in OUT.iterdir():
      if x.is_file(): x.unlink()
    population,pop_validation=old.population_export(); smoking=old.smoking_status_export(); legacy=old.smoking_history_export(); validation=old.validation_targets(smoking,legacy); counts=validate()
    old.write_csv(OUT/"irish_population_age_sex_2022.csv",list(population[0]),population); old.write_csv(OUT/"irish_smoking_status_age_sex_2022.csv",list(smoking[0]),smoking); old.write_csv(OUT/"smoking_history_strata.csv",list(legacy[0]),legacy); old.write_csv(OUT/"eligibility_model_validation_targets.csv",list(validation[0]),validation)
    for f in MODEL_FILES: shutil.copyfile(PROCESSED/f,OUT/f)
    (OUT/"smoking_history_generation_parameters.json").write_text(json.dumps(generation_parameters(),indent=2,sort_keys=True)+"\n"); (OUT/"README.md").write_text(readme())
    sources={"irish_population_age_sex_2022.csv":["data_raw/projections2057_raw.csv"],"irish_smoking_status_age_sex_2022.csv":["data_processed/cleaned_smoking_data_ag.csv"],"smoking_history_strata.csv":["data_processed/pack_year_dist_cleaned.csv"],"eligibility_model_validation_targets.csv":["data_processed/cleaned_smoking_data_ag.csv"],"smoking_history_generation_parameters.json":["data_processed/smoking_history_model_specification.json"],"README.md":[]}
    for f in MODEL_FILES:sources[f]=[f"data_processed/{f}"]
    details={}
    for f,src in sources.items():
      path=OUT/f; n=None
      if path.suffix==".csv":
        with path.open(newline="",encoding="utf-8") as h:n=sum(1 for _ in csv.DictReader(h))
      details[f]={"original_source_files":src,"row_count":n,"sha256":digest(path)}
    (OUT/"source_metadata.json").write_text(json.dumps(metadata(details,pop_validation),indent=2,sort_keys=True)+"\n")
    print(f"CEA export {VERSION}: {len(population)} population rows; "+", ".join(f"{k}={v}" for k,v in counts.items()))

if __name__=="__main__":main()
