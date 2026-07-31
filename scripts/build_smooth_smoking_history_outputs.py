#!/usr/bin/env python3
"""Build smooth, non-Bayesian Eurobarometer smoking-history outputs for the CEA."""
from __future__ import annotations

import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data_raw", ROOT / "data_processed"
AGES, SEXES, STATUSES = range(50, 80), ("female", "male"), ("current", "former")
GROUPS = tuple(f"{a}-{a+4}" for a in range(50, 80, 5))
STATUS = {"You currently smoke": "current", "You used to smoke but you have stopped": "former"}
VARS = {"current": ("start_t", "cigs_t"), "former": ("start_t", "cigs_t", "quit_t")}
SUMMARY = {"current": ("age_at_initiation", "cigarettes_per_day", "smoking_duration", "years_since_quitting", "standard_pack_years", "adjusted_pack_years"), "former": ("age_at_initiation", "age_at_stopping", "cigarettes_per_day", "smoking_duration", "years_since_quitting", "standard_pack_years", "adjusted_pack_years")}
CORE = {"current": ("age_at_initiation", "cigarettes_per_day", "smoking_duration"), "former": ("age_at_initiation", "age_at_stopping", "cigarettes_per_day", "smoking_duration", "years_since_quitting")}
ROLE = {"age_at_initiation":"derived_history_input","age_at_stopping":"chronology_input","cigarettes_per_day":"plcom2012_predictor","smoking_duration":"plcom2012_predictor","years_since_quitting":"plcom2012_predictor","standard_pack_years":"derived_validation_measure","adjusted_pack_years":"eligibility_validation_measure"}
SEED, DRAWS, MIN_START, MAX_CIGS, MAX_PY = 20260731, 250, 5.0, 80.0, 120.0
KNOTS, LAMBDAS, EPS = (55.,60.,65.,70.,75.), (.1,1.,10.,100.,1000.), 1e-5


def logit(x):
    x=np.clip(np.asarray(x,float),EPS,1-EPS); return np.log(x/(1-x))

def expit(x):
    x=np.asarray(x,float); return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))

def age_group(x):
    x=pd.Series(x).astype(int); lo=x//5*5; return lo.astype(str)+"-"+(lo+4).astype(str)

def design(age, sex):
    age=np.asarray(age,float); male=(np.asarray(sex,str)=="male").astype(float); x=(age-64.5)/10
    cols=[np.ones_like(x),x,x*x,x*x*x]
    cols += [np.maximum(x-(k-64.5)/10,0)**3 for k in KNOTS]
    cols += [male,male*x]
    return np.column_stack(cols)

def mad(x):
    x=np.asarray(x,float); m=np.median(x); return float(np.median(np.abs(x-m)))

def scale(x):
    s=1.4826*mad(x)
    return max(float(s if np.isfinite(s) and s>1e-6 else np.std(x,ddof=1) if len(x)>1 else 1e-6),1e-6)

def fit_ridge(X,y,lam):
    p=np.ones(X.shape[1]); p[0]=0; p[1]=.01; p[-2]=.1; p[-1]=.5; P=np.diag(p)
    w=np.ones(len(y)); b=np.zeros(X.shape[1])
    for _ in range(40):
        A=X.T@(X*w[:,None])+lam*P; z=X.T@(w*y)
        nb=np.linalg.pinv(A)@z; r=y-X@nb; s=scale(r); q=np.abs(r)/s; nw=np.where(q<=1.345,1,1.345/q)
        if np.max(np.abs(nb-b))<1e-9: b=nb; break
        b,w=nb,nw
    return b,scale(y-X@b)

def choose_lambda(age,sex,y):
    if len(y)<25:return 100.
    X=design(age,sex); order=np.lexsort((np.asarray(sex,str),np.asarray(age,float))); folds=np.empty(len(y),int); folds[order]=np.arange(len(y))%5
    scores=[]
    for lam in LAMBDAS:
        e=[]
        for f in range(5):
            tr=folds!=f; va=~tr
            if tr.sum()<=X.shape[1]:continue
            b,s=fit_ridge(X[tr],y[tr],lam); r=y[va]-X[va]@b; e.extend(np.minimum(r*r,25*s*s))
        scores.append((np.mean(e) if e else np.inf,lam))
    return min(scores)[1]

def fit_variable(age,sex,y):
    X=design(age,sex); lam=choose_lambda(age,sex,y); b,rs=fit_ridge(X,y,lam); r=y-X@b
    sb,_=fit_ridge(X,np.log(r*r+(.1*rs)**2),max(10.,lam))
    ps=np.sqrt(np.exp(np.clip(X@sb,-12,12))); lo,hi=np.quantile(y,[.005,.995]); spread=max(float(hi-lo),rs)
    return {"b":b,"sb":sb,"lam":lam,"rs":rs,"min_s":max(float(np.quantile(ps,.05)),.05*rs,1e-3),"lo":float(lo-.1*spread),"hi":float(hi+.1*spread)}

def predict(model,age,sex):
    X=design(age,sex); mu=X@model["b"]; s=np.maximum(np.sqrt(np.exp(np.clip(X@model["sb"],-12,12))),model["min_s"]); return mu,s


def clean_histories():
    d=pd.read_stata(RAW/"eurobarometer.dta").reset_index(drop=True); d["row"]=np.arange(len(d))
    for c in ("age_start","age_stop","age_years","cig_day_current","cig_day_past"): d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d[d.wave.eq(2017)&d.sm_status.isin(STATUS)&d.gender.isin(["Female","Male"])&d.age_years.between(50,79)].copy()
    d["smoking_status"]=d.sm_status.map(STATUS); d["sex"]=d.gender.str.lower(); d["attained_age"]=d.age_years; d["age_at_initiation"]=d.age_start; d["age_at_stopping"]=d.age_stop
    d["cigarettes_per_day"]=np.where(d.smoking_status.eq("current"),d.cig_day_current,d.cig_day_past)
    audit=[]
    def remove(mask,reason):
        nonlocal d
        for (s,t),g in d[mask].groupby(["sex","smoking_status"],dropna=False): audit.append({"sex":s,"smoking_status":t,"reason":reason,"n_records":len(g)})
        d=d[~mask].copy()
    remove(d.duplicated(["wave","gender","age_years","sm_status","age_start","age_stop","cig_day_current","cig_day_past"]),"duplicate_history")
    former=d.smoking_status.eq("former")
    remove(d[["attained_age","age_at_initiation","cigarettes_per_day"]].isna().any(axis=1)|(former&d.age_at_stopping.isna()),"missing_essential_history")
    former=d.smoking_status.eq("former")
    remove((d.age_at_initiation<MIN_START)|(d.age_at_initiation>=d.attained_age)|(former&(d.age_at_stopping<=d.age_at_initiation))|(former&(d.age_at_stopping>d.attained_age)),"invalid_chronology")
    remove((d.cigarettes_per_day<=0)|(d.cigarettes_per_day>MAX_CIGS),"invalid_or_implausible_cigarettes_per_day")
    former=d.smoking_status.eq("former")
    d["start_t"]=logit((d.age_at_initiation-MIN_START)/(d.attained_age-MIN_START)); d["cigs_t"]=np.log(d.cigarettes_per_day); d.loc[former,"quit_t"]=logit((d.loc[former,"age_at_stopping"]-d.loc[former,"age_at_initiation"])/(d.loc[former,"attained_age"]-d.loc[former,"age_at_initiation"]))
    out=pd.Series(False,index=d.index)
    for (s,t),g in d.groupby(["sex","smoking_status"]):
        for v in VARS[t]:
            x=g[v].astype(float); med=x.median(); md=mad(x)
            if len(x)<10 or md<=EPS:
                z=d[d.smoking_status.eq(t)][v].dropna().astype(float); med=z.median(); md=mad(z)
            if md>EPS: out.loc[g.index]|=(.67448975*(x-med)/md).abs()>4.5
    remove(out,"robust_multivariable_outlier")
    former=d.smoking_status.eq("former"); current=~former
    d.loc[current,"age_at_stopping"]=np.nan; d.loc[current,"smoking_duration"]=d.loc[current,"attained_age"]-d.loc[current,"age_at_initiation"]; d.loc[former,"smoking_duration"]=d.loc[former,"age_at_stopping"]-d.loc[former,"age_at_initiation"]
    d["years_since_quitting"]=0.; d.loc[former,"years_since_quitting"]=d.loc[former,"attained_age"]-d.loc[former,"age_at_stopping"]
    d["standard_pack_years"]=d.smoking_duration*d.cigarettes_per_day/20
    remove(d.standard_pack_years>MAX_PY,"implausible_standard_pack_years")
    d["adjusted_pack_years"]=(d.smoking_duration*d.cigarettes_per_day*(1-np.exp(-.1*d.smoking_duration))/20).clip(upper=60); d["age_group"]=age_group(d.attained_age)
    for s in SEXES:
        for t in STATUSES:
            n=int((d.sex.eq(s)&d.smoking_status.eq(t)).sum())
            if n<3: raise ValueError(f"Too few cleaned histories for {(s,t)}: {n}")
            audit.append({"sex":s,"smoking_status":t,"reason":"retained_for_modelling","n_records":n})
    return d.reset_index(drop=True),audit


def fit_models(d):
    models,resids,diag={}, {}, []
    for t in STATUSES:
        q=d[d.smoking_status.eq(t)].copy(); age=q.attained_age.to_numpy(float); sex=q.sex.to_numpy(str); models[t]={}; r=q[["sex"]].reset_index(drop=True)
        for v in VARS[t]:
            m=fit_variable(age,sex,q[v].to_numpy(float)); models[t][v]=m; mu,ss=predict(m,age,sex); r[v]=np.clip((q[v].to_numpy(float)-mu)/ss,-4,4)
            diag.append({"smoking_status":t,"history_variable":v,"source_n":len(q),"female_n":int(q.sex.eq("female").sum()),"male_n":int(q.sex.eq("male").sum()),"mean_penalty_lambda":m["lam"],"robust_residual_scale":m["rs"],"model":"robust_penalised_cubic_spline_location_scale"})
        resids[t]=r
    return models,resids,diag


def synthetic_pool(models,resids):
    rng=np.random.default_rng(SEED); rows=[]; donor=[]
    for t in STATUSES:
        for s in SEXES:
            same=resids[t][resids[t].sex.eq(s)][list(VARS[t])].dropna(); source="same_sex_status"
            if len(same)<10: same=resids[t][list(VARS[t])].dropna(); source="status_pooled_across_sex"
            R=same.to_numpy(float); donor.append({"sex":s,"smoking_status":t,"donor_source":source,"n_complete_residual_vectors":len(R)})
            for a in AGES:
                Z=R[rng.integers(0,len(R),DRAWS)]; age=np.full(DRAWS,a,float); sex=np.full(DRAWS,s,object); tr={}
                for j,v in enumerate(VARS[t]):
                    mu,ss=predict(models[t][v],age,sex); tr[v]=np.clip(mu+ss*Z[:,j],models[t][v]["lo"],models[t][v]["hi"])
                start=np.clip(MIN_START+expit(tr["start_t"])*(age-MIN_START),MIN_START,age-1); cigs=np.clip(np.exp(tr["cigs_t"]),1,MAX_CIGS)
                if t=="current": stop=np.full(DRAWS,np.nan); dur=age-start; quit=np.zeros(DRAWS)
                else: stop=np.clip(start+expit(tr["quit_t"])*(age-start),start+.5,age); dur=stop-start; quit=age-stop
                cigs=np.minimum(cigs,20*MAX_PY/np.maximum(dur,.5)); py=dur*cigs/20; adj=np.minimum(dur*cigs*(1-np.exp(-.1*dur))/20,60)
                for i in range(DRAWS): rows.append({"attained_age":a,"age_group":f"{a//5*5}-{a//5*5+4}","sex":s,"smoking_status":t,"synthetic_draw":i+1,"age_at_initiation":start[i],"age_at_stopping":stop[i],"cigarettes_per_day":cigs[i],"smoking_duration":dur[i],"years_since_quitting":quit[i],"standard_pack_years":py[i],"adjusted_pack_years":adj[i],"residual_donor_source":source,"source_dataset":"data_raw/eurobarometer.dta","source_processing":"robust_penalised_spline_with_complete_residual_vector_resampling"})
    return pd.DataFrame(rows),donor


def summary(x):
    x=pd.to_numeric(x,errors="coerce").dropna().astype(float); out={"n":len(x),"mean":x.mean(),"sd":x.std(ddof=1),"minimum":x.min(),"maximum":x.max()}
    for n,p in (("p01",.01),("p05",.05),("p25",.25),("p50",.5),("p75",.75),("p95",.95),("p99",.99)): out[n]=x.quantile(p)
    return out

def aggregate(pool,source):
    counts=source.groupby(["age_group","sex","smoking_status"]).size().to_dict(); total=source.groupby(["sex","smoking_status"]).size().to_dict(); params=[]; cors=[]
    for g in GROUPS:
      for s in SEXES:
       for t in STATUSES:
        q=pool[pool.age_group.eq(g)&pool.sex.eq(s)&pool.smoking_status.eq(t)]
        for v in SUMMARY[t]: params.append({"age_group":g,"sex":s,"smoking_status":t,"history_variable":v,**summary(q[v]),"source_n_exact_age_group":counts.get((g,s,t),0),"source_n_sex_status_model":total.get((s,t),0),"synthetic_n":len(q),"plco_role":ROLE[v],"distribution_recommendation":"structural_zero" if t=="current" and v=="years_since_quitting" else "synthetic_predictive_quantile","source_dataset":"data_raw/eurobarometer.dta","source_processing":"robust_penalised_spline_with_complete_residual_vector_resampling"})
        for a,b in itertools.combinations(CORE[t],2):
            z=q[[a,b]].dropna().astype(float); ok=len(z)>2 and z[a].std()>0 and z[b].std()>0
            cors.append({"age_group":g,"sex":s,"smoking_status":t,"variable_1":a,"variable_2":b,"n_complete":len(z),"correlation":z[a].corr(z[b]) if ok else "","correlation_estimable":str(ok).lower(),"joint_generation_method":"complete_standardised_residual_vector_resampling","source_dataset":"data_raw/eurobarometer.dta"})
    validation=[]
    for r in params:
        for stat in ("mean","sd","p50"): validation.append({"validation_population":"smooth_predictive_histories_ages_50_79","age_group":r["age_group"],"sex":r["sex"],"smoking_status":r["smoking_status"],"history_variable":r["history_variable"],"statistic":stat,"value":r[stat],"n":r["n"],"source_dataset":r["source_dataset"]})
    return pd.DataFrame(params),pd.DataFrame(cors),pd.DataFrame(validation)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-directory",type=Path,default=PROCESSED); a=p.parse_args(); out=a.output_directory if a.output_directory.is_absolute() else ROOT/a.output_directory; out.mkdir(parents=True,exist_ok=True)
    clean,audit=clean_histories(); models,resids,diag=fit_models(clean); pool,donor=synthetic_pool(models,resids); params,cors,val=aggregate(pool,clean)
    files={"plco_smoking_history_parameters.csv":params,"plco_smoking_history_correlations.csv":cors,"plco_smoking_history_validation_targets.csv":val,"plco_smoking_history_synthetic_pool.csv":pool,"smoking_history_model_diagnostics.csv":pd.DataFrame(diag),"smoking_history_cleaning_audit.csv":pd.DataFrame(audit),"smoking_history_residual_donor_audit.csv":pd.DataFrame(donor)}
    for name,df in files.items(): df.to_csv(out/name,index=False,lineterminator="\n")
    spec={"model_family":"non_bayesian_robust_penalised_spline_location_scale","population_role":{"smoking_status_prevalence":"Irish Census processed outputs","conditional_smoking_histories":"Eurobarometer 2017"},"cleaning":{"global_mean_imputation":False,"robust_outlier_limit":4.5,"maximum_cigarettes_per_day":MAX_CIGS,"maximum_standard_pack_years":MAX_PY,"retained_histories":len(clean)},"joint_generation":{"method":"resample complete standardised residual vectors","fallback":"same smoking status pooled across sex when fewer than 10 vectors"},"synthetic_pool":{"random_seed":SEED,"draws_per_exact_age_sex_status_cell":DRAWS,"respondent_rows_exported":False}}
    (out/"smoking_history_model_specification.json").write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    print(f"Smooth history model: retained {len(clean)} histories; generated {len(pool)} synthetic histories")

if __name__=="__main__": main()
