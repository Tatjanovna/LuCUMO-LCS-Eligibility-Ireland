#!/usr/bin/env python3
"""Fit smooth Eurobarometer smoking histories and write CEA-ready outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data_raw"; PROCESSED=ROOT/"data_processed"
AGES=range(50,80); SEXES=("female","male"); STATUSES=("current","former")
STATUS={"You currently smoke":"current","You used to smoke but you have stopped":"former"}
VARS={"current":("start_t","cigs_t"),"former":("start_t","cigs_t","quit_t")}
SEED,DRAWS,MIN_START,MAX_CIGS,MAX_PY,MAX_ADJ=20260731,250,5.0,80.0,120.0,60.0
KNOTS,LAMBDAS,EPS=(55.,60.,65.,70.,75.),(.1,1.,10.,100.,1000.),1e-5
POOL="plco_smoking_history_synthetic_pool.csv"; PARAMETERS="smoking_history_generation_parameters.json"
OBSOLETE=("plco_smoking_history_parameters.csv","plco_smoking_history_correlations.csv","plco_smoking_history_validation_targets.csv","smoking_history_model_specification.json")


def logit(x):
    x=np.clip(np.asarray(x,float),EPS,1-EPS); return np.log(x/(1-x))

def expit(x):
    x=np.asarray(x,float); return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))

def age_group(x):
    x=pd.Series(x).astype(int); lo=x//5*5; return lo.astype(str)+"-"+(lo+4).astype(str)

def design(age,sex):
    age=np.asarray(age,float); male=(np.asarray(sex,str)=="male").astype(float); x=(age-64.5)/10
    cols=[np.ones_like(x),x,x*x,x*x*x]; cols += [np.maximum(x-(k-64.5)/10,0)**3 for k in KNOTS]; cols += [male,male*x]
    return np.column_stack(cols)

def mad(x):
    x=np.asarray(x,float); m=np.median(x); return float(np.median(np.abs(x-m)))

def scale(x):
    x=np.asarray(x,float); s=1.4826*mad(x)
    return max(float(s if np.isfinite(s) and s>1e-6 else np.std(x,ddof=1) if len(x)>1 else 1e-6),1e-6)

def fit_ridge(X,y,lam):
    p=np.ones(X.shape[1]); p[0]=0; p[1]=.01; p[-2]=.1; p[-1]=.5; P=np.diag(p)
    w=np.ones(len(y)); b=np.zeros(X.shape[1])
    for _ in range(40):
        A=X.T@(X*w[:,None])+lam*P; z=X.T@(w*y); nb=np.linalg.pinv(A)@z; r=y-X@nb; s=scale(r); q=np.abs(r)/s; nw=np.where(q<=1.345,1,1.345/q)
        if np.max(np.abs(nb-b))<1e-9: b=nb; break
        b,w=nb,nw
    return b,scale(y-X@b)

def choose_lambda(age,sex,y):
    if len(y)<25:return 100.
    X=design(age,sex); order=np.lexsort((np.asarray(sex,str),np.asarray(age,float))); folds=np.empty(len(y),int); folds[order]=np.arange(len(y))%5; scores=[]
    for lam in LAMBDAS:
        errors=[]
        for fold in range(5):
            tr=folds!=fold; va=~tr
            if tr.sum()<=X.shape[1]:continue
            b,s=fit_ridge(X[tr],y[tr],lam); r=y[va]-X[va]@b; errors.extend(np.minimum(r*r,25*s*s))
        scores.append((np.mean(errors) if errors else np.inf,lam))
    return min(scores)[1]

def fit_variable(age,sex,y):
    X=design(age,sex); lam=choose_lambda(age,sex,y); b,rs=fit_ridge(X,y,lam); r=y-X@b
    sb,_=fit_ridge(X,np.log(r*r+(.1*rs)**2),max(10.,lam)); ps=np.sqrt(np.exp(np.clip(X@sb,-12,12))); lo,hi=np.quantile(y,[.005,.995]); spread=max(float(hi-lo),rs)
    return {"b":b,"sb":sb,"lam":lam,"rs":rs,"min_s":max(float(np.quantile(ps,.05)),.05*rs,1e-3),"lo":float(lo-.1*spread),"hi":float(hi+.1*spread)}

def predict(model,age,sex):
    X=design(age,sex); mu=X@model["b"]; s=np.maximum(np.sqrt(np.exp(np.clip(X@model["sb"],-12,12))),model["min_s"]); return mu,s


def clean_histories():
    d=pd.read_stata(RAW/"eurobarometer.dta")
    for c in ("age_start","age_stop","age_years","cig_day_current","cig_day_past"): d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d[d.wave.eq(2017)&d.sm_status.isin(STATUS)&d.gender.isin(["Female","Male"])&d.age_years.between(50,79)].copy()
    d["smoking_status"]=d.sm_status.map(STATUS); d["sex"]=d.gender.str.lower(); d["attained_age"]=d.age_years; d["age_at_initiation"]=d.age_start; d["age_at_stopping"]=d.age_stop
    d["cigarettes_per_day"]=np.where(d.smoking_status.eq("current"),d.cig_day_current,d.cig_day_past); audit=[]
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
    former=d.smoking_status.eq("former"); d["start_t"]=logit((d.age_at_initiation-MIN_START)/(d.attained_age-MIN_START)); d["cigs_t"]=np.log(d.cigarettes_per_day)
    d.loc[former,"quit_t"]=logit((d.loc[former,"age_at_stopping"]-d.loc[former,"age_at_initiation"])/(d.loc[former,"attained_age"]-d.loc[former,"age_at_initiation"]))
    out=pd.Series(False,index=d.index)
    for (_,status),g in d.groupby(["sex","smoking_status"]):
        for v in VARS[status]:
            x=g[v].astype(float); med=x.median(); deviation=mad(x)
            if len(x)<10 or deviation<=EPS:
                pooled=d[d.smoking_status.eq(status)][v].dropna().astype(float); med=pooled.median(); deviation=mad(pooled)
            if deviation>EPS: out.loc[g.index]|=(.67448975*(x-med)/deviation).abs()>4.5
    remove(out,"robust_multivariable_outlier")
    former=d.smoking_status.eq("former"); current=~former; d.loc[current,"age_at_stopping"]=np.nan
    d.loc[current,"smoking_duration"]=d.loc[current,"attained_age"]-d.loc[current,"age_at_initiation"]; d.loc[former,"smoking_duration"]=d.loc[former,"age_at_stopping"]-d.loc[former,"age_at_initiation"]
    d["years_since_quitting"]=0.; d.loc[former,"years_since_quitting"]=d.loc[former,"attained_age"]-d.loc[former,"age_at_stopping"]
    d["standard_pack_years"]=d.smoking_duration*d.cigarettes_per_day/20; remove(d.standard_pack_years>MAX_PY,"implausible_standard_pack_years")
    d["adjusted_pack_years"]=(d.smoking_duration*d.cigarettes_per_day*(1-np.exp(-.1*d.smoking_duration))/20).clip(upper=MAX_ADJ); d["age_group"]=age_group(d.attained_age)
    for s in SEXES:
        for t in STATUSES:
            n=int((d.sex.eq(s)&d.smoking_status.eq(t)).sum())
            if n<3: raise ValueError(f"Too few cleaned histories for {(s,t)}: {n}")
            audit.append({"sex":s,"smoking_status":t,"reason":"retained_for_modelling","n_records":n})
    return d.reset_index(drop=True),audit


def fit_models(d):
    models,resids,diag={}, {}, []
    for t in STATUSES:
        q=d[d.smoking_status.eq(t)].copy(); age=q.attained_age.to_numpy(float); sex=q.sex.to_numpy(str); models[t]={}; residual=q[["sex"]].reset_index(drop=True)
        for v in VARS[t]:
            m=fit_variable(age,sex,q[v].to_numpy(float)); models[t][v]=m; mu,ss=predict(m,age,sex); residual[v]=np.clip((q[v].to_numpy(float)-mu)/ss,-4,4)
            diag.append({"smoking_status":t,"history_variable":v,"source_n":len(q),"female_n":int(q.sex.eq("female").sum()),"male_n":int(q.sex.eq("male").sum()),"mean_penalty_lambda":m["lam"],"robust_residual_scale":m["rs"],"model":"robust_penalised_cubic_spline_location_scale"})
        resids[t]=residual
    return models,resids,diag


def synthetic_pool(models,resids):
    rng=np.random.default_rng(SEED); rows=[]; donor=[]
    for t in STATUSES:
        variables=list(VARS[t])
        for s in SEXES:
            same=resids[t][resids[t].sex.eq(s)][variables].dropna(); source="same_sex_status"
            if len(same)<10: same=resids[t][variables].dropna(); source="status_pooled_across_sex"
            R=same.to_numpy(float); donor.append({"sex":s,"smoking_status":t,"donor_source":source,"n_complete_residual_vectors":len(R)})
            for a in AGES:
                Z=R[rng.integers(0,len(R),DRAWS)]; age=np.full(DRAWS,a,float); sex=np.full(DRAWS,s,object); tr={}
                for j,v in enumerate(variables):
                    mu,ss=predict(models[t][v],age,sex); tr[v]=np.clip(mu+ss*Z[:,j],models[t][v]["lo"],models[t][v]["hi"])
                start=np.clip(MIN_START+expit(tr["start_t"])*(age-MIN_START),MIN_START,age-1); cigs=np.clip(np.exp(tr["cigs_t"]),1,MAX_CIGS)
                if t=="current": stop=np.full(DRAWS,np.nan); duration=age-start; quit_years=np.zeros(DRAWS)
                else: stop=np.clip(start+expit(tr["quit_t"])*(age-start),start+.5,age); duration=stop-start; quit_years=age-stop
                cigs=np.minimum(cigs,20*MAX_PY/np.maximum(duration,.5)); py=duration*cigs/20; adj=np.minimum(duration*cigs*(1-np.exp(-.1*duration))/20,MAX_ADJ)
                for i in range(DRAWS): rows.append({"attained_age":a,"age_group":f"{a//5*5}-{a//5*5+4}","sex":s,"smoking_status":t,"synthetic_draw":i+1,"age_at_initiation":start[i],"age_at_stopping":stop[i],"cigarettes_per_day":cigs[i],"smoking_duration":duration[i],"years_since_quitting":quit_years[i],"standard_pack_years":py[i],"adjusted_pack_years":adj[i],"residual_donor_source":source,"source_dataset":"data_raw/eurobarometer.dta","source_processing":"robust_penalised_spline_with_complete_residual_vector_resampling"})
    return pd.DataFrame(rows),donor


def generation_parameters(cleaned,donor):
    retained={f"{s}/{t}":int((cleaned.sex.eq(s)&cleaned.smoking_status.eq(t)).sum()) for s in SEXES for t in STATUSES}
    donors={f"{r['sex']}/{r['smoking_status']}":{"source":r["donor_source"],"n_complete_vectors":r["n_complete_residual_vectors"]} for r in donor}
    return {"model_version":"smooth_history_v1","model_family":"non_bayesian_robust_penalised_spline_location_scale","population_year":2022,"population_age_range":{"minimum":50,"maximum":79,"inclusive":True},"smoking_statuses":["current","former","never"],"source_roles":{"age_sex_population":"Irish Census-based population inputs","smoking_status_prevalence":"Irish Census processed outputs","conditional_smoking_histories":"Eurobarometer 2017"},"status_assignment":{"source":"irish_smoking_status_age_sex_2022.csv","matching_keys":["five_year_age_group","sex"],"method":"sample current, former, or never using assignment_probability","not_stated_handling":"renormalise current, former, and never"},"conditional_history_assignment":{"source":POOL,"matching_keys":["attained_age","sex","smoking_status"],"method":"sample one complete synthetic predictive row","current_years_since_quitting":0,"never_smoker_history":"structural_zero","respondent_rows_exported":False},"cleaning":{"global_mean_imputation":False,"robust_outlier_limit":4.5,"maximum_cigarettes_per_day":MAX_CIGS,"maximum_standard_pack_years":MAX_PY,"retained_histories":len(cleaned),"retained_histories_by_sex_status":retained},"joint_generation":{"method":"resample complete standardised residual vectors","fallback":"same smoking status pooled across sex when fewer than 10 vectors are available","donor_sources":donors},"synthetic_pool":{"random_seed":SEED,"draws_per_exact_age_sex_status_cell":DRAWS,"expected_rows":len(AGES)*len(SEXES)*len(STATUSES)*DRAWS},"constraints":["5 <= age_at_initiation < attained_age","current: age_at_stopping is missing and years_since_quitting = 0","former: age_at_initiation < age_at_stopping <= attained_age","0 < cigarettes_per_day <= 80","standard_pack_years <= 120","adjusted_pack_years <= 60"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-directory",type=Path,default=PROCESSED); args=parser.parse_args(); out=args.output_directory if args.output_directory.is_absolute() else ROOT/args.output_directory; out.mkdir(parents=True,exist_ok=True)
    for filename in OBSOLETE: (out/filename).unlink(missing_ok=True)
    clean,audit=clean_histories(); models,resids,diag=fit_models(clean); pool,donor=synthetic_pool(models,resids)
    pool.to_csv(out/POOL,index=False,lineterminator="\n"); pd.DataFrame(diag).to_csv(out/"smoking_history_model_diagnostics.csv",index=False,lineterminator="\n"); pd.DataFrame(audit).to_csv(out/"smoking_history_cleaning_audit.csv",index=False,lineterminator="\n"); pd.DataFrame(donor).to_csv(out/"smoking_history_residual_donor_audit.csv",index=False,lineterminator="\n")
    (out/PARAMETERS).write_text(json.dumps(generation_parameters(clean,donor),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"Smooth history model: retained {len(clean)} histories; generated {len(pool)} synthetic histories")

if __name__=="__main__": main()
