# CEA Irish population and smoking input export, version 3.0.0

## Purpose and regeneration

This package transfers the eligibility project's final processed estimates to the CEA. It covers exact ages **50-79 inclusive**; age 80 is excluded.

Run from the repository root:

```bash
python scripts/build_plco_smoking_history_outputs.py
python scripts/export_cea_population_inputs.py
```

The first command reproduces the final smoking-history transformations reported in `code/4. pack_year_distribution_baseline.ipynb` and writes aggregate, non-identifying PLCO-ready processed files. The second command copies only processed estimates into `export_for_cea`.

## Smoking status

Every synthetic person must receive exactly one of `current`, `former`, or `never`. The processed `smokers` category already combines daily and occasional smoking. The three probabilities are renormalised within each age-sex cell; no ever-smoker or not-stated model category is exported.

## PLCO-ready smoking histories

The package now includes age-sex-status-specific aggregate distributions for:

* age at smoking initiation;
* cigarettes per day;
* smoking duration;
* age at stopping for former smokers;
* years since quitting;
* standard and eligibility-adjusted pack-years.

`plco_smoking_history_correlations.csv` supplies the empirical within-cell correlations needed for coherent joint generation. The recommended method is a Gaussian copula with the exported empirical marginals, followed by chronology checks and redraws. The CEA must use the exact matching sex, five-year age group, and current/former status; there is no pooling or fallback.

`effective_cigarettes_per_day` and `adjusted_pack_years` reproduce the eligibility analysis. They are exported for eligibility calibration and validation and must not silently replace the published PLCOm2012 cigarettes-per-day predictor.

## Files

* `irish_population_age_sex_2022.csv`: exact-age/sex population counts, ages 50-79.
* `irish_smoking_status_age_sex_2022.csv`: three-status assignment probabilities.
* `smoking_history_strata.csv`: final adjusted pack-year estimates and cessation variants.
* `plco_smoking_history_parameters.csv`: empirical PLCO-variable marginals by exact age-sex-status cell.
* `plco_smoking_history_correlations.csv`: empirical within-cell correlations for joint generation.
* `plco_smoking_history_validation_targets.csv`: means, SDs, and medians for CEA validation.
* `smoking_history_generation_parameters.json`: assignment and joint-generation rules.
* `eligibility_model_validation_targets.csv`: Irish LHC status and adjusted pack-year targets.
* `source_metadata.json`: provenance, limitations, row counts, and checksums.

No respondent-level smoking histories or identifiers are exported.
