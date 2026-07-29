# CEA Irish population and smoking input export, version 2.0.0

## Purpose

This package transfers the eligibility project's **final processed estimates** to the CEA. It covers exact ages **50-79 inclusive**; age 80 is excluded.

Run from the repository root:

```bash
python scripts/export_cea_population_inputs.py
```

## Smoking status

Every synthetic person must receive exactly one of:

* `current`
* `former`
* `never`

The source is `data_processed/cleaned_smoking_data_ag.csv`. Its `smokers` estimate already combines daily and occasional smoking. The processed `smokers`, `quitters`, and `never_smokers` proportions use the original Census denominator and therefore leave a residual for not stated. The export renormalises these three values within each age-sex group so `assignment_probability` sums to one. No `ever_smoker`, daily/occasional, or not-stated model category is exported.

## Smoking histories

`smoking_history_strata.csv` copies the final age-sex pack-year estimates from `data_processed/pack_year_dist_cleaned.csv` without recalculation:

* current smokers: `smokers`
* former smokers: `quitters_all`, `quitters_excl_10`, and `quitters_excl_15`
* never smokers: structural zero pack-years

The default former-smoker history is `quitters_all`; the other reported variants are retained for eligibility definitions involving cessation windows. The CEA must use the exact matching sex and five-year age group. There is no age pooling, minimum-cell fallback, donor sampling, raw-data cleaning, or cross-sex substitution in this exporter.

These processed outputs provide adjusted **pack-year distributions**. They do not provide respondent-level linked values for initiation age, cigarettes per day, smoking duration, stopping age, or years since quitting.

## Files

* `irish_population_age_sex_2022.csv`: exact-age/sex population counts, ages 50-79.
* `irish_smoking_status_age_sex_2022.csv`: three-status source and assignment probabilities by age group and sex.
* `smoking_history_strata.csv`: final processed pack-year estimates by age group, sex, status, and history variant.
* `smoking_history_generation_parameters.json`: CEA assignment rules.
* `eligibility_model_validation_targets.csv`: age 55-74 status and default-history targets.
* `source_metadata.json`: provenance, transformations, limitations, row counts, and checksums.

The former raw-history cleaning audit files and donor-oriented generation files are no longer part of this package.
