# CEA Irish population input export, version 1.1.0

## Purpose and regeneration

This compact package supplies inputs for a synthetic Irish population aged **50–80 inclusive**. From the repository root run:

```bash
python scripts/export_cea_population_inputs.py
```

The committed package uses the default maximum age of 80. To build a 50–79 package instead, use `python scripts/export_cea_population_inputs.py --maximum-age 79 --output-directory export_for_cea_50_79`. Every population, Census smoking-status, Eurobarometer history, strata, parameter, and metadata export follows the selected range; Lung Health Check validation intentionally remains 55–74.

The script is deterministic, uses repository-relative paths and the Python standard library, recreates the package, validates its inputs, and records SHA-256 checksums. No notebook execution is required.

## Files

* `irish_population_age_sex_2022.csv`: exact-age/sex 2022 counts from the M2 CSO projection series baseline.
* `irish_smoking_status_age_sex_2022.csv`: Census State counts and probabilities, retaining five response categories.
* `smoking_history_strata.csv`: disclosure-safe complete-case Eurobarometer summaries with an explicit fallback level.
* `smoking_history_generation_parameters.json`: constraints, hierarchy and empirical correlations for coherent joint generation.
* `eligibility_model_validation_targets.csv`: Census and ordinary-history targets for ages 55–74.
* `smoking_history_cleaning_audit.csv` and `smoking_history_cleaning_summary.json`: aggregate cleaning counts, thresholds and maxima; neither contains respondent records or identifiers.
* `source_metadata.json`: provenance, decisions, limitations, row counts and checksums.

`smoking_history_donors.csv` is intentionally omitted. Although identifiers could be removed, this repository does not document Eurobarometer respondent-level redistribution rights. Do not copy respondent microdata to another repository without confirming the applicable licence. The disclosure-safe strata and generation parameters may be copied.

## Population and validation coverage

The population export contains every exact age from 50 through 80 for both sexes. When the maximum is 80, `80-84` is retained in the smoking-status export only because it is the source band for synthetic people aged exactly 80; the CEA must not create ages 81–84 from it. A maximum of 79 exports complete five-year source bands only. Lung Health Check validation remains restricted to 55–74 for comparability with the pilot and earlier eligibility work.

The population source is the **2022 baseline of a Census-2022-based CSO projection series under M2**, not a direct Census single-year-age table. Exact-age totals are checked against the processed five-year source totals for complete exported bands.

## Smoking status

Daily and occasional smoking remain distinct in the Census export. `not_stated` remains a fifth category, is included in `probability_all_people`, is excluded from the known-status denominator, and is never silently redistributed. The Eurobarometer extract does not support a reliable daily/occasional distinction, so current history records are labelled `current_unspecified`.

## Smoking history

The authoritative history source is `data_raw/eurobarometer.dta`, the repository's curated Irish extract. Wave-2017 respondents aged 50–80 were screened, yielding 448 candidates and 179 valid histories. Records failing sex/status, finite-value, initiation (at least 5 and before attained age), intensity (greater than 0 and at most 80), former-smoker stopping-age, chronology, or 200-pack-year rules are excluded unchanged with one documented reason—never imputed, clipped, or winsorised. This is cleaning for aggregate calibration only, not complete-history sampling.

The appropriate Irish national survey-weight variable could not be identified confidently: the original file has multiple weight labels, the curated extract has no weight, and no codebook/mapping is stored here. Weights are therefore set conceptually to 1 and every summary is labelled `unweighted_equal_weights`; no weight is guessed.

Generate smoking variables jointly and enforce the JSON constraints. The exported marginal summaries do **not** by themselves constitute a joint model. Empirical correlations are supplied as calibration targets, and the fallback sequence is five-year/sex/status, broad-age/sex/status, sex/status, then status.

## Pack-years and exclusions

Ordinary, uncapped pack-years are:

```text
cigarettes_per_day * smoking_duration / 20
```

The earlier exponential intensity adjustment (constant 0.1) and 60-pack-year cap are removed. The package also excludes lognormal eligibility calculations, Markov projections, future attenuation, quit-window eligibility exclusions, and undocumented cross-sex/age substitutions.

## Limitations

The Irish Eurobarometer sample is modest, complete-case and unweighted. Small cells (`n < 10`) are suppressed and consumers must follow the fallback hierarchy. The package provides empirical calibration inputs and constraints, not a validated causal or multivariate risk model. Confirm data licensing and the target model's handling of `not_stated` before use.
