# CEA Irish population and smoking inputs, version 4.1.0

This directory is the complete runtime input package for CEA population generation.

- `irish_population_age_sex_2022.csv` supplies exact-age and sex population counts from Irish Census-based inputs.
- `irish_smoking_status_age_sex_2022.csv` supplies current, former, and never smoking probabilities by five-year age group and sex.
- `plco_smoking_history_synthetic_pool.csv` supplies complete smoking histories conditional on exact age, sex, and current/former status.
- `smoking_history_generation_parameters.json` records source roles, matching rules, assumptions, constraints, and reproducibility settings.
- `source_metadata.json` records provenance, checksums, and population validation.

For current and former smokers, match `attained_age`, `sex`, and `smoking_status`, then sample one complete row. Never smokers receive structural-zero smoking histories.

Regenerate from the repository root:

```bash
python scripts/build_smooth_smoking_history_outputs.py
python scripts/export_cea_population_inputs.py
```

Diagnostics and cleaning audits remain in `data_processed`; they are not runtime CEA inputs. Legacy pack-year strata, marginal distributions, correlations, and validation-only files are not exported.
