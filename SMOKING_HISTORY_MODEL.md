# Smooth smoking-history model

This implementation keeps the evidence sources separate:

- Irish Census-based processed outputs determine population age, sex, and current/former/never smoking status.
- Eurobarometer 2017 determines smoking histories conditional on exact age, sex, and current/former status.

The Eurobarometer stage removes incomplete, duplicate, chronologically invalid, implausible, and robustly detected outlying histories. It then fits robust penalised cubic-spline location and scale models over continuous age for transformed initiation age, cigarettes per day, and former-smoker quitting. Complete standardised residual vectors are resampled within sex and smoking status so that dependence between history components is retained.

The sole CEA smoking-history input is `plco_smoking_history_synthetic_pool.csv`. The CEA should match exact age, sex, and current/former status and sample one complete row. Five-year marginal, correlation, legacy pack-year-strata, and validation-only files are no longer exported.

Model diagnostics, the cleaning audit, and the residual-donor audit remain in `data_processed` for eligibility-model quality assurance. They are not runtime CEA inputs.

Current implementation limits are 80 cigarettes per day, 120 standard pack-years, and 60 adjusted pack-years. These are explicit modelling assumptions and should be included in sensitivity analysis if changed.
