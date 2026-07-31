# Smooth smoking-history model

This implementation separates the two evidence sources used for population generation:

- Irish Census-based processed outputs determine population age, sex, and current/former/never smoking status.
- Eurobarometer 2017 determines smoking histories conditional on exact age, sex, and current/former status.

The Eurobarometer stage removes incomplete, duplicate, chronologically invalid, implausible, and robustly detected outlying histories. It then fits robust penalised cubic-spline location and scale models over continuous age for transformed initiation age, cigarettes per day, and former-smoker quitting. Complete standardised residual vectors are resampled within sex and smoking status so that dependence between history components is retained.

The preferred CEA input is `plco_smoking_history_synthetic_pool.csv`. The CEA should match exact age, sex, and smoking status and sample one complete row. The five-year parameter and correlation files are compatibility summaries derived from the same predictive pool.

Current implementation limits are 80 cigarettes per day and 120 standard pack-years. These are explicit modelling assumptions and should be included in sensitivity analysis if changed.
