# CEA population and smoking inputs, version 4.0.0

Irish Census-based files determine age, sex, and current/former/never smoking status. Eurobarometer 2017 is used only to estimate smoking histories conditional on exact age, sex, and current/former status.

Run:

```bash
python scripts/build_smooth_smoking_history_outputs.py
python scripts/export_cea_population_inputs_v4.py
```

The history model removes impossible, incomplete, duplicate, and robustly detected outlying histories; fits robust penalised spline location-scale models over continuous age; and resamples complete standardised residual vectors. The preferred CEA input is `plco_smoking_history_synthetic_pool.csv`: match exact age, sex, and smoking status, then sample one complete row. The parameter and correlation files are compatibility summaries derived from the same pool.

No Eurobarometer respondent rows or identifiers are exported.
