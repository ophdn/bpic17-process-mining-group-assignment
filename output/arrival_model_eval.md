# Arrival model evaluation — Simulation 1.2

Real vs simulated case arrivals, first 90 days anchored at 2016-01-01, seed=42.
Real arrivals: 6965. Parametric: 7513. MDN: 6606.

| Metric | Parametric (LogNormal, Basic) | MDN (Advanced) |
|---|---:|---:|
| Hour-of-day MAE (24 bins, lower=better) | 0.0295 | 0.0036 |
| Weekday MAE (7 bins, lower=better) | 0.0344 | 0.0029 |
| Inter-arrival KS statistic (lower=better) | 0.0870 | 0.0154 |
| Inter-arrival KS p-value | 3.18e-24 | 3.89e-01 |

## Conclusion

MDN beats the parametric model on
hour-of-day shape, beats it on
weekday shape, and beats it on the
inter-arrival KS distance.

Recommendation: flip `USE_MDN_ARRIVALS = True` in `simulation/main.py` as the team default -- coordinate with the team before changing the default, since
it changes every teammate's simulated event logs.
