# Validation record

The included SCADA-free model was run against live Open-Meteo responses for all 13 map coordinates, then reproduced from the saved responses without network access. The snapshot contains 31 GFS ensemble members, 51 ECMWF members and one JMA MSM comparison trajectory, each over 24 hourly timestamps. Both required ensemble families had complete data; none of their members were dropped.

Within each source, all 13 locations resolved to the same returned weather grid and had identical wind trajectories. This is recorded in `examples/forecast/run_metadata.json` and `weather_grid_audit.csv`.

The following **14 software tests passed** with `python -m unittest discover -s tests -v`:

1. Full residual-ensemble training, temporal holdout, save/refit and future prediction using synthetic fixtures.
2. Power-curve interpolation and explicit shutdown discontinuity.
3. High-wind restart hysteresis.
4. Converting individual wind members before averaging power.
5. Equal weather-family weighting despite unequal member counts.
6. Applying the farm grid cap to scenarios before averaging, and rejecting an incomplete farm scenario.
7. Rejecting missing required weather families.
8. Weighted quantiles retaining probability mass at zero.
9. Known outage plans and rejecting plans learned after the forecast origin.
10. Purging future targets and late-arriving labels from training folds.
11. Nonnegative stacking weights constrained to sum to one.
12. Persistence using only observations available at the forecast origin.
13. Complete spatially coherent ensemble membership in the saved snapshot.
14. Rejecting snapshot replay after turbine coordinates change.

The synthetic fixture test verifies implementation and serialization only. Its artificial accuracy scores are not included as plant results. No SCADA backtest, measured generation accuracy, empirical uncertainty coverage, or ensemble-versus-LSTM improvement has been established.

Tested package versions: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, Matplotlib 3.10.8, openpyxl 3.1.5. `requirements.txt` permits compatible versions; use `constraints-tested.txt` to request the tested versions. Serialized learned models should be loaded using the same scikit-learn version used for training.
