# Data needed to calibrate and evaluate the model

**This is optional future work. The current 24-hour weather ensemble runs without SCADA.** It obtains wind inputs from Open-Meteo and uses the supplied power curve, so missing measured ambient temperature or wind speed does not prevent forecasting.

You reported historical coverage from **29 July 2022 to 4 November 2026**. No generation/SCADA history was attached here. The first useful future data is actual hourly active power with turbine IDs and timestamps; measured wind and temperature can support further diagnostics and corrections.

Aim for 6–12 months, preferably a full year covering the site's seasons. The code's minimum history checks only establish that its temporal folds can run; they do not establish statistical adequacy.

## Required hourly generation labels

Create `data/scada.csv` with this schema (a header-only template is included):

| Column | Meaning |
|---|---|
| `valid_time_utc` | Beginning of the target hour, ISO 8601 UTC, e.g. `2026-01-01T00:00:00Z` |
| `turbine_id` | Map marker / SCADA mapping, `01` through `13` |
| `power_kw` | Actual mean active power during that hour, in kW |
| `observed_at_utc` | When that hourly label was actually available to the forecasting system, no earlier than its valid time |

Do not use turbine labels for a farm export meter. The grid connection limit applies to the whole farm, not to a separate 3.6 MW turbine meter. Obtain the farm's export meter separately for a true export-forecast backtest.

For 10-minute SCADA, first confirm whether timestamps label interval starts or ends. Convert from the source timezone to UTC, assign samples to the correct hours, and compute time-weighted hourly mean power. Record completeness and reject incomplete hours instead of filling missing power with zero. An hourly energy measurement in kWh divided by one hour gives hourly mean kW. Labels stamped at the end of an hour must be shifted to the matching hour-start key. Preserve the label's true availability time.

Keep valid zero output during downtime and curtailment if the target is actual production. Invalid sensor values, missing data, or status codes are not automatically zero generation. The implemented training stage quarantines missing/non-finite, negative, and greater-than-105%-of-rating targets; review these thresholds with the operator. Small negative auxiliary consumption would require a separate net-meter target definition.

## Required historical forecast inputs

Retain each run's `forecast_features.csv`, `run_metadata.json`, and its raw weather snapshot. Each row has `issue_time_utc`, `valid_time_utc`, and `turbine_id`. Multiple past issues may predict the same target hour. Keep the forecasts actually available at each origin; never replace them later with a newer forecast.

The latest Open-Meteo responses used by this package do not identify a single model initialization. `issue_time_utc` is the package's retrieval completion time. A historical reconstruction must use identified model runs and conservative publication/ingestion delays to establish when inputs were available. The current public-weather API cannot recreate a historical live snapshot merely by changing its dates.

The [member endpoint](https://open-meteo.com/en/docs/ensemble-api) documents only three days of past ensemble-member data. Keep your own snapshots for exact future member-feature training. The [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) documents deterministic run history from March 2024 for ECMWF IFS HRES and April 2026 for most other models. Do not assume all required forecasts or member spreads can be recovered for the entire 2022–2026 SCADA period. Historical observations or reanalysis can aid exploratory analysis, but cannot establish as-issued forecast skill. A separate deterministic calibration model is an option where only deterministic forecast archives exist.

The ML feature set contains forecast power/wind distribution statistics, circular direction, lead time, calendar cycles, turbine/location metadata, and known operating constraints. It excludes target-time measured wind, active power and future operating states. Measured nacelle wind, wind direction, pitch, RPM, alarms and curtailment flags are useful for diagnostics and data quality. They may only become predictive inputs if lagged to values available at the forecast origin.

## Optional known operating plan

Use `--plan data/plan.csv` with columns:

| Column | Meaning |
|---|---|
| `valid_time_utc` | Target hour start, UTC |
| `turbine_id` | `01`–`13` |
| `availability` | Fraction 0–1, known or planned at issue time |
| `power_limit_kw` | Known turbine power cap, 0–3,600 kW |
| `known_at_utc` | When this plan entry became available; must be at/before forecast issue |

Unspecified hours use turbine defaults. Entries with future knowledge are rejected. This version supports per-turbine schedules; the farm export limit is static in `config.json`.

## First real-data evaluation

1. Confirm the SCADA IDs, units, interval convention, hub height, turbine mode and current export limit.
2. Join labels with archived forecast inputs using valid time and turbine, retaining forecast issue/lead time.
3. Use increasing chronological training windows, with overlapping forecast horizons and late labels purged before each validation origin. Keep the final period untouched until model selection is complete.
4. Compare MAE, RMSE, capacity-normalized MAE/RMSE and bias by lead time, turbine, season, direction and operating regime. Compare persistence on exactly the rows where its origin-time observation is available.
5. Assess Q10/Q50/Q90 with pinball loss and interval coverage before calling them calibrated. Use time-block uncertainty estimates for skill comparisons; neighbouring turbines and repeated targets are dependent.

The package implements chronological stacking, holdout point metrics, horizon breakdowns and fold audits. Seasonal/directional analysis, time-block confidence intervals, probabilistic calibration, farm-meter evaluation and an LSTM benchmark remain additional research steps once the relevant data is supplied.
