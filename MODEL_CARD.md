# Model card — Nakasato ensemble forecasting baseline

**Status:** a runnable, SCADA-free, 24-hour physical weather ensemble, exercised with live API inputs. No measured wind, ambient temperature or historical generation is required to run it. It has not been calibrated or backtested against site generation, so a current forecast is not evidence of predictive accuracy.

## Site and input audit

The My Maps KML contains 13 point markers, `01`–`13`, near 41°N, 140.46°E. Coordinates were retrieved directly, rather than copied from the conversation preview. Maximum marker separation is about 3.011 km; nearest-neighbour distances are approximately 205–538 m. At a 117 m rotor diameter, these are about 1.8–4.6 rotor diameters. Geometry alone cannot establish wake losses.

The workbook has 522 records: 261 per density (1.225 and 1.223), speeds 0–26 in 0.1 increments, and maximum power 3,600. All power values match across the two density tables. The selected 1.225 curve is sorted and stored separately. Units are inferred as m/s, kW and kg/m³ because the original headers do not specify them.

Configuration uses 116.5 m hub height, 3.6 MW per turbine, 117 m rotor diameter, and a 36 MW farm export limit. The site owner reports [46.8 MW installed and 36 MW grid capacity](https://venagroup.com/vena-energy-secures-jpy14-5b-shinsei-green-loan-for-the-construction-of-the-nakasato-wind-project-in-aomori-prefecture/) and [116.5 m hubs](https://project.venaenergy.co.jp/information/1078). The generic [Vestas V117-3.45 MW specification](https://www.vestas.com/en/energy-solutions/onshore-wind-turbines/4-mw-platform/V117-3-45-MW) identifies a 3.45 MW base configuration, 3 m/s cut-in, 25 m/s cut-out and 23 m/s restart. The uploaded curve's 3.6 MW maximum is retained instead of truncating it to the generic 3.45 MW rating. Confirm the current site configuration.

## Weather ensemble and physical model

Default sources are Open-Meteo `gfs_seamless` and `ecmwf_ifs025` on its [Ensemble API](https://open-meteo.com/en/docs/ensemble-api), using 100 m wind speed/direction. The included snapshot has 31 and 51 complete members respectively. A trial of ICON's 120 m field returned null values and was excluded. JMA MSM 10 m winds are a separate benchmark with a larger extrapolation to hub height.

For turbine i, family f, member m and time t:

```
v_hub = v_reference × (hub_height/reference_height)^alpha
        × turbine_wind_multiplier × optional_direction_sector_multiplier
p_potential = supplied_power_curve(v_hub), with cut-in/out and optional restart state
p_turbine = min(p_potential × availability, known_turbine_power_limit)
p_export_scenario = min((sum_i p_turbine) × (1-electrical_loss_fraction), 36000)
weight(f,m) = family_weight(f) / number_of_complete_members(f)
forecast_mean = sum_f,m weight(f,m) × p_export_scenario(f,m)
```

Weights are normalized; defaults are 0.5 per family. Alpha defaults to 0.14, an assumption rather than a site estimate. The 100→116.5 m adjustment is modest; the JMA 10→116.5 m adjustment is substantially more sensitive to alpha and atmospheric stability. Fixed density and no fitted local corrections are the initial baseline.

The same member ID is preserved across turbine locations and times within a family. All turbines must be present for a farm scenario. Incomplete ensemble trajectories are dropped and counted; a required family with fewer than two complete members fails the run. API errors are not silently converted to zero, and old cached forecasts are not silently mixed with live ones.

Average power after conversion: `E[P(v)]` generally differs from `P(E[v])`. Similarly, cap each farm scenario before averaging; `E[min(P,36 MW)]` generally differs from `min(E[P],36 MW)`. Farm quantiles come from summed joint scenarios, not sums of turbine quantiles. The larger ECMWF member count does not increase its total family weight.

## Curve boundaries and controller approximation

Linear interpolation applies only inside the operating curve. The workbook has zero below 3 m/s, 22 kW at 3 m/s, 3,600 kW at 25 m/s, and zero at 25.1 m/s. Therefore this implementation treats **25.0 as operating and >25 as stopped**, without interpolating across that jump. An optional stateful hourly approximation keeps a turbine stopped until wind is <=23 m/s. The initial state is assumed running. Actual control settings, gust shutdowns, restart delays and high-wind options can differ.

The raw curve gives 1,715 kW at 8.3 m/s and 1,781 at 8.4, so interpolation at 8.35 gives 1,748 kW. Automated tests cover this and the discontinuity.

## Optional learned residual stack

The provided training code fits Random Forest, histogram gradient boosting and Ridge corrections to the physical mean, using only forecast/metadata features. The stack includes the physical mean itself, allowing learned weights to favour it if the data supports that choice. Corrections use normalized power residuals; final turbine point outputs respect known availability and power limits.

Four expanding chronological development blocks generate out-of-fold predictions. Before each block, all training rows whose target or observation availability reaches beyond the next forecast origin are purged. Nonnegative weights summing to one minimize squared normalized error on these development predictions. The last 20% of issue times is held out and supplies no training or weight tuning information. After evaluation, a separate production model is refitted using available labels; the evaluated model is retained for reproducibility.

This is a two-stage design: meteorological ensembles provide weather scenarios immediately; supervised stacking becomes active once historical SCADA and forecasts exist. LSTM is a possible later comparator, not a prerequisite. An ensemble's superiority must be demonstrated on held-out site data, rather than assumed from architecture alone. See [Pinson and Madsen's ensemble-to-power forecasting study](https://doi.org/10.1002/we.309) for research context; it is not validation of this implementation.

## Interpretation and remaining limitations

Q10/Q50/Q90 are conventional lower/median/upper CDF quantiles. Q10–Q90 describes raw weather spread, not validated 80% observed coverage and not an energy-finance exceedance convention. A mean can lie outside the central interval when most scenarios have zero power and a few have positive power. These intervals omit uncertainty from local model bias, outages, icing, unmodelled losses and the curve itself.

The included APIs return identical wind trajectories across the 13 locations within each source. Defaults therefore produce identical turbine predictions. Actual local wind, wakes and terrain can differ. Calibrate turbine/direction effects with SCADA instead of manufacturing differences from grid forecasts.

Hourly wind is instantaneous; mapping it through a nonlinear turbine curve approximates hourly mean power. Hourly and horizon energy values are labelled **energy proxies** and assume each power sample represents one hour. Real hourly generation requires calibration or finer-resolution inputs. The full-horizon energy quantiles sum member trajectories before summarizing; summing hourly quantiles would not give those quantiles.

Forecast `issue_time_utc` means retrieval completion, not meteorological initialization. All stored keys are UTC; the chart and convenience column use Asia/Tokyo. The included snapshot is dated and should be refreshed for a new forecast. The user's reported historical period is 29 July 2022 to 4 November 2026, with SCADA files unavailable here. The [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) can support deterministic historical comparisons, but is not a guaranteed archive of this model's individual ensemble members. Exact historical member features require saved member forecasts. Historical reconstruction is optional future work, not a dependency for the current forecast.

Known losses/operations are configurable. Defaults assume all turbines available, wind multipliers=1 and electrical losses=0. Current configuration, units, meter definitions, local corrections, forecast skill and probabilistic coverage need operator data before operational adoption.
