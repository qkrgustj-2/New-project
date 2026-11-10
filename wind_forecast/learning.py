"""Leakage-aware residual stacking, activated only when real SCADA is supplied.

Models: physical curve mean + Random Forest + histogram gradient boosting +
Ridge. Residual models are pooled across turbines. Weights use purged,
chronological out-of-fold predictions. The final 20% of issue times is held out.
"""
import glob
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

NUMERIC = ["lead_hours", "physics_mean_kw", "physics_std_kw", "wind_mean_ms", "wind_std_ms",
           "direction_sin", "direction_cos", "hour_sin", "hour_cos", "year_sin", "year_cos",
           "availability", "power_limit_kw", "rated_power_kw", "gfs_power_kw", "ecmwf_power_kw",
           "latitude", "longitude", "hub_height_m"]
MODEL_NAMES = ["physics", "random_forest", "gradient_boosting", "ridge"]
KEYS = ["issue_time_utc", "valid_time_utc", "turbine_id"]


def read_features(expression):
    files = sorted(glob.glob(str(expression), recursive=True))
    if not files:
        raise ValueError("No forecast feature files matched the input expression.")
    d = pd.concat([pd.read_csv(f, dtype={"turbine_id":str}) for f in files], ignore_index=True)
    d["turbine_id"] = d.turbine_id.str.zfill(2)
    for col in KEYS[:2]: d[col] = pd.to_datetime(d[col], utc=True, errors="raise")
    if d[KEYS].isna().any().any() or d.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate forecast keys. Select each archived issue once.")
    if not np.isfinite(d[NUMERIC].to_numpy(float)).all():
        raise ValueError("Features contain missing/infinite values; repair upstream forecasts first.")
    actual_lead = (d.valid_time_utc-d.issue_time_utc).dt.total_seconds()/3600
    if not (actual_lead > 0).all() or not np.allclose(actual_lead, d.lead_hours):
        raise ValueError("Invalid lead hours or target time at/before forecast issue.")
    return d.sort_values(KEYS).reset_index(drop=True)


def read_scada(path):
    s = pd.read_csv(path, dtype={"turbine_id":str})
    s["power_kw"] = pd.to_numeric(s.power_kw, errors="coerce")
    s["turbine_id"] = s.turbine_id.str.zfill(2)
    for col in ["valid_time_utc", "observed_at_utc"]:
        s[col] = pd.to_datetime(s[col], utc=True, errors="raise")
    if s[["valid_time_utc", "observed_at_utc", "turbine_id"]].isna().any().any():
        raise ValueError("SCADA requires valid timestamps, turbine ID and observed_at_utc.")
    if s.duplicated(["valid_time_utc", "turbine_id"]).any():
        raise ValueError("SCADA must contain one hourly target per turbine; aggregate subhourly records first.")
    if (s.observed_at_utc < s.valid_time_utc).any():
        raise ValueError("Observation cannot be available before its valid time.")
    if not s.valid_time_utc.eq(s.valid_time_utc.dt.floor("h")).all():
        raise ValueError("SCADA targets must align to whole UTC hours.")
    return s.sort_values(["turbine_id", "valid_time_utc"])


def train_before(d, boundary):
    """Purge every target/observation unavailable at the next forecast origin."""
    return d.loc[(d.issue_time_utc < boundary) & (d.valid_time_utc < boundary)
                 & (d.observed_at_utc <= boundary)].copy()


def preprocessing():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    return ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("turbine", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["turbine_id"])])


def fit_bases(d):
    residual = (d.power_kw-d.physics_mean_kw)/d.rated_power_kw
    estimators = {
        "random_forest":RandomForestRegressor(n_estimators=160, min_samples_leaf=12,
                                              max_depth=14, max_features=.85, n_jobs=2, random_state=42),
        "gradient_boosting":HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15,
                                                          min_samples_leaf=20, l2_regularization=2,
                                                          early_stopping=False, random_state=42),
        "ridge":Ridge(alpha=20),
    }
    models = {}
    for name, estimator in estimators.items():
        model = Pipeline([("features", preprocessing()), ("regressor", estimator)])
        with threadpool_limits(limits=2):
            model.fit(d, residual)
        models[name] = model
    return models


def base_predictions(models, d):
    ceiling = np.minimum(d.rated_power_kw*d.availability, d.power_limit_kw).to_numpy()
    columns = [np.clip(d.physics_mean_kw, 0, ceiling)]
    for name in MODEL_NAMES[1:]:
        with threadpool_limits(limits=2):
            residual = models[name].predict(d)
        columns.append(np.clip(d.physics_mean_kw+residual*d.rated_power_kw, 0, ceiling))
    return np.column_stack(columns)


def fit_weights(predictions, actual, capacity):
    p = np.asarray(predictions)/np.asarray(capacity)[:,None]
    y = np.asarray(actual)/np.asarray(capacity)
    result = minimize(lambda w: np.mean((p@w-y)**2), np.ones(p.shape[1])/p.shape[1],
                      method="SLSQP", bounds=[(0,1)]*p.shape[1],
                      constraints={"type":"eq", "fun":lambda w:w.sum()-1},
                      options={"ftol":1e-12, "maxiter":1000})
    if not result.success:
        raise RuntimeError("Ensemble weight optimization failed: "+result.message)
    w = np.clip(result.x, 0, 1)
    return w/w.sum()


def point_metrics(actual, predicted, capacity):
    e = np.asarray(predicted)-np.asarray(actual)
    return {"n":len(e), "mae_kw":float(np.abs(e).mean()), "rmse_kw":float(np.sqrt(np.mean(e**2))),
            "bias_kw":float(e.mean()), "nmae_pct_capacity":float(np.mean(np.abs(e)/capacity)*100),
            "nrmse_pct_capacity":float(np.sqrt(np.mean((e/capacity)**2))*100)}


def persistence_at_issue(forecasts, scada):
    """Latest valid SCADA sample actually available at each origin, not a target lag."""
    mapping = {}
    for (issue, tid), _ in forecasts.groupby(["issue_time_utc", "turbine_id"]):
        eligible = scada.loc[scada.turbine_id.eq(tid) & (scada.valid_time_utc <= issue)
                             & (scada.observed_at_utc <= issue) & np.isfinite(scada.power_kw)
                             & (scada.power_kw >= 0)]
        mapping[(issue,tid)] = np.nan if eligible.empty else eligible.iloc[-1].power_kw
    return np.array([mapping[(r.issue_time_utc,r.turbine_id)] for r in forecasts.itertuples()])


def report_metrics(d, p):
    report = {}
    for i, name in enumerate(MODEL_NAMES+["stacked_ensemble"]):
        report[name] = point_metrics(d.power_kw, p[:,i], d.rated_power_kw.to_numpy())
    return report


def train(feature_expression, scada_path, destination, min_train_rows=300):
    import sklearn
    features, scada = read_features(feature_expression), read_scada(scada_path)
    d = features.merge(scada[["valid_time_utc", "turbine_id", "power_kw", "observed_at_utc"]],
                       on=["valid_time_utc", "turbine_id"], validate="many_to_one")
    missing_scada = len(features)-len(d)
    invalid = ~np.isfinite(d.power_kw) | (d.power_kw < 0) | (d.power_kw > d.rated_power_kw*1.05)
    n_invalid = int(invalid.sum())
    d = d.loc[~invalid & (d.observed_at_utc <= pd.Timestamp.now(tz="UTC"))].copy()
    if d.empty:
        raise ValueError("No usable forecast/SCADA pairs. Check units, hourly alignment and observed_at_utc.")
    # Outages/curtailment with valid power (including zero) are retained: target is actual production.
    origins = sorted(d.issue_time_utc.unique())
    if len(origins) < 16:
        raise ValueError("Need at least 16 distinct archived issue times; months of history are recommended.")
    boundary = pd.Timestamp(origins[max(1, int(len(origins)*.8))])
    dev, test = train_before(d, boundary), d.loc[d.issue_time_utc >= boundary].copy()
    if len(dev) < min_train_rows or test.empty:
        raise ValueError("Not enough development/holdout data after horizon/label-availability purging.")
    dev_origins = np.array(sorted(dev.issue_time_utc.unique()))
    oof_parts, folds = [], []
    for fold, (_, val_indices) in enumerate(TimeSeriesSplit(n_splits=4).split(dev_origins), 1):
        val_origins = dev_origins[val_indices]
        cutoff = pd.Timestamp(val_origins[0])
        fit = train_before(dev, cutoff)
        val = dev.loc[dev.issue_time_utc.isin(val_origins)].copy()
        if len(fit) < min_train_rows:
            folds.append({"fold":fold, "status":"skipped_insufficient_purged_history", "n_train":len(fit)})
            continue
        models = fit_bases(fit)
        p = base_predictions(models, val)
        o = val[KEYS+["power_kw", "rated_power_kw"]].copy()
        for i, name in enumerate(MODEL_NAMES): o[name] = p[:,i]
        oof_parts.append(o)
        folds.append({"fold":fold, "status":"used", "n_train":len(fit), "n_validation":len(val),
                      "validation_first_issue_utc":cutoff.isoformat(),
                      "max_training_valid_time_utc":fit.valid_time_utc.max().isoformat(),
                      "max_training_label_available_utc":fit.observed_at_utc.max().isoformat()})
    if len(oof_parts) < 2:
        raise ValueError("Need at least two usable purged folds; collect more archived origins/history.")
    oof = pd.concat(oof_parts, ignore_index=True)
    weights = fit_weights(oof[MODEL_NAMES], oof.power_kw, oof.rated_power_kw)
    evaluation_models = fit_bases(dev)
    p = base_predictions(evaluation_models, test)
    predictions = np.column_stack([p, p@weights])
    metrics = report_metrics(test, predictions)
    persist = persistence_at_issue(test, scada)
    mask = np.isfinite(persist) & (persist <= test.rated_power_kw.to_numpy()*1.05)
    if mask.any():
        metrics["persistence"] = point_metrics(test.power_kw.to_numpy()[mask], persist[mask], test.rated_power_kw.to_numpy()[mask])
        metrics["persistence"]["comparison_note"] = "If persistence has missing rows, compare all methods on the matched subset."
        metrics["stacked_ensemble_on_persistence_subset"] = point_metrics(test.power_kw.to_numpy()[mask], predictions[mask,-1], test.rated_power_kw.to_numpy()[mask])
        metrics["physics_on_persistence_subset"] = point_metrics(test.power_kw.to_numpy()[mask], predictions[mask,0], test.rated_power_kw.to_numpy()[mask])
    breakdown = []
    buckets = pd.cut(test.lead_hours, bins=[0,24,48,72,168], labels=["0–24h", "24–48h", "48–72h", "72–168h"])
    for label in buckets.dropna().unique():
        m = buckets.eq(label).to_numpy()
        for i, name in enumerate(MODEL_NAMES+["stacked_ensemble"]):
            breakdown.append(dict(lead_bucket=str(label), model=name,
                                  **point_metrics(test.power_kw.to_numpy()[m], predictions[m,i], test.rated_power_kw.to_numpy()[m])))
    dest = Path(destination); dest.mkdir(parents=True, exist_ok=True)
    holdout = test[KEYS+["lead_hours", "power_kw", "rated_power_kw"]].copy()
    for i, name in enumerate(MODEL_NAMES+["stacked_ensemble"]): holdout[name+"_kw"] = predictions[:,i]
    holdout["persistence_kw"] = persist
    holdout.to_csv(dest/"holdout_predictions.csv", index=False)
    oof.to_csv(dest/"development_oof_predictions.csv", index=False)
    pd.DataFrame(breakdown).to_csv(dest/"holdout_metrics_by_horizon.csv", index=False)
    eval_artifact = {"models":evaluation_models, "weights":weights, "model_names":MODEL_NAMES,
                     "training_cutoff_utc":boundary.isoformat(), "sklearn_version":sklearn.__version__,
                     "turbine_ids":sorted(d.turbine_id.unique())}
    with (dest/"evaluation_model.pkl").open("wb") as f: pickle.dump(eval_artifact, f)
    # Refit after recording the untouched holdout evaluation; never report this refit's in-sample skill.
    production = dict(eval_artifact, models=fit_bases(d),
                      training_cutoff_utc=d.observed_at_utc.max().isoformat(),
                      refitted_after_holdout=True)
    with (dest/"model.pkl").open("wb") as f: pickle.dump(production, f)
    report = {"status":"trained_on_user_supplied_scada", "weights":dict(zip(MODEL_NAMES, weights.tolist())),
              "holdout_first_issue_utc":boundary.isoformat(), "development_rows":len(dev),
              "holdout_rows":len(test), "quarantined_invalid_target_pairs":n_invalid,
              "forecast_rows_without_scada":missing_scada,
              "folds":folds, "holdout_metrics":metrics, "scikit_learn_version":sklearn.__version__,
              "interpretation":["Weights learned from purged development folds only; holdout never used for weight tuning.",
                                "Report evaluates evaluation_model.pkl. model.pkl is subsequently refitted on all available labels.",
                                "Repeated origins and turbines share targets/weather; row counts are not independent sample counts.",
                                "Raw weather Q10–Q90 is not calibrated by this point-model training.",
                                "No claim that stacking must beat a single model; inspect holdout by horizon and season."]}
    (dest/"training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"weights":report["weights"], "holdout_metrics":metrics}, indent=2))
    return report


def predict_saved(model_path, feature_path, output):
    # Only load models you trained or otherwise trust: pickle can execute code.
    with Path(model_path).open("rb") as f: artifact = pickle.load(f)
    d = read_features(feature_path)
    if not set(d.turbine_id) <= set(artifact["turbine_ids"]):
        raise ValueError("Forecast includes turbines absent from model training; retrain with their historical labels.")
    if (d.issue_time_utc <= pd.Timestamp(artifact["training_cutoff_utc"])).any():
        raise ValueError("Prediction issue must follow model training cutoff; use holdout_predictions.csv for past validation.")
    predictions = base_predictions(artifact["models"], d)
    result = d[KEYS+["rated_power_kw"]].copy()
    result["ml_ensemble_power_kw"] = predictions@artifact["weights"]
    for i, name in enumerate(MODEL_NAMES): result[name+"_kw"] = predictions[:,i]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} calibrated turbine point forecasts. Raw weather quantiles remain separate.")
    return result
