"""Physical conversion and scenario aggregation. No network access."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if not 0 <= cfg["electrical_loss_fraction"] < 1:
        raise ValueError("electrical_loss_fraction must be in [0,1).")
    if cfg["grid_export_limit_kw"] <= 0 or not np.isfinite(cfg["grid_export_limit_kw"]):
        raise ValueError("A positive, finite grid export limit is required.")
    weights = cfg["family_weights"]
    if set(weights) != {"gfs", "ecmwf"}:
        raise ValueError("This version requires both gfs and ecmwf families.")
    if any(not np.isfinite(x) or x <= 0 for x in weights.values()):
        raise ValueError("Family weights must be positive and finite.")
    if not 0 <= cfg["restart_ms"] < cfg["cut_out_ms"]:
        raise ValueError("restart_ms must be below cut_out_ms.")
    if not 0 <= cfg["shear_exponent"] <= 0.6:
        raise ValueError("shear_exponent outside supported range [0,0.6].")
    for key, values in cfg.get("directional_wind_multipliers", {}).items():
        if len(values) != 12 or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(f"{key}: provide 12 positive, finite wind multipliers.")
    return cfg


def read_turbines(path):
    t = pd.read_csv(path, dtype={"turbine_id": str})
    t["turbine_id"] = t.turbine_id.str.zfill(2)
    required = ["latitude", "longitude", "hub_height_m", "rotor_diameter_m",
                "rated_power_kw", "wind_multiplier", "availability"]
    if t.empty or t.turbine_id.duplicated().any() or t[required].isna().any().any():
        raise ValueError("Empty, duplicate, or incomplete turbine metadata.")
    if not np.isfinite(t[required].to_numpy(float)).all():
        raise ValueError("Turbine metadata must be finite.")
    if not (t.latitude.between(-90, 90).all() and t.longitude.between(-180, 180).all()):
        raise ValueError("Invalid coordinates.")
    if (t[["hub_height_m", "rotor_diameter_m", "rated_power_kw", "wind_multiplier"]] <= 0).any().any():
        raise ValueError("Heights, diameter, rating and wind multiplier must be positive.")
    if not t.availability.between(0, 1).all():
        raise ValueError("Availability must be in [0,1].")
    return t.sort_values("turbine_id").reset_index(drop=True)


@dataclass
class PowerCurve:
    wind: np.ndarray
    power: np.ndarray
    cut_in: float = 3.0
    cut_out: float = 25.0
    restart: float = 23.0

    @classmethod
    def read(cls, path, cfg):
        df = pd.read_csv(path).sort_values("wind_speed_ms")
        if df.wind_speed_ms.duplicated().any() or not np.isfinite(df[["wind_speed_ms", "power_kw"]]).all().all():
            raise ValueError("Power curve must contain unique, finite wind speeds and powers.")
        if (df.power_kw < 0).any() or df.wind_speed_ms.min() > cfg["cut_in_ms"] or df.wind_speed_ms.max() < cfg["cut_out_ms"]:
            raise ValueError("Invalid power curve or insufficient operating-range coverage.")
        # Exclude all post-cut-out points: never interpolate over the shutdown jump.
        df = df.loc[df.wind_speed_ms <= cfg["cut_out_ms"]]
        if not np.isclose(df.wind_speed_ms.max(), cfg["cut_out_ms"]):
            raise ValueError("A curve point exactly at the configured cut-out speed is required.")
        return cls(df.wind_speed_ms.to_numpy(), df.power_kw.to_numpy(),
                   cfg["cut_in_ms"], cfg["cut_out_ms"], cfg["restart_ms"])

    def evaluate(self, wind):
        v = np.asarray(wind, dtype=float)
        p = np.interp(v, self.wind, self.power)
        return np.where((v < self.cut_in) | (v > self.cut_out), 0.0, p)

    def trajectory(self, wind, hysteresis=True):
        """Workbook convention: 25.0 operates, >25 shuts down; restart <=23.

        Initial controller state is assumed running. This is an hourly surrogate,
        not a model of gust-triggered shutdown, restart delays, or high-wind options.
        """
        v = np.asarray(wind, float)
        p = self.evaluate(v)
        if hysteresis:
            stopped = False
            for i, speed in enumerate(v):
                if not np.isfinite(speed):
                    continue
                if speed > self.cut_out:
                    stopped = True
                elif speed <= self.restart:
                    stopped = False
                if stopped:
                    p[i] = 0.0
        return p


def weighted_quantile(values, weights, q):
    """Inverse weighted empirical CDF; Q10 is the lower 10th percentile."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Quantiles require finite values and positive weights.")
    idx = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[idx]) / weights.sum()
    return float(values[idx[min(np.searchsorted(cdf, q, side="left"), len(idx)-1)]])


def summarize(scenarios, group_keys, value_col, family_weights):
    rows = []
    for key, g in scenarios.groupby(group_keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        families = sorted(g.family.unique())
        if set(families) != set(family_weights):
            raise ValueError("Missing required weather family; no silent degraded forecast.")
        counts = g.groupby("family").size().to_dict()
        w = np.array([family_weights[f] / counts[f] for f in g.family], float)
        w /= w.sum()
        v = g[value_col].to_numpy(float)
        if not np.isfinite(v).all():
            raise ValueError("Missing scenario power cannot be treated as zero.")
        mean = float(w @ v)
        row = dict(zip(group_keys, key))
        row.update(mean_power_kw=mean, q10_kw=weighted_quantile(v, w, .1),
                   q50_kw=weighted_quantile(v, w, .5), q90_kw=weighted_quantile(v, w, .9),
                   ensemble_std_kw=float(np.sqrt(w @ ((v-mean)**2))),
                   n_scenarios=len(g), n_families=len(families))
        for f, fg in g.groupby("family"):
            row[f"{f}_mean_kw"] = float(fg[value_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def apply_operations(weather, turbines, cfg, curve, issue_time, plan_path=None):
    d = weather.merge(turbines, on="turbine_id", validate="many_to_one")
    if len(d) != len(weather):
        raise ValueError("Weather contains an unknown turbine.")
    d["power_limit_kw"] = d.rated_power_kw
    if plan_path:
        plan = pd.read_csv(plan_path, dtype={"turbine_id": str})
        plan["turbine_id"] = plan.turbine_id.str.zfill(2)
        for col in ["valid_time_utc", "known_at_utc"]:
            plan[col] = pd.to_datetime(plan[col], utc=True, errors="raise")
        if plan[["valid_time_utc", "known_at_utc", "turbine_id"]].isna().any().any():
            raise ValueError("Operational plan timestamps and turbine IDs cannot be missing.")
        if plan[["valid_time_utc", "turbine_id"]].duplicated().any():
            raise ValueError("Duplicate operational plan keys.")
        if (plan.known_at_utc > issue_time).any():
            raise ValueError("Operational plan contains information unavailable at forecast issue.")
        if not set(plan.turbine_id) <= set(turbines.turbine_id):
            raise ValueError("Plan contains an unknown turbine.")
        if not plan.availability.between(0, 1).all() or not np.isfinite(plan.power_limit_kw).all() or (plan.power_limit_kw < 0).any():
            raise ValueError("Invalid operational plan values.")
        ratings = plan.turbine_id.map(turbines.set_index("turbine_id").rated_power_kw)
        if (plan.power_limit_kw > ratings).any():
            raise ValueError("Operational plan power limit exceeds the turbine rating; check units.")
        d = d.merge(plan[["valid_time_utc", "turbine_id", "availability", "power_limit_kw"]],
                    on=["valid_time_utc", "turbine_id"], how="left", suffixes=("", "_plan"), validate="many_to_one")
        for col in ["availability", "power_limit_kw"]:
            d[col] = d.pop(col+"_plan").fillna(d[col])
    d["power_limit_kw"] = np.minimum(d.power_limit_kw, d.rated_power_kw)
    d["wind_hub_ms"] = d.wind_speed_ref_ms * (d.hub_height_m / d.reference_height_m)**cfg["shear_exponent"]
    d["wind_hub_ms"] *= d.wind_multiplier
    for turbine_id, values in cfg.get("directional_wind_multipliers", {}).items():
        mask = d.turbine_id.eq(str(turbine_id).zfill(2))
        sectors = np.floor(((d.loc[mask, "wind_direction_deg"] + 15) % 360)/30).astype(int)
        d.loc[mask, "wind_hub_ms"] *= np.asarray(values)[sectors]
    d = d.sort_values(["family", "member", "turbine_id", "valid_time_utc"]).reset_index(drop=True)
    d["potential_power_kw"] = 0.0
    for _, g in d.groupby(["family", "member", "turbine_id"], sort=False):
        d.loc[g.index, "potential_power_kw"] = np.minimum(
            curve.trajectory(g.wind_hub_ms, cfg["high_wind_hysteresis"]), g.rated_power_kw)
    d["power_kw"] = np.minimum(d.potential_power_kw*d.availability, d.power_limit_kw)
    return d


def aggregate_farm(d, turbines, cfg):
    keys = ["issue_time_utc", "valid_time_utc", "family", "member"]
    count = d.groupby(keys).turbine_id.nunique()
    if not count.eq(len(turbines)).all() or d.duplicated(keys+["turbine_id"]).any():
        raise ValueError("A farm scenario requires exactly one value for every turbine.")
    farm = d.groupby(keys, as_index=False).agg(
        gross_power_kw=("potential_power_kw", "sum"), available_power_kw=("power_kw", "sum"))
    # Cap each spatially coherent member after summing turbines, before expectation.
    farm["export_power_kw"] = np.minimum(
        farm.available_power_kw*(1-cfg["electrical_loss_fraction"]), cfg["grid_export_limit_kw"])
    return farm


def make_features(d, turbines, cfg):
    """Only forecast/metadata inputs; no target-time measured wind or SCADA."""
    rows = []
    for (issue, valid, tid), g in d.groupby(["issue_time_utc", "valid_time_utc", "turbine_id"]):
        counts = g.groupby("family").size().to_dict()
        w = np.array([cfg["family_weights"][f]/counts[f] for f in g.family]); w /= w.sum()
        v, p = g.wind_hub_ms.to_numpy(), g.power_kw.to_numpy()
        direction = np.deg2rad(g.wind_direction_deg.to_numpy())
        meanv, meanp = float(w @ v), float(w @ p)
        local = valid.tz_convert(cfg["display_timezone"])
        r = {"issue_time_utc":issue, "valid_time_utc":valid, "turbine_id":tid,
             "lead_hours":(valid-issue).total_seconds()/3600,
             "physics_mean_kw":meanp, "physics_std_kw":float(np.sqrt(w @ (p-meanp)**2)),
             "wind_mean_ms":meanv, "wind_std_ms":float(np.sqrt(w @ (v-meanv)**2)),
             "direction_sin":float(w @ np.sin(direction)), "direction_cos":float(w @ np.cos(direction)),
             "hour_sin":np.sin(2*np.pi*local.hour/24), "hour_cos":np.cos(2*np.pi*local.hour/24),
             "year_sin":np.sin(2*np.pi*local.dayofyear/365.25), "year_cos":np.cos(2*np.pi*local.dayofyear/365.25),
             "availability":g.availability.iloc[0], "power_limit_kw":g.power_limit_kw.iloc[0],
             "rated_power_kw":g.rated_power_kw.iloc[0]}
        for f, fg in g.groupby("family"):
            r[f+"_power_kw"] = fg.power_kw.mean()
        rows.append(r)
    return pd.DataFrame(rows).merge(turbines[["turbine_id", "latitude", "longitude", "hub_height_m"]],
                                     on="turbine_id", validate="many_to_one")
