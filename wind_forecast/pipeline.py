"""Forecast outputs, auditable metadata, Excel workbook, and charts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .core import (PowerCurve, load_config, read_turbines, apply_operations,
                   aggregate_farm, summarize, make_features)
from .weather import read_snapshot


def csv_ready(df):
    d = df.copy()
    for col in d:
        if isinstance(d[col].dtype, pd.DatetimeTZDtype):
            d[col] = d[col].map(lambda x: x.isoformat())
    return d


def plot_forecast(farm, turbines, curve, cfg, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"nakasato-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"nakasato-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                         "axes.spines.right":False, "axes.titleweight":"bold", "font.size":10})
    fig = plt.figure(figsize=(13, 8.8), facecolor="#f4f7fa")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], hspace=.48, wspace=.23,
                          top=.87, bottom=.145, left=.07, right=.96)
    ax = fig.add_subplot(gs[0,:]); ax.set_facecolor("white")
    t = pd.to_datetime(farm.valid_time_utc, utc=True).dt.tz_convert(cfg["display_timezone"])
    ax.fill_between(t, farm.q10_kw/1000, farm.q90_kw/1000, color="#b4d9e4", alpha=.7,
                    label="Raw weather ensemble Q10–Q90")
    ax.plot(t, farm.mean_power_kw/1000, color="#087f8c", lw=2.6, label="Equal-family ensemble mean")
    for family, color in [("gfs", "#af6f2d"), ("ecmwf", "#7161a4")]:
        ax.plot(t, farm[family+"_mean_kw"]/1000, color=color, lw=1.15, alpha=.8, label=family.upper()+" mean")
    upper = max(1., min(cfg["grid_export_limit_kw"]/1000*1.08,
                        max(farm.q90_kw.max(), farm.mean_power_kw.max())/1000*1.12))
    if upper >= cfg["grid_export_limit_kw"]/1000:
        ax.axhline(cfg["grid_export_limit_kw"]/1000, color="#cc6677", ls="--", lw=1, label="Grid limit")
    ax.set_ylim(0, upper)
    ax.set_ylabel("Grid export estimate (MW)")
    ax.set_title("Hourly grid export • Asia/Tokyo", loc="left", pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=ZoneInfo(cfg["display_timezone"])))
    ax.grid(axis="y", alpha=.2); ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    site_ax = fig.add_subplot(gs[1,0]); site_ax.set_facecolor("white")
    lat0, lon0 = turbines.latitude.mean(), turbines.longitude.mean()
    x = (turbines.longitude-lon0)*111.32*np.cos(np.deg2rad(lat0))
    y = (turbines.latitude-lat0)*111.32
    site_ax.scatter(x, y, color="#087f8c", s=36)
    for xx, yy, tid in zip(x, y, turbines.turbine_id):
        site_ax.annotate(tid, (xx,yy), xytext=(5,4), textcoords="offset points", fontsize=8)
    site_ax.set_aspect("equal", adjustable="datalim"); site_ax.margins(.20)
    site_ax.set_xlabel("East of site centre (km)"); site_ax.set_ylabel("North (km)")
    site_ax.set_title("13 turbine locations", loc="left"); site_ax.grid(alpha=.15)
    ca = fig.add_subplot(gs[1,1]); ca.set_facecolor("white")
    v = np.r_[np.arange(0, 25.001, .05), 25.001, 26]
    ca.plot(v, curve.evaluate(v)/1000, color="#087f8c", lw=2)
    ca.set_xlabel("Hub-height wind (m/s)"); ca.set_ylabel("Turbine power (MW)")
    ca.set_title("Uploaded V117 reference curve", loc="left"); ca.grid(alpha=.15)
    issue = pd.Timestamp(farm.issue_time_utc.iloc[0]).tz_convert(cfg["display_timezone"])
    fig.text(.07, .95, "NAKASATO  /  WIND GENERATION FORECAST", color="#16324f", fontsize=18, weight="bold")
    fig.text(.07, .915, f"Snapshot issued {issue:%d %b %Y, %H:%M} JST  ·  46.8 MW installed  ·  36 MW export limit", color="#566b80", fontsize=10)
    fig.text(.07, .023, "Uncalibrated baseline. Q10–Q90 shows weather-member spread; 80% observed coverage has not been established.\nAll turbines available; no fitted wake, terrain, icing or electrical-loss correction. Hourly power is an instantaneous-wind approximation.",
             color="#566b80", fontsize=8.5)
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor()); plt.close(fig)


def write_workbook(tables, destination, chart_path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        for name, df in tables.items():
            csv_ready(df).to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book:
            sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
            for c in sheet[1]:
                c.fill = PatternFill("solid", fgColor="16324F")
                c.font = Font(color="FFFFFF", bold=True)
                c.alignment = Alignment(wrap_text=True, vertical="center")
            sheet.row_dimensions[1].height = 32
            for i, col in enumerate(sheet.columns, 1):
                width = min(48, max(13, max(len(str(c.value or "")) for c in list(col)[:30])+2))
                sheet.column_dimensions[get_column_letter(i)].width = width
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float): cell.number_format = "0.00"
            sheet.sheet_view.showGridLines = False
        if chart_path.exists():
            s = writer.book.create_sheet("Forecast chart")
            im = Image(str(chart_path)); im.width = 1040; im.height = 704; s.add_image(im, "A1")


def run_forecast(root, snapshot, output, plan=None, charts=True):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(root/"config.json")
    turbines = read_turbines(root/"data/turbines.csv")
    curve = PowerCurve.read(root/"data/power_curve.csv", cfg)
    weather, grid, audit, manifest = read_snapshot(snapshot, turbines)
    issue = pd.Timestamp(manifest["issue_time_utc"])
    physical = apply_operations(weather, turbines, cfg, curve, issue, plan)
    ensemble = physical.loc[physical.family.isin(cfg["family_weights"])].copy()
    farm_scenarios = aggregate_farm(ensemble, turbines, cfg)
    keys = ["issue_time_utc", "valid_time_utc"]
    farm = summarize(farm_scenarios, keys, "export_power_kw", cfg["family_weights"])
    gross = summarize(farm_scenarios, keys, "gross_power_kw", cfg["family_weights"])
    farm["gross_mean_kw"] = gross.mean_power_kw
    farm["lead_hours"] = (farm.valid_time_utc-issue).dt.total_seconds()/3600
    farm["valid_time_jst"] = farm.valid_time_utc.dt.tz_convert(cfg["display_timezone"])
    farm["hourly_energy_proxy_kwh"] = farm.mean_power_kw  # 1 h rectangular approximation
    energy_scenarios = farm_scenarios.groupby(["issue_time_utc", "family", "member"], as_index=False).export_power_kw.sum()
    energy = summarize(energy_scenarios, ["issue_time_utc"], "export_power_kw", cfg["family_weights"])
    energy = energy.rename(columns={c:("mean_energy_proxy_kwh" if c == "mean_power_kw" else
                                       c[:-3]+"_kwh" if c.endswith("_kw") else c) for c in energy.columns})
    energy["hours"] = len(farm)
    turbine_summary = summarize(ensemble, keys+["turbine_id"], "power_kw", cfg["family_weights"])
    features = make_features(ensemble, turbines, cfg)
    benchmark = pd.DataFrame()
    jma = physical.loc[physical.family.eq("jma")]
    if not jma.empty:
        complete = jma.groupby("valid_time_utc").turbine_id.nunique()
        jma = jma.loc[jma.valid_time_utc.isin(complete[complete.eq(len(turbines))].index)]
        if not jma.empty:
            benchmark = aggregate_farm(jma, turbines, cfg)
            benchmark["label"] = "JMA MSM 10 m + assumed power-law shear; benchmark only"
    outputs = {"farm_forecast.csv":farm, "turbine_forecast.csv":turbine_summary,
               "forecast_features.csv":features, "farm_scenarios.csv":farm_scenarios,
               "turbine_scenarios.csv":ensemble, "weather_grid_audit.csv":grid,
               "jma_benchmark.csv":benchmark, "horizon_energy_proxy.csv":energy}
    for name, df in outputs.items(): csv_ready(df).to_csv(output/name, index=False)
    notes = [
        ("status", "Uncalibrated physical/weather ensemble; no SCADA training or empirical accuracy claim."),
        ("target", "Per-turbine generation potential and operation-conditioned power; farm grid export after scenario aggregation/cap."),
        ("quantiles", "Q10/Q50/Q90 are lower/median/upper CDF quantiles; raw weather spread, not validated prediction intervals or exceedance P-values."),
        ("energy", "hourly_energy_proxy_kwh assumes each instantaneous mapped power sample represents one hour; it is not metered energy."),
        ("wind_height", "API 100 m winds adjusted to 116.5 m with alpha=0.14. JMA uses 10 m and is benchmark only."),
        ("density", "Fixed reference curve at 1.225 kg/m3; original units inferred and density tables identical. No density correction."),
        ("loss_assumptions", "Availability=1, wind multipliers=1, electrical loss=0 by default; no calibrated wake/terrain/icing model."),
        ("shutdown", "25.0 m/s generates per workbook; >25 shuts down; <=23 restarts if hysteresis enabled. Initial state running assumed."),
        ("point_estimate", "Mean of scenario power. Each family has equal weight, each complete member equal within family."),
        ("spatial_dependence", "Same family/member used simultaneously across every turbine. Sum scenarios, cap, then summarize."),
        ("time", "Stored in UTC, displayed in Asia/Tokyo; issue time is retrieval completion, not NWP run initialization."),
        ("JMA availability", "JMA MSM has a shorter horizon; benchmark may omit hours. It is excluded from ensemble weights."),
    ]
    metadata = {"status":"uncalibrated_physical_ensemble", "issue_time_utc":issue.isoformat(),
                "created_at_utc":pd.Timestamp.now(tz="UTC").isoformat(), "snapshot":Path(snapshot).name,
                "n_turbines":len(turbines), "n_hours":len(farm), "audit":audit,
                "config":cfg, "input_hashes":{}, "snapshot_manifest":manifest,
                "assumptions":dict(notes), "historical_generation_supplied":False,
                "empirical_accuracy_metrics":None}
    for name in ["config.json", "data/turbines.csv", "data/power_curve.csv"]:
        metadata["input_hashes"][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if plan:
        metadata["operational_plan_sha256"] = hashlib.sha256(Path(plan).read_bytes()).hexdigest()
    (output/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    chart_path = output/"forecast_overview.png"
    if charts: plot_forecast(farm, turbines, curve, cfg, chart_path)
    tables = {"Read me":pd.DataFrame(notes, columns=["item", "meaning"]),
              "Farm export kW":farm, "Turbines kW":turbine_summary,
              "Horizon energy proxy kWh":energy,
              "JMA benchmark":benchmark, "Weather grid audit":grid,
              "Turbine metadata":turbines, "Power curve":pd.read_csv(root/"data/power_curve.csv"),
              "Farm scenarios kW":farm_scenarios}
    write_workbook(tables, output/"Nakasato_forecast.xlsx", chart_path)
    return farm, turbine_summary, metadata
