"""Open-Meteo acquisition, immutable snapshots, and strict scenario validation."""
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

SOURCES = {
    "gfs": {"endpoint":"https://ensemble-api.open-meteo.com/v1/ensemble", "model":"gfs_seamless", "height":100},
    "ecmwf": {"endpoint":"https://ensemble-api.open-meteo.com/v1/ensemble", "model":"ecmwf_ifs025", "height":100},
    "jma": {"endpoint":"https://api.open-meteo.com/v1/jma", "model":"jma_msm", "height":10},
}


def get_json(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"NakasatoResearchForecast/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                body = response.read()
            return json.loads(body), hashlib.sha256(body).hexdigest()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code not in [429, 500, 502, 503, 504] or attempt == 2:
                raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {detail}") from exc
            delay = exc.headers.get("Retry-After", str(2**attempt))
            if not delay.isdigit() or int(delay) > 10:
                raise RuntimeError(f"Open-Meteo asked to retry later ({delay}); rerun acquisition later.") from exc
            time.sleep(int(delay))
    raise RuntimeError("Forecast acquisition failed.")


def fetch_snapshot(turbines, hours, destination):
    if not 1 <= hours <= 168:
        raise ValueError("Forecast horizon must be 1..168 hours.")
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    start = pd.Timestamp.now(tz="UTC").ceil("h")
    end = start + pd.Timedelta(hours=hours-1)
    manifest = {"requested_start_utc":start.isoformat(), "hours":hours,
                "nwp_initialization_time_utc":None,
                "issue_time_definition":"Time all required latest API responses had been retrieved; NWP run initializations are not supplied by these responses.",
                "turbine_ids":turbines.turbine_id.tolist(), "sources":{}, "optional_errors":{}}
    for family, source in SOURCES.items():
        h = source["height"]
        params = {"latitude":",".join(map(str, turbines.latitude)),
                  "longitude":",".join(map(str, turbines.longitude)),
                  "hourly":f"wind_speed_{h}m,wind_direction_{h}m",
                  "models":source["model"], "wind_speed_unit":"ms", "timezone":"GMT",
                  "start_hour":start.strftime("%Y-%m-%dT%H:%M"),
                  "end_hour":end.strftime("%Y-%m-%dT%H:%M"), "cell_selection":"land"}
        url = source["endpoint"]+"?"+urllib.parse.urlencode(params)
        try:
            data, digest = get_json(url)
        except (RuntimeError, OSError) as exc:
            if family != "jma":
                raise
            manifest["optional_errors"][family] = str(exc)
            continue
        saved = json.dumps(data)
        (dest/(family+".json")).write_text(saved, encoding="utf-8")
        manifest["sources"][family] = dict(source, request_url=url, response_sha256=digest,
                                               saved_payload_sha256=hashlib.sha256(saved.encode()).hexdigest(),
                                               fetched_at_utc=pd.Timestamp.now(tz="UTC").isoformat())
        print(f"Retrieved {family}: {len(data) if isinstance(data,list) else 1} locations", flush=True)
    manifest["issue_time_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    if pd.Timestamp(manifest["issue_time_utc"]) >= start:
        raise RuntimeError("Acquisition crossed the first valid hour. Run again for a new future window.")
    (dest/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest


def read_snapshot(directory, turbines):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["turbine_ids"] != turbines.turbine_id.tolist():
        raise ValueError("Snapshot location order differs from turbine metadata.")
    start = pd.Timestamp(manifest["requested_start_utc"])
    issue = pd.Timestamp(manifest["issue_time_utc"])
    times = pd.date_range(start, periods=manifest["hours"], freq="h")
    all_rows, grid_rows, audit = [], [], {}
    for family, source in manifest["sources"].items():
        payload = (directory/(family+".json")).read_bytes()
        if source.get("saved_payload_sha256") and hashlib.sha256(payload).hexdigest() != source["saved_payload_sha256"]:
            raise ValueError(f"{family}: saved weather payload checksum mismatch.")
        params = urllib.parse.parse_qs(urllib.parse.urlparse(source["request_url"]).query)
        for col in ["latitude", "longitude"]:
            coordinates = np.array([float(v) for v in params[col][0].split(",")])
            if len(coordinates) != len(turbines) or not np.allclose(coordinates, turbines[col], rtol=0, atol=1e-7):
                raise ValueError(f"{family}: snapshot requested coordinates differ from turbine metadata.")
        raw = json.loads(payload)
        locations = raw if isinstance(raw, list) else [raw]
        if len(locations) != len(turbines):
            raise ValueError(f"{family}: location count mismatch.")
        h = source["height"]
        speed_key, direction_key = f"wind_speed_{h}m", f"wind_direction_{h}m"
        members_by_site, candidates = [], set()
        for (_, turbine), response in zip(turbines.iterrows(), locations):
            hourly = response.get("hourly", {})
            if not hourly or response.get("utc_offset_seconds", 0) != 0:
                raise ValueError(f"{family}: missing hourly data or non-UTC response.")
            units = response.get("hourly_units", {})
            if units.get(speed_key) != "m/s":
                raise ValueError(f"{family}: expected m/s, got {units.get(speed_key)}.")
            source_times = pd.to_datetime(hourly["time"], utc=True)
            if source_times.duplicated().any():
                raise ValueError("Duplicate source forecast times.")
            site_members = {}
            for key in hourly:
                match = re.fullmatch(re.escape(speed_key)+r"(?:_member(\d+))?", key)
                if not match:
                    continue
                member = int(match.group(1) or 0)
                suffix = "" if match.group(1) is None else "_member"+match.group(1)
                dk = direction_key+suffix
                if dk not in hourly:
                    continue
                a = pd.DataFrame({"wind_speed_ref_ms":hourly[key], "wind_direction_deg":hourly[dk]}, index=source_times).reindex(times)
                site_members[member] = a
                candidates.add(member)
            members_by_site.append(site_members)
            grid_rows.append({"family":family, "turbine_id":turbine.turbine_id,
                              "requested_latitude":turbine.latitude, "requested_longitude":turbine.longitude,
                              "returned_latitude":response.get("latitude"), "returned_longitude":response.get("longitude"),
                              "api_ground_elevation_m":response.get("elevation"), "reference_height_m":h})
        kept, dropped = [], []
        for member in sorted(candidates):
            arrays = [site.get(member) for site in members_by_site]
            if family != "jma" and any(a is None or not np.isfinite(a.to_numpy(float)).all()
                                        or (a.wind_speed_ref_ms < 0).any() for a in arrays):
                dropped.append(member)
                continue
            kept.append(member)
            for (_, turbine), a in zip(turbines.iterrows(), arrays):
                if a is None:
                    continue
                a = a.dropna().reset_index(names="valid_time_utc")
                if (a.wind_speed_ref_ms < 0).any() or not a.wind_direction_deg.between(0,360).all():
                    raise ValueError(f"{family}: invalid wind values.")
                a["turbine_id"] = turbine.turbine_id
                a["family"], a["member"] = family, member
                a["reference_height_m"], a["issue_time_utc"] = h, issue
                all_rows.append(a)
        if family != "jma" and len(kept) < 2:
            raise ValueError(f"{family}: fewer than 2 complete spatial/time-coherent members.")
        audit[family] = {"model":source["model"], "members_used":len(kept), "dropped_members":dropped,
                         "api_reference_height_m":h}
    if not {"gfs", "ecmwf"} <= set(audit):
        raise ValueError("Both GFS and ECMWF are required. No cached/live mixing or silent fallback.")
    weather = pd.concat(all_rows, ignore_index=True)
    if (weather.valid_time_utc <= issue).any():
        raise ValueError("Snapshot contains a valid time at/before issue.")
    grid = pd.DataFrame(grid_rows)
    for family in audit:
        g = grid.loc[grid.family.eq(family)]
        audit[family]["distinct_returned_grids"] = len(g[["returned_latitude", "returned_longitude"]].drop_duplicates())
        w = weather.loc[weather.family.eq(family)]
        fingerprints = set()
        for _, site in w.groupby("turbine_id"):
            values = site.sort_values(["member", "valid_time_utc"])[["wind_speed_ref_ms", "wind_direction_deg"]].to_numpy(float)
            fingerprints.add(hashlib.sha256(values.tobytes()).hexdigest())
        audit[family]["distinct_site_wind_trajectories"] = len(fingerprints)
    return weather, grid, audit, manifest
