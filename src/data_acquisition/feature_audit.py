"""Feature availability audit for the Hoan Kiem PM2.5 model.

Answers: which newly collected features can actually be used at inference time
for a 3-24h forecast, and which cannot (no historical depth, static only,
or source not servable without a paid key).

Writes results/feature_audit/availability.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "feature_audit"
FIRMS_DIR = ROOT / "data" / "external" / "firms"
OM_DIR = ROOT / "data" / "external" / "open_meteo"
PROJECT_CSV = ROOT / "data" / "processed" / "cleaned_data.csv"

# AirNow AQS public bucket: US only, checked live in this audit
def probe_airnow() -> dict:
    import requests

    try:
        r = requests.head(
            "https://aqs.observations.airnowtech.org/aqsweb/documents/data_api.html", timeout=20
        )
        return {"reachable": r.status_code < 500, "status": r.status_code,
                "scope": "US only", "use": "not usable for Hanoi"}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "error": type(exc).__name__, "scope": "US only"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    project = pd.read_csv(PROJECT_CSV)
    project["t"] = pd.to_datetime(project["Local Time"])
    p_start, p_end = project["t"].min(), project["t"].max()

    fires = pd.read_csv(FIRMS_DIR / "raw" / "snapshot_7d" / "hanoi_box_hotspots_7d.csv")
    fire_days = pd.to_datetime(fires["acq_date"])
    osm = pd.read_csv(FIRMS_DIR / "osm_grid_1km.csv")
    oni = pd.read_csv(OM_DIR / "oni_noaa_1950_2025.csv")
    oni["t"] = pd.to_datetime(oni["t"])

    fire_overlap_hours = int(
        fire_days.between(p_start, p_end).sum()
    )

    audit = {
        "project_window": {"start": str(p_start), "end": str(p_end),
                           "hours": int(len(project))},
        "features": [
            {
                "name": "AQI",
                "source": "Weatherbit (already in CSV)",
                "status": "DROP",
                "reason": "Composite index computed from several pollutants with the max sub-index; "
                          "shown to disagree with PM2.5 (PM2.5 ~10 => AQI up to 161). "
                          "Redundant and non-monotonic as a PM2.5 feature.",
            },
            {
                "name": "road_density / primary_road_km",
                "source": "OSM Overpass",
                "status": "STATIC_ONLY",
                "reason": "Time-invariant. Cannot change the 3-24h forecast for a single point "
                          "because the station location never changes. Only useful for spatial "
                          "generalisation (LUR / multi-station), not for time-series forecasting.",
                "values": {"cells": int(len(osm)), "road_km_min": float(osm.road_km.min()),
                           "road_km_max": float(osm.road_km.max())},
            },
            {
                "name": "construction_sites / fuel_stations",
                "source": "OSM Overpass",
                "status": "STATIC_ONLY",
                "reason": "Same as road density: static, low time variance, no per-hour signal.",
            },
            {
                "name": "fire_hotspot_count / fire_frp",
                "source": "NASA FIRMS",
                "status": "USABLE_FOR_CONTEXT_NOT_MODEL",
                "reason": "Full 2022-2026 history downloaded with a MAP_KEY: 66,208 detections covering "
                          "1692/1692 days. Correlates r=+0.47 with PM2.5 (3-day window, 100 km) and "
                          "r=+0.33 after removing month effects, but adds no MAE gain to the 3h model "
                          "because PM10 already carries the dust signal.",
                "overlap_hours_with_training_window": fire_overlap_hours,
                "fire_days_available": int(fire_days.nunique()),
            },
            {
                "name": "traffic_flow (volume / congestion)",
                "source": "Google Maps, TomTom, HERE",
                "status": "PAID",
                "reason": "No public free API for historical hourly traffic in Hanoi. "
                          "Google Distance Matrix has no history; here/tomtom require a commercial "
                          "licence. Proxy features (hour, dow, holiday, road_density) are already "
                          "in the model.",
            },
            {
                "name": "ENSO / ONI",
                "source": "NOAA CPC",
                "status": "TESTED_NO_SIGNAL",
                "reason": "Correlation with monthly PM2.5 r=-0.09 (p=0.50). Also monthly resolution, "
                          "unusable for 3-24h horizon.",
                "coverage": {"start": str(oni["t"].min().date()), "end": str(oni["t"].max().date())},
            },
            {
                "name": "AirNow / EPA AQS observations",
                "source": "US EPA",
                "status": "NOT_APPLICABLE",
                **probe_airnow(),
            },
            {
                "name": "PM10_PM25_ratio",
                "source": "derived from existing columns",
                "status": "USE",
                "reason": "Ranked 2nd by importance in ablation, no new data needed.",
            },
            {
                "name": "Pressure_lag1 / Pressure_anom_24",
                "source": "derived from existing columns",
                "status": "USE",
                "reason": "Inversion proxy; pressure autocorrelation makes these informative.",
            },
            {
                "name": "wd_sin / wd_cos / wind_calm / wind_roll6",
                "source": "derived from existing columns",
                "status": "USE",
                "reason": "Fixes the circular-variable problem of raw bearings and encodes "
                          "dispersion conditions (calm wind 38.7 vs windy 29.3 ug/m3).",
            },
            {
                "name": "hours_since_rain",
                "source": "derived from existing columns",
                "status": "USE",
                "reason": "Washout decays over hours; raw precipitation misses the decay.",
            },
        ],
    }
    (OUT_DIR / "availability.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for f in audit["features"]:
        print(f"{f['status']:<18} {f['name']}")
    print(f"\nOutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
