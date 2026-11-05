"""External explanatory variables around Hoan Kiem, Hanoi.

Two sources, both keyless:

1. NASA FIRMS active-fire detections (VIIRS + MODIS) inside a Hanoi bounding box.
   Only the rolling 7-day window is served without a MAP_KEY, so a multi-year
   daily fire series needs a free MAP_KEY (see --help below).
2. OSM Overpass: road density / fuel stations / construction / land use proxies
   in a grid around the sample point.

Outputs under data/external/firms and results/external_drivers/.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
FIRMS_DIR = ROOT / "data" / "external" / "firms" / "raw" / "snapshot_7d"
OUT_DIR = ROOT / "results" / "external_drivers"

# Bounding box around Hanoi (lat_min, lon_min, lat_max, lon_max)
BBOX = (20.60, 105.30, 21.50, 106.40)

FIRMS_SETS = {
    "viirs_snpp_7d": ("suomi-npp-viirs-c2", "SUOMI_VIIRS_C2_SouthEast_Asia_7d.csv"),
    "viirs_noaa20_7d": ("noaa-20-viirs-c2", "J1_VIIRS_C2_SouthEast_Asia_7d.csv"),
    "modis_7d": ("modis-c6.1", "MODIS_C6_1_SouthEast_Asia_7d.csv"),
}
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"

OVERPASS_HOSTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_HEADERS = {"User-Agent": "air-quality-research/1.0 (student project)"}

# 1 km cells around the sample point
SAMPLE_POINT = (21.0285, 105.8542)
CELL_DEG = 0.009  # ~1 km
GRID_HALF = 5      # 11x11 = +/- 5 km


# ------------------------------------------------------------------ FIRMS
def fetch_firms() -> pd.DataFrame:
    FIRMS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (family, filename) in FIRMS_SETS.items():
        path = FIRMS_DIR / f"{name}.csv"
        if not path.exists():
            resp = requests.get(FIRMS_BASE + family + "/csv/" + filename, timeout=180)
            resp.raise_for_status()
            path.write_text(resp.text, encoding="utf-8")
            time.sleep(1.0)
        df = pd.read_csv(path)
        df["sensor"] = name
        frames.append(df)

    all_fires = pd.concat(frames, ignore_index=True)
    lat_min, lon_min, lat_max, lon_max = BBOX
    inside = all_fires[
        all_fires["latitude"].between(lat_min, lat_max)
        & all_fires["longitude"].between(lon_min, lon_max)
    ].copy()

    inside["acq_date"] = pd.to_datetime(inside["acq_date"])
    inside["dist_km"] = _haversine(
        inside["latitude"], inside["longitude"], SAMPLE_POINT[0], SAMPLE_POINT[1]
    )
    inside = inside.sort_values(["acq_date", "dist_km"])

    inside.to_csv(FIRMS_DIR / "hanoi_box_hotspots_7d.csv", index=False, encoding="utf-8")

    daily = (
        inside.groupby([inside["acq_date"].dt.date, "sensor"])
        .agg(hotspots=("latitude", "size"), frp_sum=("frp", "sum"))
        .reset_index()
    )
    daily.to_csv(FIRMS_DIR / "hanoi_daily_hotspots_7d.csv", index=False, encoding="utf-8")
    return inside


def _haversine(lat1, lon1, lat2, lon2):
    import numpy as np

    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------- Overpass
def _overpass(query: str) -> dict:
    last = None
    for host in OVERPASS_HOSTS:
        for attempt in range(3):
            try:
                resp = requests.post(
                    host, data={"data": query}, headers=OVERPASS_HEADERS, timeout=180
                )
                if resp.status_code == 200:
                    return resp.json()
                last = f"{host} HTTP {resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last = f"{host} {type(exc).__name__}"
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last}")


def fetch_osm_grid() -> pd.DataFrame:
    """Count road length, fuel stations, construction sites per 1 km cell."""
    rows = []
    for i in range(-GRID_HALF, GRID_HALF + 1):
        for j in range(-GRID_HALF, GRID_HALF + 1):
            lat = SAMPLE_POINT[0] + i * CELL_DEG
            lon = SAMPLE_POINT[1] + j * CELL_DEG
            box = f"{lat},{lon},{lat + CELL_DEG},{lon + CELL_DEG}"
            print(f"[osm] cell {i:+d},{j:+d} ({lat:.4f},{lon:.4f})", flush=True)

            roads = _overpass(
                f'[out:json][timeout:120];'
                f'way["highway"]["highway"!~"footway|cycleway|path|steps|pedestrian|track|bridleway"]'
                f'({box});out tags geom;'
            )
            road_km = 0.0
            lane_km = 0.0
            primary_km = 0.0
            for way in roads.get("elements", []):
                geom = way.get("geometry") or []
                if len(geom) < 2:
                    continue
                seg = sum(
                    _haversine(geom[k]["lat"], geom[k]["lon"], geom[k + 1]["lat"], geom[k + 1]["lon"])
                    for k in range(len(geom) - 1)
                )
                road_km += seg
                tags = way.get("tags", {})
                lanes = tags.get("lanes")
                lane_km += seg * (int(lanes) if lanes and str(lanes).isdigit() else 1)
                if tags.get("highway") in {"trunk", "primary", "secondary"}:
                    primary_km += seg

            fuel = _overpass(
                f'[out:json][timeout:60];node["amenity"="fuel"]({box});'
                f'way["amenity"="fuel"]({box});out count;'
            )
            n_fuel = int(fuel.get("elements", [{}])[0].get("tags", {}).get("total", 0)) if fuel.get("elements") else 0

            construction = _overpass(
                f'[out:json][timeout:60];'
                f'(way["landuse"="construction"]({box});'
                f'way["building"="construction"]({box}););out count;'
            )
            n_cons = int(construction.get("elements", [{}])[0].get("tags", {}).get("total", 0)) if construction.get("elements") else 0

            rows.append(
                {
                    "cell_i": i,
                    "cell_j": j,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "dist_to_sample_km": round(
                        float(_haversine(lat + CELL_DEG / 2, lon + CELL_DEG / 2, *SAMPLE_POINT)), 2
                    ),
                    "road_km": round(road_km, 3),
                    "lane_km": round(lane_km, 3),
                    "primary_km": round(primary_km, 3),
                    "fuel_stations": n_fuel,
                    "construction_sites": n_cons,
                }
            )
            time.sleep(2.0)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("== NASA FIRMS (7-day rolling window, keyless) ==")
    fires = fetch_firms()
    print(f"hotspots in Hanoi box: {len(fires)}")
    if not fires.empty:
        print(
            fires[["acq_date", "latitude", "longitude", "confidence", "frp", "dist_km", "sensor"]]
            .head(15)
            .to_string(index=False)
        )

    print("\n== OSM Overpass 1 km grid (121 cells) ==")
    grid = fetch_osm_grid()
    grid.to_csv(OUT_DIR / "osm_grid_1km.csv", index=False, encoding="utf-8")
    print(grid.describe().round(2).to_string())

    summary = {
        "firms": {
            "window": "rolling 7 days (keyless)",
            "bbox": BBOX,
            "hotspots_in_bbox": int(len(fires)),
            "hotspots_within_5km_of_sample": int((fires["dist_km"] <= 5).sum()) if not fires.empty else 0,
            "date_min": str(fires["acq_date"].min().date()) if not fires.empty else None,
            "date_max": str(fires["acq_date"].max().date()) if not fires.empty else None,
            "note": "Multi-year daily history requires a free MAP_KEY from firms.modaps.eosdis.nasa.gov/api/map_key",
        },
        "osm": {
            "cells": int(len(grid)),
            "cell_size_deg": CELL_DEG,
            "road_km_mean": round(float(grid["road_km"].mean()), 3),
            "road_km_max": round(float(grid["road_km"].max()), 3),
            "fuel_stations_total": int(grid["fuel_stations"].sum()),
            "construction_total": int(grid["construction_sites"].sum()),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
