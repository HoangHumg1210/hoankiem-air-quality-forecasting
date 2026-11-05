"""Fetch additional keyless datasets for the Hoan Kiem PM2.5 study.

Sources (all free, no registration):
  1. ERA5-Land hourly      soil moisture, boundary-layer height, radiation
                           -> surface/inversion proxies missing from Weatherbit
  2. NASA POWER hourly     T2M, RH2M, WS2M, precipitation + meteorological profile
                           -> independent cross-check of the Weatherbit weather
  3. Open-Meteo CAMS       full 2022-01-13 .. 2026-08-31 hourly air quality
                           -> extends the CAMS record past the 2024 reanalysis cutoff
  4. Open-Meteo ensemble   forecast spread for the last days
                           -> uncertainty proxy for the newest observations

Writes data/external/open_meteo/ (era5land_*, power_*, cams_full_*, ensemble_*)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "external" / "open_meteo"
LAT, LON = 21.0285, 105.8542

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
CAMS_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ENS_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

ERA5_HOURLY = [
    "soil_moisture_0_to_7cm",
    "boundary_layer_height",
    "surface_pressure",
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "precipitation",
]
POWER_PARAMS = ["T2M", "T2MDEW", "RH2M", "WS2M", "WS10M", "WD2M", "PRECTOTCORR", "PS", "CLOUD_AMT"]
CAMS_HOURLY = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide",
               "dust", "aerosol_optical_depth"]


def _get(url: str, params: dict, retries: int = 4, timeout: int = 300) -> dict:
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(4 + 6 * attempt)
    raise RuntimeError(f"Failed {url}: {last}")


def fetch_era5land() -> None:
    path = OUT_DIR / "era5land_hoankiem_hourly.csv"
    if path.exists():
        print(f"[era5land] cached -> {path.name}")
        return
    payload = _get(ERA5_URL, {
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(ERA5_HOURLY),
        "start_date": "2022-01-13", "end_date": "2026-08-31",
        "wind_speed_unit": "ms", "timezone": "Asia/Bangkok",
    })
    df = pd.DataFrame(payload["hourly"])
    df.insert(1, "source", "open_meteo_era5_land")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[era5land] {len(df)} rows -> {path.name}")


def fetch_power() -> None:
    path = OUT_DIR / "nasa_power_hoankiem_hourly.csv"
    if path.exists():
        print(f"[power] cached -> {path.name}")
        return
    payload = _get(POWER_URL, {
        "parameters": ",".join(POWER_PARAMS),
        "community": "RE", "longitude": LON, "latitude": LAT,
        "start": "20220113", "end": "20260831", "format": "JSON",
    })
    params = payload["properties"]["parameter"]
    df = pd.DataFrame(params)
    df.index.name = "key"
    df = df.reset_index()
    df["Local Time"] = pd.to_datetime(df["key"], format="%Y%m%d%H")
    df.insert(1, "source", "nasa_power")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[power] {len(df)} rows -> {path.name}")


def fetch_cams_full() -> None:
    path = OUT_DIR / "cams_full_hoankiem_hourly.csv"
    if path.exists():
        print(f"[cams] cached -> {path.name}")
        return
    payload = _get(CAMS_URL, {
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(CAMS_HOURLY),
        "start_date": "2022-01-13", "end_date": "2026-08-31",
        "timezone": "Asia/Bangkok",
    })
    df = pd.DataFrame(payload["hourly"])
    df.insert(1, "source", "open_meteo_cams")
    df["grid_lat"], df["grid_lon"] = payload.get("latitude"), payload.get("longitude")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[cams] {len(df)} rows -> {path.name}")


def fetch_ensemble() -> None:
    path = OUT_DIR / "ensemble_forecast_spread.csv"
    if path.exists():
        print(f"[ens] cached -> {path.name}")
        return
    payload = _get(ENS_URL, {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,wind_speed_10m",
        "forecast_days": 7, "models": "gfs_seamless",
        "timezone": "Asia/Bangkok",
    })
    df = pd.DataFrame(payload["hourly"])
    df.insert(1, "source", "open_meteo_gfs_ensemble")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[ens] {len(df)} rows x {df.shape[1]} cols -> {path.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (fetch_era5land, fetch_power, fetch_cams_full, fetch_ensemble):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"!! {fn.__name__} failed: {exc}")
    manifest = {
        "latitude": LAT, "longitude": LON,
        "datasets": sorted(p.name for p in OUT_DIR.glob("*.csv")),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
