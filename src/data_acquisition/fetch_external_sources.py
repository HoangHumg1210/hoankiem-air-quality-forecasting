"""Fetch multi-source air quality + weather data for 7 candidate points.

Keyless APIs (Open-Meteo):
  - air-quality  (CAMS global reanalysis)
  - archive      (ERA5 reanalysis, 1991-2025)

Writes data/external/open_meteo/*.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "external" / "open_meteo"

START, END = "2022-01-13", "2024-12-31"
POINTS = {
    "hoankiem_center": (21.0285, 105.8542),
    "hang_bai": (21.0218, 105.8503),
    "dong_xuan": (21.0341, 105.8503),
    "hang_bac": (21.0341, 105.8412),
    "ba_dinh": (21.0333, 105.8342),
    "hai_ba_trung": (21.0083, 105.8500),
    "long_bien": (21.0407, 105.8721),
}
AQ_HOURLY = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
WEA_HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
              "surface_pressure", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation"]
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEA_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get(url: str, params: dict, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 + 5 * attempt)
    raise RuntimeError(f"Request failed: {url} {last}")


def fetch_point(name: str, lat: float, lon: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aq_path = OUT_DIR / f"airq_{name}.csv"
    if not aq_path.exists():
        payload = _get(AQ_URL, {
            "latitude": lat, "longitude": lon, "hourly": ",".join(AQ_HOURLY),
            "start_date": START, "end_date": END, "timezone": "Asia/Bangkok",
        })
        df = pd.DataFrame(payload["hourly"])
        df.insert(1, "source", "open_meteo_cams")
        df["request_lat"], df["request_lon"] = lat, lon
        df["api_lat"], df["api_lon"] = payload.get("latitude"), payload.get("longitude")
        df.to_csv(aq_path, index=False, encoding="utf-8")
        print(f"[airq] {name:<16} {len(df):>6} rows")
        time.sleep(1.0)

    wea_path = OUT_DIR / f"weather_{name}.csv"
    if not wea_path.exists():
        payload = _get(WEA_URL, {
            "latitude": lat, "longitude": lon, "hourly": ",".join(WEA_HOURLY),
            "start_date": START, "end_date": END, "wind_speed_unit": "ms",
            "timezone": "Asia/Bangkok",
        })
        df = pd.DataFrame(payload["hourly"])
        df.insert(1, "source", "open_meteo_era5")
        df["request_lat"], df["request_lon"] = lat, lon
        df.to_csv(wea_path, index=False, encoding="utf-8")
        print(f"[wea ] {name:<16} {len(df):>6} rows")
        time.sleep(1.0)


def main() -> None:
    for name in (sys.argv[1:] or list(POINTS)):
        if name not in POINTS:
            raise SystemExit(f"Unknown point: {name}. Known: {list(POINTS)}")
        lat, lon = POINTS[name]
        fetch_point(name, lat, lon)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "points.json").write_text(
        json.dumps(POINTS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
