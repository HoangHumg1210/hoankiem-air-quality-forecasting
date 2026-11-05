"""Download the full historical FIRMS fire series around Hanoi and build daily features.

Requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/
(the key is read from the FIRMS_MAP_KEY environment variable or CLI argument).

FIRMS limits each area request to 5 consecutive days, so the series is built by
stepping through the study window in 5-day blocks.

Writes:
  data/external/firms/hanoi_fires_2022_2026.csv     one row per fire detection
  data/external/firms/hanoi_fire_daily_features.csv one row per day, model ready
  results/external_drivers/fire_summary.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
FIRMS_DIR = ROOT / "data" / "external" / "firms"
OUT_DIR = ROOT / "results" / "external_drivers"

PRODUCT = "VIIRS_SNPP_SP"
# Wider box so upwind fires outside the city are captured: 19.5-22.5 N, 104.0-107.5 E
BBOX = (104.0, 19.5, 107.5, 22.5)
STATION = (21.0285, 105.8542)
START, END = "2022-01-13", "2026-08-31"
BLOCK_DAYS = 5
URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{product}/{box}/{day_range}/{start}"


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _fetch_block(key: str, start: str, retries: int = 5) -> pd.DataFrame:
    box = ",".join(f"{v:.1f}" for v in BBOX)
    url = URL.format(key=key, product=PRODUCT, box=box, day_range=BLOCK_DAYS, start=start)
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")
            text = resp.text.strip()
            if not text or text.startswith("<"):
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text))
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(6 + 8 * attempt)
    raise RuntimeError(f"block {start} failed after {retries} tries: {last}")


def download(key: str) -> pd.DataFrame:
    FIRMS_DIR.mkdir(parents=True, exist_ok=True)
    cache = FIRMS_DIR / "hanoi_fires_2022_2026.csv"
    if cache.exists():
        print(f"[firms] cached -> {cache.name}")
        df = pd.read_csv(cache)
        df["acq_ts"] = pd.to_datetime(df["acq_ts"])
        return df

    starts = pd.date_range(START, END, freq=f"{BLOCK_DAYS}D")
    frames = []
    for i, start in enumerate(starts, 1):
        block = _fetch_block(key, start.strftime("%Y-%m-%d"))
        if not block.empty:
            block["block_start"] = start.strftime("%Y-%m-%d")
            frames.append(block)
        print(f"[firms] block {i}/{len(starts)} {start.date()} -> {len(block)} rows", flush=True)
        time.sleep(1.2)

    if not frames:
        raise SystemExit("no FIRMS data returned")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    df["acq_ts"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
    )
    df["dist_km"] = _haversine_km(df["latitude"], df["longitude"], *STATION)
    df = df.sort_values("acq_ts").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    print(f"[firms] total {len(df)} detections -> {cache.name}")
    return df


def daily_features(fires: pd.DataFrame, project_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily fire aggregates aligned to the project timestamp index."""
    days = pd.date_range(project_index.min().normalize(), project_index.max().normalize(), freq="D")
    feats = pd.DataFrame(index=days)
    fire_day = fires["acq_ts"].dt.normalize()

    for radius in (25, 50, 100):
        sub = fires[fires["dist_km"] <= radius]
        grp = sub.groupby(fire_day.loc[sub.index])
        feats[f"fire_count_{radius}km"] = grp.size().reindex(days, fill_value=0)
        feats[f"fire_frp_{radius}km"] = grp["frp"].sum().reindex(days, fill_value=0)

    feats["fire_count_all"] = fire_day.value_counts().reindex(days, fill_value=0)
    feats["fire_frp_all"] = fires.groupby(fire_day)["frp"].sum().reindex(days, fill_value=0)
    feats["fire_dist_min_km"] = (
        fires.groupby(fire_day)["dist_km"].min().reindex(days).ffill().fillna(999.0)
    )

    # burn season marker (straw burning peaks): May-Jun and Sep-Oct in the Red River delta
    feats["is_burn_season"] = feats.index.month.isin([5, 6, 9, 10]).astype(int)

    # rolling exposure windows (what matters for 24h transport)
    for window in (3, 7):
        feats[f"fire_count_100km_roll{window}d"] = feats["fire_count_100km"].rolling(window).sum()
        feats[f"fire_frp_100km_roll{window}d"] = feats["fire_frp_100km"].rolling(window).sum()

    return feats


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIRMS_MAP_KEY", "")
    if not key:
        raise SystemExit("Provide the FIRMS MAP_KEY as argv[1] or FIRMS_MAP_KEY env var")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fires = download(key)

    project = pd.read_csv(ROOT / "data" / "processed" / "cleaned_data.csv", usecols=["Local Time"])
    idx = pd.to_datetime(project["Local Time"])

    feats = daily_features(fires, idx)
    feats.index.name = "date"
    feats.to_csv(FIRMS_DIR / "hanoi_fire_daily_features.csv", encoding="utf-8")

    summary = {
        "product": PRODUCT,
        "bbox": list(BBOX),
        "window": [START, END],
        "detections": int(len(fires)),
        "days_with_fire": int(fires["acq_ts"].dt.normalize().nunique()),
        "days_in_window": int(len(feats)),
        "pct_days_with_fire": round(float((feats["fire_count_all"] > 0).mean() * 100), 1),
        "detections_within_50km": int((fires["dist_km"] <= 50).sum()),
        "detections_within_100km": int((fires["dist_km"] <= 100).sum()),
        "nearest_detection_km": round(float(fires["dist_km"].min()), 1),
        "frp_total_mw": round(float(fires["frp"].sum()), 1),
        "top_months_by_detections": {
            int(k): int(v) for k, v in fires["acq_ts"].dt.month.value_counts().sort_index().items()
        },
    }
    (OUT_DIR / "fire_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs -> {FIRMS_DIR}, {OUT_DIR}")


if __name__ == "__main__":
    main()
