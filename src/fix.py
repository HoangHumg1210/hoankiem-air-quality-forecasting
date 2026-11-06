import io
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def find_root():
    for p in Path(__file__).resolve().parents:
        if (p / "data").is_dir() and (p / "notebooks").is_dir():
            return p
    return Path.cwd()


FIRMS_FILE = find_root() / "data" / "external" / "firms" / "hanoi_fires_2022_2026.csv"
BOX = "104.0,19.5,107.5,22.5"          # cùng khung với fetch_firms_history.py
LAT, LON = 21.0285, 105.8542
END = pd.Timestamp("2026-08-31")


def get(url, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            return r.text.strip()
        except requests.RequestException as exc:
            print(f"  lỗi: {exc}. Thử lại {attempt + 1}/{retries}")
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"Không tải được {url}")


def main():
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("Chưa đặt FIRMS_MAP_KEY. Chạy: $env:FIRMS_MAP_KEY = \"key-cua-ban\"")

    fires = pd.read_csv(FIRMS_FILE)
    last = pd.to_datetime(fires["acq_date"]).max()
    print(f"Dữ liệu cháy hiện có tới {last:%Y-%m-%d}")
    if last >= END:
        print("Đã đủ, không cần tải thêm.")
        return

    frames = []
    for start in pd.date_range(last + pd.Timedelta(days=1), END, freq="5D"):
        text = get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/"
                   f"{BOX}/5/{start:%Y-%m-%d}")
        block = pd.read_csv(io.StringIO(text)) if text and not text.startswith("<") else pd.DataFrame()
        print(f"  {start:%Y-%m-%d}: {len(block)} điểm cháy")
        if len(block):
            block["block_start"] = f"{start:%Y-%m-%d}"
            frames.append(block)
        time.sleep(1.2)

    if not frames:
        raise SystemExit("Không nhận được điểm cháy nào. Kiểm tra lại key.")

    new = pd.concat(frames, ignore_index=True)
    acq = pd.to_datetime(new["acq_date"])
    new = new[(acq > last) & (acq <= END)].copy()
    new["acq_ts"] = pd.to_datetime(new["acq_date"] + " " + new["acq_time"].astype(str).str.zfill(4),
                                   format="%Y-%m-%d %H%M")
    lat1, lon1, lat2, lon2 = map(np.radians, (new["latitude"], new["longitude"], LAT, LON))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    new["dist_km"] = 2 * 6371 * np.arcsin(np.sqrt(a))

    bak = FIRMS_FILE.with_suffix(".csv.bak")
    if not bak.exists():
        shutil.copy(FIRMS_FILE, bak)
    out = pd.concat([fires, new[[c for c in fires.columns if c in new.columns]]], ignore_index=True)
    out = out.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    out.to_csv(FIRMS_FILE, index=False)
    print(f"Thêm {len(new)} điểm cháy, giờ có tới {out['acq_date'].max()}. Chạy lại notebooks/03_merge.ipynb.")


if __name__ == "__main__":
    main()