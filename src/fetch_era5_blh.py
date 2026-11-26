"""Tải Boundary Layer Height (ERA5 hourly) tại tâm Hoàn Kiếm qua Google Earth Engine.

Lý do: chuỗi BLH của Open-Meteo thiếu trọn 01/2024 - 06/2024 và từ 09/2024 tự chuyển
sang ECMWF IFS (lệch hệ thống so với ERA5). Dùng ERA5 cho toàn bộ giai đoạn để nhất quán.

Chạy:  py -3.13 src/fetch_era5_blh.py --project <gee-project-id>
Đầu ra: data/external/era5/era5_blh_hoankiem_hourly.csv (cột time theo giờ Việt Nam, UTC+7)
"""
import argparse
from pathlib import Path

import pandas as pd


def find_root():
    for p in Path(__file__).resolve().parents:
        if (p / "data").is_dir() and (p / "notebooks").is_dir():
            return p
    return Path.cwd()


OUT = find_root() / "data" / "external" / "era5" / "era5_blh_hoankiem_hourly.csv"
LAT, LON = 21.0285, 105.8542
START, END = "2022-01-13", "2026-08-31 23:00"
UTC_OFFSET = pd.Timedelta(hours=7)


def main(project):
    import ee
    ee.Initialize(project=project)

    point = ee.Geometry.Point(LON, LAT)
    era5 = ee.ImageCollection("ECMWF/ERA5/HOURLY").select("boundary_layer_height")

    rows = []
    start_local, end_local = pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(hours=1)
    for month in pd.date_range(start_local.to_period("M").start_time, end_local, freq="MS"):
        lo = max(month, start_local) - UTC_OFFSET
        hi = min(month + pd.offsets.MonthBegin(), end_local) - UTC_OFFSET
        if hi <= lo:
            continue
        images = era5.filterDate(lo.strftime("%Y-%m-%dT%H:%M"), hi.strftime("%Y-%m-%dT%H:%M"))
        feats = images.map(lambda img: ee.Feature(None, {
            "t": img.get("system:time_start"),
            "blh": img.reduceRegion(ee.Reducer.first(), point, 27830).get("boundary_layer_height"),
        })).getInfo()["features"]
        rows += [f["properties"] for f in feats]
        print(f"{month:%Y-%m}: {len(feats)} giờ")

    out = pd.DataFrame(rows)
    out["time"] = pd.to_datetime(out["t"], unit="ms") + UTC_OFFSET
    out = (out.rename(columns={"blh": "boundary_layer_height"})[["time", "boundary_layer_height"]]
              .drop_duplicates("time").sort_values("time"))

    expected = pd.date_range(START, END, freq="h")
    missing = expected.difference(out["time"])
    print(f"Tổng {len(out)} giờ, thiếu {len(missing)} giờ so với lưới {len(expected)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("Đã lưu:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Google Cloud project đã bật Earth Engine")
    main(parser.parse_args().project)
