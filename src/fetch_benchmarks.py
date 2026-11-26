"""Download published PM2.5 forecasts for Hoan Kiem to benchmark GRU / Transformer (notebook 10).

Two sources:

1. ``cams`` - CAMS global forecasts (ECMWF / Copernicus) from the Atmosphere Data Store.
   These are the forecasts as they were issued, with an explicit lead time, so they can be
   compared with our models at the same issue time and horizon. CAMS is the model behind the
   PM2.5 forecasts shown by Open-Meteo, Windy and many air-quality apps.
   Needs a free ADS account: https://ads.atmosphere.copernicus.eu -> accept the dataset licence
   -> put the key in ~/.cdsapirc:
       url: https://ads.atmosphere.copernicus.eu/api
       key: <your-key>
   and ``pip install cdsapi xarray netCDF4``.

2. ``openmeteo`` - Open-Meteo air-quality API (CAMS global domain, no key). Past hours are a
   continuous series stitched from the most recent CAMS runs, i.e. the shortest lead time.
   It has no horizon information, so it is an optimistic reference for CAMS.

Usage:
    py -3.11 src/fetch_benchmarks.py openmeteo
    py -3.11 src/fetch_benchmarks.py cams --start 2024-09-01 --end 2026-08-31
"""
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'external' / 'benchmarks'
LAT, LON = 21.0285, 105.8542           # Hoan Kiem
LOCAL_TZ = 'Asia/Ho_Chi_Minh'          # UTC+7, same as 'Local Time' in feature_data.csv
HORIZONS = [3, 6, 12, 24]
CAMS_RUNS_UTC = ['00:00', '12:00']     # = 07:00 and 19:00 local


def fetch_openmeteo(start: str, end: str) -> Path:
    r = requests.get('https://air-quality-api.open-meteo.com/v1/air-quality', timeout=120, params={
        'latitude': LAT, 'longitude': LON, 'hourly': 'pm2_5', 'domains': 'cams_global',
        'timezone': LOCAL_TZ, 'start_date': start, 'end_date': end})
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame({'time_local': pd.to_datetime(js['hourly']['time']),
                       'openmeteo_cams_pm25': js['hourly']['pm2_5']})
    df['grid_lat'], df['grid_lon'] = js['latitude'], js['longitude']
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / 'openmeteo_cams_pm25.csv'
    df.to_csv(out, index=False)
    print(f'{out}: {len(df)} giờ, thiếu {df.openmeteo_cams_pm25.isna().sum()}, '
          f'ô lưới ({js["latitude"]}, {js["longitude"]})')
    return out


def _open_nc(path: Path):
    import xarray as xr
    if zipfile.is_zipfile(path):
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        return xr.open_mfdataset(sorted(tmp.glob('*.nc')), combine='by_coords')
    return xr.open_dataset(path)


def _cams_month(client, first: pd.Timestamp, last: pd.Timestamp, raw_dir: Path) -> pd.DataFrame:
    target = raw_dir / f'cams_pm25_{first:%Y%m}.nc'
    if not target.exists():
        client.retrieve('cams-global-atmospheric-composition-forecasts', {
            'variable': ['particulate_matter_2.5um'],
            'date': [f'{first:%Y-%m-%d}/{last:%Y-%m-%d}'],
            'time': CAMS_RUNS_UTC,
            'leadtime_hour': [str(h) for h in HORIZONS],
            'type': ['forecast'],
            'data_format': 'netcdf_zip',
            'area': [LAT + 0.5, LON - 0.5, LAT - 0.5, LON + 0.5],   # N, W, S, E
        }, str(target))
    ds = _open_nc(target)
    var = [v for v in ds.data_vars if 'pm2p5' in v or 'pm2' in v.lower()][0]
    lat = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon = 'longitude' if 'longitude' in ds.dims else 'lon'
    point = ds[var].interp({lat: LAT, lon: LON})          # bilinear to Hoan Kiem
    df = point.to_dataframe(name='cams_pm25').reset_index()
    # Init time / lead dimension names differ between ADS versions
    init = next(c for c in ['forecast_reference_time', 'time'] if c in df.columns)
    lead = next(c for c in ['forecast_period', 'step', 'leadtime_hour'] if c in df.columns)
    hours = df[lead]
    hours = hours / np.timedelta64(1, 'h') if np.issubdtype(hours.dtype, np.timedelta64) else hours
    issue_utc = pd.to_datetime(df[init]).dt.tz_localize('UTC')
    out = pd.DataFrame({
        'issue_time': issue_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None),
        'horizon': hours.astype(int),
        'cams_pm25': df['cams_pm25'] * 1e9,                # kg/m3 -> µg/m3
    })
    out['target_time'] = out['issue_time'] + pd.to_timedelta(out['horizon'], unit='h')
    return out[out['horizon'].isin(HORIZONS)]


def fetch_cams(start: str, end: str) -> Path:
    import cdsapi
    client = cdsapi.Client()
    raw_dir = OUT_DIR / 'raw_cams'
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for first in pd.date_range(start, end, freq='MS'):
        last = min(first + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        print('CAMS', f'{first:%Y-%m}')
        frames.append(_cams_month(client, first, last, raw_dir))
    df = pd.concat(frames).sort_values(['issue_time', 'horizon']).drop_duplicates(['issue_time', 'horizon'])
    out = OUT_DIR / 'cams_forecast_pm25.csv'
    df.to_csv(out, index=False)
    print(f'{out}: {len(df)} dòng, {df.issue_time.nunique()} lần phát hành')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('source', choices=['openmeteo', 'cams'])
    ap.add_argument('--start', default='2024-09-01')   # Validation (để hiệu chỉnh bias) + Test
    ap.add_argument('--end', default='2026-08-31')
    a = ap.parse_args()
    (fetch_openmeteo if a.source == 'openmeteo' else fetch_cams)(a.start, a.end)
