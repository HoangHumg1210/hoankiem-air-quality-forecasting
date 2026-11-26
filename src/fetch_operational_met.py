"""Real-time substitutes for the reanalysis columns used in training.

Three model inputs come from reanalysis products that are published several days late,
so they are not available when an online forecast is issued:

    Boundary Layer Height  <- ERA5 (Google Earth Engine)       ~5 days late
    Soil Moisture          <- ERA5-Land (Open-Meteo archive)   ~5 days late
    Wind Speed 10m         <- NASA POWER                       a few days late

Models are trained on these columns. In operation (and in the "operational" Test
evaluation, ``forecasting.WindowData(..., operational=True)``) they are replaced by:

    Boundary Layer Height  <- Open-Meteo forecast BLH, linearly calibrated to ERA5 on Validation
    Wind Speed 10m         <- Open-Meteo forecast 10 m wind, linearly calibrated on Validation
    Soil Moisture          <- the latest ERA5-Land value that would be published (120 h old)

This script downloads the Open-Meteo forecast series from the Historical Forecast API
(model ``best_match``, values stitched from the latest runs = what was available at the time).
BLH is archived there only from 09/2024, which covers Validation (for calibration) and Test.

Usage:  py -3.11 src/fetch_operational_met.py
"""
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'external' / 'operational' / 'openmeteo_forecast_hourly.csv'
LAT, LON = 21.0285, 105.8542


def main(start='2022-01-13', end='2026-08-31'):
    r = requests.get('https://historical-forecast-api.open-meteo.com/v1/forecast', timeout=300, params={
        'latitude': LAT, 'longitude': LON, 'models': 'best_match', 'timezone': 'Asia/Ho_Chi_Minh',
        'wind_speed_unit': 'ms', 'start_date': start, 'end_date': end,
        'hourly': 'boundary_layer_height,wind_speed_10m'})
    r.raise_for_status()
    h = r.json()['hourly']
    df = pd.DataFrame({'time_local': pd.to_datetime(h['time']),
                       'blh_forecast': h['boundary_layer_height'],
                       'wind10_forecast': h['wind_speed_10m']})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'{OUT}: {len(df)} giờ | BLH thiếu {df.blh_forecast.isna().sum()} '
          f'(có từ {df.blh_forecast.first_valid_index() and df.time_local[df.blh_forecast.first_valid_index()]}) '
          f'| gió thiếu {df.wind10_forecast.isna().sum()}')


if __name__ == '__main__':
    main()
