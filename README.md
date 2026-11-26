# Hoan Kiem Air Quality Forecasting

Air quality analysis and PM2.5 forecasting research for Hoan Kiem, Hanoi.
This revision focuses on data cleaning, external-data merging, exploratory analysis, and feature engineering.

## Data workflow

Run notebooks from the repository root with a compatible Python environment:

1. `notebooks/01_extract.ipynb`: acquire Weatherbit air quality and weather data.
2. `notebooks/02_clean.ipynb`: clean the source data and export `data/processed/cleaned_data.csv`.
3. `notebooks/03_merge.ipynb`: merge cleaned observations with external weather and fire data into `data/processed/merged_data.csv`.
4. `notebooks/04_eda.ipynb`: explore the merged dataset.
5. `notebooks/05_features.ipynb`: export `data/processed/feature_data.csv` and feature-group metadata.
6. `notebooks/06_feature_selection.ipynb`: choose one shared feature set and window (GRU proxy, time-series CV inside Train). ERA5 columns are excluded by default (`USE_REANALYSIS = False`) because they are not available for online forecasting.
7. `notebooks/07_gru.ipynb`: tune the GRU on Validation (random search), then train the best config with 3 seeds.
8. `notebooks/08_transformer.ipynb`: same procedure and budget for the Transformer.
9. `notebooks/09_evaluation.ipynb`: one-time Test comparison of GRU, Transformer, persistence and seasonal naive (tables, Diebold-Mariano tests, figures in `results/report/`).
10. `notebooks/10_benchmark.ipynb`: compare both models with published forecasts (CAMS from the ADS archive, Open-Meteo CAMS, bias-corrected CAMS) at the same issue times and horizons; reports the % MAE reduction.

Notebooks 07-09 share `src/forecasting.py` (data windows, models, training loop, metrics). Training is meant for a GPU (e.g. Colab).

The extraction notebook reads its API key from the environment:

```powershell
$env:WEATHERBIT_API_KEY = "your-key"
```

Keep credentials out of notebooks and committed files. Existing raw and external datasets are under `data/`.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Select this environment as the notebook kernel. Model training requires a Python/TensorFlow-compatible environment.

## Other scripts

- `src/ghap_spatial_check.py` (run via `run_ghap.bat`): fetches GHAP 1 km PM2.5 for Hoan Kiem and compares it with Weatherbit; outputs go to `results/ghap_check/`.
- `src/fetch_era5_blh.py`: downloads ERA5 hourly Boundary Layer Height for Hoan Kiem from Google Earth Engine into `data/external/era5/` (`py -3.13 src/fetch_era5_blh.py --project <gee-project>`). It replaces the Open-Meteo BLH series, which is missing Jan-Jun 2024 and switches to ECMWF IFS from Sep 2024.
- `src/fetch_benchmarks.py`: downloads benchmark forecasts into `data/external/benchmarks/`. `openmeteo` needs no key; `cams` needs a free ADS account (key in `~/.cdsapirc`) and `pip install cdsapi xarray netCDF4`.
- `src/fix.py`: extends `data/external/firms/hanoi_fires_2022_2026.csv` up to 2026-08-31 (needs `FIRMS_MAP_KEY`). Re-run `03_merge.ipynb` afterwards.

## Current status

- Notebooks 02-05 have been executed; 06-09 have not been run yet.
