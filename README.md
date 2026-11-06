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
6. `notebooks/06_feature_selection.ipynb`: run feature-selection experiments.

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

## Current limitations

- Trained model bundles, the model registry, previous experiment outputs, and the automated test suite have been removed in this revision. Their previous versions remain in Git history.
- The Streamlit and training scripts remain, but the application is not ready to run from this checkout: its former model bundles and `data/processed/data2225_done.csv` are absent.
- The feature-selection notebook currently expects `data/processed/split_dates.csv`, while the retained split file is `results/split_dates.csv`. Align that input before executing feature selection.
- Notebook files and saved outputs are included; this revision does not establish a fresh end-to-end execution or forecasting-quality result.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the current layout.
