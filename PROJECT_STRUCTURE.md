# Project structure

| Path | Purpose |
| --- | --- |
| `notebooks/01_extract.ipynb` | Weatherbit extraction |
| `notebooks/02_clean.ipynb` | Observation cleaning |
| `notebooks/03_merge.ipynb` | External-data merging |
| `notebooks/04_eda.ipynb` | Exploratory analysis |
| `notebooks/05_features.ipynb` | Feature engineering and feature groups |
| `notebooks/06_feature_selection.ipynb` | Feature-selection experiments |
| `data/raw/` | Source observations |
| `data/external/` | Downloaded external datasets |
| `data/processed/` | Cleaned, merged, and feature datasets |
| `results/` | Retained reports and split dates |
| `app.py`, `inference.py` | Forecasting application and inference code; model artifacts currently absent |
| `train_final_model.py` | Model training code |
| `bundle_registry.py`, `promote_best_model.py`, `backfill_bundle_train_range.py` | Model bundle utilities |
| `requirements.txt` | Python dependencies |

Model bundles, acquisition scripts under `src/`, automated tests, and older experiment artifacts are available in earlier Git revisions. See the root README for execution limitations.
