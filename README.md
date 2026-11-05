# Hoan Kiem Air Quality Forecasting

An air quality analysis and PM2.5 forecasting project for Hoan Kiem, Hanoi. The repository includes a Streamlit demonstration, trained model bundles, reproducible data preparation notebooks, and analysis results.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit forecasting application |
| `inference.py` | Data preparation and model inference |
| `train_final_model.py` | Model training pipeline |
| `best_model_bundle/` | Model bundle used by the application |
| `model_registry/` | Additional trained model bundles |
| `notebooks/` | Extraction, merging, cleaning, exploration, and feature selection |
| `src/data_acquisition/` | External data collection and analysis scripts |
| `data/` | Input and processed datasets |
| `results/` | Forecasting and exploratory analysis outputs |
| `tests/` | Automated tests |

See [the notebook guide](notebooks/README.md), [the results index](results/README.md), and [the full project structure](PROJECT_STRUCTURE.md) for details.

## Run the application

Use a Python environment compatible with the pinned dependencies in `requirements.txt`. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application reads the committed model bundles and `data/processed/data2225_done.csv`. Model training uses TensorFlow and may require a separate environment from the data analysis scripts; see `PROJECT_STRUCTURE.md`.

## Data workflow

Run the numbered notebooks in `notebooks/` in order, starting with `01_extract.ipynb`. The extraction notebook reads the Weatherbit API key from the `WEATHERBIT_API_KEY` environment variable:

```powershell
$env:WEATHERBIT_API_KEY = "your-key"
```

Keep API keys in environment variables and out of committed files. The repository also contains downloaded external datasets and generated analysis outputs so the existing results can be inspected without rerunning every acquisition step.

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```
