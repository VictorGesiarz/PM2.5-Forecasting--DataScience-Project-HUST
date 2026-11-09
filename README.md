# Hanoi next-day PM2.5 forecast

Implementation of the plan in `DataPreparationReport.pdf`: build our own dataset from OpenAQ (PM2.5) and
Open-Meteo (weather), then compare persistence, seasonal-mean, Ridge, Random Forest and gradient boosting
with chronological cross-validation and a Diebold–Mariano test.

## Setup and run (uv)

```bash
uv sync                                         # creates .venv from pyproject.toml / uv.lock
uv run python src/download_openaq.py            # ~10 min, raw OpenAQ files -> data/raw/openaq (cached)
uv run python src/build_dataset.py              # hourly / daily / feature tables -> data/processed
uv run jupyter lab notebooks/                   # or run them headless:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

No API key is needed. PM2.5 comes from the public OpenAQ AWS archive (`openaq-data-archive`), which holds the same
records as the API v3 `/sensors/{id}/measurements` endpoint. If `OPENAQ_API_KEY` is set,
`openaq_archive.api_locations_near()` can be used to cross-check the station list through the API.

## Layout

| Path | Content |
|---|---|
| `src/openaq_archive.py` | List and download daily OpenAQ files; Hanoi location ids |
| `src/openmeteo.py` | Hourly weather from the Open-Meteo archive API (UTC, wind in m/s), raw JSON kept |
| `src/preprocessing.py` | Hourly cleaning, regular grid, valid days (≥ 18 h), weather aggregation, features, leakage check |
| `src/evaluation.py` | Baselines, models, expanding-window CV, metrics, AQI category, Diebold–Mariano test |
| `src/download_openaq.py`, `src/build_dataset.py` | Command-line pipeline |
| `src/plotting.py` | Shared figure style |
| `notebooks/01_station_inventory.ipynb` | OpenAQ station inventory, selection criteria, Kaggle source identification |
| `notebooks/02_dataset_construction.ipynb` | Time zone and unit checks, cleaning, excluded days, features, leakage tests |
| `notebooks/03_eda.ipynb` | Seasonality, autocorrelation, winter episodes, weather links |
| `notebooks/04_modeling.ipynb` | CV, ablation, hold-out test, DM tests, AQI category, replication on a second station |
| `data/processed/` | `station_{id}_hourly.parquet`, `_daily.csv`, `_excluded_days.csv`, `_features.csv`, `_report.json` |
| `reports/` | Result tables (CSV) and figures |

## Key results

* **Station**: US Embassy monitor, OpenAQ location **7441** (sensor 21632), Nov 2016 – Apr 2025: 2,080 valid days and
  3 winters with ≥ 85 % of hours. No Hanoi station meets all the criteria of the report; see notebook 01.
* **The Kaggle series is OpenAQ location 2161292** (Chi cục Bảo vệ Môi trường): 99.8 % of its “OpenAQ” rows are identical. Its
  PM2.5 timestamps are in UTC, but its weather columns are in local time (7 h offset), and its wind speed is in km/h.
* **Test (Nov 2023 – Apr 2025)**: Random Forest MAE 10.2 µg/m³ and gradient boosting 10.3, vs Ridge 11.0 and persistence 12.4.
  All three models beat persistence (DM p ≤ 0.002).
