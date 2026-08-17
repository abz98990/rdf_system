# Retail Demand Forecasting Pipeline

Implements the pipeline described in the Interim Progress Report: a
comparison of statistical baseline models against an LSTM deep learning
architecture for multi-variate retail demand forecasting on the M5
(Walmart) dataset, deployed behind a FastAPI endpoint.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data

Download the M5 Forecasting - Accuracy files from Kaggle
(https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place
these three files in `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

By default the pipeline filters to a single store (`store_id == "CA_1"`,
set in `config.py`) so it fits in memory on a normal dev machine — the full
dataset is ~42,840 series x 1,913+ days. Adjust `FILTER_STORE_ID` /
`FILTER_CAT_ID` in `config.py` to widen or narrow the scope.

## Running the pipeline

```bash
python -m src.train_pipeline
```

This runs all phases end to end: ingests and merges the CSVs (downcasting
dtypes), engineers lag/rolling/calendar features, chronologically splits
into train/val/test, trains the Ridge baseline and the LSTM, evaluates both
on MAE/RMSE/MAPE, and saves whichever model has the lower validation MAE
(plus both models and `models/training_summary.json`) to `models/`.

Individual phases can also be run standalone for debugging:

```bash
python -m src.data_ingestion        # -> data/processed/merged_long.parquet
python -m src.feature_engineering   # -> data/processed/{featured,train,val,test}.parquet
python -m src.baseline_models       # needs train/val.parquet, -> models/baseline_ridge.joblib
python -m src.lstm_model            # needs featured.parquet, -> models/lstm_model.pt
```

## Serving predictions

```bash
uvicorn api.main:app --reload
```

- `GET /health`
- `POST /predict/baseline` — single day of features -> next-day prediction
- `POST /predict/lstm` — a `LOOKBACK_DAYS`-length sequence of daily features
  -> next-day prediction

See `api/schemas.py` for the exact request bodies.

## Project layout

```
ml-pipeline/
├── config.py              # paths, feature windows, split sizes
├── src/
│   ├── data_ingestion.py      # Phase 1: load, downcast, merge
│   ├── feature_engineering.py # Phase 2: lags, rolling stats, one-hot, split
│   ├── baseline_models.py     # Phase 3a: Ridge/Linear regression
│   ├── sequence_utils.py      # lookback-window sequence construction (no torch import)
│   ├── lstm_model.py          # Phase 3b: PyTorch LSTM with dropout
│   ├── evaluate.py            # MAE / RMSE / MAPE
│   └── train_pipeline.py      # orchestrates all phases, picks best model
├── api/
│   ├── main.py             # FastAPI app
│   └── schemas.py          # request/response models
├── data/{raw,processed}/
└── models/
```

## Status

Implements the IPR's Phase 1-5 plan and has been run end to end
(`python -m src.train_pipeline`, plus the FastAPI endpoints) against
synthetic, M5-shaped data to confirm the full path works — ingestion,
feature engineering, both models, evaluation, model selection, and
serving. Not yet run against the real M5 dataset (see "Data" above — the
CSVs aren't in this repo, and results on synthetic data are meaningless
for the report).
