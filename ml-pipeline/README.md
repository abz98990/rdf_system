# Retail Demand Forecasting Pipeline

Implements the pipeline described in the Interim Progress Report: a
comparison of a statistical baseline model against an LSTM deep learning
architecture for multi-variate retail demand forecasting on the M5
(Walmart) dataset, deployed behind a FastAPI endpoint.

Everything below runs through one script, [`cli.py`](cli.py) — see
**Quickstart**. Every underlying step also still works standalone (see
**Project layout**) if you want to run or debug a single phase.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

python cli.py download-data     # fetch the M5 dataset — no Kaggle account needed
python cli.py train             # ingest -> features -> baseline -> LSTM -> models/
python cli.py demo              # live terminal simulation against real held-out data
```

That's the whole path from nothing to a live demo. `train` takes a few
minutes (mostly the LSTM); everything else is fast.

## Data

`python cli.py download-data` fetches `calendar.csv`, `sell_prices.csv`,
and `sales_train_validation.csv` from a GitHub mirror of the M5
competition maintained by [Nixtla](https://github.com/Nixtla/m5-forecasts)
straight into `data/raw/` — no Kaggle account or competition-rules
acceptance required. (If you'd rather use Kaggle directly, download the
[M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
files yourself and drop the same three CSVs in `data/raw/` instead.)

By default the pipeline filters to a single store (`store_id == "CA_1"`,
set in `config.py`) so ingestion and the baseline model fit comfortably in
memory on a normal dev machine — the full dataset is ~42,840 series x
1,913+ days. Adjust `FILTER_STORE_ID` / `FILTER_CAT_ID` in `config.py` to
widen or narrow the scope.

LSTM training additionally subsamples `LSTM_MAX_SERIES` (default 150)
series from whatever `FILTER_STORE_ID`/`FILTER_CAT_ID` leaves — even one
store is ~3,000 items x ~1,900 days of lookback windows, too slow to
iterate per-epoch on a laptop CPU. The model has no per-item identity
input, so it generalises to series outside its training subsample; the
baseline's closed-form fit handles the full filtered store directly. Set
`LSTM_MAX_SERIES = None` to use every series if you have a GPU or the
patience.

## Running the pipeline

```bash
python cli.py train
```

Runs all phases end to end: ingests and merges the CSVs (downcasting
dtypes), engineers lag/rolling/calendar features, chronologically splits
into train/val/test, trains the Ridge baseline and the LSTM, evaluates both
on MAE/RMSE/MAPE, and saves whichever model has the lower validation MAE
(plus both models and `models/training_summary.json`) to `models/`.

Then turn that run into report-ready assets:

```bash
python cli.py report
```

Writes a loss-curve plot, a baseline-vs-LSTM metrics bar chart, and a
markdown metrics table to `reports/`, and copies them into
`../FPR/report_assets/` next to the Final Progress Report itself.

Individual phases can also be run standalone for debugging:

```bash
python -m src.data_ingestion        # -> data/processed/merged_long.parquet
python -m src.feature_engineering   # -> data/processed/{featured,train,val,test}.parquet
python -m src.baseline_models       # needs train/val.parquet, -> models/baseline_ridge.joblib
python -m src.lstm_model            # needs featured.parquet, -> models/lstm_model.pt
```

## Serving predictions

```bash
python cli.py serve                 # http://0.0.0.0:8000, add --reload for dev auto-reload
```

- `GET /health`
- `POST /predict/baseline` — single day of features -> next-day prediction
- `POST /predict/lstm` — a `LOOKBACK_DAYS`-length sequence of daily features
  -> next-day prediction

See `api/schemas.py` for the exact request bodies, or just open
`http://127.0.0.1:8000/docs` for interactive Swagger docs once it's running.

## Live demo — walkthrough

This is the fastest way to show someone the project actually works, live,
against real data.

**One command, no setup:**

```bash
python cli.py demo
```

What it does: picks a few of the highest-selling products, then walks
day-by-day through the most recent ~28 real days in the M5 calendar — the
chronological test split neither model ever saw during training or
validation — sending each day's features to the actual running FastAPI
service (it starts one automatically if none is up, and stops it again
when the demo ends) and printing predicted vs. actual demand as it goes,
colour-coded by how close the prediction was. It closes with a running
MAE for both models on the days just simulated.

Useful flags:

```bash
python cli.py demo --items 5 --speed 1.0     # more products, slower pace for narrating live
python cli.py demo --speed 0                 # no pause between rows, for a quick self-check
```

**Presenting this to someone else:**

1. Open two terminals side by side (optional but effective): run
   `python cli.py serve` in one so they can see real request/response
   logs scroll by, then `python cli.py demo` in the other (pass
   `--port` to both if you change it). One terminal alone works fine too
   — `demo` will start its own server quietly if you skip this.
2. Narrate as it runs: point out that the dates being simulated are real,
   held-out test-period dates, and that predictions are coming from an
   actual HTTP call to the deployed model, not a canned script.
3. For a more hands-on demo, open `http://127.0.0.1:8000/docs` in a
   browser instead (or alongside) — Swagger's UI lets your audience fill
   in a request body themselves, hit "Execute", and see a real prediction
   come back, no code involved.
4. Mention the honest caveat the demo prints at the end: it's a small,
   visibility-biased sample (highest-volume products), so which model
   "wins" can vary run to run. The metrics that matter for the report are
   the full validation-set numbers in `reports/metrics_summary.md`
   (85k+ points for the baseline, 4.2k for the LSTM) — there, the LSTM
   wins on every metric (MAE 0.81 vs 1.07, RMSE 1.64 vs 2.08, MAPE 40.5%
   vs 60.9%). `reports/lstm_loss_curve.png` shows it converging cleanly
   over 14 epochs before early stopping.

## Project layout

```
ml-pipeline/
├── cli.py                  # single entry point: download-data / train / report / serve / demo / test
├── config.py                  # paths, feature windows, split sizes
├── src/
│   ├── data_ingestion.py      # Phase 1: load, downcast, merge
│   ├── feature_engineering.py # Phase 2: lags, rolling stats, one-hot, split
│   ├── baseline_models.py     # Phase 3a: Ridge/Linear regression
│   ├── sequence_utils.py      # lookback-window index construction (no torch import)
│   ├── lstm_model.py          # Phase 3b: PyTorch LSTM with dropout, lazy WindowDataset
│   ├── evaluate.py            # MAE / RMSE / MAPE
│   └── train_pipeline.py      # orchestrates all phases, picks best model
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # request/response models
├── scripts/
│   ├── download_m5.py         # Kaggle-free M5 download
│   ├── generate_report_assets.py  # training_summary.json -> plots + table
│   ├── live_demo.py           # the `demo` command's implementation
│   └── make_synthetic_data.py # tiny M5-shaped CSVs for the test suite
├── tests/                     # pytest suite, runs against synthetic data (fast, no download needed)
├── data/{raw,processed}/
├── models/
└── reports/                   # generated plots/metrics (also copied to ../FPR/report_assets/)
```

## Status

Implements the IPR's Phase 1-5 plan and has been run end to end against
the **real M5 dataset**: ingestion, feature engineering, both models,
evaluation, model selection, serving, and the live demo have all been
exercised against real data, not just the synthetic fixtures the test
suite uses. Real-data quirks that only surfaced this way (and are now
handled): this M5 mirror is missing the `id`/`d` index columns the
original Kaggle files have (reconstructed in `data_ingestion.py`),
`sell_price` is genuinely `NaN` before an item is first listed at a store
(dropped in `feature_engineering.py`), and materialising every LSTM
lookback window into one array OOM-kills training at this scale (fixed
with the lazy `WindowDataset` in `sequence_utils.py`/`lstm_model.py`).
