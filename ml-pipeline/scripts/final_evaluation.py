"""Fair, like-for-like evaluation of every model on identical held-out rows.

train_pipeline.py reports each model on whatever data that model happens to
use: the baseline on all ~3,000 series, the LSTM on its subsampled 150. Those
numbers are not comparable to each other -- different item mixes have very
different difficulty. This script scores every model on exactly the same
(series, target-date) pairs so the comparison is valid, and reports the
untouched TEST split alongside validation, since validation was already
consumed by early stopping and model selection.

Also includes three naive reference forecasts, so the report can show what
the learned models actually buy over a trivial rule.

Usage: python cli.py evaluate      (or: python -m scripts.final_evaluation)
Writes models/final_evaluation.json.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, LSTM_MAX_SERIES, MODELS_DIR
from src.evaluate import evaluate_predictions
from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from src.lstm_model import WindowDataset, load_lstm, predict_dataset, subsample_series
from src.sequence_utils import build_window_index, group_series_arrays, split_window_index


def aligned_feature_rows(featured: pd.DataFrame, window_index: list) -> pd.DataFrame:
    """The feature rows for each window's target day, in the same order as
    the window index, so per-row predictions line up element-wise.
    """
    lookup = featured.set_index(["id", "date"]).sort_index()
    keys = [(id_, pd.Timestamp(target_date)) for id_, _, target_date in window_index]
    return lookup.loc[keys]


def naive_forecasts(rows: pd.DataFrame) -> dict:
    """Trivial reference rules, computed from columns already in the feature
    matrix: yesterday's sales, same-weekday-last-week, and the 7-day mean.
    """
    return {
        "naive_last_day (lag_1)": rows["lag_1"].to_numpy(dtype="float64"),
        "naive_last_week (lag_7)": rows["lag_7"].to_numpy(dtype="float64"),
        "naive_7day_mean": rows["rolling_mean_7"].to_numpy(dtype="float64"),
    }


def evaluate_split(name: str, featured: pd.DataFrame, arrays: dict, window_index: list,
                   baseline_model, lstm_model) -> dict:
    print(f"\n--- {name} split: {len(window_index):,} windows "
          f"across {len({w[0] for w in window_index})} series ---")

    rows = aligned_feature_rows(featured, window_index)
    y_true = rows[TARGET_COLUMN].to_numpy(dtype="float64")

    results = {}

    for label, preds in naive_forecasts(rows).items():
        results[label] = evaluate_predictions(y_true, preds)

    baseline_preds = np.clip(baseline_model.predict(rows[FEATURE_COLUMNS]), 0, None)
    results["baseline_ridge"] = evaluate_predictions(y_true, baseline_preds)

    dataset = WindowDataset(arrays, window_index)
    lstm_y, lstm_preds = predict_dataset(lstm_model, dataset)
    lstm_preds = np.clip(lstm_preds, 0, None)

    # predict_dataset walks the same index in order, so the two target
    # vectors must agree; assert rather than trust it silently.
    if not np.allclose(lstm_y.astype("float64"), y_true, atol=1e-4):
        raise RuntimeError(
            "LSTM targets do not match the aligned baseline rows -- "
            "the two models would not be scored on the same data."
        )
    results["lstm"] = evaluate_predictions(y_true, lstm_preds)

    for label, metrics in results.items():
        print(f"  {label:<26} MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MAPE={metrics['MAPE']:.2f}%")

    return {
        "n_windows": len(window_index),
        "n_series": len({w[0] for w in window_index}),
        "mean_actual_sales": float(y_true.mean()),
        "models": results,
    }


def run():
    featured_path = DATA_PROCESSED_DIR / "featured.parquet"
    if not featured_path.exists():
        raise SystemExit(f"{featured_path} not found -- run `python cli.py train` first.")
    baseline_path = MODELS_DIR / "baseline_ridge.joblib"
    lstm_path = MODELS_DIR / "lstm_model.pt"
    if not baseline_path.exists() or not lstm_path.exists():
        raise SystemExit("Trained models not found -- run `python cli.py train` first.")

    print("Loading featured dataset and trained models...")
    featured = pd.read_parquet(featured_path)
    baseline_model = joblib.load(baseline_path)
    lstm_model = load_lstm(lstm_path)

    # Reproduce exactly the series subsample and windows the LSTM was
    # trained on (same seed), so val/test windows here are the same ones.
    arrays = subsample_series(group_series_arrays(featured), LSTM_MAX_SERIES)
    index = build_window_index(arrays)
    _, val_idx, test_idx = split_window_index(index)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "All models scored on identical (series, target-date) pairs. "
            "TEST is the untouched split; VALIDATION was used for early "
            "stopping and model selection, so TEST is the honest headline."
        ),
        "validation": evaluate_split("VALIDATION", featured, arrays, val_idx, baseline_model, lstm_model),
        "test": evaluate_split("TEST", featured, arrays, test_idx, baseline_model, lstm_model),
    }

    # For context: how the baseline scores across every series, which is what
    # train_pipeline.py reports and is NOT comparable to the LSTM numbers.
    full_test = featured.sort_values("date")
    max_date = full_test["date"].max()
    from config import TEST_DAYS

    test_start = max_date - pd.Timedelta(days=TEST_DAYS - 1)
    full_test_rows = full_test[full_test["date"] >= test_start]
    full_preds = np.clip(baseline_model.predict(full_test_rows[FEATURE_COLUMNS]), 0, None)
    summary["baseline_on_all_series_test"] = {
        "n_rows": len(full_test_rows),
        "n_series": int(full_test_rows["id"].nunique()),
        "metrics": evaluate_predictions(full_test_rows[TARGET_COLUMN].to_numpy(dtype="float64"), full_preds),
        "note": "Reference only -- different sample than the LSTM, not a like-for-like comparison.",
    }

    out_path = MODELS_DIR / "final_evaluation.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
