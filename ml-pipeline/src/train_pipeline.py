"""End-to-end orchestrator: ingestion -> features -> baseline vs LSTM ->
pick the best model by validation MAE and save it for the API to serve.
Also writes models/training_summary.json (metrics + LSTM loss history) so
scripts/generate_report_assets.py can turn a run into report-ready plots.
"""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from config import MODELS_DIR
from src.baseline_models import predict_baseline, train_baseline
from src.data_ingestion import build_dataset
from src.evaluate import evaluate_predictions
from src.feature_engineering import TARGET_COLUMN, build_features, chronological_split
from src.lstm_model import predict_dataset, prepare_datasets, save_lstm, train_lstm


def run():
    print("Phase 1: ingesting and merging M5 data...")
    merged = build_dataset()

    print("Phase 2: engineering features and splitting chronologically...")
    featured = build_features(merged)
    train, val, test = chronological_split(featured)
    print(f"  train={len(train):,} val={len(val):,} test={len(test):,}")

    print("Phase 3a: training baseline (Ridge Regression)...")
    baseline = train_baseline(train)
    baseline_preds = predict_baseline(baseline, val)
    baseline_metrics = evaluate_predictions(val[TARGET_COLUMN].values, baseline_preds)
    print(f"  baseline val metrics: {baseline_metrics}")

    print("Phase 3b: training LSTM...")
    train_ds, val_ds, test_ds, n_series = prepare_datasets(featured)
    print(f"  LSTM series={n_series} train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,} windows")
    lstm, lstm_history = train_lstm(train_ds, val_ds)
    y_val, lstm_preds = predict_dataset(lstm, val_ds)
    lstm_metrics = evaluate_predictions(y_val, lstm_preds)
    print(f"  LSTM val metrics: {lstm_metrics}")

    print("Phase 4: selecting best model by validation MAE...")
    best_name = "lstm" if lstm_metrics["MAE"] < baseline_metrics["MAE"] else "baseline"
    print(f"  best model: {best_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, MODELS_DIR / "baseline_ridge.joblib")
    save_lstm(lstm, MODELS_DIR / "lstm_model.pt")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "dataset_sizes": {
            "baseline_train_rows": len(train),
            "baseline_val_rows": len(val),
            "baseline_test_rows": len(test),
            "lstm_series_used": n_series,
            "lstm_train_windows": len(train_ds),
            "lstm_val_windows": len(val_ds),
            "lstm_test_windows": len(test_ds),
        },
        "baseline_val_metrics": baseline_metrics,
        "lstm_val_metrics": lstm_metrics,
        "lstm_history": lstm_history,
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved models and training_summary.json to {MODELS_DIR}")


if __name__ == "__main__":
    run()
