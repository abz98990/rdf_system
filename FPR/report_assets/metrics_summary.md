# Training Metrics Summary

Generated: 2026-08-16T23:58:00.312257+00:00
Best model (by validation MAE): **lstm**

## Validation Metrics

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Baseline (Ridge Regression) | 1.0653 | 2.0797 | 60.94 |
| LSTM | 0.8089 | 1.6370 | 40.52 |

## Dataset Sizes

- Baseline: train=4,495,233 rows, val=85,372 rows, test=85,372 rows (full filtered store)
- LSTM: 150 series subsampled, train=208,333 windows, val=4,200 windows, test=4,200 windows

Note: the LSTM trains on a subsample of series (config.LSTM_MAX_SERIES) to keep per-epoch iteration tractable on a laptop CPU; the baseline's closed-form fit handles the full filtered store directly. See ml-pipeline/README.md.