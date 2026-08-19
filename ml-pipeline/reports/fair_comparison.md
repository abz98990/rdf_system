# Like-for-like Model Comparison

Generated: 2026-08-18T21:00:21.875018+00:00

All models scored on identical (series, target-date) pairs. TEST is the untouched split; VALIDATION was used for early stopping and model selection, so TEST is the honest headline.

## Validation split

4,200 windows across 150 series; mean actual sales = 1.150 units/day.

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Naive last day | 1.0531 | 1.9863 | 63.06 |
| Naive last week | 1.1071 | 2.1637 | 65.48 |
| Naive 7-day mean | 0.9138 | 1.6379 | 55.41 |
| Ridge baseline | 0.9203 | 1.5894 | 56.96 |
| LSTM | 0.8085 | 1.6369 | 40.48 |

## Test split

4,200 windows across 150 series; mean actual sales = 1.174 units/day.

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Naive last day | 1.0964 | 2.1414 | 64.68 |
| Naive last week | 1.1279 | 2.2707 | 67.26 |
| Naive 7-day mean | 0.9601 | 1.7669 | 58.55 |
| Ridge baseline | 0.9558 | 1.7096 | 59.27 |
| LSTM | 0.8348 | 1.6885 | 44.05 |

## Reference: Ridge baseline across every series (NOT comparable)

85,372 rows across 3,049 series -- MAE 1.0859, RMSE 2.1152, MAPE 61.60%.

Reference only -- different sample than the LSTM, not a like-for-like comparison.
