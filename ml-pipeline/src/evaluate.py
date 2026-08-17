"""Shared evaluation metrics: MAE, RMSE, MAPE, as used to compare the
baseline statistical models against the LSTM in the report.
"""
import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1.0) -> float:
    """Epsilon-smoothed MAPE — plain MAPE is undefined on the many zero-sales
    days present in retail data, so the denominator is floored at 1 unit.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
