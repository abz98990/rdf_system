"""Sequence construction for the LSTM — kept free of any TensorFlow/PyTorch
import so it can be unit-tested independently of whether either is installed.

Two APIs are provided:
  - build_sequences_with_dates / split_sequences: materialise every
    lookback window as one big array up front. Simple, but memory is
    O(n_windows * lookback * n_features) — fine for the tiny synthetic
    test fixtures, but this blew up to 8+ GB and got OOM-killed on the
    real M5 CA_1 store (~4.6M rows -> ~4.5M windows). Kept only for tests.
  - group_series_arrays / build_window_index / split_window_index: build a
    lightweight index of (series, start, target_date) triples and defer
    slicing out each window until it's actually needed (see
    lstm_model.WindowDataset). Memory is O(n_rows * n_features) — the
    ~28x lookback factor is the difference between fitting in 14 GB and
    not. Use this for any real-scale training.
"""
import numpy as np
import pandas as pd

from config import LOOKBACK_DAYS, TEST_DAYS, VAL_DAYS
from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


def group_series_arrays(df: pd.DataFrame, feature_cols=FEATURE_COLUMNS, target_col: str = TARGET_COLUMN):
    """One (features, target, dates) numpy triple per id — O(n_rows) memory,
    no lookback windows materialised yet.
    """
    arrays = {}
    for id_, group in df.sort_values("date").groupby("id", observed=True):
        arrays[id_] = (
            group[feature_cols].to_numpy(dtype="float32"),
            group[target_col].to_numpy(dtype="float32"),
            group["date"].to_numpy(),
        )
    return arrays


def build_window_index(arrays: dict, lookback: int = LOOKBACK_DAYS):
    """List of (id, start, target_date) for every valid lookback window,
    without touching the feature data itself.
    """
    index = []
    for id_, (_, target, dates) in arrays.items():
        n = len(target)
        if n <= lookback:
            continue
        for start in range(n - lookback):
            index.append((id_, start, dates[start + lookback]))

    if not index:
        raise ValueError(
            "No windows could be built — need at least "
            f"{lookback + 1} rows per series."
        )
    return index


def split_window_index(index: list, val_days: int = VAL_DAYS, test_days: int = TEST_DAYS):
    """Chronological split by target date, at the index level — mirrors
    feature_engineering.chronological_split, but a lookback window is
    allowed to reach back across a split boundary into that series' own
    history, since only the *target* day decides which split it belongs to.
    """
    max_date = max(d for _, _, d in index)
    test_start = max_date - np.timedelta64(test_days - 1, "D")
    val_start = test_start - np.timedelta64(val_days, "D")

    train_idx = [w for w in index if w[2] < val_start]
    val_idx = [w for w in index if val_start <= w[2] < test_start]
    test_idx = [w for w in index if w[2] >= test_start]
    return train_idx, val_idx, test_idx


def build_sequences_with_dates(df: pd.DataFrame, lookback: int = LOOKBACK_DAYS):
    """Turn the per-day feature table into (lookback_days, n_features) ->
    next_day_sales sequences, one series (store-item id) at a time so a
    window never crosses between two different products. Each sequence
    keeps the date of its target day so it can be split chronologically
    afterwards — a lookback window is allowed to reach back across a
    train/val/test boundary into that series' own history, since only the
    *target* day determines which split a sequence belongs to.
    """
    feature_cols = FEATURE_COLUMNS
    sequences, targets, target_dates = [], [], []

    for _, group in df.sort_values("date").groupby("id", observed=True):
        values = group[feature_cols].to_numpy(dtype="float32")
        target = group[TARGET_COLUMN].to_numpy(dtype="float32")
        dates = group["date"].to_numpy()
        if len(group) <= lookback:
            continue
        for start in range(len(group) - lookback):
            sequences.append(values[start:start + lookback])
            targets.append(target[start + lookback])
            target_dates.append(dates[start + lookback])

    if not sequences:
        raise ValueError(
            "No sequences could be built — need at least "
            f"{lookback + 1} rows per series."
        )
    return np.stack(sequences), np.array(targets, dtype="float32"), np.array(target_dates)


def split_sequences(sequences, targets, target_dates, val_days: int = VAL_DAYS, test_days: int = TEST_DAYS):
    """Chronological split by target date — mirrors
    feature_engineering.chronological_split but at the sequence level, so
    lookback windows near a boundary can still use in-split history.
    """
    max_date = target_dates.max()
    test_start = max_date - np.timedelta64(test_days - 1, "D")
    val_start = test_start - np.timedelta64(val_days, "D")

    train_mask = target_dates < val_start
    val_mask = (target_dates >= val_start) & (target_dates < test_start)
    test_mask = target_dates >= test_start

    return (
        (sequences[train_mask], targets[train_mask]),
        (sequences[val_mask], targets[val_mask]),
        (sequences[test_mask], targets[test_mask]),
    )
