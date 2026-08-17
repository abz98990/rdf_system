"""Smoke tests for the pipeline using tiny synthetic M5-shaped data, so
they run without the real (multi-GB) M5 dataset present.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.make_synthetic_data import calendar, prices_df, sales_df
from src.data_ingestion import melt_sales_to_long, merge_all
from src.evaluate import mae, mape, rmse
from src.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    chronological_split,
)
from src.sequence_utils import (
    build_sequences_with_dates,
    build_window_index,
    group_series_arrays,
    split_sequences,
    split_window_index,
)


@pytest.fixture(scope="module")
def merged():
    long_sales = melt_sales_to_long(sales_df)
    return merge_all(long_sales, calendar, prices_df)


def test_merge_produces_expected_columns(merged):
    assert "sales" in merged.columns
    assert "sell_price" in merged.columns
    assert "snap_CA" in merged.columns
    assert merged["date"].is_monotonic_increasing is False  # sorted per id, not globally
    assert merged.groupby("id")["date"].apply(lambda s: s.is_monotonic_increasing).all()


def test_build_features_has_no_nulls_in_feature_columns(merged):
    featured = build_features(merged)
    assert not featured[FEATURE_COLUMNS].isna().any().any()
    assert TARGET_COLUMN in featured.columns


def test_chronological_split_is_time_ordered(merged):
    featured = build_features(merged)
    train, val, test = chronological_split(featured)
    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


def test_evaluate_metrics_are_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_mape_handles_zero_sales_without_dividing_by_zero():
    y_true = np.array([0.0, 0.0, 5.0])
    y_pred = np.array([1.0, 0.0, 5.0])
    value = mape(y_true, y_pred)
    assert np.isfinite(value)


def test_lstm_sequence_split_has_nonempty_val_and_test(merged):
    # Regression test: with VAL_DAYS == TEST_DAYS == LOOKBACK_DAYS == 28,
    # splitting the raw rows first (28 rows in val/test) can never yield a
    # 28-day lookback window. Sequences must be built across the full
    # timeline and split by target date instead.
    featured = build_features(merged)
    sequences, targets, target_dates = build_sequences_with_dates(featured)
    assert len(sequences) > 0

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_sequences(
        sequences, targets, target_dates
    )
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:]


def test_window_index_matches_materialised_sequence_count(merged):
    # The lazy (group_series_arrays + build_window_index) path used for
    # real-scale training must produce the same windows as the small-scale
    # materialising path used above, just without the O(n_windows *
    # lookback) memory spike that OOM-killed the LSTM run on real M5 data.
    featured = build_features(merged)

    sequences, _, _ = build_sequences_with_dates(featured)

    arrays = group_series_arrays(featured)
    index = build_window_index(arrays)
    assert len(index) == len(sequences)

    train_idx, val_idx, test_idx = split_window_index(index)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(index)
    assert len(val_idx) > 0
    assert len(test_idx) > 0
