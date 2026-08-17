"""Phase 2: exogenous feature engineering + chronological train/val/test split.

Turns the merged long-format table into a model-ready feature matrix: lag /
rolling-window sales features, one-hot encoded calendar events, and the
existing SNAP flags — the multi-variate dataset referenced in the IPR.
"""
import pandas as pd

from config import TEST_DAYS, VAL_DAYS

LAG_DAYS = (1, 7, 28)
ROLLING_WINDOWS = (7, 28)


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    grouped = df.groupby("id", observed=True)["sales"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = grouped.shift(1)
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df["id"], observed=True).transform(lambda s: s.rolling(window).mean())
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df["id"], observed=True).transform(lambda s: s.rolling(window).std())
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_event"] = (df["event_name_1"] != "none").astype("int8")
    df["is_weekend"] = df["wday"].isin([1, 2]).astype("int8")

    event_type_dummies = pd.get_dummies(df["event_type_1"], prefix="event_type", dtype="int8")
    df = pd.concat([df, event_type_dummies], axis=1)

    state_snap = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["snap"] = df.apply(lambda row: row[state_snap[row["state_id"]]], axis=1).astype("int8")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_and_rolling_features(df)
    df = add_calendar_features(df)
    # sell_price is NaN on days before an item was actually listed at a
    # store — those rows aren't meaningful demand signal, so drop them
    # alongside rows still warming up their lag window.
    df = df.dropna(subset=[f"lag_{lag}" for lag in LAG_DAYS] + ["sell_price"])
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    """Split strictly by date so no future information leaks into training,
    per the "chronological data split" practice noted in the report.
    """
    df = df.sort_values("date")
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)

    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test


FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["wday", "month", "is_event", "is_weekend", "snap", "sell_price"]
)
TARGET_COLUMN = "sales"


if __name__ == "__main__":
    from config import DATA_PROCESSED_DIR

    merged = pd.read_parquet(DATA_PROCESSED_DIR / "merged_long.parquet")
    featured = build_features(merged)
    train, val, test = chronological_split(featured)

    print(f"Featured dataset: {featured.shape[0]:,} rows x {featured.shape[1]} cols")
    print(f"Train: {train.shape[0]:,} | Val: {val.shape[0]:,} | Test: {test.shape[0]:,}")

    featured.to_parquet(DATA_PROCESSED_DIR / "featured.parquet", index=False)
    train.to_parquet(DATA_PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(DATA_PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(DATA_PROCESSED_DIR / "test.parquet", index=False)
