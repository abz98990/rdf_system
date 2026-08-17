"""Phase 1: Data Processing Engine — ingest the M5 CSVs, downcast dtypes to
keep memory usage manageable, and merge sales/calendar/price data into a
single multi-variate long-format table.
"""
import numpy as np
import pandas as pd

from config import CALENDAR_FILE, FILTER_CAT_ID, FILTER_STORE_ID, PRICES_FILE, SALES_FILE


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns to the smallest dtype that fits their range."""
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["date"])
    calendar = calendar.sort_values("date").reset_index(drop=True)

    if "d" not in calendar.columns:
        # Some M5 mirrors drop the precomputed "d_<n>" day-index column
        # used to join against the sales file's d_1..d_N columns; it's
        # just the row's position in date order (calendar starts at d_1).
        calendar.insert(0, "d", "d_" + (calendar.index + 1).astype(str))

    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        calendar[col] = calendar[col].fillna("none").astype("category")
    for col in ["weekday", "d"]:
        calendar[col] = calendar[col].astype("category")
    return downcast_dtypes(calendar)


def load_sell_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_FILE)
    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    return downcast_dtypes(prices)


def load_sales() -> pd.DataFrame:
    sales = pd.read_csv(SALES_FILE)

    if "id" not in sales.columns:
        # Some M5 mirrors drop the precomputed "id" column; rebuild it in
        # the same "<item_id>_<store_id>_validation" form as the original
        # Kaggle file. Built via concat (not .insert) to avoid fragmenting
        # this ~1,900-column wide frame.
        id_col = (sales["item_id"] + "_" + sales["store_id"] + "_validation").rename("id")
        sales = pd.concat([id_col, sales], axis=1)

    if FILTER_STORE_ID is not None:
        sales = sales[sales["store_id"] == FILTER_STORE_ID]
    if FILTER_CAT_ID is not None:
        sales = sales[sales["cat_id"] == FILTER_CAT_ID]

    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        sales[col] = sales[col].astype("category")

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales[day_cols] = sales[day_cols].apply(pd.to_numeric, downcast="integer")
    return sales.reset_index(drop=True)


def melt_sales_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Wide (one column per day) -> long (one row per id/day) format."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long_df = sales.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
    )
    long_df["d"] = long_df["d"].astype("category")
    long_df["sales"] = pd.to_numeric(long_df["sales"], downcast="integer")
    return long_df


def merge_all(long_sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join sales (long) with calendar (exogenous temporal features) and
    sell prices, producing the multi-variate dataset used downstream.
    """
    merged = long_sales.merge(
        calendar[
            [
                "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                "snap_CA", "snap_TX", "snap_WI",
            ]
        ],
        on="d",
        how="left",
    )
    merged = merged.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    # Items are not sold in every week; a missing price means no listing yet.
    merged["sell_price"] = merged["sell_price"].astype("float32")
    merged = merged.sort_values(["id", "date"]).reset_index(drop=True)
    return downcast_dtypes(merged)


def build_dataset() -> pd.DataFrame:
    """End-to-end Phase 1 entry point used by the pipeline orchestrator."""
    calendar = load_calendar()
    prices = load_sell_prices()
    sales = load_sales()
    long_sales = melt_sales_to_long(sales)
    return merge_all(long_sales, calendar, prices)


if __name__ == "__main__":
    from config import DATA_PROCESSED_DIR

    dataset = build_dataset()
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "merged_long.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"Merged dataset: {dataset.shape[0]:,} rows x {dataset.shape[1]} cols")
    print(f"Memory usage: {dataset.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Saved to {out_path}")
