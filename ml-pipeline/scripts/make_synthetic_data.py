"""Generates tiny CSVs matching the M5 schema, purely to smoke-test the
pipeline before the real dataset is available. Not part of the pipeline
itself — delete data/raw/*.csv and drop in the real M5 files to go live.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_DAYS = 120
N_ITEMS = 5
STORE_ID = "CA_1"
STATE_ID = "CA"

dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
d_cols = [f"d_{i+1}" for i in range(N_DAYS)]

calendar = pd.DataFrame({
    "date": dates,
    "wm_yr_wk": [11101 + (i // 7) for i in range(N_DAYS)],
    "weekday": dates.day_name(),
    "wday": (dates.dayofweek + 2) % 7 + 1,
    "month": dates.month,
    "year": dates.year,
    "d": d_cols,
    "event_name_1": [None] * N_DAYS,
    "event_type_1": [None] * N_DAYS,
    "event_name_2": [None] * N_DAYS,
    "event_type_2": [None] * N_DAYS,
    "snap_CA": rng.integers(0, 2, N_DAYS),
    "snap_TX": rng.integers(0, 2, N_DAYS),
    "snap_WI": rng.integers(0, 2, N_DAYS),
})
calendar.loc[10, ["event_name_1", "event_type_1"]] = ["NewYear", "National"]
calendar.loc[50, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]

rows = []
for item_idx in range(N_ITEMS):
    item_id = f"FOODS_1_{item_idx:03d}"
    row = {
        "id": f"{item_id}_{STORE_ID}_validation",
        "item_id": item_id,
        "dept_id": "FOODS_1",
        "cat_id": "FOODS",
        "store_id": STORE_ID,
        "state_id": STATE_ID,
    }
    base = rng.integers(2, 10)
    sales = np.clip(rng.poisson(base, N_DAYS) + rng.integers(-1, 2, N_DAYS), 0, None)
    for col, val in zip(d_cols, sales):
        row[col] = int(val)
    rows.append(row)
sales_df = pd.DataFrame(rows)

price_rows = []
for item_idx in range(N_ITEMS):
    item_id = f"FOODS_1_{item_idx:03d}"
    price = round(float(rng.uniform(1.5, 6.0)), 2)
    for wk in sorted(calendar["wm_yr_wk"].unique()):
        price_rows.append({"store_id": STORE_ID, "item_id": item_id, "wm_yr_wk": wk, "sell_price": price})
prices_df = pd.DataFrame(price_rows)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(out_dir / "calendar.csv", index=False)
    sales_df.to_csv(out_dir / "sales_train_validation.csv", index=False)
    prices_df.to_csv(out_dir / "sell_prices.csv", index=False)
    print(f"Synthetic M5-shaped CSVs written to {out_dir}")
