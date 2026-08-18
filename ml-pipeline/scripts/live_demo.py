"""Live terminal simulation: walks day-by-day through real, held-out M5
test-period dates for a handful of products, calling the actual running
FastAPI service for each day exactly as a downstream stock-alert system
would, and prints predicted vs. actual demand as it goes.

This is meant to be *watched* - run `python cli.py demo` and talk through
it live, or record the terminal. Every number is real: the dates simulated
are the chronological test split neither model was trained or tuned on.

If the API isn't already running, this script starts `uvicorn` itself
(and stops it again afterwards) so a single command is enough.
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, LOOKBACK_DAYS, MODELS_DIR, TEST_DAYS
from src.feature_engineering import FEATURE_COLUMNS

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _color_enabled() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if _color_enabled() else text


def wait_for_server(base_url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=1).status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def ensure_server_running(base_url: str, port: int):
    """Returns a subprocess handle if this call started the server (so the
    caller can shut it down again), or None if one was already running.
    """
    try:
        if requests.get(f"{base_url}/health", timeout=1).status_code == 200:
            print(f"Using API already running at {base_url}")
            return None
    except requests.exceptions.ConnectionError:
        pass

    print(f"No API running at {base_url} - starting one now...")
    root_dir = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=root_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server(base_url):
        proc.terminate()
        raise RuntimeError(f"API did not come up at {base_url} within 20s.")
    print(f"API is up at {base_url} (started for this demo, PID {proc.pid})")
    return proc


def pick_demo_items(featured: pd.DataFrame, n_items: int) -> list[str]:
    """Highest-volume items make for a livelier, more legible demo than a
    random sample of mostly near-zero-sales rows.
    """
    totals = featured.groupby("id", observed=True)["sales"].sum().sort_values(ascending=False)
    return totals.head(n_items).index.tolist()


def row_to_payload(row: pd.Series) -> dict:
    return {col: float(row[col]) if col not in ("wday", "month", "is_event", "is_weekend", "snap") else int(row[col]) for col in FEATURE_COLUMNS}


def call_baseline(base_url: str, feature_row: dict) -> float:
    r = requests.post(f"{base_url}/predict/baseline", json={"features": feature_row}, timeout=5)
    r.raise_for_status()
    return r.json()["predicted_sales"]


def call_lstm(base_url: str, sequence: list[dict]) -> float:
    r = requests.post(f"{base_url}/predict/lstm", json={"sequence": sequence}, timeout=5)
    r.raise_for_status()
    return r.json()["predicted_sales"]


def run_demo(n_items: int = 3, speed: float = 0.6, port: int = 8000, days: int | None = None):
    featured_path = DATA_PROCESSED_DIR / "featured.parquet"
    if not featured_path.exists():
        raise SystemExit(
            f"{featured_path} not found. Run `python cli.py train` first "
            "(it needs the M5 CSVs in data/raw/ - see README)."
        )
    if not (MODELS_DIR / "baseline_ridge.joblib").exists() or not (MODELS_DIR / "lstm_model.pt").exists():
        raise SystemExit("Trained models not found in models/. Run `python cli.py train` first.")

    base_url = f"http://127.0.0.1:{port}"
    started_proc = None
    try:
        started_proc = ensure_server_running(base_url, port)

        print("Loading held-out test-period data...")
        featured = pd.read_parquet(featured_path)
        n_days = days or TEST_DAYS
        demo_items = pick_demo_items(featured, n_items)

        print(_c(f"\n{'='*72}", DIM))
        print(_c(f"  LIVE DEMAND FORECASTING SIMULATION", BOLD))
        print(f"  Simulating the last {n_days} real days for {n_items} products.")
        print(f"  These dates were never seen during training or validation for")
        print(f"  either model - this is exactly the M5 chronological test split.")
        print(_c(f"{'='*72}\n", DIM))

        running_err = {"baseline": [], "lstm": []}

        for item_id in demo_items:
            series = featured[featured["id"] == item_id].sort_values("date").reset_index(drop=True)
            if len(series) < LOOKBACK_DAYS + n_days:
                continue

            print(_c(f"\n--- {item_id} ---", BOLD))
            header = f"{'date':<12}{'actual':>8}{'baseline':>10}{'lstm':>10}"
            print(_c(header, DIM))

            sim_rows = series.iloc[-n_days:]
            for _, row in sim_rows.iterrows():
                idx = row.name
                actual = float(row["sales"])
                date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

                baseline_pred = call_baseline(base_url, row_to_payload(row))

                lookback_slice = series.iloc[idx - LOOKBACK_DAYS: idx]
                lstm_seq = [row_to_payload(r) for _, r in lookback_slice.iterrows()]
                lstm_pred = call_lstm(base_url, lstm_seq)

                running_err["baseline"].append(abs(actual - baseline_pred))
                running_err["lstm"].append(abs(actual - lstm_pred))

                b_color = GREEN if abs(actual - baseline_pred) < 1 else YELLOW if abs(actual - baseline_pred) < 3 else RED
                l_color = GREEN if abs(actual - lstm_pred) < 1 else YELLOW if abs(actual - lstm_pred) < 3 else RED
                print(
                    f"{date_str:<12}{actual:>8.0f}"
                    f"{_c(f'{baseline_pred:>10.1f}', b_color)}"
                    f"{_c(f'{lstm_pred:>10.1f}', l_color)}"
                )
                time.sleep(speed)

        baseline_mae = sum(running_err["baseline"]) / len(running_err["baseline"])
        lstm_mae = sum(running_err["lstm"]) / len(running_err["lstm"])
        print(_c(f"\n{'='*72}", DIM))
        print(_c("  SIMULATION SUMMARY", BOLD))
        print(f"  Baseline (Ridge) running MAE over this demo: {baseline_mae:.3f}")
        print(f"  LSTM running MAE over this demo:              {lstm_mae:.3f}")
        winner = "LSTM" if lstm_mae < baseline_mae else "Baseline"
        print(f"  Lower error this run: {_c(winner, GREEN)}")
        print(
            "  Note: this is a tiny, cherry-picked-for-visibility sample "
            f"({len(running_err['baseline'])} points on the highest-volume "
            "items). It can go either way run to run. The metrics that "
            "matter for the report are the full validation-set numbers in "
            "reports/metrics_summary.md (85k+ points for the baseline, "
            "4.2k for the LSTM), where the LSTM wins on every metric."
        )
        print(_c(f"{'='*72}\n", DIM))

    finally:
        if started_proc is not None:
            print("Stopping the API instance this demo started...")
            started_proc.terminate()
            started_proc.wait(timeout=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live terminal demand-forecasting simulation.")
    parser.add_argument("--items", type=int, default=3, help="How many products to simulate.")
    parser.add_argument("--speed", type=float, default=0.6, help="Seconds paused between simulated days.")
    parser.add_argument("--port", type=int, default=8000, help="Port the API runs on.")
    parser.add_argument("--days", type=int, default=None, help="How many trailing days to simulate (default: TEST_DAYS).")
    args = parser.parse_args()
    run_demo(n_items=args.items, speed=args.speed, port=args.port, days=args.days)
