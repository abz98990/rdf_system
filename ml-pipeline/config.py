"""Central paths and constants for the M5 demand-forecasting pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

SALES_FILE = DATA_RAW_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_RAW_DIR / "calendar.csv"
PRICES_FILE = DATA_RAW_DIR / "sell_prices.csv"

# Keep the dataset tractable on a normal dev machine (per IPR: "millions of
# rows would cause normal dev machines to crash"). Set to None to use all series.
FILTER_STORE_ID = "CA_1"
FILTER_CAT_ID = None

LOOKBACK_DAYS = 28
VAL_DAYS = 28
TEST_DAYS = 28

# Even with FILTER_STORE_ID narrowing to one store, that's still ~3,000
# items x ~1,900 days -> millions of lookback windows, which is too slow to
# iterate per-epoch on a laptop CPU. The baseline model handles the full
# filtered dataset fine (closed-form fit), but LSTM training additionally
# subsamples to this many series. Set to None to use every series.
LSTM_MAX_SERIES = 150

RANDOM_SEED = 42
