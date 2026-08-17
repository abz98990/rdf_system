"""Fetches the M5 Forecasting - Accuracy competition files without going
through Kaggle. Nixtla (a well-known open-source time-series forecasting
project) mirrors the original competition CSVs on GitHub for reproducible
benchmarking, so this pulls from there instead.

Source: https://github.com/Nixtla/m5-forecasts
"""
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW_DIR

MIRROR_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"

WANTED_FILES = {"calendar.csv", "sell_prices.csv", "sales_train_validation.csv"}


def download_m5(directory: Path = DATA_RAW_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading M5 dataset from {MIRROR_URL} ...")
    with urlopen(MIRROR_URL) as response:
        archive_bytes = response.read()
    print(f"Downloaded {len(archive_bytes) / 1e6:.1f} MB, extracting...")

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in WANTED_FILES:
                target = directory / name
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    missing = WANTED_FILES - {p.name for p in directory.glob("*.csv")}
    if missing:
        raise RuntimeError(f"Archive did not contain expected files: {missing}")
    print(f"Done. Files are in {directory}")


if __name__ == "__main__":
    download_m5()
