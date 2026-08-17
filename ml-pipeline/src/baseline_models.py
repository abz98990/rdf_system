"""Phase 3a: statistical baseline models (Linear / Ridge Regression) — the
basis against which the LSTM's benefit is measured.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, RANDOM_SEED
from src.evaluate import evaluate_predictions
from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


def make_baseline(model_name: str = "ridge"):
    estimator = Ridge(alpha=1.0, random_state=RANDOM_SEED) if model_name == "ridge" else LinearRegression()
    return make_pipeline(StandardScaler(), estimator)


def train_baseline(train: pd.DataFrame, model_name: str = "ridge"):
    model = make_baseline(model_name)
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    model.fit(X_train, y_train)
    return model


def predict_baseline(model, df: pd.DataFrame):
    return model.predict(df[FEATURE_COLUMNS])


if __name__ == "__main__":
    from config import DATA_PROCESSED_DIR

    train = pd.read_parquet(DATA_PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "val.parquet")

    model = train_baseline(train, model_name="ridge")
    preds = predict_baseline(model, val)
    metrics = evaluate_predictions(val[TARGET_COLUMN].values, preds)

    print("Baseline (Ridge Regression) validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "baseline_ridge.joblib")
    print(f"Saved model to {MODELS_DIR / 'baseline_ridge.joblib'}")
