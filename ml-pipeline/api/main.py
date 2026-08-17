"""Phase 5: FastAPI deployment — exposes the trained baseline and LSTM
models as JSON demand-prediction endpoints for stock-alert integration.
"""
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import BaselinePredictionRequest, LSTMPredictionRequest, PredictionResponse
from config import LOOKBACK_DAYS, MODELS_DIR
from src.feature_engineering import FEATURE_COLUMNS

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Serves the Ridge baseline and LSTM models trained on the M5 dataset.",
    version="0.1.0",
)

_baseline_model = None
_lstm_model = None


def get_baseline_model():
    global _baseline_model
    if _baseline_model is None:
        import joblib

        path = MODELS_DIR / "baseline_ridge.joblib"
        if not path.exists():
            raise HTTPException(status_code=503, detail="Baseline model not trained yet. Run train_pipeline.py first.")
        _baseline_model = joblib.load(path)
    return _baseline_model


def get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        from src.lstm_model import load_lstm

        path = MODELS_DIR / "lstm_model.pt"
        if not path.exists():
            raise HTTPException(status_code=503, detail="LSTM model not trained yet. Run train_pipeline.py first.")
        _lstm_model = load_lstm(path)
    return _lstm_model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/baseline", response_model=PredictionResponse)
def predict_baseline(request: BaselinePredictionRequest):
    model = get_baseline_model()
    X = pd.DataFrame([request.features.to_ordered_values()], columns=FEATURE_COLUMNS)
    prediction = model.predict(X)[0]
    return PredictionResponse(predicted_sales=max(float(prediction), 0.0), model_used="baseline_ridge")


@app.post("/predict/lstm", response_model=PredictionResponse)
def predict_lstm_endpoint(request: LSTMPredictionRequest):
    from src.lstm_model import predict_lstm

    if len(request.sequence) != LOOKBACK_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"sequence must contain exactly {LOOKBACK_DAYS} days, got {len(request.sequence)}.",
        )
    model = get_lstm_model()
    X = np.array([[row.to_ordered_values() for row in request.sequence]], dtype="float32")
    prediction = predict_lstm(model, X)[0]
    return PredictionResponse(predicted_sales=max(float(prediction), 0.0), model_used="lstm")
