from pydantic import BaseModel, Field

from src.feature_engineering import FEATURE_COLUMNS


class FeatureRow(BaseModel):
    """One day's exogenous + lag features for a single store-item series."""
    lag_1: float
    lag_7: float
    lag_28: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_28: float
    rolling_std_28: float
    wday: int
    month: int
    is_event: int = Field(ge=0, le=1)
    is_weekend: int = Field(ge=0, le=1)
    snap: int = Field(ge=0, le=1)
    sell_price: float

    def to_ordered_values(self) -> list[float]:
        return [getattr(self, col) for col in FEATURE_COLUMNS]


class BaselinePredictionRequest(BaseModel):
    features: FeatureRow


class LSTMPredictionRequest(BaseModel):
    """A sequence of consecutive daily FeatureRows, oldest first, whose
    length must equal config.LOOKBACK_DAYS.
    """
    sequence: list[FeatureRow]


class PredictionResponse(BaseModel):
    predicted_sales: float
    model_used: str
