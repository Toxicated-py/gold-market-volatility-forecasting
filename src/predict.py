"""Model loading and prediction helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS, latest_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gold_volatility_xgboost_final.joblib"


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = MODEL_PATH):
    """Load the serialized model once per process."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_from_history(history: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> dict:
    """Predict next five-session realized volatility from raw OHLC history."""
    features = latest_features(history)
    prediction = float(load_model(str(model_path)).predict(features[FEATURE_COLUMNS])[0])
    latest_date = pd.to_datetime(history["Date"], errors="coerce").max()
    return {
        "as_of_date": latest_date.date().isoformat(),
        "observations_used": len(history),
        "predicted_future_5d_volatility_pct": prediction,
    }
