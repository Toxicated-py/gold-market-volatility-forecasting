import pandas as pd

from src.features import MINIMUM_HISTORY
from src.predict import load_model, predict_from_history


def test_model_loads_once():
    assert load_model() is load_model()


def test_prediction_with_real_history():
    raw = pd.read_csv("data/raw/gold_daily.csv").tail(MINIMUM_HISTORY)
    result = predict_from_history(raw)
    assert result["observations_used"] == MINIMUM_HISTORY
    assert result["predicted_future_5d_volatility_pct"] >= 0
    assert result["as_of_date"] == raw["Date"].iloc[-1]
