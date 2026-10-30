import pandas as pd
from fastapi.testclient import TestClient

from app import app
from src.features import MINIMUM_HISTORY

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint():
    raw = pd.read_csv("data/raw/gold_daily.csv").tail(MINIMUM_HISTORY)
    history = [
        {
            "date": row.Date,
            "open": row.Open,
            "high": row.High,
            "low": row.Low,
            "close": row.Close,
        }
        for row in raw.itertuples()
    ]
    response = client.post("/predict", json={"history": history})
    assert response.status_code == 200
    assert "predicted_future_5d_volatility_pct" in response.json()


def test_invalid_api_input():
    response = client.post("/predict", json={"history": []})
    assert response.status_code == 422
