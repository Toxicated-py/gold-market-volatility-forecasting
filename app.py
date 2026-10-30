"""FastAPI service for gold volatility forecasting."""

from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import MINIMUM_HISTORY
from src.predict import predict_from_history

app = FastAPI(
    title="Gold Market Volatility Forecasting API",
    description="Predicts realized volatility over the next five observed trading sessions.",
    version="1.0.0",
)


class OHLCObservation(BaseModel):
    date: date
    open: Annotated[float, Field(gt=0)]
    high: Annotated[float, Field(gt=0)]
    low: Annotated[float, Field(gt=0)]
    close: Annotated[float, Field(gt=0)]


class PredictionRequest(BaseModel):
    history: list[OHLCObservation] = Field(min_length=MINIMUM_HISTORY)


@app.get("/")
def root():
    return {"service": "gold-volatility-api", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        from src.predict import load_model
        load_model()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {"status": "ok", "model": "gold_volatility_xgboost_final"}


@app.post("/predict")
def predict(request: PredictionRequest):
    frame = pd.DataFrame([item.model_dump() for item in request.history])
    frame = frame.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"})
    try:
        result = predict_from_history(frame)
    except (TypeError, ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **result,
        "target_definition": "Next five observed trading sessions realized volatility",
    }
