"""Feature engineering shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "current_return_pct",
    "overnight_gap_pct",
    "intraday_range_pct",
    "open_close_return_pct",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_20d",
    "realized_vol_60d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_60d",
]
TARGET_COLUMN = "target_future_5d_volatility_pct"
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
MINIMUM_HISTORY = 61


def _validate_input(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("OHLC history must be provided as a pandas DataFrame.")
    required = ["Date", *OHLC_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = data[required].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("Date contains invalid values.")
    if frame["Date"].duplicated().any():
        raise ValueError("Date contains duplicate observations.")
    for column in OHLC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[OHLC_COLUMNS].isna().any().any():
        raise ValueError("OHLC data contains missing or non-numeric values.")
    if (frame[OHLC_COLUMNS] <= 0).any().any():
        raise ValueError("OHLC prices must all be positive.")
    return frame.sort_values("Date").reset_index(drop=True)


def build_features(data: pd.DataFrame, require_history: bool = False) -> pd.DataFrame:
    """Build the exact 17 model features from historical OHLC observations."""
    df = _validate_input(data)
    if require_history and len(df) < MINIMUM_HISTORY:
        raise ValueError(
            f"At least {MINIMUM_HISTORY} OHLC observations are required "
            "to calculate the 60-session volatility feature."
        )

    daily_return = np.log(df["Close"] / df["Close"].shift(1))
    df["current_return_pct"] = daily_return * 100
    df["overnight_gap_pct"] = np.log(df["Open"] / df["Close"].shift(1)) * 100
    df["intraday_range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["open_close_return_pct"] = np.log(df["Close"] / df["Open"]) * 100

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = daily_return.shift(lag) * 100
    for window in [5, 10, 20, 60]:
        df[f"realized_vol_{window}d"] = (
            np.sqrt(daily_return.pow(2).rolling(window=window).sum()) * 100
        )
        df[f"momentum_{window}d"] = (
            np.log(df["Close"] / df["Close"].shift(window)) * 100
        )
    return df[["Date", *FEATURE_COLUMNS]]


def latest_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return the latest complete feature row for prediction."""
    features = build_features(data, require_history=True)
    complete = features.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        raise ValueError("OHLC history does not contain a complete feature row.")
    return complete.tail(1)[FEATURE_COLUMNS]
