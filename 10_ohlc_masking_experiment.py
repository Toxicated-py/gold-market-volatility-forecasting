
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "data" / "raw" / "gold_daily.csv"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "validation.csv"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_future_5d_volatility_pct"

OHLC_FEATURES = [
    "overnight_gap_pct",
    "intraday_range_pct",
    "open_close_return_pct",
]


# --------------------------------------------------
# 1. Identify OHLC-inconsistent dates
# --------------------------------------------------

raw = pd.read_csv(RAW_PATH)
raw["Date"] = pd.to_datetime(raw["Date"])

raw["invalid_ohlc"] = (
    (raw["High"] < raw[["Open", "Close"]].max(axis=1))
    | (raw["Low"] > raw[["Open", "Close"]].min(axis=1))
    | (raw["High"] < raw["Low"])
)

invalid_dates = set(
    raw.loc[raw["invalid_ohlc"], "Date"]
)

print("INVALID OHLC DATES:", len(invalid_dates))


# --------------------------------------------------
# 2. Load existing training and validation data
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

feature_columns = [
    col for col in train.columns
    if col not in ["Date", TARGET]
]

assert all(
    col in feature_columns for col in OHLC_FEATURES
), "An expected OHLC-derived feature is missing."


# --------------------------------------------------
# 3. Mask questionable OHLC-derived features
# --------------------------------------------------

def mask_invalid_features(data, split_name):

    data = data.copy()

    mask = data["Date"].isin(invalid_dates)

    affected = int(mask.sum())

    print(
        f"{split_name}: {affected}/{len(data)} "
        f"rows affected"
    )

    data.loc[mask, OHLC_FEATURES] = np.nan

    return data


train_masked = mask_invalid_features(train, "TRAIN")
validation_masked = mask_invalid_features(
    validation, "VALIDATION"
)


# --------------------------------------------------
# 4. Define the selected XGBoost model
# --------------------------------------------------

def create_model():

    return XGBRegressor(
        subsample=0.7,
        reg_lambda=1,
        n_estimators=500,
        min_child_weight=10,
        max_depth=3,
        learning_rate=0.01,
        colsample_bytree=0.7,
        objective="reg:squarederror",
        random_state=1,
        n_jobs=-1,
    )


# --------------------------------------------------
# 5. Evaluate both versions
# --------------------------------------------------

def evaluate_model(name, train_data, validation_data):

    X_train = train_data[feature_columns]
    y_train = train_data[TARGET]

    X_val = validation_data[feature_columns]
    y_val = validation_data[TARGET]

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return {
        "Model": name,
        "RMSE": np.sqrt(
            mean_squared_error(y_val, predictions)
        ),
        "MAE": mean_absolute_error(y_val, predictions),
        "R2": r2_score(y_val, predictions),
    }


print("\nTRAINING ORIGINAL MODEL...")

original_results = evaluate_model(
    "Original",
    train,
    validation,
)

print("\nTRAINING MASKED MODEL...")

masked_results = evaluate_model(
    "OHLC Masked",
    train_masked,
    validation_masked,
)


# --------------------------------------------------
# 6. Compare with mean baseline
# --------------------------------------------------

baseline_prediction = np.full(
    len(validation),
    train[TARGET].mean(),
)

baseline_results = {
    "Model": "Mean Baseline",
    "RMSE": np.sqrt(
        mean_squared_error(
            validation[TARGET],
            baseline_prediction,
        )
    ),
    "MAE": mean_absolute_error(
        validation[TARGET],
        baseline_prediction,
    ),
    "R2": r2_score(
        validation[TARGET],
        baseline_prediction,
    ),
}

results = pd.DataFrame([
    baseline_results,
    original_results,
    masked_results,
])

print("\nVALIDATION MODEL COMPARISON")
print("---------------------------")
print(results.to_string(index=False))

results.to_csv(
    REPORT_DIR / "ohlc_masking_experiment.csv",
    index=False,
)

print("\nResults saved to reports/ohlc_masking_experiment.csv")
print("\nEXPERIMENT COMPLETED")
