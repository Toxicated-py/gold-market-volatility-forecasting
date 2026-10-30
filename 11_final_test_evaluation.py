
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_future_5d_volatility_pct"


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

for data in [train, validation, test]:
    data["Date"] = pd.to_datetime(data["Date"])

# Combine training and validation data.
# Test data remains completely separate.
train_full = pd.concat(
    [train, validation],
    ignore_index=True,
).sort_values("Date").reset_index(drop=True)

feature_columns = [
    col for col in train.columns
    if col not in ["Date", TARGET]
]

assert set(feature_columns).issubset(test.columns)
assert TARGET in test.columns

print("\nFINAL DATASET SUMMARY")
print("---------------------")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Combined training rows:", len(train_full))
print("Test rows:", len(test))

print(
    "Test period:",
    test["Date"].min().date(),
    "to",
    test["Date"].max().date(),
)


# --------------------------------------------------
# 3. Train final selected XGBoost model
# --------------------------------------------------

print("\nTRAINING FINAL XGBOOST MODEL...")

model = XGBRegressor(
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

X_train = train_full[feature_columns]
y_train = train_full[TARGET]

X_test = test[feature_columns]
y_test = test[TARGET]

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# --------------------------------------------------
# 4. Evaluation function
# --------------------------------------------------

def evaluate(y_true, y_pred):

    return {
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MAE": mean_absolute_error(
            y_true, y_pred
        ),
        "R2": r2_score(
            y_true, y_pred
        ),
    }


# --------------------------------------------------
# 5. Calculate baselines
# --------------------------------------------------

# Mean baseline: average target from training
# plus validation data.

mean_prediction = np.full(
    len(test),
    y_train.mean(),
)

mean_results = evaluate(
    y_test,
    mean_prediction,
)


# Persistence baseline: use the previous
# 5-session realized volatility.

persistence_prediction = test[
    "realized_vol_5d"
].values

persistence_results = evaluate(
    y_test,
    persistence_prediction,
)


# Selected XGBoost model.

model_results = evaluate(
    y_test,
    predictions,
)


# --------------------------------------------------
# 6. Compare final results
# --------------------------------------------------

results = pd.DataFrame([
    {
        "Model": "Mean Baseline",
        **mean_results,
    },
    {
        "Model": "5-Day Volatility Persistence",
        **persistence_results,
    },
    {
        "Model": "Final XGBoost",
        **model_results,
    },
])

print("\nFINAL TEST PERFORMANCE")
print("----------------------")

print(
    results.to_string(
        index=False,
        formatters={
            "RMSE": "{:.4f}".format,
            "MAE": "{:.4f}".format,
            "R2": "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# 7. Calculate improvement over mean baseline
# --------------------------------------------------

improvement = (
    (mean_results["RMSE"] - model_results["RMSE"])
    / mean_results["RMSE"]
) * 100

print(
    f"\nRMSE improvement over mean baseline: "
    f"{improvement:.2f}%"
)


# --------------------------------------------------
# 8. Save predictions and metrics
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Date": test["Date"],
    "Actual_Volatility": y_test,
    "Predicted_Volatility": predictions,
})

prediction_df["Residual"] = (
    prediction_df["Actual_Volatility"]
    - prediction_df["Predicted_Volatility"]
)

prediction_df["Absolute_Error"] = (
    prediction_df["Residual"].abs()
)

prediction_df.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False,
)

results.to_csv(
    REPORT_DIR / "final_test_metrics.csv",
    index=False,
)


# --------------------------------------------------
# 9. Save final model
# --------------------------------------------------

model_path = (
    MODEL_DIR / "gold_volatility_xgboost_final.joblib"
)

joblib.dump(model, model_path)

print("\nFILES SAVED")
print("-----------")
print("Final model:", model_path)
print("Predictions:", REPORT_DIR / "test_predictions.csv")
print("Metrics:", REPORT_DIR / "final_test_metrics.csv")

print("\nFINAL TEST EVALUATION COMPLETED")
