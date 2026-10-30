"""Train and evaluate the final gold volatility model."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


def evaluate(y_true, predictions):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "R2": float(r2_score(y_true, predictions)),
    }


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])
    validation = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["Date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"])
    train_full = pd.concat([train, validation], ignore_index=True).sort_values("Date")

    model = XGBRegressor(
        subsample=0.7, reg_lambda=1, n_estimators=500, min_child_weight=10,
        max_depth=3, learning_rate=0.01, colsample_bytree=0.7,
        objective="reg:squarederror", random_state=1, n_jobs=-1,
    )
    model.fit(train_full[FEATURE_COLUMNS], train_full[TARGET_COLUMN])
    predictions = model.predict(test[FEATURE_COLUMNS])

    mean_results = evaluate(test[TARGET_COLUMN], np.full(len(test), train_full[TARGET_COLUMN].mean()))
    persistence_results = evaluate(test[TARGET_COLUMN], test["realized_vol_5d"])
    model_results = evaluate(test[TARGET_COLUMN], predictions)
    results = pd.DataFrame([
        {"Model": "Mean Baseline", **mean_results},
        {"Model": "5-Day Volatility Persistence", **persistence_results},
        {"Model": "Final XGBoost", **model_results},
    ])

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "gold_volatility_xgboost_final.joblib")
    pd.DataFrame({
        "Date": test["Date"], "Actual_Volatility": test[TARGET_COLUMN],
        "Predicted_Volatility": predictions,
        "Residual": test[TARGET_COLUMN] - predictions,
        "Absolute_Error": np.abs(test[TARGET_COLUMN] - predictions),
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    results.to_csv(REPORT_DIR / "final_test_metrics.csv", index=False)
    pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_}).sort_values(
        "Importance", ascending=False
    ).to_csv(REPORT_DIR / "final_feature_importance.csv", index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_parameters": model.get_params(),
        "training_rows": len(train_full),
        "training_date_range": [train_full["Date"].min().date().isoformat(), train_full["Date"].max().date().isoformat()],
        "test_date_range": [test["Date"].min().date().isoformat(), test["Date"].max().date().isoformat()],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
