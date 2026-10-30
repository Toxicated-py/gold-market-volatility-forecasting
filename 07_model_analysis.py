
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance


# ==========================================
# 1. LOAD DATA
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

target = "target_future_5d_volatility_pct"

features = [
    col for col in train_df.columns
    if col not in ["Date", target]
]

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

val_dates = pd.to_datetime(val_df["Date"])


# ==========================================
# 2. CREATE BEST XGBOOST MODEL
# ==========================================

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
    n_jobs=-1
)


# ==========================================
# 3. TRAIN MODEL
# ==========================================

print("Training selected XGBoost model...")

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)


# ==========================================
# 4. EVALUATE MODEL
# ==========================================

def evaluate(actual, predicted):

    return {
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "R2": r2_score(
            actual, predicted
        )
    }


print("\nTRAINING PERFORMANCE")
print(evaluate(y_train, train_predictions))

print("\nVALIDATION PERFORMANCE")
print(evaluate(y_val, val_predictions))


# ==========================================
# 5. NATIVE FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nNATIVE FEATURE IMPORTANCE")
print(importance_df.to_string(index=False))


# ==========================================
# 6. PERMUTATION IMPORTANCE
# ==========================================

print("\nCalculating permutation importance...")

permutation = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=1,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
)

permutation_df = pd.DataFrame({
    "Feature": features,
    "Importance": permutation.importances_mean,
    "Std": permutation.importances_std
})

permutation_df = permutation_df.sort_values(
    "Importance",
    ascending=False
)

print("\nPERMUTATION FEATURE IMPORTANCE")
print(permutation_df.to_string(index=False))


# ==========================================
# 7. CREATE OUTPUT FOLDERS
# ==========================================

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/gold_volatility_xgboost.joblib"
)

print("\nModel saved successfully.")


# ==========================================
# 9. SAVE FEATURE IMPORTANCE
# ==========================================

importance_df.to_csv(
    "data/processed/native_feature_importance.csv",
    index=False
)

permutation_df.to_csv(
    "data/processed/permutation_importance.csv",
    index=False
)


# ==========================================
# 10. SAVE VALIDATION PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Date": val_dates,
    "Actual": y_val,
    "Predicted": val_predictions,
    "Residual": y_val - val_predictions
})

predictions_df.to_csv(
    "data/processed/validation_predictions.csv",
    index=False
)


# ==========================================
# 11. FEATURE IMPORTANCE CHART
# ==========================================

top_features = permutation_df.head(12).sort_values(
    "Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 12 Features - Permutation Importance")
plt.xlabel("Increase in RMSE when feature is shuffled")

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance.png"
)

plt.show()


# ==========================================
# 12. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    val_dates,
    y_val,
    label="Actual",
    alpha=0.8
)

plt.plot(
    val_dates,
    val_predictions,
    label="Predicted",
    alpha=0.8
)

plt.title("Actual vs Predicted Gold Volatility")
plt.xlabel("Date")
plt.ylabel("Future 5-Session Volatility (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/figures/actual_vs_predicted.png"
)

plt.show()


# ==========================================
# 13. RESIDUAL DISTRIBUTION
# ==========================================

residuals = y_val - val_predictions

plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=50
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title("Validation Residual Distribution")
plt.xlabel("Prediction Error (%)")
plt.ylabel("Frequency")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/figures/residual_distribution.png"
)

plt.show()


print("\nMODEL ANALYSIS COMPLETED")
