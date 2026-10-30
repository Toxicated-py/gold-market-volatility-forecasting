
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# XGBoost
from xgboost import XGBRegressor


# ==========================================
# 1. LOAD TRAINING AND VALIDATION DATA
# ==========================================

train_df = pd.read_csv(
    "data/processed/train.csv"
)

val_df = pd.read_csv(
    "data/processed/validation.csv"
)

target = "target_future_5d_volatility_pct"

feature_columns = [
    col for col in train_df.columns
    if col not in ["Date", target]
]

X_train = train_df[feature_columns]
y_train = train_df[target]

X_val = val_df[feature_columns]
y_val = val_df[target]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Features:", len(feature_columns))


# ==========================================
# 2. DEFINE MODELS
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=20,
        random_state=1
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=1,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1,
        objective="reg:squarederror",
        random_state=1,
        n_jobs=-1
    )
}


# ==========================================
# 3. TRAIN AND EVALUATE
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    r2 = r2_score(
        y_val,
        predictions
    )

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    trained_models[name] = model

    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))
    print("R2:", round(r2, 4))


# ==========================================
# 4. COMPARE ALL MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
).reset_index(drop=True)

print("\nMODEL COMPARISON")
print("================")

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# 5. IDENTIFY BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

print("\nBEST MODEL:", best_model_name)

print(
    "Best validation RMSE:",
    round(results_df.iloc[0]["RMSE"], 4)
)


# ==========================================
# 6. COMPARE AGAINST BASELINE
# ==========================================

baseline_rmse = 1.1494

best_rmse = results_df.iloc[0]["RMSE"]

improvement = (
    (baseline_rmse - best_rmse)
    / baseline_rmse
) * 100

print("\nBASELINE COMPARISON")

print("Mean baseline RMSE:", baseline_rmse)
print("Best model RMSE:", round(best_rmse, 4))

print(
    "Improvement over baseline:",
    round(improvement, 2),
    "%"
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("data/processed/model_comparison.csv")

print("\nMODEL TRAINING COMPLETED")
