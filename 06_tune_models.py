
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


def evaluate_model(name, model, X_val, y_val):

    predictions = model.predict(X_val)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    mae = mean_absolute_error(
        y_val, predictions
    )

    r2 = r2_score(
        y_val, predictions
    )

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def main():

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    train_df = pd.read_csv(
        "data/processed/train.csv"
    )

    val_df = pd.read_csv(
        "data/processed/validation.csv"
    )

    target = "target_future_5d_volatility_pct"

    features = [
        col for col in train_df.columns
        if col not in ["Date", target]
    ]

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))


    # ==========================================
    # 2. TIME SERIES CROSS VALIDATION
    # ==========================================

    tscv = TimeSeriesSplit(
        n_splits=4,
        gap=5
    )


    # ==========================================
    # 3. RANDOM FOREST TUNING
    # ==========================================

    print("\nTUNING RANDOM FOREST...")

    rf = RandomForestRegressor(
        random_state=1,
        n_jobs=1
    )

    rf_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [2, 5, 10, 20],
        "max_features": [0.7, 0.9, 1.0]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=1,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_

    print("\nBest Random Forest parameters:")
    print(rf_search.best_params_)

    print(
        "Best CV RMSE:",
        round(-rf_search.best_score_, 4)
    )


    # ==========================================
    # 4. XGBOOST TUNING
    # ==========================================

    print("\nTUNING XGBOOST...")

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=1,
        n_jobs=1,
        tree_method="hist",
        verbosity=0
    )

    xgb_params = {
        "n_estimators": [300, 500, 700],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [3, 5, 10],
        "reg_lambda": [1, 5, 10]
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_params,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=1,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_

    print("\nBest XGBoost parameters:")
    print(xgb_search.best_params_)

    print(
        "Best CV RMSE:",
        round(-xgb_search.best_score_, 4)
    )


    # ==========================================
    # 5. VALIDATION COMPARISON
    # ==========================================

    print("\nEVALUATING TUNED MODELS...")

    results = []

    results.append(
        evaluate_model(
            "Tuned Random Forest",
            best_rf,
            X_val,
            y_val
        )
    )

    results.append(
        evaluate_model(
            "Tuned XGBoost",
            best_xgb,
            X_val,
            y_val
        )
    )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    print("\nTUNED MODEL COMPARISON")
    print("======================")

    print(
        results_df.round(4).to_string(index=False)
    )


    # ==========================================
    # 6. BASELINE COMPARISON
    # ==========================================

    baseline_predictions = np.full(
        len(y_val),
        y_train.mean()
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            baseline_predictions
        )
    )

    best_rmse = results_df.iloc[0]["RMSE"]

    improvement = (
        (baseline_rmse - best_rmse)
        / baseline_rmse
    ) * 100

    print("\nBASELINE RMSE:", round(baseline_rmse, 4))
    print("BEST TUNED RMSE:", round(best_rmse, 4))

    print(
        "Improvement over baseline:",
        round(improvement, 2),
        "%"
    )


    # ==========================================
    # 7. SAVE RESULTS
    # ==========================================

    results_df.to_csv(
        "data/processed/tuned_model_comparison.csv",
        index=False
    )

    print("\nResults saved successfully.")

    print("\nHYPERPARAMETER TUNING COMPLETED")


if __name__ == "__main__":
    main()
