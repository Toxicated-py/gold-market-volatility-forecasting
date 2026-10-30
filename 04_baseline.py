
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD MODELLING DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/gold_volatility_features.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date").reset_index(drop=True)

target_column = "target_future_5d_volatility_pct"

feature_columns = [
    col for col in df.columns
    if col not in ["Date", target_column]
]

print("Dataset shape:", df.shape)


# ==========================================
# 2. CHRONOLOGICAL SPLITTING
# ==========================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Five-session gap to prevent overlapping labels
gap = 5

train_df = df.iloc[:train_end - gap].copy()

val_df = df.iloc[
    train_end:val_end - gap
].copy()

test_df = df.iloc[val_end:].copy()


# ==========================================
# 3. DISPLAY SPLIT INFORMATION
# ==========================================

def show_split(name, data):

    print(f"\n{name}")
    print("Rows:", len(data))
    print("Start:", data["Date"].min())
    print("End:", data["Date"].max())


show_split("TRAINING SET", train_df)
show_split("VALIDATION SET", val_df)
show_split("TEST SET", test_df)

print("\nPURGED GAP:", gap, "observations")

print(
    "\nTraining labels use market data through:",
    df.loc[train_end - 1, "Date"]
)

print(
    "Validation starts:",
    df.loc[train_end, "Date"]
)

print(
    "\nValidation labels use market data through:",
    df.loc[val_end - 1, "Date"]
)

print(
    "Test starts:",
    df.loc[val_end, "Date"]
)


# ==========================================
# 4. PREPARE X AND Y
# ==========================================

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_val = val_df[feature_columns]
y_val = val_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, actual, predictions):

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    mae = mean_absolute_error(
        actual, predictions
    )

    r2 = r2_score(
        actual, predictions
    )

    return {
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }


# ==========================================
# 6. BASELINE 1: MEAN PREDICTION
# ==========================================

dummy = DummyRegressor(strategy="mean")

dummy.fit(X_train, y_train)

dummy_predictions = dummy.predict(X_val)

dummy_results = evaluate_model(
    "Mean Baseline",
    y_val,
    dummy_predictions
)


# ==========================================
# 7. BASELINE 2: HISTORICAL VOLATILITY
# ==========================================

# Predict that future five-session volatility
# will equal the most recent five-session volatility.

persistence_predictions = (
    X_val["realized_vol_5d"].to_numpy()
)

persistence_results = evaluate_model(
    "5-Day Volatility Persistence",
    y_val,
    persistence_predictions
)


# ==========================================
# 8. COMPARE BASELINES
# ==========================================

results = pd.DataFrame([
    dummy_results,
    persistence_results
])

print("\nBASELINE MODEL COMPARISON")
print("-------------------------")

print(results.to_string(index=False))


# ==========================================
# 9. SAVE SPLIT DATASETS
# ==========================================

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

val_df.to_csv(
    "data/processed/validation.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

print("\nSaved training, validation and test datasets.")

print("\nBASELINE EVALUATION COMPLETED")
