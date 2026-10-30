
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/raw/gold_daily.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date").reset_index(drop=True)

print("Original dataset:", df.shape)


# ==========================================
# 2. DAILY RETURNS
# ==========================================

df["daily_return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

df["current_return_pct"] = df["daily_return"] * 100


# ==========================================
# 3. PRICE-BASED FEATURES
# ==========================================

# Overnight gap: today's open compared with yesterday's close
df["overnight_gap_pct"] = (
    np.log(df["Open"] / df["Close"].shift(1)) * 100
)

# Intraday price range
df["intraday_range_pct"] = (
    (df["High"] - df["Low"]) / df["Close"] * 100
)

# Today's open-to-close return
df["open_close_return_pct"] = (
    np.log(df["Close"] / df["Open"]) * 100
)


# ==========================================
# 4. LAGGED RETURNS
# ==========================================

for lag in [1, 2, 3, 5, 10]:

    df[f"return_lag_{lag}"] = (
        df["daily_return"].shift(lag) * 100
    )


# ==========================================
# 5. HISTORICAL REALIZED VOLATILITY
# ==========================================

for window in [5, 10, 20, 60]:

    df[f"realized_vol_{window}d"] = (
        np.sqrt(
            df["daily_return"]
            .pow(2)
            .rolling(window=window)
            .sum()
        ) * 100
    )


# ==========================================
# 6. PRICE MOMENTUM
# ==========================================

for window in [5, 10, 20, 60]:

    df[f"momentum_{window}d"] = (
        np.log(
            df["Close"] / df["Close"].shift(window)
        ) * 100
    )


# ==========================================
# 7. FUTURE 5-SESSION VOLATILITY TARGET
# ==========================================

# At date t, this calculates volatility using
# returns from t+1 through t+5.

future_squared_returns = (
    df["daily_return"]
    .pow(2)
    .rolling(window=5)
    .sum()
    .shift(-5)
)

df["target_future_5d_volatility_pct"] = (
    np.sqrt(future_squared_returns) * 100
)


# ==========================================
# 8. SELECT MODEL FEATURES
# ==========================================

feature_columns = [
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
    "momentum_60d"
]

target_column = "target_future_5d_volatility_pct"


# ==========================================
# 9. CREATE FINAL MODELLING DATASET
# ==========================================

model_df = df[
    ["Date"] + feature_columns + [target_column]
].copy()

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

model_df = model_df.dropna().reset_index(drop=True)


# ==========================================
# 10. SAVE PROCESSED DATA
# ==========================================

output_path = Path(
    "data/processed/gold_volatility_features.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

model_df.to_csv(output_path, index=False)


# ==========================================
# 11. RESULTS
# ==========================================

print("\nFEATURE ENGINEERING COMPLETED")

print("Original rows:", len(df))
print("Final modelling rows:", len(model_df))
print("Number of features:", len(feature_columns))

print("\nFEATURE COLUMNS:")
print(feature_columns)

print("\nTARGET SUMMARY:")
print(model_df[target_column].describe())

print("\nFIRST FIVE ROWS:")
print(model_df.head())

print("\nLAST FIVE ROWS:")
print(model_df.tail())

print("\nSaved to:", output_path)


# ==========================================
# 12. TARGET DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    model_df[target_column],
    bins=60
)

plt.title("Distribution of Future 5-Session Gold Volatility")
plt.xlabel("Future 5-Session Realized Volatility (%)")
plt.ylabel("Frequency")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/figures/future_5d_volatility_distribution.png"
)

plt.show()
