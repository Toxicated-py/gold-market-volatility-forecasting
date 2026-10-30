
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/raw/gold_daily.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

# --------------------------------
# 1. Basic dataset information
# --------------------------------

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE DATES")
print(df["Date"].duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# --------------------------------
# 2. Data quality investigation
# --------------------------------

print("\nZERO VOLUME RECORDS")
print(df["Volume"].eq(0).sum())

print("\nIDENTICAL OHLC RECORDS")
identical_ohlc = (
    df["Open"].eq(df["High"])
    & df["High"].eq(df["Low"])
    & df["Low"].eq(df["Close"])
)

print(identical_ohlc.sum())

print("\nINVALID PRICE RECORDS")
print((df[["Open", "High", "Low", "Close"]] <= 0).sum())

# --------------------------------
# 3. Daily log returns
# --------------------------------

df["daily_return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

print("\nDAILY RETURN SUMMARY")
print(df["daily_return"].describe())

# --------------------------------
# 4. Historical volatility
# --------------------------------

df["volatility_20d"] = (
    df["daily_return"]
    .rolling(window=20)
    .std()
    * np.sqrt(252)
    * 100
)

# --------------------------------
# 5. Gold closing price chart
# --------------------------------

plt.figure(figsize=(14, 6))

plt.plot(df["Date"], df["Close"])

plt.title("Gold Futures Closing Price")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/figures/gold_price.png")
plt.show()

# --------------------------------
# 6. Historical volatility chart
# --------------------------------

plt.figure(figsize=(14, 6))

plt.plot(df["Date"], df["volatility_20d"])

plt.title("20-Day Annualized Historical Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/figures/historical_volatility.png")
plt.show()

# --------------------------------
# 7. Daily returns distribution
# --------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["daily_return"].dropna(),
    bins=100
)

plt.title("Distribution of Daily Gold Returns")
plt.xlabel("Daily Log Return")
plt.ylabel("Frequency")
plt.grid(True)

plt.tight_layout()
plt.savefig("reports/figures/daily_returns.png")
plt.show()

print("\nEDA COMPLETED")
