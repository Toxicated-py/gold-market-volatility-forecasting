
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "gold_daily.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Previous closing price
df["previous_close"] = df["Close"].shift(1)

# Daily close-to-close log return
df["daily_log_return_pct"] = (
    np.log(df["Close"] / df["previous_close"]) * 100
)

# Overnight gap relative to previous close
df["overnight_gap_pct"] = (
    (df["Open"] / df["previous_close"]) - 1
) * 100

# Intraday range relative to previous close
df["intraday_range_pct"] = (
    (df["High"] - df["Low"]) / df["previous_close"]
) * 100

# Check OHLC consistency
df["invalid_ohlc"] = (
    (df["High"] < df[["Open", "Close"]].max(axis=1))
    | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    | (df["High"] < df["Low"])
)

print("\nDATA QUALITY SUMMARY")
print("--------------------")
print("Total records:", len(df))
print("Invalid OHLC records:", df["invalid_ohlc"].sum())

print(
    "Daily returns above 3%:",
    (df["daily_log_return_pct"].abs() > 3).sum()
)

print(
    "Daily returns above 5%:",
    (df["daily_log_return_pct"].abs() > 5).sum()
)

# Largest absolute daily movements
top_moves = df.loc[
    df["daily_log_return_pct"].abs().nlargest(20).index
].copy()

top_moves = top_moves.sort_values(
    "daily_log_return_pct",
    key=lambda x: x.abs(),
    ascending=False
)

columns = [
    "Date",
    "previous_close",
    "Open",
    "High",
    "Low",
    "Close",
    "daily_log_return_pct",
    "overnight_gap_pct",
    "intraday_range_pct",
]

print("\nTOP 20 LARGEST DAILY PRICE MOVEMENTS")
print("------------------------------------")

print(
    top_moves[columns].to_string(
        index=False,
        formatters={
            col: "{:.3f}".format
            for col in columns
            if col != "Date"
        }
    )
)

print("\nOUTLIER ANALYSIS COMPLETED")
