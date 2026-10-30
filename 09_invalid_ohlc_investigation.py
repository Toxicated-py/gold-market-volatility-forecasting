
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "gold_daily.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Check each OHLC consistency rule separately.
df["high_below_open"] = df["High"] < df["Open"]
df["high_below_close"] = df["High"] < df["Close"]
df["low_above_open"] = df["Low"] > df["Open"]
df["low_above_close"] = df["Low"] > df["Close"]
df["high_below_low"] = df["High"] < df["Low"]

checks = [
    "high_below_open",
    "high_below_close",
    "low_above_open",
    "low_above_close",
    "high_below_low",
]

df["invalid_ohlc"] = df[checks].any(axis=1)

print("\nOHLC VIOLATION COUNTS")
print("---------------------")

for check in checks:
    print(f"{check}: {df[check].sum()}")

print("\nTOTAL INVALID RECORDS")
print("---------------------")
print(df["invalid_ohlc"].sum())

invalid = df[df["invalid_ohlc"]].copy()

columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    *checks,
]

print("\nFIRST 30 INVALID RECORDS")
print("-------------------------")
print(invalid[columns].head(30).to_string(index=False))

print("\nLATEST 30 INVALID RECORDS")
print("-------------------------")
print(invalid[columns].tail(30).to_string(index=False))

print("\nINVALID RECORD DATE RANGE")
print("-------------------------")
if not invalid.empty:
    print("First:", invalid["Date"].min())
    print("Last:", invalid["Date"].max())

print("\nINVESTIGATION COMPLETED")
