
from pathlib import Path

import pandas as pd
import yfinance as yf


OUTPUT_PATH = Path("data/raw/gold_daily.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Downloading historical gold data...")

df = yf.Ticker("GC=F").history(
    period="max",
    interval="1d",
    auto_adjust=False,
    actions=False
)

if df.empty:
    raise RuntimeError("No data downloaded. Please try again.")

# Clean the date index
df.index = pd.to_datetime(df.index)

if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df.index.name = "Date"
df = df.reset_index()

# Keep the required columns
columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing = [col for col in columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[columns]

# Basic cleaning
df = df.dropna(subset=["Open", "High", "Low", "Close"])
df = df.sort_values("Date")
df = df.drop_duplicates(subset=["Date"])

# Save dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nDownload completed!")
print("Total records:", len(df))
print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())

print("\nFirst five records:")
print(df.head())
