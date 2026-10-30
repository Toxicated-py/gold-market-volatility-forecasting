
import pandas as pd

df = pd.read_csv("data/raw/gold_daily.csv", parse_dates=["Date"])

# Identify records with identical OHLC values
flat_ohlc = (
    (df["Open"] == df["High"])
    & (df["High"] == df["Low"])
    & (df["Low"] == df["Close"])
)

# Identify zero-volume records
zero_volume = df["Volume"] == 0

# Identify unchanged closing prices
same_close = df["Close"].eq(df["Close"].shift(1))

print("DATA QUALITY INVESTIGATION")
print("--------------------------")

print("Flat OHLC records:", flat_ohlc.sum())
print("Zero volume records:", zero_volume.sum())
print("Unchanged close from previous row:", same_close.sum())

print("\nFlat OHLC AND zero volume:")
print((flat_ohlc & zero_volume).sum())

print("\nFlat OHLC BUT non-zero volume:")
print((flat_ohlc & ~zero_volume).sum())

print("\nZero volume BUT non-flat OHLC:")
print((zero_volume & ~flat_ohlc).sum())

print("\nFIRST 20 FLAT OHLC RECORDS:")
print(
    df.loc[
        flat_ohlc,
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].head(20).to_string(index=False)
)

print("\nLAST 20 FLAT OHLC RECORDS:")
print(
    df.loc[
        flat_ohlc,
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].tail(20).to_string(index=False)
)

print("\nCONSECUTIVE UNCHANGED CLOSES:")
groups = (~same_close).cumsum()
runs = same_close.groupby(groups).sum()

print("Longest consecutive run:", int(runs.max()))
