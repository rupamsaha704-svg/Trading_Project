import pandas as pd

print("Loading XAUUSD Data...")

# Load CSV
df = pd.read_csv("XAUUSD_M5.csv")

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# EMA 200
df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

print("\nTotal Candles:", len(df))

print("\nLast 10 Candles:")

print(df[["time","close","EMA200"]].tail(10))

print("\nAnalyzer Working Successfully!")