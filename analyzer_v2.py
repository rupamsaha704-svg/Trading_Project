import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

print("Loading XAUUSD Data...")

# Load CSV
df = pd.read_csv("XAUUSD_M5.csv")

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# =========================
# EMA 200
# =========================
ema = EMAIndicator(close=df["close"], window=200)
df["EMA200"] = ema.ema_indicator()

# =========================
# ATR 14
# =========================
atr = AverageTrueRange(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    window=14
)

df["ATR14"] = atr.average_true_range()

print("\nTotal Candles:", len(df))

print("\nLast 10 Candles:\n")

print(df[["time", "close", "EMA200", "ATR14"]].tail(10))

print("\nAnalyzer V2 Working Successfully!")