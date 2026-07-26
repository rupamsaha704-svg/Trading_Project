import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

print("Loading XAUUSD Data...")

# ===========================
# Load CSV
# ===========================
df = pd.read_csv("XAUUSD_M5.csv")
df["time"] = pd.to_datetime(df["time"])

# ===========================
# EMA 200
# ===========================
ema = EMAIndicator(close=df["close"], window=200)
df["EMA200"] = ema.ema_indicator()

# ===========================
# ATR
# ===========================
atr = AverageTrueRange(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    window=14
)
df["ATR14"] = atr.average_true_range()

# ===========================
# ADX
# ===========================
adx = ADXIndicator(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    window=14
)
df["ADX14"] = adx.adx()

# ===========================
# Swing Detection
# ===========================

strength = 2

df["SwingHigh"] = False
df["SwingLow"] = False

for i in range(strength, len(df)-strength):

    high = df.iloc[i]["high"]

    if high > df.iloc[i-1]["high"] and \
       high > df.iloc[i-2]["high"] and \
       high > df.iloc[i+1]["high"] and \
       high > df.iloc[i+2]["high"]:

        df.loc[i, "SwingHigh"] = True

    low = df.iloc[i]["low"]

    if low < df.iloc[i-1]["low"] and \
       low < df.iloc[i-2]["low"] and \
       low < df.iloc[i+1]["low"] and \
       low < df.iloc[i+2]["low"]:

        df.loc[i, "SwingLow"] = True

print()

print(df[[
    "time",
    "close",
    "EMA200",
    "ATR14",
    "ADX14",
    "SwingHigh",
    "SwingLow"
]].tail(30))

print()

print("Analyzer V4 Working Successfully!")