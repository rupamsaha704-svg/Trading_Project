import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - environment validation path
    raise SystemExit(f"pandas is required for data handling: {exc}")

try:
    import MetaTrader5 as mt5
except Exception as exc:  # pragma: no cover - optional MT5 dependency guard
    raise SystemExit(
        "MetaTrader5 is not available. This script is intended for optional MT5 usage only."
    )

print("Connecting...")

if not mt5.initialize():
    print("Connection Failed")
    print(mt5.last_error())
    raise SystemExit(1)

symbol = "XAUUSD"

rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1000)

if rates is None:
    print("No Data Found")
    mt5.shutdown()
    raise SystemExit(1)

df = pd.DataFrame(rates)

df["time"] = pd.to_datetime(df["time"], unit="s")

print(df.head())

df.to_csv("XAUUSD_M5.csv", index=False)

print("Data Saved Successfully!")

mt5.shutdown()