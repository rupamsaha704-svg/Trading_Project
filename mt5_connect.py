import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import MetaTrader5 as mt5
except Exception as exc:  # pragma: no cover - optional MT5 dependency guard
    raise SystemExit(f"MetaTrader5 is not available: {exc}")

print("Connecting to MT5...")

if not mt5.initialize():
    print("MT5 connection failed")
    print(mt5.last_error())
    raise SystemExit(1)

print("MT5 Connected Successfully!")

account = mt5.account_info()

if account is not None:
    print("Account information available but not displayed in this research-safe script.")
else:
    print("Could not read account information.")

mt5.shutdown()