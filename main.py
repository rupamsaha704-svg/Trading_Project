import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.utils import load_data
    from src.indicators import Indicators
    from src.market_structure import MarketStructure
    from src.signals import SignalEngine
    from src.data_validation import validate_dataset, save_report
except Exception as exc:  # pragma: no cover - import validation path
    raise SystemExit(f"Import initialization failed: {exc}")


print("===================================")
print(" AI TRADING SYSTEM V2")
print("===================================")


# =========================
# 1. Load Data
# =========================

df = load_data("XAUUSD_M5.csv")

print("Total Candles:", len(df))

# =========================
# 1a. Validate Data (read-only)
# =========================

report = validate_dataset(df, "XAUUSD_M5.csv")
report_path = save_report(report, ROOT / "reports" / "data_quality_report.txt")
print("\n========== DATA QUALITY REPORT ==========")
print(report_path)
print("\n" + "\n".join(line for line in str(report_path).splitlines()))
print("\n" + "\n".join(line for line in __import__('src.data_validation', fromlist=['format_report']).format_report(report).splitlines()))


# =========================
# 2. Add Indicators
# =========================

indicator = Indicators()
df = indicator.add_indicators(df)


# =========================
# 3. Detect Market Structure
# =========================

market = MarketStructure()

df = market.detect_swings(df)

df = market.detect_bos(df)


# =========================
# 4. Generate Signals
# =========================

signal_engine = SignalEngine()

df = signal_engine.generate_signal(df)


# =========================
# 5. Count BOS
# =========================

bullish_bos = df["BullishBOS"].sum()
bearish_bos = df["BearishBOS"].sum()


# =========================
# 6. Display Results
# =========================

print("\n========== MARKET STRUCTURE ==========")

print("Bullish BOS:", bullish_bos)
print("Bearish BOS:", bearish_bos)

print("\nLast 20 Rows:\n")

print(
    df[
        [
            "time",
            "close",
            "Structure",
            "BullishBOS",
            "BearishBOS"
        ]
    ].tail(20)
)


print("\n===================================")
print("AI TRADING SYSTEM V2 COMPLETED!")
print("===================================")