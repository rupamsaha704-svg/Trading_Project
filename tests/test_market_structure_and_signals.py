import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.market_structure import MarketStructure
from src.signals import SignalEngine


def test_swings_detect_using_causal_past_only():
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 9.0, 13.0, 11.0],
            "low": [8.0, 7.0, 6.0, 5.0, 4.0],
        }
    )

    result = MarketStructure(strength=1).detect_swings(df)

    assert bool(result.loc[1, "SwingHigh"])
    assert bool(result.loc[3, "SwingHigh"])
    assert not bool(result.loc[1, "SwingLow"])
    assert not bool(result.loc[3, "SwingLow"])


def test_bos_detection_marks_close_breaks_after_swing():
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 13.0, 12.5],
            "low": [8.0, 9.0, 10.0, 9.5],
            "close": [9.0, 11.0, 13.5, 12.0],
        }
    )

    result = MarketStructure(strength=1).detect_swings(df)
    result = MarketStructure(strength=1).detect_bos(result)

    assert bool(result.loc[2, "SwingHigh"])
    assert bool(result.loc[2, "BullishBOS"])
    assert result.loc[2, "BullishBOSLevel"] == 13.0


def test_signal_engine_delays_signal_until_after_bos_confirmation():
    df = pd.DataFrame(
        {
            "open": [9.0, 9.5, 10.1],
            "high": [10.0, 10.8, 10.8],
            "low": [8.8, 9.0, 10.0],
            "close": [9.2, 10.2, 10.4],
            "EMA200": [9.0, 9.3, 9.8],
            "ADX14": [25.0, 25.0, 25.0],
            "BullishBOS": [False, True, False],
            "BullishBOSLevel": [None, 10.0, None],
            "BearishBOS": [False, False, False],
            "BearishBOSLevel": [None, None, None],
        }
    )

    result = SignalEngine().generate_signal(df)

    assert not bool(result.loc[1, "BuySignal"])
    assert bool(result.loc[2, "BuySignal"])


def test_signal_engine_prevents_duplicate_signals():
    df = pd.DataFrame(
        {
            "open": [9.0, 9.5, 10.1, 10.2],
            "high": [10.0, 10.8, 10.8, 10.9],
            "low": [8.8, 9.0, 10.0, 10.0],
            "close": [9.2, 10.2, 10.4, 10.5],
            "EMA200": [9.0, 9.3, 9.8, 10.0],
            "ADX14": [25.0, 25.0, 25.0, 25.0],
            "BullishBOS": [False, True, False, False],
            "BullishBOSLevel": [None, 10.0, None, None],
            "BearishBOS": [False, False, False, False],
            "BearishBOSLevel": [None, None, None, None],
        }
    )

    result = SignalEngine().generate_signal(df)

    assert result["BuySignal"].sum() == 1


def test_no_future_data_leakage_in_structure_detection():
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 9.0],
            "low": [8.0, 7.0, 6.0],
            "close": [9.0, 11.0, 8.0],
        }
    )

    result = MarketStructure(strength=1).detect_swings(df)

    assert bool(result.loc[1, "SwingHigh"])
    assert not bool(result.loc[0, "SwingHigh"])
