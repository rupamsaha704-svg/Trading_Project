"""Tests for ICT/SMC engine: Order Blocks, FVG, Liquidity Sweeps, Sessions, Signals."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ict_engine import (
    is_in_killzone,
    add_session_filter,
    detect_order_blocks,
    detect_fvg,
    detect_liquidity_sweeps,
    generate_ict_signals,
    prepare_ict_dataframe,
)


# =============================================================================
# SESSION KILLZONE TESTS
# =============================================================================


class TestKillzones:

    def test_london_open_in_killzone(self):
        assert is_in_killzone(8) is True

    def test_ny_open_in_killzone(self):
        assert is_in_killzone(13) is True

    def test_london_close_in_killzone(self):
        assert is_in_killzone(16) is True

    def test_outside_killzone(self):
        assert is_in_killzone(3) is False
        assert is_in_killzone(22) is False

    def test_add_session_filter_column(self):
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 03:00", "2024-01-01 13:00"]),
            "close": [100, 101, 102],
        })
        result = add_session_filter(df)
        assert "in_killzone" in result.columns
        assert result.iloc[0]["in_killzone"] == True   # 08:00 = London
        assert result.iloc[1]["in_killzone"] == False  # 03:00 = off
        assert result.iloc[2]["in_killzone"] == True   # 13:00 = NY


# =============================================================================
# ORDER BLOCK TESTS
# =============================================================================


class TestOrderBlocks:

    def _make_displacement_df(self):
        """Bearish candle followed by strong bullish displacement."""
        atr = 2.0
        return pd.DataFrame({
            "open":  [100.0, 101.0, 99.0, 98.0, 95.0, 96.0, 100.0],
            "high":  [101.0, 102.0, 100.0, 99.0, 96.0, 104.0, 105.0],
            "low":   [99.0, 100.0, 98.0, 97.0, 94.0, 95.0, 99.0],
            "close": [100.5, 101.5, 98.5, 97.5, 95.5, 103.5, 104.0],  # bar 5 = big up
            "ATR14": [atr] * 7,
        })

    def test_bullish_ob_detected(self):
        df = self._make_displacement_df()
        result = detect_order_blocks(df, lookback=3)
        # Strong bullish bar at index 5 (close-open=7.5 > 1.5*2=3)
        # Last bearish candle before it should be marked
        assert result["BullishOB"].sum() >= 1

    def test_ob_has_high_low(self):
        df = self._make_displacement_df()
        result = detect_order_blocks(df, lookback=3)
        ob_rows = result[result["BullishOB"]]
        if len(ob_rows) > 0:
            assert not pd.isna(ob_rows.iloc[0]["OB_High"])
            assert not pd.isna(ob_rows.iloc[0]["OB_Low"])

    def test_no_ob_without_displacement(self):
        """Flat market should not produce order blocks."""
        df = pd.DataFrame({
            "open":  [100.0] * 10,
            "high":  [100.5] * 10,
            "low":   [99.5] * 10,
            "close": [100.1] * 10,
            "ATR14": [2.0] * 10,
        })
        result = detect_order_blocks(df)
        assert result["BullishOB"].sum() == 0
        assert result["BearishOB"].sum() == 0


# =============================================================================
# FVG TESTS
# =============================================================================


class TestFVG:

    def test_bullish_fvg_detected(self):
        """Gap up: prev.high < next.low."""
        df = pd.DataFrame({
            "open":  [100, 101, 105],
            "high":  [101, 103, 106],  # bar0 high=101
            "low":   [99, 100, 104],   # bar2 low=104 > bar0 high=101 → FVG at bar1
            "close": [100.5, 102, 105],
        })
        result = detect_fvg(df)
        assert result.iloc[1]["BullishFVG"] == True
        assert result.iloc[1]["FVG_Low"] == 101.0   # prev high
        assert result.iloc[1]["FVG_High"] == 104.0  # next low

    def test_bearish_fvg_detected(self):
        """Gap down: prev.low > next.high."""
        df = pd.DataFrame({
            "open":  [105, 103, 99],
            "high":  [106, 104, 100],  # bar2 high=100
            "low":   [104, 102, 98],   # bar0 low=104 > bar2 high=100 → FVG at bar1
            "close": [105, 102.5, 99],
        })
        result = detect_fvg(df)
        assert result.iloc[1]["BearishFVG"] == True

    def test_no_fvg_in_overlapping_candles(self):
        df = pd.DataFrame({
            "open":  [100, 101, 102],
            "high":  [102, 103, 104],
            "low":   [99, 100, 101],
            "close": [101, 102, 103],
        })
        result = detect_fvg(df)
        assert result["BullishFVG"].sum() == 0
        assert result["BearishFVG"].sum() == 0


# =============================================================================
# LIQUIDITY SWEEP TESTS
# =============================================================================


class TestLiquiditySweeps:

    def test_bullish_sweep_detected(self):
        """Price wicks below recent low then closes above."""
        prices_low = [100.0] * 20 + [99.0]  # Last bar wicks to 97 but closes at 100
        df = pd.DataFrame({
            "open":  [100.0] * 20 + [100.0],
            "high":  [101.0] * 20 + [101.0],
            "low":   [99.0] * 20 + [97.0],    # wick below 99 (recent low)
            "close": [100.0] * 20 + [100.5],   # closes above 99
        })
        result = detect_liquidity_sweeps(df, lookback=20)
        assert result.iloc[20]["BullishSweep"] == True

    def test_no_sweep_in_flat_market(self):
        df = pd.DataFrame({
            "open":  [100.0] * 25,
            "high":  [101.0] * 25,
            "low":   [99.0] * 25,
            "close": [100.0] * 25,
        })
        result = detect_liquidity_sweeps(df, lookback=20)
        assert result["BullishSweep"].sum() == 0
        assert result["BearishSweep"].sum() == 0


# =============================================================================
# ICT SIGNAL GENERATION TESTS
# =============================================================================


class TestICTSignals:

    def _make_full_df(self, n=50):
        """Create a DataFrame with all required columns for signal generation."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01 08:00", periods=n, freq="5min"),
            "open":  [2500.0 + i * 0.5 for i in range(n)],
            "high":  [2501.0 + i * 0.5 for i in range(n)],
            "low":   [2499.0 + i * 0.5 for i in range(n)],
            "close": [2500.5 + i * 0.5 for i in range(n)],
            "ATR14": [3.0] * n,
            "ADX14": [25.0] * n,
            "EMA200": [2490.0] * n,
            "BullishBOS": [False] * n,
            "BearishBOS": [False] * n,
        })
        return df

    def test_no_signals_in_flat_market(self):
        df = self._make_full_df()
        df = prepare_ict_dataframe(df)
        signals = generate_ict_signals(df, require_killzone=False)
        # Flat market shouldn't produce OBs/FVGs → no signals
        assert len(signals) == 0

    def test_signal_has_required_fields(self):
        """If a signal is generated, it must have all fields."""
        df = self._make_full_df(60)
        # Inject a bullish OB manually
        df.loc[10, "BullishOB"] = True
        df.loc[10, "OB_High"] = 2504.0
        df.loc[10, "OB_Low"] = 2498.0
        # Create a retest: price dips to OB_High and closes above
        df.loc[20, "low"] = 2503.5
        df.loc[20, "close"] = 2505.0
        df.loc[20, "open"] = 2504.0

        signals = generate_ict_signals(df, require_killzone=False)
        if len(signals) > 0:
            sig = signals[0]
            assert sig.side in ("long", "short")
            assert sig.entry_price > 0
            assert sig.stop_loss > 0
            assert sig.reason != ""
            assert sig.bar_index > 0

    def test_killzone_filter_blocks_off_hours(self):
        """Signals outside killzone should be blocked when require_killzone=True."""
        df = self._make_full_df(30)
        # Set time to 3AM (outside killzone)
        df["time"] = pd.date_range("2024-01-01 03:00", periods=30, freq="5min")
        df = prepare_ict_dataframe(df)
        signals = generate_ict_signals(df, require_killzone=True)
        assert len(signals) == 0

    def test_prepare_ict_dataframe_adds_all_columns(self):
        df = self._make_full_df()
        result = prepare_ict_dataframe(df)
        expected_cols = ["in_killzone", "BullishOB", "BearishOB", "BullishFVG",
                         "BearishFVG", "BullishSweep", "BearishSweep"]
        for col in expected_cols:
            assert col in result.columns
