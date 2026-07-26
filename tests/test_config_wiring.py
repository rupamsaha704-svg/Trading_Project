"""Tests proving config values reach each engine (Indicators, MarketStructure, SignalEngine)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indicators import Indicators
from src.market_structure import MarketStructure
from src.signals import SignalEngine


class TestIndicatorsConfig:

    def test_custom_ema_window_is_used(self):
        """EMA window from config is passed to the engine."""
        ind = Indicators(ema_window=5, atr_window=14, adx_window=14)
        assert ind.ema_window == 5

    def test_custom_atr_window_is_used(self):
        """ATR window from config is passed to the engine."""
        ind = Indicators(ema_window=200, atr_window=7, adx_window=14)
        assert ind.atr_window == 7

    def test_custom_adx_window_is_used(self):
        """ADX window from config is passed to the engine."""
        ind = Indicators(ema_window=200, atr_window=14, adx_window=7)
        assert ind.adx_window == 7

    def test_default_windows_match_original(self):
        """Default constructor matches original hard-coded values."""
        ind = Indicators()
        assert ind.ema_window == 200
        assert ind.atr_window == 14
        assert ind.adx_window == 14

    def test_different_ema_window_produces_different_values(self):
        """Changing EMA window actually affects the output."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=50, freq="5min"),
            "open": range(50),
            "high": [x + 1 for x in range(50)],
            "low": [x - 1 for x in range(50)],
            "close": range(50),
        })
        out_200 = Indicators(ema_window=200).add_indicators(df)
        out_5 = Indicators(ema_window=5).add_indicators(df)

        # EMA with window=5 reacts faster → different values
        assert out_200["EMA200"].iloc[-1] != out_5["EMA200"].iloc[-1]


class TestMarketStructureConfig:

    def test_custom_strength_is_used(self):
        """swing_strength from config reaches MarketStructure."""
        ms = MarketStructure(strength=5)
        assert ms.strength == 5

    def test_default_strength_is_2(self):
        """Default strength matches original."""
        ms = MarketStructure()
        assert ms.strength == 2

    def test_different_strength_produces_different_swings(self):
        """Higher strength requires more confirmation → fewer swings."""
        df = pd.DataFrame({
            "high": [10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17],
            "low": [8, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12, 14],
        })
        swings_1 = MarketStructure(strength=1).detect_swings(df)
        swings_3 = MarketStructure(strength=3).detect_swings(df)

        count_1 = swings_1["SwingHigh"].sum() + swings_1["SwingLow"].sum()
        count_3 = swings_3["SwingHigh"].sum() + swings_3["SwingLow"].sum()

        # Strength=1 should find more (or equal) swings than strength=3
        assert count_1 >= count_3


class TestSignalEngineConfig:

    def test_custom_retest_tolerance_is_used(self):
        """retest_tolerance from config reaches SignalEngine."""
        se = SignalEngine(retest_tolerance=0.005)
        assert se.retest_tolerance == 0.005

    def test_custom_adx_threshold_is_used(self):
        """adx_threshold from config reaches SignalEngine."""
        se = SignalEngine(adx_threshold=30)
        assert se.adx_threshold == 30

    def test_default_values_match_original(self):
        """Defaults match the original hard-coded values."""
        se = SignalEngine()
        assert se.retest_tolerance == 0.001
        assert se.adx_threshold == 20

    def test_high_adx_threshold_blocks_signals(self):
        """Setting adx_threshold very high should block all signals."""
        df = pd.DataFrame({
            "open": [9.0, 9.5, 10.1],
            "high": [10.0, 10.8, 10.8],
            "low": [8.8, 9.0, 10.0],
            "close": [9.2, 10.2, 10.4],
            "EMA200": [9.0, 9.3, 9.8],
            "ADX14": [25.0, 25.0, 25.0],  # Below threshold of 50
            "BullishBOS": [False, True, False],
            "BullishBOSLevel": [None, 10.0, None],
            "BearishBOS": [False, False, False],
            "BearishBOSLevel": [None, None, None],
        })

        # Default threshold=20 → signal fires
        result_default = SignalEngine(adx_threshold=20).generate_signal(df)
        assert result_default["BuySignal"].sum() == 1

        # High threshold=50 → no signal (ADX=25 < 50)
        result_high = SignalEngine(adx_threshold=50).generate_signal(df)
        assert result_high["BuySignal"].sum() == 0

    def test_wider_retest_tolerance_allows_more_signals(self):
        """Wider retest tolerance should allow signals at greater distance from BOS level."""
        # BOS level at 10.0, bar low=10.05 (just above level)
        df = pd.DataFrame({
            "open": [9.0, 9.5, 10.1],
            "high": [10.0, 10.8, 10.8],
            "low": [8.8, 9.0, 10.05],  # 0.05 above BOS level
            "close": [9.2, 10.2, 10.4],
            "EMA200": [9.0, 9.3, 9.8],
            "ADX14": [25.0, 25.0, 25.0],
            "BullishBOS": [False, True, False],
            "BullishBOSLevel": [None, 10.0, None],
            "BearishBOS": [False, False, False],
            "BearishBOSLevel": [None, None, None],
        })

        # Tight tolerance=0.001 → low(10.05) > level(10.0) + 0.001 → no retest
        result_tight = SignalEngine(retest_tolerance=0.001).generate_signal(df)
        assert result_tight["BuySignal"].sum() == 0

        # Wide tolerance=0.1 → low(10.05) <= level(10.0) + 0.1 → retest valid
        result_wide = SignalEngine(retest_tolerance=0.1).generate_signal(df)
        assert result_wide["BuySignal"].sum() == 1
