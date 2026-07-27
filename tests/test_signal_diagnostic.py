"""Tests for signal funnel diagnostic counters and exports."""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signal_diagnostic import (
    FunnelDiagnostic,
    run_signal_diagnostic,
    add_risk_blocked_count,
    export_diagnostic,
)


def _make_bos_df():
    """DataFrame with one bullish BOS that leads to a buy signal."""
    return pd.DataFrame({
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
    })


def _make_filtered_df():
    """DataFrame where retest passes but EMA filter blocks."""
    return pd.DataFrame({
        "open": [9.0, 9.5, 10.1],
        "high": [10.0, 10.8, 10.8],
        "low": [8.8, 9.0, 10.0],
        "close": [9.2, 10.2, 10.4],
        "EMA200": [9.0, 9.3, 11.0],  # EMA above close → blocks
        "ADX14": [25.0, 25.0, 25.0],
        "BullishBOS": [False, True, False],
        "BullishBOSLevel": [None, 10.0, None],
        "BearishBOS": [False, False, False],
        "BearishBOSLevel": [None, None, None],
    })


def _make_adx_blocked_df():
    """DataFrame where retest and EMA pass but ADX is too low."""
    return pd.DataFrame({
        "open": [9.0, 9.5, 10.1],
        "high": [10.0, 10.8, 10.8],
        "low": [8.8, 9.0, 10.0],
        "close": [9.2, 10.2, 10.4],
        "EMA200": [9.0, 9.3, 9.8],
        "ADX14": [25.0, 25.0, 15.0],  # ADX below threshold
        "BullishBOS": [False, True, False],
        "BullishBOSLevel": [None, 10.0, None],
        "BearishBOS": [False, False, False],
        "BearishBOSLevel": [None, None, None],
    })


class TestFunnelDiagnostic:

    def test_total_candles_counted(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.total_candles == 3

    def test_bos_buy_candidates(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.bos_buy_candidates == 1

    def test_bos_sell_candidates_zero(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.bos_sell_candidates == 0

    def test_retest_confirmed_buy(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.retest_confirmed_buy == 1

    def test_ema_filter_passed_buy(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.ema_filter_passed_buy == 1

    def test_adx_filter_passed_buy(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.adx_filter_passed_buy == 1

    def test_final_buy_signals(self):
        df = _make_bos_df()
        diag = run_signal_diagnostic(df)
        assert diag.final_buy_signals == 1

    def test_ema_blocks_signal(self):
        """When EMA filter fails, no final signal produced."""
        df = _make_filtered_df()
        diag = run_signal_diagnostic(df)
        assert diag.retest_confirmed_buy == 1
        assert diag.ema_filter_passed_buy == 0
        assert diag.final_buy_signals == 0

    def test_adx_blocks_signal(self):
        """When ADX is below threshold, signal blocked."""
        df = _make_adx_blocked_df()
        diag = run_signal_diagnostic(df)
        assert diag.retest_confirmed_buy == 1
        assert diag.ema_filter_passed_buy == 1
        assert diag.adx_filter_passed_buy == 0
        assert diag.final_buy_signals == 0

    def test_no_bos_no_signals(self):
        """Without any BOS, all counters should be zero (except total_candles)."""
        df = pd.DataFrame({
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "EMA200": [10.0, 10.5, 11.0],
            "ADX14": [25.0, 25.0, 25.0],
            "BullishBOS": [False, False, False],
            "BullishBOSLevel": [None, None, None],
            "BearishBOS": [False, False, False],
            "BearishBOSLevel": [None, None, None],
        })
        diag = run_signal_diagnostic(df)
        assert diag.total_candles == 3
        assert diag.bos_buy_candidates == 0
        assert diag.final_buy_signals == 0
        assert diag.final_sell_signals == 0


class TestSellFunnel:

    def test_sell_bos_counted(self):
        df = pd.DataFrame({
            "open": [11.0, 10.5, 9.9],
            "high": [11.5, 11.0, 10.0],
            "low": [10.5, 10.0, 9.5],
            "close": [10.8, 9.8, 9.6],
            "EMA200": [11.0, 10.8, 10.5],
            "ADX14": [25.0, 25.0, 25.0],
            "BullishBOS": [False, False, False],
            "BullishBOSLevel": [None, None, None],
            "BearishBOS": [False, True, False],
            "BearishBOSLevel": [None, 10.0, None],
        })
        diag = run_signal_diagnostic(df)
        assert diag.bos_sell_candidates == 1

    def test_sell_signal_produced(self):
        """Full sell funnel passes → final_sell_signals = 1."""
        df = pd.DataFrame({
            "open": [11.0, 10.5, 10.1],
            "high": [11.5, 11.0, 10.0],  # high touches BOS level
            "low": [10.5, 10.0, 9.5],
            "close": [10.8, 9.8, 9.6],   # close < BOS level, < open, < EMA
            "EMA200": [11.0, 10.8, 10.5],
            "ADX14": [25.0, 25.0, 25.0],
            "BullishBOS": [False, False, False],
            "BullishBOSLevel": [None, None, None],
            "BearishBOS": [False, True, False],
            "BearishBOSLevel": [None, 10.0, None],
        })
        diag = run_signal_diagnostic(df)
        assert diag.final_sell_signals == 1


class TestRiskBlocked:

    def test_risk_blocked_count(self):
        diag = FunnelDiagnostic(final_buy_signals=5, final_sell_signals=3)
        trade_log = [{"result": "WIN"}] * 6  # 6 executed
        updated = add_risk_blocked_count(diag, trade_log, total_signals=8)
        assert updated.trades_blocked_by_risk == 2

    def test_no_blocked_when_all_executed(self):
        diag = FunnelDiagnostic(final_buy_signals=3, final_sell_signals=2)
        trade_log = [{"result": "WIN"}] * 5
        updated = add_risk_blocked_count(diag, trade_log, total_signals=5)
        assert updated.trades_blocked_by_risk == 0


class TestExportDiagnostic:

    def test_export_creates_json_and_csv(self, tmp_path):
        diag = FunnelDiagnostic(
            total_candles=1000,
            bos_buy_candidates=40,
            final_buy_signals=10,
        )
        files = export_diagnostic(diag, output_dir=tmp_path, report_date=date(2024, 6, 1))
        assert files["json"].exists()
        assert files["csv"].exists()

    def test_json_contains_all_fields(self, tmp_path):
        diag = FunnelDiagnostic(
            total_candles=500,
            bos_buy_candidates=20,
            bos_sell_candidates=15,
            ema_filter_passed_buy=10,
            adx_filter_passed_buy=8,
            retest_confirmed_buy=12,
            final_buy_signals=8,
            final_sell_signals=5,
            trades_blocked_by_risk=3,
        )
        files = export_diagnostic(diag, output_dir=tmp_path, report_date=date(2024, 6, 1))
        with open(files["json"]) as f:
            data = json.load(f)
        assert data["total_candles"] == 500
        assert data["bos_buy_candidates"] == 20
        assert data["final_buy_signals"] == 8
        assert data["trades_blocked_by_risk"] == 3

    def test_csv_has_single_row(self, tmp_path):
        diag = FunnelDiagnostic(total_candles=100)
        files = export_diagnostic(diag, output_dir=tmp_path, report_date=date(2024, 6, 1))
        df = pd.read_csv(files["csv"])
        assert len(df) == 1
        assert "total_candles" in df.columns

    def test_to_dict_complete(self):
        diag = FunnelDiagnostic(total_candles=50, final_buy_signals=3)
        d = diag.to_dict()
        expected_keys = [
            "total_candles", "bos_buy_candidates", "bos_sell_candidates",
            "ema_filter_passed_buy", "ema_filter_passed_sell",
            "adx_filter_passed_buy", "adx_filter_passed_sell",
            "retest_confirmed_buy", "retest_confirmed_sell",
            "final_buy_signals", "final_sell_signals",
            "trades_blocked_by_risk",
        ]
        for key in expected_keys:
            assert key in d
