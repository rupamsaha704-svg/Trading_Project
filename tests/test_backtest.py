"""Tests for the integrated backtest engine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import run_backtest


def _make_df_with_signal(
    prices: list,
    atr: float,
    buy_at: int = None,
    sell_at: int = None,
):
    """Helper to build a minimal DataFrame with one signal at the specified bar."""
    n = len(prices)
    df = pd.DataFrame({
        "open": prices,
        "high": [p + atr for p in prices],
        "low": [p - atr for p in prices],
        "close": prices,
        "ATR14": [atr] * n,
        "EMA200": [p - 1.0 for p in prices],
        "ADX14": [30.0] * n,
        "BuySignal": [False] * n,
        "SellSignal": [False] * n,
        "BullishBOS": [False] * n,
        "BearishBOS": [False] * n,
    })
    if buy_at is not None:
        df.loc[buy_at, "BuySignal"] = True
    if sell_at is not None:
        df.loc[sell_at, "SellSignal"] = True
    return df


def test_no_signals_produces_no_trades():
    """If there are no signals, backtest should produce zero trades."""
    df = _make_df_with_signal([100.0] * 10, atr=2.0)
    results = run_backtest(df, {"starting_balance": 10000.0})
    assert results["total_trades"] == 0
    assert results["final_balance"] == 10000.0


def test_buy_trade_wins_with_correct_pnl():
    """A buy signal where price rises to TP should produce a win with correct dollar P&L."""
    # ATR=2, multiplier=2 → stop_distance=4
    # spread=0, slippage=0 for clean math
    # entry=100, SL=96, TP=108 (RR=2)
    # position_size = (10000 * 0.01) / 4 = 25
    # Win PnL = 8 * 25 = 200
    prices = [100.0] * 3 + [108.5] * 5  # TP hit at bar 3
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    # Make high of bar 3+ reach TP (108)
    df["high"] = [102.0, 102.0, 102.0, 110.0, 110.0, 110.0, 110.0, 110.0]
    df["low"] = [98.0, 98.0, 98.0, 107.0, 107.0, 107.0, 107.0, 107.0]

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
        "risk_percent": 1.0,
        "risk_reward": 2.0,
        "atr_stop_loss_multiplier": 2.0,
    })

    assert results["wins"] == 1
    assert results["losses"] == 0
    assert results["total_trades"] == 1
    assert abs(results["net_result"] - 200.0) < 0.01


def test_buy_trade_loses_with_correct_pnl():
    """A buy signal where price drops to SL should produce a loss."""
    # entry=100, SL=96, stop_distance=4, position_size=25, loss=-100
    prices = [100.0] * 3 + [95.0] * 5
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    df["high"] = [102.0, 102.0, 102.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    df["low"] = [98.0, 98.0, 98.0, 94.0, 94.0, 94.0, 94.0, 94.0]

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
        "risk_percent": 1.0,
        "risk_reward": 2.0,
        "atr_stop_loss_multiplier": 2.0,
    })

    assert results["wins"] == 0
    assert results["losses"] == 1
    assert abs(results["net_result"] - (-100.0)) < 0.01


def test_same_bar_sl_tp_resolves_as_loss():
    """When both SL and TP are hit in the same bar, conservative rule treats it as a loss."""
    prices = [100.0] * 3 + [100.0] * 5
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    # Bar 2: high reaches TP (108) AND low reaches SL (96) — both hit
    df["high"] = [102.0, 102.0, 110.0, 102.0, 102.0, 102.0, 102.0, 102.0]
    df["low"] = [98.0, 98.0, 94.0, 98.0, 98.0, 98.0, 98.0, 98.0]

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
    })

    assert results["losses"] == 1
    assert results["wins"] == 0
    assert results["trade_log"][0]["note"] == "same-bar SL/TP → conservative loss"


def test_spread_and_slippage_affect_entry():
    """Spread and slippage should widen the effective entry, reducing profitability."""
    prices = [100.0] * 3 + [108.5] * 5
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    df["high"] = [102.0, 102.0, 102.0, 112.0, 112.0, 112.0, 112.0, 112.0]
    df["low"] = [98.0, 98.0, 98.0, 107.0, 107.0, 107.0, 107.0, 107.0]

    # Without spread/slippage
    r1 = run_backtest(df.copy(), {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
    })
    # With spread/slippage
    r2 = run_backtest(df.copy(), {
        "starting_balance": 10000.0,
        "spread_points": 0.3,
        "slippage_points": 0.1,
    })

    # Entry is higher with costs → adjusted entry > raw entry
    if r2["total_trades"] > 0:
        assert r2["trade_log"][0]["entry"] > r1["trade_log"][0]["entry"]


def test_equity_curve_tracks_balance():
    """Equity curve should start at starting balance and reflect trades."""
    prices = [100.0] * 3 + [95.0] * 5
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    df["high"] = [102.0, 102.0, 102.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    df["low"] = [98.0, 98.0, 98.0, 94.0, 94.0, 94.0, 94.0, 94.0]

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
    })

    assert results["equity_curve"][0] == 10000.0
    assert results["equity_curve"][-1] == results["final_balance"]


def test_max_drawdown_is_negative_or_zero():
    """Max drawdown should be <= 0 (it represents a loss from peak)."""
    prices = [100.0] * 3 + [95.0] * 5
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    df["high"] = [102.0, 102.0, 102.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    df["low"] = [98.0, 98.0, 98.0, 94.0, 94.0, 94.0, 94.0, 94.0]

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
    })

    assert results["max_drawdown"] <= 0.0


def test_risk_validation_blocks_trade_when_daily_limit_hit():
    """If daily loss limit is already hit, further trades should be blocked."""
    # daily_loss_limit_pct=0.5% of 10000 = 50. If daily_loss is already 50, block.
    # We simulate this by having an extremely low daily_loss_limit_pct
    prices = [100.0] * 3 + [95.0] * 5 + [95.0] * 3 + [90.0] * 5
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 2.0 for p in prices],
        "low": [p - 6.0 for p in prices],  # SL will be hit
        "close": prices,
        "ATR14": [2.0] * len(prices),
        "EMA200": [p - 1.0 for p in prices],
        "ADX14": [30.0] * len(prices),
        "BuySignal": [False] * len(prices),
        "SellSignal": [False] * len(prices),
        "BullishBOS": [False] * len(prices),
        "BearishBOS": [False] * len(prices),
    })
    # Two buy signals
    df.loc[1, "BuySignal"] = True
    df.loc[8, "BuySignal"] = True

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
        "daily_loss_limit_pct": 0.5,  # $50 limit
    })

    # First trade should lose ~$100 (risk 1% of 10000) which exceeds 0.5% limit
    # Second trade should be blocked
    assert results["total_trades"] <= 1


def test_profit_factor_and_expectancy_computed():
    """Verify that profit factor and expectancy are numeric."""
    prices = [100.0] * 3 + [108.5] * 5
    df = _make_df_with_signal(prices, atr=2.0, buy_at=1)
    df["high"] = [102.0, 102.0, 102.0, 110.0, 110.0, 110.0, 110.0, 110.0]
    df["low"] = [98.0, 98.0, 98.0, 107.0, 107.0, 107.0, 107.0, 107.0]

    results = run_backtest(df, {
        "starting_balance": 10000.0,
        "spread_points": 0.0,
        "slippage_points": 0.0,
    })

    assert isinstance(results["profit_factor"], float)
    assert isinstance(results["expectancy"], float)
    assert results["profit_factor"] > 0
