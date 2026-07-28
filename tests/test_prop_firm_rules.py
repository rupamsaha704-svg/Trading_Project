"""Tests for GFT prop firm rules compliance module."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop_firm_rules import (
    GFTRules,
    check_trade_compliance,
    calculate_safe_lot_size,
    TradeCompliance,
)


class TestGFTRules:

    def test_default_values(self):
        rules = GFTRules()
        assert rules.account_balance == 5_000.0
        assert rules.daily_drawdown_pct == 5.0
        assert rules.max_overall_drawdown_pct == 10.0
        assert rules.max_lot_size == 0.06
        assert rules.min_hold_time_minutes == 2
        assert rules.no_hedging is True
        assert rules.no_martingale is True

    def test_daily_drawdown_amount(self):
        rules = GFTRules(account_balance=5_000.0)
        assert rules.daily_drawdown_amount() == 250.0  # 5% of 5K

    def test_max_overall_floor(self):
        rules = GFTRules(account_balance=5_000.0)
        assert rules.max_overall_drawdown_amount() == 4_500.0  # 90% of 5K

    def test_step1_target(self):
        rules = GFTRules(account_balance=5_000.0)
        assert rules.step1_target_amount() == 500.0  # 10%

    def test_step2_target(self):
        rules = GFTRules(account_balance=5_000.0)
        assert rules.step2_target_amount() == 250.0  # 5%


class TestTradeCompliance:

    def test_compliant_trade_passes(self):
        rules = GFTRules()
        result = check_trade_compliance(
            rules,
            lot_size=0.03,
            hold_time_minutes=5,
            daily_loss_so_far=0.0,
            account_equity=5_000.0,
            trade_risk=50.0,
        )
        assert result.is_compliant is True
        assert result.violations == []

    def test_lot_size_violation(self):
        rules = GFTRules()
        result = check_trade_compliance(
            rules,
            lot_size=0.10,  # exceeds 0.06
            hold_time_minutes=5,
        )
        assert result.is_compliant is False
        assert any("lot_size" in v for v in result.violations)

    def test_hold_time_violation(self):
        rules = GFTRules()
        result = check_trade_compliance(
            rules,
            lot_size=0.03,
            hold_time_minutes=1,  # below 2 min
        )
        assert result.is_compliant is False
        assert any("hold_time" in v for v in result.violations)

    def test_hedging_violation(self):
        rules = GFTRules()
        result = check_trade_compliance(
            rules,
            lot_size=0.03,
            hold_time_minutes=5,
            is_hedge=True,
        )
        assert result.is_compliant is False
        assert any("hedging" in v for v in result.violations)

    def test_martingale_violation(self):
        rules = GFTRules()
        result = check_trade_compliance(
            rules,
            lot_size=0.03,
            hold_time_minutes=5,
            is_martingale=True,
        )
        assert result.is_compliant is False
        assert any("martingale" in v for v in result.violations)

    def test_daily_drawdown_violation(self):
        rules = GFTRules(account_balance=5_000.0)
        result = check_trade_compliance(
            rules,
            lot_size=0.03,
            hold_time_minutes=5,
            daily_loss_so_far=230.0,  # already lost $230
            trade_risk=30.0,          # this $30 would exceed $250 limit
            account_equity=4_770.0,
        )
        # 230 + 30 > 250 → violation
        assert result.is_compliant is False
        assert any("daily limit" in v for v in result.violations)

    def test_overall_drawdown_violation(self):
        rules = GFTRules(account_balance=5_000.0)
        result = check_trade_compliance(
            rules,
            lot_size=0.03,
            hold_time_minutes=5,
            account_equity=4_520.0,  # close to floor
            trade_risk=30.0,         # would push below 4500
        )
        assert result.is_compliant is False
        assert any("floor" in v for v in result.violations)

    def test_multiple_violations(self):
        rules = GFTRules()
        result = check_trade_compliance(
            rules,
            lot_size=0.10,
            hold_time_minutes=1,
            is_hedge=True,
            is_martingale=True,
        )
        assert result.is_compliant is False
        assert len(result.violations) >= 4


class TestSafeLotSize:

    def test_safe_lot_respects_max(self):
        rules = GFTRules(max_lot_size=0.06)
        lot = calculate_safe_lot_size(rules, account_equity=5_000.0, stop_distance_points=5.0)
        assert lot <= 0.06

    def test_safe_lot_scales_with_equity(self):
        rules = GFTRules()
        lot_high = calculate_safe_lot_size(rules, account_equity=10_000.0, stop_distance_points=5.0)
        lot_low = calculate_safe_lot_size(rules, account_equity=2_000.0, stop_distance_points=5.0)
        assert lot_high >= lot_low

    def test_safe_lot_zero_stop_returns_zero(self):
        rules = GFTRules()
        lot = calculate_safe_lot_size(rules, account_equity=5_000.0, stop_distance_points=0.0)
        assert lot == 0.0

    def test_safe_lot_minimum_is_001(self):
        rules = GFTRules()
        lot = calculate_safe_lot_size(rules, account_equity=5_000.0, stop_distance_points=100.0)
        assert lot >= 0.01

    def test_safe_lot_with_typical_gold_values(self):
        """Typical gold trade: $5K equity, 5-point SL."""
        rules = GFTRules()
        lot = calculate_safe_lot_size(
            rules,
            account_equity=5_000.0,
            stop_distance_points=5.0,
            gold_price=2500.0,
        )
        # 1% risk = $50, 5pts SL, $100/pt/lot → 0.01 lots
        assert 0.01 <= lot <= 0.06
