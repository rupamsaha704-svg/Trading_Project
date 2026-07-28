"""
GFT (Goat Funded Trader) 2-Step Standard — Prop Firm Rules Engine

Account: $25,000 (using $25K as trading capital)
Source: https://help.goatfundedtrader.com/en/articles/13575169-2-step-standard

All rules enforced as hard constraints in backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class GFTRules:
    """GFT 2-Step Standard account rules."""

    # Account
    account_balance: float = 25_000.0
    leverage_commodities: float = 20.0       # 1:20 for Gold (commodities) in evaluation
    leverage_funded: float = 10.0            # 1:10 funded phase

    # Drawdown rules (HARD BREACH)
    daily_drawdown_pct: float = 5.0          # 5% of initial balance per day
    max_overall_drawdown_pct: float = 10.0   # 10% static — equity never below 90% of start

    # Profit targets
    step1_profit_target_pct: float = 10.0    # $2,500 on $25K
    step2_profit_target_pct: float = 5.0     # $1,250 on $25K
    funded_daily_profit_cap: float = 3_000.0 # $3,000 daily cap (funded only)

    # Trading rules
    min_trading_days: int = 4                # 4 days per phase (from Jul 25, 2026)
    min_day_profit_pct: float = 0.5          # 0.5% to count as valid trading day

    # User constraints (from your requirements)
    max_lot_size: float = 0.06               # Max lot per trade
    min_hold_time_minutes: int = 2           # Minimum 2 minutes holding
    no_hedging: bool = True                  # No hedging allowed
    no_martingale: bool = True               # No martingale allowed

    # Targets
    minimum_profit_target: float = 15_000.0  # $15K minimum
    stretch_profit_target: float = 30_000.0  # $30K target
    max_margin_usage_pct: float = 80.0       # Keep within 80% margin

    def daily_drawdown_amount(self) -> float:
        """Max dollar loss per day."""
        return self.account_balance * (self.daily_drawdown_pct / 100.0)

    def max_overall_drawdown_amount(self) -> float:
        """Absolute floor — account cannot go below this."""
        return self.account_balance * (1 - self.max_overall_drawdown_pct / 100.0)

    def step1_target_amount(self) -> float:
        return self.account_balance * (self.step1_profit_target_pct / 100.0)

    def step2_target_amount(self) -> float:
        return self.account_balance * (self.step2_profit_target_pct / 100.0)

    def max_position_value(self) -> float:
        """Max position value given leverage and margin constraint."""
        return self.account_balance * self.leverage_commodities * (self.max_margin_usage_pct / 100.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeCompliance:
    """Check a single trade against prop firm rules."""

    is_compliant: bool = True
    violations: List[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


def check_trade_compliance(
    rules: GFTRules,
    lot_size: float,
    hold_time_minutes: float,
    is_hedge: bool = False,
    is_martingale: bool = False,
    daily_loss_so_far: float = 0.0,
    account_equity: float = 25_000.0,
    trade_risk: float = 0.0,
) -> TradeCompliance:
    """Validate a trade against all GFT rules before execution.

    Args:
        rules: GFT rules instance.
        lot_size: Position size in lots.
        hold_time_minutes: Expected or actual hold time.
        is_hedge: Whether this is a hedging trade.
        is_martingale: Whether this uses martingale sizing.
        daily_loss_so_far: Cumulative loss today.
        account_equity: Current account equity.
        trade_risk: Dollar risk on this trade.

    Returns:
        TradeCompliance with violations list.
    """
    compliance = TradeCompliance()

    # Lot size check
    if lot_size > rules.max_lot_size:
        compliance.violations.append(
            f"lot_size={lot_size} exceeds max={rules.max_lot_size}"
        )

    # Hold time check
    if hold_time_minutes < rules.min_hold_time_minutes:
        compliance.violations.append(
            f"hold_time={hold_time_minutes}min below min={rules.min_hold_time_minutes}min"
        )

    # Hedging check
    if is_hedge and rules.no_hedging:
        compliance.violations.append("hedging not allowed")

    # Martingale check
    if is_martingale and rules.no_martingale:
        compliance.violations.append("martingale not allowed")

    # Daily drawdown pre-check
    remaining_daily = rules.daily_drawdown_amount() - daily_loss_so_far
    if trade_risk > remaining_daily:
        compliance.violations.append(
            f"trade_risk=${trade_risk:.2f} would exceed daily limit (remaining=${remaining_daily:.2f})"
        )

    # Overall drawdown pre-check
    floor = rules.max_overall_drawdown_amount()
    if (account_equity - trade_risk) < floor:
        compliance.violations.append(
            f"trade would push equity below floor=${floor:.2f}"
        )

    # Margin check
    # Gold: 1 lot = 100 oz, price ~$2500/oz → $250,000 per lot
    # Margin required = position_value / leverage
    gold_value_per_lot = 100 * 2500  # approximate
    position_value = lot_size * gold_value_per_lot
    margin_required = position_value / rules.leverage_commodities
    margin_pct = (margin_required / account_equity) * 100
    if margin_pct > rules.max_margin_usage_pct:
        compliance.violations.append(
            f"margin_usage={margin_pct:.1f}% exceeds max={rules.max_margin_usage_pct}%"
        )

    compliance.is_compliant = len(compliance.violations) == 0
    return compliance


def calculate_safe_lot_size(
    rules: GFTRules,
    account_equity: float,
    stop_distance_points: float,
    gold_price: float = 2500.0,
) -> float:
    """Calculate the maximum safe lot size given current equity and stop distance.

    Respects:
    - Max lot size (0.06)
    - Daily drawdown limit
    - 80% margin constraint

    Args:
        rules: GFT rules.
        account_equity: Current equity.
        stop_distance_points: SL distance in price points.
        gold_price: Current gold price (for margin calc).

    Returns:
        Safe lot size (capped at max_lot_size).
    """
    if stop_distance_points <= 0:
        return 0.0

    # Risk-based sizing: risk 1% of equity per trade (conservative for prop)
    risk_per_trade = account_equity * 0.01
    # Gold: 1 lot = 100 oz, so $1 move = $100 per lot
    dollar_per_point_per_lot = 100.0
    risk_based_lots = risk_per_trade / (stop_distance_points * dollar_per_point_per_lot)

    # Margin-based cap
    margin_available = account_equity * (rules.max_margin_usage_pct / 100.0)
    gold_value_per_lot = 100 * gold_price
    margin_per_lot = gold_value_per_lot / rules.leverage_commodities
    margin_based_lots = margin_available / margin_per_lot if margin_per_lot > 0 else 0.0

    # Take the minimum of all constraints
    safe_lots = min(risk_based_lots, margin_based_lots, rules.max_lot_size)
    return round(max(0.01, safe_lots), 2)  # Min 0.01 lot
