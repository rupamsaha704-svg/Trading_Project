"""
ICT Strategy Backtest Engine — GFT $5K Challenge Simulator

Simulates trading the ICT/SMC strategy with full prop firm rule enforcement.
Tracks: equity, daily P&L, trading days, margin usage, hold times.

Flow:
1. Prepare data (indicators + ICT structures)
2. Generate ICT signals
3. Walk forward bar-by-bar, executing trades with:
   - Position sizing via calculate_safe_lot_size()
   - Pre-trade compliance check
   - SL/TP management with trailing stop
   - Daily drawdown tracking + reset
   - Overall drawdown floor monitoring
4. Stop if target hit OR account breached
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date as dt_date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop_firm_rules import GFTRules, check_trade_compliance, calculate_safe_lot_size
from src.ict_engine import (
    generate_ict_signals,
    prepare_ict_dataframe,
    ICTSignal,
)


@dataclass
class TradeRecord:
    bar_entry: int
    bar_exit: int
    side: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    hold_bars: int
    reason_entry: str
    reason_exit: str  # "tp", "sl", "trailing", "end_of_data"


@dataclass
class ChallengeResult:
    """Result of a simulated GFT challenge attempt."""
    phase: str  # "step1", "step2", "funded"
    passed: bool
    breached: bool
    breach_reason: str = ""
    starting_balance: float = 5_000.0
    final_balance: float = 5_000.0
    net_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    trading_days: int = 0
    valid_trading_days: int = 0
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)


def run_ict_challenge(
    df: pd.DataFrame,
    phase: str = "step1",
    rules: Optional[GFTRules] = None,
    risk_reward: float = 3.0,
    trailing_activation_rr: float = 1.5,
    max_bars_in_trade: int = 60,  # 5 hours max hold on M5
    require_killzone: bool = True,
    adx_threshold: float = 20.0,
) -> ChallengeResult:
    """Simulate a GFT challenge phase using ICT signals.

    Args:
        df: Prepared DataFrame (with ICT structures, indicators, signals).
        phase: "step1" (10% target), "step2" (5% target), or "funded".
        rules: GFT rules (defaults to $5K standard).
        risk_reward: Target RR ratio for TP.
        trailing_activation_rr: RR at which trailing stop activates.
        max_bars_in_trade: Maximum bars before forced exit.

    Returns:
        ChallengeResult with full trade log and metrics.
    """
    if rules is None:
        rules = GFTRules()

    # Determine target
    if phase == "step1":
        target = rules.step1_target_amount()
    elif phase == "step2":
        target = rules.step2_target_amount()
    else:
        target = rules.stretch_profit_target

    balance = rules.account_balance
    peak_balance = balance
    floor = rules.max_overall_drawdown_amount()

    equity_curve = [balance]
    trade_log: List[Dict[str, Any]] = []
    daily_pnl: Dict[str, float] = {}

    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    current_day = None
    daily_loss_today = 0.0

    # Generate signals
    signals = generate_ict_signals(df, adx_threshold=adx_threshold, require_killzone=require_killzone)

    # Index signals by bar
    signal_map: Dict[int, ICTSignal] = {s.bar_index: s for s in signals}

    breached = False
    breach_reason = ""

    i = 1
    while i < len(df) - 1:
        current = df.iloc[i]

        # --- Daily reset ---
        if "time" in df.columns:
            bar_day = str(pd.Timestamp(current["time"]).date())
            if bar_day != current_day:
                if current_day is not None and current_day not in daily_pnl:
                    daily_pnl[current_day] = 0.0
                current_day = bar_day
                daily_loss_today = 0.0

        # --- Check if target reached ---
        if balance >= rules.account_balance + target:
            break

        # --- Check if breached ---
        if balance <= floor:
            breached = True
            breach_reason = "max_overall_drawdown"
            break

        # --- Check for signal at this bar ---
        if i not in signal_map:
            i += 1
            continue

        sig = signal_map[i]

        # --- Position sizing ---
        stop_distance = abs(sig.entry_price - sig.stop_loss)
        if stop_distance <= 0:
            i += 1
            continue

        lot_size = calculate_safe_lot_size(
            rules,
            account_equity=balance,
            stop_distance_points=stop_distance,
            gold_price=sig.entry_price,
        )

        # Dollar risk for this trade
        dollar_per_point_per_lot = 100.0  # Gold: 1 lot = 100 oz
        trade_risk = stop_distance * dollar_per_point_per_lot * lot_size

        # --- Pre-trade compliance ---
        compliance = check_trade_compliance(
            rules,
            lot_size=lot_size,
            hold_time_minutes=10,  # estimate
            daily_loss_so_far=daily_loss_today,
            account_equity=balance,
            trade_risk=trade_risk,
        )
        if not compliance.is_compliant:
            i += 1
            continue

        # --- Execute trade ---
        entry_price = sig.entry_price
        sl = sig.stop_loss
        tp_distance = stop_distance * risk_reward
        if sig.side == "long":
            tp = entry_price + tp_distance
        else:
            tp = entry_price - tp_distance

        trailing_level = sl
        trailing_activated = False

        exit_bar = None
        exit_price = None
        exit_reason = ""

        for j in range(i + 1, min(i + max_bars_in_trade + 1, len(df))):
            bar = df.iloc[j]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            if sig.side == "long":
                # Update trailing stop
                unrealized_rr = (bar_high - entry_price) / stop_distance if stop_distance > 0 else 0
                if unrealized_rr >= trailing_activation_rr and not trailing_activated:
                    trailing_activated = True
                    trailing_level = entry_price  # Move to break-even
                if trailing_activated:
                    new_trail = bar_high - stop_distance * 0.5
                    if new_trail > trailing_level:
                        trailing_level = new_trail

                # Check SL / trailing
                if bar_low <= trailing_level:
                    exit_price = trailing_level
                    exit_reason = "trailing" if trailing_activated else "sl"
                    exit_bar = j
                    break
                # Check TP
                if bar_high >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    exit_bar = j
                    break
            else:  # short
                unrealized_rr = (entry_price - bar_low) / stop_distance if stop_distance > 0 else 0
                if unrealized_rr >= trailing_activation_rr and not trailing_activated:
                    trailing_activated = True
                    trailing_level = entry_price
                if trailing_activated:
                    new_trail = bar_low + stop_distance * 0.5
                    if new_trail < trailing_level:
                        trailing_level = new_trail

                if bar_high >= trailing_level:
                    exit_price = trailing_level
                    exit_reason = "trailing" if trailing_activated else "sl"
                    exit_bar = j
                    break
                if bar_low <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    exit_bar = j
                    break

        # Force exit if max bars reached
        if exit_bar is None:
            exit_bar = min(i + max_bars_in_trade, len(df) - 1)
            exit_price = float(df.iloc[exit_bar]["close"])
            exit_reason = "max_hold"

        # Calculate PnL
        if sig.side == "long":
            pnl = (exit_price - entry_price) * dollar_per_point_per_lot * lot_size
        else:
            pnl = (entry_price - exit_price) * dollar_per_point_per_lot * lot_size

        balance += pnl
        equity_curve.append(balance)

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)
            daily_loss_today += abs(pnl)

        # Track daily PnL
        if current_day:
            daily_pnl[current_day] = daily_pnl.get(current_day, 0.0) + pnl

        # Track peak
        if balance > peak_balance:
            peak_balance = balance

        trade_log.append({
            "bar_entry": i,
            "bar_exit": exit_bar,
            "side": sig.side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "lot_size": lot_size,
            "pnl": pnl,
            "hold_bars": exit_bar - i,
            "reason_entry": sig.reason,
            "reason_exit": exit_reason,
        })

        # --- Check daily DD breach ---
        if daily_loss_today >= rules.daily_drawdown_amount():
            breached = True
            breach_reason = "daily_drawdown"
            break

        # --- Check overall DD breach ---
        if balance <= floor:
            breached = True
            breach_reason = "max_overall_drawdown"
            break

        # Continue from exit bar
        i = exit_bar + 1

    # --- Compute metrics ---
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown
    eq = pd.Series(equity_curve)
    running_max = eq.cummax()
    dd = eq - running_max
    max_dd = float(dd.min())
    max_dd_pct = (max_dd / rules.account_balance * 100) if rules.account_balance > 0 else 0.0

    # Valid trading days (0.5% profit minimum)
    min_day_profit = rules.account_balance * (rules.min_day_profit_pct / 100.0)
    valid_days = sum(1 for pnl in daily_pnl.values() if pnl >= min_day_profit)

    passed = (
        not breached
        and balance >= rules.account_balance + target
        and valid_days >= rules.min_trading_days
    )

    return ChallengeResult(
        phase=phase,
        passed=passed,
        breached=breached,
        breach_reason=breach_reason,
        starting_balance=rules.account_balance,
        final_balance=balance,
        net_pnl=balance - rules.account_balance,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        trading_days=len(daily_pnl),
        valid_trading_days=valid_days,
        equity_curve=equity_curve,
        trade_log=trade_log,
        daily_pnl=daily_pnl,
    )


def prepare_ict_challenge_data(csv_path: str, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Full pipeline: load → indicators → market structure → ICT structures."""
    from src.utils import load_data
    from src.indicators import Indicators
    from src.market_structure import MarketStructure

    ind_cfg = (config or {}).get("indicators", {})
    sig_cfg = (config or {}).get("signals", {})

    df = load_data(csv_path)
    df = Indicators(
        ema_window=ind_cfg.get("ema_window", 200),
        atr_window=ind_cfg.get("atr_window", 14),
        adx_window=ind_cfg.get("adx_window", 14),
    ).add_indicators(df)
    df = MarketStructure(strength=sig_cfg.get("swing_strength", 2)).detect_swings(df)
    df = MarketStructure(strength=sig_cfg.get("swing_strength", 2)).detect_bos(df)
    df = prepare_ict_dataframe(df)
    return df
