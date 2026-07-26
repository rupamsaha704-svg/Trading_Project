"""
STRATEGY V3 BACKTEST — Risk-Integrated Version

Same-bar SL/TP Rule (Conservative):
    When both SL and TP are reachable within the same candle, the trade is
    resolved as a LOSS. Rationale: intra-bar price path is unknown; assuming
    the worst case avoids overstating strategy performance.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_data
from src.indicators import Indicators
from src.market_structure import MarketStructure
from src.signals import SignalEngine
from src.risk_manager import RiskManager


# =============================================================================
# CONFIGURATION
# =============================================================================

SETTINGS = {
    "starting_balance": 10_000.0,
    "risk_percent": 1.0,
    "risk_reward": 2.0,
    "atr_stop_loss_multiplier": 2.0,
    "max_concurrent_positions": 1,
    "daily_loss_limit_pct": 3.0,
    "max_drawdown_pct": 5.0,
    "spread_points": 0.30,       # spread in price points (e.g. 30 cents for gold)
    "slippage_points": 0.10,     # slippage in price points
}


# =============================================================================
# HELPERS
# =============================================================================

def _get_bar_date(row) -> Optional[Any]:
    """Extract the date (no time) from a bar row. Returns None if no time column."""
    if "time" in row.index:
        ts = row["time"]
        if isinstance(ts, pd.Timestamp):
            return ts.date()
        try:
            return pd.Timestamp(ts).date()
        except Exception:
            return None
    return None


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run backtest on a prepared DataFrame (indicators + signals already added).

    Trade management rules:
    - Only one position open at a time (max_concurrent_positions enforced).
    - After a trade exits at bar j, the next signal scan resumes from bar j
      (no overlapping trades, no re-scanning bars inside a trade's lifetime).
    - daily_loss resets when the candle date changes.

    Returns a dict with metrics and equity curve.
    """
    cfg = {**SETTINGS, **(settings or {})}

    risk_manager = RiskManager(
        risk_percent=cfg["risk_percent"],
        risk_reward=cfg["risk_reward"],
        atr_stop_loss_multiplier=cfg["atr_stop_loss_multiplier"],
        max_concurrent_positions=cfg["max_concurrent_positions"],
        daily_loss_limit_pct=cfg["daily_loss_limit_pct"],
        max_drawdown_pct=cfg["max_drawdown_pct"],
        account_balance=cfg["starting_balance"],
    )

    spread = cfg["spread_points"]
    slippage = cfg["slippage_points"]

    balance = cfg["starting_balance"]
    peak_equity = balance
    daily_loss = 0.0
    current_positions = 0

    equity_curve: List[float] = [balance]
    trade_log: List[Dict[str, Any]] = []

    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    # Track the current trading day for daily_loss reset
    current_day = None

    # Use index-based iteration so we can skip ahead after a trade exits
    i = 1
    while i < len(df) - 1:
        current = df.iloc[i]

        # --- Daily loss reset on date change ---
        bar_day = _get_bar_date(current)
        if bar_day is not None and bar_day != current_day:
            current_day = bar_day
            daily_loss = 0.0

        entry_price = float(current["close"])
        atr = float(current["ATR14"])

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        is_buy = bool(current["BuySignal"])
        is_sell = bool(current["SellSignal"])

        if not is_buy and not is_sell:
            i += 1
            continue

        # --- Risk validation gate (including position count) ---
        validation = risk_manager.validate_trade(
            account_balance=balance,
            equity=balance,
            current_positions=current_positions,
            daily_loss=daily_loss,
            peak_equity=peak_equity,
        )
        if not validation["allowed"]:
            i += 1
            continue

        # --- Determine side and apply spread + slippage ---
        if is_buy:
            side = "long"
            adjusted_entry = entry_price + spread + slippage
        else:
            side = "short"
            adjusted_entry = entry_price - spread - slippage

        # --- Calculate SL / TP using RiskManager ---
        stop_loss, take_profit = risk_manager.calculate_levels(
            entry_price=adjusted_entry,
            atr=atr,
            side=side,
        )

        # --- Position sizing ---
        position_size = risk_manager.calculate_position_size(
            account_balance=balance,
            entry_price=adjusted_entry,
            stop_loss=stop_loss,
        )

        stop_distance = abs(adjusted_entry - stop_loss)
        tp_distance = abs(take_profit - adjusted_entry)

        # --- Mark position as open ---
        current_positions += 1

        # --- Walk forward to resolve trade ---
        exit_bar = None
        for j in range(i + 1, len(df)):
            future = df.iloc[j]
            future_high = float(future["high"])
            future_low = float(future["low"])

            if side == "long":
                sl_hit = future_low <= stop_loss
                tp_hit = future_high >= take_profit
            else:
                sl_hit = future_high >= stop_loss
                tp_hit = future_low <= take_profit

            # --- Same-bar SL/TP: conservative rule (always assume loss) ---
            if sl_hit and tp_hit:
                pnl = -(stop_distance * position_size)
                losses += 1
                gross_loss += abs(pnl)
                balance += pnl
                daily_loss += abs(pnl)
                trade_log.append({
                    "bar_index": i,
                    "side": side,
                    "entry": adjusted_entry,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "position_size": position_size,
                    "result": "LOSS",
                    "pnl": pnl,
                    "exit_bar": j,
                    "note": "same-bar SL/TP → conservative loss",
                })
                exit_bar = j
                break
            elif sl_hit:
                pnl = -(stop_distance * position_size)
                losses += 1
                gross_loss += abs(pnl)
                balance += pnl
                daily_loss += abs(pnl)
                trade_log.append({
                    "bar_index": i,
                    "side": side,
                    "entry": adjusted_entry,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "position_size": position_size,
                    "result": "LOSS",
                    "pnl": pnl,
                    "exit_bar": j,
                    "note": "",
                })
                exit_bar = j
                break
            elif tp_hit:
                pnl = tp_distance * position_size
                wins += 1
                gross_profit += pnl
                balance += pnl
                trade_log.append({
                    "bar_index": i,
                    "side": side,
                    "entry": adjusted_entry,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "position_size": position_size,
                    "result": "WIN",
                    "pnl": pnl,
                    "exit_bar": j,
                    "note": "",
                })
                exit_bar = j
                break

        # --- Close position ---
        current_positions -= 1

        # Track equity
        equity_curve.append(balance)
        if balance > peak_equity:
            peak_equity = balance

        # --- Continue scanning from the exit bar (prevents overlapping trades) ---
        if exit_bar is not None:
            i = exit_bar
        else:
            # Trade never resolved (ran out of data) — move past entry bar
            i += 1
        i += 1

    # ==========================================================================
    # PERFORMANCE METRICS
    # ==========================================================================

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_win = (gross_profit / wins) if wins > 0 else 0.0
    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
    expectancy = (
        (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    ) if total_trades > 0 else 0.0

    # Max drawdown from equity curve
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdowns = equity_series - running_max
    max_drawdown = float(drawdowns.min())
    max_drawdown_pct = (
        (max_drawdown / running_max[drawdowns.idxmin()] * 100)
        if running_max[drawdowns.idxmin()] != 0
        else 0.0
    )

    net_result = balance - cfg["starting_balance"]

    return {
        "starting_balance": cfg["starting_balance"],
        "final_balance": balance,
        "net_result": net_result,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "settings": cfg,
    }


def prepare_dataframe(csv_path: str = "XAUUSD_M5.csv") -> pd.DataFrame:
    """Load data and add indicators + signals (shared pipeline)."""
    df = load_data(csv_path)
    df = Indicators().add_indicators(df)
    df = MarketStructure().detect_swings(df)
    df = MarketStructure().detect_bos(df)
    df = SignalEngine().generate_signal(df)
    return df


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print backtest results to console."""
    print("\n===================================")
    print(" STRATEGY V3 BACKTEST RESULTS")
    print("===================================")
    print(f"Starting Balance:  {results['starting_balance']:.2f}")
    print(f"Final Balance:     {results['final_balance']:.2f}")
    print(f"Net Result:        {results['net_result']:.2f}")
    print(f"Total Trades:      {results['total_trades']}")
    print(f"Wins:              {results['wins']}")
    print(f"Losses:            {results['losses']}")
    print(f"Win Rate:          {results['win_rate']:.2f}%")
    print(f"Profit Factor:     {results['profit_factor']:.2f}")
    print(f"Expectancy:        {results['expectancy']:.2f}")
    print(f"Gross Profit:      {results['gross_profit']:.2f}")
    print(f"Gross Loss:        {results['gross_loss']:.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.2f}")
    print(f"Max Drawdown %:    {results['max_drawdown_pct']:.2f}%")
    print("===================================")
    print()
    print("Same-bar SL/TP Rule: CONSERVATIVE (assume loss)")
    print(f"Spread:            {results['settings']['spread_points']} pts")
    print(f"Slippage:          {results['settings']['slippage_points']} pts")
    print("===================================")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("===================================")
    print(" STRATEGY V3 BACKTEST")
    print("===================================")

    df = prepare_dataframe("XAUUSD_M5.csv")
    print(f"Total Candles: {len(df)}")

    results = run_backtest(df)
    print_results(results)
