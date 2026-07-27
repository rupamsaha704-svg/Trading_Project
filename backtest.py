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
from src.report_export import export_backtest_report
from src.config_loader import load_config, get_backtest_settings, validate_config
from src.logger import setup_logger


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

    Args:
        df: DataFrame with indicators and signals already applied.
        settings: Flat settings dict (typically from get_backtest_settings()).
                  If None, loads from strategy_config.json or uses defaults.

    Returns a dict with metrics and equity curve.
    """
    # Build cfg: start from config-file defaults, then overlay any explicit settings
    config = load_config(ROOT / "strategy_config.json")
    cfg = get_backtest_settings(config)
    if settings is not None:
        cfg.update(settings)

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


def prepare_dataframe(csv_path: str = "XAUUSD_M5.csv", config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load data and add indicators + signals (shared pipeline).

    Args:
        csv_path: Path to CSV data file.
        config: Full strategy config dict. If None, uses defaults.
    """
    ind_cfg = (config or {}).get("indicators", {})
    sig_cfg = (config or {}).get("signals", {})

    df = load_data(csv_path)
    df = Indicators(
        ema_window=ind_cfg.get("ema_window", 200),
        atr_window=ind_cfg.get("atr_window", 14),
        adx_window=ind_cfg.get("adx_window", 14),
    ).add_indicators(df)
    df = MarketStructure(
        strength=sig_cfg.get("swing_strength", 2),
    ).detect_swings(df)
    df = MarketStructure(
        strength=sig_cfg.get("swing_strength", 2),
    ).detect_bos(df)
    df = SignalEngine(
        retest_tolerance=sig_cfg.get("retest_tolerance", 0.001),
        adx_threshold=sig_cfg.get("adx_threshold", 20),
    ).generate_signal(df)
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_args(argv=None):
    """Parse CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="STRATEGY V3 BACKTEST — Risk-Integrated Version",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to strategy config JSON (default: strategy_config.json in project root)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV data file (overrides config data.csv_path)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for report output (default: reports/)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    from src.data_validation import validate_strict, get_quality_summary

    # --- Parse CLI ---
    args = parse_args()

    # --- Load configuration ---
    config_path = args.config if args.config else ROOT / "strategy_config.json"
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # --- Setup structured logging ---
    logger = setup_logger("backtest", config)

    logger.info("=" * 50)
    logger.info("STRATEGY V3 BACKTEST — Starting")
    logger.info("=" * 50)

    # --- Determine paths ---
    csv_path = args.data if args.data else config["data"]["csv_path"]
    output_dir = args.output_dir if args.output_dir else ROOT / "reports"

    # --- Load raw CSV for validation (no datetime parsing yet) ---
    logger.info("Loading data from %s", csv_path)
    raw_df = pd.read_csv(str(Path(csv_path) if Path(csv_path).is_absolute() else ROOT / csv_path))

    # --- Strict data validation ---
    logger.info("Validating data quality...")
    try:
        quality_report = validate_strict(raw_df, str(csv_path))
    except ValueError as exc:
        logger.error("Data validation FAILED: %s", exc)
        sys.exit(1)

    quality_summary = get_quality_summary(quality_report)
    logger.info(
        "Data quality: %d rows, %d warnings, %d errors",
        quality_summary["row_count"],
        quality_summary["warning_count"],
        quality_summary["error_count"],
    )
    for finding in quality_summary["findings"]:
        logger.warning("  [%s] %s: %s", finding["severity"], finding["code"], finding["message"])

    # --- Prepare dataframe (indicators + signals) ---
    df = prepare_dataframe(csv_path, config=config)
    logger.info("Total candles: %d", len(df))

    # --- Run backtest ---
    settings = get_backtest_settings(config)
    logger.info(
        "Settings: balance=%.2f, risk=%.1f%%, RR=%.1f, spread=%.2f, slippage=%.2f",
        settings["starting_balance"],
        settings["risk_percent"],
        settings["risk_reward"],
        settings["spread_points"],
        settings["slippage_points"],
    )

    results = run_backtest(df, settings)

    # --- Attach data quality to results for export ---
    results["data_quality"] = quality_summary

    # --- Log results ---
    logger.info("-" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("-" * 50)
    logger.info("Starting Balance:  %.2f", results["starting_balance"])
    logger.info("Final Balance:     %.2f", results["final_balance"])
    logger.info("Net Result:        %.2f", results["net_result"])
    logger.info("Total Trades:      %d", results["total_trades"])
    logger.info("Wins:              %d", results["wins"])
    logger.info("Losses:            %d", results["losses"])
    logger.info("Win Rate:          %.2f%%", results["win_rate"])
    logger.info("Profit Factor:     %.2f", results["profit_factor"])
    logger.info("Expectancy:        %.2f", results["expectancy"])
    logger.info("Max Drawdown:      %.2f (%.2f%%)", results["max_drawdown"], results["max_drawdown_pct"])
    logger.info("Same-bar SL/TP Rule: CONSERVATIVE (assume loss)")
    logger.info("-" * 50)

    # --- Export daily report (CSV + JSON) ---
    report_files = export_backtest_report(results, output_dir=output_dir)
    logger.info("Reports exported:")
    logger.info("  CSV:  %s", report_files["csv"])
    logger.info("  JSON: %s", report_files["json"])

    logger.info("=" * 50)
    logger.info("STRATEGY V3 BACKTEST — Complete")
    logger.info("=" * 50)
