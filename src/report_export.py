"""Export backtest results to CSV and JSON report files.

Usage:
    from src.report_export import export_backtest_report
    export_backtest_report(results, output_dir="reports")

Generates:
    reports/backtest_YYYY-MM-DD.csv   — trade log
    reports/backtest_YYYY-MM-DD.json  — full metrics + settings
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def export_backtest_report(
    results: Dict[str, Any],
    output_dir: str | Path = "reports",
    report_date: Optional[date] = None,
) -> Dict[str, Path]:
    """Export backtest results to daily CSV (trade log) and JSON (metrics).

    Args:
        results: Dict returned by run_backtest().
        output_dir: Directory for output files (created if missing).
        report_date: Override date stamp (defaults to today).

    Returns:
        Dict with keys 'csv' and 'json' pointing to the created file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    day = report_date or date.today()
    date_str = day.isoformat()

    # --- CSV: trade log ---
    csv_file = output_path / f"backtest_{date_str}.csv"
    trade_log = results.get("trade_log", [])
    if trade_log:
        df = pd.DataFrame(trade_log)
        df.to_csv(csv_file, index=False)
    else:
        # Empty trade log — write header-only CSV
        pd.DataFrame(columns=[
            "bar_index", "side", "entry", "sl", "tp",
            "position_size", "result", "pnl", "exit_bar", "note",
        ]).to_csv(csv_file, index=False)

    # --- JSON: metrics + settings ---
    json_file = output_path / f"backtest_{date_str}.json"
    metrics = {
        "report_date": date_str,
        "starting_balance": results["starting_balance"],
        "final_balance": results["final_balance"],
        "net_result": results["net_result"],
        "total_trades": results["total_trades"],
        "wins": results["wins"],
        "losses": results["losses"],
        "win_rate": results["win_rate"],
        "profit_factor": results["profit_factor"],
        "expectancy": results["expectancy"],
        "gross_profit": results["gross_profit"],
        "gross_loss": results["gross_loss"],
        "max_drawdown": results["max_drawdown"],
        "max_drawdown_pct": results["max_drawdown_pct"],
        "settings": results["settings"],
        "trade_count_by_side": _count_sides(trade_log),
    }
    json_file.write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )

    return {"csv": csv_file, "json": json_file}


def _count_sides(trade_log: list) -> Dict[str, int]:
    """Count trades by side."""
    counts: Dict[str, int] = {"long": 0, "short": 0}
    for trade in trade_log:
        side = trade.get("side", "unknown")
        counts[side] = counts.get(side, 0) + 1
    return counts
