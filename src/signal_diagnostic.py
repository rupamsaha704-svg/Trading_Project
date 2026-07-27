"""
Signal Funnel Diagnostic — counts candidates at each filtering stage.

Runs the same logic as SignalEngine but instruments every filter step.
Does NOT alter signals, indicators, or entry/exit behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FunnelDiagnostic:
    """Counts at each stage of the signal funnel."""

    total_candles: int = 0
    bos_buy_candidates: int = 0
    bos_sell_candidates: int = 0
    ema_filter_passed_buy: int = 0
    ema_filter_passed_sell: int = 0
    adx_filter_passed_buy: int = 0
    adx_filter_passed_sell: int = 0
    retest_confirmed_buy: int = 0
    retest_confirmed_sell: int = 0
    final_buy_signals: int = 0
    final_sell_signals: int = 0
    trades_blocked_by_risk: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def run_signal_diagnostic(
    df: pd.DataFrame,
    retest_tolerance: float = 0.001,
    adx_threshold: float = 20,
) -> FunnelDiagnostic:
    """Analyze signal funnel without modifying the DataFrame.

    Counts how many candles pass each filter stage. Uses the same logic
    as SignalEngine.generate_signal() but only counts — never writes.

    Args:
        df: DataFrame with indicators (EMA200, ADX14) and BOS columns.
        retest_tolerance: Tolerance for BOS level retest.
        adx_threshold: Minimum ADX value for signal.

    Returns:
        FunnelDiagnostic with all stage counts.
    """
    diag = FunnelDiagnostic()
    diag.total_candles = len(df)

    bullish_bos_level = None
    bearish_bos_level = None
    bullish_bos_active = False
    bearish_bos_active = False

    for i in range(1, len(df)):
        current = df.iloc[i]

        # --- Buy funnel ---
        if bullish_bos_active and bullish_bos_level is not None:
            # Retest check
            retest_buy = (
                current["low"] <= bullish_bos_level + retest_tolerance
                and current["close"] > bullish_bos_level
                and current["close"] > current["open"]
            )
            if retest_buy:
                diag.retest_confirmed_buy += 1

                # EMA filter
                ema_pass = current["close"] > current["EMA200"]
                if ema_pass:
                    diag.ema_filter_passed_buy += 1

                    # ADX filter
                    adx_pass = current["ADX14"] >= adx_threshold
                    if adx_pass:
                        diag.adx_filter_passed_buy += 1
                        diag.final_buy_signals += 1
                        bullish_bos_active = False
                        bullish_bos_level = None

        # --- Sell funnel ---
        if bearish_bos_active and bearish_bos_level is not None:
            retest_sell = (
                current["high"] >= bearish_bos_level - retest_tolerance
                and current["close"] < bearish_bos_level
                and current["close"] < current["open"]
            )
            if retest_sell:
                diag.retest_confirmed_sell += 1

                ema_pass = current["close"] < current["EMA200"]
                if ema_pass:
                    diag.ema_filter_passed_sell += 1

                    adx_pass = current["ADX14"] >= adx_threshold
                    if adx_pass:
                        diag.adx_filter_passed_sell += 1
                        diag.final_sell_signals += 1
                        bearish_bos_active = False
                        bearish_bos_level = None

        # --- Track BOS candidates ---
        if bool(current.get("BullishBOS", False)):
            diag.bos_buy_candidates += 1
            bullish_bos_level = current.get("BullishBOSLevel", None)
            bullish_bos_active = True

        if bool(current.get("BearishBOS", False)):
            diag.bos_sell_candidates += 1
            bearish_bos_level = current.get("BearishBOSLevel", None)
            bearish_bos_active = True

    return diag


def add_risk_blocked_count(diag: FunnelDiagnostic, trade_log: List[Dict[str, Any]], total_signals: int) -> FunnelDiagnostic:
    """Add trades_blocked_by_risk by comparing signals to executed trades.

    Args:
        diag: Existing diagnostic.
        trade_log: Trade log from backtest results.
        total_signals: Total signals generated (buy + sell).

    Returns:
        Updated diagnostic with blocked count.
    """
    executed = len(trade_log)
    diag.trades_blocked_by_risk = max(0, total_signals - executed)
    return diag


# =============================================================================
# EXPORT
# =============================================================================


def export_diagnostic(
    diag: FunnelDiagnostic,
    output_dir: str | Path = "reports",
    report_date: Optional[date] = None,
) -> Dict[str, Path]:
    """Export diagnostic to JSON and CSV.

    Returns:
        Dict with 'json' and 'csv' paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    day = report_date or date.today()
    date_str = day.isoformat()

    data = diag.to_dict()

    # JSON
    json_file = output_path / f"signal_diagnostic_{date_str}.json"
    json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # CSV (single row)
    csv_file = output_path / f"signal_diagnostic_{date_str}.csv"
    pd.DataFrame([data]).to_csv(csv_file, index=False)

    return {"json": json_file, "csv": csv_file}
