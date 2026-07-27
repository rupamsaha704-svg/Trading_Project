"""
Multi-period backtest comparison runner.

Runs full validation (backtest, train/test, walk-forward, sensitivity, stress)
on 12-month, 8-month, and 6-month slices of XAUUSD M5 data.
Exports one combined JSON and one combined CSV.
"""

import sys
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_runner import run_full_validation, export_validation_report, compute_stability
from src.data_validation import validate_strict, get_quality_summary
from src.data_prep import ensure_data_ready


def slice_by_months(csv_path: str, months: int) -> str:
    """Slice the last N months from a CSV and save to a temp file. Returns path."""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    end = df["time"].max()
    start = end - pd.DateOffset(months=months)
    sliced = df[df["time"] >= start].copy()
    out_path = str(ROOT / f"_temp_slice_{months}m.csv")
    sliced.to_csv(out_path, index=False)
    return out_path


def run_multi_period(
    csv_path: str = "XAUUSD_M5_12M.csv",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Run validation on available-360-day, 8m, 6m slices and produce comparison report."""

    # Auto-prepare data from ZIP if CSV is missing
    full_path = Path(csv_path) if Path(csv_path).is_absolute() else ROOT / csv_path
    if not full_path.exists():
        ensure_data_ready(project_root=ROOT)
        full_path = ROOT / "XAUUSD_M5_12M.csv"

    full_path_str = str(full_path)
    full_df = pd.read_csv(full_path_str)
    quality_report = validate_strict(full_df, csv_path)
    quality_summary = get_quality_summary(quality_report)

    # Determine actual date span
    full_df["time"] = pd.to_datetime(full_df["time"])
    span_days = (full_df["time"].max() - full_df["time"].min()).days

    periods = {"available": None, "8m": 8, "6m": 6}
    period_results = {}

    for label, months in periods.items():
        if months is None:
            # Full available period (not claiming 12 calendar months)
            slice_path = full_path_str
            slice_df = full_df
        else:
            slice_path = slice_by_months(full_path_str, months)
            slice_df = pd.read_csv(slice_path)
            slice_df["time"] = pd.to_datetime(slice_df["time"])

        report = run_full_validation(slice_path, config_path=str(ROOT / "strategy_config.json"))

        period_key = f"{span_days}d" if label == "available" else label
        period_results[period_key] = {
            "label": f"Available {span_days}-day period" if label == "available" else f"Last {months} months",
            "candles": len(slice_df),
            "start": str(slice_df["time"].min()),
            "end": str(slice_df["time"].max()),
            "full_dataset": report["full_dataset"],
            "train_test_split": report["train_test_split"],
            "walk_forward": report["walk_forward"],
            "sensitivity": report["sensitivity"],
            "stress_test": report["stress_test"],
            "overall_stability": report["overall_stability"],
        }

        # Cleanup temp file
        if months is not None:
            Path(slice_path).unlink(missing_ok=True)

    # Build comparison
    comparison = {
        "report_date": date.today().isoformat(),
        "source_file": csv_path,
        "data_quality": quality_summary,
        "periods": period_results,
        "comparison_table": _build_comparison_table(period_results),
    }

    return comparison


def _build_comparison_table(period_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a flat table comparing key metrics across periods."""
    rows = []
    for key, data in period_results.items():
        fd = data["full_dataset"]
        tts = data["train_test_split"]
        os_stability = data["overall_stability"]
        rows.append({
            "period": key,
            "label": data.get("label", key),
            "candles": data["candles"],
            "start": data["start"],
            "end": data["end"],
            "net_result": fd["net_result"],
            "total_trades": fd["total_trades"],
            "wins": fd["wins"],
            "losses": fd["losses"],
            "win_rate": fd["win_rate"],
            "profit_factor": fd["profit_factor"],
            "expectancy": fd["expectancy"],
            "max_drawdown": fd["max_drawdown"],
            "max_drawdown_pct": fd["max_drawdown_pct"],
            "oos_net_result": tts["test"]["net_result"],
            "oos_win_rate": tts["test"]["win_rate"],
            "oos_profit_factor": tts["test"]["profit_factor"],
            "stability_profitable_pct": os_stability.get("profitability_rate", 0),
            "stability_net_cv": os_stability.get("net_result", {}).get("cv", 0),
        })
    return rows


def export_multi_period_report(
    report: Dict[str, Any],
    output_dir: str = "reports",
) -> Dict[str, Path]:
    """Export comparison report as JSON + CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = date.today().isoformat()

    # JSON
    json_file = output_path / f"multi_period_{date_str}.json"
    json_file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # CSV
    csv_file = output_path / f"multi_period_{date_str}.csv"
    if report.get("comparison_table"):
        pd.DataFrame(report["comparison_table"]).to_csv(csv_file, index=False)
    else:
        pd.DataFrame().to_csv(csv_file, index=False)

    return {"json": json_file, "csv": csv_file}


if __name__ == "__main__":
    print("=" * 60)
    print(" MULTI-PERIOD BACKTEST COMPARISON")
    print("=" * 60)

    report = run_multi_period("XAUUSD_M5_12M.csv")
    files = export_multi_period_report(report)

    print(f"\nData quality: {report['data_quality']['row_count']} rows, "
          f"{report['data_quality']['warning_count']}W, {report['data_quality']['error_count']}E")

    print("\n" + "-" * 60)
    print(f"{'Period':<12} {'Candles':<8} {'Trades':<7} {'Net':<10} {'WR%':<6} {'PF':<6} {'Exp':<8} {'MDD%':<8} {'Stab%':<6}")
    print("-" * 60)
    for row in report["comparison_table"]:
        print(f"{row['period']:<12} {row['candles']:<8} {row['total_trades']:<7} "
              f"{row['net_result']:<10.2f} {row['win_rate']:<6.1f} {row['profit_factor']:<6.2f} "
              f"{row['expectancy']:<8.2f} {row['max_drawdown_pct']:<8.2f} {row['stability_profitable_pct']:<6.1f}")
    print("-" * 60)

    print(f"\nReports: {files['json'].name}, {files['csv'].name}")
    print("=" * 60)
