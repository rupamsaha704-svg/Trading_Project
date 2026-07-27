"""
Validation Runner — Multi-mode strategy validation without optimization.

Modes:
1. Full-dataset backtest (baseline)
2. Chronological train/test split (out-of-sample)
3. Walk-forward validation (rolling windows)
4. Parameter sensitivity testing (grid)
5. Spread/slippage stress testing

Reports stability across all runs rather than optimizing for best result.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import run_backtest, prepare_dataframe
from src.config_loader import load_config, get_backtest_settings


# =============================================================================
# DATA SPLITTING
# =============================================================================


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically into train and test sets.

    Args:
        df: Full prepared DataFrame (with indicators/signals).
        train_ratio: Fraction of data for training (0.0–1.0).

    Returns:
        (train_df, test_df) tuple.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 exclusive")

    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def walk_forward_windows(
    df: pd.DataFrame,
    n_windows: int = 3,
    train_ratio: float = 0.7,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward windows (expanding or rolling train, fixed-size test).

    Each window has a training set and an out-of-sample test set.
    Windows advance chronologically with no overlap in test sets.

    Args:
        df: Full prepared DataFrame.
        n_windows: Number of walk-forward windows.
        train_ratio: Fraction of each window used for training.

    Returns:
        List of (train_df, test_df) tuples.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 exclusive")

    total_len = len(df)
    window_size = total_len // n_windows
    if window_size < 10:
        raise ValueError("Not enough data for the requested number of windows")

    windows = []
    for i in range(n_windows):
        end_idx = (i + 1) * window_size
        if i == n_windows - 1:
            end_idx = total_len  # last window takes remainder

        window_df = df.iloc[: end_idx].copy()
        split_idx = int(len(window_df) * train_ratio)
        train = window_df.iloc[:split_idx].copy()
        test = window_df.iloc[split_idx:].copy()
        windows.append((train, test))

    return windows


# =============================================================================
# SENSITIVITY GRID
# =============================================================================


def sensitivity_grid(
    df: pd.DataFrame,
    base_settings: Dict[str, Any],
    param_name: str,
    values: List[Any],
) -> List[Dict[str, Any]]:
    """Run backtest across a range of values for a single parameter.

    Args:
        df: Prepared DataFrame.
        base_settings: Base backtest settings dict.
        param_name: Key in settings to vary.
        values: List of values to test.

    Returns:
        List of result dicts, each with the varied parameter value included.
    """
    results = []
    for val in values:
        settings = {**base_settings, param_name: val}
        result = run_backtest(df, settings)
        result["varied_param"] = param_name
        result["varied_value"] = val
        results.append(result)
    return results


# =============================================================================
# STRESS TESTING
# =============================================================================


def stress_test_execution(
    df: pd.DataFrame,
    base_settings: Dict[str, Any],
    spread_values: Optional[List[float]] = None,
    slippage_values: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Run backtest under varying spread/slippage conditions.

    Args:
        df: Prepared DataFrame.
        base_settings: Base backtest settings.
        spread_values: List of spread values to test.
        slippage_values: List of slippage values to test.

    Returns:
        List of result dicts with spread/slippage noted.
    """
    if spread_values is None:
        spread_values = [0.0, 0.15, 0.30, 0.50, 1.00]
    if slippage_values is None:
        slippage_values = [0.0, 0.05, 0.10, 0.20, 0.50]

    results = []
    for spread in spread_values:
        for slippage in slippage_values:
            settings = {
                **base_settings,
                "spread_points": spread,
                "slippage_points": slippage,
            }
            result = run_backtest(df, settings)
            result["stress_spread"] = spread
            result["stress_slippage"] = slippage
            results.append(result)
    return results


# =============================================================================
# STABILITY METRICS
# =============================================================================


def compute_stability(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute stability metrics across multiple runs.

    Reports mean, std, min, max for key metrics. A stable strategy shows
    low coefficient of variation (std/mean) across conditions.
    """
    if not results:
        return {}

    metrics_keys = ["net_result", "win_rate", "profit_factor", "expectancy", "max_drawdown_pct"]
    stability = {}

    for key in metrics_keys:
        values = [r[key] for r in results if key in r and r[key] != float("inf")]
        if not values:
            stability[key] = {"mean": 0, "std": 0, "min": 0, "max": 0, "cv": 0}
            continue

        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = abs(std / mean) if mean != 0 else float("inf")
        stability[key] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "cv": round(cv, 4),
        }

    # Count profitable runs
    profitable = sum(1 for r in results if r.get("net_result", 0) > 0)
    stability["profitable_runs"] = profitable
    stability["total_runs"] = len(results)
    stability["profitability_rate"] = round(profitable / len(results) * 100, 2)

    return stability


# =============================================================================
# FULL VALIDATION RUNNER
# =============================================================================


def run_full_validation(
    csv_path: str = "XAUUSD_M5.csv",
    config_path: Optional[str] = None,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Run all validation modes and produce a combined report.

    Args:
        csv_path: Path to CSV data file.
        config_path: Path to config JSON. If None, loads ROOT/strategy_config.json.
        output_dir: Directory for report output.

    Returns:
        Dict with results from all modes + stability summary.

    Raises:
        ValueError: If CSV data fails strict validation (ERROR-level findings).
    """
    # --- Load config (default to ROOT / strategy_config.json) ---
    resolved_config_path = config_path if config_path is not None else str(ROOT / "strategy_config.json")
    config = load_config(resolved_config_path)
    base_settings = get_backtest_settings(config)

    # --- Strictly validate raw CSV before processing ---
    from src.data_validation import validate_strict, get_quality_summary

    csv_resolved = Path(csv_path) if Path(csv_path).is_absolute() else ROOT / csv_path
    raw_df = pd.read_csv(str(csv_resolved))
    quality_report = validate_strict(raw_df, str(csv_path))
    data_quality_summary = get_quality_summary(quality_report)

    # --- Prepare full dataset ---
    df = prepare_dataframe(csv_path, config=config)

    report: Dict[str, Any] = {
        "validation_date": date.today().isoformat(),
        "data_rows": len(df),
        "data_quality": data_quality_summary,
        "config_used": config,
    }

    # =========================================================================
    # 1. Full-dataset backtest (baseline)
    # =========================================================================
    full_result = run_backtest(df, base_settings)
    report["full_dataset"] = _extract_metrics(full_result)

    # =========================================================================
    # 2. Chronological train/test split
    # =========================================================================
    train_df, test_df = chronological_split(df, train_ratio=0.7)
    train_result = run_backtest(train_df, base_settings)
    test_result = run_backtest(test_df, base_settings)
    report["train_test_split"] = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train": _extract_metrics(train_result),
        "test": _extract_metrics(test_result),
    }

    # =========================================================================
    # 3. Walk-forward validation
    # =========================================================================
    n_windows = min(3, max(1, len(df) // 200))
    windows = walk_forward_windows(df, n_windows=n_windows, train_ratio=0.7)
    wf_results = []
    for i, (wf_train, wf_test) in enumerate(windows):
        wf_test_result = run_backtest(wf_test, base_settings)
        wf_results.append({
            "window": i + 1,
            "train_rows": len(wf_train),
            "test_rows": len(wf_test),
            **_extract_metrics(wf_test_result),
        })

    wf_stability = compute_stability(
        [run_backtest(wf_test, base_settings) for _, wf_test in windows]
    )
    report["walk_forward"] = {
        "n_windows": n_windows,
        "windows": wf_results,
        "stability": wf_stability,
    }

    # =========================================================================
    # 4. Parameter sensitivity
    # =========================================================================
    sensitivity_results = {}

    # Risk percent: 0.5, 1.0, 1.5, 2.0
    rp_results = sensitivity_grid(df, base_settings, "risk_percent", [0.5, 1.0, 1.5, 2.0])
    sensitivity_results["risk_percent"] = {
        "values_tested": [0.5, 1.0, 1.5, 2.0],
        "results": [_extract_metrics(r) for r in rp_results],
        "stability": compute_stability(rp_results),
    }

    # Risk reward: 1.5, 2.0, 2.5, 3.0
    rr_results = sensitivity_grid(df, base_settings, "risk_reward", [1.5, 2.0, 2.5, 3.0])
    sensitivity_results["risk_reward"] = {
        "values_tested": [1.5, 2.0, 2.5, 3.0],
        "results": [_extract_metrics(r) for r in rr_results],
        "stability": compute_stability(rr_results),
    }

    # ATR SL multiplier: 1.5, 2.0, 2.5, 3.0
    atr_results = sensitivity_grid(df, base_settings, "atr_stop_loss_multiplier", [1.5, 2.0, 2.5, 3.0])
    sensitivity_results["atr_stop_loss_multiplier"] = {
        "values_tested": [1.5, 2.0, 2.5, 3.0],
        "results": [_extract_metrics(r) for r in atr_results],
        "stability": compute_stability(atr_results),
    }

    report["sensitivity"] = sensitivity_results

    # =========================================================================
    # 5. Spread/slippage stress test
    # =========================================================================
    stress_results = stress_test_execution(df, base_settings)
    report["stress_test"] = {
        "total_scenarios": len(stress_results),
        "results": [
            {
                "spread": r["stress_spread"],
                "slippage": r["stress_slippage"],
                **_extract_metrics(r),
            }
            for r in stress_results
        ],
        "stability": compute_stability(stress_results),
    }

    # =========================================================================
    # OVERALL STABILITY
    # =========================================================================
    all_results = (
        [full_result, train_result, test_result]
        + [run_backtest(wf_test, base_settings) for _, wf_test in windows]
        + rp_results + rr_results + atr_results + stress_results
    )
    report["overall_stability"] = compute_stability(all_results)

    return report


def _extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a backtest result (no equity curve/trade log)."""
    return {
        "net_result": result["net_result"],
        "total_trades": result["total_trades"],
        "wins": result["wins"],
        "losses": result["losses"],
        "win_rate": result["win_rate"],
        "profit_factor": result["profit_factor"],
        "expectancy": result["expectancy"],
        "max_drawdown": result["max_drawdown"],
        "max_drawdown_pct": result["max_drawdown_pct"],
    }


# =============================================================================
# EXPORT
# =============================================================================


def export_validation_report(
    report: Dict[str, Any],
    output_dir: str | Path = "reports",
    report_date: Optional[date] = None,
) -> Dict[str, Path]:
    """Export validation report as JSON summary + CSV table.

    Returns:
        Dict with 'json' and 'csv' paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    day = report_date or date.today()
    date_str = day.isoformat()

    # --- JSON: full report ---
    json_file = output_path / f"validation_{date_str}.json"
    # Remove config (too verbose) from export
    export_report = {k: v for k, v in report.items() if k != "config_used"}
    json_file.write_text(
        json.dumps(export_report, indent=2, default=str),
        encoding="utf-8",
    )

    # --- CSV: summary table of all runs ---
    csv_file = output_path / f"validation_{date_str}.csv"
    rows = _flatten_to_rows(report)
    if rows:
        pd.DataFrame(rows).to_csv(csv_file, index=False)
    else:
        pd.DataFrame().to_csv(csv_file, index=False)

    return {"json": json_file, "csv": csv_file}


def _flatten_to_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten validation report into a list of rows for CSV export."""
    rows = []

    # Full dataset
    if "full_dataset" in report:
        rows.append({"mode": "full_dataset", "window": "-", **report["full_dataset"]})

    # Train/test
    if "train_test_split" in report:
        tts = report["train_test_split"]
        rows.append({"mode": "train", "window": "-", **tts["train"]})
        rows.append({"mode": "test", "window": "-", **tts["test"]})

    # Walk-forward
    if "walk_forward" in report:
        for w in report["walk_forward"]["windows"]:
            row = {k: v for k, v in w.items() if k != "train_rows" and k != "test_rows"}
            row["mode"] = "walk_forward"
            rows.append(row)

    # Sensitivity
    if "sensitivity" in report:
        for param, data in report["sensitivity"].items():
            for i, r in enumerate(data["results"]):
                rows.append({
                    "mode": f"sensitivity_{param}",
                    "window": data["values_tested"][i],
                    **r,
                })

    # Stress test
    if "stress_test" in report:
        for r in report["stress_test"]["results"]:
            rows.append({"mode": "stress_test", "window": f"s={r['spread']}/sl={r['slippage']}", **{
                k: v for k, v in r.items() if k not in ("spread", "slippage")
            }})

    return rows
