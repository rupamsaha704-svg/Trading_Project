"""Tests for the validation runner: splits, walk-forward, sensitivity, stress."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_runner import (
    chronological_split,
    walk_forward_windows,
    sensitivity_grid,
    stress_test_execution,
    compute_stability,
    export_validation_report,
    run_full_validation,
    _extract_metrics,
)
from backtest import run_backtest


def _make_signal_df(n=100):
    """Create a minimal DataFrame with periodic buy signals."""
    df = pd.DataFrame({
        "open": [100.0] * n,
        "high": [102.0] * n,
        "low": [98.0] * n,
        "close": [100.0] * n,
        "ATR14": [2.0] * n,
        "EMA200": [99.0] * n,
        "ADX14": [30.0] * n,
        "BuySignal": [False] * n,
        "SellSignal": [False] * n,
        "BullishBOS": [False] * n,
        "BearishBOS": [False] * n,
        "time": pd.date_range("2024-01-01", periods=n, freq="5min"),
    })
    # Place buy signals every 20 bars
    for i in range(5, n - 5, 20):
        df.loc[i, "BuySignal"] = True
    # Make some SL hits
    for i in range(8, n, 20):
        df.loc[i, "low"] = 94.0
    return df


# =============================================================================
# SPLIT LOGIC TESTS
# =============================================================================


class TestChronologicalSplit:

    def test_split_sizes(self):
        df = pd.DataFrame({"a": range(100)})
        train, test = chronological_split(df, train_ratio=0.7)
        assert len(train) == 70
        assert len(test) == 30

    def test_split_preserves_order(self):
        df = pd.DataFrame({"a": range(100)})
        train, test = chronological_split(df, train_ratio=0.6)
        assert train.iloc[-1]["a"] < test.iloc[0]["a"]

    def test_invalid_ratio_raises(self):
        df = pd.DataFrame({"a": range(100)})
        with pytest.raises(ValueError):
            chronological_split(df, train_ratio=0.0)
        with pytest.raises(ValueError):
            chronological_split(df, train_ratio=1.0)

    def test_split_is_copy(self):
        """Modifying split should not affect original."""
        df = pd.DataFrame({"a": range(100)})
        train, test = chronological_split(df, train_ratio=0.5)
        train.iloc[0, 0] = -999
        assert df.iloc[0]["a"] == 0


# =============================================================================
# WALK-FORWARD WINDOW TESTS
# =============================================================================


class TestWalkForwardWindows:

    def test_correct_number_of_windows(self):
        df = pd.DataFrame({"a": range(300)})
        windows = walk_forward_windows(df, n_windows=3, train_ratio=0.7)
        assert len(windows) == 3

    def test_windows_expand_chronologically(self):
        """Each window's training end should be further along."""
        df = pd.DataFrame({"a": range(300)})
        windows = walk_forward_windows(df, n_windows=3, train_ratio=0.7)
        train_ends = [len(train) for train, _ in windows]
        assert train_ends == sorted(train_ends)

    def test_test_set_after_train(self):
        """In each window, test set indices follow train set."""
        df = pd.DataFrame({"a": range(300), "idx": range(300)})
        windows = walk_forward_windows(df, n_windows=3, train_ratio=0.7)
        for train, test in windows:
            if len(test) > 0 and len(train) > 0:
                assert train.iloc[-1]["idx"] < test.iloc[0]["idx"]

    def test_insufficient_data_raises(self):
        df = pd.DataFrame({"a": range(5)})
        with pytest.raises(ValueError):
            walk_forward_windows(df, n_windows=3, train_ratio=0.7)

    def test_single_window(self):
        df = pd.DataFrame({"a": range(100)})
        windows = walk_forward_windows(df, n_windows=1, train_ratio=0.7)
        assert len(windows) == 1
        train, test = windows[0]
        assert len(train) + len(test) == 100


# =============================================================================
# SENSITIVITY GRID TESTS
# =============================================================================


class TestSensitivityGrid:

    def test_returns_correct_count(self):
        df = _make_signal_df(100)
        base = {"starting_balance": 10000.0, "spread_points": 0.0, "slippage_points": 0.0}
        results = sensitivity_grid(df, base, "risk_percent", [0.5, 1.0, 2.0])
        assert len(results) == 3

    def test_varied_param_recorded(self):
        df = _make_signal_df(100)
        base = {"starting_balance": 10000.0, "spread_points": 0.0, "slippage_points": 0.0}
        results = sensitivity_grid(df, base, "risk_percent", [0.5, 1.0])
        assert results[0]["varied_value"] == 0.5
        assert results[1]["varied_value"] == 1.0

    def test_different_params_produce_different_pnl(self):
        """Different risk_percent should lead to different net_result."""
        df = _make_signal_df(100)
        base = {"starting_balance": 10000.0, "spread_points": 0.0, "slippage_points": 0.0}
        results = sensitivity_grid(df, base, "risk_percent", [0.5, 2.0])
        # PnL should scale with risk percent
        if results[0]["total_trades"] > 0 and results[1]["total_trades"] > 0:
            assert results[0]["net_result"] != results[1]["net_result"]


# =============================================================================
# STRESS TEST TESTS
# =============================================================================


class TestStressTest:

    def test_stress_test_scenario_count(self):
        df = _make_signal_df(100)
        base = {"starting_balance": 10000.0}
        results = stress_test_execution(
            df, base,
            spread_values=[0.0, 0.5],
            slippage_values=[0.0, 0.2],
        )
        # 2 spread × 2 slippage = 4 scenarios
        assert len(results) == 4

    def test_stress_results_have_cost_labels(self):
        df = _make_signal_df(100)
        base = {"starting_balance": 10000.0}
        results = stress_test_execution(
            df, base,
            spread_values=[0.3],
            slippage_values=[0.1],
        )
        assert results[0]["stress_spread"] == 0.3
        assert results[0]["stress_slippage"] == 0.1

    def test_higher_costs_reduce_profit(self):
        """Higher spread/slippage should reduce or maintain net_result."""
        df = _make_signal_df(100)
        base = {"starting_balance": 10000.0}
        results = stress_test_execution(
            df, base,
            spread_values=[0.0, 1.0],
            slippage_values=[0.0],
        )
        low_cost = next(r for r in results if r["stress_spread"] == 0.0)
        high_cost = next(r for r in results if r["stress_spread"] == 1.0)
        assert high_cost["net_result"] <= low_cost["net_result"]


# =============================================================================
# STABILITY METRICS TESTS
# =============================================================================


class TestComputeStability:

    def test_empty_results(self):
        assert compute_stability([]) == {}

    def test_stability_keys_present(self):
        results = [
            {"net_result": 100, "win_rate": 60, "profit_factor": 1.5, "expectancy": 20, "max_drawdown_pct": -2},
            {"net_result": 50, "win_rate": 50, "profit_factor": 1.2, "expectancy": 10, "max_drawdown_pct": -3},
        ]
        stability = compute_stability(results)
        assert "net_result" in stability
        assert "profitable_runs" in stability
        assert "total_runs" in stability
        assert stability["total_runs"] == 2
        assert stability["profitable_runs"] == 2

    def test_cv_is_zero_for_constant_values(self):
        results = [{"net_result": 100, "win_rate": 50, "profit_factor": 2.0, "expectancy": 50, "max_drawdown_pct": -1}] * 3
        stability = compute_stability(results)
        assert stability["net_result"]["cv"] == 0.0


# =============================================================================
# EXPORT TESTS
# =============================================================================


class TestExportValidationReport:

    def test_export_creates_files(self, tmp_path):
        from datetime import date as d
        report = {
            "validation_date": "2024-01-01",
            "data_rows": 100,
            "full_dataset": {"net_result": 100, "total_trades": 5, "wins": 3, "losses": 2,
                             "win_rate": 60, "profit_factor": 2.0, "expectancy": 20,
                             "max_drawdown": -50, "max_drawdown_pct": -0.5},
            "train_test_split": {
                "train_rows": 70, "test_rows": 30,
                "train": {"net_result": 80, "total_trades": 4, "wins": 3, "losses": 1,
                          "win_rate": 75, "profit_factor": 3.0, "expectancy": 20,
                          "max_drawdown": -30, "max_drawdown_pct": -0.3},
                "test": {"net_result": 20, "total_trades": 1, "wins": 1, "losses": 0,
                         "win_rate": 100, "profit_factor": float("inf"), "expectancy": 20,
                         "max_drawdown": 0, "max_drawdown_pct": 0},
            },
            "walk_forward": {"n_windows": 1, "windows": [], "stability": {}},
            "sensitivity": {},
            "stress_test": {"total_scenarios": 0, "results": [], "stability": {}},
            "overall_stability": {},
        }
        files = export_validation_report(report, output_dir=tmp_path, report_date=d(2024, 1, 1))
        assert files["json"].exists()
        assert files["csv"].exists()
        assert "validation_2024-01-01" in files["json"].name



# =============================================================================
# PR #6 FIXES: default config, strict validation, quality in report
# =============================================================================


class TestDefaultConfigLoaded:

    def test_run_full_validation_uses_default_config(self):
        """When config_path is None, strategy_config.json from ROOT is loaded."""
        report = run_full_validation("XAUUSD_M5.csv", config_path=None)
        # Should succeed and produce full_dataset results
        assert "full_dataset" in report
        assert report["full_dataset"]["total_trades"] > 0
        # Config should match strategy_config.json defaults
        assert report["config_used"]["risk"]["risk_percent"] == 1.0


class TestInvalidCSVRejected:

    def test_invalid_csv_raises_value_error(self, tmp_path):
        """run_full_validation raises ValueError on CSV with invalid data."""
        # CSV missing required columns
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n")

        with pytest.raises(ValueError, match="Data validation failed"):
            run_full_validation(str(bad_csv), config_path=None)

    def test_csv_with_negative_prices_raises(self, tmp_path):
        """CSV with negative prices should fail strict validation."""
        bad_csv = tmp_path / "neg.csv"
        bad_csv.write_text(
            "time,open,high,low,close,tick_volume\n"
            "2024-01-01 00:00,-10,11,9,10,100\n"
            "2024-01-01 00:05,11,12,10,11,200\n"
        )
        with pytest.raises(ValueError, match="Data validation failed"):
            run_full_validation(str(bad_csv), config_path=None)


class TestQualitySummaryInReport:

    def test_data_quality_present_in_validation_report(self):
        """run_full_validation should include data_quality in the report dict."""
        report = run_full_validation("XAUUSD_M5.csv", config_path=None)
        assert "data_quality" in report
        assert "row_count" in report["data_quality"]
        assert "warning_count" in report["data_quality"]
        assert "error_count" in report["data_quality"]
        assert report["data_quality"]["error_count"] == 0

    def test_data_quality_exported_in_json(self, tmp_path):
        """Exported JSON should contain data_quality key."""
        import json as _json
        from datetime import date as d

        report = run_full_validation("XAUUSD_M5.csv", config_path=None)
        files = export_validation_report(report, output_dir=tmp_path, report_date=d(2024, 6, 1))

        with open(files["json"]) as f:
            data = _json.load(f)

        assert "data_quality" in data
        assert data["data_quality"]["row_count"] == 1000
