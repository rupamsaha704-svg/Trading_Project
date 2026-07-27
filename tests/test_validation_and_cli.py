"""Tests for strict CSV validation and CLI argument handling."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_validation import (
    validate_dataset,
    validate_strict,
    get_quality_summary,
    DataQualityReport,
)
from backtest import parse_args


# =============================================================================
# STRICT VALIDATION TESTS
# =============================================================================


class TestValidateStrict:

    def _valid_df(self):
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "tick_volume": [100, 200, 300, 400, 500],
        })

    def test_valid_data_passes_strict(self):
        """Valid data should not raise."""
        df = self._valid_df()
        report = validate_strict(df, "test.csv")
        assert isinstance(report, DataQualityReport)
        assert report.summary["ERROR"] == 0

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({"time": [1], "open": [1.0]})
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_missing_ohlc_values_raises(self):
        """NaN in OHLC columns should raise ValueError."""
        df = self._valid_df()
        df.loc[2, "close"] = None
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_duplicate_timestamps_raises(self):
        """Duplicate timestamps should raise ValueError."""
        df = self._valid_df()
        df.loc[3, "time"] = df.loc[2, "time"]
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_invalid_ohlc_relationships_raises(self):
        """High < Low should raise ValueError."""
        df = self._valid_df()
        df.loc[1, "high"] = 90.0  # lower than low
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_zero_prices_raises(self):
        """Zero price should raise ValueError."""
        df = self._valid_df()
        df.loc[0, "open"] = 0.0
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_negative_prices_raises(self):
        """Negative price should raise ValueError."""
        df = self._valid_df()
        df.loc[0, "low"] = -1.0
        # Also need to fix high >= low for the row to not trigger INVALID_OHLC first
        df.loc[0, "high"] = 101.0
        df.loc[0, "open"] = 100.0
        df.loc[0, "close"] = 100.5
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")


class TestNonPositivePrices:

    def test_positive_prices_pass(self):
        """All positive prices produce INFO finding."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "ALL_PRICES_POSITIVE" in codes
        assert "NON_POSITIVE_PRICES" not in codes

    def test_zero_price_detected(self):
        """Zero open price should produce ERROR finding."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [0.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "NON_POSITIVE_PRICES" in codes

    def test_negative_price_detected(self):
        """Negative close price should produce ERROR finding."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [-1.0, 11.5, 12.5],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "NON_POSITIVE_PRICES" in codes


class TestChronologicalOrder:

    def test_sorted_timestamps_pass(self):
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "TIMESTAMPS_SORTED" in codes

    def test_unsorted_timestamps_warned(self):
        df = pd.DataFrame({
            "time": [
                pd.Timestamp("2024-01-01 00:10"),
                pd.Timestamp("2024-01-01 00:05"),
                pd.Timestamp("2024-01-01 00:15"),
            ],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "UNSORTED_TIMESTAMPS" in codes


class TestGetQualitySummary:

    def test_summary_contains_expected_keys(self):
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "tick_volume": [100, 200, 300],
        })
        report = validate_dataset(df, "test.csv")
        summary = get_quality_summary(report)

        assert "file_path" in summary
        assert "row_count" in summary
        assert "info_count" in summary
        assert "warning_count" in summary
        assert "error_count" in summary
        assert "findings" in summary
        assert summary["error_count"] == 0

    def test_summary_only_includes_warnings_and_errors(self):
        """findings list should not contain INFO items."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "tick_volume": [100, 200, 300],
        })
        report = validate_dataset(df, "test.csv")
        summary = get_quality_summary(report)

        for finding in summary["findings"]:
            assert finding["severity"] in ("WARNING", "ERROR")


# =============================================================================
# CLI ARGUMENT TESTS
# =============================================================================


class TestCLIArgs:

    def test_default_args(self):
        """No args → all None defaults."""
        args = parse_args([])
        assert args.config is None
        assert args.data is None
        assert args.output_dir is None

    def test_config_arg(self):
        args = parse_args(["--config", "/path/to/config.json"])
        assert args.config == "/path/to/config.json"

    def test_data_arg(self):
        args = parse_args(["--data", "/path/to/data.csv"])
        assert args.data == "/path/to/data.csv"

    def test_output_dir_arg(self):
        args = parse_args(["--output-dir", "/tmp/reports"])
        assert args.output_dir == "/tmp/reports"

    def test_all_args_together(self):
        args = parse_args([
            "--config", "my_config.json",
            "--data", "my_data.csv",
            "--output-dir", "my_reports",
        ])
        assert args.config == "my_config.json"
        assert args.data == "my_data.csv"
        assert args.output_dir == "my_reports"


class TestDataQualityInReport:

    def test_data_quality_included_in_json_export(self, tmp_path):
        """When data_quality key is present in results, it appears in exported JSON."""
        from src.report_export import export_backtest_report
        from datetime import date

        results = {
            "starting_balance": 10000.0,
            "final_balance": 10100.0,
            "net_result": 100.0,
            "total_trades": 1,
            "wins": 1,
            "losses": 0,
            "win_rate": 100.0,
            "profit_factor": float("inf"),
            "expectancy": 100.0,
            "gross_profit": 100.0,
            "gross_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "equity_curve": [10000.0, 10100.0],
            "trade_log": [],
            "settings": {"spread_points": 0.3, "slippage_points": 0.1},
            "data_quality": {
                "file_path": "test.csv",
                "row_count": 100,
                "columns": ["time", "open", "high", "low", "close"],
                "info_count": 8,
                "warning_count": 1,
                "error_count": 0,
                "findings": [{"severity": "WARNING", "code": "MISSING_CANDLES", "message": "test"}],
            },
        }

        files = export_backtest_report(results, output_dir=tmp_path, report_date=date(2024, 1, 1))
        with open(files["json"]) as f:
            data = json.load(f)

        assert "data_quality" in data
        assert data["data_quality"]["row_count"] == 100
        assert data["data_quality"]["warning_count"] == 1



# =============================================================================
# PR #4 FIX: volume required, invalid datetime, clean CLI exit
# =============================================================================


class TestVolumeRequired:

    def test_missing_volume_column_raises(self):
        """tick_volume is now required; missing it should raise in strict mode."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            # no tick_volume
        })
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_volume_present_passes(self):
        """When tick_volume is present, no MISSING_COLUMNS error."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "tick_volume": [100, 200, 300],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings if f.severity == "ERROR"]
        assert "MISSING_COLUMNS" not in codes


class TestInvalidDatetime:

    def test_invalid_datetime_text_detected(self):
        """Unparseable datetime strings produce INVALID_DATETIME error."""
        df = pd.DataFrame({
            "time": ["2024-01-01 00:00", "not-a-date", "2024-01-01 00:10"],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "tick_volume": [100, 200, 300],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "INVALID_DATETIME" in codes
        # Check count
        finding = next(f for f in report.findings if f.code == "INVALID_DATETIME")
        assert finding.details["invalid_count"] == 1

    def test_invalid_datetime_raises_in_strict(self):
        """validate_strict raises on unparseable datetime."""
        df = pd.DataFrame({
            "time": ["2024-01-01 00:00", "garbage", "2024-01-01 00:10"],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "tick_volume": [100, 200, 300],
        })
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_strict(df, "test.csv")

    def test_valid_datetimes_no_error(self):
        """Valid datetime strings produce no INVALID_DATETIME finding."""
        df = pd.DataFrame({
            "time": ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10"],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "tick_volume": [100, 200, 300],
        })
        report = validate_dataset(df, "test.csv")
        codes = [f.code for f in report.findings]
        assert "INVALID_DATETIME" not in codes


class TestCLIExitOnInvalidDatetime:

    def test_backtest_exits_cleanly_on_invalid_data(self, tmp_path):
        """Backtest should exit with code 1 (not crash) when CSV has invalid datetimes."""
        import subprocess

        # Write a CSV with invalid datetime
        csv_content = "time,open,high,low,close,tick_volume\nnot-a-date,10,11,9,10.5,100\n"
        csv_file = tmp_path / "bad_data.csv"
        csv_file.write_text(csv_content)

        result = subprocess.run(
            [sys.executable, str(ROOT / "backtest.py"), "--data", str(csv_file)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

        # Should exit with code 1, not an unhandled exception
        assert result.returncode == 1
        # Should mention validation failure in stderr or stdout
        combined = result.stdout + result.stderr
        assert "validation" in combined.lower() or "FAILED" in combined
