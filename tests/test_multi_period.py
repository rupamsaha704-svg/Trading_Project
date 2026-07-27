"""Tests for multi-period backtest comparison runner and data preparation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_prep import prepare_data, ensure_data_ready
from run_multi_period import slice_by_months, run_multi_period, export_multi_period_report


# =============================================================================
# DATA PREPARATION TESTS (fast — uses repo ZIP)
# =============================================================================


class TestDataPrep:

    def test_prepare_data_creates_csv_from_zip(self):
        """prepare_data should create XAUUSD_M5_12M.csv from the ZIP."""
        output = ROOT / "XAUUSD_M5_12M.csv"
        output.unlink(missing_ok=True)
        result = prepare_data(project_root=ROOT, force=True)
        assert result.exists()
        assert result.name == "XAUUSD_M5_12M.csv"
        df = pd.read_csv(result)
        assert len(df) > 60000
        assert "time" in df.columns
        assert "open" in df.columns
        assert "tick_volume" in df.columns

    def test_prepare_data_skips_if_exists(self):
        """If CSV already exists, prepare_data returns immediately."""
        ensure_data_ready(project_root=ROOT)
        # Second call should be instant (no re-extraction)
        result = prepare_data(project_root=ROOT, force=False)
        assert result.exists()

    def test_ensure_data_ready_returns_path(self):
        """ensure_data_ready is the simple entry point."""
        path = ensure_data_ready(project_root=ROOT)
        assert path.exists()
        assert path.name == "XAUUSD_M5_12M.csv"

    def test_prepared_csv_has_correct_columns(self):
        """Prepared CSV should have standard column names."""
        path = ensure_data_ready(project_root=ROOT)
        df = pd.read_csv(path, nrows=5)
        expected = {"time", "open", "high", "low", "close", "tick_volume"}
        assert expected.issubset(set(df.columns))

    def test_prepared_csv_time_is_parseable(self):
        """Time column should be parseable as datetime."""
        path = ensure_data_ready(project_root=ROOT)
        df = pd.read_csv(path, nrows=10)
        parsed = pd.to_datetime(df["time"], errors="coerce")
        assert parsed.isna().sum() == 0


# =============================================================================
# SLICE TESTS (fast)
# =============================================================================


class TestSliceByMonths:

    def test_slice_returns_valid_path(self):
        ensure_data_ready(project_root=ROOT)
        path = slice_by_months(str(ROOT / "XAUUSD_M5_12M.csv"), 6)
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_slice_reduces_row_count(self):
        ensure_data_ready(project_root=ROOT)
        path = slice_by_months(str(ROOT / "XAUUSD_M5_12M.csv"), 6)
        df_full = pd.read_csv(str(ROOT / "XAUUSD_M5_12M.csv"))
        df_slice = pd.read_csv(path)
        assert len(df_slice) < len(df_full)
        Path(path).unlink(missing_ok=True)

    def test_slice_date_range_within_bounds(self):
        ensure_data_ready(project_root=ROOT)
        path = slice_by_months(str(ROOT / "XAUUSD_M5_12M.csv"), 6)
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        span_days = (df["time"].max() - df["time"].min()).days
        assert span_days <= 185
        assert span_days >= 150
        Path(path).unlink(missing_ok=True)


# =============================================================================
# FULL MULTI-PERIOD TESTS (slow — runs full validation)
# =============================================================================


class TestRunMultiPeriod:

    @pytest.mark.slow
    def test_report_has_all_periods(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        # Should have 3 period keys (NNNd, 8m, 6m)
        assert len(report["periods"]) == 3
        keys = list(report["periods"].keys())
        assert "8m" in keys
        assert "6m" in keys

    @pytest.mark.slow
    def test_report_has_data_quality(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        assert "data_quality" in report
        assert report["data_quality"]["error_count"] == 0

    @pytest.mark.slow
    def test_first_period_labeled_as_available(self):
        """First period should be labeled as 'Available N-day period'."""
        report = run_multi_period("XAUUSD_M5_12M.csv")
        table = report["comparison_table"]
        assert "Available" in table[0]["label"]
        assert "day" in table[0]["label"]

    @pytest.mark.slow
    def test_comparison_table_has_correct_columns(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        table = report["comparison_table"]
        assert len(table) == 3
        required_keys = [
            "period", "label", "candles", "net_result", "total_trades", "win_rate",
            "profit_factor", "expectancy", "max_drawdown_pct",
            "oos_net_result", "stability_profitable_pct",
        ]
        for key in required_keys:
            assert key in table[0], f"Missing key: {key}"


class TestExportMultiPeriodReport:

    @pytest.mark.slow
    def test_export_creates_json_and_csv(self, tmp_path):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        files = export_multi_period_report(report, output_dir=str(tmp_path))
        assert files["json"].exists()
        assert files["csv"].exists()

    @pytest.mark.slow
    def test_csv_has_three_rows(self, tmp_path):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        files = export_multi_period_report(report, output_dir=str(tmp_path))
        df = pd.read_csv(files["csv"])
        assert len(df) == 3
