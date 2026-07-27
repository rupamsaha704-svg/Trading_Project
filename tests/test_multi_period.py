"""Tests for multi-period backtest comparison runner."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_multi_period import slice_by_months, run_multi_period, export_multi_period_report


class TestSliceByMonths:

    def test_slice_returns_valid_path(self):
        path = slice_by_months(str(ROOT / "XAUUSD_M5_12M.csv"), 6)
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_slice_reduces_row_count(self):
        path = slice_by_months(str(ROOT / "XAUUSD_M5_12M.csv"), 6)
        df_full = pd.read_csv(str(ROOT / "XAUUSD_M5_12M.csv"))
        df_slice = pd.read_csv(path)
        assert len(df_slice) < len(df_full)
        Path(path).unlink(missing_ok=True)

    def test_slice_date_range_within_bounds(self):
        path = slice_by_months(str(ROOT / "XAUUSD_M5_12M.csv"), 6)
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        span_days = (df["time"].max() - df["time"].min()).days
        assert span_days <= 185
        assert span_days >= 150
        Path(path).unlink(missing_ok=True)


class TestRunMultiPeriod:

    @pytest.mark.slow
    def test_report_has_all_periods(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        assert "12m" in report["periods"]
        assert "8m" in report["periods"]
        assert "6m" in report["periods"]

    @pytest.mark.slow
    def test_report_has_data_quality(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        assert "data_quality" in report
        assert report["data_quality"]["error_count"] == 0

    @pytest.mark.slow
    def test_comparison_table_has_correct_columns(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        table = report["comparison_table"]
        assert len(table) == 3
        required_keys = [
            "period", "candles", "net_result", "total_trades", "win_rate",
            "profit_factor", "expectancy", "max_drawdown_pct",
            "oos_net_result", "stability_profitable_pct",
        ]
        for key in required_keys:
            assert key in table[0], f"Missing key: {key}"

    @pytest.mark.slow
    def test_12m_has_more_candles_than_6m(self):
        report = run_multi_period("XAUUSD_M5_12M.csv")
        assert report["periods"]["12m"]["candles"] > report["periods"]["6m"]["candles"]


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
