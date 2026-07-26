"""Tests for backtest report export (CSV + JSON)."""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report_export import export_backtest_report


def _sample_results():
    return {
        "starting_balance": 10000.0,
        "final_balance": 10200.0,
        "net_result": 200.0,
        "total_trades": 2,
        "wins": 1,
        "losses": 1,
        "win_rate": 50.0,
        "profit_factor": 2.0,
        "expectancy": 100.0,
        "gross_profit": 300.0,
        "gross_loss": 100.0,
        "max_drawdown": -100.0,
        "max_drawdown_pct": -1.0,
        "equity_curve": [10000.0, 9900.0, 10200.0],
        "trade_log": [
            {
                "bar_index": 1,
                "side": "long",
                "entry": 100.0,
                "sl": 96.0,
                "tp": 108.0,
                "position_size": 25.0,
                "result": "LOSS",
                "pnl": -100.0,
                "exit_bar": 3,
                "note": "",
            },
            {
                "bar_index": 5,
                "side": "short",
                "entry": 100.0,
                "sl": 104.0,
                "tp": 92.0,
                "position_size": 25.0,
                "result": "WIN",
                "pnl": 300.0,
                "exit_bar": 8,
                "note": "",
            },
        ],
        "settings": {
            "starting_balance": 10000.0,
            "risk_percent": 1.0,
            "risk_reward": 2.0,
            "atr_stop_loss_multiplier": 2.0,
            "max_concurrent_positions": 1,
            "daily_loss_limit_pct": 3.0,
            "max_drawdown_pct": 5.0,
            "spread_points": 0.3,
            "slippage_points": 0.1,
        },
    }


def test_export_creates_csv_and_json(tmp_path):
    results = _sample_results()
    files = export_backtest_report(results, output_dir=tmp_path, report_date=date(2024, 6, 15))

    assert files["csv"].exists()
    assert files["json"].exists()
    assert files["csv"].name == "backtest_2024-06-15.csv"
    assert files["json"].name == "backtest_2024-06-15.json"


def test_csv_contains_trade_log(tmp_path):
    results = _sample_results()
    files = export_backtest_report(results, output_dir=tmp_path, report_date=date(2024, 6, 15))

    df = pd.read_csv(files["csv"])
    assert len(df) == 2
    assert "side" in df.columns
    assert "pnl" in df.columns
    assert df.iloc[0]["result"] == "LOSS"
    assert df.iloc[1]["result"] == "WIN"


def test_json_contains_metrics(tmp_path):
    results = _sample_results()
    files = export_backtest_report(results, output_dir=tmp_path, report_date=date(2024, 6, 15))

    with open(files["json"]) as f:
        data = json.load(f)

    assert data["report_date"] == "2024-06-15"
    assert data["total_trades"] == 2
    assert data["win_rate"] == 50.0
    assert data["profit_factor"] == 2.0
    assert data["settings"]["risk_percent"] == 1.0
    assert data["trade_count_by_side"]["long"] == 1
    assert data["trade_count_by_side"]["short"] == 1


def test_empty_trade_log_creates_header_only_csv(tmp_path):
    results = _sample_results()
    results["trade_log"] = []
    results["total_trades"] = 0

    files = export_backtest_report(results, output_dir=tmp_path, report_date=date(2024, 6, 15))

    df = pd.read_csv(files["csv"])
    assert len(df) == 0
    assert "bar_index" in df.columns
