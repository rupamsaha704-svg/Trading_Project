"""Load and validate strategy configuration from JSON.

Usage:
    from src.config_loader import load_config, get_backtest_settings
    config = load_config("strategy_config.json")
    settings = get_backtest_settings(config)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# Default values used when keys are missing from the config file.
DEFAULTS: Dict[str, Any] = {
    "indicators": {
        "ema_window": 200,
        "atr_window": 14,
        "adx_window": 14,
    },
    "signals": {
        "retest_tolerance": 0.001,
        "adx_threshold": 20,
        "swing_strength": 2,
    },
    "risk": {
        "risk_percent": 1.0,
        "risk_reward": 2.0,
        "atr_stop_loss_multiplier": 2.0,
        "max_concurrent_positions": 1,
        "daily_loss_limit_pct": 3.0,
        "max_drawdown_pct": 5.0,
        "account_balance": 10000.0,
        "position_size_mode": "percent_of_equity",
    },
    "execution": {
        "spread_points": 0.30,
        "slippage_points": 0.10,
    },
    "data": {
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "csv_path": "XAUUSD_M5.csv",
    },
    "logging": {
        "level": "INFO",
        "log_to_file": True,
        "log_dir": "logs",
        "log_format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    },
}


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file, falling back to defaults.

    Args:
        config_path: Path to the JSON config file. If None or the file doesn't
                     exist, returns full defaults.

    Returns:
        Merged configuration dict (file values override defaults).
    """
    config = _deep_copy_defaults()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)

    # Deep-merge user config into defaults
    _deep_merge(config, user_config)
    return config


def get_backtest_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a flat settings dict suitable for run_backtest() from full config.

    Returns:
        Dict with keys matching the backtest SETTINGS format.
    """
    risk = config.get("risk", {})
    execution = config.get("execution", {})

    return {
        "starting_balance": risk.get("account_balance", 10000.0),
        "risk_percent": risk.get("risk_percent", 1.0),
        "risk_reward": risk.get("risk_reward", 2.0),
        "atr_stop_loss_multiplier": risk.get("atr_stop_loss_multiplier", 2.0),
        "max_concurrent_positions": risk.get("max_concurrent_positions", 1),
        "daily_loss_limit_pct": risk.get("daily_loss_limit_pct", 3.0),
        "max_drawdown_pct": risk.get("max_drawdown_pct", 5.0),
        "spread_points": execution.get("spread_points", 0.30),
        "slippage_points": execution.get("slippage_points", 0.10),
    }


def validate_config(config: Dict[str, Any]) -> list:
    """Validate configuration values. Returns a list of error strings (empty = valid)."""
    errors = []

    risk = config.get("risk", {})
    if risk.get("risk_percent", 1.0) <= 0:
        errors.append("risk.risk_percent must be > 0")
    if risk.get("risk_percent", 1.0) > 100:
        errors.append("risk.risk_percent must be <= 100")
    if risk.get("risk_reward", 2.0) <= 0:
        errors.append("risk.risk_reward must be > 0")
    if risk.get("atr_stop_loss_multiplier", 2.0) <= 0:
        errors.append("risk.atr_stop_loss_multiplier must be > 0")
    if risk.get("max_concurrent_positions", 1) < 1:
        errors.append("risk.max_concurrent_positions must be >= 1")
    if risk.get("account_balance", 10000.0) <= 0:
        errors.append("risk.account_balance must be > 0")

    execution = config.get("execution", {})
    if execution.get("spread_points", 0.0) < 0:
        errors.append("execution.spread_points must be >= 0")
    if execution.get("slippage_points", 0.0) < 0:
        errors.append("execution.slippage_points must be >= 0")

    indicators = config.get("indicators", {})
    for key in ("ema_window", "atr_window", "adx_window"):
        if indicators.get(key, 1) < 1:
            errors.append(f"indicators.{key} must be >= 1")

    signals = config.get("signals", {})
    if signals.get("swing_strength", 1) < 1:
        errors.append("signals.swing_strength must be >= 1")

    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_copy_defaults() -> Dict[str, Any]:
    """Return a fresh deep copy of DEFAULTS."""
    return json.loads(json.dumps(DEFAULTS))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
