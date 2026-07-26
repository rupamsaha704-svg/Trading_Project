"""Tests for strategy config loading, validation, and structured logging."""

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, get_backtest_settings, validate_config, DEFAULTS
from src.logger import setup_logger


# =============================================================================
# CONFIG LOADER TESTS
# =============================================================================


class TestLoadConfig:

    def test_load_returns_defaults_when_no_file(self, tmp_path):
        """When config file doesn't exist, returns all defaults."""
        config = load_config(tmp_path / "nonexistent.json")
        assert config["risk"]["risk_percent"] == 1.0
        assert config["execution"]["spread_points"] == 0.30
        assert config["indicators"]["ema_window"] == 200

    def test_load_returns_defaults_when_none(self):
        """When None is passed, returns all defaults."""
        config = load_config(None)
        assert config == DEFAULTS

    def test_load_merges_partial_override(self, tmp_path):
        """User config partially overrides defaults; missing keys retain defaults."""
        user = {"risk": {"risk_percent": 2.5}, "execution": {"spread_points": 0.50}}
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(user))

        config = load_config(config_file)

        # Overridden
        assert config["risk"]["risk_percent"] == 2.5
        assert config["execution"]["spread_points"] == 0.50
        # Retained defaults
        assert config["risk"]["risk_reward"] == 2.0
        assert config["indicators"]["ema_window"] == 200
        assert config["logging"]["level"] == "INFO"

    def test_load_full_config_file(self, tmp_path):
        """Loading the actual strategy_config.json works without error."""
        config = load_config(ROOT / "strategy_config.json")
        assert config["data"]["symbol"] == "XAUUSD"
        assert config["risk"]["account_balance"] == 10000.0


class TestGetBacktestSettings:

    def test_extracts_flat_settings(self):
        """get_backtest_settings produces the correct flat dict."""
        config = load_config(None)
        settings = get_backtest_settings(config)

        assert settings["starting_balance"] == 10000.0
        assert settings["risk_percent"] == 1.0
        assert settings["risk_reward"] == 2.0
        assert settings["atr_stop_loss_multiplier"] == 2.0
        assert settings["max_concurrent_positions"] == 1
        assert settings["daily_loss_limit_pct"] == 3.0
        assert settings["max_drawdown_pct"] == 5.0
        assert settings["spread_points"] == 0.30
        assert settings["slippage_points"] == 0.10

    def test_custom_values_propagate(self, tmp_path):
        """Custom config values appear in backtest settings."""
        user = {"risk": {"risk_percent": 0.5, "account_balance": 50000.0}}
        config_file = tmp_path / "custom.json"
        config_file.write_text(json.dumps(user))

        config = load_config(config_file)
        settings = get_backtest_settings(config)

        assert settings["starting_balance"] == 50000.0
        assert settings["risk_percent"] == 0.5


class TestValidateConfig:

    def test_valid_defaults_pass(self):
        """Default config passes validation."""
        config = load_config(None)
        errors = validate_config(config)
        assert errors == []

    def test_negative_risk_percent_fails(self):
        config = load_config(None)
        config["risk"]["risk_percent"] = -1.0
        errors = validate_config(config)
        assert any("risk_percent" in e for e in errors)

    def test_zero_account_balance_fails(self):
        config = load_config(None)
        config["risk"]["account_balance"] = 0.0
        errors = validate_config(config)
        assert any("account_balance" in e for e in errors)

    def test_negative_spread_fails(self):
        config = load_config(None)
        config["execution"]["spread_points"] = -0.1
        errors = validate_config(config)
        assert any("spread_points" in e for e in errors)

    def test_zero_swing_strength_fails(self):
        config = load_config(None)
        config["signals"]["swing_strength"] = 0
        errors = validate_config(config)
        assert any("swing_strength" in e for e in errors)

    def test_risk_percent_over_100_fails(self):
        config = load_config(None)
        config["risk"]["risk_percent"] = 101.0
        errors = validate_config(config)
        assert any("risk_percent" in e for e in errors)


# =============================================================================
# LOGGER TESTS
# =============================================================================


class TestLogger:

    def test_setup_logger_returns_logger_instance(self):
        """setup_logger returns a logging.Logger."""
        # Use unique name to avoid handler accumulation across tests
        logger = setup_logger("test_instance_check", {"logging": {"level": "DEBUG", "log_to_file": False}})
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_instance_check"

    def test_logger_respects_level(self):
        """Logger level is set correctly from config."""
        logger = setup_logger("test_level", {"logging": {"level": "WARNING", "log_to_file": False}})
        assert logger.level == logging.WARNING

    def test_logger_writes_to_file(self, tmp_path):
        """When log_to_file=True, a log file is created."""
        config = {
            "logging": {
                "level": "INFO",
                "log_to_file": True,
                "log_dir": str(tmp_path),
                "log_format": "%(message)s",
            }
        }
        logger = setup_logger("test_file_write", config)
        logger.info("hello from test")

        log_file = tmp_path / "test_file_write.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello from test" in content

    def test_logger_no_file_when_disabled(self, tmp_path):
        """When log_to_file=False, no file is created."""
        config = {
            "logging": {
                "level": "INFO",
                "log_to_file": False,
                "log_dir": str(tmp_path),
            }
        }
        setup_logger("test_no_file", config)
        log_file = tmp_path / "test_no_file.log"
        assert not log_file.exists()

    def test_logger_does_not_duplicate_handlers(self):
        """Calling setup_logger twice with same name doesn't add duplicate handlers."""
        config = {"logging": {"level": "INFO", "log_to_file": False}}
        logger1 = setup_logger("test_no_dup", config)
        handler_count = len(logger1.handlers)
        logger2 = setup_logger("test_no_dup", config)
        assert len(logger2.handlers) == handler_count
        assert logger1 is logger2
