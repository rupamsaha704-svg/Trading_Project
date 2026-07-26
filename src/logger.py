"""Structured logging setup for the trading system.

Usage:
    from src.logger import setup_logger
    logger = setup_logger("backtest", config)
    logger.info("Backtest started", extra={"total_candles": 1000})

Provides:
    - Console handler (always active)
    - Optional file handler (rotated daily, configured via config)
    - Structured format with timestamps
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name (e.g. 'backtest', 'signals').
        config: Full strategy config dict. Uses config['logging'] section.

    Returns:
        Configured logging.Logger instance.
    """
    log_config = (config or {}).get("logging", {})

    level_str = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_format = log_config.get(
        "log_format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(log_format)

    # --- Console handler (always) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler (optional) ---
    if log_config.get("log_to_file", False):
        log_dir = Path(log_config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / f"{name}.log"

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve an existing logger by name (no reconfiguration)."""
    return logging.getLogger(name)
