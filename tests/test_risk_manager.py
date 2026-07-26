import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_manager import RiskManager


def test_calculate_levels_uses_atr_and_reward_for_long():
    manager = RiskManager(risk_percent=1.0, risk_reward=2.0, atr_stop_loss_multiplier=2.0)

    stop_loss, take_profit = manager.calculate_levels(100.0, 2.0, side="long")

    assert stop_loss == 96.0
    assert take_profit == 108.0


def test_calculate_position_size_uses_account_balance_and_stop_distance():
    manager = RiskManager(risk_percent=1.0)

    position_size = manager.calculate_position_size(
        account_balance=10000.0,
        entry_price=100.0,
        stop_loss=96.0,
    )

    assert position_size == 25.0


def test_validate_trade_blocks_when_limits_are_reached():
    manager = RiskManager(
        risk_percent=1.0,
        risk_reward=2.0,
        max_concurrent_positions=1,
        daily_loss_limit_pct=3.0,
        max_drawdown_pct=5.0,
    )

    result = manager.validate_trade(
        account_balance=10000.0,
        equity=9500.0,
        current_positions=1,
        daily_loss=400.0,
        peak_equity=10000.0,
    )

    assert result["allowed"] is False
    assert "max_concurrent_positions" in result["reasons"]
    assert "daily_loss_limit" in result["reasons"]
    assert "max_drawdown" in result["reasons"]


def test_config_file_is_loaded_when_available():
    manager = RiskManager(config_path=str(ROOT / "src" / "risk_config.json"))

    assert manager.risk_percent == 1.0
    assert manager.risk_reward == 2.0
    assert manager.max_concurrent_positions == 1
