import json
from pathlib import Path
from typing import Dict, Optional, Tuple


class RiskManager:

    def __init__(
        self,
        risk_percent: float = 1.0,
        risk_reward: float = 2.0,
        atr_stop_loss_multiplier: float = 2.0,
        max_concurrent_positions: int = 1,
        daily_loss_limit_pct: float = 3.0,
        max_drawdown_pct: float = 5.0,
        config_path: Optional[str] = None,
        account_balance: float = 10000.0,
        position_size_mode: str = "percent_of_equity",
    ):
        self.risk_percent = risk_percent
        self.risk_reward = risk_reward
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.account_balance = account_balance
        self.position_size_mode = position_size_mode

        if config_path is not None:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.risk_percent = float(config.get("risk_percent", self.risk_percent))
        self.risk_reward = float(config.get("risk_reward", self.risk_reward))
        self.atr_stop_loss_multiplier = float(
            config.get("atr_stop_loss_multiplier", self.atr_stop_loss_multiplier)
        )
        self.max_concurrent_positions = int(
            config.get("max_concurrent_positions", self.max_concurrent_positions)
        )
        self.daily_loss_limit_pct = float(
            config.get("daily_loss_limit_pct", self.daily_loss_limit_pct)
        )
        self.max_drawdown_pct = float(
            config.get("max_drawdown_pct", self.max_drawdown_pct)
        )
        self.account_balance = float(
            config.get("account_balance", self.account_balance)
        )
        self.position_size_mode = str(
            config.get("position_size_mode", self.position_size_mode)
        )

    def calculate_levels(
        self,
        entry_price: float,
        atr: float,
        side: str = "long",
    ) -> Tuple[float, float]:
        if atr <= 0:
            raise ValueError("ATR must be positive")

        if side.lower() not in {"long", "short"}:
            raise ValueError("side must be 'long' or 'short'")

        stop_distance = atr * self.atr_stop_loss_multiplier

        if side.lower() == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * self.risk_reward)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * self.risk_reward)

        return stop_loss, take_profit

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if stop_loss <= 0:
            raise ValueError("stop_loss must be positive")

        risk_amount = account_balance * (self.risk_percent / 100.0)
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            raise ValueError("stop distance must be positive")

        position_size = risk_amount / stop_distance
        return position_size

    def validate_trade(
        self,
        account_balance: float,
        equity: float,
        current_positions: int,
        daily_loss: float,
        peak_equity: float,
    ) -> Dict[str, object]:
        reasons = []

        if current_positions >= self.max_concurrent_positions:
            reasons.append("max_concurrent_positions")

        daily_loss_limit = account_balance * (self.daily_loss_limit_pct / 100.0)
        if daily_loss >= daily_loss_limit:
            reasons.append("daily_loss_limit")

        max_drawdown_limit = peak_equity * (self.max_drawdown_pct / 100.0)
        if equity <= (peak_equity - max_drawdown_limit):
            reasons.append("max_drawdown")

        return {"allowed": not reasons, "reasons": reasons}
