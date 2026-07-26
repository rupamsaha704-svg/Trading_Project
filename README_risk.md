# Risk management layer

The research-only risk layer is implemented in src/risk_manager.py and is intentionally backward compatible with the existing interface.

## Calculations

### ATR-based stop loss
- The stop distance is derived from ATR:
  - stop_distance = ATR × atr_stop_loss_multiplier
- For a long position:
  - stop_loss = entry_price - stop_distance
  - take_profit = entry_price + (stop_distance × risk_reward)
- For a short position:
  - stop_loss = entry_price + stop_distance
  - take_profit = entry_price - (stop_distance × risk_reward)

### Position sizing
- Risk amount is computed as a percentage of account balance:
  - risk_amount = account_balance × (risk_percent / 100)
- Position size is then computed from the stop distance:
  - position_size = risk_amount / stop_distance

### Risk validation
- Maximum concurrent positions is enforced with max_concurrent_positions.
- Daily loss is compared against a percentage-based limit:
  - daily_loss_limit = account_balance × (daily_loss_limit_pct / 100)
- Maximum drawdown is evaluated against a percentage-based drawdown threshold:
  - max_drawdown_limit = peak_equity × (max_drawdown_pct / 100)

## Configuration

The default values can be overridden with src/risk_config.json or by passing parameters directly to the RiskManager constructor.
