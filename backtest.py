import pandas as pd

from src.utils import load_data
from src.indicators import Indicators
from src.market_structure import MarketStructure
from src.signals import SignalEngine


print("===================================")
print(" STRATEGY V3 BACKTEST")
print("===================================")


# =========================
# SETTINGS
# =========================

ATR_SL_MULTIPLIER = 2.0
RISK_REWARD = 2.0

STARTING_BALANCE = 10000.0


# =========================
# 1. Load Data
# =========================

df = load_data("XAUUSD_M5.csv")

print("Total Candles:", len(df))


# =========================
# 2. Indicators
# =========================

indicator = Indicators()

df = indicator.add_indicators(df)


# =========================
# 3. Market Structure
# =========================

market = MarketStructure()

df = market.detect_swings(df)

df = market.detect_bos(df)


# =========================
# 4. V3 Retest Signals
# =========================

signal_engine = SignalEngine()

df = signal_engine.generate_signal(df)


# =========================
# 5. Signal Count
# =========================

buy_signals = int(df["BuySignal"].sum())

sell_signals = int(df["SellSignal"].sum())

total_signals = buy_signals + sell_signals


print("\n========== SIGNAL RESULTS ==========")

print("Buy Signals:", buy_signals)

print("Sell Signals:", sell_signals)

print("Total Signals:", total_signals)


# =========================
# 6. Backtest Variables
# =========================

balance = STARTING_BALANCE

wins = 0

losses = 0

trade_results = []


# =========================
# 7. Trade Simulation
# =========================

for i in range(1, len(df) - 1):

    current = df.iloc[i]

    entry_price = current["close"]

    atr = current["ATR14"]


    if pd.isna(atr) or atr <= 0:

        continue


    # =========================
    # BUY TRADE
    # =========================

    if current["BuySignal"]:

        stop_loss = entry_price - (
            atr * ATR_SL_MULTIPLIER
        )

        take_profit = entry_price + (
            (entry_price - stop_loss)
            * RISK_REWARD
        )


        for j in range(i + 1, len(df)):

            future_candle = df.iloc[j]


            # SL

            if future_candle["low"] <= stop_loss:

                losses += 1

                balance -= 1

                trade_results.append("LOSS")

                break


            # TP

            elif future_candle["high"] >= take_profit:

                wins += 1

                balance += RISK_REWARD

                trade_results.append("WIN")

                break


    # =========================
    # SELL TRADE
    # =========================

    elif current["SellSignal"]:

        stop_loss = entry_price + (
            atr * ATR_SL_MULTIPLIER
        )

        take_profit = entry_price - (
            (stop_loss - entry_price)
            * RISK_REWARD
        )


        for j in range(i + 1, len(df)):

            future_candle = df.iloc[j]


            # SL

            if future_candle["high"] >= stop_loss:

                losses += 1

                balance -= 1

                trade_results.append("LOSS")

                break


            # TP

            elif future_candle["low"] <= take_profit:

                wins += 1

                balance += RISK_REWARD

                trade_results.append("WIN")

                break


# =========================
# 8. Final Results
# =========================

total_trades = wins + losses


if total_trades > 0:

    win_rate = (
        wins / total_trades
    ) * 100

else:

    win_rate = 0


net_result = balance - STARTING_BALANCE


print("\n===================================")

print(" STRATEGY V3 BACKTEST RESULTS")

print("===================================")


print("Starting Balance:", STARTING_BALANCE)

print("Final Balance:", balance)

print("Total Trades:", total_trades)

print("Wins:", wins)

print("Losses:", losses)

print("Win Rate:", round(win_rate, 2), "%")

print("Net Result:", round(net_result, 2))


print("===================================")

print("STRATEGY V3 BACKTEST COMPLETED!")

print("===================================")