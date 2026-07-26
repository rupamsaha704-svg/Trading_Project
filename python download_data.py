import MetaTrader5 as mt5
import pandas as pd

print("Connecting...")

if not mt5.initialize():
    print("Connection Failed")
    quit()

symbol = "XAUUSD"

rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1000)

if rates is None:
    print("No Data Found")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)

df['time'] = pd.to_datetime(df['time'], unit='s')

print(df.head())

df.to_csv("XAUUSD_M5.csv", index=False)

print("Data Saved Successfully!")

mt5.shutdown()