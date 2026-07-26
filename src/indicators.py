import pandas as pd

from .indicator_engine import IndicatorEngine


class Indicators:

    def __init__(self):
        self.engine = IndicatorEngine()

    def add_indicators(self, df: pd.DataFrame):

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        df["EMA200"] = self.engine.ema(df["close"], window=200)
        df["ATR14"] = self.engine.atr(df["high"], df["low"], df["close"], window=14)
        df["ADX14"] = self.engine.adx(df["high"], df["low"], df["close"], window=14)

        return df