import pandas as pd

from .indicator_engine import IndicatorEngine


class Indicators:

    def __init__(self, ema_window: int = 200, atr_window: int = 14, adx_window: int = 14):
        self.engine = IndicatorEngine()
        self.ema_window = ema_window
        self.atr_window = atr_window
        self.adx_window = adx_window

    def add_indicators(self, df: pd.DataFrame):

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        df["EMA200"] = self.engine.ema(df["close"], window=self.ema_window)
        df["ATR14"] = self.engine.atr(df["high"], df["low"], df["close"], window=self.atr_window)
        df["ADX14"] = self.engine.adx(df["high"], df["low"], df["close"], window=self.adx_window)

        return df