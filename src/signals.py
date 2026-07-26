import pandas as pd


class SignalEngine:

    def __init__(self, retest_tolerance=0.001, adx_threshold=20):
        self.retest_tolerance = retest_tolerance
        self.adx_threshold = adx_threshold

    def generate_signal(self, df: pd.DataFrame):

        df = df.copy()

        df["BuySignal"] = False
        df["SellSignal"] = False
        df["SignalState"] = "idle"

        bullish_bos_level = None
        bearish_bos_level = None

        bullish_bos_active = False
        bearish_bos_active = False

        for i in range(1, len(df)):

            current = df.iloc[i]

            if bullish_bos_active and bullish_bos_level is not None:
                buy_condition = (
                    current["low"] <= bullish_bos_level + self.retest_tolerance
                    and current["close"] > bullish_bos_level
                    and current["close"] > current["open"]
                    and current["close"] > current["EMA200"]
                    and current["ADX14"] >= self.adx_threshold
                )

                if buy_condition:
                    df.loc[df.index[i], "BuySignal"] = True
                    df.loc[df.index[i], "SignalState"] = "buy_signal"
                    bullish_bos_active = False
                    bullish_bos_level = None

            if bearish_bos_active and bearish_bos_level is not None:
                sell_condition = (
                    current["high"] >= bearish_bos_level - self.retest_tolerance
                    and current["close"] < bearish_bos_level
                    and current["close"] < current["open"]
                    and current["close"] < current["EMA200"]
                    and current["ADX14"] >= self.adx_threshold
                )

                if sell_condition:
                    df.loc[df.index[i], "SellSignal"] = True
                    df.loc[df.index[i], "SignalState"] = "sell_signal"
                    bearish_bos_active = False
                    bearish_bos_level = None

            if bool(current.get("BullishBOS", False)):
                bullish_bos_level = current.get("BullishBOSLevel", None)
                bullish_bos_active = True
                df.loc[df.index[i], "SignalState"] = "bullish_bos_active"

            if bool(current.get("BearishBOS", False)):
                bearish_bos_level = current.get("BearishBOSLevel", None)
                bearish_bos_active = True
                df.loc[df.index[i], "SignalState"] = "bearish_bos_active"

        return df