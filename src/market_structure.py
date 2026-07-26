import pandas as pd


class MarketStructure:

    def __init__(self, strength=2):
        self.strength = max(1, int(strength))

    def detect_swings(self, df: pd.DataFrame):

        df = df.copy()

        df["SwingHigh"] = False
        df["SwingLow"] = False

        strength = self.strength

        high_candidate_index = None
        high_candidate_value = None
        low_candidate_index = None
        low_candidate_value = None

        for i in range(strength, len(df)):

            current_high = float(df.iloc[i]["high"])
            current_low = float(df.iloc[i]["low"])

            window = df.iloc[max(0, i - strength):i]
            prior_high = float(window["high"].max()) if not window.empty else current_high
            prior_low = float(window["low"].min()) if not window.empty else current_low

            if high_candidate_index is None:
                if current_high > prior_high:
                    high_candidate_index = i
                    high_candidate_value = current_high
            else:
                if current_high > prior_high:
                    high_candidate_index = i
                    high_candidate_value = current_high
                elif current_high < high_candidate_value:
                    df.loc[df.index[high_candidate_index], "SwingHigh"] = True
                    high_candidate_index = None
                    high_candidate_value = None

            if low_candidate_index is None:
                if current_low < prior_low:
                    low_candidate_index = i
                    low_candidate_value = current_low
            else:
                if current_low < prior_low:
                    low_candidate_index = i
                    low_candidate_value = current_low
                elif current_low > low_candidate_value:
                    df.loc[df.index[low_candidate_index], "SwingLow"] = True
                    low_candidate_index = None
                    low_candidate_value = None

        df["Structure"] = None

        last_swing_high = None
        last_swing_low = None

        for i in range(len(df)):

            if df.iloc[i]["SwingHigh"]:
                current_high = float(df.iloc[i]["high"])

                if last_swing_high is not None:
                    if current_high > last_swing_high:
                        df.loc[df.index[i], "Structure"] = "HH"
                    elif current_high < last_swing_high:
                        df.loc[df.index[i], "Structure"] = "LH"

                last_swing_high = current_high

            if df.iloc[i]["SwingLow"]:
                current_low = float(df.iloc[i]["low"])

                if last_swing_low is not None:
                    if current_low > last_swing_low:
                        df.loc[df.index[i], "Structure"] = "HL"
                    elif current_low < last_swing_low:
                        df.loc[df.index[i], "Structure"] = "LL"

                last_swing_low = current_low

        return df

    def detect_bos(self, df: pd.DataFrame):

        df = df.copy()

        df["BullishBOS"] = False
        df["BearishBOS"] = False

        df["BullishBOSLevel"] = None
        df["BearishBOSLevel"] = None

        last_swing_high = None
        last_swing_low = None

        high_broken = False
        low_broken = False

        for i in range(len(df)):

            current_close = float(df.iloc[i]["close"])

            if bool(df.iloc[i]["SwingHigh"]):
                last_swing_high = float(df.iloc[i]["high"])
                high_broken = False

            if bool(df.iloc[i]["SwingLow"]):
                last_swing_low = float(df.iloc[i]["low"])
                low_broken = False

            if (
                last_swing_high is not None
                and current_close > last_swing_high
                and not high_broken
            ):
                df.loc[df.index[i], "BullishBOS"] = True
                df.loc[df.index[i], "BullishBOSLevel"] = last_swing_high
                high_broken = True

            if (
                last_swing_low is not None
                and current_close < last_swing_low
                and not low_broken
            ):
                df.loc[df.index[i], "BearishBOS"] = True
                df.loc[df.index[i], "BearishBOSLevel"] = last_swing_low
                low_broken = True

        return df