from __future__ import annotations

import numpy as np
import pandas as pd


class IndicatorEngine:
    """Deterministic indicator computations for research use.

    The implementation is intentionally causal: each value at index i uses only
    data from the current and earlier candles.
    """

    def ema(self, series: pd.Series, window: int) -> pd.Series:
        if window < 1:
            raise ValueError("window must be at least 1")
        if series.empty:
            return pd.Series(dtype=float)

        values = series.astype(float)
        result = pd.Series(np.nan, index=series.index, dtype=float)

        if len(values) == 0:
            return result

        result.iloc[0] = values.iloc[0]
        alpha = 2.0 / (window + 1.0)
        for i in range(1, len(values)):
            if pd.isna(values.iloc[i]):
                result.iloc[i] = result.iloc[i - 1]
            else:
                prev = result.iloc[i - 1]
                if pd.isna(prev):
                    result.iloc[i] = values.iloc[i]
                else:
                    result.iloc[i] = alpha * values.iloc[i] + (1 - alpha) * prev

        return result

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        if window < 1:
            raise ValueError("window must be at least 1")
        if len(high) == 0:
            return pd.Series(dtype=float)

        high_vals = high.astype(float)
        low_vals = low.astype(float)
        close_vals = close.astype(float)

        tr = pd.Series(np.nan, index=high.index, dtype=float)
        tr.iloc[0] = high_vals.iloc[0] - low_vals.iloc[0]
        for i in range(1, len(high_vals)):
            prev_close = close_vals.iloc[i - 1]
            tr.iloc[i] = max(
                high_vals.iloc[i] - low_vals.iloc[i],
                abs(high_vals.iloc[i] - prev_close),
                abs(low_vals.iloc[i] - prev_close),
            )

        atr_values = pd.Series(np.nan, index=high.index, dtype=float)
        if len(tr) == 0:
            return atr_values

        if len(tr) <= window:
            atr_values.iloc[-1] = tr.dropna().mean()
            return atr_values

        atr_values.iloc[window - 1] = tr.iloc[:window].mean()
        for i in range(window, len(tr)):
            prev_atr = atr_values.iloc[i - 1]
            if pd.isna(prev_atr):
                atr_values.iloc[i] = tr.iloc[i]
            else:
                atr_values.iloc[i] = ((prev_atr * (window - 1)) + tr.iloc[i]) / window

        return atr_values

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        if window < 1:
            raise ValueError("window must be at least 1")
        if len(high) == 0:
            return pd.Series(dtype=float)

        high_vals = high.astype(float)
        low_vals = low.astype(float)
        close_vals = close.astype(float)

        up_move = high_vals.diff().fillna(0.0)
        down_move = low_vals.diff().mul(-1).fillna(0.0)

        plus_dm = pd.Series(0.0, index=high.index, dtype=float)
        minus_dm = pd.Series(0.0, index=high.index, dtype=float)
        for i in range(1, len(high_vals)):
            if up_move.iloc[i] > down_move.iloc[i] and up_move.iloc[i] > 0:
                plus_dm.iloc[i] = up_move.iloc[i]
            if down_move.iloc[i] > up_move.iloc[i] and down_move.iloc[i] > 0:
                minus_dm.iloc[i] = down_move.iloc[i]

        atr = self.atr(high_vals, low_vals, close_vals, window=window)
        plus_di = pd.Series(np.nan, index=high.index, dtype=float)
        minus_di = pd.Series(np.nan, index=high.index, dtype=float)
        dx = pd.Series(np.nan, index=high.index, dtype=float)

        for i in range(window - 1, len(high_vals)):
            if pd.isna(atr.iloc[i]):
                continue
            if atr.iloc[i] == 0:
                plus_di.iloc[i] = 0.0
                minus_di.iloc[i] = 0.0
                dx.iloc[i] = 0.0
                continue

            plus_di_value = 100.0 * plus_dm.iloc[i] / atr.iloc[i]
            minus_di_value = 100.0 * minus_dm.iloc[i] / atr.iloc[i]
            plus_di.iloc[i] = plus_di_value
            minus_di.iloc[i] = minus_di_value
            denom = plus_di_value + minus_di_value
            if denom == 0:
                dx.iloc[i] = 0.0
            else:
                dx.iloc[i] = 100.0 * abs(plus_di_value - minus_di_value) / denom

        adx_values = pd.Series(np.nan, index=high.index, dtype=float)
        if len(high_vals) < (2 * window):
            return adx_values

        adx_values.iloc[2 * window - 1] = dx.iloc[window - 1 : 2 * window - 1].mean()
        for i in range(2 * window, len(high_vals)):
            prev_adx = adx_values.iloc[i - 1]
            if pd.isna(prev_adx):
                adx_values.iloc[i] = dx.iloc[i]
            else:
                adx_values.iloc[i] = ((prev_adx * (window - 1)) + dx.iloc[i]) / window

        return adx_values
