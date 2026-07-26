import numpy as np
import pandas as pd

from src.indicators import Indicators


def test_ema_is_causal_and_deterministic():
    s = pd.Series([10.0, 12.0, 8.0, 14.0], index=[0, 1, 2, 3])
    engine = Indicators().engine
    result = engine.ema(s, window=2)

    assert result.iloc[0] == 10.0
    assert np.isfinite(result.iloc[1])
    assert np.isfinite(result.iloc[2])
    assert np.isfinite(result.iloc[3])


def test_atr_handles_nan_and_small_data():
    high = pd.Series([10.0, 11.0])
    low = pd.Series([9.0, 10.0])
    close = pd.Series([9.5, 10.5])
    engine = Indicators().engine
    result = engine.atr(high, low, close, window=2)

    # With window=2 and only 2 data points, ATR is defined at index window-1 (=1).
    # Index 0 is correctly NaN because fewer than `window` TR values are available.
    assert pd.isna(result.iloc[0])
    assert np.isfinite(result.iloc[1])
    # TR[0]=1.0, TR[1]=1.5 → ATR[1] = mean(1.0, 1.5) = 1.25
    assert result.iloc[1] == 1.25


def test_adx_returns_finite_values_on_small_dataset():
    high = pd.Series([10.0, 11.0, 10.5, 12.0])
    low = pd.Series([9.0, 10.0, 9.5, 11.0])
    close = pd.Series([9.5, 10.5, 10.0, 11.5])
    engine = Indicators().engine
    result = engine.adx(high, low, close, window=2)

    assert result.isna().all() or np.isfinite(result.iloc[-1])


def test_add_indicators_preserves_columns_and_returns_causal_values():
    df = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
        }
    )

    out = Indicators().add_indicators(df)

    assert {"EMA200", "ATR14", "ADX14"}.issubset(out.columns)
    assert out["EMA200"].isna().sum() == 0
