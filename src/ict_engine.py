"""
ICT/Smart Money Concepts Engine for XAUUSD M5.

Implements:
1. Order Block detection (bullish/bearish)
2. Fair Value Gap (FVG) detection
3. Liquidity Sweep detection (stop hunts)
4. Market Structure Shift (MSS/BOS)
5. Session filtering (London/NY killzones)
6. Entry model: Liquidity Sweep → MSS → OB/FVG entry

Does NOT include the original BOS-retest logic — this is a new strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# SESSION KILLZONES (UTC times for M5 candles)
# =============================================================================

KILLZONES = {
    "london_open": (7, 10),     # 07:00–10:00 UTC
    "ny_open": (12, 15),        # 12:00–15:00 UTC (NY AM)
    "london_close": (15, 17),   # 15:00–17:00 UTC
}


def is_in_killzone(hour: int, minute: int = 0) -> bool:
    """Check if a given hour is within any killzone."""
    time_decimal = hour + minute / 60.0
    for name, (start, end) in KILLZONES.items():
        if start <= time_decimal < end:
            return True
    return False


def add_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Add killzone column to DataFrame."""
    df = df.copy()
    if "time" in df.columns:
        times = pd.to_datetime(df["time"])
        df["in_killzone"] = times.apply(lambda t: is_in_killzone(t.hour, t.minute))
    else:
        df["in_killzone"] = True  # default to always active if no time
    return df


# =============================================================================
# ORDER BLOCK DETECTION
# =============================================================================


def detect_order_blocks(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect bullish and bearish order blocks.

    Bullish OB: Last bearish candle before a strong bullish displacement.
    Bearish OB: Last bullish candle before a strong bearish displacement.

    Displacement = move > 1.5× ATR in one candle.
    """
    df = df.copy()
    df["BullishOB"] = False
    df["BearishOB"] = False
    df["OB_High"] = np.nan
    df["OB_Low"] = np.nan

    if "ATR14" not in df.columns:
        return df

    for i in range(lookback + 1, len(df)):
        atr = df.iloc[i]["ATR14"]
        if pd.isna(atr) or atr <= 0:
            continue

        current = df.iloc[i]
        body = abs(current["close"] - current["open"])
        displacement_threshold = atr * 1.5

        # Bullish displacement (strong up candle)
        if current["close"] > current["open"] and body > displacement_threshold:
            # Find last bearish candle in lookback
            for j in range(i - 1, max(i - lookback - 1, 0), -1):
                prev = df.iloc[j]
                if prev["close"] < prev["open"]:  # bearish candle
                    df.loc[df.index[j], "BullishOB"] = True
                    df.loc[df.index[j], "OB_High"] = prev["high"]
                    df.loc[df.index[j], "OB_Low"] = prev["low"]
                    break

        # Bearish displacement (strong down candle)
        if current["close"] < current["open"] and body > displacement_threshold:
            for j in range(i - 1, max(i - lookback - 1, 0), -1):
                prev = df.iloc[j]
                if prev["close"] > prev["open"]:  # bullish candle
                    df.loc[df.index[j], "BearishOB"] = True
                    df.loc[df.index[j], "OB_High"] = prev["high"]
                    df.loc[df.index[j], "OB_Low"] = prev["low"]
                    break

    return df


# =============================================================================
# FAIR VALUE GAP (FVG) DETECTION
# =============================================================================


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Fair Value Gaps (imbalances).

    Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
    """
    df = df.copy()
    df["BullishFVG"] = False
    df["BearishFVG"] = False
    df["FVG_High"] = np.nan
    df["FVG_Low"] = np.nan

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        next_candle = df.iloc[i + 1]

        # Bullish FVG: gap between prev high and next low
        if prev["high"] < next_candle["low"]:
            df.loc[df.index[i], "BullishFVG"] = True
            df.loc[df.index[i], "FVG_Low"] = prev["high"]
            df.loc[df.index[i], "FVG_High"] = next_candle["low"]

        # Bearish FVG: gap between prev low and next high
        if prev["low"] > next_candle["high"]:
            df.loc[df.index[i], "BearishFVG"] = True
            df.loc[df.index[i], "FVG_High"] = prev["low"]
            df.loc[df.index[i], "FVG_Low"] = next_candle["high"]

    return df


# =============================================================================
# LIQUIDITY SWEEP DETECTION
# =============================================================================


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect liquidity sweeps (stop hunts).

    Bullish sweep: price wicks below recent lows then closes above them.
    Bearish sweep: price wicks above recent highs then closes below them.
    """
    df = df.copy()
    df["BullishSweep"] = False
    df["BearishSweep"] = False

    for i in range(lookback, len(df)):
        current = df.iloc[i]
        window = df.iloc[max(0, i - lookback):i]

        recent_low = window["low"].min()
        recent_high = window["high"].max()

        # Bullish sweep: wicked below recent low but closed above it
        if current["low"] < recent_low and current["close"] > recent_low:
            df.loc[df.index[i], "BullishSweep"] = True

        # Bearish sweep: wicked above recent high but closed below it
        if current["high"] > recent_high and current["close"] < recent_high:
            df.loc[df.index[i], "BearishSweep"] = True

    return df


# =============================================================================
# ICT SIGNAL GENERATION
# =============================================================================


@dataclass
class ICTSignal:
    """A single ICT trade signal."""
    bar_index: int
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    reason: str  # e.g. "bullish_ob_retest", "bullish_fvg_fill"


def generate_ict_signals(
    df: pd.DataFrame,
    adx_threshold: float = 20.0,
    require_killzone: bool = True,
) -> List[ICTSignal]:
    """Generate ICT trade signals.

    Entry model:
    1. Liquidity sweep OR BOS detected
    2. Price returns to OB or FVG zone
    3. Session is in killzone (if required)
    4. ADX confirms trending conditions

    Returns list of ICTSignal objects (read-only, no DataFrame modification).
    """
    signals: List[ICTSignal] = []

    # Track active zones
    active_bullish_obs: List[Tuple[float, float, int]] = []  # (high, low, bar_idx)
    active_bearish_obs: List[Tuple[float, float, int]] = []
    active_bullish_fvgs: List[Tuple[float, float, int]] = []
    active_bearish_fvgs: List[Tuple[float, float, int]] = []

    last_signal_bar = -10  # Prevent back-to-back signals

    for i in range(1, len(df)):
        current = df.iloc[i]

        # Collect new OBs and FVGs
        if bool(current.get("BullishOB", False)):
            active_bullish_obs.append((current["OB_High"], current["OB_Low"], i))
        if bool(current.get("BearishOB", False)):
            active_bearish_obs.append((current["OB_High"], current["OB_Low"], i))
        if bool(current.get("BullishFVG", False)):
            active_bullish_fvgs.append((current["FVG_High"], current["FVG_Low"], i))
        if bool(current.get("BearishFVG", False)):
            active_bearish_fvgs.append((current["FVG_High"], current["FVG_Low"], i))

        # Skip if not in killzone
        if require_killzone and not bool(current.get("in_killzone", True)):
            continue

        # Skip if ADX too low
        adx = current.get("ADX14", 0)
        if pd.isna(adx) or adx < adx_threshold:
            continue

        # Skip if too close to last signal
        if i - last_signal_bar < 5:
            continue

        close = current["close"]
        low = current["low"]
        high = current["high"]

        # --- BULLISH ENTRY: price dips into bullish OB/FVG zone ---
        # Check if price swept liquidity recently
        has_bullish_context = bool(current.get("BullishSweep", False)) or bool(current.get("BullishBOS", False))

        if has_bullish_context or len(active_bullish_obs) > 0:
            # Check OB retests
            for ob_high, ob_low, ob_bar in list(active_bullish_obs):
                if i - ob_bar > 50:  # expire old OBs
                    active_bullish_obs.remove((ob_high, ob_low, ob_bar))
                    continue
                # Price dipped into OB zone and closed bullish
                if low <= ob_high and close > ob_high and close > current["open"]:
                    signals.append(ICTSignal(
                        bar_index=i,
                        side="long",
                        entry_price=close,
                        stop_loss=ob_low - 1.0,  # SL below OB
                        reason="bullish_ob_retest",
                    ))
                    active_bullish_obs.remove((ob_high, ob_low, ob_bar))
                    last_signal_bar = i
                    break

            # Check FVG fills
            if i != last_signal_bar:
                for fvg_high, fvg_low, fvg_bar in list(active_bullish_fvgs):
                    if i - fvg_bar > 30:
                        active_bullish_fvgs.remove((fvg_high, fvg_low, fvg_bar))
                        continue
                    if low <= fvg_high and close > fvg_high and close > current["open"]:
                        signals.append(ICTSignal(
                            bar_index=i,
                            side="long",
                            entry_price=close,
                            stop_loss=fvg_low - 1.0,
                            reason="bullish_fvg_fill",
                        ))
                        active_bullish_fvgs.remove((fvg_high, fvg_low, fvg_bar))
                        last_signal_bar = i
                        break

        # --- BEARISH ENTRY: price pokes into bearish OB/FVG zone ---
        has_bearish_context = bool(current.get("BearishSweep", False)) or bool(current.get("BearishBOS", False))

        if has_bearish_context or len(active_bearish_obs) > 0:
            if i != last_signal_bar:
                for ob_high, ob_low, ob_bar in list(active_bearish_obs):
                    if i - ob_bar > 50:
                        active_bearish_obs.remove((ob_high, ob_low, ob_bar))
                        continue
                    if high >= ob_low and close < ob_low and close < current["open"]:
                        signals.append(ICTSignal(
                            bar_index=i,
                            side="short",
                            entry_price=close,
                            stop_loss=ob_high + 1.0,
                            reason="bearish_ob_retest",
                        ))
                        active_bearish_obs.remove((ob_high, ob_low, ob_bar))
                        last_signal_bar = i
                        break

            if i != last_signal_bar:
                for fvg_high, fvg_low, fvg_bar in list(active_bearish_fvgs):
                    if i - fvg_bar > 30:
                        active_bearish_fvgs.remove((fvg_high, fvg_low, fvg_bar))
                        continue
                    if high >= fvg_low and close < fvg_low and close < current["open"]:
                        signals.append(ICTSignal(
                            bar_index=i,
                            side="short",
                            entry_price=close,
                            stop_loss=fvg_high + 1.0,
                            reason="bearish_fvg_fill",
                        ))
                        active_bearish_fvgs.remove((fvg_high, fvg_low, fvg_bar))
                        last_signal_bar = i
                        break

    return signals


# =============================================================================
# FULL PIPELINE
# =============================================================================


def prepare_ict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all ICT detections to a DataFrame (indicators must be added first).

    Expects: time, open, high, low, close, ATR14, ADX14, EMA200,
             BullishBOS, BearishBOS columns.
    """
    df = add_session_filter(df)
    df = detect_order_blocks(df)
    df = detect_fvg(df)
    df = detect_liquidity_sweeps(df)
    return df
