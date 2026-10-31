import numpy as np
import pandas as pd


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR, causal: each value uses only bars up to that row."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    previous = close.shift(1)
    tr = pd.concat([(high - low), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()


def volume_ratio(bars: pd.DataFrame, index: int, lookback: int = 20) -> float:
    start = max(0, index - lookback)
    prior = bars["volume"].iloc[start:index]
    if prior.empty or prior.mean() <= 0:
        return float("nan")
    return float(bars["volume"].iloc[index] / prior.mean())
