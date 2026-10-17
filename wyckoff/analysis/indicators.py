"""逐K线指标：全部只使用截至当前K线（含）的历史数据。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 df 上追加价量特征列（向量化，均为滚动窗口，无未来数据）。"""
    out = df.copy()
    open_, high, low, close, vol = (out[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))

    out["range"] = high - low
    out["body"] = np.abs(close - open_)
    out["body_pct"] = np.where(out["range"] > 0, out["body"] / out["range"], 0.0)
    out["close_pos"] = np.where(out["range"] > 0, (close - low) / out["range"], 0.5)  # 收盘在K线中的位置
    out["up"] = close > open_
    out["down"] = close < open_
    out["chg"] = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    out["upper_shadow"] = high - np.maximum(open_, close)
    out["lower_shadow"] = np.minimum(open_, close) - low

    # 真实波幅与ATR（14）
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out["tr"] = tr
    out["atr"] = _rolling_mean(tr, 14)

    # 成交量基准：20日均量与量比；滞后窗口用于"相对前日/相对近20日"
    out["vol_ma"] = _rolling_mean_lag(vol, 20, min_periods=5)
    out["vol_ratio"] = np.where(out["vol_ma"] > 0, vol / out["vol_ma"], 1.0)
    out["vol_max_30"] = _rolling_max_lag(vol, 30, min_periods=5)   # 此前30日最大量（不含今日）
    out["range_avg_14"] = _rolling_mean_lag(out["range"].to_numpy(float), 14, min_periods=5)
    out["body_avg_14"] = _rolling_mean_lag(out["body"].to_numpy(float), 14, min_periods=5)
    out["chg_avg_abs_14"] = _rolling_mean_lag(np.abs(out["chg"].to_numpy(float)), 14, min_periods=5)

    # 波段统计：创N日新高/新低、距N日高低点位置
    out["new_high_30"] = high >= _rolling_max_lag(high, 30, min_periods=1)
    out["new_low_30"] = low <= _rolling_min_lag(low, 30, min_periods=1)
    hh = _rolling_max(high, 60, min_periods=10)
    ll = _rolling_min(low, 60, min_periods=10)
    out["pos_in_60"] = np.where(hh > ll, (close - ll) / (hh - ll), 0.5)  # 60日内相对位置

    # 趋势背景：EMA20/60 与结构
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema60"] = out["close"].ewm(span=60, adjust=False).mean()
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=1).mean().to_numpy()


def _rolling_mean_lag(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """过去window根（不含当前根）的均值。"""
    return pd.Series(x).shift(1).rolling(window, min_periods=min_periods).mean().to_numpy()


def _rolling_max_lag(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    return pd.Series(x).shift(1).rolling(window, min_periods=min_periods).max().to_numpy()


def _rolling_min_lag(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    return pd.Series(x).shift(1).rolling(window, min_periods=min_periods).min().to_numpy()


def _rolling_max(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=min_periods).max().to_numpy()


def _rolling_min(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=min_periods).min().to_numpy()
