"""供需区（《供需交易法》Pine式简化）：基带+急速离开形成区域，逐K线维护活性与新鲜度。

规则要点（源自书）：
- 需求区：跌平跌/涨平涨中的"平"（小实体基带）后出现强势向上离开（大实体或大幅+放量）。
- 供给区：基带后急速向下离开。
- 区域被收盘贯穿即失效；回测次数越多新鲜度越低；新鲜未测区最优。
- LTS方位：区域处于近期波动区间低位→可买，高位→回避。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Zone:
    ztype: str              # 'D' 需求 | 'S' 供给
    top: float
    bottom: float
    created_idx: int
    s1: float               # 关键位：需求区=顶沿 / 供给区=底沿
    impulse_chg: float      # 离开冲动幅度（比例，绝对值）
    impulse_vol_ratio: float
    base_bars: int
    returns: int = 0        # 被回测次数
    last_touch_idx: int = -1
    broken_idx: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.broken_idx is None

    def freshness(self, t: int) -> float:
        """新鲜度评分 0-3：未测=3，首次回测=2，两次=1，更多=0。"""
        if self.returns == 0:
            return 3.0
        if self.returns == 1:
            return 2.0
        if self.returns == 2:
            return 1.0
        return 0.0

    def strength(self) -> float:
        """力度评分 0-2：离开冲动越大、量能越强。"""
        s = min(1.0, self.impulse_chg / 0.15) + min(1.0, max(0.0, self.impulse_vol_ratio - 1.0) / 2.0)
        return s

    def quality(self) -> float:
        """区域综合分 0-6：力度(0-2)+滞留(0.5-1)+新鲜度(0-3)。"""
        stay = 1.0 if self.base_bars >= 2 else 0.5
        return self.strength() + stay + self.freshness(0) * 0.67


def detect_zones(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                 body_avg: np.ndarray, vol_ratio: np.ndarray, atr: np.ndarray,
                 min_impulse_change: float = 0.03, impulse_vol_ratio: float = 1.5,
                 base_max_body_atr: float = 1.0, zone_width_atr_max: float = 3.0,
                 lookback_impulse: int = 6) -> List[Zone]:
    """扫描生成供需区（区形成于基带，离开确认后回溯登记，但可交易性从确认根开始）。"""
    n = len(close)
    zones: List[Zone] = []
    for t in range(20, n):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        chg = close[t] / close[t - lookback_impulse] - 1.0
        vr = vol_ratio[t]
        big_candle = abs(close[t] - open_proxy(low[t], high[t], close[t])) >= 1.5 * body_avg[t]

        for direction in ("up", "down"):
            if direction == "up":
                if chg < min_impulse_change or vr < impulse_vol_ratio or close[t] <= close[t - lookback_impulse]:
                    continue
                # 在冲量起点附近寻找小实体基带（最近 lookback_impulse 根内连续1-3根）
                z = _find_base(zones, high, low, close, t - lookback_impulse, t, "D",
                               body_avg, atr, base_max_body_atr, zone_width_atr_max)
                if z:
                    zones.append(Zone("D", z["top"], z["bottom"], z["start"], z["top"],
                                      abs(chg), vr, z["bars"], last_touch_idx=-1))
            else:
                if chg > -min_impulse_change or vr < impulse_vol_ratio or close[t] >= close[t - lookback_impulse]:
                    continue
                z = _find_base(zones, high, low, close, t - lookback_impulse, t, "S",
                               body_avg, atr, base_max_body_atr, zone_width_atr_max)
                if z:
                    zones.append(Zone("S", z["top"], z["bottom"], z["start"], z["bottom"],
                                      abs(chg), vr, z["bars"], last_touch_idx=-1))
    return zones


def open_proxy(low, high, close):
    """没有open时近似：取K线中点偏收盘对侧。此处low/high/close均为当根。"""
    return (high + low) / 2.0


def _find_base(zones: List[Zone], high, low, close, start: int, end: int, ztype: str,
               body_avg, atr, base_max_body_atr, zone_width_atr_max):
    """从冲量段起点向后找1-3根小实体基带。"""
    i = max(1, start)
    while i < end - 1:
        body = abs(close[i] - (high[i] + low[i]) / 2.0)
        if body <= base_max_body_atr * atr[i]:
            bars = 1
            top = high[i]
            bottom = low[i]
            j = i + 1
            while j < end - 1 and bars < 3:
                b2 = abs(close[j] - (high[j] + low[j]) / 2.0)
                if b2 <= base_max_body_atr * atr[j] and max(top, high[j]) - min(bottom, low[j]) <= zone_width_atr_max * atr[i]:
                    top = max(top, high[j])
                    bottom = min(bottom, low[j])
                    bars += 1
                    j += 1
                else:
                    break
            if top - bottom <= zone_width_atr_max * atr[i]:
                # 与已有区域重叠且同类型则跳过
                for z0 in zones:
                    if z0.ztype == ztype and z0.top > bottom and z0.bottom < top and i - z0.created_idx < 40:
                        return None
                return {"start": i, "top": float(top), "bottom": float(bottom), "bars": bars}
        i += 1
    return None


def zone_state_series(high, low, close, volume, zones: List[Zone]) -> dict:
    """逐K线：最近有效需求区（价下方）与供给区（价上方）、触碰/失效事件、LTS方位分。"""
    n = len(close)
    d_top = np.full(n, np.nan)
    d_bottom = np.full(n, np.nan)
    d_quality = np.full(n, 0.0)
    d_touch = np.zeros(n, dtype=bool)
    s_bot = np.full(n, np.nan)
    s_top = np.full(n, np.nan)
    s_quality = np.full(n, 0.0)
    s_touch = np.zeros(n, dtype=bool)
    lts_d = np.full(n, np.nan)  # 需求区相对120根波动区间的百分位

    rng_hi = np.array([high[max(0, i - 119): i + 1].max() for i in range(n)])
    rng_lo = np.array([low[max(0, i - 119): i + 1].min() for i in range(n)])

    for t in range(n):
        best_d = best_s = None
        for z in zones:
            if z.created_idx > t:
                continue
            if not z.active:
                continue
            # 失效：收盘穿透s1以外的区域边缘
            if z.ztype == "D" and close[t] < z.bottom:
                z.broken_idx = t
                continue
            if z.ztype == "S" and close[t] > z.top:
                z.broken_idx = t
                continue
            # 回测登记：进入区域
            if z.ztype == "D" and low[t] <= z.top and close[t] >= z.bottom:
                if t - z.last_touch_idx > 3:
                    z.returns += 1
                    z.last_touch_idx = t
                    d_touch[t] = True
            if z.ztype == "S" and high[t] >= z.bottom and close[t] <= z.top:
                if t - z.last_touch_idx > 3:
                    z.returns += 1
                    z.last_touch_idx = t
                    s_touch[t] = True
            if z.ztype == "D" and z.top <= close[t] * 1.02:
                if best_d is None or z.top > best_d.top:
                    best_d = z
            if z.ztype == "S" and z.bottom >= close[t] * 0.98:
                if best_s is None or z.bottom < best_s.bottom:
                    best_s = z
        if best_d is not None:
            d_top[t], d_bottom[t] = best_d.top, best_d.bottom
            d_quality[t] = best_d.quality()
            span = rng_hi[t] - rng_lo[t]
            lts_d[t] = (best_d.s1 - rng_lo[t]) / span if span > 0 else 0.5
        if best_s is not None:
            s_bot[t], s_top[t] = best_s.bottom, best_s.top
            s_quality[t] = best_s.quality()
    return {
        "demand_top": d_top, "demand_bottom": d_bottom, "demand_quality": d_quality,
        "demand_touch": d_touch, "supply_bottom": s_bot, "supply_top": s_top,
        "supply_quality": s_quality, "supply_touch": s_touch, "lts_position": lts_d,
    }
