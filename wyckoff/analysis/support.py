"""支撑/阻力【区间】：由已确认Zigzag拐点派生，逐K线维护有效性（无未来函数）。

区间定义（取拐点K线四价中的实体信息）：
- 支撑带 = [最低价, 实体上沿]   阻力带 = [实体下沿, 最高价]

合并规则（因果：拐点在 confirm_idx 到达时才加入，不追溯改写历史）：
- 支撑带【只可向下合并】：上沿在创建时锁定，后续更低拐点只拉低下沿；
- 阻力带镜像【只可向上合并】：下沿锁定，只抬上沿；
- 合并【只与相邻带】：只尝试贴价侧物理最近的那条带（支撑=上沿最高，阻力=下沿最低），
  不论它是否已破位：它已破位或合并后超宽，就另起新带，绝不跳过它去合并更远的带。
- 合并双校验：①与该带区间重合 ②合并后宽度 ≤ zone_max_width。

破位规则：
- 收盘跌破支撑带下沿tol% → 失效；破位当根仍可见（Spring发生在破位K线上）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .zigzag import Pivot


@dataclass
class Level:
    kind: str              # 'support' | 'resistance'
    bottom: float          # 区间下沿
    top: float             # 区间上沿
    created_idx: int       # 拐点价格索引
    known_from: int        # 何时可见（confirm_idx）
    broken_idx: Optional[int] = None
    touches: int = 1
    last_touch: int = -999       # 上次计数的K线：带内连续停留不重复计回踩

    @property
    def active(self) -> bool:
        return self.broken_idx is None

    @property
    def price(self) -> float:
        """代表价：支撑取上沿（贴价侧），阻力取下沿。"""
        return self.top if self.kind == "support" else self.bottom


def pivot_zone(p: Pivot) -> tuple:
    bottom, top = p.zone
    if not (np.isfinite(bottom) and np.isfinite(top)):
        bottom = top = p.price
    return float(bottom), float(top)


def merge_or_add(levels: List[Level], p: Pivot, zone_max_width: float = 0.05) -> Optional[Level]:
    """拐点并入现有带（重合+宽度双校验，支撑只向下/阻力只向上），否则新建。

    全局宽度限制：单根拐点带先裁剪到 zone_max_width 以内——
    支撑锚定下沿(最低价)，上沿最多 bottom×(1+w)；阻力锚定上沿(最高价)，下沿最多 top×(1-w)。
    """
    kind = "support" if p.kind == "low" else "resistance"
    bottom, top = pivot_zone(p)
    if top > bottom * (1 + zone_max_width):
        if kind == "support":
            top = bottom * (1 + zone_max_width)
        else:
            bottom = top / (1 + zone_max_width)
    # 严格相邻：先取【物理上最近】的那条重合带（不论是否已破位）。
    # 它已破位 → 就地另起新带（死带也算挡路的邻居，不许跳过它去合并更远的带）；
    # 它有效但合并后超宽 → 同样另起新带。
    ov = [lv for lv in levels if lv.kind == kind
          and lv.bottom <= top and lv.top >= bottom]          # ① 重合
    if ov:
        lv = max(ov, key=lambda x: x.top) if kind == "support" else min(ov, key=lambda x: x.bottom)
        if lv.active:
            if kind == "support":
                new_bottom, new_top = min(lv.bottom, bottom), lv.top   # 上沿锁定，只向下
            else:
                new_bottom, new_top = lv.bottom, max(lv.top, top)      # 下沿锁定，只向上
            center = (new_bottom + new_top) / 2
            if center > 0 and (new_top - new_bottom) / center <= zone_max_width:  # ② 宽度上限
                lv.bottom, lv.top = new_bottom, new_top
                lv.touches += 1
                return None
    new = Level(kind, bottom, top, p.price_idx, p.confirm_idx)
    levels.append(new)
    return new


def build_levels(pivots: List[Pivot], zone_max_width: float = 0.05) -> List[Level]:
    """静态视图（画图/检查用）：按 confirm 顺序增量合并——与 compute 内行为一致。"""
    levels: List[Level] = []
    for p in sorted(pivots, key=lambda x: x.confirm_idx):
        merge_or_add(levels, p, zone_max_width)
    return levels


@dataclass
class SupportSeries:
    support_bottom: np.ndarray      # 最近有效支撑带下沿（破位当根仍可见）
    support_top: np.ndarray         # 支撑带上沿
    mature_bottom: np.ndarray       # 成熟支撑带（形成>=min_age根）下沿
    mature_top: np.ndarray          # 成熟支撑带上沿
    resistance_bottom: np.ndarray   # 最近有效阻力带下沿
    resistance_top: np.ndarray
    support_touches: np.ndarray
    levels: Optional[List[Level]] = None   # 扫描结束时的支撑/阻力带全集（含已破位）

    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                pivots: List[Pivot], tol: float = 0.005, min_age: int = 12,
                zone_max_width: float = 0.05) -> "SupportSeries":
        n = len(close)
        levels: List[Level] = []
        by_confirm: dict = {}
        for p in pivots:
            by_confirm.setdefault(p.confirm_idx, []).append(p)
        s_bot = np.full(n, np.nan)
        s_top = np.full(n, np.nan)
        m_bot = np.full(n, np.nan)
        m_top = np.full(n, np.nan)
        r_bot = np.full(n, np.nan)
        r_top = np.full(n, np.nan)
        sup_touch = np.zeros(n, dtype=int)

        for t in range(n):
            # 拐点在 confirm 当根才进入世界（因果合并）
            for p in by_confirm.get(t, []):
                merge_or_add(levels, p, zone_max_width)

            known = [lv for lv in levels if lv.known_from <= t and lv.active]
            # 参考带＝"贴着现价"的那条：带下沿在现价之下即入选（现价落在带内也算），
            # 再取上沿最高者为最近支撑。旧写法 top<=close*1.01 会把"价在带内"的带排除，
            # 导致回踩深带时误用更低的带、Spring 触不到。
            sups = [lv for lv in known if lv.kind == "support" and lv.bottom <= close[t] * 1.01]
            # 读取先于破位标记：破位当根区间仍可见
            if sups:
                best = max(sups, key=lambda x: (x.top, x.touches))
                s_top[t], s_bot[t] = best.top, best.bottom
                sup_touch[t] = best.touches
                mature = [lv for lv in sups if t - lv.created_idx >= min_age]
                if mature:
                    bm = max(mature, key=lambda x: (x.top, x.touches))
                    m_top[t], m_bot[t] = bm.top, bm.bottom
            r_list = [lv for lv in known if lv.kind == "resistance" and lv.bottom >= close[t] * 0.99]
            if r_list:
                br = min(r_list, key=lambda x: x.bottom)
                r_bot[t], r_top[t] = br.bottom, br.top
                if high[t] >= br.bottom * (1 - 0.01) and close[t] <= br.top * (1 + tol):
                    br.touches += 1
            for lv in known:
                if lv.kind == "support" and close[t] < lv.bottom * (1 - tol):
                    lv.broken_idx = t
                elif lv.kind == "resistance" and close[t] > lv.top * (1 + tol):
                    lv.broken_idx = t
            if sups:
                b2 = max(sups, key=lambda x: x.top)
                if (low[t] <= b2.top * 1.01 and close[t] >= b2.bottom * (1 - tol)
                        and t - b2.last_touch >= 5):      # 回踩计数：离开带≥5根后再入带才算一次
                    b2.touches += 1
                    b2.last_touch = t
        return SupportSeries(s_bot, s_top, m_bot, m_top, r_bot, r_top, sup_touch, levels)


def pivots_for(df, threshold: float, depth: int) -> List[Pivot]:
    from .zigzag import zigzag_pivots
    return zigzag_pivots(df["high"].to_numpy(float), df["low"].to_numpy(float),
                         threshold, depth,
                         open_=df["open"].to_numpy(float), close=df["close"].to_numpy(float))
