"""静态Zigzag：拐点在出现后 delay 根K线才"确认"，确认前不可使用（无未来函数）。

约定：pivot 在索引 i 处形成极值，最早在 i+delay 根K线被观察到。
对外输出每个确认点 (price_idx, price, kind, confirm_idx)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Pivot:
    price_idx: int
    price: float
    kind: str          # 'high' | 'low'
    confirm_idx: int   # 该拐点最晚被确认的K线索引（price_idx + depth）
    body_top: float = float("nan")     # 拐点K线实体上沿 max(open, close)
    body_bottom: float = float("nan")  # 拐点K线实体下沿 min(open, close)

    @property
    def zone(self) -> tuple:
        """支撑/阻力区间：低点→[最低价, 实体上沿]；高点→[实体下沿, 最高价]。"""
        if self.kind == "low":
            return (self.price, self.body_top)
        return (self.body_bottom, self.price)


def zigzag_pivots(high: np.ndarray, low: np.ndarray, threshold, depth: int,
                  open_: np.ndarray = None, close: np.ndarray = None) -> List[Pivot]:
    """顺序扫描版Zigzag：维持候选极值，被反向波动或depth窗口确认。

    threshold 可为标量，或与 high 等长的数组（波动率自适应：每根K线自己的反转阈值）。
    """
    n = len(high)
    pivots: List[Pivot] = []
    if n <= depth + 1:
        return pivots
    adaptive_thr = not np.isscalar(threshold)

    def thr_at(idx: int) -> float:
        return float(threshold[idx]) if adaptive_thr else float(threshold)

    state = 0          # 1=候选为低点(等待向上反转), -1=候选为高点(等待向下反转)
    cand_idx = -1
    cand_price = np.nan

    # 初始化：用前 depth+1 根的相对极值定起点（先出现者作为首个候选拐点）
    seg_h = int(np.argmax(high[: depth + 1]))
    seg_l = int(np.argmin(low[: depth + 1]))
    if seg_h <= seg_l:
        cand_idx, cand_price, state = seg_h, float(high[seg_h]), -1  # 候选=高点
    else:
        cand_idx, cand_price, state = seg_l, float(low[seg_l]), 1    # 候选=低点

    i = depth + 1
    has_body = open_ is not None and close is not None

    def body_at(idx: int):
        if not has_body:
            return float("nan"), float("nan")
        return float(max(open_[idx], close[idx])), float(min(open_[idx], close[idx]))

    while i < n:
        if state == 1:  # 当前候选是低点，等待向上
            if low[i] < cand_price:       # 候选下移
                cand_idx, cand_price = i, low[i]
            elif high[i] - cand_price >= thr_at(cand_idx) * cand_price and i - cand_idx >= 1:
                # 反转成立：确认低点
                confirm = max(cand_idx + depth, i)
                bt, bb = body_at(cand_idx)
                pivots.append(Pivot(cand_idx, float(cand_price), "low", confirm, body_top=bt, body_bottom=bb))
                state = -1
                cand_idx, cand_price = i, high[i]
        elif state == -1:  # 候选高点
            if high[i] > cand_price:
                cand_idx, cand_price = i, high[i]
            elif cand_price - low[i] >= thr_at(cand_idx) * cand_price and i - cand_idx >= 1:
                confirm = max(cand_idx + depth, i)
                bt, bb = body_at(cand_idx)
                pivots.append(Pivot(cand_idx, float(cand_price), "high", confirm, body_top=bt, body_bottom=bb))
                state = 1
                cand_idx, cand_price = i, low[i]
        i += 1

    # 去重：同一根K线可能高低点都留下，保留后出现的
    dedup: List[Pivot] = []
    for p in pivots:
        if dedup and dedup[-1].price_idx == p.price_idx:
            # 保留更极端的
            if p.kind == "high" and p.price >= dedup[-1].price:
                dedup[-1] = p
            elif p.kind == "low" and p.price <= dedup[-1].price:
                dedup[-1] = p
        else:
            dedup.append(p)
    return dedup


def pivots_known_by(confirm_bar: int, pivots: List[Pivot]) -> List[Pivot]:
    """返回在 confirm_bar（含）之前已经可观察到的拐点。"""
    return [p for p in pivots if p.confirm_idx <= confirm_bar]
