"""市场周期状态机：积累→上涨→派发→下跌。

用"半衰期证据评分"代替单向状态锁定：每根K线对四个阶段分别累计
近端事件的带符号证据（过期自动衰减），取最高分；趋势背景做先验。
无未来函数：t 根只使用 confirm_idx<=t 的事件。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .events import Event

PHASES = ("markdown", "accumulation", "markup", "distribution")

# 事件 → 各阶段证据权重
EVIDENCE: Dict[str, Dict[str, float]] = {
    "SC_CANDIDATE": {"accumulation": 1.0},
    "SC_FAIL": {"markdown": 2.0},
    "SC":   {"accumulation": 3.0, "markdown": 0.5},
    "PS":   {"accumulation": 1.5, "markdown": 0.3},
    "AR":   {"accumulation": 1.5},
    "ST":   {"accumulation": 2.0},
    "SPRING": {"accumulation": 2.0, "markup": 0.8},
    "SPRING_FAIL": {"markdown": 2.0},
    "TEST": {"accumulation": 1.5, "markup": 2.5},
    "SOS":  {"markup": 2.5, "accumulation": 0.5},
    "JOC":  {"markup": 3.0},
    "LPS":  {"markup": 2.5},
    "BC":   {"distribution": 3.0, "markup": 0.3},
    "PSY":  {"distribution": 1.5},
    "UT":   {"distribution": 2.5},
    "SOW":  {"markdown": 3.0, "distribution": 0.8},
    "LPSY": {"markdown": 2.5},
    "NDR":  {"markdown": 1.5},
    "EVR":  {},
}
HALFLIFE = 40.0     # 证据半衰期（K线根数）
TREND_PRIOR = 1.2   # 趋势先验强度
PRIOR_DECAY_BARS = 120


def phase_series(n: int, events: List[Event], trend: np.ndarray, pos60: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=int)
    by_confirm: Dict[int, List[Event]] = {}
    for e in events:
        by_confirm.setdefault(e.confirm_idx, []).append(e)

    scores = np.zeros(4)          # markdown/accumulation/markup/distribution
    prior = np.zeros(4)           # 趋势先验（缓慢更新）
    prior_phase = -1
    prior_age = 10 ** 9

    for t in range(n):
        scores *= 0.5 ** (1.0 / HALFLIFE)
        prior_age += 1

        for e in by_confirm.get(t, []):
            weights = EVIDENCE.get(e.etype, {})
            w = 1.0
            if e.etype == "ST" and e.kind == "breakout_test":
                weights = {"markup": 2.0}
            for ph, v in weights.items():
                scores[PHASES.index(ph)] += v * w

        tr = int(trend[t])
        if tr != 0 and (prior_phase != tr or prior_age > PRIOR_DECAY_BARS):
            prior *= 0.3
            prior_phase = tr
            prior_age = 0
        if tr == 1:
            prior[2] = max(prior[2], TREND_PRIOR)
        elif tr == -1:
            prior[0] = max(prior[0], TREND_PRIOR)

        total = scores + prior
        # 高位放量滞涨/天量新高：强化派发判断
        if pos60[t] >= 0.85 and total[3] > 0.5:
            total[3] += 0.8
        if pos60[t] <= 0.15 and total[1] > 0.5:
            total[1] += 0.8
        out[t] = int(np.argmax(total))
    return out
