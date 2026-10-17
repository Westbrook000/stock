"""分析引擎入口：组合事件、阶段、支撑阻力与供需区，输出逐K线买卖信号与最新结论。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from .events import Event, detect_events, trend_state
from .indicators import add_candle_features
from .phase import PHASES, phase_series
from .supply_demand import Zone, detect_zones, zone_state_series
from .support import SupportSeries
from .zigzag import Pivot, day0_pivots, filter_pivots


@dataclass
class AnalysisResult:
    code: str
    name: str
    features: pd.DataFrame
    pivots: List[Pivot]
    events: List[Event]
    zones: List[Zone]
    signals: pd.DataFrame           # 逐K线: action/buy_score/sell_score/stop/entry_type/reason/phase/trend
    levels: List = field(default_factory=list)   # day0水平线历史版本
    latest: Dict = field(default_factory=dict)
    snapshot_levels: List = field(default_factory=list)  # 原版首尾端点参与的离线绘图快照

    @property
    def action(self) -> str:
        return str(self.signals["action"].iloc[-1])


def _events_asof(events: List[Event], t: int) -> List[Event]:
    return [e for e in events if e.confirm_idx <= t]


def run_analysis(df: pd.DataFrame, cfg: Optional[StrategyConfig] = None,
                code: str = "", name: str = "") -> AnalysisResult:
    cfg = cfg or StrategyConfig.load()
    feats = add_candle_features(df)
    h, l, c, v = (feats[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    n = len(feats)

    zc = cfg.zigzag
    if zc.adaptive:
        atr_ratio = feats["atr"].to_numpy(float) / np.maximum(c, np.finfo(float).eps)
        adaptive_values = atr_ratio * zc.atr_mult
        adaptive_values = np.where(np.isfinite(adaptive_values), adaptive_values, zc.threshold)
        threshold = np.clip(adaptive_values, zc.min_threshold, zc.max_threshold)
    else:
        threshold = zc.threshold
    pivots = day0_pivots(c, threshold)
    selected = filter_pivots(pivots, zc.min_bars_between_peaks, zc.peaks)
    ss = SupportSeries.compute(c, feats["atr"].to_numpy(float), selected, cfg.support)
    snapshot_levels = SupportSeries.snapshot(c, selected, cfg.support)
    live = {(level.pivot_indices,level.price):level for level in ss.levels if level.active}
    for level in snapshot_levels:
        reference = live.get((level.pivot_indices,level.price))
        if reference is not None:
            level.known_from = reference.known_from
        else:
            level.known_from,level.provisional = n,True
    events = detect_events(feats, ss, cfg)

    trend = trend_state(c, feats["ema20"].to_numpy(float), feats["ema60"].to_numpy(float))
    phases = phase_series(n, events, trend, feats["pos_in_60"].to_numpy(float))

    zones = detect_zones(
        h, l, c, v, feats["body_avg_14"].to_numpy(float), feats["vol_ratio"].to_numpy(float),
        feats["atr"].to_numpy(float),
        min_impulse_change=cfg.supply_demand.min_impulse_change,
        impulse_vol_ratio=cfg.supply_demand.impulse_vol_ratio,
        base_max_body_atr=cfg.supply_demand.base_max_body_atr,
        zone_width_atr_max=cfg.supply_demand.zone_width_atr_max,
    )
    zst = zone_state_series(h, l, c, v, zones)

    sig = _make_signals(feats, events, trend, phases, zst, ss, cfg)
    latest = _summarize(sig, feats, events, zones, cfg)
    return AnalysisResult(code or str(feats["code"].iloc[-1]), name or str(feats["name"].iloc[-1]),
                          feats, pivots, events, zones, sig, list(ss.levels or []), latest, snapshot_levels)


STRATEGY_VERSION = "day0_022_compatible_direct_v3"


def _entry_type(event, cfg):
    # Explicit booleans and pattern subtypes only. Quality and diagnostic scores are never gates.
    if event.meta.get("entry_allowed") is not True:
        return None
    if event.etype == "SPRING" and cfg.signal.entry_spring and event.kind in ("ideal", "absorption"):
        return "spring"
    if event.etype == "SC" and cfg.signal.entry_sc and event.kind == "confirmed":
        return "sc"
    if event.etype == "ST" and cfg.signal.entry_tests:
        return "st_breakout" if event.kind == "breakout_test" else "st_sc"
    if event.etype == "TEST" and cfg.signal.entry_tests:
        return "st_spring"
    if event.etype == "LPS" and cfg.signal.entry_lps:
        return "lps"
    if event.etype in ("JOC", "SOS") and cfg.signal.entry_breakouts:
        return event.etype.lower()
    return None


def _make_signals(feats, events, trend, phases, zst, ss, cfg):
    n = len(feats)
    by_bar = {}
    for event in events:
        by_bar.setdefault(event.confirm_idx, []).append(event)
    priority = {"TEST": 0, "ST": 1, "LPS": 2, "SPRING": 3, "SC": 4, "JOC": 5, "SOS": 6}
    exit_types = {"SPRING_FAIL", "SC_FAIL", "SOW", "UT", "BC", "LPSY"}
    rows = []
    for t in range(n):
        known = by_bar.get(t, [])
        eligible = [(event, _entry_type(event, cfg)) for event in known]
        eligible = [(event, typ) for event, typ in eligible if typ]
        exits = [event for event in known if event.etype in exit_types]
        action, entry, stop, origin_idx = "HOLD", "", np.nan, -1
        reasons = [f"{e.etype}({e.kind})" if e.kind else e.etype for e in known]
        if exits:
            action = "SELL"
        elif eligible:
            chosen, entry = min(eligible, key=lambda item: priority[item[0].etype])
            action, origin_idx = "BUY", chosen.idx
            stop = chosen.meta.get("stop_price", chosen.meta.get("test_low", chosen.meta.get("break_low", chosen.price)) * .99)
            reasons.append(f"事件条件确认直接入场:{entry}")
        buy_score = sum((2 + float(e.quality) for e, _ in eligible), 0.0) if cfg.signal.diagnostic_scores else 0.0
        sell_score = sum((2 + float(e.quality) for e in exits), 0.0) if cfg.signal.diagnostic_scores else 0.0
        rows.append({"date": feats["date"].iloc[t], "action": action, "entry_type": entry,
                     "stop": stop, "buy_score": buy_score, "sell_score": sell_score,
                     "reason": ";".join(reasons), "event_idx": origin_idx,
                     "phase": PHASES[int(phases[t])], "trend": int(trend[t]),
                     "strategy_version": STRATEGY_VERSION})
    return pd.DataFrame(rows)


def _summarize(sig: pd.DataFrame, feats: pd.DataFrame, events: List[Event],
               zones: List[Zone], cfg: StrategyConfig) -> Dict:
    t = len(feats) - 1
    c = feats["close"].to_numpy(float)
    recent = [e for e in events if t - e.confirm_idx <= 20]
    failed_springs = {e.meta.get("spring_idx") for e in events
                      if e.etype == "SPRING_FAIL" and e.confirm_idx <= t}
    act = sig["action"].iloc[-1]
    atr_val = feats["atr"].iloc[-1]
    targets = []
    if act == "BUY" and np.isfinite(sig["stop"].iloc[-1]):
        entry = c[t]
        risk = entry - float(sig["stop"].iloc[-1])
        targets = [round(entry + 1.5 * risk, 3), round(entry + 3 * risk, 3)]
    return {
        "date": str(pd.Timestamp(feats["date"].iloc[-1]).date()),
        "close": round(float(c[t]), 3),
        "action": str(act),
        "strategy_version": STRATEGY_VERSION,
        "entry_rule": "confirmed_event",
        "buy_score": round(float(sig["buy_score"].iloc[-1]), 2),
        "sell_score": round(float(sig["sell_score"].iloc[-1]), 2),
        "entry_type": str(sig["entry_type"].iloc[-1]),
        "stop": float(sig["stop"].iloc[-1]) if np.isfinite(sig["stop"].iloc[-1]) else None,
        "targets": targets,
        "phase": str(sig["phase"].iloc[-1]),
        "trend": int(sig["trend"].iloc[-1]),
        "reason": str(sig["reason"].iloc[-1]),
        "atr": round(float(atr_val), 4) if np.isfinite(atr_val) else None,
        "recent_events": [
            {"type": e.etype, "kind": e.kind, "idx": int(e.confirm_idx),
             "date": str(pd.Timestamp(feats["date"].iloc[e.confirm_idx]).date()),
             "price": round(e.price, 3), "quality": round(e.quality, 2),
             "failed": (e.failed_idx is not None and e.failed_idx <= t)
                       or (e.etype == "SPRING" and e.idx in failed_springs)}
            for e in recent
        ],
        "active_zones": [
            {"type": z.ztype, "top": round(z.top, 3), "bottom": round(z.bottom, 3),
             "s1": round(z.s1, 3), "returns": z.returns, "created_idx": z.created_idx}
            for z in zones if z.active and z.created_idx > t - 120
        ][:8],
    }
