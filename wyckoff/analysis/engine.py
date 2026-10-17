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
from .zigzag import Pivot, zigzag_pivots


@dataclass
class AnalysisResult:
    code: str
    name: str
    features: pd.DataFrame
    pivots: List[Pivot]
    events: List[Event]
    zones: List[Zone]
    signals: pd.DataFrame           # 逐K线: action/buy_score/sell_score/stop/entry_type/reason/phase/trend
    levels: List = field(default_factory=list)   # 合并后的支撑/阻力带
    latest: Dict = field(default_factory=dict)

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
        atr_arr = feats["atr"].to_numpy(float)
        atrp = np.where(c > 0, atr_arr / c, np.nan)
        thr = np.where(np.isfinite(atrp),
                       np.clip(zc.atr_mult * atrp, zc.min_threshold, zc.max_threshold),
                       zc.threshold)
    else:
        thr = zc.threshold
    pivots = zigzag_pivots(h, l, thr, zc.depth,
                           open_=feats["open"].to_numpy(float), close=c)
    ss = SupportSeries.compute(h, l, c, pivots, zone_max_width=cfg.zigzag.zone_max_width)
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
                          feats, pivots, events, zones, sig, list(ss.levels or []), latest)


def _make_signals(feats, events, trend, phases, zst, ss, cfg) -> pd.DataFrame:
    n = len(feats)
    c = feats["close"].to_numpy(float)
    vr = feats["vol_ratio"].to_numpy(float)
    pos60 = feats["pos_in_60"].to_numpy(float)

    buy = np.zeros(n)
    sell = np.zeros(n)
    stop = np.full(n, np.nan)
    actions = ["HOLD"] * n
    entries = [""] * n
    reasons = [""] * n

    confirmed = sorted(events, key=lambda e: e.confirm_idx)
    by_bar: Dict[int, List[Event]] = {}
    for e in confirmed:
        by_bar.setdefault(e.confirm_idx, []).append(e)

    # 失败Spring标记
    spring_fail_at = {e.failed_idx for e in events if e.etype == "SPRING" and e.failed_idx}

    for t in range(1, n):
        b, s = 0.0, 0.0
        st, ent, rsn = np.nan, "", []

        for e in by_bar.get(t, []):
            if e.etype == "SPRING" and e.failed_idx is None and e.kind in ("ideal", "needs_test"):
                # 书中铁律：熊市背景不做Spring
                if trend[t] >= 0 or phases[t] in (1, 2):
                    if e.kind == "ideal":
                        b += 3.0 * e.quality + 0.5
                    else:
                        b += 2.2 * e.quality
                    st = e.meta["break_low"] * 0.99
                    ent = "spring"
                    rsn.append(f"Spring({e.kind}@{e.price:.2f})")
            elif e.etype == "SPRING" and e.kind == "shock":
                s += 0.5  # 震仓：警示，不进场
                rsn.append("震仓Spring(需缩量二次测试)")
            elif e.etype == "TEST":
                b += 3.0
                st = e.meta.get("break_low", e.price * 0.985) * 0.99 if "break_low" in e.meta else e.price * 0.97
                ent = "test"
                rsn.append("Spring缩量二次测试成功")
            elif e.etype == "LPS":
                b += 2.5
                st = e.meta["ref"] * 0.97
                ent = "lps"
                rsn.append("LPS缩量回踩")
            elif e.etype in ("SOS", "JOC"):
                b += 1.2 * e.quality
                st = e.meta["ref"] * 0.97 if e.meta.get("ref") else c[t] * 0.95
                ent = ent or ("joc" if e.etype == "JOC" else "sos")
                rsn.append(f"{e.etype}放量突破")
            elif e.etype == "ST":
                # 缩量二次测试站回恐慌低点上方：供应消失的首个入场点（书中左侧买点）
                b += 2.4 * e.quality
                st = e.meta.get("test_low", e.price) * 0.99
                ent = ent or "st"
                rsn.append(f"ST缩量二次测试(量{e.meta.get('contraction', 0):.0%}低于恐慌)")
            elif e.etype == "SOW":
                s += 2.5
                rsn.append("SOW放量跌破支撑")
            elif e.etype == "UT":
                s += 1.8 * e.quality
                rsn.append("UT上冲回落(诱多)")
            elif e.etype == "BC":
                s += 1.5
                rsn.append("BC抢购高潮")
            elif e.etype == "NDR":
                s += 1.5
                rsn.append("无需求反弹")
            elif e.etype == "EVR":
                if e.kind == "top":
                    s += 1.0
                    rsn.append("顶部放量滞涨(努力结果背离)")
                elif e.kind == "bottom":
                    b += 0.8
                    rsn.append("底部放量滞跌(供应被吸收)")
            elif e.etype == "SC":
                b += 0.5
                rsn.append("SC恐慌抛售(观察)")

        # Spring失败回补卖出
        if t in spring_fail_at:
            s += 3.0
            rsn.append("Spring失败：跌破破位低点")

        # 供需区信号（独立策略线）
        if zst["demand_touch"][t] and zst["demand_quality"][t] >= 3.5 and trend[t] >= 0:
            lts = zst["lts_position"][t]
            if np.isfinite(lts) and lts <= 0.5:
                b += 1.5 + 0.5 * (0.5 - lts)
                ent = ent or "demand_zone"
                st = zst["demand_bottom"][t] * 0.995
                rsn.append(f"回踩新鲜需求区(s1={zst['demand_top'][t]:.2f})")
        if zst["supply_touch"][t] and zst["supply_quality"][t] >= 3.5 and trend[t] <= 0:
            s += 1.8
            rsn.append("触供给区(高位)")

        # 背景衰减：熊压买、牛压卖（书中"背景"权重；吸筹期允许左侧试仓）
        sc = cfg.signal
        if trend[t] == -1:
            b *= sc.bear_accum_mult if phases[t] == 1 else sc.bear_mult
        if trend[t] == 1:
            s *= 0.7
        if phases[t] == 0:
            b *= 0.5
        if phases[t] == 3:
            b *= sc.dist_mult

        buy[t], sell[t] = b, s
        # 信号规则：分差显著才行动
        action = "BUY" if b >= sc.buy_threshold and b > s * sc.confirm_margin else (
            "SELL" if s >= sc.sell_threshold and s > b * sc.confirm_margin else "HOLD")
        actions[t] = action
        entries[t] = ent if action == "BUY" else ""
        stop[t] = st if action == "BUY" else np.nan
        reasons[t] = ";".join(rsn)

    out = pd.DataFrame({
        "date": feats["date"].to_numpy(),
        "action": actions,
        "buy_score": buy,
        "sell_score": sell,
        "stop": stop,
        "entry_type": entries,
        "reason": reasons,
    })
    out["phase"] = [PHASES[p] for p in phases]
    out["trend"] = trend
    return out


def _summarize(sig: pd.DataFrame, feats: pd.DataFrame, events: List[Event],
               zones: List[Zone], cfg: StrategyConfig) -> Dict:
    t = len(feats) - 1
    c = feats["close"].to_numpy(float)
    recent = [e for e in events if t - e.confirm_idx <= 20]
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
             "failed": e.failed_idx is not None}
            for e in recent
        ],
        "active_zones": [
            {"type": z.ztype, "top": round(z.top, 3), "bottom": round(z.bottom, 3),
             "s1": round(z.s1, 3), "returns": z.returns, "created_idx": z.created_idx}
            for z in zones if z.active and z.created_idx > t - 120
        ][:8],
    }
