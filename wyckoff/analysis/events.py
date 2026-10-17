"""威科夫事件检测：单向前扫，事件在其"确认K线"产生交易信号，绝不引用未来数据。

事件：PS SC AR ST SPRING TEST SOS JOC LPS PSY BC UT SOW LPSY EVR NDR
依据《威科夫操盘法》（孟洪涛）第2/3/4章规则实现，参数见 StrategyConfig。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Event:
    etype: str
    idx: int                 # 事件价格所在K线（如Spring的破位K线）
    confirm_idx: int         # 可交易确认K线（信号最早可用时刻）
    price: float             # 关键价（支撑位/高低点）
    quality: float = 1.0     # 0-1 质量分（量价形态优劣）
    kind: str = ""           # 子类：ideal / needs_test / shock / top / bottom ...
    failed_idx: Optional[int] = None
    meta: Dict = field(default_factory=dict)


def trend_state(close, ema20, ema60) -> np.ndarray:
    st = np.zeros(len(close), dtype=int)
    above = (close > ema20) & (ema20 > ema60)
    below = (close < ema20) & (ema20 < ema60)
    st[above] = 1
    st[below] = -1
    return st


class EventScanner:
    def __init__(self, df, ss, cfg):
        self.n = len(df)
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.v = df["volume"].to_numpy(float)
        self.vr = df["vol_ratio"].to_numpy(float)
        self.vmax30 = df["vol_max_30"].to_numpy(float)
        self.body = df["body"].to_numpy(float)
        self.body_avg = df["body_avg_14"].to_numpy(float)
        self.upper_shadow = df["upper_shadow"].to_numpy(float)
        self.close_pos = df["close_pos"].to_numpy(float)
        self.pos60 = df["pos_in_60"].to_numpy(float)
        self.chg_avg = df["chg_avg_abs_14"].to_numpy(float)
        self.sup = ss.support_top           # 支撑带上沿（贴价侧）
        self.sup_bot = ss.support_bottom    # 支撑带下沿（影线侧）
        self.supm = ss.mature_top           # 成熟支撑带上沿（Spring参照）
        self.supm_bot = ss.mature_bottom    # 成熟支撑带下沿
        self.res = ss.resistance_bottom     # 阻力带下沿（贴价侧）
        self.trend = trend_state(self.c, df["ema20"].to_numpy(float), df["ema60"].to_numpy(float))
        self.cfg = cfg
        self.events: List[Event] = []

    # ------ 基础判定 ------
    def ultra_vol(self, t):
        return self.v[t] >= 0.8 * self.vmax30[t] or self.vr[t] >= 2.2

    def big_down(self, t):
        return self.c[t] < self.o[t] and self.body[t] >= 1.8 * self.body_avg[t]

    def big_up(self, t):
        return self.c[t] > self.o[t] and self.body[t] >= 1.8 * self.body_avg[t]

    def avg_vol_before(self, t, k=20):
        seg = self.v[max(0, t - k):t]
        return float(np.nanmean(seg)) if len(seg) else self.v[t]

    def add(self, ev):
        self.events.append(ev)

    # ------ 主扫描 ------
    def run(self) -> List[Event]:
        n = self.n
        vol_cfg, sp_cfg = self.cfg.volume, self.cfg.spring
        t = 61
        while t < n:
            last = self._recent
            handled = False

            # SC 恐慌抛售
            if (self.trend[t] <= 0 and self.pos60[t] <= 0.25 and self.c[t] < self.o[t]
                    and self.big_down(t) and self.close_pos[t] <= 0.4 and self.ultra_vol(t)
                    and not self._recent("SC", 20, t)):
                self.add(Event("SC", t, t, float(self.l[t]), quality=0.8,
                               meta={"vol": float(self.v[t]), "vol_ratio": float(self.vr[t])}))
                handled = True

            # AR 自动反弹（SC后1-6根）
            if not handled:
                sc = self._last("SC", t)
                if sc is not None and 1 <= t - sc.idx <= 6 and self.c[t] > self.o[t] and (
                        self.big_up(t) or self.vr[t] >= vol_cfg.high_vol_ratio):
                    self.add(Event("AR", t, t, float(self.h[t]), quality=0.6,
                                   meta={"sc_idx": sc.idx}))

            # Spring：必须【穿进成熟支撑带腰部或更深】才算下冲；只刮到带上沿不算Spring。
            # deep=进一步刺破带下沿（创新低）。置于ST/PSY之前：同一根回踩K线可以既是ST又是Spring。
            if not handled and np.isfinite(self.supm[t]) and np.isfinite(self.supm_bot[t]):
                waist = self.supm_bot[t] + sp_cfg.recover_frac * (self.supm[t] - self.supm_bot[t])
                if self.l[t] <= waist * (1 + sp_cfg.recover_margin):
                    deep = self.l[t] < self.supm_bot[t] * (1 - 0.002)
                    if not self._spring_cool(t, self.supm[t], sp_cfg):
                        ev = self._detect_spring(t, sp_cfg, vol_cfg, deep, waist)
                        if ev is not None:
                            self.add(ev)
                            t = ev.confirm_idx + 1
                            continue

            # ST 二次测试：SC后4-30根回踩恐慌低点不破+缩量，并在recover_window内收阳站回低点上方。
            # 书中ST成功=供应消失的首个可入场点；确认根才可用（confirm_idx），不回溯。
            if not handled:
                sc = self._last("SC", t)
                if sc is not None and 4 <= t - sc.idx <= 30 and not self._st_cool(t, sc.idx, sp_cfg):
                    if (self.l[t] <= sc.price * 1.02 and self.l[t] >= sc.price * 0.995
                            and self.v[t] < 0.75 * sc.meta["vol"]):
                        ev = self._confirm_st(t, sc, sp_cfg)
                        if ev is not None:
                            self.add(ev)
                            t = ev.confirm_idx + 1
                            continue

            # PSY 初步供给：高位放量收低且带长上影或长阴（收紧，避免误判）
            if not handled and self.pos60[t] >= 0.72 and self.vr[t] >= 1.5 and self.c[t] < self.o[t] \
                    and (self.upper_shadow[t] >= self.body[t] or self.big_down(t)) \
                    and self._last("BC", t) is None and not self._recent("PSY", 20, t):
                self.add(Event("PSY", t, t, float(self.h[t]), quality=0.5))
                handled = True
            if not handled and self.pos60[t] <= 0.35 and self.vr[t] >= 1.3 and self.c[t] > self.o[t] \
                    and self.close_pos[t] >= 0.7 and self.trend[t] == -1:
                if self._last("SC", t) is None and not self._recent("PS", 15, t):
                    self.add(Event("PS", t, t, float(self.c[t]), quality=0.5))
                    handled = True

            # Spring 的二次测试
            if not handled:
                sp_ev = self._last("SPRING", t, only_active=True)
                if sp_ev is not None and sp_ev.kind in ("needs_test", "shock") \
                        and 2 <= t - sp_ev.confirm_idx <= sp_cfg.test_window \
                        and np.isfinite(self.supm[t]):
                    brk_low = sp_ev.meta["break_low"]
                    brk_vol = sp_ev.meta["break_vol"]
                    if (self.l[t] <= self.supm[t] * 1.015 and self.l[t] >= brk_low * 0.995
                            and self.c[t] > self.o[t]
                            and self.v[t] <= 0.6 * brk_vol
                            and self.body[t] <= self.body_avg[t]):
                        self.add(Event("TEST", t, t, float(self.supm[t]), quality=0.9,
                                       meta={"spring_idx": sp_ev.idx}))
                        sp_ev.kind = "tested"
                        handled = True

            # SOS / JOC：大阳线+放量收高，或2-3连阳(更高H/L/C且量递增)；越过有效阻力记JOC
            if not handled:
                single = self.big_up(t) and self.vr[t] >= 1.3 and self.close_pos[t] >= 0.7
                chain = (t >= 2 and self.c[t] > self.o[t] and self.c[t - 1] > self.o[t - 1]
                         and self.h[t] > self.h[t - 1] and self.l[t] > self.l[t - 1] and self.c[t] > self.c[t - 1]
                         and self.v[t] > self.v[t - 1] >= self.v[t - 2])
                if single or chain:
                    kind, ref, source = "", None, "spring"
                    if np.isfinite(self.res[t]) and self.c[t] > self.res[t]:
                        kind, ref, source = "JOC", float(self.res[t]), "breakout"
                    else:
                        sp_ev = self._last("SPRING", t, only_active=True)
                        if sp_ev is not None and t - sp_ev.confirm_idx <= 20 and self.c[t] >= sp_ev.price * 1.03:
                            kind, ref = "SOS", float(sp_ev.price)
                        elif self.c[t] >= self.c[t - 20] * 1.05 and self.pos60[t] >= 0.5 \
                                and not self._recent("SOS", 15, t):
                            kind, ref, source = "SOS", float(self.c[t - 20]), "streak"
                    if kind:
                        q = float(np.clip(self.vr[t] / 3.0, 0.2, 1.0))
                        if source == "streak":
                            q *= 0.6
                        self.add(Event(kind, t, t, float(self.c[t]), quality=q,
                                       meta={"ref": ref, "chain": bool(chain and not single),
                                             "source": source}))
                        handled = True

            # LPS：SOS/JOC后缩量浅回调、不破参照位，阳线确认
            if not handled:
                sos = self._last(("SOS", "JOC"), t)
                if sos is not None and 2 <= t - sos.idx <= 15:
                    wave_low = min(self.l[sos.idx:t + 1])
                    pullback = (self.c[sos.idx] - wave_low) / max(1e-9, self.c[sos.idx] - sos.meta["ref"])
                    pull_vol = self.v[sos.idx + 1:t + 1].max()
                    if (self.c[t] > self.o[t] and pullback <= 0.618
                            and pull_vol <= 0.8 * self.avg_vol_before(sos.idx)
                            and wave_low >= sos.meta["ref"]):
                        self.add(Event("LPS", t, t, float(self.c[t]), quality=0.75,
                                       meta={"sos_idx": sos.idx, "ref": sos.meta["ref"]}))
                        handled = True

            # BC 抢购高潮：高位、天量、新高、上影或弱收
            if not handled and self.trend[t] >= 0 and self.pos60[t] >= 0.75 and self.ultra_vol(t) \
                    and self.h[t] >= np.nanmax(self.h[max(0, t - 60):t]) - 1e-9 \
                    and (self.upper_shadow[t] >= self.body[t] or self.close_pos[t] <= 0.55) \
                    and not self._recent("BC", 20, t):
                self.add(Event("BC", t, t, float(self.h[t]), quality=0.7))
                handled = True

            # UT 上冲回落：冲破阻力后1-3根内收盘收回其下
            if not handled and np.isfinite(self.res[t]) and self.h[t] > self.res[t] * 1.002 \
                    and not self._recent_confirm_after("UT", t):
                level = float(self.res[t])
                for r2 in range(t, min(t + 4, self.n)):
                    if self.c[r2] < level:
                        self.add(Event("UT", t, r2, level,
                                       quality=0.85 if r2 == t else 0.7,
                                       meta={"vol_ratio": float(self.vr[t])}))
                        handled = True
                        break

            # SOW 弱势信号：书中三条件——最低/最高均低于昨日、收盘低于昨日、量不低于昨日；
            # 再叠加跌破支撑或派发背景
            if not handled:
                book3 = (self.l[t] < self.l[t - 1] and self.h[t] < self.h[t - 1]
                         and self.c[t] < self.c[t - 1] and self.v[t] >= self.v[t - 1])
                if book3 and self.c[t] < self.o[t]:
                    broke_sup = np.isfinite(self.sup_bot[t]) and self.c[t] < self.sup_bot[t] * 0.995
                    dist_ctx = self._last("BC", t) is not None or self._last("UT", t) is not None
                    if broke_sup and (self.big_down(t) or self.vr[t] >= 1.5 or dist_ctx):
                        self.add(Event("SOW", t, t, float(self.sup_bot[t]), quality=0.85,
                                       meta={"vol_ratio": float(self.vr[t])}))
                        handled = True

            # LPSY：SOW后无需求反抽不过50%
            if not handled:
                sow = self._last("SOW", t)
                if sow is not None and 3 <= t - sow.idx <= 20 and self.c[t] > self.c[t - 3]:
                    rebound = (self.c[t] - sow.price) / max(1e-9, self.h[max(0, sow.idx - 10):sow.idx + 1].max() - sow.price)
                    if rebound <= 0.5 and self.v[t] < self.avg_vol_before(t) * 0.8 \
                            and not self._recent("LPSY", 10, t):
                        self.add(Event("LPSY", t, t, float(self.c[t]), quality=0.6))
                        handled = True

            # EVR 努力与结果背离：天量微波动
            if not handled and self.vr[t] >= 2.2 and np.isfinite(self.chg_avg[t]):
                chg_abs = abs(self.c[t] / self.c[t - 1] - 1)
                if chg_abs <= 0.5 * self.chg_avg[t]:
                    loc = "top" if self.pos60[t] >= 0.6 else ("bottom" if self.pos60[t] <= 0.4 else "mid")
                    self.add(Event("EVR", t, t, float(self.c[t]), quality=0.5, kind=loc))
                    handled = True

            # NDR 无需求反弹：空头背景下连续小阳缩量
            if not handled and self.trend[t] == -1 and t >= 3:
                if (self.c[t] > self.c[t - 3]
                        and self.v[t] < self.v[t - 1] < self.v[t - 2] and self.vr[t] < 0.8
                        and max(self.body[t], self.body[t - 1], self.body[t - 2]) <= 0.8 * self.body_avg[t]):
                    self.add(Event("NDR", t, t, float(self.c[t]), quality=0.6))

            t += 1

        self._postprocess()
        return self.events

    # ------ 辅助 ------
    def _st_cool(self, t, sc_idx: int, sp_cfg) -> bool:
        """同一次SC只保留一组二次测试，避免连续回踩逐根登记。"""
        for e in reversed(self.events):
            if e.etype == "ST" and e.meta.get("sc_idx") == sc_idx and e.confirm_idx < t:
                return t - e.confirm_idx < sp_cfg.recover_window + 1
        return False

    def _confirm_st(self, t, sc, sp_cfg) -> Optional[Event]:
        """二次测试确认：recover_window 根内出现收阳且收盘站回恐慌低点之上。"""
        end = min(t + 1 + sp_cfg.recover_window, self.n)
        for r in range(t, end):
            if self.c[r] > self.o[r] and self.c[r] >= sc.price * (1 + sp_cfg.recover_margin):
                contraction = float(np.clip((1 - self.v[t] / sc.meta["vol"]) / 0.75, 0, 1))
                quality = float(np.clip(0.55 * contraction + 0.45 * self.close_pos[r], 0, 1))
                return Event("ST", t, r, float(sc.price), quality=quality, kind="held",
                             meta={"sc_idx": sc.idx, "test_low": float(self.l[t]),
                                   "test_vol": float(self.v[t]),
                                   "contraction": contraction})
        return None

    def _spring_cool(self, t, zone_top: float, sp_cfg) -> bool:
        """同一条支撑带去重：相隔不足 recover_window+1 根的连续触带只算第一次，
        避免价格在带内横磨时逐根重复登记Spring。"""
        gap = sp_cfg.recover_window + 1
        for e in reversed(self.events):
            if e.etype == "SPRING" and e.confirm_idx < t and abs(e.meta.get("zone_top", 0.0) - zone_top) < 1e-9:
                return t - e.confirm_idx < gap
        return False

    def _detect_spring(self, t, sp_cfg, vol_cfg, deep: bool, waist: float) -> Optional[Event]:
        """Spring（支撑带版）：候选=影线穿进成熟支撑带腰部以下；recover_window 根内
        收盘站回腰部+margin 之上成立。
        分类：下冲供应小量→ideal（不必创新低）；有deep时量增→needs_test/shock。"""
        zone_top = self.supm[t]        # 支撑带上沿（实体顶）
        zone_bot = self.supm_bot[t]    # 支撑带下沿（影线底）
        recover_line = waist           # 腰部收复线（下冲须过此线，收复须站回此线）
        end = min(t + 1 + sp_cfg.recover_window, self.n)
        for r in range(t + 1, end):
            if self.c[r] >= recover_line * (1 + sp_cfg.recover_margin):
                # 书中"小量"指下冲供应：取窗口内【打到腰部以下】那些K线的最大量，
                # 与K线阴阳无关（插针下冲就是供应）；收复阳线放量属需求，不计。
                probe_bars = [i for i in range(t, r + 1) if self.l[i] <= recover_line]
                probe_vol = max((self.v[i] for i in probe_bars), default=self.v[t])
                window_vol_ratio = float(probe_vol / max(1e-9, self.avg_vol_before(t)))
                window_big = any(self.body[i] >= 1.8 * self.body_avg[i] and self.c[i] < self.o[i]
                                 for i in range(t, r + 1))
                # 触带K线不能是放量大阴（供应压倒需求）
                is_ideal = (window_vol_ratio <= vol_cfg.low_vol_ratio
                            and not self.big_down(t))
                if is_ideal:
                    kind = "ideal"
                elif deep:
                    kind = "shock" if (window_vol_ratio >= sp_cfg.vol_confirm_ratio and window_big) else "needs_test"
                else:
                    return None
                quality = float(np.clip(1.2 - 0.4 * window_vol_ratio, 0.1, 1.0))
                if self.c[r] >= zone_top * 1.005 and kind != "ideal":
                    quality = min(1.0, quality + 0.2)   # 强收复：直接吞回带上沿
                return Event("SPRING", t, r, float(zone_top), quality=quality, kind=kind,
                             meta={"break_low": float(min(self.l[i] for i in range(t, r + 1))),
                                   "break_vol": float(self.v[t]),
                                   "vol_ratio": window_vol_ratio,
                                   "zone_bottom": float(zone_bot),
                                   "zone_top": float(zone_top),
                                   "recover_line": float(recover_line),
                                   "depth": "deep" if deep else "waist"})
        return None

    def _recent(self, etype, within, t):
        e = self._last(etype, t, only_active=False)
        return e is not None and t - e.idx <= within

    def _recent_confirm_after(self, etype, t, within=3):
        """近within根内已有同型事件（避免同一上冲重复登记）。"""
        e = self._last(etype, t, only_active=False)
        return e is not None and t - e.idx <= within

    def _last(self, etype, t, only_active=False):
        names = (etype,) if isinstance(etype, str) else etype
        for e in reversed(self.events):
            if e.etype in names and e.confirm_idx <= t:
                if only_active and (e.kind in ("tested",) or e.failed_idx is not None):
                    continue
                if e.failed_idx is not None and e.failed_idx <= t:
                    continue
                return e
        return None

    def _postprocess(self):
        """Spring失败判定：确认后又跌破破位低点，或收复后4根内滞涨（无需求反弹）。"""
        for e in self.events:
            if e.etype != "SPRING":
                continue
            lo = e.meta["break_low"]
            for r in range(e.confirm_idx + 1, min(e.confirm_idx + 12, self.n)):
                if self.c[r] < lo * 0.999 and self.v[r] > self.v[r - 1] and self.c[r] < self.o[r]:
                    e.failed_idx = r
                    break
            if e.failed_idx is None:
                hi = self.c[e.confirm_idx:min(e.confirm_idx + 5, self.n)]
                if len(hi) >= 4 and hi.max() <= e.meta.get("recover_line", e.price) * 1.005:
                    e.failed_idx = e.confirm_idx + len(hi) - 1


def detect_events(df, ss, cfg) -> List[Event]:
    return EventScanner(df, ss, cfg).run()
