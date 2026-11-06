"""Append-only daily signals for the four supported bullish patterns."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from bisect import bisect_left
import time
from typing import List

import numpy as np
import pandas as pd

from .main_bottoms import (
    MAIN_RULES, PIVOT_SCALES, _make_main_pattern, _qualify_neckline_slope,
    _structure_candidates, _PotentialNecklinePairCache,
    _neckline_decline_check, _neckline_formation_coverage,
    _neckline_candidate_reversal_atr_multiple, _neckline_candidate_scale,
    _minimum_neckline_test_count, _same_neckline_boundary_phase,
    _neckline_test_tolerance_at,
)
from .patterns import PATTERN_LABELS, _line, _at
from .schemas import Pattern, Pivot

ENGINE_VERSION = "caisen-daily-continuous-v5"
BOUNDARY_FIRST_ENGINE_VERSION = "caisen-daily-boundary-first-v3"
SOURCE_RULE_VERSION = str(MAIN_RULES.get("source_rule_version", "unknown"))


class DailyRecognitionSession:
    """Causal daily recognizer. Bars, ATR and confirmed pivots are retained."""

    def __init__(self, code, kind, tick_size=0.01, progress_callback=None, total_bars=None,
                 boundary_first=False):
        self.code, self.kind = str(code), str(kind)
        self.boundary_first = bool(boundary_first)
        self.tick_size = max(float(tick_size), 1e-6)
        self.progress_callback = progress_callback
        self.total_bars = int(total_bars) if total_bars is not None else None
        self.progress_started_at = time.perf_counter()
        self.last_progress_at = 0.0
        self.bars: List[dict] = []
        self.date_strings: List[str] = []
        self.atr_values: List[float] = []
        self.scale_states = {name: self._new_zigzag() for name, _, _ in PIVOT_SCALES}
        self.scale_pivots = {name: [] for name, _, _ in PIVOT_SCALES}
        self.low_pivot_range_size = 1
        self.low_pivot_range_tree = np.full(2, np.inf, dtype=float)
        self.pressure = {"candidate": None, "pivots": []}
        self._pressure_pivot_revision = 0
        self._topology_pressure_revision = 0
        self.local_raw: List[Pivot] = []
        self.local_pivots: List[Pivot] = []
        self.active = {}
        self.consumed_neckline_phases = {}
        self.emitted_ids = set()
        self.emitted_events = []
        self.neckline_candidate_cache = {}
        self.neckline_pair_cache = {}
        self.neckline_coverage_cache = OrderedDict()
        self.neckline_potential_pair_cache = (_PotentialNecklinePairCache()
                                              if self.boundary_first else None)
        self.flag_candidates = {}
        self.last_signal_date = None
        self.performance = {"bars_appended": 0, "atr_updates": 0,
                            "pivot_updates": 0, "candidate_refreshes": 0,
                            "candidate_evaluations": 0, "prefix_frame_builds": 0}
        self._prefix_frame_cache = None
        self._prefix_frame_cache_length = -1

    def _frame_for_current_prefix(self):
        """Build one reusable frame and immutable numeric views for this prefix."""
        if (self._prefix_frame_cache is not None
                and self._prefix_frame_cache_length == len(self.bars)):
            return self._prefix_frame_cache
        frame = pd.DataFrame(self.bars)
        date_index = pd.DatetimeIndex(frame.date)
        frame.attrs["date_strings"] = self.date_strings
        frame.attrs["neckline_date_index"] = date_index
        low_values = pd.to_numeric(frame.low, errors="coerce").to_numpy(dtype=float)
        frame.attrs["neckline_low_values"] = low_values
        self._prefix_frame_cache = frame
        self._prefix_frame_cache_length = len(self.bars)
        self.performance["prefix_frame_builds"] += 1
        return frame

    def _prepare_neckline_frame(self, frame):
        """Attach the numeric history views shared by every candidate evaluation."""
        frame.attrs["neckline_candidate_cache"] = self.neckline_candidate_cache
        frame.attrs["neckline_pair_cache"] = self.neckline_pair_cache
        frame.attrs["neckline_coverage_cache"] = self.neckline_coverage_cache
        if self.neckline_potential_pair_cache is not None:
            frame.attrs["neckline_potential_pair_cache"] = self.neckline_potential_pair_cache
        frame.attrs["date_strings"] = self.date_strings
        for name in ("high", "low", "close"):
            key = f"neckline_{name}_values"
            if key not in frame.attrs:
                frame.attrs[key] = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        frame.attrs["neckline_atr_values"] = np.asarray(self.atr_values, dtype=float)
        date_index = frame.attrs.get("neckline_date_index")
        if date_index is None or len(date_index) != len(frame):
            date_index = pd.DatetimeIndex(frame.date)
            frame.attrs["neckline_date_index"] = date_index
        if "neckline_coverage_tolerances" not in frame.attrs:
            minimum_ticks = int(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2))
            tolerance_atr = float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
            atr_values = frame.attrs["neckline_atr_values"]
            frame.attrs["neckline_coverage_tolerances"] = np.maximum(
                minimum_ticks * self.tick_size,
                np.where(np.isfinite(atr_values), tolerance_atr * atr_values, 0.0))
        return frame

    @staticmethod
    def _candidate_geometry_key(candidate):
        return (candidate["family"], tuple((int(p.index), p.kind)
                                            for p in candidate["sequence"]))

    @staticmethod
    def _pattern_boundary_phase(pattern, consumed_index, reason):
        detail = {item["code"]: item["value"] for item in pattern.evidence}
        selection = detail.get("neckline_selection", {})
        tests = detail.get("neckline_tests", [])
        line = pattern.neckline or selection.get("selected_line")
        if not line:
            return None
        return {"line": dict(line), "tests": tests,
                "consumed_index": int(consumed_index),
                "candidate_id": ((tests[0].get("pair_id") if tests else None)
                                 or selection.get("selected_candidate_id")),
                "reason": reason}

    def _remember_consumed_boundary(self, candidate, phase):
        if phase is None:
            return
        phase.setdefault("phase_tolerance", _neckline_test_tolerance_at(
            phase["consumed_index"], self.atr_values, self.tick_size))
        key = self._candidate_geometry_key(candidate)
        phases = self.consumed_neckline_phases.setdefault(key, [])
        if not any(_same_neckline_boundary_phase(previous, phase, self.tick_size)
                   for previous in phases):
            phases.append(phase)

    def _boundary_is_consumed(self, candidate, phase):
        return any(_same_neckline_boundary_phase(previous, phase, self.tick_size)
                   for previous in self.consumed_neckline_phases.get(
                       self._candidate_geometry_key(candidate), ()))

    def _prepare_boundary_pairs(self):
        """Build reusable neckline pair gates before refreshing shape candidates."""
        if self.neckline_potential_pair_cache is None or len(self.bars) < 2:
            return
        by_location = {}
        for pivots in self.scale_pivots.values():
            for pivot in pivots:
                key = (int(pivot.index), pivot.kind)
                current = by_location.get(key)
                if (current is None or float(pivot.scale.split("ATR")[0])
                        > float(current.scale.split("ATR")[0])):
                    by_location[key] = pivot
        candidate_scale = _neckline_candidate_scale()
        neckline_pivots = list(by_location.values())
        for pressure in self.pressure["pivots"]:
            main = by_location.get((pressure.index, pressure.kind))
            if main is None:
                neckline_pivots.append(pressure)
            else:
                neckline_pivots.append(Pivot(
                    pressure.index, pressure.date, pressure.price, pressure.kind,
                    pressure.confirmed_at, pressure.confirmed_date,
                    main.scale + "|" + candidate_scale))
        highs = np.asarray([float(item["high"]) for item in self.bars], dtype=float)
        minimum_ticks = float(MAIN_RULES.get("neckline_tests", {}).get(
            "minimum_tolerance_ticks", 2))
        tolerance_atr = float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
        tolerances = np.maximum(minimum_ticks * self.tick_size,
                                np.where(np.isfinite(self.atr_values),
                                         tolerance_atr * np.asarray(self.atr_values, dtype=float), 0.0))
        minimum_date = pd.Timestamp(self.bars[-1]["date"]) - pd.DateOffset(years=6)
        minimum_index = bisect_left(self.date_strings, str(minimum_date.date()))
        self.neckline_potential_pair_cache.prune(minimum_index)
        self.neckline_potential_pair_cache.add(
            neckline_pivots, np.asarray(self.atr_values, dtype=float), highs, tolerances,
            self.tick_size, "1d", minimum_index=minimum_index)

    def _report_progress(self, force=False, phase="recognition"):
        callback = self.progress_callback
        if callback is None:
            return
        now = time.monotonic()
        if not force and now - self.last_progress_at < 2.0:
            return
        elapsed = max(0.0, time.perf_counter() - self.progress_started_at)
        callback({
            "phase": phase,
            "bars_completed": len(self.bars),
            "bars_total": self.total_bars,
            "current_date": self.date_strings[-1] if self.date_strings else None,
            "candidate_count": len(self.active),
            "candidate_evaluations": int(self.performance["candidate_evaluations"]),
            "elapsed_seconds": elapsed,
        })
        self.last_progress_at = now

    @staticmethod
    def _new_zigzag():
        return {"direction": 0, "high_index": 0, "low_index": 0}

    @property
    def events(self):
        return tuple(self.active.values())

    def _threshold(self, index, price, atr_multiple, percent_multiple):
        value = self.atr_values[index]
        if not np.isfinite(value) or value <= 0 or not np.isfinite(price) or price <= 0:
            return np.nan
        return max(atr_multiple * value, percent_multiple * abs(price), 2 * self.tick_size)

    def _append_scale_pivots(self, i):
        bar = self.bars[i]
        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        found = []
        for name, atr_multiple, percent_multiple in PIVOT_SCALES:
            state = self.scale_states[name]
            hi_i, lo_i = state["high_index"], state["low_index"]
            if hi > float(self.bars[hi_i]["high"]):
                hi_i = i
            if lo < float(self.bars[lo_i]["low"]):
                lo_i = i
            direction = state["direction"]
            pivot = None
            if direction == 0:
                down = self._threshold(hi_i, float(self.bars[hi_i]["high"]), atr_multiple, percent_multiple)
                up = self._threshold(lo_i, float(self.bars[lo_i]["low"]), atr_multiple, percent_multiple)
                down_move, up_move = float(self.bars[hi_i]["high"]) - close, close - float(self.bars[lo_i]["low"])
                down_ready = np.isfinite(down) and down_move >= down
                up_ready = np.isfinite(up) and up_move >= up
                if down_ready and (not up_ready or down_move / down >= up_move / up):
                    pivot = self._pivot(hi_i, "high", float(self.bars[hi_i]["high"]), i, f"{atr_multiple:g}ATR|{percent_multiple:.0%}")
                    direction, lo_i = -1, i
                elif up_ready:
                    pivot = self._pivot(lo_i, "low", float(self.bars[lo_i]["low"]), i, f"{atr_multiple:g}ATR|{percent_multiple:.0%}")
                    direction, hi_i = 1, i
            elif direction > 0:
                reverse = self._threshold(hi_i, float(self.bars[hi_i]["high"]), atr_multiple, percent_multiple)
                if hi > float(self.bars[hi_i]["high"]):
                    hi_i = i
                    reverse = self._threshold(hi_i, hi, atr_multiple, percent_multiple)
                if np.isfinite(reverse) and float(self.bars[hi_i]["high"]) - close >= reverse:
                    pivot = self._pivot(hi_i, "high", float(self.bars[hi_i]["high"]), i, f"{atr_multiple:g}ATR|{percent_multiple:.0%}")
                    direction, lo_i = -1, i
            else:
                reverse = self._threshold(lo_i, float(self.bars[lo_i]["low"]), atr_multiple, percent_multiple)
                if lo < float(self.bars[lo_i]["low"]):
                    lo_i = i
                    reverse = self._threshold(lo_i, lo, atr_multiple, percent_multiple)
                if np.isfinite(reverse) and close - float(self.bars[lo_i]["low"]) >= reverse:
                    pivot = self._pivot(lo_i, "low", float(self.bars[lo_i]["low"]), i, f"{atr_multiple:g}ATR|{percent_multiple:.0%}")
                    direction, hi_i = 1, i
            state.update(direction=direction, high_index=hi_i, low_index=lo_i)
            if pivot is not None:
                self.scale_pivots[name].append(pivot)
                self._index_low_pivot(pivot)
                found.append((name, pivot))
        self.performance["pivot_updates"] += len(found)
        return found

    def _pivot(self, index, kind, price, known, scale):
        date = str(pd.Timestamp(self.bars[index]["date"]).date())
        known_date = str(pd.Timestamp(self.bars[known]["date"]).date())
        return Pivot(index, date, float(price), kind, known, known_date, scale)

    def _ensure_low_pivot_range_capacity(self, last_index):
        while int(last_index) >= self.low_pivot_range_size:
            old_size = self.low_pivot_range_size
            old_tree = self.low_pivot_range_tree
            new_size = old_size * 2
            new_tree = np.full(2 * new_size, np.inf, dtype=float)
            new_tree[new_size:new_size + old_size] = old_tree[old_size:]
            for node in range(new_size - 1, 0, -1):
                new_tree[node] = min(new_tree[node * 2], new_tree[node * 2 + 1])
            self.low_pivot_range_size = new_size
            self.low_pivot_range_tree = new_tree

    def _index_low_pivot(self, pivot):
        """Maintain an append-updated range-min tree over confirmed low pivots."""
        if pivot.kind != "low":
            return
        index = int(pivot.index)
        self._ensure_low_pivot_range_capacity(index)
        tree = self.low_pivot_range_tree
        position = self.low_pivot_range_size + index
        if float(pivot.price) >= tree[position]:
            return
        tree[position] = float(pivot.price)
        position //= 2
        while position:
            updated = min(tree[position * 2], tree[position * 2 + 1])
            if tree[position] == updated:
                break
            tree[position] = updated
            position //= 2

    def _minimum_low_pivot(self, start, end):
        """Return minimum confirmed low-pivot price in the open index interval."""
        left, right = int(start) + 1, int(end)
        if left >= right:
            return None
        self._ensure_low_pivot_range_capacity(right - 1)
        size = self.low_pivot_range_size
        tree = self.low_pivot_range_tree
        left += size
        right += size
        minimum = np.inf
        while left < right:
            if left & 1:
                minimum = min(minimum, tree[left])
                left += 1
            if right & 1:
                right -= 1
                minimum = min(minimum, tree[right])
            left //= 2
            right //= 2
        return float(minimum) if np.isfinite(minimum) else None

    def _append_pressure_pivot(self, i):
        high = float(self.bars[i]["high"])
        close = float(self.bars[i]["close"])
        state = self.pressure
        candidate = state["candidate"]
        if candidate is None:
            if high >= float(self.bars[i - 1]["high"]):
                state["candidate"] = i
            return None
        if high > float(self.bars[candidate]["high"]):
            state["candidate"] = i
            return None
        reversal_multiple = _neckline_candidate_reversal_atr_multiple()
        reversal_distance = max(reversal_multiple * float(self.atr_values[candidate]), 2 * self.tick_size)
        if (np.isfinite(reversal_distance) and reversal_distance > 0
                and close <= float(self.bars[candidate]["high"]) - reversal_distance):
            pivot = self._pivot(candidate, "high", float(self.bars[candidate]["high"]), i,
                                f"neckline_test_{reversal_multiple:g}ATR")
            state["pivots"].append(pivot)
            state["candidate"] = None
            self.performance["pivot_updates"] += 1
            return pivot
        return None

    def _candidate_maturity_date(self, candidate):
        """Compute a bottom candidate's calendar maturity once and retain it."""
        cached = candidate.get("maturity_date")
        if cached:
            return cached
        if candidate.get("family") not in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}:
            return None
        first_low = min((pivot for pivot in candidate["sequence"] if pivot.kind == "low"),
                        key=lambda pivot: pivot.index)
        months = int(MAIN_RULES.get("minimum_duration_calendar_months_by_timeframe", {}).get("1d", 6))
        maturity = pd.Timestamp(self.bars[first_low.index]["date"]) + pd.DateOffset(months=months)
        candidate["maturity_date"] = str(maturity.date())
        return candidate["maturity_date"]

    def _append_local_pivot(self, i):
        """Right-confirm a radius-two pivot for the existing bull-flag geometry."""
        center = i - 2
        if center < 2:
            return False
        left, right = center - 2, center + 2
        highs = [float(self.bars[j]["high"]) for j in range(left, right + 1)]
        lows = [float(self.bars[j]["low"]) for j in range(left, right + 1)]
        high = float(self.bars[center]["high"])
        low = float(self.bars[center]["low"])
        is_high = high == max(highs) and highs.count(high) == 1
        is_low = low == min(lows) and lows.count(low) == 1
        if is_high == is_low or not np.isfinite(self.atr_values[center]) or self.atr_values[center] <= 0:
            return False
        kind, price = ("high", high) if is_high else ("low", low)
        candidate = (center, kind, price, self.atr_values[center])
        if self.local_raw and kind == self.local_raw[-1][1]:
            better = price > self.local_raw[-1][2] if kind == "high" else price < self.local_raw[-1][2]
            if better:
                old_index = self.local_raw[-1][0]
                self.local_raw[-1] = candidate
                self.local_pivots[-1] = self._pivot(center, kind, price, i, "1.5ATR_radius2")
                for key, sequence in list(self.flag_candidates.items()):
                    if any(p.index == old_index for p in sequence):
                        self.flag_candidates.pop(key, None)
                if len(self.local_pivots) >= 5:
                    sequence = tuple(self.local_pivots[-5:])
                    if [p.kind for p in sequence] == ["low", "high", "low", "high", "low"]:
                        self.flag_candidates[tuple(p.index for p in sequence)] = sequence
                self.performance["pivot_updates"] += 1
                return True
            return False
        if self.local_raw and abs(price - self.local_raw[-1][2]) < max(
                candidate[3], self.local_raw[-1][3]) * 1.5:
            return False
        self.local_raw.append(candidate)
        pivot = self._pivot(center, kind, price, i, "1.5ATR_radius2")
        self.local_pivots.append(pivot)
        if len(self.local_pivots) >= 5:
            sequence = tuple(self.local_pivots[-5:])
            if [p.kind for p in sequence] == ["low", "high", "low", "high", "low"]:
                self.flag_candidates[tuple(p.index for p in sequence)] = sequence
        self.performance["pivot_updates"] += 1
        return True

    def _maintain_candidates(self):
        if not self.bars:
            return
        i = len(self.bars) - 1
        current_date = pd.Timestamp(self.bars[i]["date"])
        cutoff = current_date - pd.DateOffset(years=5)
        retained = {}
        for key, candidate in self.active.items():
            sequence = candidate["sequence"]
            first = min(p.index for p in sequence)
            if pd.Timestamp(self.bars[first]["date"]) < cutoff:
                continue
            known_at = candidate.get("structure_known_at")
            if known_at is None:
                scale_pivots = self.scale_pivots[candidate["scale"]]
                indexes = [p.index for p in sequence]
                lo, hi = min(indexes), max(indexes)
                known_at = max([p.confirmed_at for p in sequence] +
                               [p.confirmed_at for p in scale_pivots if lo <= p.index <= hi])
                candidate["structure_known_at"] = known_at
            if i > known_at:
                low_points = [p for p in sequence if p.kind == "low"]
                if candidate["family"] == "triangle_bottom" and len(low_points) >= 2:
                    boundary = _at(_line((low_points[0].index, low_points[0].price),
                                         (low_points[-1].index, low_points[-1].price)), i)
                else:
                    boundary = min(float(p.price) for p in low_points)
                if float(self.bars[i]["close"]) < boundary:
                    continue
            retained[key] = candidate
        self.active = retained

    def _check_bull_flag(self):
        if len(self.flag_candidates) < 1 or len(self.bars) < 2:
            return []
        i = len(self.bars) - 1
        minimum = 2 * self.tick_size
        prior_atr = self.atr_values[i - 1]
        if not np.isfinite(prior_atr) or prior_atr <= 0:
            return []
        out = []
        cutoff = max(0, i - 500)
        for key, seq in list(self.flag_candidates.items()):
            if seq[-1].index < cutoff:
                self.flag_candidates.pop(key, None)
                continue
            if i <= max(p.confirmed_at for p in seq):
                continue
            lo0, hi0, lo1, hi1, lo2 = [float(p.price) for p in seq]
            i0, i1, i2, i3, i4 = [p.index for p in seq]
            if any(float(self.bars[j]["close"]) < lo0 for j in range(seq[-1].confirmed_at, i + 1)):
                self.flag_candidates.pop(key, None)
                continue
            impulse, correction = hi0 - lo0, hi0 - lo2
            if (i1 - i0 < 4 or i4 - i1 < 5 or impulse <= 0 or
                    correction < impulse * 0.12 or correction > impulse * 0.50 or
                    lo1 <= lo0 or lo2 <= lo0 or hi1 >= hi0 or hi1 <= lo2):
                continue
            upper = _line((i1, hi0), (i3, hi1))
            if upper["slope"] >= 0:
                continue
            formation_end = max(p.confirmed_at for p in seq)
            formation = self.bars[i1:formation_end + 1]
            coverage = _neckline_formation_coverage(
                np.asarray([b["high"] for b in formation]),
                np.asarray(self.atr_values[i1:formation_end + 1]),
                {"slope": upper["slope"], "intercept": _at(upper, i1)},
                0, len(formation)-1, self.tick_size)
            if not coverage["passed"]:
                continue
            section = self.bars[i1:i4 + 1]
            if max(float(b["high"]) for b in section) - min(float(b["low"]) for b in section) > impulse * 0.60:
                continue
            required = _at(upper, i) + max(minimum, prior_atr)
            previous_required = _at(upper, i - 1) + max(
                minimum, self.atr_values[i - 2] if i > 1 else minimum)
            if float(self.bars[i - 1]["close"]) > previous_required or float(self.bars[i]["close"]) <= required:
                continue
            pattern_id = f"{self.code}:bull_flag:{'-'.join(f'{p.index}:{p.kind}' for p in seq)}"
            if pattern_id in self.emitted_ids:
                continue
            self.emitted_ids.add(pattern_id)
            date = str(pd.Timestamp(self.bars[i]["date"]).date())
            pattern = Pattern(pattern_id=pattern_id, code=self.code, kind=self.kind, timeframe="1d",
                              pattern_type="bull_flag", direction="bullish", state="confirmed",
                              formed_at=seq[0].date, confirmed_at=i,
                              key_points={f"pivot_{n}_index": p.index for n, p in enumerate(seq, 1)},
                              neckline=dict(upper), invalidation=lo0,
                              evidence=[{"code": "flag_geometry", "value": {"impulse": impulse,
                                          "correction_ratio": correction / impulse,
                                          "known_at": seq[-1].confirmed_date}},
                                        {"code": "flag_breakout", "value": {"date": date,
                                          "close": float(self.bars[i]["close"]),
                                          "frozen_boundary": float(_at(upper, i)),
                                          "required_close": float(required),
                                          "atr14_previous": float(prior_atr)}}],
                              confirmation_date=date, last_observed_date=date,
                              rule_version=ENGINE_VERSION, source_rule_version=SOURCE_RULE_VERSION)
            out.append(self._signal(pattern, date))
        return out

    def _refresh_structures(self, changed_scales=None):
        """Refresh candidate sequences only when a confirmed large-scale pivot changes."""
        frame = self._frame_for_current_prefix()
        lows = frame.attrs["neckline_low_values"]
        pressure = self.pressure["pivots"]
        cutoff = pd.Timestamp(frame.date.iloc[-1]) - pd.DateOffset(years=5)
        start_index = int(np.searchsorted(frame.attrs["neckline_date_index"].values,
                                          cutoff.to_datetime64(), side="left"))
        self.active = {key: value for key, value in self.active.items()
                       if min(p.index for p in value["sequence"]) >= start_index}
        for scale, pivots in self.scale_pivots.items():
            if changed_scales is not None and scale not in changed_scales:
                continue
            recent = [p for p in pivots if p.index >= start_index]
            if not recent:
                continue
            for family, seq in _structure_candidates(recent, "1d", self.atr_values, self.tick_size,
                                                      low_values=lows, shoulder_pivots=pressure,
                                                      fast_candidate_enumeration=self.boundary_first):
                key = (family, scale, tuple((p.index, p.kind) for p in seq))
                if key not in self.active:
                    candidate = {"family": family, "scale": scale,
                                 "sequence": seq, "pattern": None,
                                 "needs_initial_eval": True}
                    self._candidate_maturity_date(candidate)
                    self.active[key] = candidate
            self.performance["candidate_refreshes"] += 1

    def _has_pressure_test_pair(self, candidate):
        if not self.pressure["pivots"]:
            return True
        sequence = candidate["sequence"]
        family = candidate["family"]
        low_points = sorted((p for p in sequence if p.kind == "low"), key=lambda p: p.index)
        test_rules = MAIN_RULES.get("neckline_tests", {})
        minimum_gap = int(test_rules.get("minimum_separation_daily_sessions", 10))
        required_groups = _minimum_neckline_test_count()
        minimum_ticks = float(test_rules.get("minimum_tolerance_ticks", 2)) * self.tick_size
        tolerance_atr = float(test_rules.get("tolerance_atr", 0.5))
        pressure = sorted(self.pressure["pivots"], key=lambda p: p.index)
        minimum_departure = float(test_rules.get("minimum_departure_atr", 1.0))

        def compatible_pair(left_points, right_points=None, fixed_line=None):
            same_group = right_points is None
            right_points = left_points if same_group else right_points
            for left_position, left in enumerate(left_points):
                candidates = right_points[left_position + 1:] if same_group else right_points
                for right in candidates:
                    if right.index - left.index < minimum_gap or float(left.price) <= 0:
                        continue
                    line = fixed_line or _qualify_neckline_slope(
                        _line((left.index, left.price), (right.index, right.price)),
                        left.index, right.index)
                    if line is None:
                        continue
                    between_low = self._minimum_low_pivot(left.index, right.index)
                    if between_low is None:
                        continue
                    departure = min(_at(line, left.index), _at(line, right.index)) - between_low
                    left_atr = (self.atr_values[left.confirmed_at]
                                if left.confirmed_at < len(self.atr_values) else np.nan)
                    right_atr = (self.atr_values[right.confirmed_at]
                                 if right.confirmed_at < len(self.atr_values) else np.nan)
                    reference_atr = max(float(left_atr) if np.isfinite(left_atr) else 0.0,
                                        float(right_atr) if np.isfinite(right_atr) else 0.0)
                    if (reference_atr > 0
                            and departure >= minimum_departure * reference_atr):
                        return True
            return False

        if family == "w_bottom":
            if len(low_points) < 2:
                return False
            first, last = low_points[0], low_points[-1]
            middle_highs = [p for points in self.scale_pivots.values() for p in points
                            if p.kind == "high" and first.index < p.index < last.index]
            middle_highs.extend(p for p in pressure if first.index < p.index < last.index)
            if not middle_highs:
                return False
            middle = max(middle_highs, key=lambda p: p.price)
            middle_tolerance = max(minimum_ticks, tolerance_atr * float(
                self.atr_values[middle.confirmed_at])
                if middle.confirmed_at < len(self.atr_values)
                and np.isfinite(self.atr_values[middle.confirmed_at]) else 0.0)
            minimum_level = float(middle.price) - middle_tolerance
            maximum_level = float(middle.price) * (1.0 + float(test_rules.get(
                "w_bottom_rim_association_ceiling_relative", 0.2)))
            year_before = pd.Timestamp(first.date) - pd.DateOffset(
                years=int(test_rules.get("lookback_calendar_years", 1)))
            eligible = [p for p in pressure
                        if pd.Timestamp(p.date) >= year_before and float(p.price) <= maximum_level]
            eligible = [p for p in eligible if float(p.price) >= minimum_level]
            after = [p for p in eligible if p.index > first.index]
            prior = [p for p in eligible if p.index <= first.index]
            after_pair = (len(after) >= required_groups and compatible_pair(after))
            prior_and_after = (len(prior) + len(after) >= required_groups
                               and bool(prior) and bool(after)
                               and compatible_pair(prior, after))
            return after_pair or prior_and_after

        if family == "inverse_head_shoulders":
            if len(low_points) < 3:
                return False
            left, head, right = low_points[0], low_points[1], low_points[-1]
            ihs_rules = MAIN_RULES.get("inverse_head_shoulders_neckline", {})
            left_tests = [p for p in pressure if left.index < p.index < head.index]
            right_tests = [p for p in pressure if head.index < p.index < right.index]
            require_both = bool(ihs_rules.get("require_independent_test_on_each_reaction_leg", True))
            if require_both:
                return (len(left_tests) + len(right_tests) >= required_groups
                            and bool(left_tests and right_tests)
                            and compatible_pair(left_tests, right_tests))
            after = [p for p in pressure if p.index > left.index]
            return len(after) >= required_groups and compatible_pair(after)

        if family == "triangle_bottom":
            high_points = sorted((p for p in sequence if p.kind == "high"), key=lambda p: p.index)
            if len(low_points) < 2 or len(high_points) < 2:
                return False
            upper = _qualify_neckline_slope(
                _line((high_points[0].index, high_points[0].price),
                      (high_points[-1].index, high_points[-1].price)),
                high_points[0].index, high_points[-1].index)
            if upper is None:
                return False
            lower_index, upper_index = low_points[0].index, low_points[-1].index
            eligible = []
            for pivot in pressure:
                if not lower_index <= pivot.index <= upper_index:
                    continue
                known_atr = (self.atr_values[pivot.confirmed_at]
                             if pivot.confirmed_at < len(self.atr_values) else np.nan)
                tolerance = max(minimum_ticks, tolerance_atr * float(known_atr)
                                if np.isfinite(known_atr) else 0.0)
                if abs(float(pivot.price) - _at(upper, pivot.index)) <= max(tolerance, 2 * self.tick_size):
                    eligible.append(pivot)
            return (len(eligible) >= required_groups
                    and eligible[-1].index - eligible[0].index >= minimum_gap
                    and compatible_pair(eligible, fixed_line=upper))
        return len(pressure) >= required_groups

    def _pressure_pivot_can_change_candidate(self, candidate, pressure_pivot):
        """Ignore newly confirmed touches outside this shape's search interval."""
        sequence = candidate["sequence"]
        first = min((p for p in sequence if p.kind == "low"), key=lambda p: p.index)
        index = int(pressure_pivot.index)
        family = candidate["family"]
        cached = candidate.get("pattern")
        low_points = sorted((p for p in sequence if p.kind == "low"), key=lambda p: p.index)
        test_rules = MAIN_RULES.get("neckline_tests", {})
        minimum_gap = int(test_rules.get("minimum_separation_daily_sessions", 10))
        minimum_ticks = float(test_rules.get("minimum_tolerance_ticks", 2)) * self.tick_size
        tolerance_atr = float(test_rules.get("tolerance_atr", 0.5))
        has_independent_gap_partner = any(
            pivot.index != index and abs(pivot.index - index) >= minimum_gap
            for pivot in self.pressure["pivots"])
        if (cached is not None and cached.state == "forming"
                and family in {"w_bottom", "inverse_head_shoulders"}):
            w_interior = (family == "w_bottom" and len(low_points) >= 2
                          and low_points[0].index < index < low_points[-1].index)
            if not w_interior and not has_independent_gap_partner and cached.neckline:
                known_atr = (self.atr_values[pressure_pivot.confirmed_at]
                             if pressure_pivot.confirmed_at < len(self.atr_values) else np.nan)
                touch_tolerance = max(
                    float(test_rules.get("minimum_tolerance_ticks", 2)) * self.tick_size,
                    float(test_rules.get("tolerance_atr", 0.5)) * float(known_atr)
                    if np.isfinite(known_atr) else 0.0)
                if float(pressure_pivot.price) <= _at(cached.neckline, index) + touch_tolerance:
                    return False
        if family == "triangle_bottom":
            high_points = sorted((p for p in sequence if p.kind == "high"), key=lambda p: p.index)
            if (len(low_points) < 2 or len(high_points) < 2
                    or not low_points[0].index <= index <= low_points[-1].index):
                return False
            upper = _qualify_neckline_slope(
                _line((high_points[0].index, high_points[0].price),
                      (high_points[-1].index, high_points[-1].price)),
                high_points[0].index, high_points[-1].index)
            if upper is None:
                return False
            known_atr = (self.atr_values[pressure_pivot.confirmed_at]
                         if pressure_pivot.confirmed_at < len(self.atr_values) else np.nan)
            tolerance = max(minimum_ticks, tolerance_atr * float(known_atr)
                            if np.isfinite(known_atr) else 0.0)
            return abs(float(pressure_pivot.price) - _at(upper, index)) <= tolerance
        if family == "inverse_head_shoulders":
            if index < first.index:
                return False
        if family == "w_bottom":
            lookback_years = int(test_rules.get("lookback_calendar_years", 1))
            earliest = pd.Timestamp(first.date) - pd.DateOffset(years=lookback_years)
            if pd.Timestamp(pressure_pivot.date) < earliest:
                return False
            low_points = sorted((p for p in sequence if p.kind == "low"), key=lambda p: p.index)
            middle_highs = [p for pivots in self.scale_pivots.values() for p in pivots
                            if p.kind == "high" and low_points[0].index < p.index < low_points[-1].index]
            middle_highs.extend(p for p in self.pressure["pivots"]
                                if low_points[0].index < p.index < low_points[-1].index)
            if not middle_highs:
                # The new interior touch may be the first known W rim.
                return True
            middle = max(middle_highs, key=lambda p: p.price)
            known_atr = (self.atr_values[middle.confirmed_at]
                         if middle.confirmed_at < len(self.atr_values) else np.nan)
            middle_tolerance = max(minimum_ticks,
                                   tolerance_atr * float(known_atr)
                                   if np.isfinite(known_atr) else 0.0)
            minimum_level = float(middle.price) - middle_tolerance
            maximum_level = float(middle.price) * (
                1.0 + float(test_rules.get(
                    "w_bottom_rim_association_ceiling_relative", 0.2)))
            price = float(pressure_pivot.price)
            if not minimum_level <= price <= maximum_level:
                return False
            if (cached is not None and cached.state == "forming" and cached.neckline
                    and has_independent_gap_partner):
                known_atr = (self.atr_values[pressure_pivot.confirmed_at]
                             if pressure_pivot.confirmed_at < len(self.atr_values) else np.nan)
                touch_tolerance = max(minimum_ticks,
                                      tolerance_atr * float(known_atr)
                                      if np.isfinite(known_atr) else 0.0)
                if abs(price - _at(cached.neckline, index)) <= touch_tolerance:
                    return True
            max_rise = float(MAIN_RULES.get("neckline_slope", {}).get(
                "maximum_cumulative_relative_rise", 0.05))
            for anchor in self.pressure["pivots"]:
                if anchor.index == index or abs(anchor.index - index) < minimum_gap:
                    continue
                if (pd.Timestamp(anchor.date) < earliest
                        or not minimum_level <= float(anchor.price) <= maximum_level):
                    continue
                left, right = ((anchor, pressure_pivot) if anchor.index < index
                               else (pressure_pivot, anchor))
                relative_rise = (float(right.price) - float(left.price)) / float(left.price)
                if relative_rise <= max_rise and max(anchor.index, index) > low_points[0].index:
                    return True
            return False
        if family == "inverse_head_shoulders":
            possible_anchors = [p for p in self.pressure["pivots"]
                                if p.index >= first.index and p.index != index]
            if (cached is not None and cached.state == "forming" and cached.neckline
                    and has_independent_gap_partner):
                known_atr = (self.atr_values[pressure_pivot.confirmed_at]
                             if pressure_pivot.confirmed_at < len(self.atr_values) else np.nan)
                touch_tolerance = max(minimum_ticks,
                                      tolerance_atr * float(known_atr)
                                      if np.isfinite(known_atr) else 0.0)
                if abs(float(pressure_pivot.price) - _at(cached.neckline, index)) <= touch_tolerance:
                    return True
            max_rise = float(MAIN_RULES.get("neckline_slope", {}).get(
                "maximum_cumulative_relative_rise", 0.05))
            for anchor in possible_anchors:
                if abs(anchor.index - index) < minimum_gap:
                    continue
                left, right = ((anchor, pressure_pivot) if anchor.index < index
                               else (pressure_pivot, anchor))
                relative_rise = (float(right.price) - float(left.price)) / float(left.price)
                if relative_rise <= max_rise:
                    return True
            if not possible_anchors:
                return cached is None or float(pressure_pivot.price) > (
                    _at(cached.neckline, index) + max(minimum_ticks,
                        tolerance_atr * float(self.atr_values[pressure_pivot.confirmed_at])
                        if pressure_pivot.confirmed_at < len(self.atr_values)
                        and np.isfinite(self.atr_values[pressure_pivot.confirmed_at]) else 0.0))
            # No prior pressure pivot can form an allowed line through this
            # new high. Slightly rising lines are flattened to horizontal, so
            # this point also cannot add a touch to an older qualified line.
            return False
        return True

    def _evaluate(self, pressure_pivot=None):
        if not self.active:
            return []
        frame = self._prepare_neckline_frame(self._frame_for_current_prefix())
        high_values = frame.attrs["neckline_high_values"]
        low_values = frame.attrs["neckline_low_values"]
        close_values = frame.attrs["neckline_close_values"]
        atr_values = np.asarray(self.atr_values, dtype=float)
        frame.attrs["neckline_atr_values"] = atr_values
        current_date = str(pd.Timestamp(frame.date.iloc[-1]).date())
        event_candidates = []
        all_neckline_pivots = []
        by_identity = {}
        for pivots in self.scale_pivots.values():
            for pivot in pivots:
                key = (pivot.index, pivot.kind)
                if key not in by_identity or float(pivot.scale.split("ATR")[0]) > float(by_identity[key].scale.split("ATR")[0]):
                    by_identity[key] = pivot
        all_neckline_pivots.extend(by_identity.values())
        cutoff = pd.Timestamp(frame.date.iloc[-1]) - pd.DateOffset(years=6)
        for pivot in self.pressure["pivots"]:
            if pivot.index < int(np.searchsorted(frame.attrs["neckline_date_index"].values,
                                                cutoff.to_datetime64(), side="left")):
                continue
            main = by_identity.get((pivot.index, pivot.kind))
            if main is None:
                all_neckline_pivots.append(pivot)
            else:
                all_neckline_pivots.append(Pivot(pivot.index, pivot.date, pivot.price, pivot.kind,
                                                 pivot.confirmed_at, pivot.confirmed_date,
                                                 main.scale + "|" + _neckline_candidate_scale()))
        all_neckline_pivots.sort(key=lambda p: (p.index, p.scale))
        scale_rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}

        def geometry_key(candidate):
            return (candidate["family"], tuple(
                (p.index, p.kind, float(p.price), int(p.confirmed_at), p.confirmed_date)
                for p in candidate["sequence"]))

        dominant_candidate = {}
        for candidate_key, candidate in self.active.items():
            geometry = geometry_key(candidate)
            current = dominant_candidate.get(geometry)
            if (current is None or scale_rank.get(candidate["scale"], -1)
                    > scale_rank.get(self.active[current]["scale"], -1)):
                dominant_candidate[geometry] = candidate_key
        evaluated_geometries = set()
        for key, candidate in self.active.items():
            family, seq, scale = candidate["family"], candidate["sequence"], candidate["scale"]
            geometry = geometry_key(candidate)
            if key != dominant_candidate.get(geometry):
                dominant = self.active[dominant_candidate[geometry]].get("pattern")
                cached = candidate.get("pattern")
                if (dominant is not None and dominant.confirmation_date and cached is not None
                        and not cached.confirmation_date):
                    candidate["pattern"] = None
                continue
            if family in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}:
                if current_date < (self._candidate_maturity_date(candidate) or "9999-12-31"):
                    continue
            cached = candidate.get("pattern")
            if cached is not None and cached.confirmation_date:
                continue
            if pressure_pivot is None and not candidate.get("needs_initial_eval"):
                continue
            if (pressure_pivot is not None
                    and not self._pressure_pivot_can_change_candidate(candidate, pressure_pivot)):
                continue
            if not self._has_pressure_test_pair(candidate):
                continue
            evaluated_geometries.add(geometry)
            self.performance["candidate_evaluations"] += 1
            patterns = _make_main_pattern(frame, self.code, self.kind, "1d", family, seq,
                                          all_neckline_pivots, atr_values, self.tick_size,
                                          scale, "5y", background_bars=frame,
                                          collect_diagnostics=False,
                                          consumed_boundary_phases=self.consumed_neckline_phases.get(
                                              self._candidate_geometry_key(candidate), ()))
            patterns = patterns if isinstance(patterns, list) else [patterns]
            candidate["pattern"] = next((p for p in patterns if p is not None and p.state != "invalidated"),
                                         next((p for p in patterns if p is not None), None))
            for pattern in patterns:
                if pattern is None:
                    continue
                detail = {item["code"]: item["value"] for item in pattern.evidence}
                selection = detail.get("neckline_selection", {})
                for early in selection.get("early_breakouts", []):
                    phase = {"line": early.get("line", pattern.neckline),
                             "tests": early.get("tests", detail.get("neckline_tests", [])),
                             "consumed_index": int(early["index"]),
                             "candidate_id": early.get("candidate_id"),
                             "reason": "breakout_before_minimum_structure_duration"}
                    self._remember_consumed_boundary(candidate, phase)
                if pattern.confirmation_date and pattern.confirmed_at is not None:
                    self._remember_consumed_boundary(
                        candidate, self._pattern_boundary_phase(
                            pattern, pattern.confirmed_at, "buffered_breakout"))
            candidate["needs_initial_eval"] = False
            self._report_progress()
            for pattern in patterns:
                if pattern and pattern.confirmation_date == current_date:
                    detail = {item["code"]: item["value"] for item in pattern.evidence}
                    main = detail.get("main_structure", {})
                    if main.get("signal_eligible"):
                        event_candidates.append(pattern)
        if evaluated_geometries:
            for key, candidate in self.active.items():
                geometry = geometry_key(candidate)
                if geometry in evaluated_geometries and key != dominant_candidate.get(geometry):
                    cached = candidate.get("pattern")
                    if cached is not None and not cached.confirmation_date:
                        candidate["pattern"] = None
        if not event_candidates:
            return []
        pool = [item.get("pattern") for item in self.active.values() if item.get("pattern") is not None]
        eligible = self._independent_candidates(pool, event_candidates, current_date)
        if not eligible:
            return []
        eligible = [p for p in eligible if p.pattern_id not in self.emitted_ids]
        if not eligible:
            return []
        selected = max(eligible, key=self._main_rank)
        eligible.sort(key=lambda p: p.pattern_id != selected.pattern_id)
        snapshots = [self._selected_snapshot(pattern, current_date,
                                             selected=pattern.pattern_id == selected.pattern_id,
                                             primary_id=selected.pattern_id)
                     for pattern in eligible]
        self.emitted_ids.update(pattern.pattern_id for pattern in eligible)
        return [self._signal_many(snapshots, current_date)]

    def _check_cached_breakout(self):
        """Check a frozen, already-qualified boundary on bars without new pivots."""
        if len(self.bars) < 2:
            return []
        i = len(self.bars) - 1
        current_date = str(pd.Timestamp(self.bars[i]["date"]).date())
        candidates = []
        rules = MAIN_RULES.get("breakout", {})
        buffer_multiple = float(rules.get("close_buffer_previous_atr_multiple", 1.0))
        minimum_ticks = float(rules.get("minimum_buffer_ticks", 2)) * self.tick_size
        for candidate in self.active.values():
            pattern = candidate.get("pattern")
            if pattern is None or pattern.confirmation_date:
                continue
            detail = {item["code"]: item["value"] for item in pattern.evidence}
            structure = detail.get("main_structure", {})
            selection = detail.get("neckline_selection", {})
            if (structure.get("role") == "internal" or structure.get("eligibility_reason")
                    not in {"price_breakout_confirmed", "awaiting_fresh_neckline_breakout"}):
                continue
            frozen_date = selection.get("line_frozen_at")
            structure_known = selection.get("structure_known_at")
            if (not frozen_date or not structure_known or current_date <= frozen_date
                    or current_date <= structure_known):
                continue
            line = pattern.neckline or structure.get("neckline")
            if not line:
                continue
            phase = {"line": dict(line), "tests": detail.get("neckline_tests", [])}
            if self._boundary_is_consumed(candidate, phase):
                continue
            pressure_date = selection.get("effective_start_date", pattern.formed_at)
            pressure_start = self.date_strings.index(pressure_date)
            decline = _neckline_decline_check(pattern.pattern_type, line, pressure_start, i)
            if not decline["passed"]:
                continue
            formation_start = min(int(pattern.key_points.get("first_trough_index", 0)), pressure_start)
            formation_end = max(self.date_strings.index(frozen_date),
                                self.date_strings.index(structure_known))
            guard_key = (pattern.pattern_id, line["slope"], line["intercept"],
                         formation_start, formation_end)
            if candidate.get("formation_guard_key") != guard_key:
                candidate["formation_guard"] = _neckline_formation_coverage(
                    np.asarray([b["high"] for b in self.bars]), np.asarray(self.atr_values),
                    line, formation_start, formation_end, self.tick_size,
                    cache=self.neckline_coverage_cache)
                candidate["formation_guard_key"] = guard_key
            if not candidate["formation_guard"]["passed"]:
                continue
            if float(self.bars[i]["close"]) < float(structure.get("support_lower", -np.inf)):
                continue
            previous_atr = self.atr_values[i - 1]
            if not np.isfinite(previous_atr) or previous_atr <= 0:
                continue
            previous_level = _at(line, i - 1)
            previous_required = previous_level + max(minimum_ticks, buffer_multiple * self.atr_values[i - 2]
                                                     if i > 1 else minimum_ticks)
            if float(self.bars[i - 1]["close"]) > previous_required:
                continue
            required = _at(line, i) + max(minimum_ticks, buffer_multiple * previous_atr)
            if float(self.bars[i]["close"]) > required:
                targets = next((item["value"] for item in pattern.evidence if item["code"] == "targets"), {})
                apex = targets.get("triangle_apex_index") if isinstance(targets, dict) else None
                if pattern.pattern_type == "triangle_bottom" and apex is not None:
                    trough_index = int(pattern.key_points["first_trough_index"])
                    position = (i - trough_index) / max(float(apex) - trough_index, 1.0)
                    if not 0.5 <= position <= 0.75:
                        continue
                if current_date < (self._candidate_maturity_date(candidate) or "9999-12-31"):
                    early = {"date": current_date, "index": int(i),
                             "close": float(self.bars[i]["close"]),
                             "frozen_neckline": float(_at(line, i)),
                             "required_close": float(required),
                             "atr14_previous": float(previous_atr),
                             "tolerance": float(required - _at(line, i)),
                             "candidate_id": selection.get("selected_candidate_id"),
                             "line": dict(line), "tests": detail.get("neckline_tests", [])}
                    consumed = dict(phase, consumed_index=int(i),
                                    phase_tolerance=_neckline_test_tolerance_at(
                                        i, self.atr_values, self.tick_size),
                                    candidate_id=early["candidate_id"],
                                    reason="breakout_before_minimum_structure_duration")
                    self._remember_consumed_boundary(candidate, consumed)
                    existing = selection.setdefault("early_breakouts", [])
                    if not any(item.get("date") == current_date
                               and item.get("candidate_id") == early["candidate_id"]
                               for item in existing):
                        existing.append(early)
                    structure["early_breakout_consumed"] = True
                    continue
                snapshot = deepcopy(pattern)
                snapshot.state = "confirmed"
                snapshot.confirmation_date = current_date
                snapshot.confirmed_at = i
                snapshot.last_observed_date = current_date
                for item in snapshot.evidence:
                    if item["code"] == "neckline_selection":
                        item["value"].update({"formation_coverage": dict(candidate["formation_guard"]),
                                               "decline_check": decline})
                for item in snapshot.evidence:
                    if item["code"] == "main_structure":
                        item["value"].update({"signal_eligible": True,
                                              "eligibility_reason": "price_breakout_confirmed",
                                              "observed_date": current_date})
                    elif item["code"] == "neckline_breakout":
                        item["value"] = {"date": current_date, "index": i,
                                         "close": float(self.bars[i]["close"]),
                                         "frozen_neckline": float(_at(line, i)),
                                         "required_close": float(required),
                                         "atr14_previous": float(previous_atr),
                                         "tolerance": float(required - _at(line, i))}
                self._remember_consumed_boundary(
                    candidate, self._pattern_boundary_phase(snapshot, i, "buffered_breakout"))
                candidates.append(snapshot)
        if not candidates:
            return []
        pool = [item.get("pattern") for item in self.active.values() if item.get("pattern") is not None]
        eligible = self._independent_candidates(pool, candidates, current_date)
        if not eligible:
            return []
        eligible = [p for p in eligible if p.pattern_id not in self.emitted_ids]
        if not eligible:
            return []
        selected = max(eligible, key=self._main_rank)
        eligible.sort(key=lambda p: p.pattern_id != selected.pattern_id)
        snapshots = [self._selected_snapshot(pattern, current_date,
                                             selected=pattern.pattern_id == selected.pattern_id,
                                             primary_id=selected.pattern_id)
                     for pattern in eligible]
        self.emitted_ids.update(pattern.pattern_id for pattern in eligible)
        return [self._signal_many(snapshots, current_date)]

    @staticmethod
    def _selected_snapshot(pattern, decided_at, selected=True, primary_id=None):
        snapshot = deepcopy(pattern)
        snapshot.evidence.append({"code": "main_selection", "value": {
            "selected": bool(selected), "decided_at": decided_at,
            "primary_pattern_id": primary_id or pattern.pattern_id,
            "reason": "latest_confirmation_then_pattern_family_then_coarsest_ihs_scale_then_structure_duration_then_pressure_evidence"}})
        return snapshot

    def _remember(self, signals):
        self.emitted_events.extend(deepcopy(signals))
        return signals

    def _signal(self, pattern, signal_date):
        return self._signal_many([pattern], signal_date)

    def _signal_many(self, patterns, signal_date):
        labels = [PATTERN_LABELS.get(p.pattern_type, p.pattern_type) for p in patterns]
        tags = []
        for pattern in patterns:
            detail = {item["code"]: item["value"] for item in pattern.evidence}
            tag = {"pattern_id": pattern.pattern_id, "pattern_type": pattern.pattern_type}
            for key in ("neckline_selection", "main_structure", "execution_suppression"):
                if key in detail:
                    tag[key] = deepcopy(detail[key])
            if pattern.flags:
                tag["flags"] = list(pattern.flags)
            tags.append(tag)
        return {"code": self.code, "kind": self.kind, "signal_date": signal_date,
                "entry_date": None, "pattern_ids": [p.pattern_id for p in patterns],
                "pattern_types": [p.pattern_type for p in patterns],
                "pattern_labels": labels, "pattern_tags": tags,
                "labels": labels, "tags": tags, "pattern": patterns[0].to_dict(),
                "patterns": [p.to_dict() for p in patterns],
                "engine_version": ENGINE_VERSION, "source_rule_version": SOURCE_RULE_VERSION}

    @staticmethod
    def _main_rank(pattern):
        family_order = {"w_bottom": 2, "inverse_head_shoulders": 1, "triangle_bottom": 0}
        detail = {item["code"]: item["value"] for item in pattern.evidence}
        tests = detail.get("neckline_tests", [])
        span = max((int(item.get("test_span_bars", 0)) for item in tests), default=0)
        rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
        scale = rank.get(pattern.key_points.get("pivot_scale"), -1)
        right_date = pattern.key_points.get("pivot_3_date", "") if pattern.pattern_type == "inverse_head_shoulders" else ""
        left_date = pattern.key_points.get("pivot_1_date", "") if pattern.pattern_type == "inverse_head_shoulders" else ""
        duration = (pd.Timestamp(pattern.last_observed_date) - pd.Timestamp(pattern.formed_at)).days
        return (pattern.confirmation_date or "", family_order.get(pattern.pattern_type, -1),
                scale if pattern.pattern_type == "inverse_head_shoulders" else -1,
                right_date, left_date, duration if pattern.pattern_type == "inverse_head_shoulders" else 0,
                len(tests), span, pattern.formed_at, scale, pattern.pattern_id)

    @staticmethod
    def _nested_ids(patterns, action_date):
        rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
        main_families = {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
        as_of = pd.Timestamp(action_date)
        nested = set()
        active = [p for p in patterns if p.state in {"forming", "confirmed", "target_reached"}]
        for child in active:
            child_points = child.key_points
            child_dates = [pd.Timestamp(value) for key, value in child_points.items()
                           if key.startswith("pivot_") and key.endswith("_date") and value]
            if not child_dates:
                continue
            child_detail = {item["code"]: item["value"] for item in child.evidence}
            child_structure = child_detail.get("main_structure", {})
            child_start = pd.Timestamp(child_structure.get("first_trough_date") or min(child_dates))
            child_end = pd.Timestamp(child_structure.get("structure_end_date") or max(child_dates))
            child_end = min(child_end, as_of)
            for parent in active:
                if (parent.pattern_type not in main_families
                        or parent.pattern_id == child.pattern_id):
                    continue
                points = parent.key_points
                if parent.pattern_type == "inverse_head_shoulders":
                    same_head_right = (
                        points.get("pivot_2_date") == child_points.get("pivot_2_date")
                        and points.get("pivot_3_date") == child_points.get("pivot_3_date"))
                    if same_head_right:
                        continue
                parent_detail = {item["code"]: item["value"] for item in parent.evidence}
                parent_structure = parent_detail.get("main_structure", {})
                parent_pivot_dates = [pd.Timestamp(value) for key, value in points.items()
                                      if key.startswith("pivot_") and key.endswith("_date") and value]
                parent_start = pd.Timestamp(parent_structure.get("first_trough_date") or
                    (min(parent_pivot_dates) if parent_pivot_dates else parent.formed_at))
                parent_end = pd.Timestamp(parent_structure.get("structure_end_date") or
                    (max(parent_pivot_dates) if parent_pivot_dates
                     else parent.last_observed_date or parent_start))
                parent_end = min(parent_end, as_of)
                strictly_inside = parent_start <= child_start and child_end <= parent_end and (
                    parent_start < child_start or child_end < parent_end)
                selection = parent_detail.get("neckline_selection", {})
                known_at = pd.Timestamp(selection.get("structure_known_at", parent_start))
                if (strictly_inside
                        and rank.get(parent_structure.get("pivot_scale", points.get("pivot_scale")), -1)
                        >= rank.get(child_points.get("pivot_scale"), -1)
                        and known_at <= as_of):
                    nested.add(child.pattern_id)
                    break
        return nested

    def _independent_candidates(self, pool, candidates, action_date):
        nested = self._nested_ids(pool + candidates, action_date)
        unique = {p.pattern_id: p for p in candidates if p.pattern_id not in nested
                  and p.pattern_id not in self.emitted_ids}
        by_type = {}
        for pattern in unique.values():
            current = by_type.get(pattern.pattern_type)
            if current is None or self._main_rank(pattern) > self._main_rank(current):
                by_type[pattern.pattern_type] = pattern
        return sorted(by_type.values(), key=self._main_rank, reverse=True)

    def append_bar(self, bar: dict) -> List[dict]:
        row = dict(bar)
        row["date"] = pd.Timestamp(row["date"])
        for name in ("open", "high", "low", "close"):
            row[name] = float(row[name])
        if self.bars and row["date"] <= self.bars[-1]["date"]:
            raise ValueError("daily bars must be appended in strictly increasing date order")
        self.bars.append(row)
        self._prefix_frame_cache = None
        self._prefix_frame_cache_length = -1
        self.date_strings.append(str(row["date"].date()))
        self.performance["bars_appended"] += 1
        i = len(self.bars) - 1
        previous_close = float(self.bars[i - 1]["close"]) if i else row["close"]
        true_range = max(row["high"] - row["low"], abs(row["high"] - previous_close),
                         abs(row["low"] - previous_close))
        self.atr_values.append(true_range if i == 0 else self.atr_values[-1] +
                               (true_range - self.atr_values[-1]) / 14.0)
        self.performance["atr_updates"] += 1
        refreshed = self._append_scale_pivots(i) if i else []
        pressure_pivot = self._append_pressure_pivot(i) if i else None
        if pressure_pivot is not None:
            self._pressure_pivot_revision += 1
        changed_pressure_variant = False
        if self.boundary_first and refreshed:
            pressure_locations = {(item.index, item.kind) for item in self.pressure["pivots"]}
            changed_pressure_variant = any((pivot.index, pivot.kind) in pressure_locations
                                           for _, pivot in refreshed)
        if self.boundary_first and (pressure_pivot is not None or changed_pressure_variant):
            self._prepare_boundary_pairs()
        local_added = self._append_local_pivot(i) if i else False
        if refreshed:
            changed_scales = ({name for name, _ in refreshed}
                              if (self.boundary_first and pressure_pivot is None
                                  and self._pressure_pivot_revision == self._topology_pressure_revision)
                              else None)
            if changed_scales is None:
                self._refresh_structures()
            else:
                self._refresh_structures(changed_scales)
            self._topology_pressure_revision = self._pressure_pivot_revision
        self._maintain_candidates()
        current_date = self.date_strings[-1]
        maturity_due = any(
            candidate.get("needs_initial_eval")
            and candidate.get("family") in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
            and current_date >= (self._candidate_maturity_date(candidate) or "9999-12-31")
            for candidate in self.active.values())
        if refreshed or pressure_pivot is not None or maturity_due:
            signals = self._evaluate(pressure_pivot)
        else:
            signals = []
        if not signals:
            signals = self._check_cached_breakout()
        if local_added or i >= 1:
            signals += self._check_bull_flag()
        if len(signals) > 1:
            merged = []
            for item in signals:
                if item["signal_date"] == str(pd.Timestamp(self.bars[-1]["date"]).date()):
                    merged.extend(item.get("patterns", [item["pattern"]]))
            if merged:
                patterns = [Pattern(**item) for item in merged]
                signals = [self._signal_many(patterns, str(pd.Timestamp(self.bars[-1]["date"]).date()))]
        self._report_progress()
        return self._remember(signals)


def generate_daily_signals(bars, code, kind, trade_start=None, progress_callback=None,
                           recognizer="v7", performance=None):
    """Replay all supplied bars and return confirmed signals; history is never reset."""
    if recognizer not in {"v7", "boundary_first"}:
        raise ValueError("recognizer must be 'v7' or 'boundary_first'")
    frame = pd.DataFrame(bars).copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return []
    metadata_tick = getattr(bars, "attrs", {}).get("quality", {}).get("tick_size")
    tick = float(metadata_tick if metadata_tick is not None else (0.001 if kind == "etf" else 0.01))
    session = DailyRecognitionSession(code, kind, tick, progress_callback=progress_callback,
                                      total_bars=len(frame),
                                      boundary_first=recognizer == "boundary_first")
    session._report_progress(force=True, phase="recognition")
    signals = []
    first = pd.Timestamp(trade_start).date().isoformat() if trade_start is not None else None
    for index, row in enumerate(frame.to_dict("records")):
        found = session.append_bar(row)
        if first is None or found and found[0]["signal_date"] >= first:
            signals.extend(found)
        session._report_progress(force=(index + 1 == len(frame)))
    dates = [str(pd.Timestamp(value).date()) for value in frame.date]
    for signal in signals:
        try:
            next_index = dates.index(signal["signal_date"]) + 1
            signal["entry_date"] = dates[next_index] if next_index < len(dates) else None
        except ValueError:
            signal["entry_date"] = None
    session._report_progress(force=True, phase="recognition_complete")
    if performance is not None:
        performance.update(session.performance)
        if session.neckline_potential_pair_cache is not None:
            performance.update(session.neckline_potential_pair_cache.stats)
        performance["engine_version"] = (BOUNDARY_FIRST_ENGINE_VERSION
                                          if recognizer == "boundary_first" else ENGINE_VERSION)
    return signals


def generate_daily_signals_boundary_first(bars, code, kind, trade_start=None,
                                          progress_callback=None, performance=None):
    """Causal pair-first recognizer; the default remains the v7 reference path."""
    return generate_daily_signals(bars, code, kind, trade_start=trade_start,
                                  progress_callback=progress_callback,
                                  recognizer="boundary_first", performance=performance)
