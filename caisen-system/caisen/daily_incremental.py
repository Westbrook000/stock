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
    detect_main_bottoms,
    _structure_candidates, _PotentialNecklinePairCache,
    _neckline_decline_check, _neckline_formation_coverage,
    _neckline_candidate_reversal_atr_multiple, _neckline_candidate_scale,
    _minimum_neckline_test_count, _same_neckline_boundary_phase,
    _neckline_test_tolerance_at,
)
from .patterns import PATTERN_LABELS, _line, _at
from .schemas import Pattern, Pivot

ENGINE_VERSION = "caisen-joint-pattern-session-v10-premature-boundary-retirement"
BOUNDARY_FIRST_ENGINE_VERSION = ENGINE_VERSION
SOURCE_RULE_VERSION = str(MAIN_RULES.get("source_rule_version", "unknown"))
from .research_config import recognition_context_snapshot


class DailyRecognitionSession:
    """Causal daily recognizer. Bars, ATR and confirmed pivots are retained."""

    def __init__(self, code, kind, tick_size=0.01, progress_callback=None, total_bars=None,
                 boundary_first=False, include_main_structures=True, timeframe="1d",
                 provenance=None, prior_price_context=None, coordinate_offset=0):
        self.code, self.kind = str(code), str(kind)
        self.timeframe = str(timeframe)
        if self.timeframe not in {"1d", "1w"}:
            raise ValueError("timeframe must be '1d' or '1w'")
        self.boundary_first = bool(boundary_first)
        self.include_main_structures = bool(include_main_structures)
        self.tick_size = max(float(tick_size), 1e-6)
        self.provenance = deepcopy(provenance or {})
        self.prior_price_context = prior_price_context
        self.coordinate_offset = int(coordinate_offset)
        provided_context = self.provenance.get("recognition_context")
        self.recognition_context = deepcopy(provided_context if provided_context is not None
                                            else recognition_context_snapshot())
        self.recognition_context.setdefault("recognition_experiment", None)
        self.recognition_config = None
        from .research_config import current_recognition_experiment
        self.recognition_config = current_recognition_experiment()
        self.recognition_context.setdefault("engine_version", ENGINE_VERSION)
        self.recognition_context.setdefault("source_rule_version", SOURCE_RULE_VERSION)
        self.recognition_context.setdefault("tick_size", self.tick_size)
        self.recognition_context.setdefault("tick_size_source",
                                            self.provenance.get("tick_size_source", "explicit_or_session"))
        self.recognition_context.setdefault("data_version", self.provenance.get("data_version", "unknown"))
        self.provenance["recognition_context"] = deepcopy(self.recognition_context)
        self.progress_callback = progress_callback
        self.total_bars = int(total_bars) if total_bars is not None else None
        self.progress_started_at = time.perf_counter()
        self.last_progress_at = 0.0
        self.bars: List[dict] = []
        self.date_strings: List[str] = []
        self._date_index_cache = {}
        self._date_index_cache_length = 0
        self._pressure_window_cache_key = None
        self._pressure_window_cache = []
        self.high_values: List[float] = []
        self.low_values: List[float] = []
        self.close_values: List[float] = []
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
        self.retired_patterns = {}
        self.pattern_lifecycle_registry = {}
        self.consumed_neckline_phases = {}
        self.boundary_crossing_cache = {}
        self.emitted_ids = set()
        self.emitted_events = []
        self.neckline_candidate_cache = {}
        self.neckline_pair_cache = {}
        self.neckline_coverage_cache = OrderedDict()
        self.neckline_potential_pair_cache = (_PotentialNecklinePairCache()
                                              if self.boundary_first else None)
        self.flag_candidates = {}
        self.flag_formations = {}
        self.flag_instances = {}
        self.last_signal_date = None
        self.performance = {"bars_appended": 0, "atr_updates": 0,
                            "pivot_updates": 0, "candidate_refreshes": 0,
                            "candidate_evaluations": 0, "prefix_frame_builds": 0,
                            "pressure_window_builds": 0, "pressure_window_cache_hits": 0,
                            "date_index_builds": 0}
        self._prefix_frame_cache = None
        self._prefix_frame_cache_length = -1

    def _frame_for_current_prefix(self):
        """Build one reusable frame and immutable numeric views for this prefix."""
        if (self._prefix_frame_cache is not None
                and self._prefix_frame_cache_length == len(self.bars)):
            return self._prefix_frame_cache
        frame = pd.DataFrame(self.bars)
        date_index = pd.DatetimeIndex(frame.date)
        frame.attrs["provenance"] = deepcopy(self.provenance)
        frame.attrs["date_strings"] = self.date_strings
        if self.prior_price_context is not None:
            frame.attrs["prior_price_context"] = self.prior_price_context
        frame.attrs["neckline_date_index"] = date_index
        frame.attrs["neckline_high_values"] = np.asarray(self.high_values, dtype=float)
        frame.attrs["neckline_low_values"] = np.asarray(self.low_values, dtype=float)
        frame.attrs["neckline_close_values"] = np.asarray(self.close_values, dtype=float)
        self._prefix_frame_cache = frame
        self._prefix_frame_cache_length = len(self.bars)
        self.performance["prefix_frame_builds"] += 1
        return frame

    def _index_for_date(self, date):
        """Resolve a date through a prefix-scoped map instead of a linear scan."""
        if self._date_index_cache_length != len(self.date_strings):
            self._date_index_cache = {value: index for index, value in enumerate(self.date_strings)}
            self._date_index_cache_length = len(self.date_strings)
            self.performance["date_index_builds"] += 1
        return self._date_index_cache.get(str(date))

    def _pressure_pivots_in_observation_window(self):
        """Reuse the sorted, observation-anchored pressure window per appended bar."""
        if not self.bars or not self.pressure["pivots"]:
            return []
        bar_index = len(self.bars) - 1
        cache_key = (bar_index, self._pressure_pivot_revision, self.timeframe)
        if cache_key == self._pressure_window_cache_key:
            self.performance["pressure_window_cache_hits"] += 1
            return self._pressure_window_cache
        years = int(MAIN_RULES.get("timeframes", {}).get(self.timeframe, {}).get(
            "structure_window_calendar_years", 5 if self.timeframe == "1d" else 10))
        cutoff = pd.Timestamp(self.bars[-1]["date"]) - pd.DateOffset(years=years)
        search_start = bisect_left(self.date_strings, str(cutoff.date()))
        window = [pivot for pivot in self.pressure["pivots"] if pivot.index >= search_start]
        if any(window[index - 1].index > window[index].index for index in range(1, len(window))):
            window.sort(key=lambda pivot: pivot.index)
        self._pressure_window_cache_key = cache_key
        self._pressure_window_cache = window
        self.performance["pressure_window_builds"] += 1
        return window

    def _prepare_neckline_frame(self, frame):
        """Attach the numeric history views shared by every candidate evaluation."""
        frame.attrs["neckline_candidate_cache"] = self.neckline_candidate_cache
        frame.attrs["neckline_pair_cache"] = self.neckline_pair_cache
        frame.attrs["neckline_coverage_cache"] = self.neckline_coverage_cache
        frame.attrs["joint_first_crossing_cache"] = self.boundary_crossing_cache
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
    def _keep_frozen_boundary(current, replacement):
        """Keep the first paired boundary for a live structure instance.

        A later pressure touch can produce a better ranked alternative, but it
        cannot silently redraw a boundary that was already frozen for this
        structure. A distinct boundary can only be considered after the
        current instance reaches a terminal state and is re-created from new
        confirmed evidence.
        """
        if current is None:
            return replacement
        if replacement is None:
            return current
        if current.state not in {"forming", "confirmed"}:
            return replacement
        current_details = {item.get("code"): item.get("value") for item in current.evidence}
        selection = current_details.get("neckline_selection") or {}
        if not selection.get("line_frozen_at"):
            return replacement
        from .structure_boundary import attach_joint_lifecycle
        current_lifecycle = attach_joint_lifecycle(current)
        replacement_lifecycle = attach_joint_lifecycle(replacement)
        if current_lifecycle["instance_id"] != replacement_lifecycle["instance_id"]:
            return current
        current_details = {item.get("code"): item.get("value") for item in current.evidence}
        replacement_details = {item.get("code"): item.get("value") for item in replacement.evidence}
        old_selection = current_details.get("neckline_selection") or {}
        new_selection = replacement_details.get("neckline_selection")
        if isinstance(new_selection, dict):
            for field in ("selected_line", "line_frozen_at", "freeze_date", "effective_start_date",
                          "selected_candidate_id", "independent_tests"):
                if field in old_selection:
                    new_selection[field] = deepcopy(old_selection[field])
        replacement.neckline = deepcopy(current.neckline)
        old_structure = current_details.get("main_structure") or {}
        new_structure = replacement_details.get("main_structure")
        if isinstance(new_structure, dict):
            for field in ("structure_known_at", "frozen_at", "frozen_at_index"):
                if field in old_structure:
                    new_structure[field] = deepcopy(old_structure[field])
        replacement_lifecycle = attach_joint_lifecycle(replacement)
        prior_events = current_lifecycle.get("events", [])
        prior_pair = next((event for event in prior_events if event.get("kind") == "paired"), None)
        if prior_pair:
            replacement_lifecycle["paired_at"] = current_lifecycle.get("paired_at")
            replacement_lifecycle["paired_at_status"] = current_lifecycle.get(
                "paired_at_status", "unknown_legacy_discovery_date")
            replacement_lifecycle["instance_discovered_at"] = current_lifecycle.get(
                "instance_discovered_at", current_lifecycle.get("paired_at"))
            replacement_lifecycle["instance_discovery_kind"] = current_lifecycle.get(
                "instance_discovery_kind", replacement_lifecycle["paired_at_status"])
            replacement_lifecycle["line_known_at"] = current_lifecycle.get("line_known_at")
            replacement_lifecycle["structure_known_at"] = current_lifecycle.get("structure_known_at")
            replacement_lifecycle["boundary_selected_at"] = current_lifecycle.get("boundary_selected_at")
        combined = []
        seen = set()
        new_events = replacement_lifecycle.get("events", [])
        if prior_pair:
            new_events = [event for event in new_events
                          if event.get("kind") != "paired" or
                          event.get("boundary_version_id") != prior_pair.get("boundary_version_id")]
        for event in prior_events + new_events:
            identity = (event.get("occurred_at", event.get("date")), event.get("kind"),
                        event.get("boundary_version_id"), event.get("close"))
            if identity not in seen:
                combined.append(deepcopy(event))
                seen.add(identity)
        combined.sort(key=lambda event: (event.get("date", ""), event.get("sequence", 0)))
        for sequence, event in enumerate(combined, start=1):
            event["sequence"] = sequence
            event["instance_id"] = replacement_lifecycle["instance_id"]
        replacement_lifecycle["events"] = combined
        for item in replacement.evidence:
            if item.get("code") == "joint_lifecycle":
                item["value"] = replacement_lifecycle
                break
        return replacement

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

    def _record_premature_boundary(self, candidate, pattern, index, level, buffer, previous_atr):
        details = {item["code"]: item["value"] for item in pattern.evidence}
        selection = details["neckline_selection"]
        line = dict(pattern.neckline)
        tests = details.get("neckline_tests", [])
        date = self.date_strings[index]
        early = {"date": date, "index": index, "close": self.close_values[index],
                 "frozen_neckline": level, "required_close": level + buffer,
                 "atr14_previous": previous_atr, "tolerance": buffer,
                 "candidate_id": selection.get("selected_candidate_id"),
                 "line": line, "tests": tests}
        phase = {"line": line, "tests": tests, "consumed_index": index,
                 "candidate_id": early["candidate_id"],
                 "reason": "breakout_before_minimum_structure_duration"}
        new_phase = not self._boundary_is_consumed(candidate, phase)
        self._remember_consumed_boundary(candidate, phase)
        existing = selection.setdefault("early_breakouts", [])
        if not any(item.get("date") == date and item.get("candidate_id") == early["candidate_id"]
                   for item in existing):
            existing.append(early)
        details["main_structure"].update(early_breakout_consumed=True, terminal_state="expired",
                                         structure_end_date=date, signal_eligible=False,
                                         eligibility_reason="terminal_state")
        pattern.state = "expired"
        pattern.last_observed_date = self.date_strings[-1]
        return new_phase

    def _audit_proposed_boundary(self, candidate, pattern):
        """Check this line's own history, including dates when another line ranked first."""
        if pattern is None or pattern.state != "forming" or not pattern.neckline:
            return False
        from .structure_boundary import FrozenBoundary, first_buffered_crossing
        details = {item["code"]: item["value"] for item in pattern.evidence}
        selection = details.get("neckline_selection") or {}
        frozen = self._index_for_date(selection.get("line_frozen_at"))
        structure_known = self._index_for_date(selection.get("structure_known_at"))
        maturity = self._candidate_maturity_date(candidate)
        if frozen is None or structure_known is None or not maturity:
            return False
        end = bisect_left(self.date_strings, maturity) - 1
        ready = max(frozen, structure_known)
        if end < ready:
            return False
        if len(self.close_values) != len(self.bars):
            self.close_values = [float(bar["close"]) for bar in self.bars]
        rules = MAIN_RULES["breakout"]
        multiple = float(rules.get("close_buffer_previous_atr_multiple", 1.0))
        ticks = float(rules.get("minimum_buffer_ticks", 2))
        boundary = FrozenBoundary("audit", pattern.neckline["slope"], pattern.neckline["intercept"])
        first = first_buffered_crossing(
            self.close_values, self.atr_values, boundary, ready + 1, end, self.tick_size,
            self.boundary_crossing_cache, buffer_multiple=multiple, minimum_ticks=ticks)
        if first is None:
            return False
        previous_atr = self.atr_values[first - 1]
        return self._record_premature_boundary(candidate, pattern, first, boundary.value_at(first),
                                               max(ticks * self.tick_size, multiple * previous_atr),
                                               previous_atr)

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
        minimum_date = pd.Timestamp(self.bars[-1]["date"]) - pd.DateOffset(
            years=10 if self.timeframe == "1w" else 6)
        minimum_index = bisect_left(self.date_strings, str(minimum_date.date()))
        self.neckline_potential_pair_cache.prune(minimum_index)
        self.neckline_potential_pair_cache.add(
            neckline_pivots, np.asarray(self.atr_values, dtype=float), highs, tolerances,
            self.tick_size, self.timeframe, minimum_index=minimum_index)

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
        months = int(MAIN_RULES.get("minimum_duration_calendar_months_by_timeframe", {}).get(
            self.timeframe, 6 if self.timeframe == "1d" else 12))
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

    def _retire_main_pattern(self, pattern, reason=None):
        if reason and pattern.state == "forming" and not pattern.confirmation_date:
            day = self.date_strings[-1]
            pattern.state = "invalidated"
            pattern.last_observed_date = day
            details = {item["code"]: item.get("value") for item in pattern.evidence}
            structure = details.get("main_structure")
            if isinstance(structure, dict):
                structure.update(terminal_state="invalidated", structure_end_date=day,
                                 signal_eligible=False, eligibility_reason="terminal_state")
            pattern.evidence.append({"code": "structure_window_invalidation", "value": {
                "date": day, "reason": reason}})
        lifecycle = self._attach_session_lifecycle(pattern)
        self.retired_patterns[lifecycle["instance_id"]] = deepcopy(pattern)

    def _triangle_window_reason(self, pattern, details, i, close):
        if pattern.pattern_type != "triangle_bottom" or not pattern.neckline:
            return None
        apex = (details.get("targets") or {}).get("triangle_apex_index")
        start = pattern.key_points.get("first_trough_index")
        if apex is None or start is None or apex <= start:
            return None
        ratio = (i - start) / (apex - start)
        prior_atr = self.atr_values[i - 1] if i > 0 else np.nan
        rules = MAIN_RULES["breakout"]
        if np.isfinite(prior_atr) and prior_atr > 0:
            required = _at(pattern.neckline, i) + max(
                float(rules["minimum_buffer_ticks"]) * self.tick_size,
                float(rules["close_buffer_previous_atr_multiple"]) * prior_atr)
            if close > required and not 0.5 <= ratio <= 0.75:
                return "triangle_first_breakout_outside_window"
        if ratio > 0.75:
            return "triangle_breakout_window_ended"
        return None

    def _maintain_candidates(self):
        if not self.bars:
            return
        i = len(self.bars) - 1
        current_date = pd.Timestamp(self.bars[i]["date"])
        years = int(MAIN_RULES.get("timeframes", {}).get(self.timeframe, {}).get(
            "structure_window_calendar_years", 5 if self.timeframe == "1d" else 10))
        cutoff = current_date - pd.DateOffset(years=years)
        retained = {}
        for key, candidate in self.active.items():
            sequence = candidate["sequence"]
            first = min(p.index for p in sequence)
            if pd.Timestamp(self.bars[first]["date"]) < cutoff:
                pattern = candidate.get("pattern")
                if pattern is not None:
                    self._retire_main_pattern(pattern, "structure_window_ended")
                continue
            known_at = candidate.get("structure_known_at")
            if known_at is None:
                scale_pivots = self.scale_pivots[candidate["scale"]]
                indexes = [p.index for p in sequence]
                lo, hi = min(indexes), max(indexes)
                known_at = max([p.confirmed_at for p in sequence] +
                               [p.confirmed_at for p in scale_pivots if lo <= p.index <= hi])
                candidate["structure_known_at"] = known_at
            pattern = candidate.get("pattern")
            if (pattern is not None and pattern.state in {"forming", "confirmed"}
                    and i > known_at):
                details = {item.get("code"): item.get("value") for item in pattern.evidence}
                structure = details.get("main_structure") or {}
                close = float(self.bars[i]["close"])
                terminal_code = None
                terminal_value = None
                if pattern.state == "confirmed" and pattern.neckline:
                    line_value = float(_at(pattern.neckline, i))
                    if close < line_value:
                        terminal_code = "neckline_invalidation"
                        terminal_value = {"date": self.date_strings[i], "index": i,
                                          "level": line_value, "close": close,
                                          "reason": "close_below_instance_frozen_boundary"}
                    else:
                        targets = details.get("targets") or {}
                        prices = targets.get("prices") or []
                        first_target = float(prices[0]) if prices else None
                        if (first_target is not None
                                and float(self.bars[i]["high"]) >= first_target):
                            terminal_code = "target_reached"
                            terminal_value = {"date": self.date_strings[i], "index": i,
                                              "price": first_target}
                elif pattern.state == "forming":
                    support = structure.get("support_lower", pattern.invalidation)
                    if support is not None and close < float(support):
                        terminal_code = "structure_invalidation"
                        terminal_value = {"date": self.date_strings[i], "index": i,
                                          "level": float(support), "close": close,
                                          "reason": "close_below_instance_structural_support"}
                    elif reason := self._triangle_window_reason(pattern, details, i, close):
                        self._remember_consumed_boundary(candidate, self._pattern_boundary_phase(pattern, i, reason))
                        self._retire_main_pattern(pattern, reason)
                        continue
                if terminal_code is not None:
                    evidence = next((item for item in pattern.evidence
                                     if item.get("code") == terminal_code), None)
                    if evidence is None:
                        pattern.evidence.append({"code": terminal_code, "value": terminal_value})
                    else:
                        evidence["value"] = terminal_value
                    terminal_state = ("target_reached" if terminal_code == "target_reached"
                                      else "invalidated")
                    structure.update({"terminal_state": terminal_state,
                                      "structure_end_date": self.date_strings[i],
                                      "signal_eligible": False,
                                      "eligibility_reason": "terminal_state"})
                    pattern.state = terminal_state
                    pattern.last_observed_date = self.date_strings[i]
                    lifecycle = self._attach_session_lifecycle(pattern)
                    terminal_snapshot = deepcopy(pattern)
                    self.retired_patterns[lifecycle["instance_id"]] = terminal_snapshot
                    continue
            elif pattern is None and i > known_at:
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
        usable_atr = np.isfinite(prior_atr) and prior_atr > 0
        out = []
        cutoff = max(0, i - 500)
        for key, seq in list(self.flag_candidates.items()):
            # Keep flag IDs stable when online analysis omits a leading history prefix.
            flag_prefix = "bull_flag" if self.timeframe == "1d" else "bull_flag:1w"
            anchor_key = "-".join(f"{p.index + self.coordinate_offset}:{p.kind}" for p in seq)
            pattern_id = f"{self.code}:{flag_prefix}:{anchor_key}"
            if seq[-1].index < cutoff:
                self.flag_candidates.pop(key, None)
                formation = self.flag_formations.get(pattern_id)
                if formation is not None and formation.state == "forming":
                    self._terminate_flag_formation(formation, "invalidated", self.date_strings[i],
                                                   "flag_search_window_exceeded", None)
                continue
            if not usable_atr or i < max(p.confirmed_at for p in seq):
                continue
            lo0, hi0, lo1, hi1, lo2 = [float(p.price) for p in seq]
            i0, i1, i2, i3, i4 = [p.index for p in seq]
            if any(float(self.bars[j]["close"]) < lo0 for j in range(seq[-1].confirmed_at, i + 1)):
                self.flag_candidates.pop(key, None)
                formation = self.flag_formations.get(pattern_id)
                if formation is not None and formation.state == "forming":
                    self._terminate_flag_formation(formation, "invalidated", self.date_strings[i],
                                                   "close_below_impulse_floor", lo0)
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
            formation = self.flag_formations.get(pattern_id)
            if formation is None:
                known_index = max(p.confirmed_at for p in seq)
                known_date = self.date_strings[known_index]
                geometry = {"impulse": impulse, "correction_ratio": correction / impulse,
                            "known_at": known_date,
                            "anchors": [{"date": p.date, "price": float(p.price), "kind": p.kind}
                                        for p in seq]}
                candidate = Pattern(
                    pattern_id=pattern_id, code=self.code, kind=self.kind, timeframe=self.timeframe,
                    pattern_type="bull_flag", direction="bullish", state="forming",
                    formed_at=seq[0].date, confirmed_at=None,
                    key_points={**{key: value for n, p in enumerate(seq, 1) for key, value in (
                        (f"pivot_{n}_index", p.index), (f"pivot_{n}_date", p.date),
                        (f"pivot_{n}", float(p.price)), (f"pivot_{n}_kind", p.kind))},
                                "first_trough_index": i0, "latest_anchor_index": i4,
                                "pivot_scale": "flag"},
                    neckline=dict(upper), invalidation=lo0,
                    evidence=[{"code": "flag_geometry", "value": geometry},
                              {"code": "flag_lifecycle", "value": {"state": "forming",
                               "known_at": known_date, "anchors": geometry["anchors"], "events": []}},
                              {"code": "recognition_context", "value": deepcopy(
                                  self.provenance.get("recognition_context") or {
                                      "recognition_experiment": None})}],
                    data_version=self.provenance.get("data_version", "unknown"),
                    last_observed_date=known_date, rule_version=ENGINE_VERSION,
                    source_rule_version=SOURCE_RULE_VERSION)
                self.flag_formations[pattern_id] = candidate
                self._attach_session_lifecycle(candidate, discovered_at=known_date)
                required_at_pairing = _at(upper, i) + max(minimum, prior_atr)
                if float(self.bars[i]["close"]) > required_at_pairing:
                    lifecycle = next(item["value"] for item in candidate.evidence
                                     if item["code"] == "flag_lifecycle")
                    lifecycle["buffered_crossing_consumed_at"] = self.date_strings[i]
                    lifecycle["events"].append({"date": self.date_strings[i],
                                                "kind": "rejected_too_early",
                                                "boundary": float(_at(upper, i)),
                                                "required_close": float(required_at_pairing),
                                                "close": float(self.bars[i]["close"]),
                                                "reason": "buffered_crossing_on_pairing_day"})
                    self._terminate_flag_formation(candidate, "expired", self.date_strings[i],
                                                   "buffered_crossing_on_pairing_day",
                                                   float(_at(upper, i)))
                    self.flag_candidates.pop(key, None)
                    self.emitted_ids.add(pattern_id)
                continue
            upper = dict(formation.neckline)
            formation_lifecycle = next(item["value"] for item in formation.evidence
                                       if item["code"] == "flag_lifecycle")
            if formation_lifecycle.get("buffered_crossing_consumed_at"):
                continue
            required = _at(upper, i) + max(minimum, prior_atr)
            previous_required = _at(upper, i - 1) + max(
                minimum, self.atr_values[i - 2] if i > 1 else minimum)
            if float(self.bars[i]["close"]) > required and float(
                    self.bars[i - 1]["close"]) > previous_required:
                formation_lifecycle["buffered_crossing_consumed_at"] = self.date_strings[i]
                formation_lifecycle["events"].append({"date": self.date_strings[i],
                    "kind": "rejected_too_early", "boundary": float(_at(upper, i)),
                    "required_close": float(required), "close": float(self.bars[i]["close"]),
                    "reason": "buffered_crossing_previously_consumed"})
                continue
            if float(self.bars[i]["close"]) <= required:
                continue
            if pattern_id in self.emitted_ids:
                continue
            self.emitted_ids.add(pattern_id)
            date = str(pd.Timestamp(self.bars[i]["date"]).date())
            unlocked = float(self.bars[i]["close"]) > hi0
            first_target = float(_at(upper, i) + impulse) if unlocked else None
            pattern = Pattern(pattern_id=pattern_id, code=self.code, kind=self.kind, timeframe=self.timeframe,
                              pattern_type="bull_flag", direction="bullish", state="confirmed",
                              formed_at=seq[0].date, confirmed_at=i,
                              key_points={**deepcopy(formation.key_points),
                                          "first_trough_index": i0,
                                          "latest_anchor_index": i4,
                                          "pivot_scale": "flag"},
                              neckline=dict(upper), invalidation=lo0,
                              evidence=[{"code": "flag_geometry", "value": next(
                                          item["value"] for item in formation.evidence
                                          if item["code"] == "flag_geometry")},
                                        {"code": "flag_breakout", "value": {"date": date,
                                          "close": float(self.bars[i]["close"]),
                                          "frozen_boundary": float(_at(upper, i)),
                                          "required_close": float(required),
                                          "atr14_previous": float(prior_atr),
                                          "prior_high": hi0, "impulse_size": impulse,
                                          "measured_target_locked": not unlocked,
                                          "target": first_target}},
                                        {"code": "flag_lifecycle", "value": {
                                          "state": "confirmed", "known_at": formation.last_observed_date,
                                          "entry_eligible_at_confirmation": True,
                                          "entry_eligibility_at_confirmation_status": "known",
                                          "confirmation_selected_at": date,
                                          "target": first_target,
                                          "anchors": next(item["value"].get("anchors", [])
                                                           for item in formation.evidence
                                                           if item["code"] == "flag_geometry"),
                                          "events": [{"date": date, "kind": "confirmed",
                                                      "boundary": float(_at(upper, i))}]}},
                                        {"code": "joint_lifecycle", "value": deepcopy(next(
                                          item["value"] for item in formation.evidence
                                          if item.get("code") == "joint_lifecycle"))},
                                        {"code": "recognition_context", "value": deepcopy(
                                          next(item["value"] for item in formation.evidence
                                               if item["code"] == "recognition_context"))}],
                              data_version=formation.data_version,
                              confirmation_date=date, last_observed_date=date,
                              rule_version=ENGINE_VERSION, source_rule_version=SOURCE_RULE_VERSION)
            parent = self._main_parent_for_flag(pattern, date)
            flag_lifecycle = next(item["value"] for item in pattern.evidence
                                  if item.get("code") == "flag_lifecycle")
            if parent is not None:
                pattern.parent_id = parent.pattern_id
                flag_lifecycle["entry_eligible_at_confirmation"] = False
                flag_lifecycle["selection_at_confirmation"] = {
                    "date": date, "selected": False, "parent_pattern_id": parent.pattern_id,
                    "rule": "active_main_structure_nested_suppression_v1"}
                pattern.evidence.append({"code": "execution_suppression", "value": {
                    "suppressed": True, "reason": "nested_in_main_bottom",
                    "parent_id": parent.pattern_id, "suppressed_at": date}})
            else:
                flag_lifecycle["selection_at_confirmation"] = {
                    "date": date, "selected": True, "parent_pattern_id": None,
                    "rule": "active_main_structure_nested_suppression_v1"}
            lifecycle = self._attach_session_lifecycle(
                pattern, discovered_at=formation.last_observed_date or date)
            self.flag_instances[pattern_id] = deepcopy(pattern)
            if parent is None and lifecycle.get("entry_eligible"):
                out.append(self._signal(pattern, date))
        return out

    def _main_parent_for_flag(self, flag, action_date):
        """Find an active main structure that suppresses this flag as of its confirmation."""
        pool = [candidate.get("pattern") for candidate in self.active.values()
                if candidate.get("pattern") is not None]
        rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
        pool = sorted((pattern for pattern in pool
                       if pattern.state in {"forming", "confirmed"}
                       and pattern.pattern_type in {"w_bottom", "inverse_head_shoulders",
                                                    "triangle_bottom"}),
                      key=lambda pattern: (rank.get(
                          (pattern.key_points or {}).get("pivot_scale"), -1),
                          pattern.pattern_id), reverse=True)
        for parent in pool:
            if flag.pattern_id in self._nested_ids([parent, flag], action_date):
                return parent
        return None

    def _attach_session_lifecycle(self, pattern, discovered_at=None):
        """Bind one observed Pattern object to its stable in-session instance history."""
        from .structure_boundary import JOINT_LIFECYCLE_SCHEMA_VERSION, attach_joint_lifecycle

        lifecycle = attach_joint_lifecycle(
            pattern, discovered_at=discovered_at,
            discovery_kind="session_observation" if discovered_at else "unknown_legacy_discovery_date")
        instance_id = lifecycle["instance_id"]
        previous = self.pattern_lifecycle_registry.get(instance_id)
        previous_selection = (previous or {}).get("selection_at_confirmation")
        current_selection = lifecycle.get("selection_at_confirmation")
        if previous_selection and current_selection != previous_selection:
            self._write_main_selection_evidence(pattern, previous_selection)
            lifecycle = attach_joint_lifecycle(
                pattern, discovered_at=discovered_at,
                discovery_kind="session_observation" if discovered_at else "unknown_legacy_discovery_date")
        if previous and previous.get("schema_version") == JOINT_LIFECYCLE_SCHEMA_VERSION:
            for key in ("paired_at", "paired_at_status", "instance_discovered_at",
                        "instance_discovery_kind", "line_known_at", "structure_known_at",
                        "boundary_selected_at"):
                lifecycle[key] = deepcopy(previous.get(key))
            combined = []
            seen = {}
            conflicts = deepcopy(previous.get("event_conflicts", []))
            prior_pairs = [event for event in previous.get("events", [])
                           if event.get("kind") == "paired"]
            new_events = lifecycle.get("events", [])
            if prior_pairs:
                prior_boundary_ids = {event.get("boundary_version_id") for event in prior_pairs}
                new_events = [event for event in new_events
                              if event.get("kind") != "paired"
                              or event.get("boundary_version_id") not in prior_boundary_ids]
            for event in previous.get("events", []):
                identity = (event.get("kind"), event.get("boundary_version_id"),
                            event.get("occurred_at", event.get("date")))
                seen[identity] = event
                combined.append(deepcopy(event))
            previous_terminal = previous.get("state_at") in {"expired", "invalidated", "target_reached"}
            for event in new_events:
                identity = (event.get("kind"), event.get("boundary_version_id"),
                            event.get("occurred_at", event.get("date")))
                old = seen.get(identity)
                if old is not None:
                    business_fields = ("close", "boundary", "required_close", "atr", "price",
                                       "reason", "line")
                    changed = {field: {"canonical": old.get(field), "observed": event.get(field)}
                               for field in business_fields if old.get(field) != event.get(field)}
                    if changed:
                        conflicts.append({"kind": "event_payload_conflict", "identity": identity,
                                          "recorded_at": previous.get("paired_at"),
                                          "fields": changed})
                    continue
                if previous_terminal and event.get("kind") in {
                        "confirmed", "invalidated", "target_reached"}:
                    conflicts.append({"kind": "terminal_instance_transition_blocked",
                                      "identity": identity, "observed_event": deepcopy(event),
                                      "recorded_at": previous.get("paired_at")})
                    continue
                combined.append(deepcopy(event))
                seen[identity] = event
            combined.sort(key=lambda event: (event.get("date", ""),
                                             event.get("sequence", 0), event.get("kind", "")))
            for sequence, event in enumerate(combined, start=1):
                event["sequence"] = sequence
                event["instance_id"] = instance_id
            lifecycle["events"] = combined
            if conflicts:
                lifecycle["event_conflicts"] = conflicts
                lifecycle["entry_eligible"] = False
                details = {item.get("code"): item.get("value") for item in pattern.evidence}
                structure = details.get("main_structure")
                if isinstance(structure, dict):
                    structure["signal_eligible"] = False
                    structure["eligibility_reason"] = "lifecycle_event_conflict"
                flag_lifecycle = details.get("flag_lifecycle")
                if isinstance(flag_lifecycle, dict):
                    flag_lifecycle["entry_eligible"] = False

            terminal_state = previous.get("state_at")
            if terminal_state in {"expired", "invalidated", "target_reached"}:
                pattern.state = terminal_state
                pattern.confirmation_date = previous.get("canonical_confirmation_date")
                pattern.confirmed_at = previous.get("canonical_confirmed_at")
                details = {item.get("code"): item.get("value") for item in pattern.evidence}
                structure = details.get("main_structure")
                if isinstance(structure, dict):
                    structure.update({"terminal_state": terminal_state, "signal_eligible": False,
                                      "eligibility_reason": "terminal_state"})
                flag_lifecycle = details.get("flag_lifecycle")
                if isinstance(flag_lifecycle, dict):
                    flag_lifecycle["state"] = terminal_state
                lifecycle["state_at"] = terminal_state
                lifecycle["canonical_confirmation_date"] = previous.get(
                    "canonical_confirmation_date")
                lifecycle["canonical_confirmed_at"] = previous.get("canonical_confirmed_at")
                lifecycle["entry_eligible"] = False
                if previous.get("entry_eligible_at_confirmation") is not None:
                    lifecycle["entry_eligible_at_confirmation"] = previous[
                        "entry_eligible_at_confirmation"]
                    lifecycle["entry_eligibility_at_confirmation_status"] = previous.get(
                        "entry_eligibility_at_confirmation_status")
            if previous_selection:
                lifecycle["selection_at_confirmation"] = deepcopy(previous_selection)
                lifecycle["selection_reason"] = previous.get("selection_reason")
                lifecycle["entry_eligible_at_confirmation"] = previous.get(
                    "entry_eligible_at_confirmation")
                lifecycle["entry_eligibility_at_confirmation_status"] = previous.get(
                    "entry_eligibility_at_confirmation_status")
            for item in pattern.evidence:
                if item.get("code") == "joint_lifecycle":
                    item["value"] = lifecycle
                    break
        self.pattern_lifecycle_registry[instance_id] = deepcopy(lifecycle)
        return lifecycle

    @staticmethod
    def _write_main_selection_evidence(pattern, selection):
        structure = None
        for item in pattern.evidence:
            if item.get("code") == "main_structure" and isinstance(item.get("value"), dict):
                structure = item["value"]
                structure["selection_at_confirmation"] = deepcopy(selection)
                structure["entry_eligible_at_confirmation"] = bool(selection.get("selected"))
                break
        if structure is None:
            return
        selected = bool(selection.get("selected"))
        value = {
            "selected": selected,
            "decided_at": selection.get("date"),
            "primary_pattern_id": selection.get("primary_pattern_id"),
            "primary_instance_id": selection.get("primary_instance_id"),
            "reason": selection.get("reason"),
        }
        for item in pattern.evidence:
            if item.get("code") == "main_selection":
                item["value"] = value
                break
        else:
            pattern.evidence.append({"code": "main_selection", "value": value})

    def _persist_main_selection(self, pattern, selection):
        lifecycle = self._attach_session_lifecycle(pattern)
        instance_id = lifecycle["instance_id"]
        previous = self.pattern_lifecycle_registry.get(instance_id) or {}
        canonical = previous.get("selection_at_confirmation") or selection
        aliases = [pattern]
        aliases.extend(candidate.get("pattern") for candidate in self.active.values())
        aliases.extend(self.retired_patterns.values())
        seen = set()
        for alias in aliases:
            if alias is None or id(alias) in seen:
                continue
            seen.add(id(alias))
            alias_lifecycle = next((item.get("value", {}) for item in alias.evidence
                                   if item.get("code") == "joint_lifecycle"), {})
            if alias_lifecycle.get("instance_id") != instance_id:
                continue
            self._write_main_selection_evidence(alias, canonical)
            self._attach_session_lifecycle(alias)
        return canonical

    def _terminate_flag_formation(self, pattern, state, date, reason, boundary):
        pattern.state = state
        pattern.last_observed_date = str(date)
        lifecycle = next(item["value"] for item in pattern.evidence
                         if item.get("code") == "flag_lifecycle")
        lifecycle["state"] = state
        lifecycle.setdefault("events", []).append({"date": str(date), "kind": state,
                                                     "boundary": boundary, "reason": reason})
        attached = self._attach_session_lifecycle(pattern)
        self.retired_patterns[attached["instance_id"]] = deepcopy(pattern)

    def _advance_flag_lifecycles(self, index):
        """Advance confirmed flag instances on their frozen upper boundary."""
        if index < 0 or index >= len(self.bars):
            return
        date = self.date_strings[index] if index < len(self.date_strings) else str(
            pd.Timestamp(self.bars[index]["date"]).date())
        row = self.bars[index]
        from .structure_boundary import attach_joint_lifecycle
        for pattern in self.flag_instances.values():
            if pattern.state != "confirmed" or index <= int(pattern.confirmed_at):
                continue
            boundary = float(_at(pattern.neckline, index))
            detail = {item.get("code"): item.get("value") for item in pattern.evidence}
            breakout = detail.get("flag_breakout", {}) or {}
            lifecycle = detail.get("flag_lifecycle", {}) or {}
            target = lifecycle.get("target")
            event = None
            if float(row["close"]) < boundary:
                pattern.state = "invalidated"
                event = {"date": date, "kind": "invalidated", "boundary": boundary,
                         "close": float(row["close"])}
                pattern.invalidation = boundary
            else:
                if target is None and not breakout.get("measured_target_locked"):
                    target = breakout.get("target")
                    lifecycle["target"] = target
                elif target is None and float(row["close"]) > float(
                        breakout.get("prior_high", float("inf"))):
                    impulse = float(breakout.get("impulse_size", 0.0))
                    target = boundary + impulse
                    lifecycle["target"] = target
                    breakout.update({"measured_target_locked": False, "target": target,
                                     "target_unlocked_date": date})
                    pattern.targets = [float(target)]
                if target is not None and float(row["high"]) >= float(target):
                    pattern.state = "target_reached"
                    event = {"date": date, "kind": "target_reached", "price": float(target),
                             "high": float(row["high"])}
            if event is None:
                continue
            lifecycle.setdefault("events", []).append(event)
            lifecycle["state"] = pattern.state
            pattern.last_observed_date = date
            for item in pattern.evidence:
                if item.get("code") == "flag_lifecycle":
                    item["value"] = lifecycle
                elif item.get("code") == "flag_breakout":
                    item["value"] = breakout
            attached = self._attach_session_lifecycle(pattern)
            if pattern.state in {"invalidated", "target_reached"}:
                self.retired_patterns[attached["instance_id"]] = deepcopy(pattern)

    def _refresh_structures(self, changed_scales=None):
        """Refresh candidate sequences only when a confirmed large-scale pivot changes."""
        current_date = pd.Timestamp(self.bars[-1]["date"])
        pressure = self.pressure["pivots"]
        years = int(MAIN_RULES.get("timeframes", {}).get(self.timeframe, {}).get(
            "structure_window_calendar_years", 5 if self.timeframe == "1d" else 10))
        cutoff = current_date - pd.DateOffset(years=years)
        start_index = bisect_left(self.date_strings, str(cutoff.date()))
        retained = {}
        for key, value in self.active.items():
            if min(p.index for p in value["sequence"]) >= start_index:
                retained[key] = value
            elif value.get("pattern") is not None:
                self._retire_main_pattern(value["pattern"], "structure_window_ended")
        self.active = retained
        for scale, pivots in self.scale_pivots.items():
            if changed_scales is not None and scale not in changed_scales:
                continue
            recent = [p for p in pivots if p.index >= start_index]
            if not recent:
                continue
            for family, seq in _structure_candidates(recent, self.timeframe, self.atr_values, self.tick_size,
                                                      low_values=np.asarray(self.low_values, dtype=float),
                                                      shoulder_pivots=pressure,
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
        gap_key = ("minimum_separation_weekly_bars" if self.timeframe == "1w"
                   else "minimum_separation_daily_sessions")
        minimum_gap = int(test_rules.get(gap_key, 2 if self.timeframe == "1w" else 10))
        required_groups = _minimum_neckline_test_count()
        minimum_ticks = float(test_rules.get("minimum_tolerance_ticks", 2)) * self.tick_size
        tolerance_atr = float(test_rules.get("tolerance_atr", 0.5))
        pressure = self._pressure_pivots_in_observation_window()
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
                self.atr_values[middle.index])
                if middle.index < len(self.atr_values)
                and np.isfinite(self.atr_values[middle.index]) else 0.0)
            minimum_level = float(middle.price) - middle_tolerance
            maximum_level = float(middle.price) * (1.0 + float(test_rules.get(
                "w_bottom_rim_association_ceiling_relative", 0.2)))
            # The cached pressure slice already uses the same observation-anchored
            # 5y/10y geometry window as the batch selector.
            eligible = [p for p in pressure if float(p.price) <= maximum_level]
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
        gap_key = ("minimum_separation_weekly_bars" if self.timeframe == "1w"
                   else "minimum_separation_daily_sessions")
        minimum_gap = int(test_rules.get(gap_key, 2 if self.timeframe == "1w" else 10))
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
        search_years = int(MAIN_RULES.get("timeframes", {}).get(self.timeframe, {}).get(
            "structure_window_calendar_years", 5 if self.timeframe == "1d" else 10))
        search_start_date = (pd.Timestamp(self.bars[-1]["date"]) -
                             pd.DateOffset(years=search_years)) if self.bars else pd.Timestamp.min
        search_start_index = bisect_left(self.date_strings, str(search_start_date.date()))
        if family == "w_bottom":
            earliest = search_start_date
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
            known_atr = (self.atr_values[middle.index]
                         if middle.index < len(self.atr_values) else np.nan)
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
                                if p.index >= search_start_index and p.index != index]
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
        # Some focused callers construct a stream from an already materialized
        # prefix. Keep the same date/index invariant as append_bar before the
        # candidate evaluator indexes the current date or builds cached frames.
        if len(self.date_strings) != len(self.bars):
            self.date_strings = [str(pd.Timestamp(bar["date"]).date())
                                 for bar in self.bars]
        if not self.date_strings:
            return []
        current_index = len(self.bars) - 1
        current_date = self.date_strings[current_index]
        instances_paired_before_evaluation = set(self.pattern_lifecycle_registry)
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

        preeligible = []
        for key, candidate in self.active.items():
            geometry = geometry_key(candidate)
            if key != dominant_candidate.get(geometry):
                continue
            family = candidate["family"]
            if family in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}:
                if current_date < (self._candidate_maturity_date(candidate) or "9999-12-31"):
                    continue
            cached = candidate.get("pattern")
            if cached is not None and (cached.confirmation_date or cached.state in
                                       {"expired", "invalidated", "target_reached"}):
                continue
            if pressure_pivot is None and not candidate.get("needs_initial_eval"):
                continue
            if (pressure_pivot is not None
                    and not self._pressure_pivot_can_change_candidate(candidate, pressure_pivot)):
                continue
            if not self._has_pressure_test_pair(candidate):
                continue
            preeligible.append((key, candidate, geometry))
        if not preeligible:
            return []

        frame = self._prepare_neckline_frame(self._frame_for_current_prefix())
        high_values = frame.attrs["neckline_high_values"]
        low_values = frame.attrs["neckline_low_values"]
        close_values = frame.attrs["neckline_close_values"]
        atr_values = np.asarray(self.atr_values, dtype=float)
        frame.attrs["neckline_atr_values"] = atr_values
        event_candidates = []
        all_neckline_pivots = []
        by_identity = {}
        for pivots in self.scale_pivots.values():
            for pivot in pivots:
                key = (pivot.index, pivot.kind)
                if key not in by_identity or float(pivot.scale.split("ATR")[0]) > float(by_identity[key].scale.split("ATR")[0]):
                    by_identity[key] = pivot
        all_neckline_pivots.extend(by_identity.values())
        cutoff = pd.Timestamp(frame.date.iloc[-1]) - pd.DateOffset(
            years=10 if self.timeframe == "1w" else 6)
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
        evaluated_geometries = set()
        for key, candidate, geometry in preeligible:
            family, seq, scale = candidate["family"], candidate["sequence"], candidate["scale"]
            evaluated_geometries.add(geometry)
            self.performance["candidate_evaluations"] += 1
            patterns = []
            retried_boundary = False
            while True:
                batch = _make_main_pattern(
                    frame, self.code, self.kind, self.timeframe, family, seq,
                    all_neckline_pivots, atr_values, self.tick_size, scale, "5y",
                    background_bars=frame, collect_diagnostics=False,
                    consumed_boundary_phases=self.consumed_neckline_phases.get(
                        self._candidate_geometry_key(candidate), ()))
                batch = batch if isinstance(batch, list) else [batch]
                rerank = False
                for proposed in batch:
                    rerank = self._audit_proposed_boundary(candidate, proposed) or rerank
                patterns.extend(batch)
                # Exclude each newly consumed line and try another qualified boundary.
                if not rerank or any(p and p.state in {"forming", "confirmed"} for p in batch):
                    break
                retried_boundary = True
            for pattern in patterns:
                if pattern is None:
                    continue
                if not any(item.get("code") == "recognition_context" for item in pattern.evidence):
                    pattern.evidence.append({"code": "recognition_context",
                                             "value": deepcopy(self.recognition_context)})
                details = {item.get("code"): item.get("value") for item in pattern.evidence}
                structure = details.get("main_structure")
                if isinstance(structure, dict):
                    structure.setdefault("instance_discovered_at", current_date)
                from .structure_boundary import attach_joint_lifecycle
                lifecycle = self._attach_session_lifecycle(pattern, discovered_at=current_date)
                if pattern.state == "forming" and lifecycle.get("paired_at") == current_date:
                    reason = self._triangle_window_reason(
                        pattern, details, current_index, float(self.bars[current_index]["close"]))
                    if reason:
                        self._remember_consumed_boundary(candidate, self._pattern_boundary_phase(pattern, current_index, reason))
                        self._retire_main_pattern(pattern, reason)
                if (pattern.confirmation_date == current_date
                        and lifecycle.get("paired_at") == current_date
                        and lifecycle.get("instance_id") not in instances_paired_before_evaluation):
                    pairing_index = pattern.confirmed_at
                    breakout = details.get("neckline_breakout")
                    if isinstance(breakout, dict):
                        breakout["consumed_on_pairing_day"] = True
                    structure = details.get("main_structure")
                    if isinstance(structure, dict):
                        structure.update({"terminal_state": "expired", "structure_end_date": current_date,
                                          "signal_eligible": False,
                                          "entry_eligible_at_confirmation": False,
                                          "eligibility_reason": "breakout_on_instance_pairing_date"})
                    pattern.state = "expired"
                    pattern.confirmation_date = None
                    pattern.confirmed_at = None
                    pattern.last_observed_date = current_date
                    pattern.evidence.append({"code": "pairing_day_cross_consumed", "value": {
                        "date": current_date, "occurred_at": current_date, "recorded_at": current_date,
                        "close": breakout.get("close") if isinstance(breakout, dict) else None,
                        "required_close": (breakout.get("required_close")
                                           if isinstance(breakout, dict) else None),
                        "reason": "breakout_on_instance_pairing_date"}})
                    prior_canonical = self.pattern_lifecycle_registry.get(lifecycle["instance_id"])
                    if prior_canonical is not None:
                        prior_canonical["events"] = [event for event in prior_canonical.get("events", [])
                                                      if event.get("kind") != "confirmed"]
                        prior_canonical["state_at"] = "forming"
                        prior_canonical["canonical_confirmation_date"] = None
                        prior_canonical["canonical_confirmed_at"] = None
                        prior_canonical["entry_eligible_at_confirmation"] = None
                        prior_canonical["entry_eligibility_at_confirmation_status"] = "not_confirmed"
                    pattern.evidence = [item for item in pattern.evidence
                                        if item.get("code") != "joint_lifecycle"]
                    lifecycle = self._attach_session_lifecycle(pattern)
                    self._remember_consumed_boundary(candidate, self._pattern_boundary_phase(
                        pattern, pairing_index, "pairing_day_cross_consumed"))
                    self.retired_patterns[lifecycle["instance_id"]] = deepcopy(pattern)
            replacement = next((p for p in batch if p is not None
                                and p.state not in {"invalidated", "target_reached", "expired"}),
                               next((p for p in batch if p is not None), None))
            if retried_boundary and replacement is not None and replacement.state in {
                    "invalidated", "target_reached", "expired"}:
                self._retire_main_pattern(replacement)
                replacement = None
            candidate["pattern"] = self._keep_frozen_boundary(candidate.get("pattern"), replacement)
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
                if selection.get("early_breakouts"):
                    lifecycle = self._attach_session_lifecycle(pattern)
                    if lifecycle.get("state_at") == "expired":
                        self.retired_patterns[lifecycle["instance_id"]] = deepcopy(pattern)
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
        selected_lifecycle = self._attach_session_lifecycle(selected)
        primary_instance_id = selected_lifecycle["instance_id"]
        for pattern in eligible:
            selected_instance = self._attach_session_lifecycle(pattern)["instance_id"] == primary_instance_id
            self._persist_main_selection(pattern, {
                "date": current_date, "selected": selected_instance,
                "primary_pattern_id": selected.pattern_id,
                "primary_instance_id": primary_instance_id,
                "reason": "latest_confirmation_then_pattern_family_then_coarsest_ihs_scale_then_structure_duration_then_pressure_evidence"})
        self._persist_main_selection(selected, {
            "date": current_date, "selected": True,
            "primary_pattern_id": selected.pattern_id,
            "primary_instance_id": primary_instance_id,
            "reason": "latest_confirmation_then_pattern_family_then_coarsest_ihs_scale_then_structure_duration_then_pressure_evidence"})
        snapshots = [self._selected_snapshot(pattern, current_date,
                                             selected=(self._attach_session_lifecycle(pattern)["instance_id"]
                                                       == primary_instance_id),
                                             primary_id=selected.pattern_id,
                                             primary_instance_id=primary_instance_id)
                     for pattern in eligible]
        self.emitted_ids.update(pattern.pattern_id for pattern in eligible)
        return [self._signal_many(snapshots, current_date)]

    def _check_cached_breakout(self):
        """Check a frozen, already-qualified boundary on bars without new pivots."""
        if len(self.bars) < 2:
            return []
        i = len(self.bars) - 1
        if len(self.close_values) != len(self.bars):
            self.close_values = [float(bar["close"]) for bar in self.bars]
        current_date = str(pd.Timestamp(self.bars[i]["date"]).date())
        candidates = []
        rules = MAIN_RULES.get("breakout", {})
        buffer_multiple = float(rules.get("close_buffer_previous_atr_multiple", 1.0))
        minimum_ticks = float(rules.get("minimum_buffer_ticks", 2)) * self.tick_size
        for candidate in self.active.values():
            pattern = candidate.get("pattern")
            if (pattern is None or pattern.confirmation_date
                    or pattern.state in {"expired", "invalidated", "target_reached"}):
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
            pressure_start = self._index_for_date(pressure_date)
            frozen_index = self._index_for_date(frozen_date)
            structure_known_index = self._index_for_date(structure_known)
            if pressure_start is None or frozen_index is None or structure_known_index is None:
                continue
            decline = _neckline_decline_check(pattern.pattern_type, line, pressure_start, i)
            if not decline["passed"]:
                continue
            formation_start = pressure_start
            formation_end = max(frozen_index, structure_known_index)
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
                from .structure_boundary import FrozenBoundary, first_buffered_crossing
                boundary = FrozenBoundary("audit", float(line["slope"]), float(line["intercept"]))
                ready = max(frozen_index, structure_known_index)
                first = first_buffered_crossing(
                    self.close_values, self.atr_values, boundary, ready + 1, i, self.tick_size,
                    self.boundary_crossing_cache, buffer_multiple=buffer_multiple,
                    minimum_ticks=float(rules.get("minimum_buffer_ticks", 2)))
                if first != i:
                    continue
                targets = next((item["value"] for item in pattern.evidence if item["code"] == "targets"), {})
                apex = targets.get("triangle_apex_index") if isinstance(targets, dict) else None
                if pattern.pattern_type == "triangle_bottom" and apex is not None:
                    trough_index = int(pattern.key_points["first_trough_index"])
                    position = (i - trough_index) / max(float(apex) - trough_index, 1.0)
                    if not 0.5 <= position <= 0.75:
                        continue
                if current_date < (self._candidate_maturity_date(candidate) or "9999-12-31"):
                    self._record_premature_boundary(candidate, pattern, i, _at(line, i),
                                                    required - _at(line, i), previous_atr)
                    self._retire_main_pattern(pattern)
                    candidate.update(pattern=None, needs_initial_eval=True)
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
                candidate["pattern"] = deepcopy(snapshot)
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
        selected_lifecycle = self._attach_session_lifecycle(selected)
        primary_instance_id = selected_lifecycle["instance_id"]
        for pattern in eligible:
            selected_instance = self._attach_session_lifecycle(pattern)["instance_id"] == primary_instance_id
            self._persist_main_selection(pattern, {
                "date": current_date, "selected": selected_instance,
                "primary_pattern_id": selected.pattern_id,
                "primary_instance_id": primary_instance_id,
                "reason": "latest_confirmation_then_pattern_family_then_coarsest_ihs_scale_then_structure_duration_then_pressure_evidence"})
        self._persist_main_selection(selected, {
            "date": current_date, "selected": True,
            "primary_pattern_id": selected.pattern_id,
            "primary_instance_id": primary_instance_id,
            "reason": "latest_confirmation_then_pattern_family_then_coarsest_ihs_scale_then_structure_duration_then_pressure_evidence"})
        snapshots = [self._selected_snapshot(pattern, current_date,
                                             selected=(self._attach_session_lifecycle(pattern)["instance_id"]
                                                       == primary_instance_id),
                                             primary_id=selected.pattern_id,
                                             primary_instance_id=primary_instance_id)
                     for pattern in eligible]
        self.emitted_ids.update(pattern.pattern_id for pattern in eligible)
        return [self._signal_many(snapshots, current_date)]

    def _selected_snapshot(self, pattern, decided_at, selected=True, primary_id=None,
                           primary_instance_id=None):
        snapshot = deepcopy(pattern)
        lifecycle = next((item.get("value", {}) for item in snapshot.evidence
                          if item.get("code") == "joint_lifecycle"), {})
        self._write_main_selection_evidence(snapshot, {
            "date": decided_at, "selected": bool(selected),
            "primary_pattern_id": primary_id or pattern.pattern_id,
            "primary_instance_id": primary_instance_id or lifecycle.get("instance_id"),
            "reason": "latest_confirmation_then_pattern_family_then_coarsest_ihs_scale_then_structure_duration_then_pressure_evidence"})
        return snapshot

    def _remember(self, signals):
        self.emitted_events.extend(deepcopy(signals))
        return signals

    def _signal(self, pattern, signal_date):
        return self._signal_many([pattern], signal_date)

    def _signal_many(self, patterns, signal_date):
        from .structure_boundary import attach_joint_lifecycle
        for pattern in patterns:
            self._attach_session_lifecycle(pattern)
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
    def _structure_anchor_signature(pattern):
        points = pattern.key_points or {}
        signature = []
        for number in range(1, 10):
            date = points.get(f"pivot_{number}_date")
            kind = points.get(f"pivot_{number}_kind")
            if date and kind:
                signature.append((str(date), str(kind)))
        return (pattern.pattern_type, tuple(signature)) if signature else None

    @staticmethod
    def _price_contains_structure(parent, child, parent_structure, child_structure):
        """Require a proposed parent bottom to contain the child's price envelope."""
        parent_floor = parent_structure.get("support_lower")
        child_floor = child_structure.get("support_lower")
        parent_line, child_line = parent.neckline, child.neckline
        parent_points, child_points = parent.key_points or {}, child.key_points or {}
        child_start = child_points.get("first_trough_index", child_points.get("pivot_1_index"))
        child_end = child_points.get("latest_anchor_index", child_points.get("pivot_2_index"))
        if (parent_floor is None or child_floor is None or parent_line is None or child_line is None
                or child_start is None or child_end is None):
            return False
        if float(parent_floor) > float(child_floor):
            return False
        parent_slope = float(parent_line.get("slope", 0.0))
        parent_intercept = float(parent_line.get("intercept", 0.0))
        child_slope = float(child_line.get("slope", 0.0))
        child_intercept = float(child_line.get("intercept", 0.0))
        for index in (int(child_start), int(child_end)):
            if parent_slope * index + parent_intercept < child_slope * index + child_intercept:
                return False
        return True

    @staticmethod
    def _nested_ids(patterns, action_date):
        rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
        main_families = {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
        as_of = pd.Timestamp(action_date)
        nested = set()
        # Terminal parents must release their nested-structure suppression.
        active = [p for p in patterns if p.state in {"forming", "confirmed"}]
        for child in active:
            child_points = child.key_points
            child_dates = [pd.Timestamp(value) for key, value in child_points.items()
                           if key.startswith("pivot_") and key.endswith("_date") and value]
            if not child_dates:
                continue
            child_detail = {item["code"]: item["value"] for item in child.evidence}
            child_structure = child_detail.get("main_structure", {})
            if child.pattern_type == "bull_flag":
                flag_geometry = child_detail.get("flag_geometry", {}) or {}
                anchors = flag_geometry.get("anchors", []) or []
                child_structure = {**child_structure,
                                   "first_trough_date": (anchors[0].get("date") if anchors else
                                                          child_points.get("pivot_1_date")),
                                   "structure_end_date": (anchors[-1].get("date") if anchors else
                                                           child_points.get("pivot_5_date")),
                                   "support_lower": child.invalidation}
            child_start = pd.Timestamp(child_structure.get("first_trough_date") or min(child_dates))
            child_end = pd.Timestamp(child_structure.get("structure_end_date") or max(child_dates))
            child_end = min(child_end, as_of)
            for parent in active:
                if (parent.pattern_type not in main_families
                        or parent.pattern_id == child.pattern_id):
                    continue
                if (parent.pattern_type in main_families
                        and child.pattern_type in main_families):
                    parent_signature = DailyRecognitionSession._structure_anchor_signature(parent)
                    child_signature = DailyRecognitionSession._structure_anchor_signature(child)
                    # The same confirmed pivots at another ATR scale are one
                    # structure with multiple measurements, not a parent and
                    # an internal child competing for a buy event.
                    suppression_rules = MAIN_RULES.get("main_structure_suppression", {})
                    if (suppression_rules.get("merge_identical_anchors_across_scales", True)
                            and parent_signature is not None
                            and parent_signature == child_signature):
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
                        and known_at <= as_of
                        and (child.pattern_type not in main_families
                             or not MAIN_RULES.get("main_structure_suppression", {}).get(
                                 "require_parent_price_envelope_containment", True)
                             or DailyRecognitionSession._price_contains_structure(
                                 parent, child, parent_structure, child_structure))):
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
        from .research_config import recognition_experiment
        with recognition_experiment(self.recognition_config):
            return self._append_bar(bar)

    def _append_bar(self, bar: dict) -> List[dict]:
        row = dict(bar)
        row["date"] = pd.Timestamp(row["date"])
        for name in ("open", "high", "low", "close"):
            row[name] = float(row[name])
        if self.bars and row["date"] <= self.bars[-1]["date"]:
            raise ValueError("daily bars must be appended in strictly increasing date order")
        self.bars.append(row)
        self.high_values.append(float(row["high"]))
        self.low_values.append(float(row["low"]))
        self.close_values.append(float(row["close"]))
        self._prefix_frame_cache = None
        self._prefix_frame_cache_length = -1
        date_text = str(row["date"].date())
        previous_date_count = len(self.date_strings)
        self.date_strings.append(date_text)
        if self._date_index_cache_length == previous_date_count:
            self._date_index_cache[date_text] = previous_date_count
            self._date_index_cache_length = previous_date_count + 1
        self.performance["bars_appended"] += 1
        i = len(self.bars) - 1
        previous_close = float(self.bars[i - 1]["close"]) if i else row["close"]
        true_range = max(row["high"] - row["low"], abs(row["high"] - previous_close),
                         abs(row["low"] - previous_close))
        self.atr_values.append(true_range if i == 0 else self.atr_values[-1] +
                               (true_range - self.atr_values[-1]) / 14.0)
        self.performance["atr_updates"] += 1
        refreshed = self._append_scale_pivots(i) if i and self.include_main_structures else []
        pressure_pivot = self._append_pressure_pivot(i) if i and self.include_main_structures else None
        if pressure_pivot is not None:
            self._pressure_pivot_revision += 1
        changed_pressure_variant = False
        if self.boundary_first and refreshed:
            pressure_locations = {(item.index, item.kind) for item in self.pressure["pivots"]}
            changed_pressure_variant = any((pivot.index, pivot.kind) in pressure_locations
                                           for _, pivot in refreshed)
        if self.include_main_structures and self.boundary_first and (pressure_pivot is not None or changed_pressure_variant):
            self._prepare_boundary_pairs()
        local_added = self._append_local_pivot(i) if i else False
        if self.include_main_structures and refreshed:
            changed_scales = ({name for name, _ in refreshed}
                              if (self.boundary_first and pressure_pivot is None
                                  and self._pressure_pivot_revision == self._topology_pressure_revision)
                              else None)
            if changed_scales is None:
                self._refresh_structures()
            else:
                self._refresh_structures(changed_scales)
            self._topology_pressure_revision = self._pressure_pivot_revision
        if self.include_main_structures:
            self._maintain_candidates()
        current_date = self.date_strings[-1]
        maturity_due = self.include_main_structures and any(
            candidate.get("needs_initial_eval")
            and candidate.get("family") in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
            and current_date >= (self._candidate_maturity_date(candidate) or "9999-12-31")
            for candidate in self.active.values())
        if self.include_main_structures and (refreshed or pressure_pivot is not None or maturity_due):
            signals = self._evaluate(pressure_pivot)
        else:
            signals = []
        if not signals:
            signals = self._check_cached_breakout()
        if local_added or i >= 1:
            signals += self._check_bull_flag()
        self._advance_flag_lifecycles(i)
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


def recognize_daily_history(bars, code, kind, trade_start=None, progress_callback=None,
                            recognizer="boundary_first", performance=None, tick_size=None,
                            timeframe="1d"):
    """Return patterns and append-only events from the shared OHLCV session.

    The scanner and backtest use this same causal event stream. In particular,
    later pivots or a rolling search window cannot remove an event already
    emitted by an earlier prefix. ``recognizer`` remains a compatibility
    switch for the pair-precomputation optimization; it does not select a
    different recognition algorithm. Weekly callers must pass completed week
    bars and ``timeframe='1w'``.
    """
    if recognizer not in {"v7", "boundary_first"}:
        raise ValueError("recognizer must be 'v7' or 'boundary_first'")
    if timeframe not in {"1d", "1w"}:
        raise ValueError("timeframe must be '1d' or '1w'")
    started = time.perf_counter()
    frame = pd.DataFrame(bars).copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        empty = {"patterns": [], "signals": [], "performance": {}}
        if performance is not None:
            performance.update(empty["performance"])
        return empty
    input_attrs = getattr(bars, "attrs", {})
    frame.attrs.update(input_attrs)
    metadata_tick = tick_size if tick_size is not None else input_attrs.get("quality", {}).get("tick_size")
    tick_size_source = ("explicit_argument" if tick_size is not None else
                        "input_quality" if input_attrs.get("quality", {}).get("tick_size") is not None
                        else "kind_default")
    tick = float(metadata_tick if metadata_tick is not None else (0.001 if kind == "etf" else 0.01))
    provenance = deepcopy(input_attrs.get("provenance", {}) or {})
    provenance.setdefault("data_version", "unknown")
    provenance["tick_size_source"] = tick_size_source
    provenance["tick_size"] = tick
    provided_context = input_attrs.get("recognition_context")
    if provided_context is None:
        provided_context = provenance.get("recognition_context")
    recognition_context = deepcopy(provided_context if provided_context is not None
                                   else recognition_context_snapshot())
    recognition_context.setdefault("recognition_experiment", None)
    recognition_context.setdefault("data_version", provenance["data_version"])
    recognition_context.setdefault("engine_version", ENGINE_VERSION)
    recognition_context.setdefault("source_rule_version", SOURCE_RULE_VERSION)
    recognition_context.setdefault("tick_size", tick)
    recognition_context.setdefault("tick_size_source", tick_size_source)
    if "online_history_scope" in input_attrs:
        recognition_context["online_history_scope"] = deepcopy(input_attrs["online_history_scope"])
    provenance["recognition_context"] = recognition_context
    if progress_callback is not None:
        progress_callback({"phase": "recognition", "bars_completed": 0,
                           "bars_total": len(frame), "current_date": None})
    first = pd.Timestamp(trade_start).date().isoformat() if trade_start is not None else None
    dates = [str(pd.Timestamp(value).date()) for value in frame.date]
    session_started = time.perf_counter()
    session = DailyRecognitionSession(
        code, kind, tick_size=tick, total_bars=len(frame), progress_callback=progress_callback,
        boundary_first=(recognizer == "boundary_first"), timeframe=timeframe,
        provenance=provenance, prior_price_context=input_attrs.get("prior_price_context"),
        coordinate_offset=input_attrs.get("chart_offset", 0))
    signals = []
    records = frame.to_dict("records")
    for row in records:
        for signal in session.append_bar(row):
            if first is None or signal["signal_date"] >= first:
                signals.append(signal)
    session_seconds = time.perf_counter() - session_started
    date_index = {value: index for index, value in enumerate(dates)}
    for signal in signals:
        next_index = date_index.get(signal["signal_date"], len(dates)) + 1
        signal["entry_date"] = dates[next_index] if next_index < len(dates) else None

    patterns_by_instance = {}

    def remember_pattern(pattern):
        if pattern is None:
            return
        lifecycle = session._attach_session_lifecycle(pattern)
        identity = lifecycle.get("instance_id", pattern.pattern_id)
        current = patterns_by_instance.get(identity)
        if current is None:
            patterns_by_instance[identity] = pattern
            return
        state_rank = {"forming": 0, "confirmed": 1, "expired": 2,
                      "target_reached": 3, "invalidated": 4}
        if (state_rank.get(pattern.state, -1), pattern.last_observed_date or "") > (
                state_rank.get(current.state, -1), current.last_observed_date or ""):
            patterns_by_instance[identity] = pattern

    for candidate in session.active.values():
        pattern = candidate.get("pattern")
        if pattern is not None:
            if pattern.state not in {"invalidated", "expired", "target_reached"}:
                pattern.last_observed_date = dates[-1]
            remember_pattern(deepcopy(pattern))
    for pattern in session.flag_formations.values():
        if pattern.pattern_id not in session.flag_instances:
            if pattern.state not in {"invalidated", "expired", "target_reached"}:
                pattern.last_observed_date = dates[-1]
            remember_pattern(deepcopy(pattern))
    for pattern in session.flag_instances.values():
        remember_pattern(deepcopy(pattern))
    for pattern in session.retired_patterns.values():
        remember_pattern(deepcopy(pattern))
    for event in session.emitted_events:
        for payload in event.get("patterns", [event.get("pattern")]):
            if payload:
                remember_pattern(Pattern(**payload))
    patterns = sorted(patterns_by_instance.values(),
                      key=lambda pattern: (pattern.formed_at, pattern.pattern_type, pattern.pattern_id),
                      reverse=True)
    elapsed = time.perf_counter() - started
    final_performance = {**session.performance,
                         "confirmed_instances": sum(len(item.get("pattern_ids", []))
                                                     for item in signals),
                         "flag_signals": sum("bull_flag" in item.get("pattern_types", [])
                                             for item in signals),
                         "session_seconds": session_seconds,
                         "recognition_seconds": elapsed,
                         "requested_recognizer": recognizer,
                         "effective_recognizer": ENGINE_VERSION,
                         "timeframe": timeframe,
                         "engine_version": ENGINE_VERSION,
                         "source_rule_version": SOURCE_RULE_VERSION,
                         "data_version": provenance["data_version"],
                         "tick_size": tick,
                         "tick_size_source": tick_size_source}
    if session.neckline_potential_pair_cache is not None:
        final_performance.update(session.neckline_potential_pair_cache.stats)
    if performance is not None:
        performance.update(final_performance)
    if progress_callback is not None:
        progress_callback({"phase": "recognition_complete", "bars_completed": len(frame),
                           "bars_total": len(frame), "current_date": dates[-1],
                           "elapsed_seconds": elapsed})
    previous_atr = float(session.atr_values[-2]) if len(frame) > 1 else np.nan
    return {"patterns": patterns, "signals": signals, "performance": final_performance,
            "latest_bar": {"date": dates[-1], "index": len(frame) - 1,
                           "previous_atr": previous_atr if np.isfinite(previous_atr) else None}}


def recognize_weekly_history(bars, code, kind, progress_callback=None, performance=None,
                             tick_size=None):
    """Recognize complete weekly bars with the same instance/event engine."""
    return recognize_daily_history(bars, code, kind, progress_callback=progress_callback,
                                   performance=performance, tick_size=tick_size, timeframe="1w")


def generate_daily_signals(bars, code, kind, trade_start=None, progress_callback=None,
                           recognizer="boundary_first", performance=None):
    """Compatibility wrapper returning daily confirmation events only."""
    result = recognize_daily_history(bars, code, kind, trade_start=trade_start,
                                     progress_callback=progress_callback,
                                     recognizer=recognizer, performance=performance)
    return result["signals"]


def generate_daily_signals_boundary_first(bars, code, kind, trade_start=None,
                                          progress_callback=None, performance=None):
    """Causal pair-first recognizer used by the production default path."""
    return generate_daily_signals(bars, code, kind, trade_start=trade_start,
                                  progress_callback=progress_callback,
                                  recognizer="boundary_first", performance=performance)
