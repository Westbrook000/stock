"""Causal recognizer for long-cycle major bottoms.

This is additive to the v16 pattern families. It consumes one already-sliced
daily or complete-weekly prefix and does not inspect bars beyond that prefix.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import combinations
from bisect import bisect_left, bisect_right
from typing import Iterable

import numpy as np
import pandas as pd

from .indicators import atr, volume_ratio
from .rules.catalog import load_rule_catalog
from .research_config import current_recognition_experiment
from .schemas import Pattern, Pivot
from .pattern_instances import pattern_identity_metadata, pattern_instance_is_frozen


MAIN_RULES = load_rule_catalog().get("main_structure_extension", {})
MAIN_PATTERN_TYPES = {
    "w_bottom": "w_bottom",
    "inverse_head_shoulders": "inverse_head_shoulders",
    "triangle_bottom": "triangle_bottom",
}
PIVOT_SCALES = tuple(
    (item["name"], float(item["atr_multiple"]), float(item["percent_multiple"]))
    for item in MAIN_RULES.get("pivot_scales", [])
)
MIN_STRUCTURE_MONTHS = int(MAIN_RULES.get("minimum_duration_calendar_months", 6))
NECKLINE_LOOKBACK_YEARS = int(MAIN_RULES.get("neckline_tests", {}).get("lookback_calendar_years", 1))
NECKLINE_TEST_COUNT = int(MAIN_RULES.get("neckline_tests", {}).get("minimum_independent_groups", 2))


def _minimum_neckline_test_count():
    """Return the baseline minimum, raised only inside an explicit experiment."""
    experiment = current_recognition_experiment()
    configured = experiment.minimum_neckline_tests if experiment is not None else None
    return max(NECKLINE_TEST_COUNT, int(configured)) if configured is not None else NECKLINE_TEST_COUNT


def _recognition_experiment_cache_key():
    experiment = current_recognition_experiment()
    return experiment.cache_key() if experiment is not None else None


def _same_neckline_boundary_phase(consumed, candidate, tick_size=0.01):
    """Match a consumed phase only when its frozen line is unchanged.

    Touch tolerances judge price contact; they do not decide whether frozen
    boundaries are the same. A changed neckline is a new boundary revision
    and may independently confirm after its freeze date.
    """
    left_line, right_line = consumed.get("line"), candidate.get("line")
    if not left_line or not right_line:
        return False
    epsilon = 1e-12
    return (abs(float(left_line["slope"]) - float(right_line["slope"])) <= epsilon
            and abs(float(left_line["intercept"]) - float(right_line["intercept"])) <= epsilon)


def _consumed_neckline_phase_matcher(phases, current_index):
    """Build a fast, tolerance-preserving lookup for already consumed lines."""
    epsilon = 1e-12
    lines = set()
    for phase in phases:
        consumed_index = phase.get("consumed_index")
        if (consumed_index is not None and int(consumed_index) >= int(current_index)):
            continue
        if not phase.get("selected", True):
            continue
        line = phase.get("line")
        if line:
            lines.add((float(line["slope"]), float(line["intercept"])))
    ordered = sorted(lines)

    def matches(line):
        if not line:
            return False
        slope = float(line["slope"])
        intercept = float(line["intercept"])
        start = bisect_left(ordered, (slope - epsilon, float("-inf")))
        stop = bisect_right(ordered, (slope + epsilon, float("inf")))
        return any(abs(existing_intercept - intercept) <= epsilon
                   for _, existing_intercept in ordered[start:stop])

    return matches


def _failed_premature_neckline_phase(phases, close_values, index):
    """Return a pre-maturity breakout phase after price closes back below it."""
    for phase in phases:
        if phase.get("reason") != "breakout_before_minimum_structure_duration":
            continue
        consumed_index = phase.get("consumed_index")
        line = phase.get("line")
        if consumed_index is None or line is None or int(consumed_index) >= index:
            continue
        if float(close_values[index]) < _at(line, index):
            return phase
    return None


def _neckline_candidate_reversal_atr_multiple():
    return float(MAIN_RULES.get("neckline_tests", {}).get("candidate_reversal_atr_multiple", 2.0))


def _neckline_candidate_scale():
    return f"neckline_test_{_neckline_candidate_reversal_atr_multiple():g}ATR"


def _neckline_test_tolerance_at(index, atr_values, tick_size):
    rules = MAIN_RULES.get("neckline_tests", {})
    known_atr = atr_values[index] if 0 <= int(index) < len(atr_values) else np.nan
    return max(float(rules.get("minimum_tolerance_ticks", 2)) * float(tick_size),
               float(rules.get("tolerance_atr", 0.5)) * float(known_atr)
               if np.isfinite(known_atr) else 0.0)


def _w_rebound_peak_associated(low_pivots, tests, line, high_values, atr_values, tick_size):
    """Require each W rebound peak to be a touch of its selected resistance."""
    if len(low_pivots) < 2:
        return False
    highs = np.asarray(high_values, dtype=float)
    for left, right in zip(low_pivots, low_pivots[1:]):
        start, end = int(left.index) + 1, int(right.index)
        if start >= end or end > len(highs):
            return False
        peak_index = start + int(np.nanargmax(highs[start:end]))
        peak_price = float(highs[peak_index])
        tolerance = _neckline_test_tolerance_at(peak_index, atr_values, tick_size)
        if abs(_at(line, peak_index) - peak_price) <= tolerance + 1e-12:
            continue
        # A selected independent resistance touch within this rebound interval
        # can identify a valid peak even when the raw daily high differs by a
        # small amount from the pivot's reported level.
        associated_test = any(
            int(test.get("pivot_index", -1)) == peak_index
            and abs(float(test.get("level", peak_price)) - peak_price)
                <= max(tolerance, float(test.get("tolerance", 0.0))) + 1e-12
            and abs(_at(line, peak_index) - float(test.get("level", peak_price)))
                <= max(tolerance, float(test.get("tolerance", 0.0))) + 1e-12
            for test in tests if isinstance(test, dict))
        if not associated_test:
            return False
    return True


def _date_text(dates, index, date_strings=None):
    """Format a bar date without repeatedly boxing a Timestamp from an index."""
    index = int(index)
    return date_strings[index] if date_strings is not None else str(dates[index].date())


def detect_multiscale_pivots(bars: pd.DataFrame, timeframe="1d", tick_size=0.01):
    """Return causally frozen swing pivots at three ATR-and-percent scales.

    An extreme is published only when a later close reverses by the scale
    threshold. A still-forming extreme at the right edge is never published.
    """
    if bars.empty:
        return {name: [] for name, _, _ in PIVOT_SCALES}
    if len(bars) < 3:
        return {name: [] for name, _, _ in PIVOT_SCALES}
    frame = bars.reset_index(drop=True)
    atr_values = atr(frame, period=14).to_numpy(dtype=float)
    highs = pd.to_numeric(frame.high, errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame.low, errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    date_strings = dates.strftime("%Y-%m-%d").tolist()
    output = {}
    for scale_name, atr_multiple, percent_multiple in PIVOT_SCALES:
        def threshold(index, price):
            scale = atr_values[index]
            if not np.isfinite(scale) or scale <= 0 or not np.isfinite(price) or price <= 0:
                return np.nan
            return max(atr_multiple * float(scale), percent_multiple * abs(float(price)), 2.0 * float(tick_size))

        pivots = []
        high_index = 0
        low_index = 0
        direction = 0
        for index in range(1, len(frame)):
            if not np.isfinite(highs[index]) or not np.isfinite(lows[index]) or not np.isfinite(closes[index]):
                continue
            if highs[index] > highs[high_index]:
                high_index = index
            if lows[index] < lows[low_index]:
                low_index = index
            if direction == 0:
                down_threshold = threshold(high_index, highs[high_index])
                up_threshold = threshold(low_index, lows[low_index])
                down_move = highs[high_index] - closes[index]
                up_move = closes[index] - lows[low_index]
                down_ready = np.isfinite(down_threshold) and down_move >= down_threshold
                up_ready = np.isfinite(up_threshold) and up_move >= up_threshold
                if down_ready and (not up_ready or down_move / down_threshold >= up_move / up_threshold):
                    pivots.append(Pivot(high_index, _date_text(dates, high_index, date_strings), float(highs[high_index]),
                                        "high", index, _date_text(dates, index, date_strings),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction, low_index = -1, index
                elif up_ready:
                    pivots.append(Pivot(low_index, _date_text(dates, low_index, date_strings), float(lows[low_index]),
                                        "low", index, _date_text(dates, index, date_strings),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction, high_index = 1, index
            elif direction > 0:
                reversal = threshold(high_index, highs[high_index])
                if np.isfinite(reversal) and highs[index] > highs[high_index]:
                    high_index = index
                    reversal = threshold(high_index, highs[high_index])
                if np.isfinite(reversal) and highs[high_index] - closes[index] >= reversal:
                    pivots.append(Pivot(high_index, _date_text(dates, high_index, date_strings), float(highs[high_index]),
                                        "high", index, _date_text(dates, index, date_strings),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction = -1
                    low_index = index
            else:
                reversal = threshold(low_index, lows[low_index])
                if lows[index] < lows[low_index]:
                    low_index = index
                    reversal = threshold(low_index, lows[low_index])
                if np.isfinite(reversal) and closes[index] - lows[low_index] >= reversal:
                    pivots.append(Pivot(low_index, _date_text(dates, low_index, date_strings), float(lows[low_index]),
                                        "low", index, _date_text(dates, index, date_strings),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction = 1
                    high_index = index
        output[scale_name] = pivots
    return output


def _line(p1, p2):
    if p1[0] == p2[0]:
        return {"slope": 0.0, "intercept": float(p1[1])}
    slope = (float(p2[1]) - float(p1[1])) / (int(p2[0]) - int(p1[0]))
    return {"slope": float(slope), "intercept": float(p1[1]) - slope * int(p1[0])}


def _at(line, index):
    return float(line["slope"]) * index + float(line["intercept"])


def _neckline_decline_check(family, line, pressure_start, observation_index):
    """Measure fall over the effective pressure segment, including its extension."""
    start, end = _at(line, pressure_start), _at(line, observation_index)
    decline = (start - end) / start if start > 0 else float('inf')
    limited = family in {"w_bottom", "inverse_head_shoulders"}
    maximum = float(MAIN_RULES.get("neckline_slope", {}).get(
        "maximum_cumulative_relative_decline_w_ihs", .20))
    return {"passed": start > 0 and end > 0 and (not limited or decline <= maximum + 1e-12),
            "relative_decline": float(decline), "maximum_decline": maximum if limited else None,
            "start_index": int(pressure_start), "end_index": int(observation_index)}


def _neckline_formation_coverage(high_values, atr_values, line, start, end, tick_size, cache=None,
                                 tolerances=None):
    """Contain every formation wick; subsequent breakout attempts are separate."""
    start, end = int(start), int(end)
    if start < 0 or end >= len(high_values) or start > end:
        return {"passed": False, "reason": "formation_range_unavailable"}
    rules = MAIN_RULES.get("neckline_tests", {})
    if tolerances is None:
        tolerances = np.maximum(float(rules.get("minimum_tolerance_ticks", 2)) * tick_size,
            np.where(np.isfinite(atr_values), float(rules.get("tolerance_atr", .5)) * np.asarray(atr_values), 0.))
    excess, worst = _line_coverage_prefixes(high_values, tolerances,
        line["slope"], line["intercept"], start, [end], cache=cache)[end]
    return {"passed": excess <= 1e-10, "start_index": start, "end_index": end,
            "maximum_excess": float(excess), "worst_index": worst,
            "reason": "formation_covered" if excess <= 1e-10 else "formation_high_crosses_neckline"}


def _neckline_structure_anchor_coverage(line, anchors, atr_values, tick_size):
    """Check only frozen structural pivots against a resistance boundary."""
    rules = MAIN_RULES.get("neckline_tests", {})
    minimum_ticks = float(rules.get("minimum_tolerance_ticks", 2)) * float(tick_size)
    tolerance_atr = float(rules.get("tolerance_atr", 0.5))
    excess, worst_index = 0.0, None
    for anchor in anchors:
        known_at = int(anchor.confirmed_at)
        index = int(anchor.index)
        if (known_at < 0 or known_at >= len(atr_values) or index < 0
                or known_at < index):
            return {"passed": False, "reason": "structure_anchor_unavailable",
                    "maximum_excess": None, "worst_index": index}
        known_atr = float(atr_values[known_at])
        tolerance = max(minimum_ticks,
                        tolerance_atr * known_atr if np.isfinite(known_atr) else 0.0)
        anchor_excess = float(anchor.price) - _at(line, index) - tolerance
        if anchor_excess > excess:
            excess, worst_index = anchor_excess, index
    return {"passed": excess <= 1e-10, "maximum_excess": float(max(0.0, excess)),
            "worst_index": worst_index,
            "reason": "structure_anchors_covered" if excess <= 1e-10
            else "structure_anchor_above_neckline"}


def _qualify_neckline_slope(line, start_index, end_index):
    """Reject rising resistance above the configured relative two-test limit.

    Small positive drift is treated as one horizontal resistance zone. The
    calculation is relative to the first test level, so nominal price does not
    affect the threshold.
    """
    rules = MAIN_RULES.get("neckline_slope", {})
    maximum_rise = float(rules.get("maximum_cumulative_relative_rise", 0.05))
    start_level = _at(line, start_index)
    end_level = _at(line, end_index)
    if start_level <= 0:
        return None
    relative_rise = (end_level - start_level) / start_level
    if relative_rise > maximum_rise:
        return None
    if relative_rise > 0:
        return {"slope": 0.0, "intercept": (start_level + end_level) / 2.0}
    return line


def _triangle_apex_index(upper, lower):
    """Return the upper/lower boundary intersection for a converging triangle."""
    denominator = float(lower["slope"]) - float(upper["slope"])
    if denominator <= 0:
        return None
    return (float(upper["intercept"]) - float(lower["intercept"])) / denominator


def _detect_neckline_test_pivots(bars, atr_values, tick_size):
    """Freeze each pressure candidate after its configured ATR reversal."""
    frame = bars.reset_index(drop=True)
    highs = pd.to_numeric(frame.high, errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    reversal_multiple = _neckline_candidate_reversal_atr_multiple()
    candidate_scale = _neckline_candidate_scale()
    result = []
    candidate_index = None
    for index in range(1, len(frame)):
        if not np.isfinite(highs[index]) or not np.isfinite(closes[index]):
            continue
        if candidate_index is None:
            if highs[index] >= highs[index - 1]:
                candidate_index = index
            continue
        if highs[index] > highs[candidate_index]:
            candidate_index = index
            continue
        known_atr = atr_values[candidate_index]
        if not np.isfinite(known_atr) or known_atr <= 0:
            candidate_index = None
            continue
        reversal_distance = max(reversal_multiple * float(known_atr), 2.0 * float(tick_size))
        if closes[index] <= highs[candidate_index] - reversal_distance:
            result.append(Pivot(candidate_index, str(dates[candidate_index].date()),
                                float(highs[candidate_index]), "high", index,
                                str(dates[index].date()), candidate_scale))
            candidate_index = None
    return result


def _tick_size(bars, requested):
    if requested is not None:
        return max(float(requested), 0.000001)
    return max(float(bars.attrs.get("quality", {}).get("tick_size", 0.01)), 0.000001)


def _build_range_minimum_index(values):
    """Build an immutable sparse table for exact O(1) inclusive range minima."""
    values = np.asarray(values, dtype=float)
    levels = [values]
    width = 1
    while width * 2 <= len(values):
        previous = levels[-1]
        levels.append(np.minimum(previous[:-width], previous[width:]))
        width *= 2
    return tuple(levels)


def _range_minimum(index, start, end):
    """Return the minimum on the half-open interval [start, end)."""
    length = int(end) - int(start)
    if length <= 0:
        return None
    level = length.bit_length() - 1
    width = 1 << level
    return float(min(index[level][start], index[level][end - width]))


def _touch_interval_minima(touch_indices, low_indices, low_prices, low_range_index, previous=None):
    """Reuse minima between unchanged consecutive pressure touches on append-only pivots."""
    touches = tuple(int(index) for index in touch_indices)
    low_signature = tuple((int(index), float(price))
                          for index, price in zip(low_indices, low_prices))
    minima = []
    old_touch_count = 0
    changed_low_indexes = []
    if previous is not None:
        old_touches = tuple(previous.get("touch_indices", ()))
        old_lows = dict(previous.get("low_signature", ()))
        current_lows = dict(low_signature)
        stable_lows = all(index in current_lows and current_lows[index] <= price
                          for index, price in old_lows.items())
        if (touches[:len(old_touches)] == old_touches and stable_lows
                and len(previous.get("interval_minima", ())) == max(0, len(old_touches) - 1)):
            old_touch_count = len(old_touches)
            minima = list(previous["interval_minima"])
            changed_low_indexes = [index for index, price in low_signature
                                   if index not in old_lows or price < old_lows[index]]
    first_new_interval = max(0, old_touch_count - 1)
    for position in range(first_new_interval, len(touches) - 1):
        left, right = touches[position], touches[position + 1]
        low_start = bisect_right(low_indices, left)
        low_end = bisect_left(low_indices, right)
        value = _range_minimum(low_range_index, low_start, low_end)
        minima.append(float(value) if value is not None else np.inf)
    for index in changed_low_indexes:
        position = bisect_right(touches, index) - 1
        if (position < first_new_interval and 0 <= position < len(minima)
                and touches[position] < index < touches[position + 1]):
            value = dict(low_signature)[index]
            minima[position] = min(minima[position], value)
    state = {"touch_indices": touches, "interval_minima": tuple(minima),
             "low_signature": low_signature}
    return tuple(minima), state


def _build_touch_gap_minimum_index(touch_indexes, interval_minima, low_by_index):
    """Build an RMQ index over open intervals plus lows on internal touch bars."""
    values = []
    for position, minimum in enumerate(interval_minima):
        values.append(float(minimum))
        internal_touch = position + 1
        if internal_touch < len(touch_indexes) - 1:
            values.append(float(low_by_index.get(int(touch_indexes[internal_touch]), np.inf)))
    return _build_range_minimum_index(values)


def _touch_gap_minimum(index, left_position, right_position):
    """Return the low minimum strictly between two pressure-touch positions."""
    left_position, right_position = int(left_position), int(right_position)
    if right_position <= left_position:
        return None
    return _range_minimum(index, 2 * left_position, 2 * right_position - 1)


def _cached_touch_gap_minimum(index, left_position, right_position, cache):
    """Memoize immutable pressure-touch gap queries within one neckline search."""
    left_position, right_position = int(left_position), int(right_position)
    key = (left_position, right_position)
    if key not in cache:
        cache[key] = _touch_gap_minimum(index, left_position, right_position)
    return cache[key]


def _line_coverage_prefixes(high_values, tolerances, slope, intercept, start, coverage_ends,
                            cache=None):
    """Check requested prefixes, extending cached append-only line scans when available."""
    ends = sorted({int(end) for end in coverage_ends})
    if not ends:
        return {}
    first, last = int(start), ends[-1]
    if first < 0 or last >= len(high_values) or first > last:
        return {end: (0.0, None) for end in ends}
    line_key = (first, float(slope), float(intercept))
    checkpoints = None
    if cache is not None:
        entry = cache.get(line_key)
        if entry is None:
            entry = {"checkpoints": {}}
            cache[line_key] = entry
        elif hasattr(cache, "move_to_end"):
            cache.move_to_end(line_key)
        checkpoints = entry["checkpoints"]
    result = {}
    for end in ends:
        cached = checkpoints.get(end) if checkpoints is not None else None
        if cached is not None:
            maximum, worst_index = cached
        else:
            prior_ends = [value for value in checkpoints if value < end] if checkpoints is not None else []
            if prior_ends:
                prior_end = max(prior_ends)
                maximum, worst_index = checkpoints[prior_end]
                scan_start = prior_end + 1
            else:
                maximum, worst_index = -np.inf, None
                scan_start = first
            indexes = np.arange(scan_start, end + 1, dtype=int)
            highs = np.asarray(high_values, dtype=float)[scan_start:end + 1]
            allowances = np.asarray(tolerances, dtype=float)[scan_start:end + 1]
            excesses = highs - (float(slope) * indexes + float(intercept)) - allowances
            excesses = np.where(np.isfinite(highs) & np.isfinite(allowances), excesses, -np.inf)
            if len(excesses):
                local_position = int(np.argmax(excesses))
                local_maximum = float(excesses[local_position])
                if local_maximum > maximum:
                    maximum = local_maximum
                    worst_index = scan_start + local_position
            if checkpoints is not None:
                checkpoints[end] = (float(maximum), worst_index)
        result[end] = (max(0.0, float(maximum)), worst_index)
    if cache is not None and hasattr(cache, "popitem"):
        while len(cache) > 512:
            try:
                cache.popitem(last=False)
            except TypeError:
                break
    return result


class _PotentialNecklinePairCache:
    """Precompute necessary pair geometry and anchor-span coverage before matching shapes."""

    def __init__(self):
        self.entries = {}
        self.pivot_keys = set()
        self.eligible_by_left = {}
        self.anchor_eligible_by_left = {}
        self.stats = {"boundary_pairs_generated": 0, "boundary_pairs_reused": 0,
                      "coverage_checks": 0, "coverage_reused": 0,
                      "pair_index_fallbacks": 0}

    @staticmethod
    def _pivot_key(pivot):
        return (int(pivot.index), float(pivot.price), int(pivot.confirmed_at))

    def add(self, pivots, atr_values, high_values, coverage_tolerances, tick_size,
            timeframe, minimum_index=0):
        rules = MAIN_RULES.get("neckline_tests", {})
        weekly = timeframe in {"1w", "weekly", "week"}
        minimum_gap = int(rules.get(
            "minimum_separation_weekly_bars" if weekly else "minimum_separation_daily_sessions",
            2 if weekly else 10))
        minimum_ticks = float(rules.get("minimum_tolerance_ticks", 2)) * tick_size
        tolerance_atr = float(rules.get("tolerance_atr", 0.5))
        candidate_scale = _neckline_candidate_scale()
        pressure = sorted((pivot for pivot in pivots
                           if pivot.kind == "high" and candidate_scale in pivot.scale
                           and pivot.index >= int(minimum_index)
                           and pivot.confirmed_at < len(high_values)),
                          key=lambda pivot: (pivot.index, pivot.scale))
        by_key = {self._pivot_key(pivot): pivot for pivot in pressure}
        current_keys = set(by_key)
        expired = self.pivot_keys - current_keys
        if expired:
            self.entries = {key: value for key, value in self.entries.items()
                            if key[0] not in expired and key[1] not in expired}
            self.anchor_eligible_by_left = {}
            self.eligible_by_left = {}
            for key, entry in self.entries.items():
                if entry["line"] is not None and entry["anchors_within_tolerance"]:
                    self.anchor_eligible_by_left.setdefault(key[0], []).append(key[1])
                if (entry["line"] is not None and entry["anchors_within_tolerance"]
                        and entry["coverage_passed"] is True):
                    self.eligible_by_left.setdefault(key[0], []).append(key[1])
        new_pivots = [by_key[key] for key in current_keys - self.pivot_keys]
        self.pivot_keys = current_keys
        for pivot in new_pivots:
            for other in pressure:
                if other.index == pivot.index:
                    continue
                left, right = (pivot, other) if pivot.index < other.index else (other, pivot)
                if right.index - left.index < minimum_gap:
                    continue
                key = (self._pivot_key(left), self._pivot_key(right))
                if key in self.entries:
                    continue
                self.stats["boundary_pairs_generated"] += 1
                base = _line((left.index, left.price), (right.index, right.price))
                line = _qualify_neckline_slope(base, left.index, right.index)
                entry = {"line": line, "coverage_end": max(left.confirmed_at, right.confirmed_at),
                         "anchors_within_tolerance": False, "coverage_passed": None}
                if line is not None:
                    def touch_tolerance(pivot):
                        known_atr = atr_values[pivot.confirmed_at]
                        return max(minimum_ticks, tolerance_atr * float(known_atr)
                                   if np.isfinite(known_atr) else 0.0)

                    entry["anchors_within_tolerance"] = all(
                        abs(float(pivot.price) - _at(line, pivot.index)) <= touch_tolerance(pivot)
                        for pivot in (left, right))
                if line is not None and entry["anchors_within_tolerance"]:
                    self.stats["coverage_checks"] += 1
                    excess, _ = _line_coverage_prefixes(
                        high_values, coverage_tolerances, line["slope"], line["intercept"],
                        left.index, [entry["coverage_end"]])[entry["coverage_end"]]
                    entry["coverage_passed"] = excess <= 1e-12
                self.entries[key] = entry
                if line is not None and entry["anchors_within_tolerance"]:
                    self.anchor_eligible_by_left.setdefault(key[0], []).append(key[1])
                if (line is not None and entry["anchors_within_tolerance"]
                        and entry["coverage_passed"] is True):
                    self.eligible_by_left.setdefault(key[0], []).append(key[1])

    def prune(self, minimum_index):
        minimum_index = int(minimum_index)
        self.pivot_keys = {key for key in self.pivot_keys if key[0] >= minimum_index}
        self.entries = {key: entry for key, entry in self.entries.items()
                        if key[0][0] >= minimum_index}
        self.eligible_by_left = {}
        self.anchor_eligible_by_left = {}
        for key, entry in self.entries.items():
            if entry["line"] is not None and entry["anchors_within_tolerance"]:
                self.anchor_eligible_by_left.setdefault(key[0], []).append(key[1])
            if (entry["line"] is not None and entry["anchors_within_tolerance"]
                    and entry["coverage_passed"] is True):
                self.eligible_by_left.setdefault(key[0], []).append(key[1])

    def matching_pairs(self, pivots, minimum_gap, anchor_only=False):
        """List prequalified pairs in the same order as combinations(pivots, 2)."""
        positions = {self._pivot_key(pivot): position for position, pivot in enumerate(pivots)}
        known = self.pivot_keys
        pairs = []
        for left_position, left in enumerate(pivots):
            left_key = self._pivot_key(left)
            if anchor_only:
                right_keys = self.anchor_eligible_by_left.get(left_key, ())
            else:
                right_keys = self.eligible_by_left.get(left_key, ())
            for right_key in right_keys:
                right_position = positions.get(right_key)
                if right_position is None or right_position <= left_position:
                    continue
                pairs.append((left_position, right_position, pivots[right_position]))
                self.stats["boundary_pairs_reused"] += 1
                self.stats["coverage_reused"] += 1
        unknown = {key for key in positions if key not in known}
        if unknown:
            for left_position, left in enumerate(pivots):
                for right_position in range(left_position + 1, len(pivots)):
                    right = pivots[right_position]
                    if (self._pivot_key(left) not in unknown and self._pivot_key(right) not in unknown):
                        continue
                    if right.index - left.index >= minimum_gap:
                        pairs.append((left_position, right_position, right))
            self.stats["pair_index_fallbacks"] += sum(
                1 for left_position, left in enumerate(pivots)
                for right in pivots[left_position + 1:]
                if (self._pivot_key(left) in unknown or self._pivot_key(right) in unknown)
                and right.index - left.index >= minimum_gap)
        pairs.sort(key=lambda item: (item[0], item[1]))
        return [(pivots[left_position], right)
                for left_position, _, right in pairs]

    def get(self, left, right):
        key = (self._pivot_key(left), self._pivot_key(right))
        entry = self.entries.get(key)
        if entry is None:
            return None
        self.stats["boundary_pairs_reused"] += 1
        self.stats["coverage_reused"] += int(entry["coverage_passed"] is not None)
        return entry


def _independent_neckline_tests(pivots, first_trough, last_anchor, atr_values, dates, tick_size, timeframe,
                                lower_index=None, upper_index=None, minimum_level=None,
                                maximum_level=None, bars=None,
                                return_candidates=False, diagnostics=None,
                                trusted_readonly_cache=False, coverage_mode="full_interval",
                                structure_anchors=None):
    """Select a causal pressure boundary by independent tests and structural coverage.

    Candidate lines are anchored by pairs of confirmed touches, then collect
    every additional independent touch they explain. ``full_interval`` requires
    all intervening highs to stay below the line. ``structure_anchors_only``
    instead checks selected pattern anchors when they become knowable; ordinary
    highs between anchors do not veto the structural pairing.
    """
    if coverage_mode not in {"full_interval", "structure_anchors_only"}:
        raise ValueError(f"unsupported neckline coverage mode: {coverage_mode}")
    structure_anchors = tuple(structure_anchors or ())
    pair_memo = bars.attrs.get("neckline_pair_cache") if bars is not None else None
    potential_pair_cache = (bars.attrs.get("neckline_potential_pair_cache")
                            if bars is not None else None)
    date_strings = bars.attrs.get("date_strings") if bars is not None else None
    minimum_test_count = _minimum_neckline_test_count()
    experiment_cache_key = _recognition_experiment_cache_key()
    if date_strings is None or len(date_strings) != len(dates):
        date_strings = dates.strftime("%Y-%m-%d").tolist()
    rejection_seen = set()
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"rejection_counts": {}, "rejection_count_by_date": {},
                            "rejected_candidates": []})

    def reject(reason, candidate, known_date=None):
        if diagnostics is None:
            return
        sample = dict(candidate)
        if known_date is not None:
            sample["known_date"] = str(pd.Timestamp(known_date).date())
        if pair_memo is not None:
            key = (reason, repr(sample))
            if key in rejection_seen:
                return
            rejection_seen.add(key)
        counts = diagnostics["rejection_counts"]
        counts[reason] = int(counts.get(reason, 0)) + 1
        if known_date is not None:
            date_counts = diagnostics["rejection_count_by_date"].setdefault(reason, {})
            date_key = str(pd.Timestamp(known_date).date())
            date_counts[date_key] = int(date_counts.get(date_key, 0)) + 1
        items = diagnostics["rejected_candidates"]
        items.append(dict(sample, reason=reason))
        if len(items) > 128:
            del items[128:]

    date_values = dates.values
    lookback = pd.Timestamp(date_values[first_trough.index]) - pd.DateOffset(years=NECKLINE_LOOKBACK_YEARS)
    lookback_value = lookback.to_datetime64()
    candidate_scale = _neckline_candidate_scale()
    candidate_pool_cache = bars.attrs.get("neckline_pressure_candidate_pool") if bars is not None else None
    if (candidate_pool_cache is not None and candidate_pool_cache.get("source") is pivots
            and candidate_pool_cache.get("scale") == candidate_scale):
        pressure_candidate_pool = candidate_pool_cache["pivots"]
    else:
        # This input list is shared by every shape evaluation in a causal prefix.
        # Cache only the broad pressure-pivot pool; per-structure date, index,
        # and level restrictions remain applied independently below.
        pressure_candidate_pool = [pivot for pivot in pivots
                                   if pivot.kind == "high" and candidate_scale in pivot.scale
                                   and pivot.confirmed_at < len(dates)]
        if bars is not None:
            candidate_indices = [pivot.index for pivot in pressure_candidate_pool]
            bars.attrs["neckline_pressure_candidate_pool"] = {
                "source": pivots, "scale": candidate_scale, "pivots": pressure_candidate_pool,
                "indices": candidate_indices,
                "indices_sorted": all(left <= right for left, right in zip(
                    candidate_indices, candidate_indices[1:])),
            }
    lookback_index = int(np.searchsorted(date_values, lookback_value, side="left"))
    pool_indices = (candidate_pool_cache.get("indices")
                    if candidate_pool_cache is not None
                    and candidate_pool_cache.get("source") is pivots
                    and candidate_pool_cache.get("scale") == candidate_scale else None)
    if pool_indices is None:
        pool_indices = [pivot.index for pivot in pressure_candidate_pool]
    pool_is_sorted = (candidate_pool_cache.get("indices_sorted")
                      if candidate_pool_cache is not None
                      and candidate_pool_cache.get("source") is pivots
                      and candidate_pool_cache.get("scale") == candidate_scale else None)
    if pool_is_sorted is None:
        pool_is_sorted = all(left <= right for left, right in zip(pool_indices, pool_indices[1:]))
    if pool_is_sorted:
        first_index = max(lookback_index, int(lower_index) if lower_index is not None else 0)
        start = bisect_left(pool_indices, first_index)
        end = (bisect_right(pool_indices, int(upper_index))
               if upper_index is not None else len(pool_indices))
        candidates = [pivot for pivot in pressure_candidate_pool[start:end]
                      if maximum_level is None or float(pivot.price) <= float(maximum_level)]
    else:
        candidates = [pivot for pivot in pressure_candidate_pool
                      if date_values[pivot.index] >= lookback_value
                      and (lower_index is None or pivot.index >= lower_index)
                      and (upper_index is None or pivot.index <= upper_index)
                      and (maximum_level is None or float(pivot.price) <= float(maximum_level))]
    neckline_rules = MAIN_RULES.get("neckline_tests", {})
    weekly = timeframe in {"1w", "weekly", "week"}
    minimum_gap = int(neckline_rules.get(
        "minimum_separation_weekly_bars" if weekly else "minimum_separation_daily_sessions",
        2 if weekly else 10))
    tolerance_atr = float(neckline_rules.get("tolerance_atr", 0.5))
    minimum_ticks = int(neckline_rules.get("minimum_tolerance_ticks", 2))
    shared_vectors = (bars.attrs.get("neckline_pivot_vectors") if bars is not None else None)
    if shared_vectors is not None and shared_vectors.get("source") is pivots:
        high_by_index = shared_vectors["high_by_index"]
        low_by_index = shared_vectors["low_by_index"]
        low_indices = shared_vectors["low_indices"]
        low_prices = shared_vectors["low_prices"]
    else:
        high_by_index = {}
        low_by_index = {}
        for pivot in pivots:
            target = high_by_index if pivot.kind == "high" else low_by_index if pivot.kind == "low" else None
            if target is not None:
                comparator = max if pivot.kind == "high" else min
                target[pivot.index] = comparator(
                    target.get(pivot.index, -np.inf if pivot.kind == "high" else np.inf),
                    float(pivot.price))
        low_indices = np.asarray(sorted(low_by_index), dtype=int)
        low_prices = np.asarray([low_by_index[index] for index in low_indices], dtype=float)
        low_range_index = _build_range_minimum_index(low_prices)
        if bars is not None:
            bars.attrs["neckline_pivot_vectors"] = {
                "source": pivots, "high_by_index": high_by_index, "low_by_index": low_by_index,
                "low_indices": low_indices, "low_prices": low_prices,
                "low_range_index": low_range_index,
            }
    low_range_index = (shared_vectors.get("low_range_index") if shared_vectors is not None
                       and shared_vectors.get("source") is pivots else low_range_index)
    # One wick can produce the same local high at several pivot scales. It is
    # one test group, represented by its strongest confirmed scale.
    unique = {}
    def scale_strength(pivot):
        try:
            return float(pivot.scale.split("ATR")[0])
        except (TypeError, ValueError):
            return 0.0
    for pivot in candidates:
        old = unique.get(pivot.index)
        if old is None or scale_strength(pivot) > scale_strength(old):
            unique[pivot.index] = pivot
    candidates = [unique[index] for index in sorted(unique)]
    if len(candidates) < minimum_test_count:
        return []

    candidate_indices = np.asarray([pivot.index for pivot in candidates], dtype=int)
    candidate_positions = {int(index): position for position, index in enumerate(candidate_indices)}
    candidate_prices = np.asarray([float(pivot.price) for pivot in candidates], dtype=float)
    candidate_tolerances = np.asarray([
        max(minimum_ticks * tick_size,
            tolerance_atr * float(atr_values[pivot.confirmed_at])
            if np.isfinite(atr_values[pivot.confirmed_at]) else 0.0)
        for pivot in candidates], dtype=float)
    max_known = min(len(dates) - 1, max(p.confirmed_at for p in candidates))
    candidate_end = int(np.searchsorted(candidate_indices, max_known, side="right"))
    # An append-only session owns this cache; covered OHLC/ATR values are immutable.
    memo = bars.attrs.get("neckline_candidate_cache") if bars is not None else None
    memo_key = None
    interval_minima = []
    current_signature = tuple((p.index, p.price, p.confirmed_at, p.scale) for p in candidates)
    anchor_signature = tuple((int(p.index), float(p.price), int(p.confirmed_at), p.kind)
                             for p in structure_anchors)
    pool_key = (str(dates[first_trough.index]), lower_index,
                upper_index if upper_index is not None and upper_index < max_known else None,
                minimum_level, maximum_level, float(tick_size), timeframe,
                diagnostics is not None, experiment_cache_key, coverage_mode, anchor_signature)
    prior = pair_memo.get(pool_key) if pair_memo is not None else None
    interval_minima, interval_state = _touch_interval_minima(
        candidate_indices, low_indices, low_prices, low_range_index,
        previous=(prior.get("interval_state") if prior is not None else None))
    if memo is not None or pair_memo is not None:
        # Departures depend only on interval minima between pressure touches;
        # a new, higher internal low must not force the same pool to be rebuilt.
        memo_key = (tuple((p.index, p.price, p.confirmed_at, p.scale) for p in candidates),
                    tuple(interval_minima), minimum_level, float(tick_size), timeframe,
                    diagnostics is not None, minimum_test_count, experiment_cache_key,
                    coverage_mode, anchor_signature)
        if memo is not None and memo_key in memo:
            cached, cached_diagnostics = memo.pop(memo_key)
            memo[memo_key] = (cached, cached_diagnostics)
            if diagnostics is not None:
                diagnostics.update(deepcopy(cached_diagnostics))
            # `_make_main_pattern` treats candidate records as read-only and
            # builds its shape-specific filtered copies. Returning the stored
            # object there avoids copying the same large evidence tree for each
            # refresh. Other callers keep the defensive-copy default.
            result = cached if trusted_readonly_cache and return_candidates else deepcopy(cached)
            return result if return_candidates else (result[0]["tests"] if result else [])
    high_values = (bars.attrs.get("neckline_high_values") if bars is not None else None)
    if high_values is None or len(high_values) != len(dates):
        high_values = (pd.to_numeric(bars.high, errors="coerce").to_numpy(dtype=float)
                       if bars is not None else np.asarray([
                       high_by_index.get(index, np.nan) for index in range(len(dates))], dtype=float))
    high_values = np.asarray(high_values, dtype=float)
    minimum_departure = float(neckline_rules.get("minimum_departure_atr", 1.0))
    atr_array = (bars.attrs.get("neckline_atr_values") if bars is not None else None)
    if atr_array is None or len(atr_array) != len(dates):
        atr_array = np.asarray(atr_values, dtype=float)
    coverage_tolerances = (bars.attrs.get("neckline_coverage_tolerances")
                           if bars is not None else None)
    if coverage_tolerances is None or len(coverage_tolerances) != len(dates):
        coverage_tolerances = np.maximum(minimum_ticks * tick_size,
                                        np.where(np.isfinite(atr_array), tolerance_atr * atr_array, 0.0))
    touch_gap_range_index = _build_touch_gap_minimum_index(candidate_indices, interval_minima, low_by_index)
    touch_gap_cache = {}
    lines, pair_lines = {}, {}
    incremental_pair_keys = None
    use_potential_pairs = potential_pair_cache is not None and diagnostics is None
    if pair_memo is not None:
        if prior is not None:
            old_size = len(prior["touch_signature"])
            can_extend = current_signature[:old_size] == prior["touch_signature"]
            if can_extend:
                lines = dict(prior["lines"])
                pair_lines = dict(prior["pair_lines"])
                rejection_seen.update(prior["rejection_seen"])
                if diagnostics is not None:
                    diagnostics.update(deepcopy(prior["diagnostics"]))
                comparable_intervals = min(max(0, old_size - 1), len(interval_minima))
                changed_intervals = [index for index in range(comparable_intervals)
                                     if interval_minima[index] != prior["interval_minima"][index]]
                if changed_intervals:
                    # A newly confirmed lower trough can change whether later
                    # resistance touches are independent. Invalidate all grouped
                    # outputs beginning before the changed interval; their
                    # pressure-line anchor pairs are re-evaluated below.
                    last_changed_interval = max(changed_intervals)
                    invalidated_through = candidates[last_changed_interval].index
                    lines = {signature: value for signature, value in lines.items()
                             if signature[0] > invalidated_through}
                else:
                    last_changed_interval = None
                new_touches = candidates[old_size:]
                by_index = {pivot.index: pivot for pivot in candidates}
                positions = {pivot.index: position for position, pivot in enumerate(candidates)}
                affected_keys = set()
                # Enumerate only pairs that include newly confirmed touches.
                # The previous implementation materialized every historical
                # pair on each append, then discarded almost all of them.
                for left_position, left in enumerate(candidates):
                    right_start = old_size if left_position < old_size else left_position + 1
                    for right in candidates[right_start:]:
                        affected_keys.add((left.index, right.index))
                # An old pair also needs re-grouping when the new touch lies on
                # its cached line. Keep this exact check for alternate lines.
                if new_touches and pair_lines:
                    pair_keys = list(pair_lines)
                    slopes = np.asarray([pair_lines[key]["slope"] for key in pair_keys], dtype=float)
                    intercepts = np.asarray([pair_lines[key]["intercept"] for key in pair_keys], dtype=float)
                    for touch in new_touches:
                        tolerance = max(minimum_ticks * tick_size,
                                        tolerance_atr * float(atr_values[touch.confirmed_at]))
                        deviations = np.abs(float(touch.price) -
                                            (slopes * int(touch.index) + intercepts))
                        affected_keys.update(pair_keys[position]
                                             for position in np.flatnonzero(deviations <= tolerance))
                if last_changed_interval is not None:
                    for pair_key in pair_lines:
                        if positions[pair_key[0]] <= last_changed_interval:
                            affected_keys.add(pair_key)
                incremental_pair_keys = affected_keys
                anchor_pairs = [
                    (by_index[left], by_index[right])
                    for left, right in sorted(
                        affected_keys,
                        key=lambda pair: (positions[pair[0]], positions[pair[1]]),
                    )
                ]
            else:
                anchor_pairs = ([] if use_potential_pairs else list(combinations(candidates, 2)))
        else:
            anchor_pairs = ([] if use_potential_pairs else list(combinations(candidates, 2)))
    else:
        anchor_pairs = ([] if use_potential_pairs else list(combinations(candidates, 2)))
    prechecked_pairs = None
    if use_potential_pairs:
        matching_pairs = potential_pair_cache.matching_pairs(
            candidates, minimum_gap, anchor_only=coverage_mode == "structure_anchors_only")
        if incremental_pair_keys is not None:
            matching_pairs = [pair for pair in matching_pairs
                              if (pair[0].index, pair[1].index) in incremental_pair_keys]
        anchor_pairs = matching_pairs
        prechecked_pairs = {
            (potential_pair_cache._pivot_key(left), potential_pair_cache._pivot_key(right)):
                potential_pair_cache.entries[(potential_pair_cache._pivot_key(left),
                                              potential_pair_cache._pivot_key(right))]
            for left, right in matching_pairs
            if (potential_pair_cache._pivot_key(left), potential_pair_cache._pivot_key(right))
            in potential_pair_cache.entries
        }
    for left, right in anchor_pairs:
        if right.index - left.index < minimum_gap:
            continue
        if minimum_level is not None and min(float(left.price), float(right.price)) < float(minimum_level):
            continue
        if maximum_level is not None and max(float(left.price), float(right.price)) > float(maximum_level):
            continue
        cached_pair = (prechecked_pairs.get((potential_pair_cache._pivot_key(left),
                                             potential_pair_cache._pivot_key(right)))
                       if prechecked_pairs is not None else None)
        if cached_pair is not None:
            if (cached_pair["line"] is None
                    or not cached_pair["anchors_within_tolerance"]
                    or (coverage_mode == "full_interval"
                        and cached_pair["coverage_passed"] is False)):
                continue
            line = cached_pair["line"]
            base_line = line
        else:
            base_line = _line((left.index, left.price), (right.index, right.price))
            line = _qualify_neckline_slope(base_line, left.index, right.index)
        if line is None:
            start_level = _at(base_line, left.index)
            end_level = _at(base_line, right.index)
            relative_rise = ((end_level - start_level) / start_level) if start_level > 0 else None
            reject("slope_limit_exceeded", {
                "anchor_dates": [_date_text(dates, left.index, date_strings),
                                 _date_text(dates, right.index, date_strings)],
                "anchor_levels": [float(left.price), float(right.price)],
                "relative_rise": float(relative_rise) if relative_rise is not None else None,
                "maximum_relative_rise": float(MAIN_RULES.get("neckline_slope", {}).get(
                    "maximum_cumulative_relative_rise", 0.05)),
            }, known_date=dates[right.confirmed_at])
            continue
        pair_lines[(left.index, right.index)] = dict(line)
        # Convert tolerance-band touches into one ordered list of independent
        # groups. An untested high above the line is not part of the fit.
        candidate_start = candidate_positions[left.index]
        positions = np.arange(candidate_start, candidate_end, dtype=int)
        deviations = np.abs(candidate_prices[positions] -
                             (float(line["slope"]) * candidate_indices[positions]
                              + float(line["intercept"])))
        matching_offsets = np.flatnonzero(
            deviations <= candidate_tolerances[positions])
        touch_points = [(candidates[candidate_start + offset],
                         float(candidate_tolerances[candidate_start + offset]),
                         float(deviations[offset])) for offset in matching_offsets]
        if len(touch_points) < minimum_test_count:
            reject("insufficient_independent_tests", {
                "anchor_dates": [_date_text(dates, left.index, date_strings),
                                 _date_text(dates, right.index, date_strings)],
                "test_count": len(touch_points),
            }, known_date=dates[max((item[0].confirmed_at for item in touch_points), default=right.confirmed_at)])
            continue
        # Apply minimum spacing and require an ATR retreat between touches.
        # Save each causal prefix so a touch discovered after a breakout cannot
        # retroactively replace the line that was knowable on that date.
        grouped = []
        eligible_prefixes = []
        for item in touch_points:
            pivot = item[0]
            if grouped:
                previous = grouped[-1][0]
                if pivot.index - previous.index < minimum_gap:
                    continue
                previous_position = candidate_positions[previous.index]
                pivot_position = candidate_positions[pivot.index]
                between_low = _cached_touch_gap_minimum(
                    touch_gap_range_index, previous_position, pivot_position, touch_gap_cache)
                if between_low is None or not np.isfinite(between_low):
                    continue
                departure = min(_at(line, previous.index), _at(line, pivot.index)) - between_low
                previous_atr = atr_values[previous.confirmed_at]
                pivot_atr = atr_values[pivot.confirmed_at]
                reference_atr = max(float(previous_atr) if np.isfinite(previous_atr) else 0.0,
                                    float(pivot_atr) if np.isfinite(pivot_atr) else 0.0)
                if reference_atr <= 0 or departure < minimum_departure * reference_atr:
                    continue
            grouped.append(item)
            if (len(grouped) >= minimum_test_count
                    and grouped[0][0].index == left.index
                    and any(entry[0].index == right.index for entry in grouped)):
                coverage_end = max(p.confirmed_at for p, _, _ in grouped)
                if coverage_mode == "structure_anchors_only" and structure_anchors:
                    coverage_end = max(coverage_end,
                                       max(int(anchor.confirmed_at) for anchor in structure_anchors))
                if coverage_end < len(dates):
                    # Check every causal touch prefix after grouping. One cumulative
                    # maximum per anchor line replaces a full interval scan for each
                    # newly added touch while preserving earlier valid prefixes.
                    eligible_prefixes.append((tuple(grouped), coverage_end))
        if not eligible_prefixes:
            continue
        start = eligible_prefixes[0][0][0][0].index
        coverage_ends = [coverage_end for _, coverage_end in eligible_prefixes]
        if coverage_mode == "full_interval":
            coverage_by_end = _line_coverage_prefixes(
                high_values, coverage_tolerances, line["slope"], line["intercept"], start,
                coverage_ends,
                cache=(bars.attrs.get("neckline_coverage_cache") if bars is not None else None))
        else:
            coverage_by_end = {}
            for coverage_end in coverage_ends:
                if any(int(anchor.confirmed_at) > coverage_end for anchor in structure_anchors):
                    continue
                coverage = _neckline_structure_anchor_coverage(
                    line, structure_anchors, atr_values, tick_size)
                coverage_by_end[int(coverage_end)] = (
                    float(coverage["maximum_excess"] or 0.0), coverage["worst_index"])
        for grouped_prefix, coverage_end in eligible_prefixes:
            if coverage_end not in coverage_by_end:
                continue
            max_excess, worst_index = coverage_by_end[coverage_end]
            if max_excess > 1e-12:
                if diagnostics is not None:
                    reject("whole_interval_high_crossed", {
                        "anchor_dates": [_date_text(dates, left.index, date_strings),
                                         _date_text(dates, right.index, date_strings)],
                        "test_dates": [_date_text(dates, entry[0].index, date_strings)
                                       for entry in grouped_prefix],
                        "coverage_start_date": _date_text(dates, start, date_strings),
                        "coverage_end_date": _date_text(dates, coverage_end, date_strings),
                        "line": dict(line),
                        "crossing_date": (_date_text(dates, worst_index, date_strings)
                                          if worst_index is not None else None),
                        "crossing_high": (float(high_values[worst_index])
                                          if worst_index is not None else None),
                        "crossing_allowance": (float(coverage_tolerances[worst_index])
                                               if worst_index is not None else None),
                        "maximum_excess": max_excess,
                    }, known_date=date_strings[coverage_end])
                continue
            residual = float(np.mean([deviation / tolerance
                                      for _, tolerance, deviation in grouped_prefix]))
            span = grouped_prefix[-1][0].index - grouped_prefix[0][0].index
            signature = tuple(entry[0].index for entry in grouped_prefix)
            score = (-len(grouped_prefix), -span, residual,
                     -max(p.confirmed_at for p, _, _ in grouped_prefix),
                     grouped_prefix[0][0].index, grouped_prefix[-1][0].index)
            candidate = {"line": line, "grouped": list(grouped_prefix), "score": score,
                         "max_excess": max_excess, "residual": residual,
                         "coverage_start": start, "coverage_end": coverage_end,
                         "span": span, "signature": signature}
            current = lines.get(signature)
            if current is None or candidate["score"] < current["score"]:
                lines[signature] = candidate
    if pair_memo is not None:
        pair_memo.pop(pool_key, None)
        if len(pair_memo) >= 128:
            pair_memo.pop(next(iter(pair_memo)))
        pair_memo[pool_key] = {
            "touch_signature": current_signature, "interval_minima": tuple(interval_minima),
            "interval_state": interval_state,
            "lines": dict(lines), "pair_lines": dict(pair_lines),
            "rejection_seen": set(rejection_seen), "diagnostics": deepcopy(diagnostics or {}),
        }
    if not lines:
        return []
    ranked = sorted(lines.values(), key=lambda item: item["score"])

    def format_candidate(chosen, rank):
        selected = chosen["grouped"]
        pair_id = "neckline-candidate-" + "-".join(str(item[0].index) for item in selected)
        result = []
        for position, (pivot, tolerance, deviation) in enumerate(selected):
            adjacent_departures = []
            for pair_pos in {position - 1, position}:
                if 0 <= pair_pos < len(selected) - 1:
                    p0, p1 = selected[pair_pos][0], selected[pair_pos + 1][0]
                    lo = bisect_right(low_indices, p0.index)
                    hi = bisect_left(low_indices, p1.index)
                    if hi > lo:
                        drop = min(_at(chosen["line"], p0.index), _at(chosen["line"], p1.index)) - _range_minimum(
                            low_range_index, lo, hi)
                        a0, a1 = atr_values[p0.confirmed_at], atr_values[p1.confirmed_at]
                        ref = max(float(a0) if np.isfinite(a0) else 0.0,
                                  float(a1) if np.isfinite(a1) else 0.0)
                        if ref > 0:
                            adjacent_departures.append(drop / ref)
            result.append({
                "group_id": f"neck-touch-{pivot.index}",
                "pair_id": pair_id,
                "pivot_index": int(pivot.index),
                "pivot_date": _date_text(dates, pivot.index, date_strings),
                "known_index": int(pivot.confirmed_at),
                "known_date": _date_text(dates, pivot.confirmed_at, date_strings),
                "pivot_scale": pivot.scale,
                "level": float(pivot.price),
                "departure_atr": float(min(adjacent_departures)) if adjacent_departures else 0.0,
                "tolerance": float(tolerance),
                "normalized_residual": float(deviation / tolerance) if tolerance > 0 else 0.0,
                "valid": True,
                "source_pivot_indices": [int(pivot.index)],
                "line": dict(chosen["line"]),
                "selection_rank": rank,
                "selection_score": [float(value) if isinstance(value, (int, float, np.number)) else value
                                    for value in chosen["score"]],
                "selection_reason": "most_independent_tests_then_longest_span_then_lowest_residual_then_latest_test",
                "independent_test_count": len(selected),
                "test_span_bars": chosen["span"],
                "coverage_start_index": chosen["coverage_start"],
                "coverage_start_date": _date_text(dates, chosen["coverage_start"], date_strings),
                "coverage_end_index": chosen["coverage_end"],
                "coverage_end_date": _date_text(dates, chosen["coverage_end"], date_strings),
                "coverage_max_excess": chosen["max_excess"],
                "coverage_mode": coverage_mode,
                "candidate_count": len(lines),
            })
        return {"line": dict(chosen["line"]), "tests": result,
                "score": chosen["score"], "frozen_index": max(
                    item["known_index"] for item in result), "selection_rank": rank}

    formatted = [format_candidate(candidate, rank)
                 for rank, candidate in enumerate(ranked, start=1)]
    if memo is not None:
        if len(memo) >= 128:
            memo.pop(next(iter(memo)))
        cached_formatted = (formatted if trusted_readonly_cache and return_candidates
                            else deepcopy(formatted))
        memo[memo_key] = (cached_formatted, deepcopy(diagnostics or {}))
    if return_candidates:
        return formatted
    return formatted[0]["tests"] if formatted else []


def _pivot_tolerance(pivot, atr_values=None, tick_size=0.01):
    known_atr = (atr_values[pivot.confirmed_at]
                 if atr_values is not None and pivot.confirmed_at < len(atr_values) else np.nan)
    return max(float(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2)) * tick_size,
               float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
               * float(known_atr) if np.isfinite(known_atr) else 0.0)


def _triangle_pivot_vectors(pivots, atr_values=None, tick_size=0.01):
    highs, lows = {}, {}
    for pivot in pivots:
        if pivot.kind == "high":
            highs[pivot.index] = max(highs.get(pivot.index, -np.inf), float(pivot.price))
        elif pivot.kind == "low":
            lows[pivot.index] = min(lows.get(pivot.index, np.inf), float(pivot.price))
    high_indices = np.asarray(sorted(highs), dtype=int)
    low_indices = np.asarray(sorted(lows), dtype=int)
    high_prices = np.asarray([highs[index] for index in high_indices], dtype=float)
    low_prices = np.asarray([lows[index] for index in low_indices], dtype=float)
    if atr_values is None:
        high_tolerances = np.full(len(high_indices), 2.0 * tick_size)
        low_tolerances = np.full(len(low_indices), 2.0 * tick_size)
    else:
        tolerances = {}
        for pivot in pivots:
            if pivot.kind not in {"high", "low"}:
                continue
            tolerances[pivot.index] = max(tolerances.get(pivot.index, 0.0),
                                          _pivot_tolerance(pivot, atr_values, tick_size))
        high_tolerances = np.asarray([tolerances[index] for index in high_indices], dtype=float)
        low_tolerances = np.asarray([tolerances[index] for index in low_indices], dtype=float)
    return (high_indices, high_prices, high_tolerances,
            low_indices, low_prices, low_tolerances)


def _triangle_boundaries_intact(sequence, pivots, atr_values=None, tick_size=0.01, vectors=None):
    lows = [pivot for pivot in sequence if pivot.kind == "low"]
    highs = [pivot for pivot in sequence if pivot.kind == "high"]
    if len(lows) < 3 or len(highs) < 2:
        return False
    upper = _line((highs[0].index, highs[0].price), (highs[-1].index, highs[-1].price))
    lower = _line((lows[0].index, lows[0].price), (lows[-1].index, lows[-1].price))
    if upper["slope"] >= lower["slope"]:
        return False
    apex = _triangle_apex_index(upper, lower)
    if apex is None or apex <= lows[-1].index:
        return False
    high_indices, high_prices, high_tolerances, low_indices, low_prices, low_tolerances = (
        vectors or _triangle_pivot_vectors(pivots, atr_values, tick_size))
    high_start = bisect_left(high_indices, lows[0].index)
    high_end = bisect_right(high_indices, lows[-1].index)
    h_indices = high_indices[high_start:high_end]
    if len(h_indices) and np.any(high_prices[high_start:high_end] >
                                  upper["slope"] * h_indices + upper["intercept"]
                                  + high_tolerances[high_start:high_end]):
        return False
    low_start = bisect_left(low_indices, lows[0].index)
    low_end = bisect_right(low_indices, lows[-1].index)
    l_indices = low_indices[low_start:low_end]
    if len(l_indices) and np.any(low_prices[low_start:low_end] <
                                  lower["slope"] * l_indices + lower["intercept"]
                                  - low_tolerances[low_start:low_end]):
        return False
    return True


def _ihs_valley_intact(left, head, right, pivots, atr_values=None, tick_size=0.01,
                       low_values=None, low_pivot_indexes=None):
    tolerance = _pivot_tolerance(head, atr_values, tick_size)
    if low_pivot_indexes is not None:
        start = bisect_left(low_pivot_indexes, left.index)
        end = bisect_right(low_pivot_indexes, right.index)
        possible_lows = pivots[start:end]
    else:
        possible_lows = (pivot for pivot in pivots
                         if pivot.kind == "low" and left.index <= pivot.index <= right.index)
    if any(float(pivot.price) < float(head.price) - tolerance for pivot in possible_lows):
        return False
    if low_values is not None:
        end = min(len(low_values), right.index + 1)
        lows = np.asarray(low_values[left.index:end], dtype=float)
        lows = lows[np.isfinite(lows)]
        if len(lows) and float(np.min(lows)) < float(head.price) - tolerance:
            return False
    return True


def _ihs_right_shoulder_intact(head, right, pivots, atr_values=None, tick_size=0.01,
                               low_pivot_indexes=None):
    """Reject a later shoulder if an earlier significant right-side trough exists."""
    rules = MAIN_RULES.get("inverse_head_shoulders", {})
    minimum_atr = float(rules.get(
        "minimum_intervening_right_trough_head_separation_atr_multiple", 2.0))
    minimum_ticks = float(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2))
    right_tolerance = _pivot_tolerance(right, atr_values, tick_size)
    if low_pivot_indexes is not None:
        start = bisect_right(low_pivot_indexes, head.index)
        end = bisect_left(low_pivot_indexes, right.index)
        candidate_lows = pivots[start:end]
    else:
        candidate_lows = pivots
    for pivot in candidate_lows:
        if pivot.kind != "low" or not head.index < pivot.index < right.index:
            continue
        known_atr = (atr_values[pivot.index]
                     if atr_values is not None and pivot.index < len(atr_values) else np.nan)
        minimum_lift = max(minimum_ticks * tick_size,
                           minimum_atr * float(known_atr) if np.isfinite(known_atr) else 0.0)
        is_significant_shoulder = float(pivot.price) - float(head.price) >= minimum_lift
        if is_significant_shoulder and float(pivot.price) < float(right.price) - right_tolerance:
            return False
    return True


def _ihs_shoulders_have_depth(left, head, right, atr_values=None, tick_size=0.01):
    minimum_atr = float(MAIN_RULES.get("inverse_head_shoulders", {}).get(
        "minimum_shoulder_head_separation_atr_multiple", 1.0))
    minimum_ticks = float(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2))
    for shoulder in (left, right):
        known_atr = (atr_values[shoulder.index]
                     if atr_values is not None and shoulder.index < len(atr_values) else np.nan)
        required_lift = max(minimum_ticks * tick_size,
                            minimum_atr * float(known_atr) if np.isfinite(known_atr) else 0.0)
        if float(shoulder.price) - float(head.price) < required_lift:
            return False
    return True


def _structure_candidates(pivots, timeframe, atr_values=None, tick_size=0.01,
                          low_values=None, shoulder_pivots=None,
                          fast_candidate_enumeration=False):
    candidates = []
    lows = [pivot for pivot in pivots if pivot.kind == "low"]
    highs = [pivot for pivot in pivots if pivot.kind == "high"]
    low_indices_ordered = [pivot.index for pivot in lows]
    high_indices_ordered = [pivot.index for pivot in highs]
    low_indices_sorted = all(left <= right for left, right in zip(
        low_indices_ordered, low_indices_ordered[1:]))
    high_indices_sorted = all(left <= right for left, right in zip(
        high_indices_ordered, high_indices_ordered[1:]))
    shoulder_pool = shoulder_pivots or lows
    shoulder_lows = ([pivot for pivot in shoulder_pool if pivot.kind == "low"]
                     if shoulder_pool is not lows else lows)
    shoulder_low_indices = [pivot.index for pivot in shoulder_lows]
    shoulder_low_indices_sorted = all(left <= right for left, right in zip(
        shoulder_low_indices, shoulder_low_indices[1:]))

    def interval_pivots(pool, indexes, is_sorted, start, end):
        if is_sorted:
            return pool[bisect_right(indexes, start):bisect_left(indexes, end)]
        return [pivot for pivot in pool if start < pivot.index < end]

    vectors = _triangle_pivot_vectors(pivots, atr_values, tick_size)
    high_indices, high_prices = vectors[0], vectors[1]
    high_pivot_by_index = {}
    for pivot in highs:
        current = high_pivot_by_index.get(pivot.index)
        if current is None or pivot.price > current.price:
            high_pivot_by_index[pivot.index] = pivot
    for first in lows:
        seconds = []
        maximum_gap = float(MAIN_RULES.get("w_bottom", {}).get("maximum_trough_gap_ratio", 0.15))
        possible_seconds = []
        for second in lows:
            if second.index <= first.index:
                continue
            gap_ratio = abs(float(first.price) - float(second.price)) / max(
                float(first.price), float(second.price))
            if gap_ratio <= maximum_gap:
                possible_seconds.append((gap_ratio, -(second.index - first.index),
                                         second.index, second))
        if fast_candidate_enumeration:
            possible_seconds.sort(key=lambda item: item[:3])
        for gap_ratio, negative_span, _, second in possible_seconds:
            between_highs = interval_pivots(
                highs, high_indices_ordered, high_indices_sorted, first.index, second.index)
            if not between_highs:
                continue
            inner_lows = interval_pivots(
                lows, low_indices_ordered, low_indices_sorted, first.index, second.index)
            support = min(float(first.price), float(second.price))
            if any(float(pivot.price) < support - _pivot_tolerance(pivot, atr_values, tick_size)
                   for pivot in inner_lows):
                continue
            middle = max(between_highs, key=lambda pivot: pivot.price)
            known_atr = (atr_values[middle.confirmed_at]
                         if atr_values is not None and middle.confirmed_at < len(atr_values) else np.nan)
            rim_threshold = min(float(known_atr) if np.isfinite(known_atr) else 0.0,
                                max(float(first.price), float(second.price)) * 0.05)
            if float(middle.price) - max(float(first.price), float(second.price)) >= rim_threshold:
                seconds.append((gap_ratio, negative_span, second))
                if fast_candidate_enumeration:
                    break
        if seconds:
            second = (seconds[0][2] if fast_candidate_enumeration else
                      min(seconds, key=lambda item: (item[0], item[1], item[2].index))[2])
            candidates.append(("w_bottom", [first, second]))
    for head in lows:
        lefts = [pivot for pivot in lows if pivot.index < head.index and pivot.price > head.price]
        rights = [pivot for pivot in lows if pivot.index > head.index and pivot.price > head.price]
        shoulder_pairs = []
        ihs_rules = MAIN_RULES.get("inverse_head_shoulders", {})
        maximum_shoulder_gap = float(ihs_rules.get("maximum_shoulder_gap_to_head_depth_ratio", 0.50))
        maximum_leg_ratio = float(ihs_rules.get("maximum_leg_duration_ratio", 2.5))
        selected_pair = None
        pair_order = (((left, right) for right in reversed(rights) for left in reversed(lefts))
                      if fast_candidate_enumeration else
                      ((left, right) for left in lefts for right in rights))
        for left, right in pair_order:
                left_leg = head.index - left.index
                right_leg = right.index - head.index
                if min(left_leg, right_leg) <= 0 or max(left_leg, right_leg) / min(left_leg, right_leg) > maximum_leg_ratio:
                    continue
                neck_highs = interval_pivots(
                    highs, high_indices_ordered, high_indices_sorted, left.index, right.index)
                if len(neck_highs) < 2:
                    continue
                depth = max(float(pivot.price) for pivot in neck_highs) - float(head.price)
                if depth <= 0:
                    continue
                shoulder_gap = abs(float(left.price) - float(right.price)) / depth
                if shoulder_gap > maximum_shoulder_gap:
                    continue
                if not _ihs_shoulders_have_depth(left, head, right, atr_values, tick_size):
                    continue
                if not _ihs_valley_intact(
                        left, head, right, lows, atr_values, tick_size,
                        low_values=low_values,
                        low_pivot_indexes=(low_indices_ordered if low_indices_sorted else None)):
                    continue
                if not _ihs_right_shoulder_intact(
                        head, right, shoulder_lows, atr_values, tick_size,
                        low_pivot_indexes=(shoulder_low_indices
                                           if shoulder_low_indices_sorted else None)):
                    continue
                local_span = (head.index - left.index) + (right.index - head.index)
                shoulder_pairs.append((local_span, shoulder_gap, -(right.index - left.index),
                                       left.index, [left, head, right]))
                if fast_candidate_enumeration:
                    selected_pair = [left, head, right]
                    break
        if shoulder_pairs:
            # Keep the most recently completed right shoulder, then the nearest
            # significant left shoulder. This preserves a known large structure
            # without letting a later post-breakout low replace its right shoulder.
            selected = (selected_pair if fast_candidate_enumeration else min(
                shoulder_pairs, key=lambda item: (
                    -item[4][2].index, -item[4][0].index, item[1], item[3]))[4])
            candidates.append(("inverse_head_shoulders", selected))
    for first in lows:
        triangle_candidates = []
        seconds = [pivot for pivot in lows if pivot.index > first.index
                   and float(pivot.price) > float(first.price)]
        for second in seconds:
            first_start = bisect_right(high_indices, first.index)
            first_end = bisect_left(high_indices, second.index)
            if first_end <= first_start:
                continue
            first_high_index = first_start + int(np.argmax(high_prices[first_start:first_end]))
            first_high_pivot = high_pivot_by_index[int(high_indices[first_high_index])]
            for third in lows:
                if third.index <= second.index or float(third.price) <= float(second.price):
                    continue
                second_start = bisect_right(high_indices, second.index)
                second_end = bisect_left(high_indices, third.index)
                if second_end <= second_start:
                    continue
                second_high_index = second_start + int(np.argmax(high_prices[second_start:second_end]))
                second_high_pivot = high_pivot_by_index[int(high_indices[second_high_index])]
                if float(first_high_pivot.price) <= float(second_high_pivot.price):
                    continue
                sequence = [first, first_high_pivot, second, second_high_pivot, third]
                if not _triangle_boundaries_intact(sequence, pivots, atr_values, tick_size, vectors):
                    continue
                width = float(first_high_pivot.price) - float(first.price)
                triangle_candidates.append((-(third.index - first.index), -width,
                                            second.index, ["triangle_bottom", sequence]))
        if triangle_candidates:
            candidates.append(min(triangle_candidates, key=lambda item: item[:3])[3])
    return candidates


def _make_main_pattern(bars, code, kind, timeframe, family, sequence, pivots,
                       atr_values, tick_size, pivot_scale, scale_name, background_bars=None,
                       collect_diagnostics=True, consumed_boundary_phases=None,
                       neckline_coverage_mode="full_interval"):
    index = bars.index
    frame = (bars if isinstance(index, pd.RangeIndex) and index.start == 0
             and index.step == 1 and index.stop == len(bars)
             else bars.reset_index(drop=True))
    dates = frame.attrs.get("neckline_date_index")
    if dates is None or len(dates) != len(frame):
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    date_strings = frame.attrs.get("date_strings")
    if date_strings is None or len(date_strings) != len(frame):
        date_strings = dates.strftime("%Y-%m-%d").tolist()
    frame.attrs["date_strings"] = date_strings
    close = frame.attrs.get("neckline_close_values")
    if close is None or len(close) != len(frame):
        close = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=float)
    else:
        close = np.asarray(close, dtype=float)
    high_values = frame.attrs.get("neckline_high_values")
    if high_values is None or len(high_values) != len(frame):
        high_values = pd.to_numeric(frame.high, errors="coerce").to_numpy(dtype=float)
    else:
        high_values = np.asarray(high_values, dtype=float)
    low_values = frame.attrs.get("neckline_low_values")
    if low_values is None or len(low_values) != len(frame):
        low_values = pd.to_numeric(frame.low, errors="coerce").to_numpy(dtype=float)
    else:
        low_values = np.asarray(low_values, dtype=float)
    first_trough = min((pivot for pivot in sequence if pivot.kind == "low"), key=lambda pivot: pivot.index)
    last_anchor = max(sequence, key=lambda pivot: pivot.index)
    low_pivots = [pivot for pivot in sequence if pivot.kind == "low"]
    if family == "w_bottom" and len(low_pivots) >= 2:
        maximum_gap = float(MAIN_RULES.get("w_bottom", {}).get("maximum_trough_gap_ratio", 0.15))
        if max(float(pivot.price) for pivot in low_pivots) - min(float(pivot.price) for pivot in low_pivots) > max(low_pivots[0].price, low_pivots[-1].price) * maximum_gap:
            return None
        middle_highs = [pivot for pivot in pivots if pivot.kind == "high"
                        and low_pivots[0].index < pivot.index < low_pivots[-1].index]
        if not middle_highs:
            return None
        w_support_lower = min(float(item.price) for item in low_pivots)
        for pivot in pivots:
            if (pivot.kind == "low" and low_pivots[0].index < pivot.index < low_pivots[-1].index
                    and float(pivot.price) < w_support_lower
                    - _pivot_tolerance(pivot, atr_values, tick_size)):
                return None
        known_atrs = [atr_values[pivot.confirmed_at] for pivot in low_pivots
                      if pivot.confirmed_at < len(atr_values) and np.isfinite(atr_values[pivot.confirmed_at])]
        meaningful_rim = min(float(min(known_atrs, default=0.0)), max(float(low_pivots[0].price), float(low_pivots[-1].price)) * 0.05)
        if max(float(pivot.price) for pivot in middle_highs) - max(float(pivot.price) for pivot in low_pivots) < meaningful_rim:
            return None
    w_middle_highs = ([pivot for pivot in pivots if pivot.kind == "high"
                       and low_pivots[0].index < pivot.index < low_pivots[-1].index]
                      if family == "w_bottom" else [])
    minimum_neckline = None
    if w_middle_highs:
        middle = max(w_middle_highs, key=lambda pivot: pivot.price)
        middle_atr = atr_values[middle.confirmed_at] if middle.confirmed_at < len(atr_values) else np.nan
        middle_tolerance = max(float(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2)) * tick_size,
                               float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
                               * float(middle_atr) if np.isfinite(middle_atr) else 0.0)
        # A resistance retest must reach the intervening W rim; tolerance
        # applies to independent touch clustering, not to lowering the
        # candidate neckline below the actual peak.
        minimum_neckline = float(middle.price) - middle_tolerance
    if family == "triangle_bottom":
        triangle_highs = [pivot for pivot in sequence if pivot.kind == "high"]
        triangle_lows = [pivot for pivot in sequence if pivot.kind == "low"]
        if len(triangle_highs) < 2 or len(triangle_lows) < 2:
            return None
    if family == "inverse_head_shoulders":
        head = next(pivot for pivot in sequence if pivot is not low_pivots[0]
                    and pivot is not low_pivots[-1] and pivot.kind == "low")
        left_leg = head.index - low_pivots[0].index
        right_leg = low_pivots[-1].index - head.index
        maximum_leg_ratio = float(MAIN_RULES.get("inverse_head_shoulders", {}).get(
            "maximum_leg_duration_ratio", 2.5))
        if (min(left_leg, right_leg) <= 0
                or max(left_leg, right_leg) / min(left_leg, right_leg) > maximum_leg_ratio):
            return None
        if not _ihs_shoulders_have_depth(low_pivots[0], head, low_pivots[-1],
                                         atr_values, tick_size):
            return None
        if not _ihs_valley_intact(low_pivots[0], head, low_pivots[-1],
                                  pivots, atr_values, tick_size, low_values=low_values):
            return None
        if not _ihs_right_shoulder_intact(head, low_pivots[-1], pivots,
                                          atr_values, tick_size):
            return None
    minimum_test_count = _minimum_neckline_test_count()
    neckline_diagnostics = {}

    def record_neckline_rejection(reason, details):
        if not collect_diagnostics:
            return
        counts = neckline_diagnostics.setdefault("rejection_counts", {})
        counts[reason] = int(counts.get(reason, 0)) + 1
        sample = dict(details, reason=reason)
        known_date = sample.get("known_date")
        if known_date:
            by_date = neckline_diagnostics.setdefault("rejection_count_by_date", {}).setdefault(reason, {})
            by_date[str(pd.Timestamp(known_date).date())] = by_date.get(str(pd.Timestamp(known_date).date()), 0) + 1
        samples = neckline_diagnostics.setdefault("rejected_candidates", [])
        samples.append(sample)
        if len(samples) > 128:
            del samples[128:]

    neckline_candidates = _independent_neckline_tests(
        pivots, first_trough, last_anchor, atr_values, dates, tick_size, timeframe,
        # Causal independent pressure tests may precede the first trough or
        # finish after it; their known dates still precede the frozen pair.
        lower_index=(int(np.searchsorted(
            dates.values,
            (pd.Timestamp(dates[first_trough.index]) - pd.DateOffset(years=NECKLINE_LOOKBACK_YEARS)).to_datetime64(),
            side="left")) if family == "w_bottom" else first_trough.index),
        # Pressure tests after the final structure trough can finish freezing
        # the boundary, provided they are known before the breakout event.
        upper_index=(len(frame) - 1 if family in {"w_bottom", "inverse_head_shoulders"}
                     else last_anchor.index),
        minimum_level=minimum_neckline,
        maximum_level=(float(max(pivot.price for pivot in w_middle_highs)) *
                       (1.0 + float(MAIN_RULES.get("neckline_tests", {}).get(
                           "w_bottom_rim_association_ceiling_relative", 0.2)))
                       if w_middle_highs else None),
        bars=frame, return_candidates=True,
        diagnostics=neckline_diagnostics if collect_diagnostics else None,
        trusted_readonly_cache=True,
        coverage_mode=neckline_coverage_mode,
        structure_anchors=sequence if neckline_coverage_mode == "structure_anchors_only" else None,
    )
    neckline_candidates = [candidate for candidate in neckline_candidates
                           if len(candidate["tests"]) >= minimum_test_count]
    if not neckline_candidates:
        return None
    if family == "triangle_bottom":
        high_pivots = triangle_highs
        low_pivots = triangle_lows
        neckline = _line((high_pivots[0].index, high_pivots[0].price),
                         (high_pivots[-1].index, high_pivots[-1].price))
        neckline = _qualify_neckline_slope(neckline, high_pivots[0].index, high_pivots[-1].index)
        if neckline is None:
            return None
        filtered_candidates = []
        for candidate in neckline_candidates:
            candidate_tests = [test for test in candidate["tests"]
                               if abs(test["level"] - _at(neckline, test["pivot_index"]))
                               <= max(test["tolerance"], 2.0 * tick_size)]
            if len(candidate_tests) < minimum_test_count:
                record_neckline_rejection("tests_do_not_match_triangle_upper_boundary", {
                    "candidate_id": candidate["tests"][0]["pair_id"],
                    "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                    "matched_test_count": len(candidate_tests),
                    "known_date": max(test["known_date"] for test in candidate["tests"]),
                })
                continue
            coverage_start = min(test["pivot_index"] for test in candidate_tests)
            coverage_end = max(test["known_index"] for test in candidate_tests)
            idx = np.arange(coverage_start, coverage_end + 1, dtype=int)
            interval_atr = np.asarray(atr_values, dtype=float)[idx]
            tolerance = np.maximum(
                float(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2)) * tick_size,
                np.where(np.isfinite(interval_atr),
                         float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5)) * interval_atr, 0.0))
            excesses = high_values[idx] - _at(neckline, idx) - tolerance
            if np.any(excesses > 1e-12):
                worst = int(np.argmax(excesses))
                record_neckline_rejection("triangle_upper_envelope_crossed", {
                    "candidate_id": candidate["tests"][0]["pair_id"],
                    "test_dates": [test["pivot_date"] for test in candidate_tests],
                    "coverage_start_date": _date_text(dates, coverage_start, date_strings),
                    "coverage_end_date": _date_text(dates, coverage_end, date_strings),
                    "line": dict(neckline),
                    "crossing_date": _date_text(dates, idx[worst], date_strings),
                    "crossing_high": float(high_values[idx[worst]]),
                    "maximum_excess": float(excesses[worst]),
                    "known_date": _date_text(dates, coverage_end, date_strings),
                })
                continue
            filtered_candidates.append(dict(
                candidate, tests=candidate_tests, line=dict(neckline),
                frozen_index=max(test["known_index"] for test in candidate_tests),
                coverage_start_index=coverage_start, coverage_end_index=coverage_end))
        neckline_candidates = filtered_candidates
        if not neckline_candidates:
            return None
        neckline_level = _at(neckline, last_anchor.index)
        lower_boundary = _line((low_pivots[0].index, low_pivots[0].price),
                               (low_pivots[-1].index, low_pivots[-1].price))
        if not _triangle_boundaries_intact(sequence, pivots, atr_values, tick_size):
            return None
        apex_index = _triangle_apex_index(neckline, lower_boundary)
        if apex_index is None:
            return None
        if apex_index <= first_trough.index:
            return None
        neckline_rules = MAIN_RULES.get("neckline_tests", {})
        weekly = timeframe in {"1w", "weekly", "week"}
        minimum_gap = int(neckline_rules.get(
            "minimum_separation_weekly_bars" if weekly else "minimum_separation_daily_sessions",
            2 if weekly else 10))
        between = [pivot for pivot in pivots if pivot.kind == "high"
                   and low_pivots[0].index < pivot.index < low_pivots[-1].index]
        support_departure = (_at(neckline, low_pivots[1].index) - float(low_pivots[1].price)
                             if len(low_pivots) >= 3 else 0.0)
        middle_atr = atr_values[low_pivots[1].confirmed_at] if len(low_pivots) >= 3 else np.nan
        if (low_pivots[-1].index - low_pivots[0].index < minimum_gap
                or not between or not np.isfinite(middle_atr) or support_departure < middle_atr):
            return None
        support_tests = [{"group_id": f"support-touch-{pivot.index}",
                          "pivot_index": int(pivot.index), "pivot_date": pivot.date,
                          "known_index": int(pivot.confirmed_at), "known_date": pivot.confirmed_date,
                          "level": float(pivot.price), "valid": True}
                         for pivot in (low_pivots[0], low_pivots[-1])]
    else:
        support_tests = []
        qualified = []
        maximum_shoulder_gap = float(MAIN_RULES.get("inverse_head_shoulders", {}).get(
            "maximum_shoulder_gap_to_head_depth_ratio", 0.50))
        for candidate in neckline_candidates:
            candidate_line = _qualify_neckline_slope(
                candidate["line"], candidate["tests"][0]["pivot_index"],
                candidate["tests"][-1]["pivot_index"])
            if candidate_line is None:
                record_neckline_rejection("slope_limit_exceeded_after_horizontal_conversion", {
                    "candidate_id": candidate["tests"][0]["pair_id"],
                    "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                    "line": dict(candidate["line"]),
                    "known_date": max(test["known_date"] for test in candidate["tests"]),
                })
                continue
            if family == "w_bottom":
                if not _w_rebound_peak_associated(
                        low_pivots, candidate["tests"], candidate_line,
                        high_values, atr_values, tick_size):
                    peaks = []
                    for left, right in zip(low_pivots, low_pivots[1:]):
                        start, end = int(left.index) + 1, int(right.index)
                        peak_index = (start + int(np.nanargmax(high_values[start:end]))
                                      if start < end else start)
                        peaks.append({
                            "date": date_strings[peak_index],
                            "price": float(high_values[peak_index]),
                            "line_level": float(_at(candidate_line, peak_index)),
                            "tolerance": float(_neckline_test_tolerance_at(
                                peak_index, atr_values, tick_size)),
                        })
                    record_neckline_rejection("w_rebound_peak_not_associated_with_boundary", {
                        "candidate_id": candidate["tests"][0]["pair_id"],
                        "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                        "intervening_rebound_peaks": peaks,
                        "line": dict(candidate_line),
                        "known_date": max(test["known_date"] for test in candidate["tests"]),
                    })
                    continue
                intact = True
                for pivot in w_middle_highs:
                    tolerance = _pivot_tolerance(pivot, atr_values, tick_size)
                    if _at(candidate_line, pivot.index) < float(pivot.price) - tolerance:
                        intact = False
                        break
                if not intact:
                    record_neckline_rejection("w_intervening_rim_not_covered", {
                        "candidate_id": candidate["tests"][0]["pair_id"],
                        "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                        "line": dict(candidate_line),
                        "uncovered_rim_dates": [pivot.date for pivot in w_middle_highs
                                                 if _at(candidate_line, pivot.index)
                                                 < float(pivot.price) - _pivot_tolerance(pivot, atr_values, tick_size)],
                        "known_date": max(test["known_date"] for test in candidate["tests"]),
                    })
                    continue
            else:
                ihs_neckline_rules = MAIN_RULES.get("inverse_head_shoulders_neckline", {})
                if ihs_neckline_rules.get("require_independent_test_on_each_reaction_leg", True):
                    reaction_ranges = [
                        (low_pivots[0].index, head.index),
                        (head.index, low_pivots[-1].index),
                    ]
                    test_indices = [int(test["pivot_index"]) for test in candidate["tests"]]
                    missing_reaction_ranges = [position + 1 for position, (left, right) in
                                               enumerate(reaction_ranges)
                                               if not any(left < test_index < right
                                                          for test_index in test_indices)]
                    if missing_reaction_ranges:
                        record_neckline_rejection("missing_ihs_reaction_side_test", {
                            "candidate_id": candidate["tests"][0]["pair_id"],
                            "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                            "missing_reaction_legs": missing_reaction_ranges,
                            "known_date": max(test["known_date"] for test in candidate["tests"]),
                        })
                        continue
                uncovered_lows = []
                if ihs_neckline_rules.get("require_neckline_above_all_three_troughs", True):
                    for pivot in low_pivots:
                        tolerance = _pivot_tolerance(pivot, atr_values, tick_size)
                        line_level = _at(candidate_line, pivot.index)
                        if line_level < float(pivot.price) - tolerance:
                            uncovered_lows.append({
                                "date": pivot.date, "low": float(pivot.price),
                                "neckline": float(line_level), "tolerance": float(tolerance),
                            })
                if uncovered_lows:
                    record_neckline_rejection("neckline_below_ihs_structure", {
                        "candidate_id": candidate["tests"][0]["pair_id"],
                        "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                        "uncovered_structure_lows": uncovered_lows,
                        "known_date": max(test["known_date"] for test in candidate["tests"]),
                    })
                    continue
                neck_at_head = _at(candidate_line, head.index)
                depth = neck_at_head - float(head.price)
                shoulder_gap = abs(float(low_pivots[0].price) - float(low_pivots[-1].price))
                if depth <= 0 or shoulder_gap / depth > maximum_shoulder_gap:
                    record_neckline_rejection("shoulder_geometry_not_supported_by_boundary", {
                        "candidate_id": candidate["tests"][0]["pair_id"],
                        "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                        "line": dict(candidate_line),
                        "head_depth": float(depth),
                        "shoulder_gap_ratio": float(shoulder_gap / depth) if depth > 0 else None,
                        "known_date": max(test["known_date"] for test in candidate["tests"]),
                    })
                    continue
            qualified.append(dict(candidate, line=candidate_line))
        neckline_candidates = qualified
        if not neckline_candidates:
            return None

    geometry_start = min(pivot.index for pivot in sequence)
    geometry_end = max(pivot.index for pivot in sequence)
    geometry_pivots = [pivot for pivot in pivots
                       if geometry_start <= pivot.index <= geometry_end]
    structure_known_at = max(max(pivot.confirmed_at for pivot in sequence),
                             max((pivot.confirmed_at for pivot in geometry_pivots), default=0))
    if structure_known_at >= len(frame):
        return None

    def paired_freeze_index(candidate):
        if neckline_coverage_mode == "structure_anchors_only":
            return max(int(candidate["frozen_index"]), int(structure_known_at))
        return int(candidate["frozen_index"])

    covered_candidates = []
    for candidate in neckline_candidates:
        pair_freeze = paired_freeze_index(candidate)
        formation_end = max(structure_known_at, pair_freeze)
        formation_start = min(geometry_start, candidate["tests"][0]["coverage_start_index"])
        if neckline_coverage_mode == "structure_anchors_only":
            coverage = _neckline_structure_anchor_coverage(
                candidate["line"], sequence, atr_values, tick_size)
            coverage.update({"start_index": formation_start, "end_index": formation_end,
                             "coverage_mode": "structure_anchors_only"})
        else:
            coverage = _neckline_formation_coverage(
                high_values, atr_values, candidate["line"], formation_start, formation_end,
                tick_size, cache=frame.attrs.get("neckline_coverage_cache"),
                tolerances=frame.attrs.get("neckline_coverage_tolerances"))
        if not coverage["passed"]:
            record_neckline_rejection(
                "formation_high_crosses_neckline" if neckline_coverage_mode == "full_interval"
                else "structure_anchor_above_neckline", {
                "candidate_id": candidate["tests"][0]["pair_id"],
                "formation_coverage": coverage,
                "known_date": date_strings[formation_end],
            })
            continue
        covered_candidates.append(dict(candidate, paired_frozen_index=pair_freeze,
                                       formation_coverage=coverage))
    neckline_candidates = covered_candidates
    if not neckline_candidates:
        return None
    duration_rules = MAIN_RULES.get("minimum_duration_calendar_months_by_timeframe", {})
    minimum_duration_months = int(duration_rules.get(
        "1w" if timeframe in {"1w", "weekly", "week"} else "1d", MIN_STRUCTURE_MONTHS))
    maturity_date = pd.Timestamp(dates[first_trough.index]) + pd.DateOffset(months=minimum_duration_months)
    maturity_indices = np.flatnonzero(dates >= maturity_date)
    maturity_index = int(maturity_indices[0]) if len(maturity_indices) else len(frame)
    support_values = [float(pivot.price) for pivot in sequence if pivot.kind == "low"]
    support_lower = min(support_values)
    support_upper = max(support_values)
    direction = "bullish"
    def support_boundary(index):
        return _at(lower_boundary, index) if family == "triangle_bottom" else support_lower

    # Rank one boundary for the primary event; other candidates only consume
    # their first crossing so an unselected phase cannot reappear later.
    weak_crossings = []
    breakout = None
    early_breakouts = []
    neckline_history = []
    rejected_boundaries = []
    consumed_phases = [dict(item) for item in (consumed_boundary_phases or [])]
    chosen_candidate = None
    pending_breakout = None
    historical_only = False
    early_breakout_invalidation = None
    locked_candidate = None
    for index in range(structure_known_at, len(frame)):
        if close[index] < support_boundary(index):
            break
        if pending_breakout is not None:
            prior_candidate = pending_breakout["candidate"]
            prior_support = float(pending_breakout["item"]["frozen_neckline"])
            if close[index] < prior_support:
                invalidation = {
                    "index": int(index), "price": prior_support,
                    "tolerance": max(2.0 * tick_size,
                                     float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
                                     * float(atr_values[index]) if np.isfinite(atr_values[index]) else 0.0),
                    "reason": "breakout_reentered_below_frozen_neckline",
                    "candidate_id": pending_breakout["item"]["candidate_id"],
                }
                rejected_boundaries.append(invalidation)
                neckline_history.append({
                    "candidate_id": pending_breakout["item"]["candidate_id"],
                    "breakout": dict(pending_breakout["item"]),
                    "candidate": prior_candidate,
                    "invalidation": invalidation,
                    "breakout_date": pending_breakout["item"]["date"],
                    "invalidation_date": _date_text(dates, index, date_strings),
                    "state": "invalidated",
                })
                pending_breakout = None
                chosen_candidate = None
                # The old boundary phase has reached a terminal state. Keep
                # its historical record, but allow another distinct frozen
                # boundary for the same structure anchors to start a new
                # lifecycle and produce its own first breakout event.
                locked_candidate = None
                continue
            # Keep the primary pending lifecycle unchanged. Sidechannel
            # candidates are only considered when no primary is pending.
            continue
        failed_phase = _failed_premature_neckline_phase(consumed_phases, close, index)
        if failed_phase is not None:
            early_breakout_invalidation = {
                "date": _date_text(dates, index, date_strings), "index": int(index),
                "level": float(_at(failed_phase["line"], index)),
                "reason": "premature_breakout_fell_back_below_frozen_neckline",
                "breakout_date": date_strings[int(failed_phase["consumed_index"])],
                "candidate_id": failed_phase.get("candidate_id"),
                "line": dict(failed_phase["line"]),
            }
            chosen_candidate = next((candidate for candidate in neckline_candidates
                                     if _same_neckline_boundary_phase(
                                         failed_phase, candidate, tick_size)), chosen_candidate)
            break
        available = [candidate for candidate in neckline_candidates
                     if paired_freeze_index(candidate) < index
                     and _neckline_decline_check(family, candidate["line"],
                         candidate["tests"][0]["coverage_start_index"], index)["passed"]]
        # A previously frozen instance must not lock this evaluation to its
        # line: keep that instance separately and let a newly paired, distinct
        # boundary establish its own lifecycle.
        consumed_line_matches = _consumed_neckline_phase_matcher(consumed_phases, index)
        available = [candidate for candidate in available
                     if not consumed_line_matches(candidate["line"])]
        if locked_candidate is None and available:
            # Pair the structure with the first knowable available boundary.
            # When several freeze together, evidence rank breaks the tie.
            available.sort(key=lambda candidate: (
                paired_freeze_index(candidate), candidate["score"]))
            locked_candidate = available[0]
        elif locked_candidate is not None:
            available = [candidate for candidate in available if candidate is locked_candidate]
        available = [candidate for candidate in available if not any(
            abs(_at(candidate["line"], rejected["index"]) - rejected["price"])
            <= rejected["tolerance"] for rejected in rejected_boundaries)]
        if not available:
            continue
        available.sort(key=lambda candidate: candidate["score"])
        primary_candidate = min(available, key=lambda candidate: candidate["score"])
        chosen_candidate = primary_candidate
        for candidate in available:
            selected_tests = candidate["tests"]
            neckline = dict(candidate["line"])
            if index <= structure_known_at or index <= paired_freeze_index(candidate):
                continue
            if any(abs(_at(neckline, rejected["index"]) - rejected["price"])
                   <= rejected["tolerance"] for rejected in rejected_boundaries):
                continue
            phase = {"line": neckline, "tests": selected_tests,
                     "consumed_index": int(index),
                     "phase_tolerance": _neckline_test_tolerance_at(index, atr_values, tick_size),
                     "candidate_id": selected_tests[0]["pair_id"]}
            matching_consumed = [existing for existing in consumed_phases
                                 if (existing.get("consumed_index") is None
                                     or int(existing["consumed_index"]) < index)
                                 and _same_neckline_boundary_phase(existing, phase, tick_size)]
            if any(existing.get("selected", True) for existing in matching_consumed):
                continue
            sidephase_already_consumed = bool(matching_consumed)
            previous_level = _at(neckline, index - 1)
            previous_atr_for_buffer = atr_values[index - 2] if index > 1 else np.nan
            if not np.isfinite(previous_atr_for_buffer) or previous_atr_for_buffer <= 0:
                continue
            previous_buffer = max(
                float(MAIN_RULES.get("breakout", {}).get("minimum_buffer_ticks", 2)) * tick_size,
                float(MAIN_RULES.get("breakout", {}).get("close_buffer_previous_atr_multiple", 1.0))
                * float(previous_atr_for_buffer),
            )
            if close[index - 1] > previous_level + previous_buffer:
                continue
            previous_atr = atr_values[index - 1] if index > 0 else np.nan
            if not np.isfinite(previous_atr) or previous_atr <= 0:
                continue
            level = _at(neckline, index)
            breakout_rules = MAIN_RULES.get("breakout", {})
            tolerance = max(float(breakout_rules.get("minimum_buffer_ticks", 2)) * tick_size,
                            float(breakout_rules.get("close_buffer_previous_atr_multiple", 1.0))
                            * float(previous_atr))
            required_close = level + tolerance
            if close[index] > level and close[index] <= required_close:
                weak_crossings.append({
                    "date": _date_text(dates, index, date_strings), "close": float(close[index]),
                    "neckline": float(level), "required_close": float(required_close),
                    "atr14_previous": float(previous_atr), "tolerance": float(tolerance),
                    "candidate_id": selected_tests[0]["pair_id"],
                })
            if close[index] <= required_close:
                continue
            if sidephase_already_consumed:
                # This exact frozen boundary has already had its first
                # buffered crossing consumed; another candidate line remains
                # free to confirm its own lifecycle.
                continue
            item = {
                "date": _date_text(dates, index, date_strings), "index": int(index),
                "close": float(close[index]), "frozen_neckline": float(level),
                "required_close": float(required_close),
                "atr14_previous": float(previous_atr), "tolerance": float(tolerance),
                "candidate_id": selected_tests[0]["pair_id"],
            }
            if pd.Timestamp(dates[index]) < maturity_date:
                early_event = dict(item, line=dict(candidate["line"]),
                                   tests=candidate["tests"])
                early_breakouts.append(early_event)
                phase["reason"] = "breakout_before_minimum_structure_duration"
                consumed_phases.append(phase)
                rejected_boundaries.append({
                    "index": int(index), "price": float(level),
                    "tolerance": max(2.0 * tick_size,
                                     float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
                                     * float(previous_atr)),
                    "reason": "breakout_before_minimum_structure_duration",
                    "candidate_id": item["candidate_id"],
                })
                neckline_history.append({"candidate_id": item["candidate_id"],
                                         "breakout_date": item["date"], "state": "too_early"})
                continue
            else:
                phase["reason"] = "buffered_breakout"
                consumed_phases.append(phase)
                pending_breakout = {"candidate": candidate, "item": item}
                chosen_candidate = candidate
                # The reported pattern must carry the same frozen neckline
                # that generated its first valid breakout.
                break
    if chosen_candidate is None:
        final_candidates = [candidate for candidate in neckline_candidates
                            if locked_candidate is None or candidate is locked_candidate
                            if paired_freeze_index(candidate) <= len(frame) - 1
                            and _neckline_decline_check(family, candidate["line"],
                                candidate["tests"][0]["coverage_start_index"], len(frame)-1)["passed"]
                            and not any(phase.get("selected", True)
                                        and _same_neckline_boundary_phase(phase, candidate, tick_size)
                                        and (phase.get("consumed_index") is None
                                             or int(phase["consumed_index"]) < len(frame) - 1)
                                        for phase in consumed_phases)
                            and not any(
                                abs(_at(candidate["line"], rejected["index"]) - rejected["price"])
                                <= rejected["tolerance"] for rejected in rejected_boundaries)]
        if not final_candidates:
            retired = [entry for entry in neckline_history
                       if entry.get("state") == "invalidated" and entry.get("candidate")]
            if not retired:
                return None
            # A retired boundary is still a historical event. The lack of a
            # current eligible candidate must not erase an earlier order.
            chosen_candidate = retired[-1]["candidate"]
            historical_only = True
        else:
            chosen_candidate = min(final_candidates, key=lambda candidate: candidate["score"])
    if pending_breakout is not None:
        chosen_candidate = pending_breakout["candidate"]
        breakout = pending_breakout["item"]
        neckline_history.append({"candidate_id": breakout["candidate_id"],
                                 "breakout_date": breakout["date"], "state": "confirmed"})
    selected_tests = chosen_candidate["tests"]
    selected_candidate_id = selected_tests[0]["pair_id"]
    selected_weak_crossings = [item for item in weak_crossings
                               if item.get("candidate_id") == selected_candidate_id]
    selected_early_breakouts = [item for item in early_breakouts
                                 if item.get("candidate_id") == selected_candidate_id]
    selected_retired_segments = [item for item in neckline_history
                                 if item.get("candidate_id") == selected_candidate_id]
    selected_consumed_phases = [item for item in consumed_phases
                                if item.get("candidate_id") == selected_candidate_id]
    neckline = dict(chosen_candidate["line"])
    neckline_level = _at(neckline, last_anchor.index)
    frozen_date_value = max(
        pd.Timestamp(dates[structure_known_at]),
        pd.Timestamp(dates[paired_freeze_index(chosen_candidate)]),
    )
    frozen_at = int(np.flatnonzero(dates == frozen_date_value)[0])
    if frozen_at >= len(frame):
        return None
    frozen_date = _date_text(dates, frozen_at, date_strings)
    if breakout is None and pd.Timestamp(dates[-1]) < maturity_date:
        return None

    structure_invalidation = None
    neckline_invalidation = early_breakout_invalidation
    target_reached = None
    target_values = []
    structure_low = min(support_values)
    sequence_known_at = max(pivot.confirmed_at for pivot in sequence)
    invalidation_end = breakout["index"] if breakout is not None else (
        int(early_breakout_invalidation["index"]) if early_breakout_invalidation is not None
        else len(frame) - 1)
    first_structure_break = next((index for index in range(sequence_known_at, invalidation_end + 1)
                                  if close[index] < support_boundary(index)), None)
    if first_structure_break is not None:
        structure_invalidation = {"date": _date_text(dates, first_structure_break, date_strings),
                                  "index": int(first_structure_break),
                                  "level": float(support_boundary(first_structure_break))}
        breakout = None
    if breakout is not None and family == "triangle_bottom":
        breakout_position = (breakout["index"] - first_trough.index) / max(apex_index - first_trough.index, 1.0)
        if not 0.5 <= breakout_position <= 0.75:
            # The first close beyond the buffered boundary was outside the
            # book's effective triangle window; later price cannot re-label it.
            return None
    terminal_dates = [event["date"] for event in
                      (structure_invalidation, neckline_invalidation) if event]
    quality_date = (min(terminal_dates) if terminal_dates else
                    (breakout["date"] if breakout else date_strings[-1]))
    if historical_only:
        quality_date = min(entry["breakout_date"] for entry in neckline_history
                           if entry.get("candidate") and entry.get("state") == "invalidated")
    background = background_bars if background_bars is not None else bars
    background = background.copy()
    background["date"] = pd.to_datetime(background.date, errors="coerce")
    quality_history = background[background.date <= pd.Timestamp(quality_date)]
    finite_low = pd.to_numeric(quality_history.low, errors="coerce").dropna()
    finite_high = pd.to_numeric(quality_history.high, errors="coerce").dropna()
    low_position = {"passed": False, "position_pass": False, "drawdown_pass": False,
                    "history_end": quality_date, "range_low": None, "range_high": None,
                    "position_ratio": None, "prepattern_high": None, "prepattern_drawdown": None}
    if len(finite_low) and len(finite_high):
        range_low, range_high = float(finite_low.min()), float(finite_high.max())
        ratio = ((structure_low - range_low) / (range_high - range_low)
                 if range_high > range_low else np.inf)
        low_rules = MAIN_RULES.get("low_position", {})
        cutoff = pd.Timestamp(dates[first_trough.index]) - pd.DateOffset(
            years=int(low_rules.get("prepattern_lookback_calendar_years", 2)))
        prepattern = quality_history[(quality_history.date >= cutoff)
                                     & (quality_history.date < pd.Timestamp(dates[first_trough.index]))]
        prior_high = pd.to_numeric(prepattern.high, errors="coerce").dropna()
        pre_high = float(prior_high.max()) if len(prior_high) else None
        drawdown = (pre_high - structure_low) / pre_high if pre_high and pre_high > 0 else None
        low_position.update({"range_low": range_low, "range_high": range_high,
                             "position_ratio": float(ratio), "position_pass": bool(
                                 0 <= ratio <= float(low_rules.get("full_history_position_max", 0.30))),
                             "prepattern_high": pre_high, "prepattern_drawdown": drawdown,
                             "drawdown_pass": bool(drawdown is not None and drawdown >= float(
                                 low_rules.get("minimum_prepattern_drawdown", 0.30)))})
        low_position["passed"] = bool(low_position["position_pass"] or low_position["drawdown_pass"])
    if not low_position["passed"]:
        return None

    if breakout is not None:
        if family == "triangle_bottom":
            height = max(0.0, _at(neckline, first_trough.index) - _at(lower_boundary, first_trough.index))
        elif family == "inverse_head_shoulders":
            head = min(float(pivot.price) for pivot in low_pivots)
            height = max(0.0, _at(neckline, breakout["index"]) - head)
        else:
            height = max(0.0, _at(neckline, breakout["index"]) - float(np.mean(support_values)))
        first_target = float(breakout["close"] + height)
        second_target = float(breakout["close"] + 2.0 * height)
        if first_target > 0 and second_target > 0:
            target_values = [first_target, second_target]
    if breakout is not None:
        post_breakout_support_level = float(_at(neckline, breakout["index"]))
        for index in range(breakout["index"] + 1, len(frame)):
            if close[index] < post_breakout_support_level:
                neckline_invalidation = {"date": _date_text(dates, index, date_strings), "index": int(index),
                                         "level": post_breakout_support_level,
                                         "reason": "close_below_post_breakout_support"}
                break
            if target_values and float(frame.high.iloc[index]) >= target_values[0]:
                target_reached = {"date": _date_text(dates, index, date_strings), "index": int(index),
                                  "price": target_values[0]}
                break
    terminal = "invalidated" if structure_invalidation or neckline_invalidation else (
        "target_reached" if target_reached else None)
    state = terminal or ("confirmed" if breakout else "forming")
    volume = None
    volume_date = None
    if breakout:
        ratio = volume_ratio(frame, breakout["index"])
        if np.isfinite(ratio):
            volume = float(ratio)
            volume_date = breakout["date"]

    end_dates = [event["date"] for event in
                 (structure_invalidation, neckline_invalidation, target_reached) if event]
    structure_end_date = (min(end_dates) if end_dates else
                          (breakout["date"] if breakout else date_strings[-1]))
    signal_eligible = bool(breakout is not None and terminal is None)
    observed_index = len(frame) - 1
    post_breakout_support_level = (float(_at(neckline, breakout["index"]))
                                   if breakout is not None else None)
    neckline_at_observed = (post_breakout_support_level if post_breakout_support_level is not None
                            else float(_at(neckline, observed_index)))
    observed_atr = atr_values[observed_index] if observed_index < len(atr_values) else np.nan
    stop_rules = MAIN_RULES.get("neckline_tests", {})
    plan_stop_buffer = max(float(stop_rules.get("minimum_tolerance_ticks", 2)) * tick_size,
                           float(stop_rules.get("tolerance_atr", 0.5))
                           * float(observed_atr) if np.isfinite(observed_atr) else 0.0)
    plan_stop_level = (neckline_at_observed - plan_stop_buffer) if signal_eligible else float(support_lower)
    decision_index = int(breakout["index"]) if breakout else len(frame) - 1
    def rejection_summary(as_of_index):
        as_of_date = _date_text(dates, int(as_of_index), date_strings)
        counts = {}
        for reason, by_date in neckline_diagnostics.get("rejection_count_by_date", {}).items():
            count = sum(int(number) for date, number in by_date.items() if date <= as_of_date)
            if count:
                counts[reason] = count
        samples = [item for item in neckline_diagnostics.get("rejected_candidates", [])
                   if item.get("known_date") and item["known_date"] <= as_of_date]
        samples.sort(key=lambda item: (
            -float(item.get("maximum_excess", 0.0)), item.get("known_date", ""),
            item.get("reason", ""), item.get("anchor_dates", [])))
        return counts, samples[:12]

    rejected_counts_as_of, rejected_samples_as_of = rejection_summary(decision_index)
    ranked_as_of_decision = sorted(
        [candidate for candidate in neckline_candidates
         if paired_freeze_index(candidate) < decision_index], key=lambda candidate: candidate["score"])

    def candidate_rank_reason(candidate, selected):
        if candidate["tests"][0]["pair_id"] == selected["tests"][0]["pair_id"]:
            return "selected_by_ordered_evidence"
        score, best = candidate["score"], selected["score"]
        if score[0] != best[0]:
            return "fewer_independent_tests" if score[0] > best[0] else "ordered_after_selected"
        if score[1] != best[1]:
            return "shorter_test_span" if score[1] > best[1] else "ordered_after_selected"
        if score[2] != best[2]:
            return "larger_normalized_residual" if score[2] > best[2] else "ordered_after_selected"
        if score[3] != best[3]:
            return "older_latest_test" if score[3] > best[3] else "ordered_after_selected"
        return "anchor_date_tie_break"

    candidate_summaries = [{
        "candidate_id": candidate["tests"][0]["pair_id"],
        "rank": position,
        "selected": candidate["tests"][0]["pair_id"] == selected_tests[0]["pair_id"],
        "line": dict(candidate["line"]),
        "independent_test_count": len(candidate["tests"]),
        "test_span_bars": candidate["tests"][0]["test_span_bars"],
        "normalized_residual": candidate["tests"][0]["selection_score"][2],
        "coverage_start_date": candidate["tests"][0]["coverage_start_date"],
        "coverage_end_date": candidate["tests"][0]["coverage_end_date"],
        "coverage_start_index": candidate["tests"][0]["coverage_start_index"],
        "coverage_end_index": candidate["tests"][0]["coverage_end_index"],
        "paired_frozen_at": _date_text(dates, paired_freeze_index(candidate), date_strings),
        "pressure_boundary_known_at": _date_text(dates, candidate["frozen_index"], date_strings),
        "test_dates": [item["pivot_date"] for item in candidate["tests"]],
        "reason": candidate_rank_reason(candidate, chosen_candidate),
    } for position, candidate in enumerate(ranked_as_of_decision[:8], start=1)]
    candidate_scale = _neckline_candidate_scale()
    older_pressure_tests = [{
        "date": pivot.date, "known_date": pivot.confirmed_date, "level": float(pivot.price),
        "pivot_index": int(pivot.index), "known_index": int(pivot.confirmed_at),
    } for pivot in sorted(
        [pivot for pivot in pivots if pivot.kind == "high"
        and candidate_scale in pivot.scale
         and pivot.index < int(selected_tests[0]["coverage_start_index"])],
        key=lambda pivot: pivot.index, reverse=True)[:12]]
    selection_evidence = {
        "rule_version": str(MAIN_RULES.get(
            "neckline_selection_rule_version", "pattern_matched_pressure_envelope_v7")),
        "consumed_boundary_phase_rule": str(MAIN_RULES.get(
            "neckline_tests", {}).get("consumed_boundary_phase_rule",
                                      "same_structure_exact_frozen_neckline_v2")),
        "breakout_candidate_rule": str(MAIN_RULES.get(
            "breakout_candidate_rule", "candidate_own_frozen_neckline_v1")),
        "selected_candidate_id": selected_candidate_id,
        "selected_rank": next((item["rank"] for item in candidate_summaries
                               if item["selected"]), None),
        "selected_line": dict(neckline),
        "effective_start_date": selected_tests[0]["coverage_start_date"],
        "freeze_date": frozen_date,
        "line_frozen_at": _date_text(dates, paired_freeze_index(chosen_candidate), date_strings),
        "pressure_boundary_known_at": _date_text(
            dates, chosen_candidate["frozen_index"], date_strings),
        "structure_known_at": _date_text(dates, structure_known_at, date_strings),
        "formation_coverage": dict(chosen_candidate["formation_coverage"],
            start_date=date_strings[chosen_candidate["formation_coverage"]["start_index"]],
            end_date=date_strings[chosen_candidate["formation_coverage"]["end_index"]]),
        "decline_check": _neckline_decline_check(family, neckline,
            selected_tests[0]["coverage_start_index"], breakout["index"] if breakout else len(frame)-1),
        "independent_tests": selected_tests,
        "coverage": {"passed": True,
                     "start_date": selected_tests[0]["coverage_start_date"],
                     "end_date": selected_tests[0]["coverage_end_date"],
                     "maximum_excess": selected_tests[0]["coverage_max_excess"]},
        "score_order": ["independent_test_count_desc", "test_span_desc",
                        "normalized_residual_asc", "latest_test_desc", "anchor_dates_asc"],
        "candidates": candidate_summaries,
        "diagnostics_as_of": _date_text(dates, decision_index, date_strings),
        "rejection_counts": rejected_counts_as_of,
        "rejected_candidates": rejected_samples_as_of,
        "older_pressure_tests": older_pressure_tests,
        "retired_segments": selected_retired_segments,
        "early_breakouts": selected_early_breakouts,
        "consumed_boundary_phases": [{
            "candidate_id": item.get("candidate_id"),
            "consumed_index": int(item["consumed_index"]),
            "consumed_date": date_strings[int(item["consumed_index"])],
            "reason": item.get("reason"), "line": dict(item["line"]),
            "tests": item.get("tests", []),
            "phase_tolerance": float(item.get("phase_tolerance", 0.0)),
            "selected": bool(item.get("selected", True)),
        } for item in selected_consumed_phases],
    }
    key_points = {"pivot_scale": pivot_scale, "scale": scale_name,
                  "support_lower": float(support_lower), "support_upper": float(support_upper),
                  "first_trough_index": int(first_trough.index),
                  "first_trough_date": _date_text(dates, first_trough.index, date_strings),
                  "latest_anchor_index": int(last_anchor.index),
                  "latest_anchor_date": _date_text(dates, last_anchor.index, date_strings),
                  "neckline_test_indices": [item["pivot_index"] for item in selected_tests]}
    for number, pivot in enumerate(sequence, 1):
        key_points[f"pivot_{number}_index"] = int(pivot.index)
        key_points[f"pivot_{number}_date"] = _date_text(dates, pivot.index, date_strings)
        key_points[f"pivot_{number}"] = float(pivot.price)
        key_points[f"pivot_{number}_kind"] = pivot.kind
    evidence = [
        {"code": "main_structure", "value": {
            "role": "main", "family": family, "scale": scale_name, "pivot_scale": pivot_scale,
            "signal_eligible": signal_eligible,
            "eligibility_reason": "price_breakout_confirmed" if signal_eligible else (
                "terminal_state" if terminal else "awaiting_fresh_neckline_breakout"),
            "support_lower": float(support_lower), "support_upper": float(support_upper),
            "lower_boundary": (dict(lower_boundary) if family == "triangle_bottom" else None),
            "neckline": neckline, "frozen_at": frozen_date,
            "frozen_at_index": int(frozen_at), "first_trough_date": _date_text(dates, first_trough.index, date_strings),
            "latest_anchor_date": _date_text(dates, last_anchor.index, date_strings),
            "maturity_date": str(maturity_date.date()), "structure_end_date": structure_end_date,
            "minimum_duration_months": minimum_duration_months,
            "signal_is_volume_independent": True,
            "low_position": low_position,
            "terminal_state": terminal,
            "observed_date": _date_text(dates, observed_index, date_strings),
            "neckline_at_observed": neckline_at_observed,
            "post_breakout_support_level": post_breakout_support_level,
            "post_breakout_support_date": breakout["date"] if breakout else None,
            "plan_stop_level": float(plan_stop_level),
            "neckline_rule_version": selection_evidence["rule_version"],
            "neckline_effective_start": selection_evidence["effective_start_date"],
            "neckline_independent_test_count": len(selected_tests),
            "neckline_selection_rank": selection_evidence["selected_rank"],
            "neckline_selection_score": selected_tests[0]["selection_score"],
        }},
        {"code": "neckline_tests", "value": selected_tests},
        {"code": "neckline_selection", "value": selection_evidence},
        {"code": "neckline_breakout", "value": breakout},
        {"code": "weak_crossings", "value": selected_weak_crossings},
        {"code": "support_tests", "value": support_tests},
        {"code": "volume_support", "value": {
            "required": bool(MAIN_RULES.get("volume_required", False)),
            "status": "supportive" if volume is not None and volume >= float(
                MAIN_RULES.get("volume_support_ratio", 1.5)) else (
                "unknown" if volume is None else "not_supportive"),
            "ratio": volume, "date": volume_date,
        }},
        {"code": "targets", "value": {"prices": target_values, "available": bool(breakout),
                                       "formula": "neckline_height_projection" if family != "triangle_bottom"
                                       else "triangle_width_projection",
                                       "triangle_apex_index": float(apex_index) if family == "triangle_bottom" else None,
                                       "triangle_base_width": (float(_at(neckline, first_trough.index)
                                                                      - _at(lower_boundary, first_trough.index))
                                                               if family == "triangle_bottom" else None)}},
        {"code": "low_position", "value": low_position},
        {"code": "main_bottom_reference", "value": {
            "support_lower": float(support_lower), "support_upper": float(support_upper),
            "first_trough_date": _date_text(dates, first_trough.index, date_strings),
            "latest_anchor_date": _date_text(dates, last_anchor.index, date_strings),
            "neckline_at_observed": neckline_at_observed,
            "observed_date": _date_text(dates, observed_index, date_strings),
            "plan_stop_level": float(plan_stop_level),
        }},
        {"code": "structure_invalidation", "value": structure_invalidation},
        {"code": "neckline_invalidation", "value": neckline_invalidation},
        {"code": "target_reached", "value": target_reached},
        {"code": "multiscale_threshold", "value": {
            "scale": pivot_scale, "atr_multiple": next(item[1] for item in PIVOT_SCALES if item[0] == pivot_scale),
            "percent_multiple": next(item[2] for item in PIVOT_SCALES if item[0] == pivot_scale),
        }},
    ]
    identity = pattern_identity_metadata({
        "code": code, "kind": kind, "timeframe": timeframe,
        "pattern_type": MAIN_PATTERN_TYPES[family], "key_points": key_points,
        "neckline": neckline, "evidence": evidence,
    })
    pattern = Pattern(
        pattern_id=identity["instance_id"], code=code, kind=kind, timeframe=timeframe,
        pattern_type=MAIN_PATTERN_TYPES[family], direction=direction, state=state,
        formed_at=_date_text(dates, first_trough.index, date_strings),
        confirmed_at=breakout["index"] if breakout else None, key_points=key_points,
        neckline=neckline, invalidation=float(plan_stop_level), targets=target_values, evidence=evidence,
        volume_ratio=volume, volume_verified=False,
        rule_version=str(MAIN_RULES.get("main_pattern_rule_version", "caisen-v29-main-bottom-v1")),
        data_version=bars.attrs.get("provenance", {}).get("data_version", "unknown"),
        last_observed_date=date_strings[-1],
        confirmation_date=breakout["date"] if breakout else None,
        source_rule_version=str(MAIN_RULES.get("source_rule_version", "caisen-main-bottoms-v16")),
        structure_key=identity["structure_key"],
        boundary_version_id=identity["boundary_version_id"],
    )
    patterns = [] if historical_only else [pattern]
    for retired in neckline_history:
        if retired.get("state") != "invalidated" or not retired.get("candidate") or not retired.get("breakout"):
            continue
        old_candidate = retired["candidate"]
        old_tests = old_candidate["tests"]
        old_breakout = retired["breakout"]
        old_invalidation = retired["invalidation"]
        old_line = dict(old_candidate["line"])
        old_post_breakout_support = float(old_breakout["frozen_neckline"])
        old_structure_frozen_index = max(
            paired_freeze_index(old_candidate), int(structure_known_at))
        snapshot = deepcopy(pattern)
        snapshot.neckline = old_line
        snapshot.state = "invalidated"
        snapshot.confirmed_at = int(old_breakout["index"])
        snapshot.confirmation_date = old_breakout["date"]
        snapshot.key_points["neckline_test_indices"] = [item["pivot_index"] for item in old_tests]
        if family == "triangle_bottom":
            old_height = max(0.0, _at(old_line, first_trough.index) - _at(lower_boundary, first_trough.index))
        elif family == "inverse_head_shoulders":
            old_height = max(0.0, _at(old_line, old_breakout["index"])
                             - min(float(pivot.price) for pivot in low_pivots))
        else:
            old_height = max(0.0, _at(old_line, old_breakout["index"]) - float(np.mean(support_values)))
        old_targets = ([float(old_breakout["close"] + old_height),
                        float(old_breakout["close"] + 2.0 * old_height)] if old_height > 0 else [])
        snapshot.targets = old_targets
        for item in snapshot.evidence:
            value = item["value"]
            if item["code"] == "main_structure":
                value.update({
                    "neckline": old_line,
                    "frozen_at": _date_text(dates, old_structure_frozen_index, date_strings),
                    "frozen_at_index": old_structure_frozen_index,
                    "structure_end_date": retired["invalidation_date"],
                    "signal_eligible": False,
                    "eligibility_reason": "terminal_state",
                    "terminal_state": "invalidated",
                    "neckline_at_observed": old_post_breakout_support,
                    "post_breakout_support_level": old_post_breakout_support,
                    "post_breakout_support_date": old_breakout["date"],
                    "neckline_effective_start": old_tests[0]["coverage_start_date"],
                    "neckline_independent_test_count": len(old_tests),
                    "neckline_selection_score": old_tests[0]["selection_score"],
                })
            elif item["code"] == "neckline_tests":
                item["value"] = old_tests
            elif item["code"] == "neckline_selection":
                historical_candidates = sorted(
                    [candidate for candidate in neckline_candidates
                     if paired_freeze_index(candidate) < int(old_breakout["index"])],
                    key=lambda candidate: candidate["score"])
                historical_summaries = [{
                    "candidate_id": candidate["tests"][0]["pair_id"],
                    "rank": position,
                    "selected": candidate["tests"][0]["pair_id"] == old_tests[0]["pair_id"],
                    "line": dict(candidate["line"]),
                    "independent_test_count": len(candidate["tests"]),
                    "test_span_bars": candidate["tests"][0]["test_span_bars"],
                    "normalized_residual": candidate["tests"][0]["selection_score"][2],
                    "coverage_start_date": candidate["tests"][0]["coverage_start_date"],
                    "coverage_end_date": candidate["tests"][0]["coverage_end_date"],
                    "coverage_start_index": candidate["tests"][0]["coverage_start_index"],
                    "coverage_end_index": candidate["tests"][0]["coverage_end_index"],
                    "test_dates": [test["pivot_date"] for test in candidate["tests"]],
                    "reason": candidate_rank_reason(candidate, old_candidate),
                } for position, candidate in enumerate(historical_candidates, start=1)][:8]
                value.update({
                    "selected_candidate_id": old_tests[0]["pair_id"],
                    "selected_line": old_line,
                    "effective_start_date": old_tests[0]["coverage_start_date"],
                    "freeze_date": _date_text(dates, old_structure_frozen_index, date_strings),
                    "line_frozen_at": _date_text(
                        dates, paired_freeze_index(old_candidate), date_strings),
                    "pressure_boundary_known_at": _date_text(
                        dates, old_candidate["frozen_index"], date_strings),
                    "structure_known_at": _date_text(dates, structure_known_at, date_strings),
                    "formation_coverage": dict(old_candidate["formation_coverage"],
                        start_date=date_strings[old_candidate["formation_coverage"]["start_index"]],
                        end_date=date_strings[old_candidate["formation_coverage"]["end_index"]]),
                    "decline_check": _neckline_decline_check(family, old_line,
                        old_tests[0]["coverage_start_index"], int(old_breakout["index"])),
                    "diagnostics_as_of": old_breakout["date"],
                    "rejection_counts": rejection_summary(int(old_breakout["index"]))[0],
                    "rejected_candidates": rejection_summary(int(old_breakout["index"]))[1],
                    "independent_tests": old_tests,
                    "selected_rank": next((entry["rank"] for entry in historical_summaries
                                           if entry["selected"]), None),
                    "candidates": historical_summaries,
                    "retired_segments": [entry for entry in neckline_history
                                         if entry.get("candidate_id") == old_tests[0]["pair_id"]
                                         and entry.get("breakout_date", "") <= old_breakout["date"]],
                    "early_breakouts": [entry for entry in early_breakouts
                                         if entry.get("candidate_id") == old_tests[0]["pair_id"]
                                         and entry.get("date", "") <= old_breakout["date"]],
                    "coverage": {"passed": True,
                                 "start_date": old_tests[0]["coverage_start_date"],
                                 "end_date": old_tests[0]["coverage_end_date"],
                                 "maximum_excess": old_tests[0]["coverage_max_excess"]},
                })
            elif item["code"] == "neckline_breakout":
                item["value"] = old_breakout
            elif item["code"] == "weak_crossings":
                item["value"] = [crossing for crossing in weak_crossings
                                  if crossing.get("candidate_id") == old_tests[0]["pair_id"]
                                  and crossing.get("date", "") <= old_breakout["date"]]
            elif item["code"] == "neckline_invalidation":
                item["value"] = {"date": retired["invalidation_date"],
                                 "index": int(old_invalidation["index"]),
                                 "level": float(old_invalidation["price"])}
            elif item["code"] == "targets":
                value.update({"prices": old_targets, "available": True})
            elif item["code"] == "target_reached":
                item["value"] = None
        snapshot.structure_key = None
        snapshot.boundary_version_id = None
        old_identity = pattern_identity_metadata(snapshot.to_dict())
        snapshot.pattern_id = old_identity["instance_id"]
        snapshot.structure_key = old_identity["structure_key"]
        snapshot.boundary_version_id = old_identity["boundary_version_id"]
        patterns.append(snapshot)
    return patterns


def detect_main_bottoms(bars: pd.DataFrame, code="", kind="stock", timeframe="1d", tick_size=None,
                        background_bars=None):
    """Detect long-cycle W, inverse head-and-shoulders, and triangle bottoms.

    `bars` must be a causal daily prefix or calendar-complete weekly prefix.
    The complete supplied prefix is used for as-of low-position evidence; only
    the trailing five daily or ten weekly years supply structure pivots.
    """
    if bars is None or bars.empty:
        return []
    frame = bars.copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return []
    full_history = frame.copy()
    background = (background_bars.copy() if background_bars is not None else full_history.copy())
    background["date"] = pd.to_datetime(background.date, errors="coerce")
    background = background[background.date.notna() & (background.date <= frame.date.iloc[-1])].copy()
    timeframe_rules = MAIN_RULES.get("timeframes", {})
    weekly = timeframe in {"1w", "weekly", "week"}
    history_years = int(timeframe_rules.get("1w" if weekly else "1d", {}).get(
        "structure_window_calendar_years", 10 if weekly else 5))
    history_start = pd.Timestamp(frame.date.iloc[-1]) - pd.DateOffset(years=history_years)
    frame = frame[frame.date >= history_start].reset_index(drop=True)
    if "segment_id" in frame and frame.segment_id.nunique() > 1:
        segments = []
        for _, group in frame.groupby("segment_id", sort=True):
            segments.extend(detect_main_bottoms(group, code, kind, timeframe, tick_size,
                                                background_bars=background))
        return segments
    tick = _tick_size(bars, tick_size)
    atr_values = atr(frame, period=14).to_numpy(dtype=float)
    events = []
    horizon = f"{history_years}y"
    all_scale_pivots = detect_multiscale_pivots(frame, timeframe=timeframe, tick_size=tick)
    pivot_by_identity = {}
    for scale_pivots in all_scale_pivots.values():
        for pivot in scale_pivots:
            identity = (pivot.index, pivot.kind, pivot.price)
            current = pivot_by_identity.get(identity)
            rank = float(pivot.scale.split("ATR")[0])
            current_rank = float(current.scale.split("ATR")[0]) if current else -1.0
            if current is None or rank > current_rank:
                pivot_by_identity[identity] = pivot
    neckline_pivots = sorted(pivot_by_identity.values(), key=lambda pivot: pivot.index)
    scale_by_location = {(pivot.index, pivot.kind): pivot for pivot in pivot_by_identity.values()}
    for touch in _detect_neckline_test_pivots(frame, atr_values, tick):
        scale_pivot = scale_by_location.get((touch.index, touch.kind))
        if scale_pivot is None:
            neckline_pivots.append(touch)
        else:
            neckline_pivots.append(Pivot(touch.index, touch.date, touch.price, touch.kind,
                                         touch.confirmed_at, touch.confirmed_date,
                                         f"{scale_pivot.scale}|{_neckline_candidate_scale()}"))
    neckline_pivots.sort(key=lambda pivot: (pivot.index, pivot.scale))
    for pivot_scale, _, _ in PIVOT_SCALES:
        pivots = all_scale_pivots[pivot_scale]
        for family, sequence in _structure_candidates(
                pivots, timeframe, atr_values, tick,
                low_values=pd.to_numeric(frame.low, errors="coerce").to_numpy(dtype=float),
                shoulder_pivots=neckline_pivots):
            boundary_phases = []
            seen_instances = set()
            # The weekly detail/scan path must preserve every independently
            # frozen boundary for a dated structure. Re-run the causal lifecycle
            # with each prior frozen line excluded to discover its next line.
            # Daily streaming has its own incremental owner and remains single-pass.
            while True:
                event = _make_main_pattern(
                    frame, code, kind, timeframe, family, sequence, neckline_pivots,
                    atr_values, tick, pivot_scale, horizon, background_bars=background,
                    consumed_boundary_phases=boundary_phases)
                if event is None:
                    break
                candidates = event if isinstance(event, list) else [event]
                new_candidates = [item for item in candidates
                                  if item is not None and item.pattern_id not in seen_instances]
                if not new_candidates:
                    break
                events.extend(new_candidates)
                seen_instances.update(item.pattern_id for item in new_candidates)
                if not weekly:
                    break
                frozen_count = 0
                for item in new_candidates:
                    if not pattern_instance_is_frozen(item.to_dict()):
                        continue
                    detail = {row["code"]: row["value"] for row in item.evidence}
                    selection = detail.get("neckline_selection", {})
                    line = item.neckline or selection.get("selected_line")
                    if not line:
                        continue
                    phase = {
                        "line": dict(line), "tests": detail.get("neckline_tests", []),
                        "selected": True,
                    }
                    if any(_same_neckline_boundary_phase(previous, phase, tick)
                           for previous in boundary_phases):
                        continue
                    boundary_phases.append(phase)
                    frozen_count += 1
                if frozen_count == 0:
                    break
    # Stable IDs deduplicate repeated sequence walks. Prefer the complete base
    # with stronger pressure evidence before using pivot scale as a tie-break.
    unique = {event.pattern_id: event for event in events}
    events = list(unique.values())
    if events:
        rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
        active = [event for event in events if event.state in {"forming", "confirmed"}]
        pool = active or events
        internal_parents = _nested_main_parent_map(active, active)
        primary_pool = [event for event in pool if event.pattern_id not in internal_parents] or pool

        def primary_rank(event):
            test_evidence = next((item["value"] for item in event.evidence
                                  if item["code"] == "neckline_tests"), [])
            test_count = len(test_evidence)
            test_span = max((int(item.get("test_span_bars", 0)) for item in test_evidence), default=0)
            duration = (pd.Timestamp(event.last_observed_date) - pd.Timestamp(event.formed_at)).days
            head_depth = 0.0
            if event.pattern_type == "inverse_head_shoulders":
                left_shoulder, head, right_shoulder = (event.key_points.get(f"pivot_{number}")
                                                       for number in (1, 2, 3))
                if all(value is not None for value in (left_shoulder, head, right_shoulder)):
                    head_depth = max(0.0, (float(left_shoulder) + float(right_shoulder)) / 2.0
                                     - float(head))
            event_date = event.confirmation_date or event.last_observed_date
            duration = (pd.Timestamp(event.last_observed_date) - pd.Timestamp(event.formed_at)).days
            scale_rank = rank.get(event.key_points.get("pivot_scale"), -1)
            right_shoulder_date = (event.key_points.get("pivot_3_date", "")
                                   if event.pattern_type == "inverse_head_shoulders" else "")
            left_shoulder_date = (event.key_points.get("pivot_1_date", "")
                                  if event.pattern_type == "inverse_head_shoulders" else "")
            return (event.state in {"confirmed", "target_reached"}, event_date,
                    event.pattern_type == "w_bottom",
                    scale_rank if event.pattern_type == "inverse_head_shoulders" else -1,
                    right_shoulder_date, left_shoulder_date,
                    duration if event.pattern_type == "inverse_head_shoulders" else 0,
                    test_count, test_span, head_depth, event.formed_at,
                    scale_rank, event.pattern_id)
        primary = max(primary_pool, key=lambda event: (
            *primary_rank(event),
        ))
        for event in events:
            structure = next(item["value"] for item in event.evidence if item["code"] == "main_structure")
            parent = internal_parents.get(event.pattern_id)
            if parent is not None:
                structure.update({"role": "internal", "signal_eligible": False,
                                  "eligibility_reason": "nested_in_known_main_structure",
                                  "parent_pattern_id": parent.pattern_id})
            event.evidence.append({"code": "main_selection", "value": {
                "selected": event.pattern_id == primary.pattern_id,
                "decided_at": str(frame.date.iloc[-1].date()),
                "primary_pattern_id": primary.pattern_id,
                "parent_pattern_id": parent.pattern_id if parent is not None else None,
                "reason": ("nested_in_known_main_structure" if parent is not None else
                           "latest_confirmation_then_pattern_family_then_coarsest_structure_then_duration_then_pressure_evidence"),
            }})
    return sorted(events, key=lambda event: (event.formed_at, event.pattern_type, event.pattern_id))


def _pattern_dates(pattern, bars_by_timeframe):
    dates = []
    for key, value in pattern.key_points.items():
        if key.startswith("pivot_") and key.endswith("_date") and isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce")
            if not pd.isna(parsed):
                dates.append(pd.Timestamp(parsed))
    if dates:
        return [pd.Timestamp(value).normalize() for value in dates]
    bars = bars_by_timeframe.get(pattern.timeframe) if isinstance(bars_by_timeframe, dict) else bars_by_timeframe
    if bars is not None and not bars.empty:
        index_to_date = pd.to_datetime(bars.date, errors="coerce").reset_index(drop=True)
        for key, value in pattern.key_points.items():
            if key.startswith("pivot_") and key.endswith("_index") and isinstance(value, int) and 0 <= value < len(index_to_date):
                dates.append(index_to_date.iloc[value])
    if not dates:
        dates = [pd.Timestamp(pattern.formed_at)]
        if pattern.confirmation_date:
            dates.append(pd.Timestamp(pattern.confirmation_date))
    return [pd.Timestamp(value).normalize() for value in dates if not pd.isna(value)]


def _main_parent_eligible(pattern):
    if (pattern.pattern_type not in MAIN_PATTERN_TYPES or pattern.direction != "bullish"
            or pattern.state not in {"forming", "confirmed"} or pattern.parent_id):
        return False
    evidence = {item.get("code"): item.get("value")
                for item in pattern.evidence if isinstance(item, dict)}
    structure = evidence.get("main_structure")
    if not isinstance(structure, dict):
        return False
    selection = evidence.get("main_selection") or {}
    suppression = evidence.get("execution_suppression") or {}
    if structure.get("role") not in {None, "main"} or selection.get("selected") is False:
        return False
    if suppression.get("suppressed") is True:
        return False
    # signal_eligible describes breakout readiness, not whether a forming main
    # structure can contain and suppress one of its internal candidates.
    return True


def _pattern_anchor_window(pattern, parent=False):
    # Suppression is about structural nesting, not overlapping observation
    # lifetimes. `structure_end_date` and `last_observed_date` can be much
    # later than the actual pivot set; using either as the parent's right
    # edge makes an old confirmed bottom suppress every later bottom. Bound
    # both sides by the actual structural low anchors (or explicit first /
    # latest anchor dates when available) so containment remains directional.
    dates = list(_pattern_low_anchor_dates(pattern))
    evidence = {item.get("code"): item.get("value")
                for item in pattern.evidence if isinstance(item, dict)}
    structure = evidence.get("main_structure") or {}
    for key in ("first_trough_date", "latest_anchor_date"):
        value = structure.get(key)
        if value:
            parsed = pd.to_datetime(value, errors="coerce")
            if not pd.isna(parsed):
                dates.append(pd.Timestamp(parsed).normalize())
    if not dates:
        dates = _pattern_dates(pattern, None)
    if not dates:
        return None
    return min(dates), max(dates)


def _pattern_structure_signature(pattern):
    return pattern_identity_metadata(pattern.to_dict()).get("structure_key")


def _pattern_low_anchor_dates(pattern):
    dates = set()
    for key, value in pattern.key_points.items():
        if not (key.startswith("pivot_") and key.endswith("_date")
                and isinstance(value, str)):
            continue
        number = key[len("pivot_"):-len("_date")]
        if pattern.key_points.get(f"pivot_{number}_kind") != "low":
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            dates.add(pd.Timestamp(parsed).normalize())
    return dates


def _nested_main_parent_map(parents, children, action_date=None):
    """Map each contained bullish structure to one known, live outer main structure.

    Containment is directional: the selected outer structure owns an inner
    candidate. Pivot detection scale does not reverse that relationship; a
    smaller-scale outer shape can still be the parent when its time window
    strictly contains the child's.
    """
    parent_rows = []
    for parent in parents:
        if not _main_parent_eligible(parent):
            continue
        window = _pattern_anchor_window(parent, parent=True)
        if window is None:
            continue
        evidence = {item.get("code"): item.get("value")
                    for item in parent.evidence if isinstance(item, dict)}
        selection = evidence.get("neckline_selection") or {}
        known_values = [selection.get(key) for key in (
            "line_frozen_at", "freeze_date", "structure_known_at") if selection.get(key)]
        structure = evidence.get("main_structure") or {}
        if structure.get("frozen_at"):
            known_values.append(structure["frozen_at"])
        available = (evidence.get("boundary_available_at") or {}).get("date")
        if available:
            known_values.append(available)
        parent_known_at = max((pd.Timestamp(value).normalize() for value in known_values),
                              default=window[0])
        structure_start = pd.to_datetime(
            structure.get("first_trough_date"), errors="coerce")
        if pd.isna(structure_start):
            structure_start = window[0]
        parent_rows.append((parent, window, parent_known_at,
                            _pattern_structure_signature(parent),
                            _pattern_low_anchor_dates(parent),
                            pd.Timestamp(structure_start).normalize()))

    def enclosing_parents(child, child_window, child_start, child_signature, child_lows,
                          current_date):
        matches = []
        for parent, parent_window, known_at, parent_signature, parent_lows, parent_start in parent_rows:
            if parent.pattern_id == child.pattern_id or parent.timeframe != child.timeframe:
                continue
            if parent_signature == child_signature:
                continue
            # Competing readings that share the actual structural lows are
            # alternatives, not parent/child suppressions.
            if (parent.pattern_type == child.pattern_type == "w_bottom"
                    and parent_lows.intersection(child_lows)):
                continue
            if len(parent_lows.intersection(child_lows)) >= 2:
                continue
            if known_at > current_date:
                continue
            if (parent_start < child_start and parent_window[0] < child_window[0]
                    and child_window[1] < parent_window[1]):
                matches.append((parent_window[0], parent_window[1], parent))
        return matches

    # Determine suppression before choosing suppressors. Otherwise a middle
    # structure that is itself suppressed by an outer structure can still act
    # as a parent for another child in this same pass, creating a cascading
    # parent/child chain from candidates that are not selected as main.
    currently_suppressed_parents = set()
    for candidate, candidate_window, candidate_known, candidate_signature, candidate_lows, candidate_start in parent_rows:
        candidate_date = (pd.Timestamp(action_date).normalize() if action_date is not None else
                          pd.Timestamp(candidate.confirmation_date or candidate.last_observed_date
                                       or candidate_window[1]).normalize())
        if candidate_known > candidate_date:
            continue
        if enclosing_parents(candidate, candidate_window, candidate_start,
                             candidate_signature, candidate_lows, candidate_date):
            currently_suppressed_parents.add(candidate.pattern_id)

    nested = {}
    for child in children:
        if child.direction != "bullish" or child.state not in {"forming", "confirmed"}:
            continue
        child_evidence = {item.get("code"): item.get("value")
                          for item in child.evidence if isinstance(item, dict)}
        child_window = _pattern_anchor_window(child)
        if child_window is None:
            continue
        current_date = (pd.Timestamp(action_date).normalize() if action_date is not None else
                        pd.Timestamp(child.confirmation_date or child.last_observed_date
                                     or child_window[1]).normalize())
        child_structure = child_evidence.get("main_structure") or {}
        child_start = pd.to_datetime(child_structure.get("first_trough_date"), errors="coerce")
        if pd.isna(child_start):
            child_start = child_window[0]
        child_start = pd.Timestamp(child_start).normalize()
        child_signature = _pattern_structure_signature(child)
        child_lows = _pattern_low_anchor_dates(child)
        matches = [match for match in enclosing_parents(
            child, child_window, child_start, child_signature, child_lows, current_date)
            if match[2].pattern_id not in currently_suppressed_parents]
        if matches:
            nested[child.pattern_id] = max(
                matches, key=lambda item: (item[0], item[1], item[2].pattern_id))[2]
    return nested


def link_main_bottom_children(main_events: Iterable[Pattern], existing_patterns: Iterable[Pattern], bars_by_timeframe=None):
    """Return copied events with causal same-timeframe parent/child relations.

    A selected main bottom suppresses only a strictly contained, same-timeframe
    bullish child. Use the parent's recorded structural end; an unended forming
    parent extends through its current observation. Timeframes are never crossed.
    """
    mains = [replace(event, key_points=deepcopy(event.key_points), evidence=deepcopy(event.evidence), flags=list(event.flags))
             for event in main_events]
    legacy = [replace(event, key_points=deepcopy(event.key_points), evidence=deepcopy(event.evidence), flags=list(event.flags))
              for event in existing_patterns]
    all_events = mains + legacy
    parents = [event for event in mains
               if next((item.get("value", {}).get("selected") for item in event.evidence
                        if item.get("code") == "main_selection"), False)]
    child_candidates = [child for child in legacy
                        if child.direction == "bullish"
                        and not any(item.get("code") == "main_structure" for item in child.evidence)]
    parent_by_child = _nested_main_parent_map(parents, child_candidates)
    for child in all_events:
        parent = parent_by_child.get(child.pattern_id)
        if parent is None:
            continue
        child_dates = _pattern_dates(child, bars_by_timeframe)
        if not child_dates:
            continue
        child_start, child_end = min(child_dates), max(child_dates)
        structure = next((item.get("value", {}) for item in parent.evidence
                          if item.get("code") == "main_structure"), {})
        child.parent_id = parent.pattern_id
        child.evidence.append({"code": "execution_suppression", "value": {
            "suppressed": True, "reason": "nested_in_main_bottom", "parent_id": parent.pattern_id,
            "first_trough_date": structure["first_trough_date"],
            "structure_end_date": structure.get("structure_end_date"),
        }})
        child_ids = [item["value"] for item in parent.evidence if item["code"] == "main_children"]
        if child_ids:
            child_ids = child_ids[0]
        else:
            child_ids = []
        child_ids.append({"pattern_id": child.pattern_id, "pattern_type": child.pattern_type,
                          "anchor_start_date": str(child_start.date()), "anchor_end_date": str(child_end.date())})
        parent.evidence = [item for item in parent.evidence if item["code"] != "main_children"]
        parent.evidence.append({"code": "main_children", "value": child_ids})
    return all_events
