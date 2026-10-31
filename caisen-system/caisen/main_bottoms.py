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
from .schemas import Pattern, Pivot


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
                    pivots.append(Pivot(high_index, str(dates[high_index].date()), float(highs[high_index]),
                                        "high", index, str(dates[index].date()),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction, low_index = -1, index
                elif up_ready:
                    pivots.append(Pivot(low_index, str(dates[low_index].date()), float(lows[low_index]),
                                        "low", index, str(dates[index].date()),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction, high_index = 1, index
            elif direction > 0:
                reversal = threshold(high_index, highs[high_index])
                if np.isfinite(reversal) and highs[index] > highs[high_index]:
                    high_index = index
                    reversal = threshold(high_index, highs[high_index])
                if np.isfinite(reversal) and highs[high_index] - closes[index] >= reversal:
                    pivots.append(Pivot(high_index, str(dates[high_index].date()), float(highs[high_index]),
                                        "high", index, str(dates[index].date()),
                                        f"{atr_multiple:g}ATR|{percent_multiple:.0%}"))
                    direction = -1
                    low_index = index
            else:
                reversal = threshold(low_index, lows[low_index])
                if lows[index] < lows[low_index]:
                    low_index = index
                    reversal = threshold(low_index, lows[low_index])
                if np.isfinite(reversal) and closes[index] - lows[low_index] >= reversal:
                    pivots.append(Pivot(low_index, str(dates[low_index].date()), float(lows[low_index]),
                                        "low", index, str(dates[index].date()),
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
    return float(line["slope"]) * int(index) + float(line["intercept"])


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
    """Freeze each resistance touch after a causal one-ATR departure."""
    frame = bars.reset_index(drop=True)
    highs = pd.to_numeric(frame.high, errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
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
        departure = max(float(known_atr), 2.0 * float(tick_size))
        if closes[index] <= highs[candidate_index] - departure:
            result.append(Pivot(candidate_index, str(dates[candidate_index].date()),
                                float(highs[candidate_index]), "high", index,
                                str(dates[index].date()), "neckline_test_1ATR"))
            candidate_index = None
    return result


def _tick_size(bars, requested):
    if requested is not None:
        return max(float(requested), 0.000001)
    return max(float(bars.attrs.get("quality", {}).get("tick_size", 0.01)), 0.000001)


def _independent_neckline_tests(pivots, first_trough, last_anchor, atr_values, dates, tick_size, timeframe,
                                lower_index=None, upper_index=None, minimum_level=None,
                                maximum_level=None, target_level=None, latest_close=None,
                                latest_buffer=0.0):
    """Find two independent resistance touches and their departure evidence."""
    lookback = pd.Timestamp(dates[first_trough.index]) - pd.DateOffset(years=NECKLINE_LOOKBACK_YEARS)
    candidates = [pivot for pivot in pivots if pivot.kind == "high" and "neckline_test_1ATR" in pivot.scale
                  and dates[pivot.index] >= lookback
                  and (lower_index is None or pivot.index >= lower_index)
                  and (upper_index is None or pivot.index <= upper_index)
                  and (maximum_level is None or float(pivot.price) <= float(maximum_level))
                  and pivot.confirmed_at < len(dates)]
    neckline_rules = MAIN_RULES.get("neckline_tests", {})
    weekly = timeframe in {"1w", "weekly", "week"}
    minimum_gap = int(neckline_rules.get(
        "minimum_separation_weekly_bars" if weekly else "minimum_separation_daily_sessions",
        2 if weekly else 10))
    tolerance_atr = float(neckline_rules.get("tolerance_atr", 0.5))
    minimum_ticks = int(neckline_rules.get("minimum_tolerance_ticks", 2))
    high_by_index = {}
    high_tolerance_by_index = {}
    low_by_index = {}
    for pivot in pivots:
        target = high_by_index if pivot.kind == "high" else low_by_index if pivot.kind == "low" else None
        if target is not None:
            comparator = max if pivot.kind == "high" else min
            target[pivot.index] = comparator(target.get(pivot.index, -np.inf if pivot.kind == "high" else np.inf),
                                             float(pivot.price))
            if pivot.kind == "high":
                known_atr = atr_values[pivot.confirmed_at] if pivot.confirmed_at < len(atr_values) else np.nan
                tolerance = max(minimum_ticks * tick_size,
                                tolerance_atr * float(known_atr) if np.isfinite(known_atr) else 0.0)
                high_tolerance_by_index[pivot.index] = max(high_tolerance_by_index.get(pivot.index, 0.0),
                                                           tolerance)
    high_indices = np.asarray(sorted(high_by_index), dtype=int)
    high_prices = np.asarray([high_by_index[index] for index in high_indices], dtype=float)
    high_tolerances = np.asarray([high_tolerance_by_index[index] for index in high_indices], dtype=float)
    low_indices = np.asarray(sorted(low_by_index), dtype=int)
    low_prices = np.asarray([low_by_index[index] for index in low_indices], dtype=float)
    tests = []
    for left, right in combinations(candidates, 2):
        if right.index - left.index < minimum_gap:
            continue
        left_atr = atr_values[left.confirmed_at]
        right_atr = atr_values[right.confirmed_at]
        left_tolerance = max(minimum_ticks * tick_size, tolerance_atr * left_atr) if np.isfinite(left_atr) else minimum_ticks * tick_size
        right_tolerance = max(minimum_ticks * tick_size, tolerance_atr * right_atr) if np.isfinite(right_atr) else minimum_ticks * tick_size
        # Do not require the two touches to cluster around a horizontal level:
        # a genuinely inclined neckline is also valid. Family geometry below
        # validates the fitted line against the intervening structure.
        if minimum_level is not None and min(float(left.price), float(right.price)) < float(minimum_level):
            continue
        line = _line((left.index, left.price), (right.index, right.price))
        # A candidate line is not resistance if an intervening confirmed swing
        # high already exceeded it by more than its own known tolerance.
        high_start = bisect_right(high_indices, max(left.index, first_trough.index))
        high_end = bisect_left(high_indices, right.index)
        if high_end > high_start:
            indices = high_indices[high_start:high_end]
            prices = high_prices[high_start:high_end]
            tolerances = high_tolerances[high_start:high_end]
            if np.any(prices > float(line["slope"]) * indices + float(line["intercept"]) + tolerances):
                continue
        low_start = bisect_right(low_indices, left.index)
        low_end = bisect_left(low_indices, right.index)
        if low_end <= low_start:
            continue
        left_level, right_level = _at(line, left.index), _at(line, right.index)
        departure = min(left_level, right_level) - float(np.min(low_prices[low_start:low_end]))
        reference_atr = max(float(left_atr) if np.isfinite(left_atr) else 0.0,
                            float(right_atr) if np.isfinite(right_atr) else 0.0)
        if reference_atr <= 0 or departure < float(neckline_rules.get("minimum_departure_atr", 1.0)) * reference_atr:
            continue
        tests.append((left, right, line, departure, reference_atr, left_tolerance, right_tolerance))
    if not tests:
        return []
    current_horizontal_pairs = [item for item in tests
                                if item[1].index >= first_trough.index
                                and abs(float(item[0].price) - float(item[1].price))
                                <= item[5] + item[6]]
    if current_horizontal_pairs:
        tests = current_horizontal_pairs
    def pivot_scale_rank(pivot):
        try:
            return float(pivot.scale.split("ATR")[0])
        except (TypeError, ValueError):
            return 0.0
    def associated_rim(item):
        pair_mean = float(np.mean([item[0].price, item[1].price]))
        snap = float(MAIN_RULES.get("neckline_tests", {}).get(
            "w_bottom_rim_snap_relative", 0.05))
        if target_level is not None and abs(pair_mean - float(target_level)) / max(abs(float(target_level)), 1e-9) <= snap:
            return float(target_level)
        return min(float(item[0].price), float(item[1].price))
    def rim_rank(item):
        rim = associated_rim(item)
        close_pass = latest_close is not None and rim + float(latest_buffer) < float(latest_close)
        return (0 if close_pass else 1, -rim,
                0 if abs(float(item[0].price) - float(item[1].price))
                <= item[5] + item[6] else 1,
                -(item[1].index - item[0].index),
                -min(pivot_scale_rank(item[0]), pivot_scale_rank(item[1])),
                max(item[0].confirmed_at, item[1].confirmed_at), item[0].index)
    prior_pairs = [item for item in tests
                   if item[1].index < first_trough.index
                   and max(item[0].confirmed_at, item[1].confirmed_at) <= first_trough.confirmed_at]
    if prior_pairs:
        # Left-side tests are valid when they associate with the current W
        # rim. Within that association band, retain the highest horizontal
        # pre-bottom rim; this lets a major outer resistance remain distinct
        # from a lower inner swing peak.
        chosen = min(prior_pairs, key=rim_rank)
    else:
        # Select the best tested resistance level within the candidate's
        # tolerance band, then freeze it. This avoids letting one stale
        # high-price touch tilt a newer horizontal neckline.
        chosen = min(tests, key=rim_rank)
    left, right, line, departure, reference_atr, left_tolerance, right_tolerance = chosen
    result = []
    for pivot, tolerance in ((left, left_tolerance), (right, right_tolerance)):
        result.append({
            "group_id": f"neck-touch-{pivot.index}",
            "pair_id": f"neck-pair-{left.index}-{right.index}",
            "pivot_index": int(pivot.index),
            "pivot_date": str(dates[pivot.index].date()),
            "known_index": int(pivot.confirmed_at),
            "known_date": str(dates[pivot.confirmed_at].date()),
            "pivot_scale": pivot.scale,
            "level": float(pivot.price),
            "departure_atr": float(departure / reference_atr),
            "tolerance": float(tolerance),
            "valid": True,
            "source_pivot_indices": [int(pivot.index)],
        })
    return result


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


def _ihs_valley_intact(left, head, right, pivots, atr_values=None, tick_size=0.01):
    return not any(pivot.kind == "low" and left.index <= pivot.index <= right.index
                   and float(pivot.price) < float(head.price) - _pivot_tolerance(
                       pivot, atr_values, tick_size)
                   for pivot in pivots)


def _structure_candidates(pivots, timeframe, atr_values=None, tick_size=0.01):
    candidates = []
    lows = [pivot for pivot in pivots if pivot.kind == "low"]
    highs = [pivot for pivot in pivots if pivot.kind == "high"]
    vectors = _triangle_pivot_vectors(pivots, atr_values, tick_size)
    high_indices, high_prices = vectors[0], vectors[1]
    high_pivot_by_index = {}
    for pivot in highs:
        current = high_pivot_by_index.get(pivot.index)
        if current is None or pivot.price > current.price:
            high_pivot_by_index[pivot.index] = pivot
    for first in lows:
        seconds = []
        for second in lows:
            if second.index <= first.index:
                continue
            maximum_gap = float(MAIN_RULES.get("w_bottom", {}).get("maximum_trough_gap_ratio", 0.15))
            if abs(float(first.price) - float(second.price)) / max(float(first.price), float(second.price)) > maximum_gap:
                continue
            between_highs = [pivot for pivot in highs if first.index < pivot.index < second.index]
            if not between_highs:
                continue
            inner_lows = [pivot for pivot in lows if first.index < pivot.index < second.index]
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
                gap_ratio = abs(float(first.price) - float(second.price)) / max(float(first.price), float(second.price))
                seconds.append((gap_ratio, -(second.index - first.index), second))
        if seconds:
            second = min(seconds, key=lambda item: (item[0], item[1], item[2].index))[2]
            candidates.append(("w_bottom", [first, second]))
    for head in lows:
        lefts = [pivot for pivot in lows if pivot.index < head.index and pivot.price > head.price]
        rights = [pivot for pivot in lows if pivot.index > head.index and pivot.price > head.price]
        shoulder_pairs = []
        for left in lefts:
            for right in rights:
                neck_highs = [pivot for pivot in highs if left.index < pivot.index < right.index]
                if len(neck_highs) < 2:
                    continue
                depth = max(float(pivot.price) for pivot in neck_highs) - float(head.price)
                if depth <= 0:
                    continue
                shoulder_gap = abs(float(left.price) - float(right.price)) / depth
                if shoulder_gap > 0.07:
                    continue
                if not _ihs_valley_intact(left, head, right, lows, atr_values, tick_size):
                    continue
                local_span = (head.index - left.index) + (right.index - head.index)
                shoulder_pairs.append((local_span, shoulder_gap, -(right.index - left.index),
                                       left.index, [left, head, right]))
        if shoulder_pairs:
            candidates.append(("inverse_head_shoulders",
                               min(shoulder_pairs, key=lambda item: item[:4])[4]))
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
                       atr_values, tick_size, pivot_scale, scale_name, background_bars=None):
    frame = bars.reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    close = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=float)
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
    tests = _independent_neckline_tests(
        pivots, first_trough, last_anchor, atr_values, dates, tick_size, timeframe,
        # Allow one prior-year touch to anchor the level, but require the
        # second independent touch to occur after this bottom begins.
        lower_index=(int(np.searchsorted(
            dates.values,
            (pd.Timestamp(dates[first_trough.index]) - pd.DateOffset(years=NECKLINE_LOOKBACK_YEARS)).to_datetime64(),
            side="left")) if family == "w_bottom" else first_trough.index),
        # Later independent retests can establish an outer rim after the
        # nominal W anchors; they remain usable only when confirmed within
        # this observed prefix and before the breakout frozen below.
        upper_index=(len(frame) - 1 if family == "w_bottom" else last_anchor.index),
        minimum_level=minimum_neckline,
        maximum_level=(float(max(pivot.price for pivot in w_middle_highs)) *
                       (1.0 + float(MAIN_RULES.get("neckline_tests", {}).get(
                           "w_bottom_rim_association_ceiling_relative", 0.2)))
                       if w_middle_highs else None),
        target_level=(float(max(pivot.price for pivot in w_middle_highs))
                      if w_middle_highs else None),
        latest_close=float(close[-1]),
        latest_buffer=max(
            float(MAIN_RULES.get("breakout", {}).get("minimum_buffer_ticks", 2)) * tick_size,
            float(MAIN_RULES.get("breakout", {}).get("close_buffer_previous_atr_multiple", 1.0))
            * float(atr_values[-2]) if len(atr_values) > 1 and np.isfinite(atr_values[-2]) else 0.0,
        ),
    )
    if len(tests) < NECKLINE_TEST_COUNT:
        return None
    if family == "triangle_bottom":
        high_pivots = [pivot for pivot in sequence if pivot.kind == "high"]
        low_pivots = [pivot for pivot in sequence if pivot.kind == "low"]
        if len(high_pivots) < 2 or len(low_pivots) < 2:
            return None
        neckline = _line((high_pivots[0].index, high_pivots[0].price),
                         (high_pivots[-1].index, high_pivots[-1].price))
        # Independent touches corroborate the sloped resistance within the
        # same known ATR/tick tolerance used for horizontal tests.
        tests = [test for test in tests if abs(test["level"] - _at(neckline, test["pivot_index"]))
                 <= max(test["tolerance"], 2.0 * tick_size)]
        if len(tests) < NECKLINE_TEST_COUNT:
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
        selected_tests = tests[:2]
        neckline_pivots = [next(pivot for pivot in pivots if pivot.index == item["pivot_index"]
                                and "neckline_test_1ATR" in pivot.scale)
                           for item in selected_tests]
        neckline = _line((neckline_pivots[0].index, neckline_pivots[0].price),
                         (neckline_pivots[1].index, neckline_pivots[1].price))
        if family in {"w_bottom", "inverse_head_shoulders"}:
            neckline = _qualify_neckline_slope(
                neckline, neckline_pivots[0].index, neckline_pivots[1].index)
            if neckline is None:
                return None
        if family == "w_bottom":
            # Two independent tests within the accepted tolerance describe one
            # horizontal resistance zone. Extrapolating a tiny historical
            # difference across years creates a false steep neckline drift.
            test_gap = abs(float(neckline_pivots[0].price) - float(neckline_pivots[1].price))
            test_tolerances = []
            for pivot in neckline_pivots:
                known_atr = (atr_values[pivot.confirmed_at]
                             if pivot.confirmed_at < len(atr_values) else np.nan)
                test_tolerances.append(max(2.0 * tick_size,
                                           0.5 * float(known_atr) if np.isfinite(known_atr) else 0.0))
            # Each touch may sit at either edge of the same tolerance zone;
            # the total allowed gap is therefore the sum of both per-touch
            # tolerances. A materially inclined pair still retains its slope.
            if test_gap <= sum(test_tolerances):
                tested_level = float(np.mean([pivot.price for pivot in neckline_pivots]))
                middle_rim = float(max(pivot.price for pivot in w_middle_highs))
                rim_match = float(MAIN_RULES.get("neckline_tests", {}).get(
                    "w_bottom_rim_snap_relative", 0.05))
                # A tested band coincident with the current W rim uses the
                # structural peak. A distinct, independently retested outer
                # band remains its own neckline; its lower tested edge gives
                # a conservative frozen breakout boundary.
                horizontal_level = (middle_rim if abs(tested_level - middle_rim) / middle_rim <= rim_match
                                    else min(float(pivot.price) for pivot in neckline_pivots))
                neckline = {"slope": 0.0,
                            "intercept": horizontal_level}
        neckline_level = _at(neckline, last_anchor.index)
        support_tests = []
        if family == "w_bottom":
            for pivot in w_middle_highs:
                known_atr = atr_values[pivot.confirmed_at] if pivot.confirmed_at < len(atr_values) else np.nan
                tolerance = max(float(MAIN_RULES.get("neckline_tests", {}).get("minimum_tolerance_ticks", 2)) * tick_size,
                                float(MAIN_RULES.get("neckline_tests", {}).get("tolerance_atr", 0.5))
                                * float(known_atr) if np.isfinite(known_atr) else 0.0)
                if _at(neckline, pivot.index) < float(pivot.price) - tolerance:
                    return None
        if family == "inverse_head_shoulders":
            head = next(pivot for pivot in sequence if pivot is not low_pivots[0]
                        and pivot is not low_pivots[-1] and pivot.kind == "low")
            if not _ihs_valley_intact(low_pivots[0], head, low_pivots[-1],
                                      pivots, atr_values, tick_size):
                return None
            neck_at_head = _at(neckline, head.index)
            depth = neck_at_head - float(head.price)
            shoulder_gap = abs(float(low_pivots[0].price) - float(low_pivots[-1].price))
            if depth <= 0 or shoulder_gap / depth > 0.07:
                return None

    selected_tests = tests[:NECKLINE_TEST_COUNT]
    test_known_indices = [int(item["known_index"]) for item in selected_tests]
    geometry_start = min(pivot.index for pivot in sequence)
    geometry_end = max(pivot.index for pivot in sequence)
    geometry_pivots = [pivot for pivot in pivots
                       if geometry_start <= pivot.index <= geometry_end]
    frozen_at = max(max(pivot.confirmed_at for pivot in sequence), max(test_known_indices),
                    max((pivot.confirmed_at for pivot in geometry_pivots), default=0))
    if frozen_at >= len(frame):
        return None
    frozen_date = str(dates[frozen_at].date())
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

    # First valid break is searched from the moment all geometry became known.
    # If it clears the buffered line before the six-month boundary, the event
    # is rejected permanently instead of being relabeled at a later breakout.
    weak_crossings = []
    breakout = None
    early_breakout = None
    for index in range(frozen_at, len(frame)):
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
                "date": str(dates[index].date()), "close": float(close[index]),
                "neckline": float(level), "required_close": float(required_close),
                "atr14_previous": float(previous_atr), "tolerance": float(tolerance),
            })
        if close[index] > required_close:
            item = {
                "date": str(dates[index].date()), "index": int(index),
                "close": float(close[index]), "frozen_neckline": float(level),
                "required_close": float(required_close),
                "atr14_previous": float(previous_atr), "tolerance": float(tolerance),
            }
            if pd.Timestamp(dates[index]) < maturity_date:
                early_breakout = item
            else:
                breakout = item
            break
    if early_breakout is not None:
        return None
    if breakout is None and pd.Timestamp(dates[-1]) < maturity_date:
        return None

    structure_invalidation = None
    neckline_invalidation = None
    target_reached = None
    target_values = []
    structure_low = min(support_values)
    sequence_known_at = max(pivot.confirmed_at for pivot in sequence)
    invalidation_end = breakout["index"] if breakout is not None else len(frame) - 1
    def support_boundary(index):
        return _at(lower_boundary, index) if family == "triangle_bottom" else support_lower
    first_structure_break = next((index for index in range(sequence_known_at, invalidation_end + 1)
                                  if close[index] < support_boundary(index)), None)
    if first_structure_break is not None:
        structure_invalidation = {"date": str(dates[first_structure_break].date()),
                                  "index": int(first_structure_break),
                                  "level": float(support_boundary(first_structure_break))}
        breakout = None
    if breakout is not None and family == "triangle_bottom":
        breakout_position = (breakout["index"] - first_trough.index) / max(apex_index - first_trough.index, 1.0)
        if not 0.5 <= breakout_position <= 0.75:
            # The first close beyond the buffered boundary was outside the
            # book's effective triangle window; later price cannot re-label it.
            return None
    quality_date = ((structure_invalidation or {}).get("date")
                    or (breakout["date"] if breakout else str(dates[-1].date())))
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
        for index in range(breakout["index"] + 1, len(frame)):
            if close[index] < _at(neckline, index):
                neckline_invalidation = {"date": str(dates[index].date()), "index": int(index),
                                         "level": float(_at(neckline, index))}
                break
            if target_values and float(frame.high.iloc[index]) >= target_values[0]:
                target_reached = {"date": str(dates[index].date()), "index": int(index),
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

    structure_end_date = ((structure_invalidation or neckline_invalidation or target_reached or {}).get("date")
                          or (breakout["date"] if breakout else str(dates[-1].date())))
    signal_eligible = bool(breakout is not None and terminal is None)
    observed_index = len(frame) - 1
    neckline_at_observed = float(_at(neckline, observed_index))
    observed_atr = atr_values[observed_index] if observed_index < len(atr_values) else np.nan
    stop_rules = MAIN_RULES.get("neckline_tests", {})
    plan_stop_buffer = max(float(stop_rules.get("minimum_tolerance_ticks", 2)) * tick_size,
                           float(stop_rules.get("tolerance_atr", 0.5))
                           * float(observed_atr) if np.isfinite(observed_atr) else 0.0)
    plan_stop_level = (neckline_at_observed - plan_stop_buffer) if signal_eligible else float(support_lower)
    identity_dates = [pivot.date for pivot in sequence] + [
        str(dates[index].date()) for index in selected_tests[0]["source_pivot_indices"] + selected_tests[1]["source_pivot_indices"]
    ]
    identity = "|".join([code, kind, timeframe, MAIN_PATTERN_TYPES[family], pivot_scale,
                         ":".join(identity_dates)])
    pattern_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    key_points = {"pivot_scale": pivot_scale, "scale": scale_name,
                  "support_lower": float(support_lower), "support_upper": float(support_upper),
                  "first_trough_index": int(first_trough.index),
                  "first_trough_date": str(dates[first_trough.index].date()),
                  "latest_anchor_index": int(last_anchor.index),
                  "latest_anchor_date": str(dates[last_anchor.index].date()),
                  "neckline_test_indices": [item["pivot_index"] for item in selected_tests]}
    for number, pivot in enumerate(sequence, 1):
        key_points[f"pivot_{number}_index"] = int(pivot.index)
        key_points[f"pivot_{number}_date"] = str(dates[pivot.index].date())
        key_points[f"pivot_{number}"] = float(pivot.price)
        key_points[f"pivot_{number}_kind"] = pivot.kind
    evidence = [
        {"code": "main_structure", "value": {
            "role": "main", "family": family, "scale": scale_name, "pivot_scale": pivot_scale,
            "signal_eligible": signal_eligible,
            "eligibility_reason": "price_breakout_confirmed" if signal_eligible else (
                "terminal_state" if terminal else "awaiting_buffered_price_breakout"),
            "support_lower": float(support_lower), "support_upper": float(support_upper),
            "neckline": neckline, "frozen_at": frozen_date,
            "frozen_at_index": int(frozen_at), "first_trough_date": str(dates[first_trough.index].date()),
            "latest_anchor_date": str(dates[last_anchor.index].date()),
            "maturity_date": str(maturity_date.date()), "structure_end_date": structure_end_date,
            "minimum_duration_months": MIN_STRUCTURE_MONTHS,
            "signal_is_volume_independent": True,
            "low_position": low_position,
            "terminal_state": terminal,
            "observed_date": str(dates[observed_index].date()),
            "neckline_at_observed": neckline_at_observed,
            "plan_stop_level": float(plan_stop_level),
        }},
        {"code": "neckline_tests", "value": selected_tests},
        {"code": "neckline_breakout", "value": breakout},
        {"code": "weak_crossings", "value": weak_crossings},
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
            "first_trough_date": str(dates[first_trough.index].date()),
            "latest_anchor_date": str(dates[last_anchor.index].date()),
            "neckline_at_observed": neckline_at_observed,
            "observed_date": str(dates[observed_index].date()),
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
    return Pattern(
        pattern_id=f"{code}:{pattern_id}", code=code, kind=kind, timeframe=timeframe,
        pattern_type=MAIN_PATTERN_TYPES[family], direction=direction, state=state,
        formed_at=str(dates[first_trough.index].date()),
        confirmed_at=breakout["index"] if breakout else None, key_points=key_points,
        neckline=neckline, invalidation=float(plan_stop_level), targets=target_values, evidence=evidence,
        volume_ratio=volume, volume_verified=False,
        rule_version="caisen-v17-main-bottom-v1", data_version=bars.attrs.get("provenance", {}).get("data_version", "unknown"),
        last_observed_date=str(dates[-1].date()),
        confirmation_date=breakout["date"] if breakout else None,
        source_rule_version=str(MAIN_RULES.get("source_rule_version", "caisen-main-bottoms-v1")),
    )


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
                                         f"{scale_pivot.scale}|neckline_test_1ATR"))
    neckline_pivots.sort(key=lambda pivot: (pivot.index, pivot.scale))
    for pivot_scale, _, _ in PIVOT_SCALES:
        pivots = all_scale_pivots[pivot_scale]
        for family, sequence in _structure_candidates(pivots, timeframe, atr_values, tick):
            event = _make_main_pattern(frame, code, kind, timeframe, family, sequence, neckline_pivots,
                                       atr_values, tick, pivot_scale, horizon, background_bars=background)
            if event is not None:
                events.append(event)
    # Stable IDs deduplicate repeated sequence walks. Select one primary event
    # from the causal set, preferring longer formed structures then coarser turns.
    unique = {event.pattern_id: event for event in events}
    events = list(unique.values())
    if events:
        rank = {"2atr_5pct": 0, "3atr_8pct": 1, "5atr_12pct": 2}
        active = [event for event in events if event.state in {"forming", "confirmed"}]
        pool = active or events
        primary = max(pool, key=lambda event: (
            event.state == "confirmed",
            -float(event.key_points.get("support_lower", float("inf"))),
            rank.get(event.key_points.get("pivot_scale"), -1),
            (pd.Timestamp(event.last_observed_date) - pd.Timestamp(event.formed_at)).days,
            event.pattern_id,
        ))
        for event in events:
            structure = next(item["value"] for item in event.evidence if item["code"] == "main_structure")
            event.evidence.append({"code": "main_selection", "value": {
                "selected": event.pattern_id == primary.pattern_id,
                "decided_at": str(frame.date.iloc[-1].date()),
                "primary_pattern_id": primary.pattern_id,
                "reason": "lowest_support_then_longest_structure",
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


def link_main_bottom_children(main_events: Iterable[Pattern], existing_patterns: Iterable[Pattern], bars_by_timeframe=None):
    """Return copied events with causal same-timeframe parent/child relations.

    A selected main bottom suppresses bullish structures whose known anchors
    lie from its first trough through the breakout (or current observation if
    still forming). Timeframes are never crossed.
    """
    mains = [replace(event, key_points=deepcopy(event.key_points), evidence=deepcopy(event.evidence), flags=list(event.flags))
             for event in main_events]
    legacy = [replace(event, key_points=deepcopy(event.key_points), evidence=deepcopy(event.evidence), flags=list(event.flags))
              for event in existing_patterns]
    all_events = mains + legacy
    parents = []
    for event in mains:
        evidence = {item["code"]: item["value"] for item in event.evidence}
        structure = evidence.get("main_structure", {})
        selection = evidence.get("main_selection", {})
        if selection.get("selected"):
            parents.append((event, structure))
    for child in all_events:
        if child.direction != "bullish" or any(item["code"] == "main_structure" for item in child.evidence):
            continue
        child_dates = _pattern_dates(child, bars_by_timeframe)
        if not child_dates:
            continue
        child_start, child_end = min(child_dates), max(child_dates)
        matches = []
        for parent, structure in parents:
            if child.timeframe != parent.timeframe:
                continue
            start = pd.Timestamp(structure["first_trough_date"]).normalize()
            end_value = (structure.get("structure_end_date") or parent.confirmation_date
                         or parent.last_observed_date or parent.formed_at)
            end = pd.Timestamp(end_value).normalize()
            known_at = pd.Timestamp(structure.get("frozen_at", parent.formed_at)).normalize()
            actionable_date = pd.Timestamp(child.confirmation_date or child.last_observed_date or child_end).normalize()
            if known_at > actionable_date:
                continue
            if start <= child_start and child_end <= end:
                matches.append((start, end, parent, structure))
        if not matches:
            continue
        _, _, parent, structure = max(matches, key=lambda item: (item[0], item[1], item[2].pattern_id))
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
