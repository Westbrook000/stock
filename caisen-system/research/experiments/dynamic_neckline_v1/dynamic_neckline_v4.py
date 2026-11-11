"""Triangle-specific validation for independently revised necklines.

This experiment leaves W bottoms, inverse head-and-shoulders and flags on the
V2 path. For each triangle boundary version, it recomputes the intersection
with the structure's frozen lower rail and only permits a breakout in the
book's 50%-75% formation window.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite

import pandas as pd


def triangle_apex_index(upper: dict, lower: dict) -> float | None:
    """Return the intersection of two converging lines, or None if invalid."""
    try:
        upper_slope = float(upper["slope"])
        upper_intercept = float(upper["intercept"])
        lower_slope = float(lower["slope"])
        lower_intercept = float(lower["intercept"])
    except (KeyError, TypeError, ValueError):
        return None
    denominator = lower_slope - upper_slope
    if denominator <= 0:
        return None
    apex = (upper_intercept - lower_intercept) / denominator
    return apex if isfinite(apex) else None


def _triangle_start_index(structure: dict) -> int | None:
    value = structure.get("first_trough_index")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    lows = []
    for point in structure.get("anchors") or []:
        if str(point.get("kind", point.get("role", ""))).lower() != "low":
            continue
        try:
            lows.append(int(point["index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return min(lows) if lows else None


def filter_triangle_boundary_revisions(structure: dict, revisions: list[dict]) -> list[dict]:
    """Attach an as-of apex to each valid triangle revision.

    Non-triangle revisions are returned unchanged. A triangle revision is
    discarded if the frozen lower rail is missing, the rails do not converge,
    the apex is not after the first trough, or the apex is not still in the
    future when this boundary version becomes available.
    """
    if structure.get("pattern_type") != "triangle_bottom":
        return list(revisions)
    lower = structure.get("lower_boundary")
    start = _triangle_start_index(structure)
    if not isinstance(lower, dict) or start is None:
        return []

    result = []
    for revision in revisions:
        try:
            known_index = int(revision["known_index"])
        except (KeyError, TypeError, ValueError):
            continue
        apex = triangle_apex_index(revision.get("line") or {}, lower)
        if apex is None or apex <= start or apex <= known_index:
            continue
        item = deepcopy(revision)
        item["triangle_apex_index"] = float(apex)
        item["triangle_start_index"] = int(start)
        result.append(item)
    return result


def triangle_breakout_in_window(index: int, start_index: int, apex_index: float,
                                minimum_progress: float = 0.5,
                                maximum_progress: float = 0.75) -> bool:
    """Whether a breakout falls in the 50%-75% start-to-apex interval."""
    try:
        index = int(index)
        start_index = int(start_index)
        apex_index = float(apex_index)
        minimum_progress = float(minimum_progress)
        maximum_progress = float(maximum_progress)
    except (TypeError, ValueError):
        return False
    span = apex_index - start_index
    if (not isfinite(apex_index) or span <= 0
            or not 0.0 <= minimum_progress <= maximum_progress <= 1.0):
        return False
    progress = (index - start_index) / span
    return minimum_progress <= progress <= maximum_progress


def filter_triangle_breakout_signals(structure: dict, revisions: list[dict],
                                     signals: list[dict], bars) -> list[dict]:
    """Keep triangle signals only when their own frozen line has valid timing."""
    if structure.get("pattern_type") != "triangle_bottom":
        return list(signals)
    by_version = {item.get("boundary_version_id"): item for item in revisions}
    dates = pd.to_datetime(bars.date, errors="coerce")
    date_to_index = {pd.Timestamp(value).date().isoformat(): index
                     for index, value in enumerate(dates) if not pd.isna(value)}
    result = []
    for signal in signals:
        revision = by_version.get(signal.get("boundary_version_id"))
        signal_index = date_to_index.get(str(signal.get("signal_date") or ""))
        if revision is None or signal_index is None:
            continue
        start = revision.get("triangle_start_index", _triangle_start_index(structure))
        apex = revision.get("triangle_apex_index")
        if start is None or apex is None or not triangle_breakout_in_window(
                signal_index, start, apex):
            continue
        item = deepcopy(signal)
        item["triangle_apex_index"] = float(apex)
        item["triangle_progress"] = (signal_index - int(start)) / (float(apex) - int(start))
        result.append(item)
    return result
