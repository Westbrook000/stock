"""Research helpers linking neckline pressure to confirmed shape pivots."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


def _iso(value):
    return pd.Timestamp(value).date().isoformat()


def _kind(point):
    return str(point.get("kind", point.get("role", ""))).lower()


def link_neckline_points(structure, pressure_points, bars, atr_values, tick_size=0.01):
    """Attach causal neckline-specific pivots and delay pairing until all are known.

    W bottoms use the dominant confirmed rebound between troughs. Inverse H&S
    uses one confirmed rebound between each adjacent pair of troughs. Triangles
    and flags already archive their alternating high/low geometry, so their
    high pivots are used directly.
    """
    linked = deepcopy(structure)
    dates = pd.to_datetime(bars.date).reset_index(drop=True)
    if "known_index" in linked:
        structure_known_index = int(linked["known_index"])
    else:
        date_to_index = {_iso(date): i for i, date in enumerate(dates)}
        structure_known_index = date_to_index.get(str(linked.get("known_at") or ""))
    if structure_known_index is None or not 0 <= structure_known_index < len(bars):
        linked.update({"neckline_points": [], "neckline_link_pass": False,
                       "neckline_link_reason": "structure_known_date_missing"})
        return linked

    anchors = linked.get("anchors") or []
    pattern_type = str(linked.get("pattern_type") or "")
    lows = sorted((dict(point) for point in anchors if _kind(point) == "low"),
                  key=lambda point: int(point["index"]))
    points = []
    required = 0

    if pattern_type in {"w_bottom", "inverse_head_shoulders"}:
        segments = list(zip(lows, lows[1:]))
        required = 1 if pattern_type == "w_bottom" else 2
        for left, right in segments:
            left_index, right_index = int(left["index"]), int(right["index"])
            candidates = [dict(point) for point in pressure_points
                          if point.get("kind") == "high"
                          and left_index < int(point.get("index", -1)) < right_index
                          and 0 <= int(point.get("confirmed_at", len(bars))) < len(bars)]
            if not candidates:
                continue
            # The highest confirmed 2ATR swing in each rebound interval is the
            # neckline-related pivot; ties resolve to the earlier swing.
            peak = min(candidates, key=lambda point: (-float(point["price"]),
                                                       int(point["index"]),
                                                       int(point["confirmed_at"])))
            peak.update({"role": "neckline_rebound", "index": int(peak["index"]),
                         "confirmed_at": int(peak["confirmed_at"]),
                         "price": float(peak["price"])})
            points.append(peak)
    else:
        # The archived triangle/flag geometries already include alternating
        # swing highs; lows remain structure anchors but are not neckline tests.
        points = [dict(point, index=int(point["index"]), price=float(point["price"]),
                       confirmed_at=structure_known_index,
                       role="neckline_structure_high")
                  for point in anchors if _kind(point) == "high"]
        required = len(points)

    points.sort(key=lambda point: (int(point["index"]), int(point["confirmed_at"])))
    complete = bool(points) and len(points) >= required
    if pattern_type == "inverse_head_shoulders" and len(points) != required:
        complete = False
    known_index = max([structure_known_index] + [int(point["confirmed_at"]) for point in points])
    if known_index >= len(bars):
        complete = False
        known_index = structure_known_index
    linked["neckline_points"] = points if complete else []
    linked["neckline_link_pass"] = complete
    linked["neckline_link_reason"] = ("all_required_rebound_points_confirmed" if complete
                                      else "missing_confirmed_neckline_structure_point")
    linked["known_index"] = known_index
    linked["known_at"] = _iso(dates.iloc[known_index])
    return linked


def boundary_matches_neckline_points(line, points, atr_values, asof_index,
                                     max_gap_atr=2.0, tick_size=0.01):
    """Require a candidate pressure line to remain close to every linked pivot."""
    if not points:
        return False
    values = np.asarray(atr_values, dtype=float)
    for point in points:
        index = int(point["index"])
        if (int(point.get("confirmed_at", 0)) > int(asof_index)
                or not 0 <= index < len(values)):
            return False
        atr_at_pivot = float(values[index])
        if not np.isfinite(atr_at_pivot) or atr_at_pivot <= 0:
            return False
        line_value = float(line["intercept"]) + float(line["slope"]) * index
        tolerance = max(2.0 * float(tick_size), float(max_gap_atr) * atr_at_pivot)
        if abs(line_value - float(point["price"])) > tolerance + 1e-12:
            return False
    return True
