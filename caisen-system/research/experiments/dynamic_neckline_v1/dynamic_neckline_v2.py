"""Causal, research-only neckline revisions with material-change gating."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from caisen.main_bottoms import atr


def _iso(value):
    return pd.Timestamp(value).date().isoformat()


def _line_value(line, index):
    return float(line["intercept"]) + float(line["slope"]) * int(index)


def _covers_anchors(line, anchors, tick_size):
    return all(_line_value(line, point["index"]) + 2 * tick_size >= point["price"]
               for point in anchors)


def _candidate_pair(available_points, minimum_span_bars):
    """Return the latest independent pair, allowing either point to predate the base."""
    points_by_index = {}
    for point in available_points:
        index = int(point["index"])
        if index not in points_by_index or int(point["confirmed_at"]) < int(
                points_by_index[index]["confirmed_at"]):
            points_by_index[index] = point
    points = sorted(points_by_index.values(), key=lambda point: int(point["index"]))
    if len(points) < 2:
        return None
    right = points[-1]
    right_index = int(right["index"])
    left = next((point for point in reversed(points[:-1])
                 if right_index - int(point["index"]) >= minimum_span_bars), None)
    return (left, right) if left is not None else None


def build_boundary_revisions_v2(
        bars, structure, pressure_points, tick_size=0.01,
        revision_shift_atr_multiple=1.0, minimum_shift_ticks=2.0,
        minimum_pressure_span_bars=10, minimum_revision_gap_bars=10,
        atr_values=None, association_points=None,
        maximum_association_gap_atr=2.0):
    """Pair a causal pressure boundary with a known structure, then revise materially.

    The initial line may be supported by pressure highs confirmed before the base's
    first structural point. It only becomes available after all structure anchors
    are known. Later confirmed pressure highs can create a new boundary version,
    but only when the line moves by at least the configured ATR/tick threshold.
    A revision never creates a breakout on its own availability bar.
    """
    if bars.empty or not pressure_points or not structure.get("anchors"):
        return []

    dates = pd.to_datetime(bars.date).reset_index(drop=True)
    date_to_index = {_iso(value): index for index, value in enumerate(dates)}
    if "known_index" in structure:
        structure_known_index = int(structure["known_index"])
    else:
        structure_known_index = date_to_index.get(str(structure.get("known_at") or ""))
    if structure_known_index is None or not 0 <= structure_known_index < len(bars):
        return []

    anchors = []
    for point in structure["anchors"]:
        try:
            raw_index = point.get("index")
            if raw_index is None:
                raw_index = date_to_index[str(point["date"])]
            index = int(raw_index)
            price = float(point["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(bars):
            anchors.append({"index": index, "price": price})
    if not anchors:
        return []

    association_points = [dict(point) for point in (association_points or [])]
    if association_points:
        for point in association_points:
            try:
                point["index"] = int(point["index"])
                point["confirmed_at"] = int(point.get("confirmed_at", 0))
                point["price"] = float(point["price"])
            except (KeyError, TypeError, ValueError):
                return []
            if not 0 <= point["index"] < len(bars):
                return []

    close = pd.to_numeric(bars.close, errors="coerce").to_numpy(dtype=float)
    if atr_values is None:
        atr_values = atr(bars, period=14).to_numpy(dtype=float)
    else:
        atr_values = np.asarray(atr_values, dtype=float)
        if len(atr_values) != len(bars):
            raise ValueError("atr_values length must match bars")
    support = structure.get("support_lower")
    invalidated_at = len(bars)
    if support is not None:
        invalidated_at = next((index for index in range(structure_known_index, len(bars))
                               if np.isfinite(close[index]) and close[index] < float(support)),
                              len(bars))
    if structure_known_index >= invalidated_at:
        return []

    points = []
    for point in pressure_points:
        if point.get("kind") != "high":
            continue
        try:
            index = int(point["index"])
            confirmed_at = int(point["confirmed_at"])
            price = float(point["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if (0 <= index < len(bars) and 0 <= confirmed_at < len(bars)
                and confirmed_at < invalidated_at):
            item = dict(point)
            item.update({"index": index, "confirmed_at": confirmed_at, "price": price})
            points.append(item)
    points.sort(key=lambda point: (int(point["confirmed_at"]), int(point["index"])))

    candidate_days = sorted({structure_known_index} |
                            {int(point["confirmed_at"]) for point in points
                             if structure_known_index < int(point["confirmed_at"]) < invalidated_at})
    revisions = []
    active_points = []
    cursor = 0
    active_line = None
    last_revision_index = None
    last_geometry = None
    active_pair = None
    pattern_type = str(structure.get("pattern_type") or "")

    for asof in candidate_days:
        added_point = False
        while cursor < len(points) and int(points[cursor]["confirmed_at"]) <= asof:
            active_points.append(points[cursor])
            cursor += 1
            added_point = True
        if added_point:
            active_pair = _candidate_pair(active_points, minimum_pressure_span_bars)
        pair = active_pair
        if pair is None:
            continue
        left, right = pair
        left_index, right_index = int(left["index"]), int(right["index"])
        if right_index <= left_index:
            continue
        slope = (float(right["price"]) - float(left["price"])) / (right_index - left_index)
        line = {"slope": slope, "intercept": float(left["price"]) - slope * left_index}
        if not _covers_anchors(line, anchors, tick_size):
            continue

        # Candidate boundaries must describe the actual pattern rebound(s),
        # not merely sit above the trough anchors. A rejected candidate is not
        # allowed to become active, so later material-shift checks compare
        # against the last structurally valid version.
        association_pass = True
        for point in association_points:
            index = int(point["index"])
            confirmed_at = int(point["confirmed_at"])
            atr_at_pivot = float(atr_values[index])
            if confirmed_at > asof or not np.isfinite(atr_at_pivot) or atr_at_pivot <= 0:
                association_pass = False
                break
            allowed_gap = max(2.0 * float(tick_size),
                              float(maximum_association_gap_atr) * atr_at_pivot)
            if abs(_line_value(line, index) - float(point["price"])) > allowed_gap + 1e-12:
                association_pass = False
                break
        if not association_pass:
            continue

        # Preserve the existing falling-line cap for W and inverse head-and-shoulders.
        if pattern_type in {"w_bottom", "inverse_head_shoulders"} and slope < 0:
            anchor_start = min(point["index"] for point in anchors)
            anchor_end = max(point["index"] for point in anchors)
            start_level = _line_value(line, anchor_start)
            end_level = _line_value(line, anchor_end)
            relative_change = (end_level - start_level) / max(abs(start_level), tick_size)
            if relative_change < -0.20:
                continue

        geometry = (round(slope, 12), round(float(line["intercept"]), 8))
        if geometry == last_geometry:
            continue
        if active_line is not None:
            if asof - int(last_revision_index) < minimum_revision_gap_bars:
                continue
            previous_level = _line_value(active_line, asof)
            candidate_level = _line_value(line, asof)
            atr_now = atr_values[asof]
            required_shift = max(float(minimum_shift_ticks) * float(tick_size),
                                 float(revision_shift_atr_multiple) * float(atr_now)
                                 if np.isfinite(atr_now) and atr_now > 0 else 0.0)
            if abs(candidate_level - previous_level) + 1e-12 < required_shift:
                continue

        known_at = _iso(dates.iloc[asof])
        version_seed = (f"{structure.get('structure_key')}|{known_at}|"
                        f"{geometry[0]}|{geometry[1]}")
        revision_reason = ("initial_boundary_after_structure_known" if active_line is None
                           else "material_pressure_boundary_shift")
        revisions.append({
            "boundary_version_id": "dyn2-" + hashlib.sha256(
                version_seed.encode("utf-8")).hexdigest()[:16],
            "known_at": known_at,
            "known_index": int(asof),
            "line": line,
            "supporting_point_indices": [left_index, right_index],
            "supporting_point_dates": [
                str(left.get("date") or _iso(dates.iloc[left_index])),
                str(right.get("date") or _iso(dates.iloc[right_index]))],
            "supporting_confirmed_dates": [
                str(left.get("confirmed_date") or _iso(dates.iloc[int(left["confirmed_at"])])),
                str(right.get("confirmed_date") or _iso(dates.iloc[int(right["confirmed_at"])]))],
            "coverage": "structure_anchors_only",
            "revision_reason": revision_reason,
            "revision_shift_atr_multiple": float(revision_shift_atr_multiple),
            "minimum_revision_gap_bars": int(minimum_revision_gap_bars),
            "structure_key": structure.get("structure_key"),
            "instance_id": structure.get("instance_id"),
            "pattern_type": pattern_type,
            "association_point_dates": [
                str(point.get("date") or _iso(dates.iloc[int(point["index"])]))
                for point in association_points],
            "maximum_association_gap_atr": float(maximum_association_gap_atr),
        })
        active_line = line
        last_revision_index = asof
        last_geometry = geometry

    return revisions
