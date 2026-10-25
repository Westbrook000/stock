import hashlib
from dataclasses import replace
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd

from . import RULE_VERSION
from .indicators import atr, volume_ratio
from .pivots import detect_pivots
from .schemas import Pattern
from .rules.catalog import load_rule_catalog


BOOK_PATTERNS = {
    "w_bottom": "W底", "spring": "破底翻", "w_bottom_spring": "W底破底翻",
    "bull_flag": "下飘旗形上涨中继", "inverse_head_shoulders": "头肩底", "triangle_bottom": "收敛三角形底部",
    "m_top": "M头", "false_breakout": "假突破", "false_breakout_head_shoulders": "假突破（头肩顶）",
    "bear_flag": "上飘旗形下跌中继", "head_shoulders": "头肩顶", "triangle_top": "收敛三角形头部",
}


@lru_cache(maxsize=1)
def _active_rule_ids():
    return {item["rule_id"] for item in load_rule_catalog()["patterns"]}


@lru_cache(maxsize=1)
def _rule_set_version():
    return load_rule_catalog()["rule_set_version"]


@lru_cache(maxsize=1)
def _engineering_parameters():
    return {item["rule_id"]: {parameter["name"]: parameter["value"]
                              for parameter in item.get("engineering_parameters", [])}
            for item in load_rule_catalog()["patterns"]}


def _duration_balance_ratio(pattern_type, seq):
    if pattern_type in ("w_bottom", "triangle_bottom"):
        outer_kind, middle_kind = "low", "high"
    elif pattern_type in ("m_top", "triangle_top"):
        outer_kind, middle_kind = "high", "low"
    else:
        return None
    if pattern_type.startswith("triangle_"):
        gaps = [b.index - a.index for a, b in zip(seq, seq[1:])]
    else:
        outer = [pivot for pivot in seq if pivot.kind == outer_kind]
        if len(outer) < 2:
            return None
        first, last = outer[0], outer[-1]
        centers = [pivot for pivot in seq if pivot.kind == middle_kind and first.index < pivot.index < last.index]
        if not centers:
            return None
        center = (max(centers, key=lambda pivot: pivot.price) if middle_kind == "high"
                  else min(centers, key=lambda pivot: pivot.price))
        gaps = [center.index - first.index, last.index - center.index]
    if not gaps or min(gaps) <= 0:
        return None
    return max(gaps) / min(gaps)


def _passes_duration_balance(pattern_type, seq):
    limit = _engineering_parameters().get(pattern_type, {}).get("max_leg_duration_ratio")
    ratio = _duration_balance_ratio(pattern_type, seq)
    return limit is None or ratio is None or ratio <= float(limit)


def _passes_trough_tolerance(pattern_type, seq):
    if pattern_type != "w_bottom":
        return True
    limit = _engineering_parameters().get(pattern_type, {}).get("max_trough_gap_ratio")
    if limit is None:
        return True
    lows = [float(pivot.price) for pivot in seq if pivot.kind == "low"]
    if len(lows) < 2 or max(lows) <= 0:
        return False
    return (max(lows) - min(lows)) / max(lows) <= float(limit)


def _passes_shoulder_tolerance(pattern_type, seq, neckline):
    limit = _engineering_parameters().get(pattern_type, {}).get("shoulder_tolerance")
    if limit is None or len(seq) != 5:
        return True
    head = seq[2]
    neck_at_head = _at(neckline, head.index)
    depth = (head.price - neck_at_head if pattern_type == "head_shoulders"
             else neck_at_head - head.price)
    if depth <= 0:
        return False
    ratio = abs(float(seq[0].price) - float(seq[4].price)) / depth
    return ratio <= float(limit)


def current_pattern_snapshot(patterns):
    """Keep the latest event per rule for a current-analysis screen.

    Historical/replay callers can retain the complete event list returned by
    ``detect_patterns``. The snapshot keeps a terminal latest event as history
    instead of reviving an older same-rule event. It ranks only frozen pivot
    anchor indices already present in each event.
    """
    latest = {}
    for pattern in patterns:
        anchor_indices = [value for key, value in pattern.key_points.items()
                          if key.startswith("pivot_") and key.endswith("_index") and isinstance(value, int)]
        anchor = max(anchor_indices, default=-1)
        rank = (anchor, pattern.formed_at, pattern.pattern_id)
        current = latest.get(pattern.pattern_type)
        if current is None or rank > current[0]:
            latest[pattern.pattern_type] = (rank, pattern)
    return sorted((item[1] for item in latest.values()), key=lambda pattern: (pattern.formed_at, pattern.pattern_type), reverse=True)


def _line(p1, p2):
    if p1[0] == p2[0]:
        return {"slope": 0.0, "intercept": float(p1[1])}
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return {"slope": float(slope), "intercept": float(p1[1] - slope * p1[0])}


def _at(line, index):
    return line["slope"] * index + line["intercept"]


def _valid_projection_geometry(seq, neck, targets, direction, end):
    """Reject impossible boundary geometry and rules without a positive first target."""
    last_pivot = seq[-1]
    boundary = _at(neck, last_pivot.index)
    if direction == "bearish" and boundary >= float(last_pivot.price):
        return False
    if direction == "bullish" and boundary <= float(last_pivot.price):
        return False
    target_is_valid = bool(targets) and np.isfinite(float(targets[0])) and float(targets[0]) > 0
    if not target_is_valid:
        return False
    return True


def detect_patterns(bars: pd.DataFrame, code="", kind="stock", timeframe="1d", asof=None, _atr_multiple=2.0) -> List[Pattern]:
    """Causal first-pass recognizer for the 12 book families; all matches retain evidence."""
    frame = bars.copy().reset_index(drop=True)
    if asof is not None:
        frame = frame[pd.to_datetime(frame.date) <= pd.Timestamp(asof)].reset_index(drop=True)
    if len(frame) < 8:
        return []
    provenance = frame.attrs.get("provenance", {})
    pivots = detect_pivots(frame, radius=2, atr_multiple=_atr_multiple)
    if "segment_id" in frame:
        pivots = [p for p in pivots if int(frame.segment_id.iloc[p.index]) == int(frame.segment_id.iloc[p.confirmed_at])]
    if not pivots:
        return []
    found = []
    close = frame.close.to_numpy()
    # Consecutive 5-point topology: valley/peak/valley/peak/valley or its mirror.
    for start in range(max(0, len(pivots) - 12), max(0, len(pivots) - 2)):
        seq = pivots[start:start + 5]
        if len(seq) != 5:
            continue
        if "segment_id" in frame and frame.segment_id.iloc[[p.index for p in seq]].nunique() > 1:
            continue
        types = [p.kind for p in seq]
        i0, i1, i2, i3, i4 = [p.index for p in seq]
        if types == ["low", "high", "low", "high", "low"]:
            lo1, hi1, lo2, hi2, lo3 = [p.price for p in seq]
            neck_hs=_line((i1, hi1), (i3, hi2))
            if (lo2 < lo1 and lo2 < lo3
                    and _passes_shoulder_tolerance("inverse_head_shoulders", seq, neck_hs)):
                neck_level=_at(neck_hs,i4)
                distance=neck_level-lo2
                _add(found, frame, code, kind, timeframe, "inverse_head_shoulders", seq, close[-1],
                     neck_hs, min(lo1, lo2, lo3), [neck_level+distance,neck_level+2*distance], i4)
            if (_passes_trough_tolerance("w_bottom", seq)
                    and _passes_duration_balance("w_bottom", seq)):
                neck = _line((i1, hi1), (i3, hi2))
                neck_level=(hi1+hi2)/2.0; bottom_level=(lo1+lo2+lo3)/3.0
                distance=neck_level-bottom_level; target = neck_level + distance
                _add(found, frame, code, kind, timeframe, "w_bottom", seq, close[-1], neck,
                     min(lo1, lo2, lo3), [target,target+distance], i4)
            # Both descending upper and ascending lower bounds indicate a converging base.
            lower = _line((i0, lo1), (i4, lo3)); upper = _line((i1, hi1), (i3, hi2))
            if (lower["slope"] > 0 and upper["slope"] < 0
                    and _passes_duration_balance("triangle_bottom", seq)):
                cross = (lower["intercept"] - upper["intercept"]) / (upper["slope"] - lower["slope"])
                position = (i4 - i0) / max(cross - i0, 1)
                if 0.5 <= position <= 0.75:
                    base_width=max(0.0,_at(upper,i0)-_at(lower,i0))
                    _add(found, frame, code, kind, timeframe, "triangle_bottom", seq, close[-1], upper,
                         min(lo1, lo2, lo3), [_at(upper,i4)+base_width], i4)
        if types == ["high", "low", "high", "low", "high"]:
            hi1, lo1, hi2, lo2, hi3 = [p.price for p in seq]
            hs_neck=_line((i1, lo1), (i3, lo2))
            if (hi2 > hi1 and hi2 > hi3
                    and _passes_shoulder_tolerance("head_shoulders", seq, hs_neck)):
                neck_level=_at(hs_neck,i4); head_level=hi2
                distance=head_level-neck_level
                _add(found, frame, code, kind, timeframe, "head_shoulders", seq, close[-1],
                     hs_neck, max(hi1, hi2, hi3), [neck_level-distance,neck_level-2*distance], i4, direction="bearish")
                broke=[k for k in range(i4+1,len(frame)) if close[k]<_at(hs_neck,k)]
                if broke and any(close[k]>_at(hs_neck,k) for k in range(broke[0]+1,len(frame))):
                    _add_reclaim_event(found,frame,code,kind,timeframe,"false_breakout_head_shoulders",seq,
                        hs_neck, min(lo1,lo2), [max(hi1,hi3)], broke[0], bullish=True)
            if (abs(hi1 - hi3) <= max(hi2 - min(lo1, lo2), 1e-9) * 0.75
                    and _passes_duration_balance("m_top", seq)):
                neck = _line((i1, lo1), (i3, lo2)); target = min(lo1, lo2) - (max(hi1, hi2, hi3) - min(lo1, lo2))
                _add(found, frame, code, kind, timeframe, "m_top", seq, close[-1], neck,
                     max(hi1, hi2, hi3), [target], i4, direction="bearish")
            upper = _line((i0, hi1), (i4, hi3)); lower = _line((i1, lo1), (i3, lo2))
            if (upper["slope"] < 0 and lower["slope"] > 0
                    and _passes_duration_balance("triangle_top", seq)):
                cross = (lower["intercept"] - upper["intercept"]) / (upper["slope"] - lower["slope"])
                if 0.5 <= (i4-i0) / max(cross-i0, 1) <= 0.75:
                    base_width=max(0.0,_at(upper,i0)-_at(lower,i0))
                    _add(found, frame, code, kind, timeframe, "triangle_top", seq, close[-1], lower,
                         max(hi1, hi2, hi3), [_at(lower,i4)-base_width], i4, direction="bearish")
    # Two-point W/M are also classified from the trailing three-turn topology.
    for j in range(max(0, len(pivots)-8), len(pivots)-2):
        seq = pivots[j:j+3]
        if len(seq) != 3: continue
        if "segment_id" in frame and frame.segment_id.iloc[[p.index for p in seq]].nunique() > 1:
            continue
        kinds = [p.kind for p in seq]; a,b,c = seq
        if (kinds == ["low", "high", "low"]
                and _passes_trough_tolerance("w_bottom", seq)
                and _passes_duration_balance("w_bottom", seq)):
            line = {"slope": 0.0, "intercept": b.price}
            bottom_mid=(a.price+c.price)/2.0
            distance=b.price-bottom_mid; first=b.price+distance
            _add(found, frame, code, kind, timeframe, "w_bottom", seq, close[-1], line,
                 min(a.price,c.price), [first,first+distance], c.index)
        if (kinds == ["high", "low", "high"]
                and abs(a.price-c.price) < (max(a.price,c.price)-b.price)*0.55
                and _passes_duration_balance("m_top", seq)):
            _add(found, frame, code, kind, timeframe, "m_top", seq, close[-1], {"slope":0.0,"intercept":b.price},
                 max(a.price,c.price), [b.price-(max(a.price,c.price)-b.price)], c.index, direction="bearish")
    box_events = _detect_box_reclaims(frame, code, kind, timeframe, pivots)
    found.extend(box_events)
    found.extend(_nested_w_bottom_springs(found, box_events, frame))
    # Continuation flag candidates: prior impulse followed by bounded counter-trend range and breakout.
    found.extend(_detect_flags(frame, code, kind, timeframe))
    # Remove duplicate same-type event identities while retaining distinct book labels.
    unique = {}
    for p in found:
        unique[p.pattern_id] = p
    if _atr_multiple == 2.0:
        for p in detect_patterns(frame,code,kind,timeframe,asof,_atr_multiple=3.0):
            unique[p.pattern_id]=p
    active = _active_rule_ids()
    recognized = [pattern for pattern in unique.values() if pattern.pattern_type in active]
    topology_groups = {}
    for pattern in recognized:
        indices = tuple(pattern.key_points.get("pivot_%d_index" % n) for n in range(1, 6))
        indices = tuple(index for index in indices if index is not None)
        if len(indices) >= 3:
            topology_groups.setdefault(indices, []).append(pattern)
    for group in topology_groups.values():
        labels = sorted({pattern.pattern_type for pattern in group})
        if len(labels) > 1:
            for pattern in group:
                if "manual_review_required" not in pattern.flags:
                    pattern.flags.append("manual_review_required")
                pattern.evidence.append({"code": "competing_topology_labels", "value": labels})
    for pattern in recognized:
        pattern.last_observed_date = str(pd.to_datetime(frame.date.iloc[-1]).date())
        pattern.source_rule_version = _rule_set_version()
    return sorted(recognized, key=lambda p: (p.formed_at, p.pattern_id), reverse=True)


def _add(out, frame, code, kind, tf, ptype, seq, current_close, neck, invalidation, targets, anchor,
         direction="bullish", extra=None):
    index = len(frame)-1
    # Freeze topology at the last anchor, then find the first causal close break and later invalidation.
    anchor=max(p.index for p in seq)
    known_at=max(p.confirmed_at for p in seq)
    segment = int(frame.segment_id.iloc[anchor]) if "segment_id" in frame else 0
    end = index
    if "segment_id" in frame:
        later = frame.index[(frame.segment_id != segment) & (frame.index > anchor)]
        if len(later): end = int(later[0]) - 1
    if known_at > end:
        return
    if not _valid_projection_geometry(seq, neck, targets, direction, end):
        return
    first_target_valid = bool(targets) and np.isfinite(float(targets[0])) and float(targets[0]) > 0
    target_values = [float(target) for target in targets] if first_target_valid else []
    breaks=[]
    for k in range(known_at,end+1):
        above=frame.close.iloc[k] > _at(neck,k)
        if above if direction=="bullish" else not above:
            breaks.append(k)
    confirm_idx=breaks[0] if breaks else max(p.confirmed_at for p in seq)
    crossed_data_break = end < index
    terminal_state = None
    target_index = None
    structure_invalidation_index = None
    neckline_invalidation_index = None
    price_floor = min(float(p.price) for p in seq)
    price_ceiling = max(float(p.price) for p in seq)

    first_target_valid = bool(targets) and np.isfinite(float(targets[0])) and float(targets[0]) > 0
    def projection_outside_structure(k):
        projected = _at(neck, k)
        return projected <= 0 or projected < price_floor or projected > price_ceiling

    if breaks:
        first_target = target_values[0] if target_values else None
        preconfirmation_failure = next((k for k in range(anchor + 1, confirm_idx + 1)
            if ((frame.close.iloc[k] < float(invalidation) if direction == "bullish"
                 else frame.close.iloc[k] > float(invalidation)) or projection_outside_structure(k))), None)
        if preconfirmation_failure is not None:
            terminal_state = "invalidated"
            if projection_outside_structure(preconfirmation_failure):
                neckline_invalidation_index = preconfirmation_failure
            else:
                structure_invalidation_index = preconfirmation_failure
        for k in range(confirm_idx + 1, end + 1) if terminal_state is None else ():
            invalidated_here = ((frame.close.iloc[k] < _at(neck,k)) if direction == "bullish"
                                else (frame.close.iloc[k] > _at(neck,k)))
            # Daily OHLC cannot establish which level traded first within one bar;
            # resolve a same-bar stop/target touch conservatively as invalidation.
            if invalidated_here:
                terminal_state = "invalidated"
                neckline_invalidation_index = k
                break
            if projection_outside_structure(k):
                terminal_state = "invalidated"
                neckline_invalidation_index = k
                break
            target_reached_here = (first_target is not None and
                                   (frame.high.iloc[k] >= first_target if direction == "bullish"
                                    else frame.low.iloc[k] <= first_target))
            if target_reached_here:
                terminal_state = "target_reached"
                target_index = k
                break
    else:
        for k in range(anchor + 1, end + 1):
            invalidated_here = (frame.close.iloc[k] < float(invalidation) if direction == "bullish"
                                else frame.close.iloc[k] > float(invalidation))
            if invalidated_here:
                terminal_state = "invalidated"
                structure_invalidation_index = k
                break
            if projection_outside_structure(k):
                terminal_state = "invalidated"
                neckline_invalidation_index = k
                break
    if terminal_state is None and crossed_data_break:
        terminal_state = "invalidated"
    state = terminal_state or ("confirmed" if breaks else "forming")
    invalidated = state == "invalidated"
    confirmed=bool(breaks) and not invalidated
    volume = float("nan")
    if confirmed:
        volume = volume_ratio(frame, confirm_idx)
    points = {"%s_%d" % (p.kind, n+1): p.price for n,p in enumerate(seq)}
    for n,p in enumerate(seq,1):
        points["pivot_%d_index"%n]=p.index
        points["pivot_%d"%n]=p.price
    if ptype in ("w_bottom", "w_bottom_spring"):
        lows = [p.price for p in seq if p.kind == "low"]
        if len(lows) >= 2:
            points["first_low"], points["second_low"] = lows[0], lows[1]
    if ptype == "m_top":
        highs = [p.price for p in seq if p.kind == "high"]
        if len(highs) >= 2:
            points["first_high"], points["second_high"] = highs[0], highs[1]
    identity = "%s|%s|%s|%s|%s" % (code, kind, tf, ptype, ":".join(p.date for p in seq))
    pid = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    dates = pd.to_datetime(frame.date)
    evidence = [{"code":"pivot_topology", "value":[p.kind for p in seq]},
                {"code":"neckline", "value":_at(neck,end)}, {"code":"close_confirmation", "value":bool(breaks)},
                {"code":"confirmation_index", "value":confirm_idx}]
    if ptype == "w_bottom":
        lows = [p for p in seq if p.kind == "low"]
        highs = [p for p in seq if p.kind == "high"]
        if len(lows) >= 2 and highs:
            first, last = lows[0], lows[-1]
            middle = max((p for p in highs if first.index < p.index < last.index),
                         key=lambda pivot: pivot.price, default=None)
            if middle is not None:
                first_leg = middle.index - first.index
                second_leg = last.index - middle.index
                evidence.append({"code": "w_geometry_metrics", "value": {
                    "first_leg_bars": first_leg,
                    "second_leg_bars": second_leg,
                    "leg_duration_ratio": round(max(first_leg, second_leg) / max(min(first_leg, second_leg), 1), 3),
                    "trough_gap_pct": round(abs(first.price - last.price) / max(min(first.price, last.price), 1e-9) * 100, 2),
                }})
    if crossed_data_break: evidence.append({"code":"data_discontinuity", "value":True})
    if structure_invalidation_index is not None:
        evidence.append({"code":"structure_invalidation", "value":{
            "index": structure_invalidation_index, "level": float(invalidation),
            "date": str(dates.iloc[structure_invalidation_index].date())}})
    if neckline_invalidation_index is not None:
        evidence.append({"code":"neckline_projection_invalidation", "value":{
            "index": neckline_invalidation_index, "level": _at(neck, neckline_invalidation_index),
            "date": str(dates.iloc[neckline_invalidation_index].date()),
            "structure_low": price_floor, "structure_high": price_ceiling}})
    if target_index is not None:
        evidence.append({"code":"target_reached", "value":{"index":target_index,"price":float(targets[0]),"date":str(dates.iloc[target_index].date())}})
    if extra: evidence.append({"code":"manual_review", "value":extra})
    provenance = frame.attrs.get("provenance", {})
    flags = []
    if extra and extra.get("manual_review_required", True) and "manual_review_required" not in flags:
        flags.append("manual_review_required")
    out.append(Pattern("%s:%s"%(code,pid), code, kind, tf, ptype, direction, state,
        str(dates.iloc[seq[0].index].date()), confirm_idx, points, neck,
        float(invalidation), target_values, evidence,
        None if np.isnan(volume) else volume, bool(frame.attrs.get("quality",{}).get("volume_unit_verified", False) and frame.attrs.get("quality",{}).get("corporate_actions_verified", False)),
        RULE_VERSION, provenance.get("data_version","unknown"),
        flags=flags,
        execution_rules_verified=bool(frame.attrs.get("quality",{}).get("trading_rules_verified",False)),
        last_observed_date=str(dates.iloc[-1].date()),
        confirmation_date=str(dates.iloc[confirm_idx].date()) if confirmed else None,
        source_rule_version=_rule_set_version()))
    out[-1].evidence.append({"code":"confirmed_date", "value":str(dates.iloc[confirm_idx].date())})


def _detect_box_reclaims(frame, code, kind, tf, pivots):
    """Detect break-and-reclaim events only after an explicit consolidation box.

    These are engineering-start values pending blind-label calibration:
    four alternating boundary touches, at least ten bars between first and
    last touch, boundary agreement within max(0.5 ATR, 5% of price), box
    width no greater than 6 ATR, and at least 80% in-box closes. A breakout
    must return inside the frozen box within 20 bars. Candidate events always
    require visual review; the thresholds are not attributed to the book.
    """
    if len(pivots) < 4:
        return []
    atr_values = atr(frame, period=14).to_numpy()
    close = frame.close.to_numpy()
    dates = pd.to_datetime(frame.date)
    found = []
    for offset in range(len(pivots) - 3):
        seq = pivots[offset:offset + 4]
        kinds = [pivot.kind for pivot in seq]
        if kinds not in (["low", "high", "low", "high"], ["high", "low", "high", "low"]):
            continue
        if "segment_id" in frame and frame.segment_id.iloc[[p.index for p in seq]].nunique() != 1:
            continue
        first, last = seq[0].index, seq[-1].index
        if last - first < 10:
            continue
        highs = [pivot.price for pivot in seq if pivot.kind == "high"]
        lows = [pivot.price for pivot in seq if pivot.kind == "low"]
        upper, lower = float(np.mean(highs)), float(np.mean(lows))
        midpoint = max((upper + lower) / 2.0, 0.01)
        tolerance = max(float(atr_values[last]) * 0.5, midpoint * 0.05)
        if max(highs) - min(highs) > tolerance or max(lows) - min(lows) > tolerance:
            continue
        width = upper - lower
        if width <= 0 or width > float(atr_values[last]) * 6.0:
            continue
        box_closes = close[first:last + 1]
        inside = (box_closes >= lower - tolerance) & (box_closes <= upper + tolerance)
        if float(inside.mean()) < 0.8:
            continue
        segment = int(frame.segment_id.iloc[last]) if "segment_id" in frame else 0
        end = len(frame)
        if "segment_id" in frame:
            later = frame.index[(frame.segment_id != segment) & (frame.index > last)]
            if len(later):
                end = int(later[0])
        if kinds[0] == "low":
            break_indices = [i for i in range(last + 1, end) if close[i] < lower]
            ptype, direction, boundary = "spring", "bullish", lower
            in_box = lambda value: lower <= value <= upper
            invalidation = lower
        else:
            break_indices = [i for i in range(last + 1, end) if close[i] > upper]
            ptype, direction, boundary = "false_breakout", "bearish", upper
            in_box = lambda value: lower <= value <= upper
            invalidation = upper
        if not break_indices:
            continue
        break_index = break_indices[0]
        stop = min(end, break_index + 21)
        reclaims = [i for i in range(break_index + 1, stop) if in_box(close[i])]
        if not reclaims:
            continue
        event_seq = seq
        _add_reclaim_event(
            found, frame, code, kind, tf, ptype, event_seq,
            {"slope": 0.0, "intercept": boundary}, invalidation, [], break_index,
            bullish=(direction == "bullish"),
            extra={"box_lower": lower, "box_upper": upper,
                   "box_first_touch": str(dates.iloc[first].date()),
                   "box_last_touch": str(dates.iloc[last].date()),
                   "box_width_atr": width / max(float(atr_values[last]), 1e-9)},
        )
    return found


def _nested_w_bottom_springs(patterns, box_events, frame):
    """Join a verified inner spring box to an overlapping outer W candidate."""
    found = []
    for event in box_events:
        if event.pattern_type != "spring":
            continue
        box_indices = [event.key_points.get("pivot_%d_index" % n)
                       for n in range(1, 5)]
        if any(not isinstance(index, int) for index in box_indices):
            continue
        for parent in patterns:
            if parent.pattern_type != "w_bottom":
                continue
            parent_invalidation = next((item for item in parent.evidence
                                        if item.get("code") == "structure_invalidation"), None)
            if parent.state == "invalidated" and parent_invalidation is None:
                continue
            # The W pattern's low pivots are explicitly labeled by its topology.
            topology = next((item.get("value") for item in parent.evidence
                             if item.get("code") == "pivot_topology"), [])
            low_indices = [index for index, pivot_type in zip(
                [parent.key_points.get("pivot_%d_index" % n) for n in range(1, 6)], topology
            ) if pivot_type == "low" and isinstance(index, int)]
            if len(low_indices) < 2:
                continue
            outer_start, outer_end = min(low_indices), max(low_indices)
            box_start, box_end = min(box_indices), max(box_indices)
            if not (outer_start <= box_start <= outer_end <= box_end):
                continue
            event_evidence = {item.get("code"): item.get("value") for item in event.evidence}
            break_index = event_evidence.get("break_index")
            if not isinstance(break_index, int) or break_index < outer_end:
                continue
            identity = "%s|%s|w_bottom_spring" % (parent.pattern_id, event.pattern_id)
            pid = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
            points = dict(parent.key_points)
            points.update({"inner_box_lower": event_evidence.get("box_lower"),
                           "inner_box_upper": event_evidence.get("box_upper"),
                           "inner_break_index": break_index})
            revived_from_spring = parent.state == "invalidated" and parent_invalidation is not None
            evidence = [
                {**item, "code": "outer_w_invalidation"}
                if revived_from_spring and item.get("code") == "structure_invalidation" else item
                for item in parent.evidence
            ]
            evidence.extend({"code": "inner_spring_" + str(item.get("code")),
                             "value": item.get("value")}
                            for item in event.evidence
                            if item.get("code") in {"box_lower", "box_upper", "box_first_touch",
                                                     "box_last_touch", "break_index", "confirmation_index",
                                                     "close_reclaim"})
            evidence.append({"code": "outer_w_pattern_id", "value": parent.pattern_id})
            evidence.append({"code": "inner_spring_pattern_id", "value": event.pattern_id})
            composite_state = ("confirmed" if event.state == "followthrough_confirmed" else event.state)
            composite = replace(
                parent, pattern_id="%s:%s" % (parent.code, pid),
                pattern_type="w_bottom_spring", key_points=points,
                evidence=evidence, flags=sorted(set(parent.flags + event.flags +
                                                    ["manual_review_required", "nested_event"])),
                parent_id=parent.pattern_id, targets=list(parent.targets),
                state=composite_state if revived_from_spring else parent.state,
                confirmed_at=event.confirmed_at if revived_from_spring else parent.confirmed_at,
                confirmation_date=event.confirmation_date if revived_from_spring else parent.confirmation_date,
                invalidation=event.invalidation if revived_from_spring else parent.invalidation,
                last_observed_date=str(pd.to_datetime(frame.date.iloc[-1]).date()),
            )
            found.append(composite)
    return found


def _detect_flags(frame, code, kind, tf):
    if len(frame) < 30:
        return []
    out = []
    tail = len(frame) - 1
    pivots = detect_pivots(frame, radius=2, atr_multiple=1.5)
    if "segment_id" in frame:
        pivots = [p for p in pivots if int(frame.segment_id.iloc[p.index]) == int(frame.segment_id.iloc[p.confirmed_at])]
    # Search the full configured history window. Restricting this to the last
    # ten turns hid still-active flag structures that formed earlier in the
    # 500-bar analysis window.
    search_floor = max(0, tail - 500)
    pivots = [pivot for pivot in pivots if pivot.index >= search_floor]
    for offset in range(0, len(pivots) - 4):
        seq = pivots[offset:offset + 5]
        if len(seq) != 5:
            continue
        if "segment_id" in frame and frame.segment_id.iloc[[p.index for p in seq]].nunique() != 1:
            continue
        kinds = [p.kind for p in seq]
        i0, i1, i2, i3, i4 = [p.index for p in seq]
        if kinds == ["low", "high", "low", "high", "low"]:
            lo0, hi0, lo1, hi1, lo2 = [float(p.price) for p in seq]
            impulse = hi0 - lo0
            correction = hi0 - lo2
            if (i1 - i0 < 4 or i4 - i1 < 5 or impulse <= 0 or
                    correction < impulse * 0.12 or correction > impulse * 0.50 or
                    lo1 <= lo0 or lo2 <= lo0 or hi1 >= hi0 or hi1 <= lo2):
                continue
            upper = _line((i1, hi0), (i3, hi1))
            # Require a real counter-trend channel and an actual close beyond it.
            if upper["slope"] >= 0:
                continue
            section = frame.iloc[i1:i4 + 1]
            if section.high.max() - section.low.min() > impulse * 0.60:
                continue
            unlocked = bool(frame.close.iloc[tail] > hi0)
            first_target = float(_at(upper, tail) + impulse) if unlocked else float(hi0)
            _add(out, frame, code, kind, tf, "bull_flag", seq, float(frame.close.iloc[-1]), upper,
                 min(lo0, lo1, lo2), [first_target], i4,
                 extra={"measured_target_locked": not unlocked, "impulse_size": impulse,
                        "correction_ratio": correction / impulse, "manual_review_required": False})
        elif kinds == ["high", "low", "high", "low", "high"]:
            hi0, lo0, hi1, lo1, hi2 = [float(p.price) for p in seq]
            impulse = hi0 - lo0
            correction = hi2 - lo0
            if (i1 - i0 < 4 or i4 - i1 < 5 or impulse <= 0 or
                    correction < impulse * 0.12 or correction > impulse * 0.50 or
                    hi1 >= hi0 or hi2 >= hi0 or lo1 <= lo0 or lo1 >= hi1):
                continue
            lower = _line((i1, lo0), (i3, lo1))
            if lower["slope"] <= 0:
                continue
            section = frame.iloc[i1:i4 + 1]
            if section.high.max() - section.low.min() > impulse * 0.60:
                continue
            unlocked = bool(frame.close.iloc[tail] < lo0)
            first_target = float(_at(lower, tail) - impulse) if unlocked else float(lo0)
            _add(out, frame, code, kind, tf, "bear_flag", seq, float(frame.close.iloc[-1]), lower,
                 max(hi0, hi1, hi2), [first_target], i4, direction="bearish",
                 extra={"measured_target_locked": not unlocked, "impulse_size": impulse,
                        "correction_ratio": correction / impulse, "manual_review_required": False})
    return out


def _add_reclaim_event(out,frame,code,kind,tf,ptype,seq,neck,invalidation,targets,break_index,bullish,extra=None):
    if targets and (not np.isfinite(float(targets[0])) or float(targets[0]) <= 0):
        return
    if "segment_id" in frame:
        segment = int(frame.segment_id.iloc[break_index])
        if any(int(frame.segment_id.iloc[p.index]) != segment for p in seq): return
        end = len(frame)
        later = frame.index[(frame.segment_id != segment) & (frame.index > break_index)]
        if len(later): end = int(later[0])
    else:
        end = len(frame)
    dates=pd.to_datetime(frame.date); level=_at(neck,break_index)
    reclaim=[k for k in range(break_index+1,end) if (frame.close.iloc[k]>_at(neck,k) if bullish else frame.close.iloc[k]<_at(neck,k))]
    if not reclaim: return
    confirmation=reclaim[0]
    crossed_data_break = end < len(frame)
    terminal_state = None
    target_index = None
    followthrough_index = None
    first_target = float(targets[0]) if targets else None
    followthrough_level = None
    if extra and "box_upper" in extra and "box_lower" in extra:
        followthrough_level = float(extra["box_upper"] if bullish else extra["box_lower"])
    for k in range(confirmation + 1, end):
        invalidated_here = (frame.close.iloc[k] < _at(neck,k) if bullish
                            else frame.close.iloc[k] > _at(neck,k))
        if invalidated_here:
            terminal_state = "invalidated"
            break
        target_reached_here = (first_target is not None and
                               (frame.high.iloc[k] >= first_target if bullish
                                else frame.low.iloc[k] <= first_target))
        if target_reached_here:
            terminal_state = "target_reached"
            target_index = k
            break
        followed_through = (followthrough_level is not None and
                            (frame.close.iloc[k] > followthrough_level if bullish
                             else frame.close.iloc[k] < followthrough_level))
        if followed_through:
            terminal_state = "followthrough_confirmed"
            followthrough_index = k
            break
    if terminal_state is None and crossed_data_break:
        terminal_state = "invalidated"
    state = terminal_state or "confirmed"
    identity="%s|%s|%s|%s|%s"%(code,kind,tf,ptype,":".join(p.date for p in seq))
    pid=hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    key_points={"pivot_%d_index"%(n+1):p.index for n,p in enumerate(seq)}
    key_points.update({"pivot_%d_price"%(n+1):p.price for n,p in enumerate(seq)})
    key_points.update({"pivot_%d"%(n+1):p.price for n,p in enumerate(seq)})
    evidence=[{"code":"reclaim_level","value":level},{"code":"break_index","value":break_index},
              {"code":"confirmation_index","value":confirmation},{"code":"confirmed_date","value":str(dates.iloc[confirmation].date())},
              {"code":"close_reclaim","value":True},{"code":"manual_review","value":"visual confirmation required"}]
    if extra:
        evidence.extend({"code":key,"value":value} for key,value in extra.items())
    if crossed_data_break: evidence.append({"code":"data_discontinuity", "value":True})
    if target_index is not None:
        evidence.append({"code":"target_reached", "value":{"index":target_index,"price":first_target,"date":str(dates.iloc[target_index].date())}})
    if followthrough_index is not None:
        evidence.append({"code":"followthrough_confirmed", "value":{"index":followthrough_index,
                        "level":followthrough_level,"date":str(dates.iloc[followthrough_index].date())}})
    volume=volume_ratio(frame,confirmation); verified=bool(frame.attrs.get("quality",{}).get("volume_unit_verified",False) and frame.attrs.get("quality",{}).get("corporate_actions_verified",False))
    provenance=frame.attrs.get("provenance",{})
    out.append(Pattern("%s:%s"%(code,pid),code,kind,tf,ptype,"bullish" if bullish else "bearish",state,
       str(dates.iloc[seq[0].index].date()),confirmation,key_points,neck,float(invalidation),targets,evidence,
       None if np.isnan(volume) else volume,verified,RULE_VERSION,provenance.get("data_version","unknown"),
       flags=["manual_review_required","reclaim_event"],
       execution_rules_verified=bool(frame.attrs.get("quality",{}).get("trading_rules_verified",False)),
       last_observed_date=str(dates.iloc[-1].date()), confirmation_date=str(dates.iloc[confirmation].date()),
       source_rule_version=_rule_set_version()))
