"""Scan-time BUY and neckline-attention assessments from saved pattern events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd

from .indicators import atr


ASSESSMENT_VERSION = "scan-assessment-v2"
_PRIORITY = {"BUY": 0, "FOCUS_NECKLINE": 1, "INSUFFICIENT_EVIDENCE": 2, "NONE": 3}
_BUY_SETUPS = {"w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag", "spring"}


def _mapping(value):
    if is_dataclass(value):
        return asdict(value)
    return value if isinstance(value, dict) else {}


def _evidence(pattern, code):
    return next((item.get("value") for item in pattern.get("evidence", [])
                 if item.get("code") == code), None)


def _date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.normalize()


def _asof_bars(bars, as_of_date):
    if not isinstance(bars, pd.DataFrame) or bars.empty or not {"date", "close", "high", "low"}.issubset(bars.columns):
        return None
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date", "close", "high", "low"]).sort_values("date")
    return frame.reset_index(drop=True) if not frame.empty else None


def _event(pattern, bars):
    """Read only the recognizer's saved first-break event; never rediscover one."""
    saved_event = _evidence(pattern, "neckline_breakout")
    if isinstance(saved_event, dict):
        return saved_event
    is_main = any(item.get("code") == "main_structure" for item in pattern.get("evidence", []))
    if is_main:
        return None
    confirmation_date = pattern.get("confirmation_date")
    if not confirmation_date and pattern.get("state") == "confirmed":
        return None
    if not confirmation_date:
        return None
    event = {"date": confirmation_date}
    if bars is not None:
        date = _date(confirmation_date)
        match = bars[bars.date == date]
        if not match.empty:
            row = match.iloc[0]
            event["close"] = float(row.close)
            event["index"] = int(row.get("index", match.index[0]))
    return event


def _line(pattern):
    line = pattern.get("neckline")
    if not isinstance(line, dict):
        main = _evidence(pattern, "main_structure")
        line = main.get("neckline") if isinstance(main, dict) else None
    if not isinstance(line, dict):
        return None
    try:
        slope, intercept = float(line["slope"]), float(line["intercept"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return slope, intercept


def _eligible_setup(pattern, cutoff, event=None):
    if pattern.get("direction") != "bullish" or pattern.get("pattern_type") not in _BUY_SETUPS:
        return False, "non_buy_setup"
    if "ambiguous_structure" in (pattern.get("flags") or []):
        return False, "ambiguous_structure"
    suppression = _evidence(pattern, "execution_suppression") or {}
    if pattern.get("parent_id") or (isinstance(suppression, dict) and suppression.get("suppressed")):
        return False, "execution_suppressed"
    main = _evidence(pattern, "main_structure")
    if isinstance(main, dict):
        if main.get("role") != "main":
            return False, "non_primary_structure"
        selection = _evidence(pattern, "main_selection")
        if not isinstance(selection, dict):
            return False, "main_selection_evidence_missing"
        decided = _date(selection.get("decided_at"))
        selected_as_of_cutoff = selection.get("selected") is True and decided is not None and decided <= cutoff
        event_date = _date(event.get("date")) if isinstance(event, dict) else None
        if not selected_as_of_cutoff and (event_date is None or event_date > cutoff):
            if selection.get("selected") is True:
                return False, "primary_selection_after_cutoff"
            return False, "not_selected_primary_structure"
    return True, None


def _row_for_date(frame, value):
    date = _date(value)
    if date is None:
        return None
    matches = frame.index[frame.date == date]
    return int(matches[-1]) if len(matches) else None


def _recognizer_index(pattern, frame, row_index):
    """Map a bar in the supplied history to the recognizer's sliced-window index."""
    key_points = pattern.get("key_points") or {}
    pivot_index = key_points.get("pivot_1_index")
    pivot_date = key_points.get("pivot_1_date") or pattern.get("formed_at")
    try:
        pivot_index = int(pivot_index)
    except (TypeError, ValueError):
        if abs(float(_line(pattern)[0])) < 1e-15:
            return int(row_index)
        return None
    pivot_row = _row_for_date(frame, pivot_date)
    if pivot_row is None:
        return None
    return int(row_index) - (pivot_row - pivot_index)


def _main_boundary(pattern, frame, cutoff_row, line):
    main = _evidence(pattern, "main_structure")
    post_breakout_support = main.get("post_breakout_support_level") if isinstance(main, dict) else None
    support_date = _date(main.get("post_breakout_support_date")) if isinstance(main, dict) else None
    cutoff_date = _date(frame.date.iloc[int(cutoff_row)]) if 0 <= int(cutoff_row) < len(frame) else None
    if (post_breakout_support is not None and support_date is not None
            and cutoff_date is not None and cutoff_date >= support_date):
        return float(post_breakout_support)
    observed_date = main.get("observed_date") if isinstance(main, dict) else None
    observed_boundary = main.get("neckline_at_observed") if isinstance(main, dict) else None
    observed_row = _row_for_date(frame, observed_date)
    if observed_row is not None and observed_boundary is not None:
        return float(observed_boundary) + line[0] * (int(cutoff_row) - observed_row)
    index = _recognizer_index(pattern, frame, cutoff_row)
    return None if index is None else line[0] * index + line[1]


def _known_by_cutoff(pattern, frame, cutoff):
    main = _evidence(pattern, "main_structure")
    if isinstance(main, dict):
        selection = _evidence(pattern, "main_selection") or {}
        decided = _date(selection.get("decided_at"))
        if decided is None or decided > cutoff:
            return False, "primary_selection_after_cutoff"
        return _main_freeze_by(pattern, cutoff)

    # Classic patterns store the last anchor's confirmation index while forming.
    # It is the causal time at which their line became usable.
    if pattern.get("state") == "forming":
        try:
            known_index = int(pattern.get("confirmed_at"))
        except (TypeError, ValueError):
            return False, "formation_freeze_evidence_missing"
        known_date = _index_date(pattern, frame, known_index)
        if known_date is None:
            return False, "formation_freeze_date_unmapped"
        if known_date > cutoff:
            return False, "formation_not_frozen_by_cutoff"
        return True, None
    return False, "formation_freeze_evidence_missing"


def _main_freeze_by(pattern, cutoff):
    main = _evidence(pattern, "main_structure") or {}
    neckline_selection = _evidence(pattern, "neckline_selection") or {}
    freeze = _date(neckline_selection.get("line_frozen_at") or neckline_selection.get("freeze_date")
                   or main.get("frozen_at"))
    known = _date(neckline_selection.get("structure_known_at"))
    if freeze is None or known is None:
        return False, "formation_freeze_evidence_missing"
    if freeze > cutoff or known > cutoff:
        return False, "formation_not_frozen_by_cutoff"
    return True, None


def _index_date(pattern, frame, recognizer_index):
    index = _recognizer_index(pattern, frame, _row_for_date(frame, pattern.get("formed_at")) or 0)
    if index is None:
        return None
    offset = int(recognizer_index) - index
    row = (_row_for_date(frame, pattern.get("formed_at")) or 0) + offset
    return frame.date.iloc[row] if 0 <= row < len(frame) else None


def _historical_primary_proof(pattern, frame, event_date, cache, tick_size=None, background_bars=None):
    """Replay the recognizer only when current selection evidence is not as-of."""
    selection = _evidence(pattern, "main_selection") or {}
    decided = _date(selection.get("decided_at"))
    if selection.get("selected") is True and decided is not None and decided <= event_date:
        return True, {"verified": True, "basis": "saved_selection_at_or_before_event"}

    neckline_selection = _evidence(pattern, "neckline_selection") or {}
    candidate_id = neckline_selection.get("selected_candidate_id")
    if not candidate_id:
        return False, None
    if tick_size is None or not isinstance(background_bars, pd.DataFrame):
        return False, {"missing_replay_inputs": True}
    replay_background = _asof_bars(background_bars, event_date)
    if replay_background is None:
        return False, {"missing_replay_inputs": True}
    replay_background = replay_background[replay_background.date <= event_date]
    if replay_background.empty or replay_background.date.iloc[-1] != event_date:
        return False, {"missing_replay_inputs": True}
    code, kind, timeframe = pattern.get("code", ""), pattern.get("kind", ""), pattern.get("timeframe", "")
    cache_key = (code, kind, timeframe, str(event_date.date()))
    if cache_key not in cache:
        cutoff_bars = frame[frame.date <= event_date].copy()
        if cutoff_bars.empty or cutoff_bars.date.iloc[-1] != event_date:
            cache[cache_key] = set()
        else:
            from .main_bottoms import detect_main_bottoms

            replay = detect_main_bottoms(cutoff_bars, code=code, kind=kind, timeframe=timeframe,
                                         tick_size=tick_size, background_bars=replay_background)
            selected = set()
            for item in replay:
                item_evidence = {part.get("code"): part.get("value") for part in item.evidence}
                item_main = item_evidence.get("main_structure") or {}
                item_selection = item_evidence.get("main_selection") or {}
                item_decided = _date(item_selection.get("decided_at"))
                item_neckline = item_evidence.get("neckline_selection") or {}
                item_breakout = item_evidence.get("neckline_breakout") or {}
                if (item_main.get("role") == "main"
                        and item_selection.get("selected") is True
                        and item_decided is not None and item_decided <= event_date
                        and item_breakout.get("date") == str(event_date.date())
                        and item_neckline.get("selected_candidate_id")):
                    selected.add((item.pattern_type, item_breakout["date"],
                                  item_neckline["selected_candidate_id"]))
            cache[cache_key] = selected
    key = (pattern.get("pattern_type"), str(event_date.date()), candidate_id)
    matched = key in cache[cache_key]
    return matched, ({"verified": True, "basis": "recognizer_replay_at_breakout",
                      "date": str(event_date.date()), "candidate_id": candidate_id} if matched else None)


def _prepare_snapshot(bars, as_of_date):
    frame = _asof_bars(bars, as_of_date)
    if frame is None:
        return {"error": "cutoff_ohlc_snapshot_missing"}
    cutoff = _date(as_of_date) if as_of_date is not None else frame.date.iloc[-1]
    full_frame = frame
    frame = frame[frame.date <= cutoff].reset_index(drop=True)
    if frame.empty:
        return {"error": "no_ohlc_at_cutoff"}
    latest = frame.iloc[-1]
    previous_atr = float(atr(frame, period=14).iloc[-2]) if len(frame) >= 2 else None
    return {"cutoff": cutoff, "full_frame": full_frame, "frame": frame,
            "latest": latest, "previous_atr": previous_atr}


def assess_pattern(pattern, bars, as_of_date=None, *, tick_size=None, background_bars=None):
    """Assess one saved event against its own cutoff snapshot.

    ``bars`` must be the scan run's saved daily/weekly OHLC snapshot, not current
    market data substituted for a historical run.
    """
    return _assess_prepared(_mapping(pattern), _prepare_snapshot(bars, as_of_date),
                            tick_size=tick_size, background_bars=background_bars)


def _assess_prepared(pattern, prepared, historical_primary_cache=None, *, tick_size=None, background_bars=None):
    if prepared.get("error"):
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": prepared["error"]}
    cutoff = prepared["cutoff"]
    full_frame = prepared["full_frame"]
    frame = prepared["frame"]
    latest = prepared["latest"]
    latest_close = float(latest.close)
    latest_date = latest.date
    evidence = {"as_of_date": str(latest_date.date()), "latest_close": latest_close}

    event = _event(pattern, frame)
    eligible, exclusion_reason = _eligible_setup(pattern, cutoff, event)
    if not eligible:
        if exclusion_reason in {"primary_selection_after_cutoff", "main_selection_evidence_missing"}:
            return {"status": "INSUFFICIENT_EVIDENCE", "reason": exclusion_reason, "evidence": evidence}
        return {"status": "NONE", "reason": exclusion_reason, "evidence": evidence}
    formation_date = _date(pattern.get("formed_at"))
    if formation_date is None or formation_date > cutoff:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "formation_after_cutoff", "evidence": evidence}

    if event is not None:
        breakout_date = _date(event.get("date"))
        if breakout_date is None:
            return {"status": "INSUFFICIENT_EVIDENCE", "reason": "breakout_date_missing", "evidence": evidence}
        if breakout_date > cutoff:
            event = None
        else:
            breakout_close = event.get("close")
            if breakout_close is None:
                match = frame[frame.date == breakout_date]
                if not match.empty:
                    breakout_close = float(match.iloc[0].close)
            try:
                breakout_close = float(breakout_close)
            except (TypeError, ValueError):
                return {"status": "INSUFFICIENT_EVIDENCE", "reason": "breakout_close_missing", "evidence": evidence}
            if not np.isfinite(breakout_close) or breakout_close <= 0:
                return {"status": "INSUFFICIENT_EVIDENCE", "reason": "breakout_close_invalid", "evidence": evidence}
            evidence.update({"first_break_date": str(breakout_date.date()), "first_break_close": breakout_close})
            if latest_date == breakout_date:
                buy_reason = "first_valid_breakout_day"
            elif latest_date > breakout_date and latest_close < breakout_close:
                drawdown = (breakout_close - latest_close) / breakout_close
                invalidation = pattern.get("invalidation")
                if invalidation is None:
                    structure = _evidence(pattern, "main_structure") or {}
                    invalidation = structure.get("support_lower")
                if invalidation is not None and latest_close <= float(invalidation):
                    return {"status": "NONE", "reason": "post_breakout_structure_invalidated",
                            "evidence": {**evidence, "drawdown_from_breakout": drawdown,
                                         "invalidation": float(invalidation)}}
                if drawdown < 0.08:
                    return {"status": "FOCUS_NECKLINE", "reason": "post_breakout_focus_within_8pct",
                            "evidence": {**evidence, "drawdown_from_breakout": drawdown,
                                         "invalidation": invalidation}}
                return {"status": "NONE", "reason": "post_breakout_drawdown_at_least_8pct",
                        "evidence": {**evidence, "drawdown_from_breakout": drawdown,
                                     "invalidation": invalidation}}
            else:
                return {"status": "NONE", "reason": "breakout_retest_rule_not_met", "evidence": evidence}
            if _evidence(pattern, "main_structure") is not None:
                known, reason = _main_freeze_by(pattern, breakout_date)
                if not known:
                    return {"status": "INSUFFICIENT_EVIDENCE", "reason": reason, "evidence": evidence}
                selection = _evidence(pattern, "main_selection") or {}
                decided = _date(selection.get("decided_at"))
                if selection.get("selected") is not True or decided is None or decided > breakout_date:
                    cache = historical_primary_cache if historical_primary_cache is not None else {}
                    verified, proof = _historical_primary_proof(
                        pattern, full_frame, breakout_date, cache, tick_size, background_bars)
                    if not verified:
                        reason = ("historical_primary_replay_inputs_missing"
                                  if proof and proof.get("missing_replay_inputs")
                                  else "historical_primary_not_proven")
                        return {"status": "INSUFFICIENT_EVIDENCE", "reason": reason,
                                "evidence": evidence}
                    evidence["historical_primary_proof"] = proof
            return {"status": "BUY", "reason": buy_reason, "evidence": evidence}

    if pattern.get("state") in {"invalidated", "target_reached"}:
        return {"status": "NONE", "reason": "terminal_pattern_state", "evidence": evidence}
    if pattern.get("state") == "confirmed":
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "confirmed_event_not_saved", "evidence": evidence}

    line = _line(pattern)
    if line is None:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "frozen_boundary_missing", "evidence": evidence}
    known, reason = _known_by_cutoff(pattern, full_frame, cutoff)
    if not known:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": reason, "evidence": evidence}
    # For bull_flag, the recognizer stores the upper channel edge as neckline.
    cutoff_row = int(full_frame.index[full_frame.date == latest_date][-1])
    if _evidence(pattern, "main_structure") is not None:
        boundary = _main_boundary(pattern, full_frame, cutoff_row, line)
    else:
        bar_index = _recognizer_index(pattern, full_frame, cutoff_row)
        if bar_index is None:
            return {"status": "INSUFFICIENT_EVIDENCE", "reason": "frozen_boundary_index_unmapped", "evidence": evidence}
        boundary = line[0] * bar_index + line[1]
    if not np.isfinite(boundary) or boundary <= 0:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "frozen_boundary_invalid", "evidence": evidence}
    if prepared["previous_atr"] is None:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "previous_complete_bar_missing", "evidence": evidence}
    atr_value = prepared["previous_atr"]
    if not np.isfinite(atr_value) or atr_value <= 0:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "previous_complete_bar_atr_missing", "evidence": evidence}
    evidence.update({"boundary": boundary, "previous_complete_bar_atr": atr_value})
    if latest_close < boundary and latest_close >= boundary - atr_value:
        return {"status": "FOCUS_NECKLINE", "reason": "within_one_previous_bar_atr_below_boundary", "evidence": evidence}
    return {"status": "NONE", "reason": "outside_neckline_attention_band", "evidence": evidence}


def assess_scan(patterns, bars_by_timeframe, as_of_date=None, *, tick_size_by_timeframe=None,
                background_bars_by_timeframe=None, instance_events_by_timeframe=None):
    """Combine daily/weekly event assessments, preferring BUY over attention."""
    grouped = defaultdict(list)
    for item in patterns or []:
        pattern = _mapping(item)
        timeframe = str(pattern.get("timeframe") or "unknown")
        grouped[timeframe].append(pattern)

    timeframes = {}
    historical_primary_cache = {}
    for timeframe, candidates in grouped.items():
        prepared = _prepare_snapshot((bars_by_timeframe or {}).get(timeframe), as_of_date)
        tick_size = (tick_size_by_timeframe or {}).get(timeframe)
        background_bars = (background_bars_by_timeframe or {}).get(timeframe)
        results = []
        for pattern in candidates:
            shared_events = (instance_events_by_timeframe or {}).get(timeframe, [])
            is_shared_daily = (timeframe == "1d" and pattern.get("pattern_type") in
                               {"w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"})
            matched_events = [event for event in shared_events
                              if event.get("instance_id") == pattern.get("pattern_id")
                              and _date(event.get("event_date")) is not None
                              and _date(event.get("event_date")) <= prepared.get("cutoff", _date("1900-01-01"))]
            if is_shared_daily:
                if matched_events:
                    result = _assess_instance_lifecycle(pattern, matched_events, prepared,
                                                        historical_primary_cache, tick_size, background_bars)
                else:
                    result = {"status": "INSUFFICIENT_EVIDENCE",
                              "reason": "shared_instance_event_missing",
                              "evidence": {"instance_id": pattern.get("pattern_id")}}
            else:
                result = _assess_prepared(pattern, prepared, historical_primary_cache,
                                          tick_size=tick_size, background_bars=background_bars)
            results.append((pattern, result))
        pattern, result = min(results, key=lambda pair: (_PRIORITY.get(pair[1]["status"], 4),
                                                         pair[0].get("pattern_id", "")))
        selected_result = {**result, "pattern_id": pattern.get("pattern_id")}
        if result.get("instance_id") is not None:
            selected_result["instance_id"] = result["instance_id"]
        if result.get("event_id") is not None:
            selected_result["event_id"] = result["event_id"]
        timeframes[timeframe] = selected_result
    overall = min(timeframes, key=lambda tf: (_PRIORITY.get(timeframes[tf]["status"], 4), tf), default=None)
    return {
        "status": timeframes[overall]["status"] if overall else "INSUFFICIENT_EVIDENCE",
        "selected_timeframe": overall,
        "timeframes": timeframes,
        "as_of_date": str(_date(as_of_date).date()) if _date(as_of_date) is not None else None,
    }


def _assess_instance_lifecycle(pattern, events, prepared, historical_primary_cache,
                               tick_size, background_bars):
    """Derive daily action state from the shared immutable instance event stream."""
    events = sorted(events, key=lambda event: (int(event.get("event_seq", 0)),
                                               str(event.get("event_date", ""))))
    cutoff = prepared.get("cutoff")
    if prepared.get("error"):
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": prepared["error"]}
    latest_event = events[-1]
    entry_event = next((event for event in events
                        if event.get("entry_eligible") is True), None)
    evidence = {"instance_id": latest_event.get("instance_id"),
                "event_id": latest_event.get("event_id"),
                "event_date": latest_event.get("event_date"),
                "lifecycle_state": latest_event.get("state"),
                "entry_eligible": bool(latest_event.get("entry_eligible"))}
    if latest_event.get("state") == "invalidated":
        return {"status": "NONE", "reason": "instance_invalidated", "evidence": evidence,
                "event_id": latest_event.get("event_id"), "instance_id": latest_event.get("instance_id")}
    if entry_event is not None:
        breakout_date = _date(entry_event.get("event_date"))
        if breakout_date == cutoff:
            evidence["entry_event_id"] = entry_event.get("event_id")
            return {"status": "BUY", "reason": "first_valid_breakout_day", "evidence": evidence,
                    "event_id": entry_event.get("event_id"), "instance_id": entry_event.get("instance_id")}
        if latest_event.get("state") == "target_reached":
            return {"status": "NONE", "reason": "instance_target_reached", "evidence": evidence,
                    "event_id": latest_event.get("event_id"),
                    "instance_id": latest_event.get("instance_id")}
        if latest_event.get("state") == "focus":
            return {"status": "FOCUS_NECKLINE", "reason": "post_breakout_focus_within_8pct",
                    "evidence": evidence, "event_id": latest_event.get("event_id"),
                    "instance_id": latest_event.get("instance_id")}
        return {"status": "NONE", "reason": "entry_event_not_current_day", "evidence": evidence,
                "event_id": latest_event.get("event_id"), "instance_id": latest_event.get("instance_id")}
    if latest_event.get("state") == "forming":
        snapshot = latest_event.get("pattern")
        if isinstance(snapshot, dict):
            return _assess_prepared(snapshot, prepared, historical_primary_cache,
                                    tick_size=tick_size, background_bars=background_bars)
    return {"status": "NONE", "reason": "no_entry_eligible_instance_event", "evidence": evidence,
            "event_id": latest_event.get("event_id"), "instance_id": latest_event.get("instance_id")}
