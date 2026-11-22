"""Causal lifecycle primitives for a structure and its frozen boundary.

This module is experimental until all recognition consumers use the same events.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from itertools import combinations
from typing import List, Optional

from .research_config import recognition_context_snapshot


TERMINAL_PATTERN_STATES = frozenset({"expired", "invalidated", "target_reached"})
JOINT_LIFECYCLE_SCHEMA_VERSION = "joint-pattern-instance-v4"


def derive_lifecycle_state(pattern):
    """Return the authoritative state implied by explicit terminal evidence."""
    values = {item.get("code"): item.get("value") for item in pattern.evidence}
    structure = values.get("main_structure", {}) or {}
    terminal = structure.get("terminal_state")
    if terminal in TERMINAL_PATTERN_STATES:
        return terminal
    flag_terminal = (values.get("flag_lifecycle", {}) or {}).get("state")
    if pattern.pattern_type == "bull_flag" and flag_terminal in TERMINAL_PATTERN_STATES:
        return flag_terminal
    if pattern.state in TERMINAL_PATTERN_STATES:
        return pattern.state
    selection = values.get("neckline_selection", {}) or {}
    early_breakouts = selection.get("early_breakouts", []) or []
    matched_early = [item for item in early_breakouts
                     if _early_breakout_matches_pattern(item, pattern, selection, values)]
    if (matched_early and not values.get("neckline_breakout")
            and not values.get("flag_breakout")):
        return "expired"
    return pattern.state


def sync_lifecycle_state(pattern):
    """Persist a derived terminal state so recognition consumers agree."""
    state = derive_lifecycle_state(pattern)
    if state == "expired":
        values = {item.get("code"): item.get("value") for item in pattern.evidence}
        structure = values.get("main_structure", {})
        selection = values.get("neckline_selection", {}) or {}
        early = [item for item in (selection.get("early_breakouts", []) or [])
                 if _early_breakout_matches_pattern(item, pattern, selection, values)]
        expired_at = min((item.get("recorded_at", item.get("date"))
                          for item in early if item.get("recorded_at", item.get("date"))), default=None)
        if isinstance(structure, dict):
            structure.update({"terminal_state": "expired", "signal_eligible": False,
                              "eligibility_reason": "terminal_state"})
            if expired_at:
                structure["structure_end_date"] = expired_at
        pattern.state = "expired"
    return state


def _early_breakout_matches_pattern(item, pattern, selection, values=None):
    """Match a consumed early cross only to the exact selected frozen boundary."""
    values = values or {entry.get("code"): entry.get("value") for entry in pattern.evidence}
    selected_line = selection.get("selected_line") or pattern.neckline or {}
    early_line = item.get("line")
    if not isinstance(early_line, dict):
        return False
    if (round(float(early_line.get("slope", float("nan"))), 9),
            round(float(early_line.get("intercept", float("nan"))), 9)) != (
            round(float(selected_line.get("slope", float("nan"))), 9),
            round(float(selected_line.get("intercept", float("nan"))), 9)):
        return False
    candidate_id = item.get("candidate_id")
    selected_id = selection.get("selected_candidate_id")
    if candidate_id and selected_id:
        if candidate_id != selected_id:
            return False
    old_tests = item.get("tests") or []
    current_tests = selection.get("independent_tests") or values.get("neckline_tests") or []
    if not old_tests or not current_tests:
        return False
    def signature(tests):
        return tuple((test.get("pivot_date"), test.get("group_id"),
                      round(float(test.get("level", 0.0)), 9)) for test in tests[:2])
    return signature(old_tests) == signature(current_tests)


@dataclass(frozen=True)
class FrozenBoundary:
    boundary_id: str
    slope: float
    intercept: float
    test_indices: tuple = ()
    frozen_at: int = -1

    @property
    def test_span(self) -> int:
        return self.test_indices[-1] - self.test_indices[0] if self.test_indices else 0

    def value_at(self, index: int) -> float:
        return self.slope * index + self.intercept


@dataclass(frozen=True)
class PressureTouch:
    index: int
    known_at: int
    price: float


def select_boundary(touches, highs, atr, observed_index, tick_size):
    """Research primitive: callers must provide already independent pressure tests."""
    known = sorted((t for t in touches if t.known_at <= observed_index), key=lambda t: t.index)
    candidates = []
    for left, right in combinations(known, 2):
        if right.index <= left.index:
            continue
        slope = (right.price - left.price) / (right.index - left.index)
        intercept = left.price - slope * left.index
        groups = [t for t in known if abs(t.price - (slope*t.index + intercept))
                  <= max(2*tick_size, .5*atr[t.index])]
        frozen_at = max(t.known_at for t in groups)
        if any(highs[i] > slope*i + intercept + max(2*tick_size, .5*atr[i])
               for i in range(groups[0].index, frozen_at + 1)):
            continue
        residual = sum(abs(t.price - (slope*t.index + intercept)) /
                       max(2*tick_size, .5*atr[t.index]) for t in groups) / len(groups)
        indices = tuple(t.index for t in groups)
        score = (-len(groups), -(indices[-1] - indices[0]), residual, -frozen_at, indices)
        boundary = FrozenBoundary(f'{left.index}:{right.index}', slope, intercept, indices, frozen_at)
        candidates.append((score, boundary))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


@dataclass(frozen=True)
class PatternEvent:
    instance_id: str
    boundary_id: str
    index: int
    kind: str
    close: float
    boundary_value: float
    date: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self):
        return {"instance_id": self.instance_id, "boundary_version_id": self.boundary_id,
                "index": self.index, "date": self.date, "kind": self.kind,
                "reason": self.reason, "close": self.close,
                "boundary_value": self.boundary_value}


@dataclass
class JointPatternInstance:
    instance_id: str
    boundary: FrozenBoundary
    ready_index: int
    maturity_index: int
    target: Optional[float] = None
    state: str = 'waiting_breakout'
    events: List[PatternEvent] = field(default_factory=list)
    last_index: int = -1
    weak_crossing_active: bool = False


def advance_instance(instance: JointPatternInstance, index: int, close: float,
                     previous_atr: float, tick_size: float) -> List[PatternEvent]:
    if index <= instance.last_index:
        raise ValueError('bars must be appended in strictly increasing order')
    instance.last_index = index
    if index <= instance.ready_index or instance.state in ('expired', 'invalidated', 'target_reached'):
        return []
    line = instance.boundary.value_at(index)
    kind = None
    if instance.state == 'waiting_breakout' and close > line + max(previous_atr, 2 * tick_size):
        if index < instance.maturity_index:
            kind, instance.state = 'rejected_too_early', 'expired'
        else:
            kind, instance.state = 'confirmed', 'confirmed'
    elif instance.state == 'confirmed' and close < line:
        kind, instance.state = 'invalidated', 'invalidated'
    elif instance.state == 'confirmed' and instance.target is not None and close >= instance.target:
        kind, instance.state = 'target_reached', 'target_reached'
    elif instance.state == 'waiting_breakout':
        if close > line and not instance.weak_crossing_active:
            kind = 'weak_crossing'
        instance.weak_crossing_active = close > line
    if kind is None:
        return []
    event = PatternEvent(instance.instance_id, instance.boundary.boundary_id, index, kind, close, line)
    instance.events.append(event)
    return [event]


def pressure_search_start(dates, timeframe):
    """Window is anchored at observation time, independent of structure age."""
    import pandas as pd
    import numpy as np
    years = 10 if timeframe in {'1w', 'weekly', 'week'} else 5
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=years)
    return int(np.searchsorted(dates.values, cutoff.to_datetime64(), side='left'))


def formation_start_index(geometry_start, tests):
    """Coverage starts at the first independently tested pressure point."""
    return int(tests[0]["coverage_start_index"])


def first_buffered_crossing(closes, atr, boundary, start, end, tick_size, cache=None,
                            buffer_multiple=1.0, minimum_ticks=2.0):
    """First eligible close on one frozen line; cached future dates stay hidden."""
    import math
    start = max(1, int(start))
    end = min(int(end), len(closes) - 1, len(atr) - 1)
    if end < start - 1:
        return None
    multiple, ticks = float(buffer_multiple), float(minimum_ticks) * float(tick_size)
    key = (boundary.slope, boundary.intercept, start, float(tick_size), multiple, ticks)
    memo = cache.get(key) if cache is not None else None
    if memo is None:
        memo = {'through': start - 1, 'first': None}
        if cache is not None:
            cache[key] = memo
        ready = start - 1
        if ready > 0 and ready <= end:
            known_atr = float(atr[ready - 1])
            if (math.isfinite(known_atr) and known_atr > 0
                    and float(closes[ready]) > boundary.value_at(ready)
                    + max(ticks, multiple * known_atr)):
                memo['first'] = ready
    if memo['first'] is not None:
        return memo['first'] if memo['first'] <= end else None
    for i in range(max(start, memo['through'] + 1), end + 1):
        previous = float(atr[i - 1])
        if (math.isfinite(previous) and previous > 0
                and float(closes[i]) > boundary.value_at(i) + max(ticks, multiple * previous)):
            memo['first'] = i
            break
    memo['through'] = max(memo['through'], end)
    return memo['first'] if memo['first'] is not None and memo['first'] <= end else None


def has_new_tests_after_failure(tests, failure_index, observed_index):
    """Require two distinct post-failure tests already confirmed at observation."""
    return len({int(test['pivot_index']) for test in tests
                if int(test['pivot_index']) > failure_index
                and int(test['known_index']) > failure_index
                and int(test['known_index']) <= observed_index}) >= 2


def attach_joint_lifecycle(pattern, discovered_at=None, discovery_kind="session_observation"):
    """Attach deterministic structure-boundary identity and its known events.

    The source pattern remains immutable in meaning: this adapter records the
    lifecycle facts already calculated by the recognizer; it never recomputes
    a breakout from a later or substitute boundary.
    """
    values = {item.get("code"): item.get("value") for item in pattern.evidence}
    previous_lifecycle = values.get("joint_lifecycle") or {}
    structure = values.get("main_structure", {}) or {}
    selection = values.get("neckline_selection", {}) or {}
    tests = values.get("neckline_tests", []) or []
    line = pattern.neckline or selection.get("selected_line") or {}
    anchors = []
    number = 1
    while f"pivot_{number}_date" in pattern.key_points:
        point = pattern.key_points
        anchors.append((point.get(f"pivot_{number}_date"),
                        point.get(f"pivot_{number}_kind"),
                        point.get(f"pivot_{number}")))
        number += 1
    if not anchors:
        anchors = [(str(pattern.formed_at), str(pattern.pattern_type), str(pattern.pattern_id))]
    saved_lifecycle = values.get("joint_lifecycle") or {}
    supplied_context = values.get("recognition_context")
    if supplied_context is None:
        supplied_context = saved_lifecycle.get("recognition_context")
    recognition_context = dict(supplied_context if supplied_context is not None
                               else recognition_context_snapshot())
    recognition_context.setdefault("recognition_experiment", None)
    recognition_context.setdefault("data_version", str(pattern.data_version or "unknown"))
    recognition_context.setdefault("engine_version", str(pattern.rule_version or "unknown"))
    recognition_context.setdefault("source_rule_version", str(pattern.source_rule_version or "unknown"))
    structure_payload = {
        "instrument": str(pattern.code), "kind": str(pattern.kind),
        "timeframe": str(pattern.timeframe), "family": str(pattern.pattern_type),
        "anchors": anchors,
    }
    structure_key = hashlib.sha256(json.dumps(
        structure_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()[:24]
    anchor_tests = [{key: test.get(key) for key in ("pivot_date", "level", "group_id")}
                    for test in tests[:2]]
    boundary_payload = {
        "line": [round(float(line.get("slope", 0.0)), 9),
                 round(float(line.get("intercept", 0.0)), 9)],
        "effective_start_date": selection.get("effective_start_date"),
        "anchor_tests": anchor_tests,
    }
    boundary_version_id = hashlib.sha256(json.dumps(
        boundary_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()[:24]
    instance_id = hashlib.sha256(
        f"{structure_key}|{boundary_version_id}".encode("utf-8")
    ).hexdigest()[:24]
    prior_instance_matches = (previous_lifecycle.get("schema_version") in {
                                  "joint-pattern-instance-v2", "joint-pattern-instance-v3",
                                  JOINT_LIFECYCLE_SCHEMA_VERSION}
                              and previous_lifecycle.get("instance_id") == instance_id)
    if (prior_instance_matches and previous_lifecycle.get("schema_version") == JOINT_LIFECYCLE_SCHEMA_VERSION
            and previous_lifecycle.get("paired_at")):
        paired_at = previous_lifecycle["paired_at"]
        paired_at_status = previous_lifecycle.get("paired_at_status", "known")
    elif discovered_at:
        paired_at = str(discovered_at)
        paired_at_status = str(discovery_kind or "observed")
    else:
        # Source geometry dates establish when facts became knowable, not when
        # this instance was actually selected by a causal consumer.
        paired_at = None
        paired_at_status = ("unknown_legacy_discovery_date" if prior_instance_matches
                            else "unknown_missing_discovery_date")
    flag_geometry = values.get("flag_geometry", {}) or {}
    line_known_at = selection.get("line_frozen_at") or flag_geometry.get("known_at")
    structure_known_at = structure.get("structure_known_at") or selection.get("structure_known_at")
    boundary_selected_at = (paired_at if paired_at_status != "unknown_missing_discovery_date"
                            and paired_at_status != "unknown_legacy_discovery_date" else None)
    for item in selection.get("early_breakouts", []) or []:
        if isinstance(item, dict) and item.get("date"):
            item.setdefault("occurred_at", str(item["date"]))
            if paired_at:
                item.setdefault("recorded_at", str(paired_at))
    state = sync_lifecycle_state(pattern)

    events = []

    def prior_event(kind, occurred_at):
        return next((event for event in previous_lifecycle.get("events", [])
                     if event.get("kind") == kind
                     and event.get("boundary_version_id") == boundary_version_id
                     and str(event.get("occurred_at", event.get("date"))) == str(occurred_at)), None)

    def record_event(kind, occurred_at, **fields):
        occurred_at = str(occurred_at)
        old = prior_event(kind, occurred_at)
        if old:
            events.append(dict(old))
            return
        recorded_at = str(paired_at) if paired_at and occurred_at < str(paired_at) else occurred_at
        event = {"date": recorded_at, "kind": kind, "occurred_at": occurred_at,
                 "recorded_at": recorded_at if paired_at or discovered_at else None,
                 "boundary_version_id": boundary_version_id, **fields}
        if recorded_at != occurred_at:
            event["date_semantics"] = "historical_diagnostic"
        elif not paired_at and not discovered_at:
            event["date_semantics"] = "inferred_legacy"
        events.append(event)

    if paired_at:
        old_pair = next((event for event in previous_lifecycle.get("events", [])
                         if event.get("kind") == "paired"
                         and event.get("boundary_version_id") == boundary_version_id), None)
        events.append(dict(old_pair) if old_pair else {
            "date": str(paired_at), "kind": "paired", "recorded_at": str(paired_at),
            "date_semantics": paired_at_status, "reason": selection.get("rule_version"),
            "boundary_version_id": boundary_version_id, "line": dict(line),
            "structure_key": structure_key})
    weak = values.get("weak_crossings", []) or []
    if isinstance(weak, list):
        for item in weak:
            if isinstance(item, dict) and item.get("date"):
                record_event("weak_crossing", item["date"], close=item.get("close"),
                             boundary=item.get("neckline", item.get("frozen_neckline")),
                             required_close=item.get("required_close"),
                             atr=item.get("atr14_previous"))
    early = [item for item in (selection.get("early_breakouts", []) or [])
             if _early_breakout_matches_pattern(item, pattern, selection, values)]
    if isinstance(early, list):
        for item in early:
            if isinstance(item, dict) and item.get("date"):
                record_event("rejected_too_early", item.get("occurred_at", item["date"]),
                             close=item.get("close"),
                             boundary=item.get("frozen_neckline", item.get("frozen_boundary")),
                             required_close=item.get("required_close"),
                             reason="breakout_before_minimum_structure_duration")
    breakout = values.get("neckline_breakout") or values.get("flag_breakout")
    if isinstance(breakout, dict) and breakout.get("date"):
        event_kind = ("pairing_day_cross_consumed" if breakout.get("consumed_on_pairing_day")
                      else "confirmed")
        record_event(event_kind, breakout["date"], close=breakout.get("close"),
                     boundary=breakout.get("frozen_neckline", breakout.get("frozen_boundary")),
                     required_close=breakout.get("required_close"),
                     atr=breakout.get("atr14_previous"),
                     reason=("breakout_on_instance_pairing_date"
                             if event_kind == "pairing_day_cross_consumed" else None))
    pairing_day_cross = values.get("pairing_day_cross_consumed")
    if isinstance(pairing_day_cross, dict) and pairing_day_cross.get("date"):
        record_event("pairing_day_cross_consumed", pairing_day_cross["date"],
                     close=pairing_day_cross.get("close"),
                     required_close=pairing_day_cross.get("required_close"),
                     reason=pairing_day_cross.get("reason"))
    invalidation = (values.get("neckline_invalidation") or values.get("structure_invalidation")
                    or values.get("structure_window_invalidation"))
    if isinstance(invalidation, dict) and invalidation.get("date"):
        record_event("invalidated", invalidation["date"], boundary=invalidation.get("level"),
                     reason=invalidation.get("reason", "close_below_instance_boundary"))
    target = values.get("target_reached")
    if isinstance(target, dict) and target.get("date"):
        record_event("target_reached", target["date"], price=target.get("price"))
    priority = {"paired": 0, "weak_crossing": 1, "rejected_too_early": 2,
                "pairing_day_cross_consumed": 2,
                "confirmed": 3, "invalidated": 4, "target_reached": 5}
    events.sort(key=lambda event: (event.get("date", ""), priority.get(event["kind"], 99)))
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
        event["instance_id"] = instance_id

    main_selection = values.get("main_selection", {}) or {}
    flag_breakout = values.get("flag_breakout") or {}
    flag_geometry = values.get("flag_geometry") or {}
    flag_lifecycle = values.get("flag_lifecycle", {}) or {}
    flag_known_at = flag_geometry.get("known_at")
    flag_selection = flag_lifecycle.get("selection_at_confirmation")
    flag_selection_recorded = (isinstance(flag_selection, dict)
                               and isinstance(flag_selection.get("selected"), bool))
    canonical_flag_selection = (previous_lifecycle.get("selection_at_confirmation")
                                if prior_instance_matches
                                and previous_lifecycle.get("selection_at_confirmation") is not None
                                else flag_selection)
    canonical_flag_selection_recorded = (isinstance(canonical_flag_selection, dict)
                                         and isinstance(canonical_flag_selection.get("selected"), bool))
    flag_breakout_date = flag_breakout.get("date")
    flag_close = flag_breakout.get("close")
    flag_required = flag_breakout.get("required_close")
    flag_breakout_qualified = bool(pattern.pattern_type == "bull_flag"
                                   and flag_breakout_date and pattern.confirmation_date
                                   and flag_breakout_date == pattern.confirmation_date
                                   and flag_known_at and flag_known_at <= flag_breakout_date
                                   and flag_close is not None and flag_required is not None
                                   and float(flag_close) > float(flag_required))
    flag_eligible = bool(flag_breakout_qualified and state == "confirmed"
                         and not pattern.parent_id
                         and (not canonical_flag_selection_recorded
                              or canonical_flag_selection.get("selected") is True))
    main_eligible = bool(structure.get("role") == "main" and structure.get("signal_eligible")
                         and main_selection.get("selected") is True and state == "confirmed")
    suppression = values.get("execution_suppression") or {}
    suppressed = bool(suppression.get("suppressed"))
    eligible = (flag_eligible or main_eligible) and not suppressed and not pattern.parent_id
    suppression_date = suppression.get("suppressed_at")
    suppression_at_confirmation = bool(suppressed and (
        not suppression_date or not pattern.confirmation_date
        or suppression_date <= pattern.confirmation_date))
    main_selection_recorded = (structure.get("role") == "main"
                               and isinstance(main_selection, dict)
                               and isinstance(main_selection.get("selected"), bool))
    flag_confirmation_evidence_present = bool(
        flag_known_at and flag_breakout_date and flag_close is not None and flag_required is not None)
    flag_lifecycle_entry_recorded = (
        "entry_eligible_at_confirmation" in flag_lifecycle
        and flag_lifecycle.get("entry_eligibility_at_confirmation_status") == "known")
    flag_entry_at_confirmation = bool(flag_lifecycle.get(
        "entry_eligible_at_confirmation", False) and flag_breakout_qualified
        and (not flag_selection_recorded or flag_selection.get("selected") is True)
        and not suppression_at_confirmation
        and not pattern.parent_id)
    if pattern.pattern_type != "bull_flag":
        flag_entry_status = "not_applicable"
        flag_entry_historical = None
    elif (prior_instance_matches
          and previous_lifecycle.get("schema_version") == JOINT_LIFECYCLE_SCHEMA_VERSION
          and str(previous_lifecycle.get("entry_eligibility_at_confirmation_status", ""))
          .startswith("known")
          and previous_lifecycle.get("entry_eligible_at_confirmation") is not None):
        # Once a confirmation fact has been published, later evidence cannot rewrite it.
        flag_entry_status = previous_lifecycle["entry_eligibility_at_confirmation_status"]
        flag_entry_historical = previous_lifecycle["entry_eligible_at_confirmation"]
    elif (prior_instance_matches
          and previous_lifecycle.get("schema_version") == JOINT_LIFECYCLE_SCHEMA_VERSION
          and previous_lifecycle.get("canonical_confirmation_date")
          and str(previous_lifecycle.get("entry_eligibility_at_confirmation_status", ""))
          .startswith("unknown")):
        # A confirmed legacy instance with unknown evidence stays unknown.
        flag_entry_status = previous_lifecycle["entry_eligibility_at_confirmation_status"]
        flag_entry_historical = previous_lifecycle.get("entry_eligible_at_confirmation")
    elif (flag_confirmation_evidence_present and flag_breakout_qualified
          and flag_lifecycle_entry_recorded):
        flag_entry_status = "known"
        flag_entry_historical = flag_entry_at_confirmation
    elif flag_confirmation_evidence_present and not flag_breakout_qualified:
        flag_entry_status = "known_not_eligible_geometry_or_breakout_gate"
        flag_entry_historical = False
    elif suppression_at_confirmation:
        flag_entry_status = "known_suppressed_at_confirmation"
        flag_entry_historical = False
    elif previous_lifecycle.get("schema_version") in {
            "joint-pattern-instance-v2", "joint-pattern-instance-v3"}:
        flag_entry_status = "unknown_legacy_lifecycle_schema"
        flag_entry_historical = None
    elif (not flag_confirmation_evidence_present
          or "entry_eligible_at_confirmation" not in flag_lifecycle
          or flag_lifecycle.get("entry_eligibility_at_confirmation_status") != "known"):
        flag_entry_status = ("unknown_missing_confirmation_evidence"
                             if not flag_confirmation_evidence_present
                             else "unknown_legacy_confirmation_eligibility")
        flag_entry_historical = None
    else:
        flag_entry_status = "known"
        flag_entry_historical = bool(flag_entry_at_confirmation or structure.get(
            "entry_eligible_at_confirmation", False))
    if structure.get("role") == "main":
        if main_selection_recorded:
            main_entry_status = "known"
            if (prior_instance_matches
                    and previous_lifecycle.get("entry_eligibility_at_confirmation_status") == "known"):
                main_entry_historical = previous_lifecycle.get("entry_eligible_at_confirmation")
            else:
                main_entry_historical = bool(
                    main_selection.get("selected")
                    and structure.get("signal_eligible")
                    and not suppression_at_confirmation
                    and not pattern.parent_id)
        else:
            main_entry_status = ("unknown_missing_confirmation_selection"
                                 if pattern.confirmation_date else "not_confirmed")
            main_entry_historical = None
    else:
        main_entry_status = flag_entry_status
        main_entry_historical = flag_entry_historical
    current_selection = (structure.get("selection_at_confirmation")
                         if structure.get("role") == "main" else canonical_flag_selection)
    current_selection_reason = (main_selection.get("reason") if structure.get("role") == "main"
                                else flag_selection.get("rule")
                                if isinstance(flag_selection, dict) else None)
    selection_at_confirmation = (previous_lifecycle.get("selection_at_confirmation")
                                 if prior_instance_matches
                                 and previous_lifecycle.get("selection_at_confirmation") is not None
                                 else current_selection)
    selection_reason = (previous_lifecycle.get("selection_reason")
                       if prior_instance_matches
                       and previous_lifecycle.get("selection_at_confirmation") is not None
                       else current_selection_reason)
    flag_events = ((values.get("flag_lifecycle", {}) or {}).get("events") or [])
    lifecycle_events = sorted(
        events + [dict(event, boundary_version_id=boundary_version_id)
                  for event in flag_events if event.get("kind") not in {"confirmed", "paired"}],
        key=lambda event: (event.get("date", ""), event.get("kind", "")))
    if (previous_lifecycle.get("schema_version") in {
            "joint-pattern-instance-v2", "joint-pattern-instance-v3",
            JOINT_LIFECYCLE_SCHEMA_VERSION}
            and previous_lifecycle.get("instance_id") == instance_id):
        merged_events = []
        seen_events = set()
        for event in previous_lifecycle.get("events", []) + lifecycle_events:
            identity = (event.get("date"), event.get("kind"), event.get("boundary_version_id"),
                        event.get("close"), event.get("required_close"), event.get("reason"))
            if identity not in seen_events:
                merged_events.append(dict(event))
                seen_events.add(identity)
        lifecycle_events = sorted(merged_events,
                                  key=lambda event: (event.get("date", ""),
                                                     event.get("sequence", 0), event.get("kind", "")))
    for sequence, event in enumerate(lifecycle_events, start=1):
        event["sequence"] = sequence
        event["instance_id"] = instance_id
    lifecycle = {
        "schema_version": JOINT_LIFECYCLE_SCHEMA_VERSION,
        "instance_id": instance_id,
        "structure_key": structure_key,
        "boundary_version_id": boundary_version_id,
        "paired_at": paired_at,
        "paired_at_status": paired_at_status,
        "instance_discovered_at": paired_at,
        "instance_discovery_kind": paired_at_status,
        "line_known_at": line_known_at,
        "structure_known_at": structure_known_at,
        "boundary_selected_at": boundary_selected_at,
        "state_at": state,
        "canonical_confirmation_date": pattern.confirmation_date,
        "canonical_confirmed_at": pattern.confirmed_at,
        "entry_eligible": eligible,
        "entry_eligible_at_confirmation": (main_entry_historical if structure.get("role") == "main"
                                            else flag_entry_historical),
        "entry_eligibility_at_confirmation_status": main_entry_status,
        "selection_at_confirmation": deepcopy(selection_at_confirmation),
        "selection_reason": selection_reason,
        "boundary": dict(line),
        "structure_anchors": anchors,
        "boundary_anchor_tests": anchor_tests,
        "recognition_context": recognition_context or {"recognition_experiment": None},
        "events": lifecycle_events,
    }
    pattern.evidence = [item for item in pattern.evidence if item.get("code") != "joint_lifecycle"]
    pattern.evidence.append({"code": "joint_lifecycle", "value": lifecycle})
    return lifecycle


def confirmation_snapshot(pattern):
    """Return the pattern exactly as it was knowable at its first confirmation."""
    from copy import deepcopy
    snapshot = deepcopy(pattern)
    if not snapshot.confirmation_date:
        return snapshot
    cutoff = str(snapshot.confirmation_date)
    snapshot.state = "confirmed"
    snapshot.last_observed_date = cutoff
    structure = None
    primary_id = None
    for item in snapshot.evidence:
        code, value = item.get("code"), item.get("value")
        if code == "main_structure" and isinstance(value, dict):
            structure = value
            history = value.get("selection_at_confirmation") or {}
            primary_id = history.get("primary_pattern_id")
            value.update({"signal_eligible": True,
                          "eligibility_reason": "price_breakout_confirmed",
                          "terminal_state": None,
                          "structure_end_date": cutoff,
                          "observed_date": cutoff,
                          "role": "main" if value.get("entry_eligible_at_confirmation") else value.get("role")})
        elif code in {"structure_invalidation", "neckline_invalidation", "structure_window_invalidation", "target_reached"}:
            if isinstance(value, dict) and value.get("date", "") > cutoff:
                item["value"] = None
        elif code == "weak_crossings" and isinstance(value, list):
            item["value"] = [entry for entry in value
                             if isinstance(entry, dict) and entry.get("date", "") <= cutoff]
        elif code == "neckline_selection" and isinstance(value, dict):
            value["diagnostics_as_of"] = cutoff
            value["retired_segments"] = [entry for entry in value.get("retired_segments", [])
                                          if entry.get("breakout_date", "") <= cutoff]
            value["early_breakouts"] = [entry for entry in value.get("early_breakouts", [])
                                         if entry.get("date", "") <= cutoff]
            value["rejected_candidates"] = [entry for entry in value.get("rejected_candidates", [])
                                            if entry.get("known_date", "") <= cutoff]
        elif code == "main_selection" and isinstance(value, dict):
            value["selected"] = bool(structure and structure.get("entry_eligible_at_confirmation"))
            value["primary_pattern_id"] = primary_id or snapshot.pattern_id
            value["decided_at"] = cutoff
        elif code == "joint_lifecycle" and isinstance(value, dict):
            value["events"] = [entry for entry in value.get("events", [])
                               if entry.get("date", "") <= cutoff]
            value["state_at"] = "confirmed"
            value["entry_eligible"] = bool(structure and structure.get("entry_eligible_at_confirmation"))
    if structure is not None:
        breakout = next((item.get("value") for item in snapshot.evidence
                         if item.get("code") == "neckline_breakout"), None)
        if isinstance(breakout, dict):
            structure["neckline_at_observed"] = breakout.get("frozen_neckline")
        snapshot.invalidation = float(structure.get("support_lower", snapshot.invalidation or 0.0))
    attach_joint_lifecycle(snapshot)
    return snapshot
