"""Immutable, append-only instances shared by daily scan and backtest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


_STATES = {"forming", "confirmed", "focus", "invalidated", "target_reached"}
RECOGNITION_SCHEMA_VERSION = "caisen-pattern-instance-v2"
STATE_SCHEMA_VERSION = "caisen-pattern-state-event-v2"


def _get(pattern, key, default=None):
    if isinstance(pattern, dict):
        return pattern.get(key, default)
    return getattr(pattern, key, default)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def _frozen_geometry(pattern):
    """Project invariant anchors and frozen bounds, excluding scale/index data."""
    key_points = pattern.get("key_points") or {}
    anchors = []
    for key in sorted(key_points):
        if not key.startswith("pivot_") or not key.endswith("_date"):
            continue
        number = key[len("pivot_"):-len("_date")]
        price = key_points.get(f"pivot_{number}")
        anchors.append({"role": key_points.get(f"pivot_{number}_kind") or f"pivot_{number}",
                        "date": str(key_points[key]),
                        "price": round(float(price), 10) if price is not None else None})
    evidence = {item.get("code"): item.get("value")
                for item in pattern.get("evidence", []) if isinstance(item, dict)}
    structure = evidence.get("main_structure") or {}
    lower_boundary = (structure.get("lower_boundary") or pattern.get("lower_boundary")
                      or pattern.get("support"))
    if lower_boundary is None and pattern.get("pattern_type") == "triangle_bottom":
        lower_boundary = {key: (pattern.get("key_points") or {}).get(key)
                          for key in ("support_lower", "support_upper")}
    return {
        "pattern_type": pattern.get("pattern_type"),
        "timeframe": pattern.get("timeframe"),
        "anchors": anchors,
        "neckline": pattern.get("neckline"),
        "lower_boundary": lower_boundary,
        "support_boundary": {key: structure.get(key, key_points.get(key))
                             for key in ("support_lower", "support_upper")},
    }


def pattern_identity_metadata(pattern):
    """Build stable structure, frozen-boundary and instance identities."""
    cached_structure = _get(pattern, "structure_key")
    cached_boundary = _get(pattern, "boundary_version_id")
    if cached_structure and cached_boundary:
        instance_id = _get(pattern, "pattern_id")
        if not instance_id:
            digest = hashlib.sha256(_canonical({"structure_key": cached_structure,
                                                "boundary_version_id": cached_boundary}).encode("utf-8")).hexdigest()[:20]
            instance_id = f"{_get(pattern, 'code') or 'unknown'}:{digest}"
        return {"structure_key": cached_structure,
                "boundary_version_id": cached_boundary,
                "instance_id": str(instance_id)}
    key_points = _get(pattern, "key_points") or {}
    anchors = []
    for key in sorted(key_points):
        if not key.startswith("pivot_") or not key.endswith("_date"):
            continue
        number = key[len("pivot_"):-len("_date")]
        role = key_points.get(f"pivot_{number}_kind") or f"pivot_{number}"
        anchors.append((str(role), str(key_points[key])))
    structure_material = {
        "code": _get(pattern, "code"), "kind": _get(pattern, "kind"),
        "timeframe": _get(pattern, "timeframe"), "pattern_type": _get(pattern, "pattern_type"),
        "anchors": anchors,
    }
    structure_key = "structure-" + hashlib.sha256(
        _canonical(structure_material).encode("utf-8")).hexdigest()[:20]
    evidence = {item.get("code"): item.get("value")
                for item in (_get(pattern, "evidence", []) or []) if isinstance(item, dict)}
    selection = evidence.get("neckline_selection") or {}
    line = selection.get("selected_line") or _get(pattern, "neckline")
    frozen_at = selection.get("line_frozen_at") or selection.get("freeze_date")
    selected_candidate_id = selection.get("selected_candidate_id")
    test_points = []
    for test in evidence.get("neckline_tests") or []:
        if not isinstance(test, dict):
            continue
        test_date = test.get("pivot_date")
        level = test.get("level")
        if (selected_candidate_id and test.get("pair_id")
                and test.get("pair_id") != selected_candidate_id):
            continue
        if frozen_at and test_date and str(test_date) > str(frozen_at):
            continue
        if test_date is not None:
            test_points.append((str(test_date),
                                round(float(level), 10) if level is not None else None))
    structure = evidence.get("main_structure") or {}
    lower_boundary = (structure.get("lower_boundary") or _get(pattern, "lower_boundary")
                      or _get(pattern, "support"))
    if lower_boundary is None and _get(pattern, "pattern_type") == "triangle_bottom":
        lower_boundary = {key: key_points.get(key) for key in ("support_lower", "support_upper")}
    boundary_material = {"pattern_type": _get(pattern, "pattern_type"), "line": line,
                         "lower_boundary": lower_boundary,
                         "line_frozen_at": frozen_at,
                         "support_boundary": {key: structure.get(key, key_points.get(key))
                                              for key in ("support_lower", "support_upper")},
                         "test_points": sorted(test_points)}
    boundary_version_id = "boundary-" + hashlib.sha256(
        _canonical(boundary_material).encode("utf-8")).hexdigest()[:20]
    instance_digest = hashlib.sha256(_canonical({
        "structure_key": structure_key,
        "boundary_version_id": boundary_version_id,
    }).encode("utf-8")).hexdigest()[:20]
    code = _get(pattern, "code") or "unknown"
    return {"structure_key": structure_key,
            "boundary_version_id": boundary_version_id,
            "instance_id": f"{code}:{instance_digest}"}


def _instance_metadata(pattern, event_date):
    evidence = {item.get("code"): item.get("value")
                for item in pattern.get("evidence", []) if isinstance(item, dict)}
    selection = evidence.get("neckline_selection") or {}
    flag_geometry = evidence.get("flag_geometry") or {}
    identity = pattern_identity_metadata(pattern)
    selected = next((item for item in selection.get("candidates", [])
                     if item.get("selected") is True or item.get("candidate_id") == selection.get("selected_candidate_id")), {})
    selection_reason = (selection.get("selection_reason") or selected.get("reason")
                        or selected.get("selection_reason")
                        or ("frozen_flag_geometry" if pattern.get("pattern_type") == "bull_flag"
                            else "selected_frozen_boundary"))
    boundary_available = evidence.get("boundary_available_at") or {}
    pairing_dates = [value for value in (selection.get("line_frozen_at"),
                                         selection.get("freeze_date"),
                                         selection.get("structure_known_at"),
                                         flag_geometry.get("known_at"),
                                         boundary_available.get("date")) if value]
    # ISO date strings sort chronologically. Pairing is only complete once
    # both the boundary and its full structure anchors are knowable.
    paired_at = max(pairing_dates, key=lambda value: str(value)[:10]) if pairing_dates else None
    return {"structure_key": identity["structure_key"],
            "boundary_version_id": identity["boundary_version_id"],
            "paired_at": str(paired_at) if paired_at else None,
            "state_at": str(event_date),
            "selection_reason": str(selection_reason),
            "recognition_schema_version": RECOGNITION_SCHEMA_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION}


def derive_instance_metadata(event):
    """Expose the instance contract beside an immutable event without editing it."""
    event = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    pattern = event.get("pattern") or {}
    keys = ("structure_key", "boundary_version_id", "paired_at", "state_at",
            "selection_reason", "recognition_schema_version", "state_schema_version")
    if all(event.get(key) is not None for key in keys):
        metadata = {key: event[key] for key in keys}
        metadata_source = "native_event_snapshot"
        source_event_schema_version = event.get("state_schema_version")
    else:
        metadata = _instance_metadata(pattern, event.get("event_date"))
        metadata["state_schema_version"] = STATE_SCHEMA_VERSION + "-derived"
        metadata_source = "derived_from_immutable_event_snapshot"
        source_event_schema_version = "legacy-event-v1"
    return {**metadata,
            "instance_id": event.get("instance_id"),
            "event_id": event.get("event_id"),
            "event_seq": event.get("event_seq"),
            "event_date": event.get("event_date"),
            "metadata_source": metadata_source,
            "source_event_schema_version": source_event_schema_version}


def pattern_instance_is_frozen(pattern):
    """Whether the instance has its final paired boundary and known anchors."""
    pattern_type = _get(pattern, "pattern_type")
    evidence = {item.get("code"): item.get("value")
                for item in (_get(pattern, "evidence", []) or []) if isinstance(item, dict)}
    if pattern_type == "bull_flag":
        points = _get(pattern, "key_points") or {}
        geometry = evidence.get("flag_geometry") or {}
        return (all("pivot_%d_index" % number in points for number in range(1, 6))
                and bool(geometry.get("known_at"))
                and bool(_get(pattern, "neckline")))
    if pattern_type not in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}:
        return False
    structure = evidence.get("main_structure") or {}
    selection = evidence.get("neckline_selection") or {}
    tests = evidence.get("neckline_tests") or []
    selected_id = selection.get("selected_candidate_id")
    paired_ids = {item.get("pair_id") for item in tests if isinstance(item, dict)}
    return (structure.get("role") == "main"
            and bool(selected_id)
            and selected_id in paired_ids
            and bool(selection.get("selected_line"))
            and _canonical(selection.get("selected_line")) == _canonical(_get(pattern, "neckline"))
            and bool(selection.get("line_frozen_at"))
            and bool(selection.get("structure_known_at"))
            and len(tests) >= 2
            and bool(_get(pattern, "neckline")))


@dataclass(frozen=True)
class PatternInstanceEvent:
    """One immutable observation of an instance's lifecycle."""

    instance_id: str
    event_id: str
    event_seq: int
    state: str
    event_date: str
    entry_eligible: bool
    snapshot_json: str
    event_type: str = None

    @classmethod
    def from_pattern(cls, pattern, state, event_date, entry_eligible=False, event_seq=1,
                     event_type=None):
        state = str(state)
        if state not in _STATES:
            raise ValueError("unsupported pattern instance state: %s" % state)
        if event_type not in {None, "weak_crossing", "early_breakout_consumed"}:
            raise ValueError("unsupported pattern instance event type: %s" % event_type)
        instance_id = str(pattern.get("pattern_id") or "")
        if not instance_id:
            raise ValueError("pattern instance requires a stable pattern_id")
        geometry = _frozen_geometry(pattern)
        payload = {
            "instance_id": instance_id,
            "event_seq": int(event_seq),
            "state": state,
            "event_date": str(event_date),
            "entry_eligible": bool(entry_eligible),
            **_instance_metadata(pattern, event_date),
            "frozen_geometry": geometry,
            "scale_evidence": {key: value for key, value in (pattern.get("key_points") or {}).items()
                               if "scale" in key or key.endswith("_index") or "test_indices" in key},
            "pattern": json.loads(_canonical(pattern)),
        }
        if event_type is not None:
            payload["event_type"] = str(event_type)
        snapshot_json = _canonical(payload)
        digest_payload = dict(payload)
        digest_payload.pop("event_seq", None)
        # Descriptive schema metadata is additive and must not rename events
        # created by the prior event schema for the same causal snapshot.
        for key in ("structure_key", "boundary_version_id", "paired_at", "state_at",
                    "selection_reason", "recognition_schema_version", "state_schema_version",
                    "scale_evidence"):
            digest_payload.pop(key, None)
        event_id = hashlib.sha256(_canonical(digest_payload).encode("utf-8")).hexdigest()
        payload["event_id"] = event_id
        snapshot_json = _canonical(payload)
        return cls(instance_id, event_id, int(event_seq), state, str(event_date),
                   bool(entry_eligible), snapshot_json, event_type)

    def to_dict(self):
        """Return a detached JSON value; callers cannot mutate the stored event."""
        return json.loads(self.snapshot_json)


class PatternInstanceLedger:
    """Append-only state events with stable geometry and single-entry gating."""

    def __init__(self):
        self._events = []
        self._geometry = {}
        self._last_date = {}
        self._entry_eligible = set()
        self._event_ids = set()
        self._terminal = set()
        self._latest = {}
        self._sequences = {}

    @property
    def events(self):
        return tuple(self._events)

    @property
    def event_count(self):
        """Number of append-only events without copying the event collection."""
        return len(self._events)

    @property
    def latest_events(self):
        return tuple(self._latest.values())

    def append(self, pattern, state, event_date, entry_eligible=False, event_type=None):
        instance_id = str(pattern.get("pattern_id") or "")
        event = PatternInstanceEvent.from_pattern(pattern, state, event_date,
                                                  entry_eligible=entry_eligible,
                                                  event_seq=self._sequences.get(instance_id, 0) + 1,
                                                  event_type=event_type)
        if event.event_id in self._event_ids:
            return next(item for item in self._events if item.event_id == event.event_id)
        payload = event.to_dict()
        geometry = _canonical(payload["frozen_geometry"])
        previous_geometry = self._geometry.get(instance_id)
        if previous_geometry is not None and previous_geometry != geometry:
            raise ValueError("pattern instance frozen geometry cannot change")
        if instance_id in self._terminal:
            raise ValueError("terminal pattern instance cannot change state")
        if state == "forming" and instance_id in self._entry_eligible:
            raise ValueError("confirmed pattern instance cannot return to forming")
        if entry_eligible and state != "confirmed":
            raise ValueError("entry eligibility is only valid on confirmed events")
        if entry_eligible and instance_id in self._entry_eligible:
            raise ValueError("entry eligibility was already emitted for this instance")
        previous_date = self._last_date.get(instance_id)
        if previous_date is not None and event.event_date < previous_date:
            raise ValueError("pattern instance events must be date ordered")
        self._geometry.setdefault(instance_id, geometry)
        self._last_date[instance_id] = event.event_date
        self._event_ids.add(event.event_id)
        self._events.append(event)
        self._latest[instance_id] = event
        self._sequences[instance_id] = event.event_seq
        if entry_eligible:
            self._entry_eligible.add(instance_id)
        if state in {"invalidated", "target_reached"}:
            self._terminal.add(instance_id)
        return event

