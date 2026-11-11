import json
import hashlib

import pytest

from caisen.pattern_instances import (PatternInstanceEvent, PatternInstanceLedger,
                                      derive_instance_metadata, pattern_identity_metadata,
                                      pattern_instance_is_frozen)


def test_pattern_instance_events_keep_an_immutable_snapshot():
    pattern = {
        "pattern_id": "000001:db1",
        "code": "000001",
        "kind": "stock",
        "timeframe": "1d",
        "pattern_type": "w_bottom",
        "state": "forming",
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "key_points": {"pivot_1_date": "2025-01-02"},
        "evidence": [{"code": "touch_count", "value": 2}],
    }
    event = PatternInstanceEvent.from_pattern(
        pattern, state="forming", event_date="2025-06-01", entry_eligible=False
    )
    pattern["neckline"]["intercept"] = 99.0
    pattern["evidence"][0]["value"] = 99

    snapshot = event.to_dict()
    assert snapshot["frozen_geometry"]["neckline"]["intercept"] == 12.0
    assert snapshot["pattern"]["evidence"][0]["value"] == 2
    assert snapshot["entry_eligible"] is False

    snapshot["pattern"]["neckline"]["intercept"] = 77.0
    assert event.to_dict()["pattern"]["neckline"]["intercept"] == 12.0


def test_event_contract_exposes_instance_pairing_and_schema_metadata():
    pattern = {
        "pattern_id": "000001:db2", "code": "000001", "kind": "stock",
        "timeframe": "1d", "pattern_type": "w_bottom", "state": "forming",
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_index": 12,
                       "pivot_2_date": "2025-02-03", "pivot_2_index": 34},
        "evidence": [{"code": "neckline_selection", "value": {
            "selected_candidate_id": "candidate-7", "selected_line": {"slope": 0.0, "intercept": 12.0},
            "line_frozen_at": "2025-05-01", "structure_known_at": "2025-05-03",
            "selection_reason": "most independent tests"}}],
    }
    event = PatternInstanceEvent.from_pattern(pattern, "forming", "2025-05-01").to_dict()
    confirmed = PatternInstanceEvent.from_pattern(pattern, "confirmed", "2025-05-05",
                                                 entry_eligible=True).to_dict()

    assert event["structure_key"].startswith("structure-")
    assert event["boundary_version_id"].startswith("boundary-")
    assert event["paired_at"] == "2025-05-03"
    assert event["state_at"] == "2025-05-01"
    assert event["selection_reason"] == "most independent tests"
    assert event["recognition_schema_version"] == "caisen-pattern-instance-v2"
    assert event["state_schema_version"] == "caisen-pattern-state-event-v2"
    assert confirmed["structure_key"] == event["structure_key"]
    assert confirmed["boundary_version_id"] == event["boundary_version_id"]
    assert confirmed["paired_at"] == event["paired_at"]
    assert confirmed["state_at"] == "2025-05-05"
    legacy_digest = dict(event)
    legacy_digest.pop("event_id")
    legacy_digest.pop("event_seq")
    for key in ("structure_key", "boundary_version_id", "paired_at", "state_at",
                "selection_reason", "recognition_schema_version", "state_schema_version",
                "scale_evidence"):
        legacy_digest.pop(key)
    from caisen.pattern_instances import _canonical
    assert event["event_id"] == hashlib.sha256(_canonical(legacy_digest).encode("utf-8")).hexdigest()


def test_pairing_time_includes_causal_boundary_discovery_date():
    pattern = {
        "pattern_id": "000001:new-boundary", "code": "000001", "kind": "stock",
        "timeframe": "1d", "pattern_type": "w_bottom", "state": "forming",
        "neckline": {"slope": 0.0, "intercept": 12.5},
        "key_points": {"pivot_1_date": "2025-01-02"},
        "evidence": [
            {"code": "neckline_selection", "value": {
                "line_frozen_at": "2025-05-01", "structure_known_at": "2025-05-03"}},
            {"code": "boundary_available_at", "value": {
                "date": "2025-05-08", "index": 90}},
        ],
    }

    event = PatternInstanceEvent.from_pattern(pattern, "forming", "2025-05-08").to_dict()

    assert event["paired_at"] == "2025-05-08"


def test_structure_and_boundary_identity_use_anchor_roles_and_frozen_line_only():
    base = {
        "code": "516630", "kind": "etf", "timeframe": "1d", "pattern_type": "w_bottom",
        "key_points": {"pivot_1_index": 530, "pivot_1_date": "2024-02-05",
                       "pivot_1_kind": "low", "pivot_2_index": 549,
                       "pivot_2_date": "2024-04-22", "pivot_2_kind": "low",
                       "pivot_scale": "2atr_5pct"},
        "neckline": {"slope": 0.0, "intercept": 0.9665},
        "evidence": [{"code": "neckline_selection", "value": {
            "selected_candidate_id": "candidate-a",
            "selected_line": {"slope": 0.0, "intercept": 0.9665}}}],
    }
    another_scale = {**base, "key_points": {**base["key_points"],
                                           "pivot_1_index": 1060, "pivot_2_index": 1098,
                                           "pivot_scale": "3atr_8pct"},
                     "evidence": [{"code": "neckline_selection", "value": {
                         "selected_candidate_id": "candidate-b",
                         "selected_line": {"slope": 0.0, "intercept": 0.9665}}}]}
    other_boundary = {**another_scale, "neckline": {"slope": 0.001, "intercept": 0.95},
                      "evidence": [{"code": "neckline_selection", "value": {
                          "selected_candidate_id": "candidate-c",
                          "selected_line": {"slope": 0.001, "intercept": 0.95}}}]}

    first = pattern_identity_metadata(base)
    scale_variant = pattern_identity_metadata(another_scale)
    boundary_variant = pattern_identity_metadata(other_boundary)

    assert first["structure_key"] == scale_variant["structure_key"] == boundary_variant["structure_key"]
    assert first["boundary_version_id"] == scale_variant["boundary_version_id"]
    assert first["instance_id"] == scale_variant["instance_id"]
    assert first["boundary_version_id"] != boundary_variant["boundary_version_id"]
    assert first["instance_id"] != boundary_variant["instance_id"]


def test_triangle_boundary_identity_includes_lower_support_line():
    triangle = {
        "code": "000001", "kind": "stock", "timeframe": "1d",
        "pattern_type": "triangle_bottom",
        "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_kind": "low"},
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "support": {"slope": 0.001, "intercept": 9.0},
        "evidence": [{"code": "main_structure", "value": {
            "lower_boundary": {"slope": 0.001, "intercept": 9.0}}}],
    }
    moved_support = {**triangle, "support": {"slope": 0.002, "intercept": 8.9},
                     "evidence": [{"code": "main_structure", "value": {
                         "lower_boundary": {"slope": 0.002, "intercept": 8.9}}}]}
    assert pattern_identity_metadata(triangle)["structure_key"] == \
        pattern_identity_metadata(moved_support)["structure_key"]
    assert pattern_identity_metadata(triangle)["boundary_version_id"] != \
        pattern_identity_metadata(moved_support)["boundary_version_id"]


def test_later_pressure_tests_do_not_rename_a_frozen_boundary():
    pattern = {
        "code": "000001", "kind": "stock", "timeframe": "1d", "pattern_type": "w_bottom",
        "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_kind": "low"},
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "evidence": [
            {"code": "neckline_selection", "value": {
                "selected_candidate_id": "pair-1", "line_frozen_at": "2025-05-01",
                "selected_line": {"slope": 0.0, "intercept": 12.0}}},
            {"code": "neckline_tests", "value": [
                {"pair_id": "pair-1", "pivot_date": "2025-03-01", "level": 12.0},
                {"pair_id": "pair-1", "pivot_date": "2025-05-01", "level": 12.0}]},
        ],
    }
    later_test = {**pattern, "evidence": [
        pattern["evidence"][0],
        {"code": "neckline_tests", "value": [*pattern["evidence"][1]["value"],
            {"pair_id": "pair-1", "pivot_date": "2025-06-01", "level": 12.0}]}]}

    assert pattern_identity_metadata(pattern) == pattern_identity_metadata(later_test)


def test_frozen_geometry_ignores_scale_and_bar_indices_but_keeps_anchor_and_bounds():
    from caisen.pattern_instances import _frozen_geometry
    base = {
        "pattern_type": "w_bottom", "timeframe": "1d",
        "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_kind": "low",
                       "pivot_1": 9.0, "pivot_1_index": 5,
                       "pivot_scale": "2atr_5pct", "neckline_test_indices": [20, 30]},
        "neckline": {"slope": 0.0, "intercept": 12.0}, "support": 8.0,
    }
    another_scale = {**base, "key_points": {**base["key_points"],
                                            "pivot_1_index": 500,
                                            "pivot_scale": "3atr_8pct",
                                            "neckline_test_indices": [200, 300]}}
    assert _frozen_geometry(base) == _frozen_geometry(another_scale)
    assert _frozen_geometry(base)["anchors"] == [{"role": "low", "date": "2025-01-02", "price": 9.0}]


def test_bull_flag_identity_is_stable_and_pattern_ledger_event_ids_match():
    from caisen.schemas import Pattern
    base = {
        "code": "000001", "kind": "stock", "timeframe": "1d", "pattern_type": "bull_flag",
        "key_points": {f"pivot_{i}_date": f"2025-01-{i:02d}" for i in range(1, 6)},
        "neckline": {"slope": 0.0, "intercept": 12.0}, "evidence": [],
    }
    shifted = {**base, "key_points": {**base["key_points"], "pivot_1_index": 999}}
    first = pattern_identity_metadata(base)
    assert first["instance_id"].startswith("000001:")
    assert first["instance_id"] == pattern_identity_metadata(shifted)["instance_id"]
    pattern = Pattern(pattern_id=first["instance_id"], code="000001", kind="stock",
                      timeframe="1d", pattern_type="bull_flag", direction="bullish",
                      state="forming", formed_at="2025-01-01", confirmed_at=None,
                      key_points=base["key_points"], neckline=base["neckline"]).to_dict()
    event = PatternInstanceLedger().append(pattern, "forming", "2025-01-06").to_dict()
    assert event["instance_id"] == pattern["pattern_id"] == first["instance_id"]
    assert event["pattern"]["pattern_id"] == event["instance_id"]


def test_weak_crossing_is_a_forming_event_without_entry_eligibility():
    ledger = PatternInstanceLedger()
    pattern = {
        "pattern_id": "000001:weak-cross", "code": "000001", "kind": "stock",
        "timeframe": "1d", "pattern_type": "w_bottom", "state": "forming",
        "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_kind": "low"},
        "neckline": {"slope": 0.0, "intercept": 12.0}, "evidence": [],
    }

    event = ledger.append(pattern, "forming", "2025-05-01", event_type="weak_crossing").to_dict()

    assert event["event_type"] == "weak_crossing"
    assert event["state"] == "forming"
    assert event["entry_eligible"] is False


def test_premature_breakout_consumption_is_an_explicit_non_entry_event():
    pattern = {
        "pattern_id": "000001:premature", "code": "000001", "kind": "stock",
        "timeframe": "1d", "pattern_type": "w_bottom", "state": "forming",
        "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_kind": "low"},
        "neckline": {"slope": 0.0, "intercept": 12.0}, "evidence": [],
    }
    ledger = PatternInstanceLedger()
    event = ledger.append(pattern, "forming", "2025-05-01",
                          event_type="early_breakout_consumed").to_dict()

    assert event["event_type"] == "early_breakout_consumed"
    assert event["state"] == "forming"
    assert event["entry_eligible"] is False
    assert ledger.latest_events[0].state == "forming"


def test_legacy_event_metadata_is_derived_without_mutating_raw_event():
    event = {
        "instance_id": "000001:legacy", "event_id": "old-event", "event_seq": 3,
        "event_date": "2025-05-05", "state": "confirmed", "entry_eligible": True,
        "pattern": {"pattern_id": "000001:legacy", "pattern_type": "w_bottom", "timeframe": "1d",
                    "key_points": {"pivot_1_date": "2025-01-02", "pivot_1_index": 1},
                    "neckline": {"slope": 0.0, "intercept": 12.0},
                    "evidence": [{"code": "neckline_selection", "value": {
                        "selected_candidate_id": "paired-7", "selected_line": {"slope": 0.0, "intercept": 12.0},
                        "line_frozen_at": "2025-05-01", "structure_known_at": "2025-05-03",
                        "selection_reason": "ordered pressure pair"}}]},
        "frozen_geometry": {"neckline": {"slope": 0.0, "intercept": 12.0}},
    }
    original = json.dumps(event, sort_keys=True)
    metadata = derive_instance_metadata(event)

    assert metadata["metadata_source"] == "derived_from_immutable_event_snapshot"
    assert metadata["source_event_schema_version"] == "legacy-event-v1"
    assert metadata["paired_at"] == "2025-05-03"
    assert metadata["state_at"] == event["event_date"]
    assert metadata["boundary_version_id"].startswith("boundary-")
    assert metadata["event_id"] == "old-event"
    assert json.dumps(event, sort_keys=True) == original


def test_pattern_instance_ledger_rejects_relabeling_same_frozen_instance():
    ledger = PatternInstanceLedger()
    forming = {
        "pattern_id": "000001:db1",
        "code": "000001",
        "kind": "stock",
        "timeframe": "1d",
        "pattern_type": "w_bottom",
        "state": "forming",
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "key_points": {"pivot_1_date": "2025-01-02"},
        "evidence": [],
    }
    first = ledger.append(forming, state="forming", event_date="2025-06-01")
    later = dict(forming, neckline={"slope": 0.01, "intercept": 11.0})

    with pytest.raises(ValueError, match="frozen geometry"):
        ledger.append(later, state="confirmed", event_date="2025-06-02",
                      entry_eligible=True)

    assert len(ledger.events) == 1
    assert first.instance_id == "000001:db1"


def test_pattern_instance_ledger_allows_one_buy_eligibility_event_per_instance():
    ledger = PatternInstanceLedger()
    pattern = {
        "pattern_id": "000001:db1",
        "code": "000001",
        "kind": "stock",
        "timeframe": "1d",
        "pattern_type": "w_bottom",
        "state": "confirmed",
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "key_points": {"pivot_1_date": "2025-01-02"},
        "evidence": [],
    }
    buy = ledger.append(pattern, state="confirmed", event_date="2025-06-02",
                        entry_eligible=True)
    replayed = ledger.append(pattern, state="confirmed", event_date="2025-06-02",
                             entry_eligible=True)
    retreat = ledger.append(pattern, state="focus", event_date="2025-06-03",
                            entry_eligible=False)

    with pytest.raises(ValueError, match="entry eligibility"):
        ledger.append(pattern, state="confirmed", event_date="2025-06-04",
                      entry_eligible=True)

    assert buy.entry_eligible is True
    assert replayed.event_id == buy.event_id
    assert replayed.event_seq == buy.event_seq
    assert retreat.entry_eligible is False
    assert [item.state for item in ledger.events] == ["confirmed", "focus"]
    assert json.loads(buy.snapshot_json)["instance_id"] == retreat.instance_id


def test_target_reached_is_a_terminal_non_entry_lifecycle_event():
    ledger = PatternInstanceLedger()
    pattern = {
        "pattern_id": "000001:target", "code": "000001", "kind": "stock",
        "timeframe": "1d", "pattern_type": "w_bottom", "state": "confirmed",
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "key_points": {"pivot_1_date": "2025-01-02"}, "evidence": [],
    }
    ledger.append(pattern, "confirmed", "2025-06-02", entry_eligible=True)

    target = ledger.append(pattern, "target_reached", "2025-06-03")

    assert target.event_seq == 2
    assert target.entry_eligible is False
    with pytest.raises(ValueError, match="terminal pattern instance"):
        ledger.append(pattern, "focus", "2025-06-04")


def test_pre_freeze_candidate_line_updates_do_not_create_pattern_instances():
    candidate = {
        "pattern_id": "000001:pending",
        "pattern_type": "w_bottom",
        "state": "forming",
        "neckline": {"slope": 0.0, "intercept": 12.0},
        "evidence": [
            {"code": "main_structure", "value": {"role": "main"}},
            {"code": "neckline_selection", "value": {"selected_line": None}},
        ],
    }
    assert pattern_instance_is_frozen(candidate) is False
    candidate["neckline"] = {"slope": 0.01, "intercept": 11.0}
    candidate["evidence"][1]["value"].update({
        "selected_line": dict(candidate["neckline"]),
        "selected_candidate_id": "paired-touch-1-2",
        "line_frozen_at": "2025-06-01",
        "structure_known_at": "2025-06-01",
    })
    candidate["evidence"].append({"code": "neckline_tests", "value": [
        {"pair_id": "paired-touch-1-2"}, {"pair_id": "paired-touch-1-2"}]})

    assert pattern_instance_is_frozen(candidate) is True
