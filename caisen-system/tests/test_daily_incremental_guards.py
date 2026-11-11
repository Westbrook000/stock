"""Guard tests for causal lifecycles that must survive the rolling session."""

import pandas as pd
import pytest
from dataclasses import replace
from copy import deepcopy

from caisen.daily_incremental import DailyRecognitionSession
from caisen.schemas import Pattern, Pivot
from caisen.pattern_instances import pattern_identity_metadata, pattern_instance_is_frozen


def _bar(day, close=10.0):
    return {"date": pd.Timestamp(day), "open": close, "high": close + 0.1,
            "low": close - 0.1, "close": close, "volume": 1000}


def _quiet_pivot_updates(session, monkeypatch):
    monkeypatch.setattr(session, "_append_scale_pivots", lambda _index: [])
    monkeypatch.setattr(session, "_append_pressure_pivot", lambda _index: None)
    monkeypatch.setattr(session, "_append_local_pivot", lambda _index: False)


def _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                   known="2024-06-28", support_lower=9.5):
    key_points = {"first_trough_date": first_trough, "first_trough_index": 0,
                  "pivot_1_date": first_trough, "pivot_1_kind": "low", "pivot_1": support_lower,
                  "pivot_2_date": frozen, "pivot_2_kind": "low", "pivot_2": support_lower,
                  "pivot_scale": "2atr_5pct"}
    line = {"slope": 0.0, "intercept": 10.0}
    tests = [{"pair_id": "paired-boundary", "group_id": f"test-{i}",
              "pivot_date": date, "level": 10.0, "valid": True,
              "normalized_residual": 0.1, "test_span_bars": 20}
             for i, date in enumerate((first_trough, frozen))]
    evidence = [
        {"code": "main_structure", "value": {
            "role": "main", "eligibility_reason": "awaiting_fresh_neckline_breakout",
            "support_lower": support_lower}},
        {"code": "neckline_selection", "value": {
            "selected_candidate_id": "paired-boundary", "selected_line": line,
            "line_frozen_at": frozen, "structure_known_at": known,
            "effective_start_date": first_trough}},
        {"code": "neckline_tests", "value": tests},
    ]
    identity = pattern_identity_metadata({"code": "000001", "kind": "stock", "timeframe": "1d",
        "pattern_type": "w_bottom", "key_points": key_points, "neckline": line, "evidence": evidence})
    return Pattern(
        pattern_id=identity["instance_id"], code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at=first_trough, confirmed_at=None,
        key_points=key_points, neckline=line, invalidation=support_lower, evidence=evidence,
        last_observed_date=frozen, structure_key=identity["structure_key"],
        boundary_version_id=identity["boundary_version_id"],
    )


def _install_cached_candidate(session, pattern):
    first_date = pattern.key_points["first_trough_date"]
    pivot = Pivot(0, first_date, float(pattern.invalidation), "low", 1,
                  str(pd.Timestamp(session.bars[1]["date"]).date()), "2ATR|5%")
    session.active[(pattern.pattern_type, "2atr_5pct", (0, "low"))] = {
        "family": pattern.pattern_type, "scale": "2atr_5pct", "sequence": [pivot],
        "structure_known_at": 1, "pattern": pattern, "needs_initial_eval": False,
    }


def test_cached_signal_rejects_a_high_crossing_during_formation():
    session = DailyRecognitionSession('fixture', 'stock')
    session.bars = [_bar(d, c) for d, c in zip(
        ['2024-01-01', '2024-07-01', '2024-07-02', '2024-07-03'], [9, 9, 9, 11])]
    session.bars[1]['high'] = 11.0
    session.date_strings = [str(b['date'].date()) for b in session.bars]
    session.atr_values = [.2] * 4
    _install_cached_candidate(session, _cached_bottom(frozen='2024-07-01', known='2024-07-02',
                                                      support_lower=8))
    assert session._check_cached_breakout() == []


def test_cached_signal_keeps_weak_crossing_after_complete_formation():
    session = DailyRecognitionSession('fixture', 'stock')
    session.bars = [_bar(d, c) for d, c in zip(
        ['2024-01-01', '2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04'],
        [9, 9, 9, 10.1, 10.4])]
    session.date_strings = [str(b['date'].date()) for b in session.bars]
    session.atr_values = [.2] * 5
    _install_cached_candidate(session, _cached_bottom(frozen='2024-07-01', known='2024-07-02',
                                                      support_lower=8))
    result = session._check_cached_breakout()
    assert len(result) == 1
    assert result[0]['signal_date'] == '2024-07-04'
    detail = {x['code']: x['value'] for x in result[0]['pattern']['evidence']}
    assert detail['neckline_selection']['formation_coverage']['end_index'] == 2


def test_cached_weak_crossing_appends_forming_event_without_buy(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.bars = [_bar("2024-01-01", 9.5), _bar("2024-06-28", 9.8),
                    _bar("2024-07-01", 9.9), _bar("2024-07-02", 10.1)]
    session.date_strings = [str(bar["date"].date()) for bar in session.bars]
    session.atr_values = [0.2] * len(session.bars)
    pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                             known="2024-06-28", support_lower=8.0)
    _install_cached_candidate(session, pattern)
    session._record_instance_states([])

    signals = session._check_cached_breakout()

    assert signals == []
    weak = [event for event in session.instance_events() if event.get("event_type") == "weak_crossing"]
    assert len(weak) == 1
    assert weak[0]["event_date"] == "2024-07-02"
    assert weak[0]["state"] == "forming" and weak[0]["entry_eligible"] is False
    assert session.current_instances()[0].state == "forming"
    session._check_cached_breakout()
    assert len([event for event in session.instance_events()
                if event.get("event_type") == "weak_crossing"]) == 1


@pytest.mark.parametrize("terminal_state", ["invalidated", "target_reached"])
def test_cached_breakout_does_not_append_weak_crossing_after_terminal_event(terminal_state):
    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar(day, close) for day, close in zip(
        ["2024-01-01", "2024-06-28", "2024-07-01", "2024-07-02", "2024-07-03"],
        [9.5, 9.8, 9.9, 9.9, 10.1])]
    session.date_strings = [str(bar["date"].date()) for bar in session.bars]
    session.atr_values = [0.2] * len(session.bars)
    pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                             known="2024-06-28", support_lower=8.0)
    _install_cached_candidate(session, pattern)
    session.instance_ledger.append(pattern.to_dict(), "forming", "2024-07-01")
    session.instance_ledger.append(pattern.to_dict(), terminal_state, "2024-07-02")

    assert session._check_cached_breakout() == []

    events = [event for event in session.instance_events()
              if event["instance_id"] == pattern.pattern_id]
    assert [event["state"] for event in events] == ["forming", terminal_state]
    assert not any(event.get("event_type") == "weak_crossing" for event in events)


def test_cached_breakout_waits_for_six_calendar_months_from_first_trough(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-01-01", 9.5))
    session.append_bar(_bar("2024-06-28"))
    _install_cached_candidate(session, _cached_bottom())

    before_six_months = session.append_bar(_bar("2024-06-30", 10.0))
    on_six_month_boundary = session.append_bar(_bar("2024-07-01", 11.0))

    assert before_six_months == []
    assert len(on_six_month_boundary) == 1
    assert on_six_month_boundary[0]["signal_date"] == "2024-07-01"


def test_cached_breakout_pre_maturity_cross_consumes_same_frozen_boundary(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-01-01", 9.5))
    session.append_bar(_bar("2024-06-28", 10.0))
    pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                             known="2024-06-28", support_lower=8.0)
    pattern.neckline = {"slope": 0.0, "intercept": 10.0}
    _install_cached_candidate(session, pattern)

    early = session.append_bar(_bar("2024-06-30", 10.5))
    failed_retest = session.append_bar(_bar("2024-07-01", 9.9))
    after_maturity = session.append_bar(_bar("2024-07-02", 10.5))

    assert early == []
    assert failed_retest == []
    assert after_maturity == []
    assert session.consumed_neckline_phases
    pattern = next(iter(session.active.values()))["pattern"]
    detail = {item["code"]: item["value"] for item in pattern.evidence}
    assert pattern.state == "invalidated"
    assert detail["neckline_invalidation"]["date"] == "2024-07-01"
    assert detail["neckline_invalidation"]["reason"] == \
        "premature_breakout_fell_back_below_frozen_neckline"
    assert detail["neckline_selection"]["early_breakouts"][0]["date"] == "2024-06-30"
    next_low = Pivot(1, session.date_strings[1], 9.0, "low", 2,
                     session.date_strings[1], "3ATR|8%")
    new_structure = {"family": "w_bottom", "sequence": [next_low]}
    assert not session._boundary_is_consumed(new_structure, {"line": pattern.neckline, "tests": []})


def test_mature_confirmation_consumes_equivalent_line_across_candidate_ids(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-01-01", 9.5))
    session.append_bar(_bar("2024-06-28", 10.0))
    first_pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                                   known="2024-06-28", support_lower=8.0)
    first_pattern.neckline = {"slope": 0.0, "intercept": 10.0}
    _install_cached_candidate(session, first_pattern)

    first_breakout = session.append_bar(_bar("2024-07-01", 10.5))
    assert len(first_breakout) == 1
    assert first_breakout[0]["signal_date"] == "2024-07-01"

    later_pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                                   known="2024-06-28", support_lower=8.0)
    later_pattern.pattern_id = "same-structure-new-candidate-id"
    later_pattern.neckline = {"slope": 0.0, "intercept": 10.0}
    pivot = Pivot(0, "2024-01-01", 9.5, "low", 1, "2024-06-28", "3ATR|8%")
    session.active[("w_bottom", "3atr_8pct", ((0, "low"),))] = {
        "family": "w_bottom", "scale": "3atr_8pct", "sequence": [pivot],
        "structure_known_at": 1, "pattern": later_pattern, "needs_initial_eval": False,
    }

    session.append_bar(_bar("2024-07-02", 9.9))
    recross = session.append_bar(_bar("2024-07-03", 10.5))

    assert recross == []


def test_new_bottom_candidate_caches_its_calendar_maturity_date(monkeypatch):
    import caisen.daily_incremental as incremental

    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-01-31"), _bar("2024-02-02")]
    first = Pivot(0, "2024-01-31", 9.0, "low", 1, "2024-02-02", "2ATR|5%")
    second = Pivot(1, "2024-02-02", 9.1, "low", 1, "2024-02-02", "2ATR|5%")
    session.scale_pivots["2atr_5pct"] = [first, second]
    monkeypatch.setattr(incremental, "_structure_candidates",
                        lambda *_args, **_kwargs: [("w_bottom", [first, second])])

    session._refresh_structures()

    candidate = next(iter(session.active.values()))
    assert candidate["maturity_date"] == "2024-07-31"


def test_support_breach_permanently_invalidates_cached_structure_after_recovery(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-01-01", 9.5))
    session.append_bar(_bar("2024-07-01"))
    _install_cached_candidate(session, _cached_bottom(first_trough="2024-01-01",
                                                       frozen="2024-07-01", known="2024-06-30"))

    breach = session.append_bar(_bar("2024-07-02", 9.4))
    recovery = session.append_bar(_bar("2024-07-03", 10.0))
    later_neckline_break = session.append_bar(_bar("2024-07-04", 11.0))

    assert breach == []
    assert recovery == []
    assert later_neckline_break == []


def test_active_candidate_removal_writes_invalidated_event_for_frozen_instance(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-01-01", 9.5))
    session.append_bar(_bar("2024-07-01", 9.8))
    pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                             known="2024-06-28", support_lower=9.5)
    selection = next(item["value"] for item in pattern.evidence
                     if item["code"] == "neckline_selection")
    selection.update({
        "selected_candidate_id": "pair-1", "selected_line": pattern.neckline,
        "line_frozen_at": "2024-06-28", "structure_known_at": "2024-06-28"})
    tests = next(item["value"] for item in pattern.evidence if item["code"] == "neckline_tests")
    for test in tests:
        test["pair_id"] = "pair-1"
    identity = pattern_identity_metadata(pattern.to_dict())
    pattern.pattern_id = identity["instance_id"]
    pattern.structure_key = identity["structure_key"]
    pattern.boundary_version_id = identity["boundary_version_id"]
    _install_cached_candidate(session, pattern)
    session._record_instance_states([])
    assert session.instance_events()[-1]["state"] == "forming"

    session.append_bar(_bar("2024-07-02", 9.4))

    assert not session.active
    events = [event for event in session.instance_events()
              if event["instance_id"] == pattern.pattern_id]
    assert [event["state"] for event in events] == ["forming", "invalidated"]
    assert session.current_instances()[0].state == "invalidated"


def test_current_instances_use_latest_lifecycle_event_state():
    session = DailyRecognitionSession("000001", "stock")
    pattern = _cached_bottom()
    session.instance_ledger.append(pattern.to_dict(), "forming", "2024-06-28")
    session.instance_ledger.append(pattern.to_dict(), "confirmed", "2024-07-01",
                                    entry_eligible=True)
    session.instance_ledger.append(pattern.to_dict(), "focus", "2024-07-02")

    current = session.current_instances()

    assert len(current) == 1
    assert current[0].state == "focus"
    assert current[0].neckline == pattern.neckline


def test_distinct_pending_boundaries_and_confirmed_reline_remain_independent():
    def candidate(line, tests):
        freeze_date = tests[-1][0]
        selection = {"selected_candidate_id": "paired", "selected_line": line,
                     "line_frozen_at": freeze_date, "structure_known_at": freeze_date,
                     "selection_reason": "paired pressure evidence"}
        evidence = [
            {"code": "main_structure", "value": {"role": "main", "support_lower": 8.0}},
            {"code": "neckline_selection", "value": selection},
            {"code": "neckline_tests", "value": [
                {"pair_id": "paired", "group_id": f"g{i}", "pivot_date": date,
                 "level": line["intercept"], "normalized_residual": residual,
                 "test_span_bars": span, "valid": True}
                for i, (date, residual, span) in enumerate(tests)]},
        ]
        key_points = {"pivot_1_date": "2024-01-01", "pivot_1_kind": "low",
                      "pivot_1": 8.0, "pivot_2_date": "2024-02-01",
                      "pivot_2_kind": "low", "pivot_2": 8.1,
                      "scale": "5y", "pivot_scale": "2atr_5pct"}
        metadata = pattern_identity_metadata({"code": "000001", "kind": "stock",
            "timeframe": "1d", "pattern_type": "w_bottom", "key_points": key_points,
            "neckline": line, "evidence": evidence})
        return Pattern(pattern_id=metadata["instance_id"], code="000001", kind="stock",
            timeframe="1d", pattern_type="w_bottom", direction="bullish", state="forming",
            formed_at="2024-01-01", confirmed_at=None, key_points=key_points,
            neckline=line, invalidation=7.5, evidence=evidence,
            structure_key=metadata["structure_key"],
            boundary_version_id=metadata["boundary_version_id"])

    shorter = candidate({"slope": 0.0, "intercept": 10.0}, [
        ("2024-03-01", 0.1, 10), ("2024-04-01", 0.1, 10)])
    best_at_freeze = candidate({"slope": 0.0, "intercept": 10.1}, [
        ("2024-02-01", 0.2, 10), ("2024-03-01", 0.1, 30), ("2024-05-01", 0.05, 30)])
    later_better = candidate({"slope": 0.0, "intercept": 9.9}, [
        ("2024-02-01", 0.0, 50), ("2024-04-01", 0.0, 50), ("2024-06-01", 0.0, 50),
        ("2024-07-01", 0.0, 50)])
    same_line_new_tests = candidate({"slope": 0.0, "intercept": 10.0}, [
        ("2024-03-01", 0.0, 20), ("2024-05-01", 0.0, 40)])
    assert same_line_new_tests.boundary_version_id != shorter.boundary_version_id
    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-07-01")]
    session.date_strings = ["2024-07-01"]
    session.active["shorter"] = {"pattern": shorter}
    session.active["best"] = {"pattern": best_at_freeze}
    session.active["later"] = {"pattern": later_better}
    same_boundary_scale = replace(later_better,
        key_points={**best_at_freeze.key_points, "pivot_1_index": 900,
                    "pivot_scale": "3atr_8pct"})
    same_boundary_scale = replace(same_boundary_scale,
        key_points={**later_better.key_points, "pivot_1_index": 900,
                    "pivot_scale": "3atr_8pct"})
    session.active["best_scale_copy"] = {"pattern": same_boundary_scale}

    session._record_instance_states([])

    structure_key = pattern_identity_metadata(best_at_freeze.to_dict())["structure_key"]
    chosen_ids = session.selected_boundary_by_structure[structure_key]
    assert chosen_ids == {shorter.boundary_version_id, best_at_freeze.boundary_version_id,
                          later_better.boundary_version_id}
    session._record_instance_states([])
    filtered = session._selected_boundary_patterns([shorter, best_at_freeze, later_better])
    assert {item.pattern_id for item in filtered} == {
        shorter.pattern_id, best_at_freeze.pattern_id, later_better.pattern_id}
    assert {event["instance_id"] for event in session.instance_events()} == {
        shorter.pattern_id, best_at_freeze.pattern_id, later_better.pattern_id}
    assert {item["pivot_scale"] for item in next(event for event in session.instance_events()
            if event["instance_id"] == later_better.pattern_id)["pattern"]["multiscale_evidence"]} == {
        "2atr_5pct", "3atr_8pct"}
    merged_scale = session._selected_boundary_patterns([later_better, same_boundary_scale])
    assert len(merged_scale) == 1
    assert {item["pivot_scale"] for item in merged_scale[0].multiscale_evidence} == {
        "2atr_5pct", "3atr_8pct"}

    session_with_confirmed_line = DailyRecognitionSession("000001", "stock")
    session_with_confirmed_line.instance_ledger.append(
        shorter.to_dict(), "confirmed", "2024-05-01", entry_eligible=True)
    session_with_confirmed_line._select_structure_boundaries(
        [shorter, best_at_freeze, later_better])
    retained_ids = session_with_confirmed_line.selected_boundary_by_structure[structure_key]
    assert retained_ids == {shorter.boundary_version_id, best_at_freeze.boundary_version_id,
                            later_better.boundary_version_id}
    confirmed_line_session = DailyRecognitionSession("000001", "stock")
    confirmed_line_session.instance_ledger.append(
        shorter.to_dict(), "confirmed", "2024-05-01", entry_eligible=True)
    chosen_after_same_line_retest = confirmed_line_session._selected_boundary_patterns(
        [shorter, same_line_new_tests, later_better])
    assert {item.pattern_id for item in chosen_after_same_line_retest} == {
        shorter.pattern_id, later_better.pattern_id}


def test_reline_keeps_prior_pending_instance_and_allows_new_line_entry():
    def candidate(line, frozen_date):
        pattern = _cached_bottom(first_trough="2024-01-01", frozen=frozen_date,
                                 known=frozen_date, support_lower=8.0)
        pattern.key_points["pivot_2_date"] = "2024-06-01"
        pattern.neckline = line
        selection = next(item["value"] for item in pattern.evidence
                         if item["code"] == "neckline_selection")
        selection.update({"selected_candidate_id": "pair-" + frozen_date,
                          "selected_line": line, "line_frozen_at": frozen_date,
                          "structure_known_at": frozen_date})
        tests_detail = next(item["value"] for item in pattern.evidence
                            if item["code"] == "neckline_tests")
        for index, test in enumerate(tests_detail):
            test.update({"pair_id": "pair-" + frozen_date, "group_id": f"{frozen_date}-{index}",
                         "pivot_date": frozen_date, "level": line["intercept"],
                         "valid": True})
        identity = pattern_identity_metadata({**pattern.to_dict(), "pattern_id": None,
            "structure_key": None, "boundary_version_id": None})
        pattern.pattern_id = identity["instance_id"]
        pattern.structure_key = identity["structure_key"]
        pattern.boundary_version_id = identity["boundary_version_id"]
        return pattern

    old_line = candidate({"slope": 0.0, "intercept": 10.0}, "2024-06-28")
    new_line = candidate({"slope": 0.0, "intercept": 10.4}, "2024-07-10")
    assert old_line.structure_key == new_line.structure_key
    assert old_line.boundary_version_id != new_line.boundary_version_id

    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-06-28"), _bar("2024-07-10")]
    session.date_strings = ["2024-06-28", "2024-07-10"]
    session.active["old"] = {"pattern": old_line}
    session._record_instance_states([])
    assert session.instance_ledger.latest_events[0].state == "forming"

    session.active["new"] = {"pattern": new_line}
    session.date_strings[-1] = "2024-07-11"
    session._record_instance_states([])

    states = {event.instance_id: event.state for event in session.instance_ledger.latest_events}
    assert states == {old_line.pattern_id: "forming", new_line.pattern_id: "forming"}
    session.instance_ledger.append(old_line.to_dict(), "confirmed", "2024-07-12",
                                   entry_eligible=True)
    session.instance_ledger.append(new_line.to_dict(), "confirmed", "2024-07-12",
                                   entry_eligible=True)
    assert {event.instance_id: event.state for event in session.instance_ledger.latest_events} == {
        old_line.pattern_id: "confirmed", new_line.pattern_id: "confirmed"}


def test_new_frozen_line_on_same_lows_can_emit_a_separate_buy_after_old_line_fails():
    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-01-01", 9.0), _bar("2024-06-28", 9.4),
                    _bar("2024-07-01", 9.8), _bar("2024-07-02", 11.0)]
    session.date_strings = [str(bar["date"].date()) for bar in session.bars]
    session.atr_values = [0.2] * len(session.bars)
    old_line = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                              known="2024-06-28", support_lower=8.0)
    session.instance_ledger.append(old_line.to_dict(), "forming", "2024-06-28")
    session.instance_ledger.append(old_line.to_dict(), "confirmed", "2024-06-29",
                                   entry_eligible=True)
    session.instance_ledger.append(old_line.to_dict(), "invalidated", "2024-07-01")

    new_line = replace(old_line, neckline={"slope": 0.0, "intercept": 10.5},
                       evidence=deepcopy(old_line.evidence),
                       pattern_id=None, boundary_version_id=None, structure_key=None)
    selection = next(item["value"] for item in new_line.evidence
                     if item["code"] == "neckline_selection")
    selection.update({"selected_candidate_id": "new-pair", "selected_line": new_line.neckline,
                      "line_frozen_at": "2024-07-01", "structure_known_at": "2024-07-01"})
    for test in next(item["value"] for item in new_line.evidence
                     if item["code"] == "neckline_tests"):
        test["pair_id"] = "new-pair"
        test["level"] = 10.5
    new_line.evidence.append({"code": "boundary_available_at", "value": {
        "date": "2024-07-01", "index": 2}})
    identity = pattern_identity_metadata(new_line.to_dict())
    new_line.pattern_id = identity["instance_id"]
    new_line.structure_key = identity["structure_key"]
    new_line.boundary_version_id = identity["boundary_version_id"]
    _install_cached_candidate(session, new_line)

    signals = session._check_cached_breakout()

    assert len(signals) == 1
    assert signals[0]["signal_date"] == "2024-07-02"
    assert signals[0]["pattern_ids"] == [new_line.pattern_id]
    session._record_instance_states(signals)
    latest = {event.instance_id: event.state for event in session.instance_ledger.latest_events}
    assert latest == {old_line.pattern_id: "invalidated", new_line.pattern_id: "confirmed"}


def test_nested_suppression_is_directional_and_independent_of_input_order():
    outer = Pattern(
        pattern_id="outer-structure", code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming", formed_at="2023-01-01",
        confirmed_at=None,
        key_points={"pivot_1_date": "2023-01-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-01-01", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom", "first_trough_date": "2023-01-01"}},
                  {"code": "main_selection", "value": {"selected": True}}])
    inner = Pattern(
        pattern_id="inner-structure", code="000001", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="forming", formed_at="2023-03-01",
        confirmed_at=None,
        key_points={"pivot_1_date": "2023-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2023-06-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2023-09-01", "pivot_3_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "triangle_bottom", "first_trough_date": "2023-03-01"}},
                  {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids([outer, inner], "2024-01-01") == {"inner-structure"}
    assert DailyRecognitionSession._nested_ids([inner, outer], "2024-01-01") == {"inner-structure"}


def test_suppressed_middle_structure_cannot_suppress_its_own_child():
    from caisen.main_bottoms import _nested_main_parent_map

    def shape(pattern_id, family, lows, other_dates=()):
        key_points = {}
        number = 1
        for day in lows:
            key_points.update({f"pivot_{number}_date": day,
                               f"pivot_{number}_kind": "low"})
            number += 1
        for day in other_dates:
            key_points.update({f"pivot_{number}_date": day,
                               f"pivot_{number}_kind": "high"})
            number += 1
        return Pattern(
            pattern_id=pattern_id, code="000001", kind="stock", timeframe="1d",
            pattern_type=family, direction="bullish", state="forming",
            formed_at=min(lows), confirmed_at=None, key_points=key_points,
            evidence=[{"code": "main_structure", "value": {
                "role": "main", "family": family,
                "first_trough_date": min(lows)}},
                {"code": "main_selection", "value": {"selected": True}}])

    outer = shape("outer", "w_bottom", ["2023-01-01", "2025-01-01"])
    middle = shape("middle", "inverse_head_shoulders",
                   ["2023-03-01", "2023-11-01", "2024-07-01"])
    inner = shape("inner", "triangle_bottom",
                  ["2023-05-01", "2023-09-01", "2024-01-01"],
                  ["2023-07-01", "2023-12-01"])

    nested = _nested_main_parent_map([outer, middle, inner], [outer, middle, inner],
                                     "2025-02-01")

    assert {key: value.pattern_id for key, value in nested.items()} == {
        "middle": "outer", "inner": "outer"}


def test_parent_hierarchy_does_not_use_outer_structure_before_it_was_known():
    from caisen.main_bottoms import _nested_main_parent_map

    def shape(pattern_id, family, lows, known_at, observed_at):
        key_points = {}
        for number, day in enumerate(lows, 1):
            key_points.update({f"pivot_{number}_date": day,
                               f"pivot_{number}_kind": "low"})
        return Pattern(
            pattern_id=pattern_id, code="000001", kind="stock", timeframe="1d",
            pattern_type=family, direction="bullish", state="forming",
            formed_at=min(lows), last_observed_date=observed_at, confirmed_at=None,
            key_points=key_points,
            evidence=[{"code": "main_structure", "value": {
                "role": "main", "family": family,
                "first_trough_date": min(lows)}},
                {"code": "main_selection", "value": {"selected": True}},
                {"code": "neckline_selection", "value": {
                    "structure_known_at": known_at}}])

    outer = shape("outer-late", "w_bottom", ["2023-01-01", "2025-01-01"],
                  "2024-12-15", "2025-01-01")
    middle = shape("middle-known", "inverse_head_shoulders",
                   ["2023-03-01", "2023-06-01", "2023-10-01"],
                   "2023-11-01", "2023-12-01")
    inner = shape("inner-known", "triangle_bottom",
                  ["2023-05-01", "2023-07-01", "2023-09-01"],
                  "2024-06-10", "2024-07-01")

    nested = _nested_main_parent_map([outer, middle, inner], [outer, middle, inner])

    assert nested["inner-known"].pattern_id == "middle-known"


def test_already_suppressed_structure_cannot_suppress_a_third_structure():
    suppressed_parent = Pattern(
        pattern_id="suppressed-middle", code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming", formed_at="2023-03-01",
        confirmed_at=None, parent_id="outer-parent",
        key_points={"pivot_1_date": "2023-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2023-09-01", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom", "first_trough_date": "2023-03-01"}},
                  {"code": "main_selection", "value": {"selected": True}},
                  {"code": "execution_suppression", "value": {
                      "suppressed": True, "parent_id": "outer-parent"}}])
    inner = Pattern(
        pattern_id="inner-child", code="000001", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="forming",
        formed_at="2023-04-01", confirmed_at=None,
        key_points={"pivot_1_date": "2023-04-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2023-06-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2023-08-01", "pivot_3_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "triangle_bottom", "first_trough_date": "2023-04-01"}},
                  {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids(
        [suppressed_parent, inner], "2023-09-01") == set()


def test_boundary_selection_cache_reuses_unchanged_frozen_instance_set(monkeypatch):
    def candidate(line, offset):
        pattern = _cached_bottom()
        changed_line = dict(line)
        selection = next(item["value"] for item in pattern.evidence
                         if item["code"] == "neckline_selection")
        selection["selected_line"] = changed_line
        selection["line_frozen_at"] = "2024-06-28"
        tests = next(item["value"] for item in pattern.evidence
                     if item["code"] == "neckline_tests")
        for test in tests:
            test["pair_id"] = f"boundary-{offset}"
        selection["selected_candidate_id"] = f"boundary-{offset}"
        for test in tests:
            test["level"] = changed_line["intercept"]
        pattern.neckline = changed_line
        identity = pattern_identity_metadata({**pattern.to_dict(),
            "structure_key": None, "boundary_version_id": None, "pattern_id": None})
        pattern.pattern_id = identity["instance_id"]
        pattern.structure_key = identity["structure_key"]
        pattern.boundary_version_id = identity["boundary_version_id"]
        return pattern

    first = candidate({"slope": 0.0, "intercept": 10.0}, "a")
    second = candidate({"slope": 0.0, "intercept": 10.1}, "b")
    assert first.structure_key == second.structure_key
    assert first.boundary_version_id != second.boundary_version_id
    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-07-01")]
    session.date_strings = ["2024-07-01"]
    session.active["first"] = {"pattern": first}
    session.active["second"] = {"pattern": second}
    original_rank = session._boundary_pattern_rank
    calls = []
    monkeypatch.setattr(session, "_boundary_pattern_rank",
                        lambda pattern: (calls.append(pattern.pattern_id), original_rank(pattern))[1])

    session._record_instance_states([])
    assert calls

    calls.clear()
    session.bars.append(_bar("2024-07-02"))
    session.date_strings.append("2024-07-02")
    session._record_instance_states([])
    assert calls  # The first forming events changed the ledger revision.

    calls.clear()
    session.bars.append(_bar("2024-07-03"))
    session.date_strings.append("2024-07-03")
    session._record_instance_states([])
    assert calls == []
    assert session.selected_boundary_by_structure[first.structure_key] == {
        first.boundary_version_id, second.boundary_version_id}


def test_terminal_boundary_is_not_revived_while_distinct_live_line_remains_eligible():
    first = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                           known="2024-06-28", support_lower=8.0)
    second = replace(first, neckline={"slope": 0.0, "intercept": 10.5}, evidence=[
        {"code": "main_structure", "value": {"role": "main", "support_lower": 8.0}},
        {"code": "neckline_selection", "value": {
            "selected_candidate_id": "paired-second", "selected_line": {"slope": 0.0, "intercept": 10.5},
            "line_frozen_at": "2024-07-10", "structure_known_at": "2024-07-10"}},
        {"code": "neckline_tests", "value": [
            {"pair_id": "paired-second", "group_id": "g1", "pivot_date": "2024-06-01",
             "level": 10.5, "normalized_residual": 0.1, "valid": True},
            {"pair_id": "paired-second", "group_id": "g2", "pivot_date": "2024-06-10",
             "level": 10.5, "normalized_residual": 0.1, "valid": True}]}],
        boundary_version_id=None, structure_key=None)
    second.evidence[-1]["value"][0]["normalized_residual"] = 0.0
    second.evidence[-1]["value"][1]["normalized_residual"] = 0.0
    second.evidence[-1]["value"][0]["pivot_date"] = "2024-07-10"
    second.evidence[-1]["value"][1]["pivot_date"] = "2024-07-10"
    second.evidence[-1]["value"][0]["test_span_bars"] = 40
    second.evidence[-1]["value"][1]["test_span_bars"] = 40
    second.evidence[1]["value"]["line_frozen_at"] = "2024-07-10"
    for pattern in (first, second):
        identity = pattern_identity_metadata(pattern.to_dict())
        pattern.pattern_id = identity["instance_id"]
        pattern.structure_key = identity["structure_key"]
        pattern.boundary_version_id = identity["boundary_version_id"]
    assert pattern_instance_is_frozen(first)
    assert pattern_instance_is_frozen(first)

    session = DailyRecognitionSession("000001", "stock")
    chosen = session._selected_boundary_patterns([first, second])
    assert {item.pattern_id for item in chosen} == {first.pattern_id, second.pattern_id}
    session.instance_ledger.append(first.to_dict(), "forming", "2024-06-28")
    session.instance_ledger.append(first.to_dict(), "invalidated", "2024-07-01")

    replacement = session._selected_boundary_patterns([first, second])

    assert [item.pattern_id for item in replacement] == [second.pattern_id]
    assert session._selected_boundary_patterns([first]) == []


def test_same_level_same_anchor_boundary_variants_are_not_mutually_suppressed():
    first = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                           known="2024-06-28", support_lower=8.0)
    second = replace(first, pattern_id="variant-2",
                     neckline={"slope": 0.0, "intercept": 10.5})

    assert DailyRecognitionSession._nested_ids([first, second], "2024-07-01") == set()


def test_temporally_outer_main_structure_suppresses_inner_even_if_its_pivot_scale_is_finer():
    outer = Pattern(
        pattern_id="outer-fine-scale", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at="2023-01-01", confirmed_at=None,
        key_points={"pivot_1_date": "2023-01-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2023-12-01", "pivot_2_kind": "low",
                    "pivot_scale": "2atr_5pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom", "pivot_scale": "2atr_5pct",
            "first_trough_date": "2023-01-01", "structure_end_date": "2024-01-01"}},
            {"code": "main_selection", "value": {"selected": True}}],
        last_observed_date="2024-01-01")
    inner = Pattern(
        pattern_id="inner-coarse-scale", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed",
        formed_at="2023-04-01", confirmed_at=100, confirmation_date="2023-10-01",
        key_points={"pivot_1_date": "2023-04-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2023-06-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2023-08-01", "pivot_3_kind": "low",
                    "pivot_4_date": "2023-10-01", "pivot_4_kind": "high",
                    "pivot_scale": "5atr_12pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "triangle_bottom", "pivot_scale": "5atr_12pct"}},
            {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids([outer, inner], "2023-10-01") == {"inner-coarse-scale"}


def test_same_level_ihs_with_same_outer_anchors_are_not_mutually_suppressed():
    def candidate(pattern_id, middle):
        return Pattern(
            pattern_id=pattern_id, code="600633", kind="stock", timeframe="1d",
            pattern_type="inverse_head_shoulders", direction="bullish", state="forming",
            formed_at="2024-01-22", confirmed_at=None, last_observed_date="2025-02-06",
            key_points={"pivot_1_date": "2024-01-22", "pivot_1_kind": "low",
                        "pivot_2_date": middle, "pivot_2_kind": "low",
                        "pivot_3_date": "2025-01-13", "pivot_3_kind": "low"},
            evidence=[
                {"code": "main_structure", "value": {
                    "role": "main", "family": "inverse_head_shoulders",
                    "pivot_scale": "2atr_5pct"}},
                {"code": "neckline_selection", "value": {
                    "structure_known_at": "2025-01-13"}},
            ])

    left = candidate("left", "2024-07-18")
    right = candidate("right", "2024-09-10")

    assert DailyRecognitionSession._nested_ids([left, right], "2025-02-06") == set()


def test_ihs_and_w_sharing_head_and_right_shoulder_are_overlapping_alternatives():
    ihs = Pattern(
        pattern_id="outer-ihs", code="516630", kind="etf", timeframe="1d",
        pattern_type="inverse_head_shoulders", direction="bullish", state="forming",
        formed_at="2023-10-24", confirmed_at=None,
        last_observed_date="2024-10-08",
        key_points={"pivot_1_date": "2023-10-24", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-02-05", "pivot_2_kind": "low",
                    "pivot_3_date": "2024-09-09", "pivot_3_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "inverse_head_shoulders",
            "first_trough_date": "2023-10-24", "structure_end_date": "2024-10-08"}},
                  {"code": "main_selection", "value": {"selected": True}}])
    w_bottom = Pattern(
        pattern_id="inner-w", code="516630", kind="etf", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-02-05", confirmed_at=744, confirmation_date="2024-10-08",
        last_observed_date="2024-10-08",
        key_points={"pivot_1_date": "2024-02-05", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-09-09", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom",
            "first_trough_date": "2024-02-05", "structure_end_date": "2024-09-09"}},
                  {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids(
        [ihs, w_bottom], "2024-10-08") == set()


def test_cross_family_nested_child_sharing_only_one_low_remains_suppressed():
    ihs = Pattern(
        pattern_id="outer-ihs-one-shared", code="516630", kind="etf", timeframe="1d",
        pattern_type="inverse_head_shoulders", direction="bullish", state="forming",
        formed_at="2023-10-24", confirmed_at=None, last_observed_date="2024-10-08",
        key_points={"pivot_1_date": "2023-10-24", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-02-05", "pivot_2_kind": "low",
                    "pivot_3_date": "2024-09-09", "pivot_3_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "inverse_head_shoulders",
            "first_trough_date": "2023-10-24", "structure_end_date": "2024-10-08"}},
                  {"code": "main_selection", "value": {"selected": True}}])
    inner_w = Pattern(
        pattern_id="inner-w-one-shared", code="516630", kind="etf", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at="2024-02-05", confirmed_at=None, last_observed_date="2024-07-01",
        key_points={"pivot_1_date": "2024-02-05", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-07-01", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom",
            "first_trough_date": "2024-02-05", "structure_end_date": "2024-07-01"}},
                  {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids(
        [ihs, inner_w], "2024-10-08") == {"inner-w-one-shared"}


def test_live_observation_window_does_not_make_an_old_pattern_parent_of_a_new_bottom():
    old = Pattern(
        pattern_id="old-confirmed-bottom", code="516630", kind="etf", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2022-04-27", confirmed_at=100, confirmation_date="2022-10-27",
        last_observed_date="2024-10-08",
        key_points={"pivot_1_date": "2022-04-27", "pivot_1_kind": "low",
                    "pivot_2_date": "2022-10-11", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom",
            "first_trough_date": "2022-04-27", "latest_anchor_date": "2022-10-11",
            # This is an observation/lifecycle endpoint, not a structural anchor.
            "structure_end_date": "2024-10-08"}},
                  {"code": "main_selection", "value": {"selected": True}}])
    newer = Pattern(
        pattern_id="new-2024-bottom", code="516630", kind="etf", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-02-05", confirmed_at=744, confirmation_date="2024-10-08",
        last_observed_date="2024-10-08",
        key_points={"pivot_1_date": "2024-02-05", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-09-09", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom",
            "first_trough_date": "2024-02-05", "latest_anchor_date": "2024-09-09",
            "structure_end_date": "2024-10-08"}},
                  {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids([old, newer], "2024-10-08") == set()


def test_overlapping_main_structures_with_same_start_are_not_mutually_suppressed():
    outer = Pattern(
        pattern_id="outer-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at="2023-01-01", confirmed_at=None, last_observed_date="2025-01-01",
        key_points={"pivot_1_date": "2023-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2023-12-01", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom", "first_trough_date": "2023-01-01",
            "structure_end_date": "2025-01-01"}},
                  {"code": "main_selection", "value": {"selected": True}}])
    overlapping = Pattern(
        pattern_id="overlapping-triangle", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="forming",
        formed_at="2023-02-01", confirmed_at=None, last_observed_date="2025-01-01",
        key_points={"pivot_1_date": "2023-02-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-11-01", "pivot_2_kind": "low"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "triangle_bottom", "first_trough_date": "2023-01-01",
            "structure_end_date": "2025-01-01"}},
                  {"code": "main_selection", "value": {"selected": True}}])

    assert DailyRecognitionSession._nested_ids([outer, overlapping], "2025-01-01") == set()


def test_changed_frozen_line_is_separate_even_when_within_two_tick_tolerance():
    first = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                           known="2024-06-28", support_lower=8.0)
    second = replace(first, neckline={"slope": 0.0, "intercept": 10.01},
                     evidence=deepcopy(first.evidence),
                     boundary_version_id=None, structure_key=None)
    second.evidence[1]["value"]["selected_line"] = dict(second.neckline)
    second.evidence[2]["value"] = deepcopy(first.evidence[2]["value"])
    for index, test in enumerate(second.evidence[2]["value"]):
        test["pivot_index"] = index
        test["tolerance"] = 0.02
    for index, test in enumerate(first.evidence[2]["value"]):
        test["pivot_index"] = index
        test["tolerance"] = 0.02
    for pattern in (first, second):
        identity = pattern_identity_metadata(pattern.to_dict())
        pattern.pattern_id = identity["instance_id"]
        pattern.structure_key = identity["structure_key"]
        pattern.boundary_version_id = identity["boundary_version_id"]

    session = DailyRecognitionSession("000001", "stock")
    session.instance_ledger.append(first.to_dict(), "confirmed", "2024-06-28",
                                    entry_eligible=True)
    selected = session._selected_boundary_patterns([first, second])

    assert len(selected) == 2


def test_live_distinct_frozen_line_can_enter_and_same_line_scale_copy_is_one_entry():
    first = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                           known="2024-06-28", support_lower=8.0)
    second = replace(first, neckline={"slope": 0.0, "intercept": 10.5},
                     evidence=deepcopy(first.evidence),
                     boundary_version_id=None, structure_key=None)
    identity = pattern_identity_metadata(second.to_dict())
    second.pattern_id = identity["instance_id"]
    second.structure_key = identity["structure_key"]
    second.boundary_version_id = identity["boundary_version_id"]
    same_line_scale = replace(second,
        key_points={**second.key_points, "pivot_scale": "3atr_8pct"})
    session = DailyRecognitionSession("000001", "stock")
    session.instance_ledger.append(first.to_dict(), "forming", "2024-06-28")

    independent = session._independent_candidates([first, second],
                                                   [first, second, same_line_scale],
                                                   "2024-07-01")

    assert {pattern.pattern_id for pattern in independent} == {
        first.pattern_id, second.pattern_id}


def test_new_frozen_boundary_revision_remains_buy_eligible_after_old_instance_terminates():
    old = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                         known="2024-06-28", support_lower=8.0)
    revised = replace(old, neckline={"slope": 0.0, "intercept": 10.5},
                      evidence=deepcopy(old.evidence), boundary_version_id=None,
                      structure_key=None)
    revised.evidence[1]["value"].update({
        "selected_candidate_id": "paired-revised",
        "selected_line": dict(revised.neckline),
        "line_frozen_at": "2024-07-10", "structure_known_at": "2024-07-10"})
    revised.evidence[2]["value"] = [
        {"pair_id": "paired-revised", "group_id": "g1", "pivot_date": "2024-06-01",
         "level": 10.5, "normalized_residual": 0.1, "valid": True},
        {"pair_id": "paired-revised", "group_id": "g2", "pivot_date": "2024-06-10",
         "level": 10.5, "normalized_residual": 0.1, "valid": True},
    ]
    identity = pattern_identity_metadata(revised.to_dict())
    revised.pattern_id = identity["instance_id"]
    revised.structure_key = identity["structure_key"]
    revised.boundary_version_id = identity["boundary_version_id"]
    revised.state = "confirmed"
    revised.confirmation_date = "2024-07-15"
    revised.confirmed_at = 100
    assert pattern_instance_is_frozen(revised)
    assert revised.structure_key == old.structure_key
    assert revised.boundary_version_id != old.boundary_version_id
    assert revised.pattern_id != old.pattern_id

    session = DailyRecognitionSession("000001", "stock")
    session.instance_ledger.append(old.to_dict(), "forming", "2024-06-28")
    session.emitted_ids.add(old.pattern_id)
    assert session._independent_candidates([old, revised], [revised], "2024-07-09") == []
    while_old_line_is_live = session._independent_candidates(
        [old, revised], [revised], "2024-07-15")
    assert [item.pattern_id for item in while_old_line_is_live] == [revised.pattern_id]

    session.instance_ledger.append(old.to_dict(), "invalidated", "2024-07-16")
    candidates = session._independent_candidates([old, revised], [revised], "2024-07-17")

    assert [item.pattern_id for item in candidates] == [revised.pattern_id]
    entry = session.instance_ledger.append(revised.to_dict(), "confirmed", "2024-07-15",
                                           entry_eligible=True)
    assert entry.instance_id == revised.pattern_id
    assert entry.entry_eligible is True


def test_same_bottom_anchors_can_emit_again_for_a_changed_frozen_neckline():
    old = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                         known="2024-06-28", support_lower=8.0)
    revised = replace(old, neckline={"slope": 0.0, "intercept": 10.04},
                      evidence=deepcopy(old.evidence), boundary_version_id=None,
                      structure_key=None)
    revised.evidence[1]["value"]["selected_line"] = dict(revised.neckline)
    revised.evidence[2]["value"] = [
        {**item, "level": 10.04} for item in revised.evidence[2]["value"]]
    identity = pattern_identity_metadata(revised.to_dict())
    revised.pattern_id = identity["instance_id"]
    revised.structure_key = identity["structure_key"]
    revised.boundary_version_id = identity["boundary_version_id"]
    session = DailyRecognitionSession("000001", "stock")
    session.emitted_ids.add(old.pattern_id)
    old_detail = {item["code"]: item["value"] for item in old.evidence}
    revised_detail = {item["code"]: item["value"] for item in revised.evidence}
    from caisen.main_bottoms import _same_neckline_boundary_phase
    old_tests = [{**item, "pivot_index": index, "tolerance": 0.1}
                 for index, item in enumerate(old_detail["neckline_tests"], start=10)]
    revised_tests = [{**item, "pivot_index": index, "tolerance": 0.1}
                     for index, item in enumerate(revised_detail["neckline_tests"], start=10)]
    old_phase = {"line": old.neckline, "tests": old_tests,
                 "consumed_index": 100}
    revised_phase = {"line": revised.neckline, "tests": revised_tests,
                     "consumed_index": 100}

    assert not _same_neckline_boundary_phase(old_phase, revised_phase, 0.01)

    eligible = session._independent_candidates([old], [revised], "2024-07-01")

    assert [item.pattern_id for item in eligible] == [revised.pattern_id]


def test_new_neckline_on_same_troughs_emits_fresh_buy_after_its_freeze_while_old_instance_is_live():
    old = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                         known="2024-06-28", support_lower=8.0)
    revised = replace(old, neckline={"slope": 0.0, "intercept": 10.04},
                      evidence=deepcopy(old.evidence), boundary_version_id=None,
                      structure_key=None, confirmation_date=None, confirmed_at=None)
    revised.evidence[1]["value"].update({
        "selected_candidate_id": "paired-revised",
        "selected_line": dict(revised.neckline),
        "line_frozen_at": "2024-07-10", "structure_known_at": "2024-07-10",
        "effective_start_date": "2024-01-01"})
    revised.evidence[2]["value"] = [
        {"pair_id": "paired-revised", "group_id": "retest-1", "pivot_date": "2024-01-01",
         "pivot_index": 0, "level": 10.04, "tolerance": 0.1,
         "valid": True, "normalized_residual": 0.1},
        {"pair_id": "paired-revised", "group_id": "retest-2", "pivot_date": "2024-06-28",
         "pivot_index": 1, "level": 10.04, "tolerance": 0.1,
         "valid": True, "normalized_residual": 0.1},
    ]
    revised.evidence.append({"code": "boundary_available_at", "value": {
        "date": "2024-07-10", "index": 1,
        "reason": "paired_boundary_discovered_causally_on_this_prefix"}})
    identity = pattern_identity_metadata(revised.to_dict())
    revised.pattern_id = identity["instance_id"]
    revised.structure_key = identity["structure_key"]
    revised.boundary_version_id = identity["boundary_version_id"]

    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-01-01", 9.0), _bar("2024-07-10", 9.8),
                    _bar("2024-07-15", 10.8)]
    session.date_strings = [str(bar["date"].date()) for bar in session.bars]
    session.atr_values = [0.2, 0.2, 0.2]
    sequence = [Pivot(0, "2024-01-01", 8.0, "low", 0, "2024-01-01", "2ATR|5%"),
                Pivot(1, "2024-07-10", 8.1, "low", 1, "2024-07-10", "2ATR|5%")]
    candidate = {"family": "w_bottom", "scale": "2atr_5pct", "sequence": sequence,
                 "maturity_date": "2024-07-01", "pattern": revised,
                 "frozen_patterns": {old.pattern_id: old, revised.pattern_id: revised},
                 "needs_initial_eval": False}
    session.active[("w_bottom", "2atr_5pct", ((0, "low"), (1, "low")))] = candidate
    session.instance_ledger.append(old.to_dict(), "forming", "2024-06-28")
    session.instance_ledger.append(old.to_dict(), "confirmed", "2024-07-02", entry_eligible=True)
    session._remember_consumed_boundary(
        candidate, session._pattern_boundary_phase(old, 1, "buffered_breakout"))

    signals = session._check_cached_breakout()
    session._record_instance_states(signals)

    assert len(signals) == 1
    assert signals[0]["signal_date"] == "2024-07-15"
    assert signals[0]["pattern_ids"] == [revised.pattern_id]
    session.bars.append(_bar("2024-07-16", 10.9))
    session.date_strings.append("2024-07-16")
    session.atr_values.append(0.2)
    assert session._check_cached_breakout() == []
    confirmed = [event for event in session.instance_events()
                 if event["entry_eligible"] is True]
    assert [(event["instance_id"], event["event_date"]) for event in confirmed] == [
        (old.pattern_id, "2024-07-02"), (revised.pattern_id, "2024-07-15")]
    old_history = [event for event in session.instance_events()
                   if event["instance_id"] == old.pattern_id]
    assert max(old_history, key=lambda event: event["event_seq"])["state"] == "confirmed"


def test_new_pressure_pair_rechecks_frozen_candidate_for_an_independent_line(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-01-01", 9.0), _bar("2024-07-01", 9.2)]
    session.date_strings = [str(bar["date"].date()) for bar in session.bars]
    session.atr_values = [0.2, 0.2]
    sequence = [Pivot(0, "2024-01-01", 8.0, "low", 0, "2024-01-01", "2ATR|5%"),
                Pivot(1, "2024-07-01", 8.1, "low", 1, "2024-07-01", "2ATR|5%")]
    candidate = {"family": "w_bottom", "scale": "2atr_5pct", "sequence": sequence,
                 "pattern": None, "needs_initial_eval": True}
    session.active[("w_bottom", "2atr_5pct", ((0, "low"), (1, "low")))] = candidate
    first = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                           known="2024-06-28", support_lower=8.0)
    second = replace(first, neckline={"slope": 0.0, "intercept": 10.5},
                     evidence=deepcopy(first.evidence),
                     boundary_version_id=None, structure_key=None)
    second.evidence[1]["value"].update({
        "selected_candidate_id": "paired-next", "selected_line": second.neckline,
        "line_frozen_at": "2024-07-02", "structure_known_at": "2024-07-02"})
    for index, test in enumerate(second.evidence[2]["value"]):
        test["pair_id"] = "paired-next"
        test["pivot_date"] = "2024-07-02" if index else test["pivot_date"]
    for pattern in (first, second):
        identity = pattern_identity_metadata(pattern.to_dict())
        pattern.pattern_id = identity["instance_id"]
        pattern.structure_key = identity["structure_key"]
        pattern.boundary_version_id = identity["boundary_version_id"]
    assert pattern_instance_is_frozen(first)
    calls = []

    def make_pattern(*_args, consumed_boundary_phases=(), **_kwargs):
        calls.append(list(consumed_boundary_phases))
        return [first] if len(calls) == 1 else [second]

    monkeypatch.setattr("caisen.daily_incremental._make_main_pattern", make_pattern)
    monkeypatch.setattr(session, "_has_pressure_test_pair", lambda _candidate: True)
    monkeypatch.setattr(session, "_pressure_pivot_can_change_candidate", lambda *_args: True)

    session._evaluate(Pivot(2, "2024-07-02", 10.0, "high", 2, "2024-07-02", "test"))
    assert candidate["pattern"] is first
    session._record_instance_states([])
    session.bars.append(_bar("2024-07-02", 9.3))
    session.date_strings.append("2024-07-02")
    session.atr_values.append(0.2)
    session._evaluate(Pivot(3, "2024-07-02", 10.5, "high", 3, "2024-07-02", "test"))
    session._record_instance_states([])

    assert len(calls) == 2
    assert calls[1][0]["line"] == first.neckline
    latest = {event.instance_id: event.state for event in session.instance_ledger.latest_events}
    assert latest == {first.pattern_id: "forming", second.pattern_id: "forming"}


def test_boundary_discovered_after_its_first_breakout_cannot_backfill_a_buy():
    session = DailyRecognitionSession("000001", "stock")
    session.bars = [_bar("2024-01-01", 9.8), _bar("2024-06-28", 9.9),
                    _bar("2024-06-29", 11.0)]
    session.date_strings = [str(bar["date"].date()) for bar in session.bars]
    session.atr_values = [0.2, 0.2, 0.2]
    pattern = _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                             known="2024-06-28", support_lower=8.0)
    selection = next(item["value"] for item in pattern.evidence
                     if item["code"] == "neckline_selection")
    tests = next(item["value"] for item in pattern.evidence
                 if item["code"] == "neckline_tests")
    selection["consumed_boundary_phases"] = [{
        "line": dict(pattern.neckline), "tests": deepcopy(tests),
        "consumed_index": 2, "candidate_id": selection["selected_candidate_id"],
        "reason": "breakout_before_boundary_available", "selected": True}]
    pattern.evidence.append({"code": "boundary_available_at", "value": {
        "date": "2024-07-01", "index": 3,
        "reason": "paired_boundary_discovered_causally_on_this_prefix"}})
    _install_cached_candidate(session, pattern)
    candidate = next(iter(session.active.values()))
    session._remember_consumed_boundary(candidate,
                                        selection["consumed_boundary_phases"][0])

    # The line is discovered after the June 29 buffered breakout. Neither the
    # discovery bar nor a later recross may turn that earlier move into a buy.
    session.bars.append(_bar("2024-07-01", 10.1))
    session.date_strings.append("2024-07-01")
    session.atr_values.append(0.2)
    assert session._check_cached_breakout() == []
    session.bars.append(_bar("2024-07-02", 11.0))
    session.date_strings.append("2024-07-02")
    session.atr_values.append(0.2)

    assert session._check_cached_breakout() == []
    assert not session.instance_ledger.latest_events


def test_boundary_selector_does_not_serialize_candidate_patterns():
    pattern = _cached_bottom()
    pattern.to_dict = lambda: (_ for _ in ()).throw(AssertionError("unexpected full serialization"))
    session = DailyRecognitionSession("000001", "stock")

    selected = session._select_structure_boundaries([pattern])

    assert selected[pattern.structure_key] == {pattern.boundary_version_id}


def test_target_reached_event_uses_first_shape_target_without_creating_a_buy():
    session = DailyRecognitionSession("000001", "stock")
    bar = _bar("2024-07-01", 10.6)
    bar["high"] = 10.8
    session.bars = [bar]
    session.date_strings = ["2024-07-01"]
    pattern = Pattern(
        pattern_id="target-bottom", code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-01-01", confirmed_at=1, confirmation_date="2024-07-01",
        key_points={"pivot_1_date": "2024-01-01"},
        neckline={"slope": 0.0, "intercept": 10.0}, invalidation=9.5,
        targets=[10.5], evidence=[{"code": "neckline_breakout", "value": {"close": 10.2}}],
    ).to_dict()
    session.instance_ledger.append(pattern, "confirmed", "2024-07-01", entry_eligible=True)
    session.confirmed_instance_patterns["target-bottom"] = pattern
    session.instance_entry_signals["target-bottom"] = {"signal_date": "2024-07-01"}

    session._record_instance_states([])
    assert [event["state"] for event in session.instance_events()] == ["confirmed"]

    next_bar = _bar("2024-07-02", 10.6)
    next_bar["high"] = 10.8
    session.bars.append(next_bar)
    session.date_strings.append("2024-07-02")
    session._record_instance_states([])

    assert [(event["state"], event["entry_eligible"])
            for event in session.instance_events()] == [("confirmed", True), ("target_reached", False)]
    assert session.current_instances()[0].state == "target_reached"


def test_support_invalidation_precedes_target_on_the_same_bar():
    session = DailyRecognitionSession("000001", "stock")
    bar = _bar("2024-07-02", 9.4)
    bar["high"] = 10.8
    session.bars = [_bar("2024-07-01", 10.2), bar]
    session.date_strings = ["2024-07-01", "2024-07-02"]
    pattern = Pattern(
        pattern_id="invalidated-before-target", code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-01-01", confirmed_at=1, confirmation_date="2024-07-01",
        key_points={"pivot_1_date": "2024-01-01"},
        neckline={"slope": 0.0, "intercept": 10.0}, invalidation=9.5,
        targets=[10.5], evidence=[{"code": "neckline_breakout", "value": {"close": 10.2}}],
    ).to_dict()
    session.instance_ledger.append(pattern, "confirmed", "2024-07-01", entry_eligible=True)
    session.confirmed_instance_patterns["invalidated-before-target"] = pattern
    session.instance_entry_signals["invalidated-before-target"] = {"signal_date": "2024-07-01"}

    session._record_instance_states([])

    assert [event["state"] for event in session.instance_events()] == ["confirmed", "invalidated"]


def test_cached_breakout_is_not_eligible_on_structure_known_date(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-01-01", 9.5))
    session.append_bar(_bar("2024-06-30"))
    _install_cached_candidate(session, _cached_bottom(
        first_trough="2024-01-01", frozen="2024-06-29", known="2024-07-01"))

    same_day_as_structure_known = session.append_bar(_bar("2024-07-01", 11.0))

    assert same_day_as_structure_known == []


def test_cached_structure_expires_after_five_year_window_without_new_pivot(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2020-12-01", 9.5))
    session.append_bar(_bar("2025-12-01"))
    _install_cached_candidate(session, _cached_bottom(
        first_trough="2020-12-01", frozen="2025-12-01", known="2025-11-30"))

    aged_out_breakout = session.append_bar(_bar("2025-12-02", 11.0))

    assert aged_out_breakout == []
    assert not session.active


def test_bull_flag_does_not_emit_from_pivots_not_yet_right_confirmed(monkeypatch):
    session = DailyRecognitionSession("000001", "stock")
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    session.bars = [_bar(day, 10.0) for day in dates]
    session.atr_values = [0.2] * len(session.bars)
    centers = [(0, "low", 8.0), (4, "high", 12.0), (8, "low", 9.7),
               (12, "high", 11.3), (16, "low", 10.7)]
    session.local_pivots = [Pivot(index, str(dates[index].date()), price, kind,
                                  len(session.bars) + 1, "2025-01-22", "1.5ATR_radius2")
                            for index, kind, price in centers]
    monkeypatch.setattr(session, "_append_scale_pivots", lambda _index: [])
    monkeypatch.setattr(session, "_append_pressure_pivot", lambda _index: None)
    monkeypatch.setattr(session, "_append_local_pivot", lambda _index: False)

    emitted = session.append_bar(_bar("2025-01-21", 11.5))

    assert emitted == []
    assert not session.emitted_ids


def test_same_day_independent_main_patterns_are_merged_into_one_signal(monkeypatch):
    import caisen.daily_incremental as incremental

    session = DailyRecognitionSession("000001", "stock")
    _quiet_pivot_updates(session, monkeypatch)
    session.append_bar(_bar("2024-04-01"))
    current_date = "2025-04-02"
    first = Pattern(
        pattern_id="same-day-w", code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2025-01-01", confirmed_at=1, confirmation_date=current_date,
        key_points={"pivot_1_date": "2025-01-01", "pivot_scale": "2atr_5pct"},
        neckline={"slope": 0.01, "intercept": 10.5},
        evidence=[{"code": "main_structure", "value": {"role": "main", "signal_eligible": True}},
                 {"code": "neckline_selection", "value": {"selected_candidate_id": "w-line"}}],
        last_observed_date=current_date,
    )
    second = Pattern(
        pattern_id="same-day-triangle", code="000001", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed",
        formed_at="2025-01-10", confirmed_at=1, confirmation_date=current_date,
        key_points={"pivot_1_date": "2025-01-10", "pivot_scale": "2atr_5pct"},
        neckline={"slope": -0.01, "intercept": 9.5},
        evidence=[{"code": "main_structure", "value": {"role": "main", "signal_eligible": True}},
                 {"code": "neckline_selection", "value": {"selected_candidate_id": "triangle-line"}}],
        last_observed_date=current_date,
    )
    for pattern, candidate_id, anchor_start, anchor_end in (
            (first, "w-line", "2024-10-01", "2025-01-10"),
            (second, "triangle-line", "2025-02-01", "2025-02-15")):
        pattern.key_points.update({"pivot_1_date": anchor_start, "pivot_1_kind": "low",
                                   "pivot_1": 8.5, "pivot_2_date": anchor_end,
                                   "pivot_2_kind": "low", "pivot_2": 8.6})
        tests = [{"pair_id": candidate_id, "group_id": f"{candidate_id}-{index}",
                  "pivot_date": date, "level": pattern.neckline["intercept"],
                  "valid": True, "normalized_residual": 0.1}
                 for index, date in enumerate((anchor_start, anchor_end))]
        selection = {"selected_candidate_id": candidate_id, "selected_line": pattern.neckline,
                     "line_frozen_at": "2025-03-01", "structure_known_at": "2025-03-02",
                     "effective_start_date": anchor_start}
        pattern.evidence.extend([{"code": "neckline_tests", "value": tests},
                                 {"code": "neckline_selection", "value": selection}])
        identity = pattern_identity_metadata(pattern.to_dict())
        pattern.pattern_id = identity["instance_id"]
        pattern.structure_key = identity["structure_key"]
        pattern.boundary_version_id = identity["boundary_version_id"]
        assert pattern_instance_is_frozen(pattern.to_dict())
    candidates = {
        ("triangle_bottom", "2atr_5pct", ()): {"family": "triangle_bottom", "scale": "2atr_5pct",
            "sequence": [Pivot(0, "2025-04-01", 9.5, "low", 1, current_date, "2ATR|5%")],
            "structure_known_at": 1, "pattern": None, "needs_initial_eval": True},
        ("w_bottom", "2atr_5pct", ()): {"family": "w_bottom", "scale": "2atr_5pct",
            "sequence": [Pivot(0, "2025-04-01", 9.5, "low", 1, current_date, "2ATR|5%")],
            "structure_known_at": 1, "pattern": None, "needs_initial_eval": True},
    }
    monkeypatch.setattr(session, "_append_scale_pivots", lambda _index: [("2atr_5pct", Pivot(
        0, "2025-04-01", 10.0, "low", 1, current_date, "2ATR|5%"))])
    monkeypatch.setattr(session, "_refresh_structures", lambda: session.active.update(candidates))
    monkeypatch.setattr(incremental, "_make_main_pattern",
                        lambda _frame, _code, _kind, _tf, family, *_args, **_kwargs:
                        [first if family == "w_bottom" else second])

    emitted = session.append_bar(_bar(current_date))

    assert len(emitted) == 1
    assert set(emitted[0]["pattern_ids"]) == {first.pattern_id, second.pattern_id}
    assert set(emitted[0]["pattern_types"]) == {"w_bottom", "triangle_bottom"}
    assert emitted[0]["pattern"]["pattern_type"] == "w_bottom"
    selections = [next(item["value"] for item in pattern["evidence"]
                       if item["code"] == "main_selection") for pattern in emitted[0]["patterns"]]
    assert all(item["primary_pattern_id"] == emitted[0]["pattern"]["pattern_id"]
               for item in selections)
    assert sum(item["selected"] for item in selections) == 1
    grouped = {pattern["pattern_type"]: pattern for pattern in emitted[0]["patterns"]}
    assert grouped["w_bottom"]["neckline"]["intercept"] == 10.5
    assert grouped["triangle_bottom"]["neckline"]["intercept"] == 9.5
    line_ids = {pattern["pattern_type"]: next(item["value"]["selected_candidate_id"]
                   for item in pattern["evidence"] if item["code"] == "neckline_selection")
                for pattern in emitted[0]["patterns"]}
    assert line_ids == {"w_bottom": "w-line", "triangle_bottom": "triangle-line"}


@pytest.mark.parametrize("child_scale", ["3atr_8pct", "5atr_12pct"])
def test_strictly_enclosing_w_suppresses_an_inner_triangle_at_same_or_smaller_scale(child_scale):
    parent = Pattern(
        pattern_id="main-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-02-06", confirmed_at=1, confirmation_date="2026-01-09",
        key_points={"pivot_1_date": "2025-03-31", "pivot_1_kind": "low",
                    "pivot_2_date": "2025-12-17", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[
            {"code": "main_structure", "value": {
                "role": "main", "family": "w_bottom", "pivot_scale": "5atr_12pct",
                "first_trough_date": "2025-03-31", "structure_end_date": "2025-12-30"}},
            {"code": "neckline_selection", "value": {"structure_known_at": "2025-12-30"}},
        ],
        last_observed_date="2026-01-09",
    )
    child = Pattern(
        pattern_id="inner-triangle", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed",
        formed_at="2025-04-30", confirmed_at=2, confirmation_date="2026-01-09",
        key_points={"pivot_1_date": "2025-04-30", "pivot_1_kind": "low",
                    "pivot_2_date": "2025-04-09", "pivot_2_kind": "high",
                    "pivot_3_date": "2025-06-15", "pivot_3_kind": "low",
                    "pivot_4_date": "2025-11-17", "pivot_4_kind": "high",
                    "pivot_5_date": "2025-11-20", "pivot_5_kind": "low",
                    "pivot_scale": child_scale},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "triangle_bottom", "pivot_scale": child_scale,
            "first_trough_date": "2025-04-30", "structure_end_date": "2025-11-20"}}],
        last_observed_date="2026-01-09",
    )

    assert DailyRecognitionSession._nested_ids([parent, child], "2026-01-09") == {"inner-triangle"}


def test_internal_or_unselected_structure_cannot_suppress_another_buy_candidate():
    parent = Pattern(
        pattern_id="internal-parent", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-01-01", confirmed_at=1, confirmation_date="2025-01-01",
        key_points={"pivot_1_date": "2024-01-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2025-01-01", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[
            {"code": "main_structure", "value": {
                "role": "internal", "signal_eligible": False,
                "pivot_scale": "5atr_12pct"}},
            {"code": "main_selection", "value": {"selected": False}},
        ])
    child = Pattern(
        pattern_id="valid-child", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed",
        formed_at="2024-03-01", confirmed_at=2, confirmation_date="2024-12-01",
        key_points={"pivot_1_date": "2024-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-06-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2024-09-01", "pivot_3_kind": "low",
                    "pivot_4_date": "2024-12-01", "pivot_4_kind": "high",
                    "pivot_scale": "2atr_5pct"},
        evidence=[
            {"code": "main_structure", "value": {
                "role": "main", "signal_eligible": True,
                "pivot_scale": "2atr_5pct"}},
            {"code": "main_selection", "value": {"selected": True}},
        ])

    assert DailyRecognitionSession._nested_ids([parent, child], "2025-02-01") == set()


def test_forming_outer_main_structure_suppresses_inner_candidate_before_its_own_breakout():
    parent = Pattern(
        pattern_id="forming-outer-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming", confirmed_at=None,
        formed_at="2024-01-01", last_observed_date="2025-01-01",
        key_points={"pivot_1_date": "2024-01-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2025-01-01", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[
            {"code": "main_structure", "value": {
                "role": "main", "signal_eligible": False, "pivot_scale": "5atr_12pct"}},
            {"code": "main_selection", "value": {"selected": True}},
            {"code": "neckline_selection", "value": {"structure_known_at": "2025-01-01"}},
        ])
    child = Pattern(
        pattern_id="confirmed-inner-triangle", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed", confirmed_at=240,
        formed_at="2024-03-01", last_observed_date="2024-12-01",
        confirmation_date="2024-12-01",
        key_points={"pivot_1_date": "2024-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-06-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2024-09-01", "pivot_3_kind": "low",
                    "pivot_4_date": "2024-11-01", "pivot_4_kind": "high",
                    "pivot_scale": "2atr_5pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True, "pivot_scale": "2atr_5pct"}}])

    assert DailyRecognitionSession._nested_ids([parent, child], "2025-01-01") == {
        "confirmed-inner-triangle"}


def test_candidate_after_last_anchor_is_not_suppressed_by_observation_end():
    parent = Pattern(
        pattern_id="outer-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming", confirmed_at=None,
        formed_at="2024-01-01", last_observed_date="2025-01-01",
        key_points={"pivot_1_date": "2024-01-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-07-01", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": False, "pivot_scale": "5atr_12pct",
            "first_trough_date": "2024-01-01", "structure_end_date": "2025-01-01"}},
                  {"code": "main_selection", "value": {"selected": True}},
                  {"code": "neckline_selection", "value": {
                      "structure_known_at": "2024-07-01"}}])
    child = Pattern(
        pattern_id="post-anchor-inner-triangle", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed", confirmed_at=240,
        formed_at="2024-09-01", confirmation_date="2024-12-01",
        key_points={"pivot_1_date": "2024-09-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-10-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2024-11-01", "pivot_3_kind": "low",
                    "pivot_4_date": "2024-12-01", "pivot_4_kind": "high",
                    "pivot_scale": "2atr_5pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True, "pivot_scale": "2atr_5pct"}}])

    assert DailyRecognitionSession._nested_ids([parent, child], "2025-01-01") == set()


def test_terminal_major_structure_releases_its_nested_suppression():
    parent = Pattern(
        pattern_id="finished-parent", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="target_reached",
        formed_at="2024-01-01", confirmed_at=1, confirmation_date="2024-10-01",
        key_points={"pivot_1_date": "2024-01-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2025-01-01", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True,
            "pivot_scale": "5atr_12pct"}}])
    child = Pattern(
        pattern_id="new-child", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed",
        formed_at="2024-03-01", confirmed_at=2, confirmation_date="2024-12-01",
        key_points={"pivot_1_date": "2024-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-06-01", "pivot_2_kind": "high",
                    "pivot_3_date": "2024-09-01", "pivot_3_kind": "low",
                    "pivot_4_date": "2024-12-01", "pivot_4_kind": "high",
                    "pivot_scale": "2atr_5pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True,
            "pivot_scale": "2atr_5pct"}}])

    assert DailyRecognitionSession._nested_ids([parent, child], "2025-02-01") == set()


def test_outer_w_with_only_one_shared_trough_does_not_suppress_distinct_child_w():
    outer = Pattern(
        pattern_id="outer-2021-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at="2021-02-04", confirmed_at=None,
        key_points={"pivot_1_date": "2021-02-04", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-09-18", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom", "pivot_scale": "5atr_12pct",
            "first_trough_date": "2021-02-04", "structure_end_date": "2024-09-26"}},
            {"code": "neckline_selection", "value": {"structure_known_at": "2024-09-26"}}],
        last_observed_date="2026-01-09")
    inner = Pattern(
        pattern_id="inner-2024-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at="2024-02-06", confirmed_at=None,
        key_points={"pivot_1_date": "2024-02-06", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-09-18", "pivot_2_kind": "low",
                    "pivot_scale": "2atr_5pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "w_bottom", "pivot_scale": "2atr_5pct",
            "first_trough_date": "2024-02-06", "structure_end_date": "2024-09-19"}},
            {"code": "neckline_selection", "value": {"structure_known_at": "2024-09-19"}}],
        last_observed_date="2026-01-09")

    assert DailyRecognitionSession._nested_ids([outer, inner], "2026-01-09") == set()


def test_same_day_distinct_w_instances_remain_in_one_buy_signal():
    from caisen.pattern_instances import pattern_identity_metadata

    def w_candidate(start, end):
        line = {"slope": 0.0, "intercept": 10.0}
        points = {"pivot_1_date": start, "pivot_1_kind": "low", "pivot_1": 8.0,
                  "pivot_2_date": end, "pivot_2_kind": "low", "pivot_2": 8.1,
                  "pivot_scale": "2atr_5pct"}
        evidence = [
            {"code": "main_structure", "value": {"role": "main", "support_lower": 8.0}},
            {"code": "neckline_selection", "value": {"selected_candidate_id": f"line-{start}",
                "selected_line": line, "line_frozen_at": "2025-02-01",
                "structure_known_at": "2025-02-02"}},
            {"code": "neckline_tests", "value": [
                {"pair_id": f"line-{start}", "group_id": f"{start}-{i}",
                 "pivot_date": date, "level": 10.0, "valid": True,
                 "test_span_bars": 20, "normalized_residual": 0.1}
                for i, date in enumerate((start, end))]},
        ]
        identity = pattern_identity_metadata({"code": "000001", "kind": "stock",
            "timeframe": "1d", "pattern_type": "w_bottom", "key_points": points,
            "neckline": line, "evidence": evidence})
        return Pattern(pattern_id=identity["instance_id"], code="000001", kind="stock",
            timeframe="1d", pattern_type="w_bottom", direction="bullish", state="confirmed",
            formed_at=start, confirmed_at=1, confirmation_date="2025-06-01",
            key_points=points, neckline=line, evidence=evidence,
            structure_key=identity["structure_key"],
            boundary_version_id=identity["boundary_version_id"])

    older = w_candidate("2024-01-01", "2024-02-01")
    newer = w_candidate("2024-03-01", "2024-04-01")
    session = DailyRecognitionSession("000001", "stock")

    selected = session._independent_candidates([], [newer, older], "2025-06-01")
    signal = session._signal_many(selected, "2025-06-01")

    assert [item.pattern_id for item in selected] == [older.pattern_id, newer.pattern_id]
    assert set(signal["pattern_ids"]) == {older.pattern_id, newer.pattern_id}
