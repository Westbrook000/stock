from caisen.structure_boundary import FrozenBoundary, JointPatternInstance, advance_instance


def test_frozen_instance_uses_own_line_and_cannot_revive():
    line = FrozenBoundary('b1', slope=-0.1, intercept=10.0)
    instance = JointPatternInstance('p1', line, ready_index=2, maturity_index=3)
    assert advance_instance(instance, 2, 12, 1, .01) == []
    assert [e.kind for e in advance_instance(instance, 3, 11, 1, .01)] == ['confirmed']
    assert [e.kind for e in advance_instance(instance, 4, 9.5, 1, .01)] == ['invalidated']
    assert advance_instance(instance, 5, 12, 1, .01) == []
    assert instance.state == 'invalidated'


def test_early_breakout_consumes_boundary():
    instance = JointPatternInstance('p', FrozenBoundary('b', 0, 10), ready_index=1, maturity_index=5)
    assert [e.kind for e in advance_instance(instance, 2, 12, 1, .01)] == ['rejected_too_early']
    assert advance_instance(instance, 5, 12, 1, .01) == []
    assert instance.state == 'expired'


def test_threshold_is_strict_and_events_cannot_be_replayed():
    import pytest
    instance = JointPatternInstance('p', FrozenBoundary('b', 0, 10), ready_index=0, maturity_index=0)
    assert [event.kind for event in advance_instance(instance, 1, 11, 1, .01)] == ['weak_crossing']
    assert instance.state == 'waiting_breakout'
    assert [event.kind for event in advance_instance(instance, 2, 11.01, 1, .01)] == ['confirmed']
    with pytest.raises(ValueError):
        advance_instance(instance, 2, 9, 1, .01)
    assert [event.kind for event in instance.events] == ['weak_crossing', 'confirmed']


def test_weak_crossing_records_event_without_confirmation():
    instance = JointPatternInstance('p', FrozenBoundary('b', 0, 10), ready_index=0, maturity_index=0)
    assert [e.kind for e in advance_instance(instance, 1, 10.5, 1, .01)] == ['weak_crossing']
    assert instance.state == 'waiting_breakout'
    assert advance_instance(instance, 2, 10.6, 1, .01) == []
    assert [e.kind for e in advance_instance(instance, 3, 11.1, 1, .01)] == ['confirmed']


def test_confirmed_instance_records_target_and_terminal_state():
    instance = JointPatternInstance('p', FrozenBoundary('b', 0, 10), ready_index=0, maturity_index=0, target=15)
    advance_instance(instance, 1, 12, 1, .01)
    assert [e.kind for e in advance_instance(instance, 2, 15, 1, .01)] == ['target_reached']
    assert instance.state == 'target_reached'
    assert advance_instance(instance, 3, 8, 1, .01) == []


def test_boundary_selection_uses_confirmed_touches_and_full_high_coverage():
    from caisen.structure_boundary import PressureTouch, select_boundary
    touches = [PressureTouch(0, 1, 10), PressureTouch(3, 4, 10), PressureTouch(6, 7, 10)]
    highs = [10, 9, 9, 10, 9, 9, 10, 9]
    line = select_boundary(touches, highs, [1] * len(highs), observed_index=7, tick_size=.01)
    assert line is not None
    assert line.test_indices == (0, 3, 6)
    assert line.frozen_at == 7
    highs[5] = 11
    assert select_boundary(touches, highs, [1] * len(highs), observed_index=7, tick_size=.01) is None


def test_future_touch_is_not_used_and_extension_not_counted_as_span():
    from caisen.structure_boundary import PressureTouch, select_boundary
    touches = [PressureTouch(0, 1, 10), PressureTouch(3, 8, 10), PressureTouch(6, 9, 10)]
    highs = [10, 9, 9, 10, 9, 9, 10, 9, 9, 9]
    assert select_boundary(touches, highs, [1] * 10, observed_index=7, tick_size=.01) is None
    line = select_boundary(touches, highs, [1] * 10, observed_index=9, tick_size=.01)
    assert line.test_span == 6


def test_bidirectional_search_uses_observation_window_not_first_trough():
    import pandas as pd
    from caisen.structure_boundary import pressure_search_start
    dates = pd.bdate_range('2010-01-01', '2026-10-30')
    i = pressure_search_start(dates, '1d')
    assert dates[i] >= pd.Timestamp('2021-10-30')
    assert dates[i-1] < pd.Timestamp('2021-10-30')
    w = pressure_search_start(dates, '1w')
    assert dates[w] >= pd.Timestamp('2016-10-30')
    assert dates[w-1] < pd.Timestamp('2016-10-30')


def test_formation_starts_at_first_independent_pressure_test():
    from caisen.structure_boundary import formation_start_index
    assert formation_start_index(50, [{'coverage_start_index': 100}]) == 100
    assert formation_start_index(150, [{'coverage_start_index': 100}]) == 100


def test_nearby_distinct_lines_are_not_same_boundary():
    from caisen.main_bottoms import _same_neckline_boundary_phase
    tests = [{'pivot_index': 10, 'coverage_start_index': 10, 'coverage_end_index': 20, 'tolerance': .5}, {'pivot_index': 20, 'coverage_start_index': 10, 'coverage_end_index': 20, 'tolerance': .5}]
    old = {'line': {'slope': 0, 'intercept': 10}, 'tests': tests}
    shifted = {'line': {'slope': 0, 'intercept': 10.1}, 'tests': tests}
    numeric = {'line': {'slope': 0, 'intercept': 10.001}, 'tests': tests}
    assert not _same_neckline_boundary_phase(old, shifted, .01)
    assert _same_neckline_boundary_phase(old, numeric, .01)


def test_failed_boundary_does_not_reject_distinct_price_zone():
    from caisen.main_bottoms import _boundary_was_rejected
    tests = [{'pivot_index': 10}, {'pivot_index': 20}]
    old = {'line': {'slope': 0, 'intercept': 10}, 'tests': tests}
    other = {'line': {'slope': 0, 'intercept': 10.1}, 'tests': tests}
    assert _boundary_was_rejected(old, [old], .01)
    assert not _boundary_was_rejected(other, [old], .01)
    assert not _boundary_was_rejected(other, [{'index': 15, 'price': 10, 'tolerance': .5}], .01)


def test_first_crossing_is_causal_and_repeated_breakout_is_not_first():
    from caisen.structure_boundary import first_buffered_crossing
    closes = [9, 9, 12, 9, 12]
    atr = [1, 1, 1, 1, 1]
    line = FrozenBoundary('b', 0, 10)
    cache = {}
    assert first_buffered_crossing(closes, atr, line, 1, 4, .01, cache) == 2
    assert first_buffered_crossing(closes, atr, line, 1, 1, .01, cache) is None
    assert first_buffered_crossing(closes, atr, line, 3, 4, .01, cache) == 2


def test_joint_recognition_uses_new_cache_engine_versions():
    from caisen.daily_incremental import ENGINE_VERSION, BOUNDARY_FIRST_ENGINE_VERSION
    assert ENGINE_VERSION != 'caisen-daily-continuous-v5'
    assert BOUNDARY_FIRST_ENGINE_VERSION != 'caisen-daily-boundary-first-v3'
    assert ENGINE_VERSION == 'caisen-joint-pattern-session-v9-flag-confirmation-history'


def test_line_already_broken_at_freeze_cannot_be_rearmed():
    from caisen.structure_boundary import first_buffered_crossing
    line = FrozenBoundary('b', 0, 10)
    assert first_buffered_crossing([9, 12, 9, 12], [1, 1, 1, 1], line, 2, 3, .01, {}) == 1


def test_short_prefix_before_freeze_does_not_poison_first_crossing_cache():
    from caisen.structure_boundary import first_buffered_crossing
    line = FrozenBoundary('b', 0, 10)
    cache = {}
    closes, atr = [9, 12, 9, 12], [1, 1, 1, 1]
    assert first_buffered_crossing(closes, atr, line, 2, 0, .01, cache) is None
    assert first_buffered_crossing(closes, atr, line, 2, 3, .01, cache) == 1


def test_first_crossing_cache_advances_only_new_bars_and_is_rule_scoped():
    from caisen.structure_boundary import first_buffered_crossing
    line = FrozenBoundary('b', 0, 10)
    cache = {}
    closes, atr = [9, 9], [1, 1]
    assert first_buffered_crossing(closes, atr, line, 1, 1, .01, cache) is None
    assert first_buffered_crossing(closes, atr, line, 1, 1, .01, cache) is None
    closes.append(12)
    atr.append(1)
    assert first_buffered_crossing(closes, atr, line, 1, 2, .01, cache) == 2
    assert first_buffered_crossing(closes, atr, line, 1, 2, .01, cache,
                                   buffer_multiple=2.0) is None
    assert len(cache) == 2


def test_new_line_after_failure_needs_two_new_confirmed_tests():
    from caisen.structure_boundary import has_new_tests_after_failure
    tests = [{'pivot_index': 2, 'known_index': 3}, {'pivot_index': 11, 'known_index': 12}, {'pivot_index': 14, 'known_index': 16}]
    assert not has_new_tests_after_failure(tests, failure_index=10, observed_index=15)
    assert has_new_tests_after_failure(tests, failure_index=10, observed_index=16)
    assert not has_new_tests_after_failure([{'pivot_index': 9, 'known_index': 12}, tests[-1]], 10, 16)


def test_joint_instance_identity_binds_structure_to_frozen_boundary_and_event_history():
    from copy import deepcopy
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    pattern = Pattern("p1", "600108", "stock", "1d", "w_bottom", "bullish", "invalidated",
                      "2024-02-06", 100,
                      key_points={"pivot_1_date": "2024-02-06", "pivot_1": 2.22,
                                  "pivot_1_kind": "low", "pivot_2_date": "2024-03-15",
                                  "pivot_2": 2.87, "pivot_2_kind": "high",
                                  "pivot_3_date": "2024-09-18", "pivot_3": 2.25,
                                  "pivot_3_kind": "low"},
                      neckline={"slope": 0.0, "intercept": 2.87},
                      confirmation_date="2024-10-08", last_observed_date="2025-01-06",
                      evidence=[{"code": "main_structure", "value": {
                          "role": "main", "signal_eligible": False,
                          "structure_known_at": "2024-09-25", "structure_end_date": "2025-01-06"}},
                                {"code": "neckline_selection", "value": {
                                    "line_frozen_at": "2024-09-26", "effective_start_date": "2024-03-15",
                                    "selected_line": {"slope": 0.0, "intercept": 2.87}}},
                                {"code": "neckline_tests", "value": [
                                    {"group_id": "a", "pivot_date": "2024-03-15", "known_date": "2024-03-20",
                                     "level": 2.87},
                                    {"group_id": "b", "pivot_date": "2024-09-18", "known_date": "2024-09-25",
                                     "level": 2.86}]},
                                {"code": "neckline_breakout", "value": {
                                    "date": "2024-10-08", "close": 2.90, "frozen_neckline": 2.87}},
                                {"code": "neckline_invalidation", "value": {
                                    "date": "2025-01-06", "level": 2.87}}])

    attach_joint_lifecycle(pattern, discovered_at="2024-09-26")
    lifecycle = next(item["value"] for item in pattern.evidence
                     if item["code"] == "joint_lifecycle")
    assert lifecycle["paired_at"] == "2024-09-26"
    assert lifecycle["events"][-1]["kind"] == "invalidated"
    assert lifecycle["events"][-2]["kind"] == "confirmed"
    original_instance_id = lifecycle["instance_id"]

    changed_boundary = deepcopy(pattern)
    changed_boundary.neckline = {"slope": 0.0, "intercept": 3.0}
    attach_joint_lifecycle(changed_boundary)
    changed = next(item["value"] for item in changed_boundary.evidence
                   if item["code"] == "joint_lifecycle")
    assert lifecycle["structure_key"] == changed["structure_key"]
    assert original_instance_id != changed["instance_id"]


def test_lifecycle_structure_known_at_falls_back_to_selected_neckline_evidence():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    pattern = Pattern("fallback-known", "600108", "stock", "1d", "w_bottom", "bullish", "forming",
                      "2024-02-06", None,
                      key_points={"pivot_1_date": "2024-02-06", "pivot_1": 2.2,
                                  "pivot_1_kind": "low", "pivot_2_date": "2024-09-18",
                                  "pivot_2": 2.25, "pivot_2_kind": "low"},
                      neckline={"slope": 0.0, "intercept": 2.87},
                      evidence=[{"code": "main_structure", "value": {"role": "main"}},
                                {"code": "neckline_selection", "value": {
                                    "structure_known_at": "2024-09-20",
                                    "selected_line": {"slope": 0.0, "intercept": 2.87}}}])

    lifecycle = attach_joint_lifecycle(pattern, discovered_at="2024-09-20")

    assert lifecycle["structure_known_at"] == "2024-09-20"


def test_lifecycle_expires_instance_after_a_consumed_early_breakout():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    pattern = Pattern(
        "early-w", "000008", "stock", "1d", "w_bottom", "bullish", "forming",
        "2024-10-18", None,
        key_points={"pivot_1_date": "2024-10-18", "pivot_1": 2.2, "pivot_1_kind": "low",
                    "pivot_2_date": "2024-11-27", "pivot_2": 2.36, "pivot_2_kind": "low"},
        neckline={"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
        evidence=[
            {"code": "main_structure", "value": {"role": "main", "signal_eligible": False,
                "early_breakout_consumed": True}},
            {"code": "neckline_selection", "value": {
                "line_frozen_at": "2024-12-02", "structure_known_at": "2024-12-02",
                "selected_candidate_id": "neckline-candidate-old",
                "independent_tests": [{"pivot_date": "2024-03-14", "group_id": "old-a", "level": 2.95},
                                      {"pivot_date": "2024-11-07", "group_id": "old-b", "level": 2.92}],
                "selected_line": {"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
                "early_breakouts": [{"date": "2024-12-03", "close": 3.06,
                    "frozen_neckline": 2.91656050955414,
                    "required_close": 3.036994212307247,
                    "candidate_id": "neckline-candidate-old",
                    "line": {"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
                    "tests": [{"pivot_date": "2024-03-14", "group_id": "old-a", "level": 2.95},
                              {"pivot_date": "2024-11-07", "group_id": "old-b", "level": 2.92}],
                    "reason": "breakout_before_minimum_structure_duration"}]}}],
    )

    lifecycle = attach_joint_lifecycle(pattern)

    assert pattern.state == "expired"
    assert lifecycle["state_at"] == "expired"
    assert lifecycle["entry_eligible"] is False
    assert lifecycle["events"][-1]["kind"] == "rejected_too_early"


def test_delayed_materialization_records_pair_and_early_cross_at_discovery_date():
    from copy import deepcopy
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    pattern = Pattern(
        "delayed-w", "000008", "stock", "1d", "w_bottom", "bullish", "forming",
        "2024-10-18", None,
        key_points={"pivot_1_date": "2024-10-18", "pivot_1": 2.2, "pivot_1_kind": "low",
                    "pivot_2_date": "2024-11-27", "pivot_2": 2.36, "pivot_2_kind": "low"},
        neckline={"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
        evidence=[
            {"code": "main_structure", "value": {"role": "main", "signal_eligible": False,
                "maturity_date": "2025-04-18", "structure_known_at": "2024-12-02"}},
            {"code": "neckline_selection", "value": {
                "line_frozen_at": "2024-11-14", "freeze_date": "2024-12-02",
                "selected_candidate_id": "neckline-candidate-7357-7514",
                "selected_line": {"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
                "independent_tests": [{"pivot_date": "2024-03-14", "group_id": "old-a", "level": 2.95},
                                      {"pivot_date": "2024-11-07", "group_id": "old-b", "level": 2.92}],
                "early_breakouts": [{"date": "2024-12-03", "index": 7532,
                    "close": 3.06, "frozen_neckline": 2.91656050955414,
                    "required_close": 3.036994212307247,
                    "candidate_id": "neckline-candidate-7357-7514",
                    "line": {"slope": -0.0001910828025477723,
                             "intercept": 4.355796178343961},
                    "tests": [{"pivot_date": "2024-03-14", "group_id": "old-a", "level": 2.95},
                              {"pivot_date": "2024-11-07", "group_id": "old-b", "level": 2.92}]}]}}],
    )

    lifecycle = attach_joint_lifecycle(pattern, discovered_at="2025-04-18")
    pair = next(event for event in lifecycle["events"] if event["kind"] == "paired")
    rejected = next(event for event in lifecycle["events"]
                    if event["kind"] == "rejected_too_early")
    assert lifecycle["paired_at"] == "2025-04-18"
    assert pair["date"] == pair["recorded_at"] == "2025-04-18"
    assert rejected["date"] == rejected["recorded_at"] == "2025-04-18"
    assert rejected["occurred_at"] == "2024-12-03"
    assert pattern.state == "expired"

    original_events = deepcopy(lifecycle["events"])
    refreshed = attach_joint_lifecycle(pattern, discovered_at="2025-04-21")
    assert refreshed["paired_at"] == "2025-04-18"
    assert refreshed["events"][:len(original_events)] == original_events


def test_early_breakout_from_old_line_does_not_expire_a_new_boundary_version():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    pattern = Pattern(
        "new-line-w", "000008", "stock", "1d", "w_bottom", "bullish", "forming",
        "2024-10-18", None,
        key_points={"pivot_1_date": "2024-10-18", "pivot_1": 2.2, "pivot_1_kind": "low",
                    "pivot_2_date": "2024-11-27", "pivot_2": 2.36, "pivot_2_kind": "low"},
        neckline={"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
        evidence=[
            {"code": "main_structure", "value": {"role": "main", "signal_eligible": False}},
            {"code": "neckline_tests", "value": [
                {"pivot_date": "2025-03-03", "group_id": "new-a", "level": 3.15},
                {"pivot_date": "2025-05-06", "group_id": "new-b", "level": 3.10}]},
            {"code": "neckline_selection", "value": {
                "selected_candidate_id": "neckline-candidate-new",
                "selected_line": {"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
                "independent_tests": [
                    {"pivot_date": "2025-03-03", "group_id": "new-a", "level": 3.15},
                    {"pivot_date": "2025-05-06", "group_id": "new-b", "level": 3.10}],
                "early_breakouts": [{"date": "2024-12-03", "candidate_id": "neckline-candidate-old",
                    "line": {"slope": -0.0001910828025477723, "intercept": 4.355796178343961},
                    "tests": [{"pivot_date": "2024-03-14", "group_id": "old-a", "level": 2.95},
                              {"pivot_date": "2024-11-07", "group_id": "old-b", "level": 2.92}]}]}}],
    )

    lifecycle = attach_joint_lifecycle(pattern)

    assert pattern.state == "forming"
    assert lifecycle["state_at"] == "forming"
    assert lifecycle["boundary"]["intercept"] == 4.355796178343961
    assert all(event["kind"] != "rejected_too_early" for event in lifecycle["events"])


def test_parent_suppressed_flag_keeps_historical_confirmation_eligibility():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "nested-flag", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        rule_version="caisen-joint-pattern-session-v6",
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                  {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                    "frozen_boundary": 12.0, "required_close": 12.1}},
                  {"code": "flag_lifecycle", "value": {
                    "entry_eligible_at_confirmation": True,
                    "entry_eligibility_at_confirmation_status": "known"}},
                  {"code": "recognition_context", "value": {"recognition_experiment": None,
                    "engine_version": "caisen-joint-pattern-session-v7"}}],
    )

    first = attach_joint_lifecycle(flag)
    flag.parent_id = "main-parent"
    flag.evidence.append({"code": "execution_suppression", "value": {"suppressed": True,
                         "parent_id": "main-parent", "suppressed_at": "1994-03-20"}})
    lifecycle = attach_joint_lifecycle(flag)

    assert first["entry_eligible_at_confirmation"] is True
    assert lifecycle["entry_eligible_at_confirmation"] is True
    assert lifecycle["entry_eligible"] is False


def test_flag_suppressed_at_confirmation_records_historical_ineligibility():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "suppressed-flag", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        rule_version="caisen-joint-pattern-session-v6",
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                  {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                    "frozen_boundary": 12.0, "required_close": 12.1}},
                  {"code": "flag_lifecycle", "value": {
                    "entry_eligible_at_confirmation": True,
                    "entry_eligibility_at_confirmation_status": "known"}},
                  {"code": "execution_suppression", "value": {"suppressed": True,
                    "parent_id": "main-parent", "suppressed_at": "1994-03-14"}},
                  {"code": "recognition_context", "value": {"recognition_experiment": None,
                    "engine_version": "caisen-joint-pattern-session-v6"}}],
        parent_id="main-parent",
    )

    first = attach_joint_lifecycle(flag)
    lifecycle = attach_joint_lifecycle(flag)

    assert first["entry_eligible_at_confirmation"] is False
    assert lifecycle["entry_eligible_at_confirmation"] is False
    assert lifecycle["entry_eligible"] is False


def test_invalidated_flag_preserves_recorded_confirmation_eligibility():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "later-invalidated-flag", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        rule_version="caisen-joint-pattern-session-v6",
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                  {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                    "frozen_boundary": 12.0, "required_close": 12.1}},
                  {"code": "flag_lifecycle", "value": {
                    "entry_eligible_at_confirmation": True,
                    "entry_eligibility_at_confirmation_status": "known"}},
                  {"code": "recognition_context", "value": {"recognition_experiment": None,
                    "engine_version": "caisen-joint-pattern-session-v6"}}],
    )
    first = attach_joint_lifecycle(flag)
    flag.state = "invalidated"

    lifecycle = attach_joint_lifecycle(flag)

    assert first["entry_eligible_at_confirmation"] is True
    assert lifecycle["entry_eligible_at_confirmation"] is True
    assert lifecycle["entry_eligible"] is False


def test_flag_breakout_is_not_eligible_when_geometry_was_not_yet_known():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "unknown-geometry-flag", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        rule_version="caisen-joint-pattern-session-v6",
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-17"}},
                  {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                    "frozen_boundary": 12.0, "required_close": 12.1}},
                  {"code": "recognition_context", "value": {"recognition_experiment": None,
                    "engine_version": "caisen-joint-pattern-session-v6"}}],
    )

    lifecycle = attach_joint_lifecycle(flag)

    assert lifecycle["entry_eligible_at_confirmation"] is False
    assert lifecycle["entry_eligible"] is False


def test_legacy_flag_without_confirmation_provenance_is_unknown_not_inferred():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "legacy-flag-no-provenance", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        evidence=[{"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                  "frozen_boundary": 12.0, "required_close": 12.1}}],
    )

    lifecycle = attach_joint_lifecycle(flag)

    assert lifecycle["entry_eligible_at_confirmation"] is None
    assert lifecycle["entry_eligibility_at_confirmation_status"] == "unknown_missing_confirmation_evidence"


def test_legacy_v2_flag_lifecycle_true_is_not_trusted_as_historical_qualification():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "legacy-v2-flag", "600684", "stock", "1d", "bull_flag", "bullish", "invalidated",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        rule_version="caisen-v17",
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                  {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                    "frozen_boundary": 12.0, "required_close": 12.1}},
                  {"code": "joint_lifecycle", "value": {"schema_version": "joint-pattern-instance-v2",
                    "entry_eligible_at_confirmation": True}}],
    )

    lifecycle = attach_joint_lifecycle(flag)

    assert lifecycle["entry_eligible_at_confirmation"] is None
    assert lifecycle["entry_eligibility_at_confirmation_status"] == "unknown_legacy_lifecycle_schema"


def test_pattern_geometry_identity_is_stable_across_data_versions():
    from copy import deepcopy
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    def pattern(data_version):
        return Pattern(
            "same-shape", "600684", "stock", "1d", "w_bottom", "bullish", "forming",
            "1994-01-03", None, data_version=data_version,
            key_points={"pivot_1_date": "1994-01-03", "pivot_1": 10.0, "pivot_1_kind": "low",
                        "pivot_2_date": "1994-02-01", "pivot_2": 10.1, "pivot_2_kind": "low"},
            neckline={"slope": 0.0, "intercept": 12.0},
        )

    first = pattern("freeze-A")
    second = deepcopy(first)
    second.data_version = "freeze-B"

    first_id = attach_joint_lifecycle(first)["instance_id"]
    second_id = attach_joint_lifecycle(second)["instance_id"]

    assert first_id == second_id
    first_context = next(item["value"] for item in first.evidence
                         if item["code"] == "joint_lifecycle")["recognition_context"]
    second_context = next(item["value"] for item in second.evidence
                          if item["code"] == "joint_lifecycle")["recognition_context"]
    assert first_context["data_version"] == "freeze-A"
    assert second_context["data_version"] == "freeze-B"


def test_explicit_production_context_is_not_overwritten_by_ambient_experiment():
    from caisen.research_config import RecognitionExperimentConfig, recognition_experiment
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    pattern = Pattern("production-shape", "000001", "stock", "1d", "w_bottom", "bullish",
                      "forming", "2024-01-01", None,
                      key_points={"pivot_1_date": "2024-01-01", "pivot_1": 10,
                                  "pivot_1_kind": "low"},
                      evidence=[{"code": "recognition_context", "value": {
                          "recognition_experiment": None, "engine_version": "caisen-joint-pattern-session-v6"}}])

    with recognition_experiment(RecognitionExperimentConfig(
            "internal_rebound_coverage_g1_v1", require_internal_rebound_coverage=True)):
        lifecycle = attach_joint_lifecycle(pattern)

    assert lifecycle["recognition_context"]["recognition_experiment"] is None


def test_g1_pattern_context_survives_reattachment_in_default_scope():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    context = {"recognition_experiment": {"experiment_id": "internal_rebound_coverage_g1_v1",
                "require_internal_rebound_coverage": True,
                "cache_key": ["internal_rebound_coverage_g1_v1", None, "test_count_first", None, True, False]},
               "engine_version": "caisen-joint-pattern-session-v6"}
    pattern = Pattern("g1-shape", "000001", "stock", "1d", "w_bottom", "bullish",
                      "forming", "2024-01-01", None,
                      key_points={"pivot_1_date": "2024-01-01", "pivot_1": 10,
                                  "pivot_1_kind": "low"},
                      evidence=[{"code": "recognition_context", "value": context}])

    lifecycle = attach_joint_lifecycle(pattern)

    assert lifecycle["recognition_context"]["recognition_experiment"] == context["recognition_experiment"]
    assert lifecycle["recognition_context"]["engine_version"] == context["engine_version"]


def test_confirmed_bull_flag_lifecycle_marks_its_real_breakout_entry_eligible():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "flag", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        rule_version="caisen-joint-pattern-session-v6",
        key_points={"pivot_1_date": "1994-01-03", "pivot_1_kind": "low", "pivot_1": 10.0},
        neckline={"slope": -0.01, "intercept": 12.0},
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                  {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                    "frozen_boundary": 12.0, "required_close": 12.1}},
                  {"code": "flag_lifecycle", "value": {
                    "entry_eligible_at_confirmation": True,
                    "entry_eligibility_at_confirmation_status": "known"}},
                  {"code": "recognition_context", "value": {"recognition_experiment": None,
                    "engine_version": "caisen-joint-pattern-session-v6"}}],
    )

    lifecycle = attach_joint_lifecycle(flag)

    assert lifecycle["entry_eligible"] is True
    assert lifecycle["entry_eligible_at_confirmation"] is True
    assert lifecycle["events"][-1]["kind"] == "confirmed"
    assert lifecycle["events"][-1]["boundary"] == 12.0
    assert lifecycle["schema_version"] == "joint-pattern-instance-v4"


def test_confirmed_legacy_flag_without_history_stays_unknown_on_reattachment():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    flag = Pattern(
        "legacy-confirmed-flag", "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "1994-01-03", 20, confirmation_date="1994-03-14",
        key_points={"pivot_1_date": "1994-01-03", "pivot_1_kind": "low", "pivot_1": 10.0},
        neckline={"slope": -0.01, "intercept": 12.0},
        evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                  {"code": "flag_lifecycle", "value": {"state": "confirmed"}}],
    )

    first = attach_joint_lifecycle(flag)
    assert first["entry_eligible_at_confirmation"] is None
    assert first["entry_eligibility_at_confirmation_status"] == "unknown_missing_confirmation_evidence"

    flag.evidence.extend([
        {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
          "frozen_boundary": 12.0, "required_close": 12.1}},
        {"code": "flag_lifecycle", "value": {"state": "confirmed",
          "entry_eligible_at_confirmation": True,
          "entry_eligibility_at_confirmation_status": "known",
          "selection_at_confirmation": {"date": "1994-03-14", "selected": True}}},
    ])
    second = attach_joint_lifecycle(flag)

    assert second["entry_eligible_at_confirmation"] is None
    assert second["entry_eligibility_at_confirmation_status"] == \
        "unknown_missing_confirmation_evidence"
    assert second["selection_at_confirmation"]["selected"] is True

    flag.state = "invalidated"
    third = attach_joint_lifecycle(flag)
    assert third["entry_eligible"] is False
    assert third["entry_eligible_at_confirmation"] is None
    assert third["entry_eligibility_at_confirmation_status"] == \
        "unknown_missing_confirmation_evidence"


def test_flag_confirmation_selection_and_history_are_immutable_across_later_evidence():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    def make_flag(pattern_id, selected):
        return Pattern(
            pattern_id, "600684", "stock", "1d", "bull_flag", "bullish", "confirmed",
            "1994-01-03", 20, confirmation_date="1994-03-14",
            key_points={"pivot_1_date": "1994-01-03", "pivot_1_kind": "low", "pivot_1": 10.0},
            neckline={"slope": -0.01, "intercept": 12.0},
            evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                      {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                        "frozen_boundary": 12.0, "required_close": 12.1}},
                      {"code": "flag_lifecycle", "value": {
                        "state": "confirmed", "entry_eligible_at_confirmation": selected,
                        "entry_eligibility_at_confirmation_status": "known",
                        "selection_at_confirmation": {"date": "1994-03-14",
                                                       "selected": selected}}}],
        )

    from caisen.daily_incremental import DailyRecognitionSession
    session = DailyRecognitionSession("600684", "stock", tick_size=0.01)

    selected = make_flag("selected-flag", True)
    first = session._attach_session_lifecycle(selected, discovered_at="1994-03-11")
    assert first["entry_eligible_at_confirmation"] is True
    selected_flag_lifecycle = next(item["value"] for item in selected.evidence
                                   if item["code"] == "flag_lifecycle")
    selected_flag_lifecycle["selection_at_confirmation"] = None
    selected.evidence.append({"code": "execution_suppression", "value": {
        "suppressed": True, "suppressed_at": "1994-03-14"}})
    second = session._attach_session_lifecycle(selected)
    assert second["entry_eligible"] is False
    assert second["entry_eligible_at_confirmation"] is True
    assert second["selection_at_confirmation"]["selected"] is True

    selected.state = "invalidated"
    selected_flag_lifecycle["state"] = "invalidated"
    terminal = session._attach_session_lifecycle(selected)
    assert terminal["entry_eligible"] is False
    assert terminal["entry_eligible_at_confirmation"] is True

    rejected = make_flag("rejected-flag", False)
    rejected.key_points["pivot_1_date"] = "1994-01-04"
    rejected.key_points["pivot_1"] = 10.1
    rejected_first = session._attach_session_lifecycle(rejected, discovered_at="1994-03-11")
    assert rejected_first["entry_eligible"] is False
    assert rejected_first["entry_eligible_at_confirmation"] is False
    rejected_flag_lifecycle = next(item["value"] for item in rejected.evidence
                                   if item["code"] == "flag_lifecycle")
    rejected_flag_lifecycle["entry_eligible_at_confirmation"] = True
    rejected_flag_lifecycle["selection_at_confirmation"]["selected"] = True
    rejected_second = session._attach_session_lifecycle(rejected)
    assert rejected_second["entry_eligible"] is False
    assert rejected_second["entry_eligible_at_confirmation"] is False
    assert rejected_second["selection_at_confirmation"]["selected"] is False
    rejected.state = "invalidated"
    rejected_flag_lifecycle["state"] = "invalidated"
    rejected_terminal = session._attach_session_lifecycle(rejected)
    assert rejected_terminal["entry_eligible"] is False
    assert rejected_terminal["entry_eligible_at_confirmation"] is False
