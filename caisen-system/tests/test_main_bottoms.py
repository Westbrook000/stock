from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caisen.main_bottoms import (
    MAIN_RULES,
    PIVOT_SCALES,
    _independent_neckline_tests,
    _ihs_right_shoulder_intact,
    _ihs_valley_intact,
    _structure_candidates,
    _triangle_boundaries_intact,
    _triangle_apex_index,
    _w_rebound_peak_associated,
    detect_main_bottoms,
    detect_multiscale_pivots,
    link_main_bottom_children,
)
from caisen.research_config import RecognitionExperimentConfig, recognition_experiment
from caisen.schemas import Pattern, Pivot


def synthetic_w_bars():
    dates = pd.bdate_range("2021-01-04", periods=1300)
    anchors = [(0, 3.0), (80, 3.30), (160, 2.8), (240, 3.33),
               (320, 2.9), (390, 2.22), (470, 3.30), (570, 2.25),
               (660, 3.28), (760, 3.0), (820, 3.34), (930, 3.0), (1040, 3.5),
               (1100, 3.34), (1180, 4.4), (1299, 4.6)]
    x = np.arange(len(dates))
    close = np.interp(x, [point[0] for point in anchors], [point[1] for point in anchors])
    # Small deterministic movement produces realistic true ranges without new large swings.
    close = close + 0.012 * np.sin(x * 0.31) + 0.006 * np.sin(x * 0.071)
    frame = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 0.025,
        "low": close - 0.025,
        "close": close,
        "volume": 1_000_000 + 100_000 * np.sin(x * 0.17),
    })
    frame.loc[[390, 570], "low"] = [2.22, 2.25]
    frame.loc[[470, 820, 1100], "high"] = [3.30, 3.34, 3.34]
    return frame


def evidence(pattern):
    return {item["code"]: item["value"] for item in pattern.evidence}


def main_events(patterns, family="w_bottom"):
    return [item for item in patterns if item.pattern_type == family
            and evidence(item).get("main_structure", {}).get("role") == "main"]


def is_selected(pattern):
    return evidence(pattern).get("main_selection", {}).get("selected", False)


def test_main_bottom_thresholds_are_declared_in_the_rule_catalog():
    assert MAIN_RULES["source_rule_version"] == "caisen-main-bottoms-v17"
    assert MAIN_RULES["main_pattern_rule_version"] == "caisen-v30-main-bottom-v1"
    assert MAIN_RULES["neckline_tests"]["candidate_reversal_atr_multiple"] == 2.0
    assert MAIN_RULES["neckline_tests"]["minimum_departure_atr"] == 1.0
    assert MAIN_RULES["timeframes"]["1d"]["structure_window_calendar_years"] == 5
    assert MAIN_RULES["timeframes"]["1w"]["structure_window_calendar_years"] == 10
    assert MAIN_RULES["minimum_duration_calendar_months_by_timeframe"] == {"1d": 6, "1w": 12}
    assert MAIN_RULES["minimum_duration_calendar_months"] == 6
    assert MAIN_RULES["neckline_tests"]["minimum_independent_groups"] == 2
    assert MAIN_RULES["neckline_tests"]["consumed_boundary_phase_rule"] == \
        "same_structure_exact_frozen_neckline_v2"
    assert MAIN_RULES["inverse_head_shoulders"]["maximum_leg_duration_ratio"] == 2.5
    assert MAIN_RULES["inverse_head_shoulders"][
        "minimum_intervening_right_trough_head_separation_atr_multiple"] == 2.0
    assert MAIN_RULES["breakout"]["close_buffer_previous_atr_multiple"] == 1.0
    assert tuple((name, atr_multiple, percent_multiple) for name, atr_multiple, percent_multiple in PIVOT_SCALES) == (
        ("2atr_5pct", 2.0, 0.05), ("3atr_8pct", 3.0, 0.08), ("5atr_12pct", 5.0, 0.12))


def test_consumed_neckline_phase_lookup_matches_naive_lifecycle_rules():
    from caisen.main_bottoms import _consumed_neckline_phase_matcher

    phases = [
        {"line": {"slope": -0.1, "intercept": 5.0}, "selected": True},
        {"line": {"slope": -0.2, "intercept": 6.0}, "selected": True,
         "consumed_index": 9},
        {"line": {"slope": -0.3, "intercept": 7.0}, "selected": False},
        {"line": {"slope": -0.4, "intercept": 8.0}, "selected": True,
         "consumed_index": 11},
        {"line": {"slope": -0.5, "intercept": 9.0}, "selected": True,
         "consumed_index": 4},
    ]
    matches = _consumed_neckline_phase_matcher(phases, current_index=10)

    assert matches({"slope": -0.1 + 5e-13, "intercept": 5.0})
    assert matches({"slope": -0.2, "intercept": 6.0})
    assert matches({"slope": -0.5, "intercept": 9.0})
    assert not matches({"slope": -0.3, "intercept": 7.0})
    assert not matches({"slope": -0.4, "intercept": 8.0})
    assert not matches({"slope": -0.5, "intercept": 9.001})


def test_516630_same_anchor_geometry_keeps_distinct_frozen_boundary_reentries():
    from caisen.daily_incremental import recognize_daily_instances

    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-10-08"].reset_index(drop=True)
    found = recognize_daily_instances(bars, "516630", "etf", recognizer="boundary_first")
    same_structure = [item for item in found["instances"]
        if item.pattern_type == "inverse_head_shoulders"
        and [item.key_points.get(f"pivot_{number}_date") for number in (1, 2, 3)] ==
        ["2023-10-24", "2024-02-05", "2024-04-22"]]

    assert len(same_structure) >= 2
    assert len({item.boundary_version_id for item in same_structure}) == len(same_structure)
    assert len({(item.neckline["slope"], item.neckline["intercept"])
                for item in same_structure}) == len(same_structure)
    assert all(item.state == "confirmed" and item.confirmation_date for item in same_structure)
    for item in same_structure:
        selection = evidence(item)["neckline_selection"]
        assert pd.Timestamp(selection["line_frozen_at"]) < pd.Timestamp(item.confirmation_date)
    assert all(item.multiscale_evidence for item in same_structure)
    matching_entries = [event for event in found["entry_events"]
                        if event["pattern"].get("pattern_type") == "inverse_head_shoulders"
                        and [event["pattern"].get("key_points", {}).get(f"pivot_{number}_date")
                             for number in (1, 2, 3)] == [
                                 "2023-10-24", "2024-02-05", "2024-04-22"]]
    assert {event["instance_id"] for event in matching_entries} == {
        item.pattern_id for item in same_structure}


def test_w_frozen_neckline_must_be_associated_with_the_intervening_rebound():
    lows = [Pivot(10, "2025-01-01", 2.5, "low", 11, "2025-01-02"),
            Pivot(30, "2025-02-01", 2.6, "low", 31, "2025-02-02")]
    highs = np.array([3.0] * 41, dtype=float)
    highs[20] = 3.03
    atr = np.full(41, 0.05, dtype=float)
    candidate = {"tests": [{"pivot_index": 15, "level": 3.26, "tolerance": 0.025}]}

    assert not _w_rebound_peak_associated(
        lows, candidate["tests"], {"slope": 0.0, "intercept": 3.225}, highs, atr, 0.001)
    highs[20] = 3.225
    assert _w_rebound_peak_associated(
        lows, [], {"slope": 0.0, "intercept": 3.225}, highs, atr, 0.001)


def test_batch_w_pattern_can_reissue_on_a_new_neckline_after_old_line_fails(monkeypatch):
    import caisen.main_bottoms as main_bottoms

    dates = pd.bdate_range("2020-01-01", periods=171)
    close = np.full(len(dates), 10.1, dtype=float)
    close[150] = 12.3  # First line at 12.0 gets a buffered breakout.
    close[151] = 11.5  # The first boundary fails before the revised line triggers.
    close[170] = 12.8  # New line at 12.5 gets its own buffered breakout.
    high = close + 0.05
    low = close - 0.05
    high[0], low[0] = 20.0, 8.0  # Supplies a qualifying low-history position.
    high[20] = 12.0  # The W rebound is covered by both candidate boundaries.
    low[10], low[30] = 10.0, 10.1
    frame = pd.DataFrame({"date": dates, "open": close, "high": high,
                          "low": low, "close": close, "volume": 1_000_000.0})
    first = Pivot(10, str(dates[10].date()), 10.0, "low", 11,
                  str(dates[11].date()), "2ATR|5%")
    rebound = Pivot(20, str(dates[20].date()), 12.0, "high", 21,
                    str(dates[21].date()), "2ATR|5%")
    second = Pivot(30, str(dates[30].date()), 10.1, "low", 31,
                   str(dates[31].date()), "2ATR|5%")

    def boundary(level, pair_id, frozen_index, test_indices):
        tests = []
        for index in test_indices:
            day = str(dates[index].date())
            tests.append({
                "pivot_index": index, "pivot_date": day,
                "known_index": index, "known_date": day,
                "coverage_start_index": index, "coverage_end_index": index,
                "coverage_start_date": day, "coverage_end_date": day,
                "coverage_max_excess": 0.0,
                "level": level, "tolerance": 0.1, "valid": True,
                "pair_id": pair_id, "group_id": f"{pair_id}-{index}",
                "test_span_bars": test_indices[-1] - test_indices[0],
                "selection_score": (0, 0, 0.1, -index),
            })
        return {"line": {"slope": 0.0, "intercept": level}, "tests": tests,
                "score": (-2, -(test_indices[-1] - test_indices[0]), 0.1, -test_indices[-1]),
                "frozen_index": frozen_index}

    candidates = [boundary(12.0, "first-line", 6, [5, 6]),
                  boundary(12.5, "replacement-line", 145, [140, 145])]
    monkeypatch.setattr(main_bottoms, "_independent_neckline_tests",
                        lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(main_bottoms, "_w_rebound_peak_associated", lambda *_args: True)
    monkeypatch.setattr(main_bottoms, "_neckline_formation_coverage",
                        lambda _high, _atr, _line, start, end, *_args, **_kwargs: {
                            "passed": True, "start_index": start, "end_index": end,
                            "maximum_excess": 0.0})

    result = main_bottoms._make_main_pattern(
        frame, "000001", "stock", "1d", "w_bottom", [first, rebound, second],
        [first, rebound, second], np.full(len(frame), 0.2), 0.01,
        "2atr_5pct", "5y", background_bars=frame)

    assert result is not None
    pattern = result[0] if isinstance(result, list) else result
    assert pattern.state == "confirmed"
    assert pattern.confirmation_date == str(dates[170].date())
    assert pattern.neckline["intercept"] == 12.5


def test_r02_late_boundary_discovery_does_not_backfill_or_emit_a_buy():
    from caisen.daily_incremental import recognize_daily_instances

    bars = pd.read_csv(Path(__file__).parent / "fixtures" / "main_bottoms" / "R02-000008.csv",
                       parse_dates=["date"])
    result = recognize_daily_instances(bars, "000008", "etf", recognizer="boundary_first")
    target = next(item for item in result["instances"]
                  if item.pattern_type == "w_bottom"
                  and item.key_points.get("pivot_1_date") == "2025-04-29"
                  and item.key_points.get("pivot_2_date") == "2025-06-23"
                  and any(record.get("date") == "2025-08-08"
                          for record in evidence(item)["neckline_selection"].get(
                              "early_breakouts", [])))
    late_instance_id = target.pattern_id
    target_events = [event for event in result["instance_events"]
                     if event["instance_id"] == late_instance_id]

    # This boundary pairs on 2025-11-19, after its 2025-08-08 crossing.
    # It may start a forming lifecycle then, but cannot backfill a buy.
    assert target.confirmation_date is None
    assert target_events
    assert min(event["event_date"] for event in target_events) == "2025-11-19"
    assert not any(event.get("event_date") == "2025-08-08" for event in target_events)
    assert not any(event["instance_id"] == late_instance_id for event in result["entry_events"])


def test_streaming_boundary_discovery_does_not_backdate_late_early_crossings():
    from caisen.daily_incremental import DailyRecognitionSession

    bars = pd.read_csv(Path(__file__).parent / "fixtures" / "main_bottoms" / "R02-000008.csv",
                       parse_dates=["date"])
    session = DailyRecognitionSession("000008", "etf", tick_size=0.001,
                                      boundary_first=True)
    prior_event_ids = []
    target_availability = None
    target_early_dates = []
    target_instance_id = None
    for _, row in bars.iterrows():
        bar = row.to_dict()
        bar["date"] = pd.Timestamp(bar["date"])
        for key in ("open", "high", "low", "close"):
            bar[key] = float(bar[key])
        session.append_bar(bar)
        for candidate in session.active.values():
            pattern = candidate.get("pattern")
            if (pattern is None or pattern.pattern_type != "w_bottom"
                    or pattern.key_points.get("pivot_1_date") != "2025-04-29"
                    or pattern.key_points.get("pivot_2_date") != "2025-06-23"):
                continue
            metadata = {item["code"]: item["value"] for item in pattern.evidence}
            if metadata.get("boundary_available_at") and target_availability is None:
                target_availability = metadata["boundary_available_at"]["date"]
                target_instance_id = pattern.pattern_id
                target_early_dates = [item["date"] for item in
                                      metadata["neckline_selection"].get("early_breakouts", [])]
        events = session.instance_events()
        assert [event["event_id"] for event in events[:len(prior_event_ids)]] == prior_event_ids
        newly_added = events[len(prior_event_ids):]
        current_date = str(bar["date"].date())
        for event in newly_added:
            if event.get("event_type") == "early_breakout_consumed":
                assert event["event_date"] == current_date
        prior_event_ids = [event["event_id"] for event in events]

    assert target_availability == "2025-11-19"
    assert target_early_dates == ["2025-08-08"]
    assert not any(event["event_date"] == "2025-08-08"
                   and event.get("event_type") == "early_breakout_consumed"
                   for event in session.instance_events())
    assert not any(event.get("entry_eligible") and event["instance_id"] == target_instance_id
                   for event in session.instance_events())


def test_000538_weekly_frozen_boundary_and_state_are_prefix_stable():
    from caisen.market import aggregate_weekly, default_china_calendar

    source = Path(__file__).parents[2] / "data" / "stocks" / "000538.csv"
    full_daily = pd.read_csv(source, parse_dates=["date"])
    snapshots = []
    events_by_cutoff = {}
    stable_instance_id = None
    for cutoff in ("2024-08-30", "2024-09-27", "2024-11-29", "2026-11-06"):
        daily = full_daily[full_daily.date <= cutoff].reset_index(drop=True)
        weekly = aggregate_weekly(daily, calendar=default_china_calendar())
        found = detect_main_bottoms(weekly, code="000538", kind="stock", timeframe="1w",
                                    background_bars=daily)
        events_by_cutoff[cutoff] = found
        matching = [item for item in found
                    if item.pattern_type == "w_bottom"
                    and item.key_points.get("pivot_1_date") == "2023-03-24"
                    and item.key_points.get("pivot_2_date") == "2024-02-02"
                    and evidence(item).get("neckline_selection", {}).get(
                        "line_frozen_at") == "2023-06-02"]
        target = (next(item for item in matching if item.pattern_id == stable_instance_id)
                  if stable_instance_id else matching[0])
        stable_instance_id = target.pattern_id
        detail = evidence(target)
        selection = detail["neckline_selection"]
        snapshots.append((cutoff, target, selection, detail))

    first_id = snapshots[0][1].pattern_id
    first_line = snapshots[0][1].neckline
    assert all(item.pattern_id == first_id for _, item, _, _ in snapshots), [
        (cutoff, item.pattern_id, item.neckline, item.state, item.confirmation_date)
        for cutoff, item, _, _ in snapshots]
    assert all(item.neckline == first_line for _, item, _, _ in snapshots)
    assert snapshots[0][1].state == "forming"
    assert snapshots[0][1].confirmation_date is None
    # The original 49.85 boundary remains the same historical instance; its
    # later invalidation does not borrow the new line's September confirmation.
    assert snapshots[1][1].confirmation_date is None
    assert snapshots[2][1].state == "confirmed"
    assert snapshots[2][1].confirmation_date == "2024-09-30"
    assert all(selection["line_frozen_at"] == "2023-06-02"
               for _, _, selection, _ in snapshots)
    assert all([test["pivot_date"] for test in detail["neckline_tests"] if test["valid"]]
               == ["2023-02-03", "2023-05-05"]
               for _, _, _, detail in snapshots)
    september_events = [item for item in events_by_cutoff["2024-09-27"]
                    if item.pattern_type == "w_bottom"
                    and item.key_points.get("pivot_1_date") == "2023-03-24"
                    and item.key_points.get("pivot_2_date") == "2024-02-02"]
    assert len({item.boundary_version_id for item in september_events}) >= 2
    assert any(item.pattern_id != first_id and item.confirmation_date == "2024-09-27"
               for item in september_events)
    later_revisions = [item for item in events_by_cutoff["2026-11-06"]
                       if item.pattern_type == "w_bottom"
                       and item.key_points.get("pivot_1_date") == "2023-03-24"
                       and item.key_points.get("pivot_2_date") == "2024-02-02"
                       and item.pattern_id != first_id]
    assert any(item.neckline != first_line and item.confirmation_date
               for item in later_revisions)
    assert all(pd.Timestamp(evidence(item)["neckline_selection"]["line_frozen_at"])
               < pd.Timestamp(item.confirmation_date)
               for item in later_revisions if item.confirmation_date)
    assert any(item.state == "forming" for item in september_events)


def test_weekly_detection_keeps_a_second_frozen_line_for_the_same_anchors(monkeypatch):
    from caisen import main_bottoms
    from caisen.pattern_instances import pattern_identity_metadata

    dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
    bars = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                         "low": 9.8, "close": 10.0, "volume": 1000})
    first = Pivot(2, str(dates[2].date()), 8.0, "low", 3,
                  str(dates[3].date()), "2ATR|5%")
    second = Pivot(10, str(dates[10].date()), 8.1, "low", 11,
                   str(dates[11].date()), "2ATR|5%")
    sequence = [first, second]
    scales = {name: list(sequence) for name, _atr_multiple, _percent in PIVOT_SCALES}
    monkeypatch.setattr(main_bottoms, "detect_multiscale_pivots",
                        lambda *_args, **_kwargs: scales)
    monkeypatch.setattr(main_bottoms, "_detect_neckline_test_pivots", lambda *_args: [])
    monkeypatch.setattr(main_bottoms, "_structure_candidates",
                        lambda scale_pivots, *_args, **_kwargs:
                        [("w_bottom", sequence)]
                        if scale_pivots is scales[PIVOT_SCALES[0][0]] else [])
    lines = [{"slope": 0.0, "intercept": 10.0 + 0.5 * index} for index in range(10)]

    def make_pattern(_frame, code, kind, timeframe, family, anchors, _pressure,
                     _atr, _tick, scale, _horizon, **kwargs):
        excluded = [item.get("line") for item in kwargs.get("consumed_boundary_phases", ())]
        line_index = next((index for index, line in enumerate(lines) if line not in excluded), None)
        if line_index is None:
            return None
        line = lines[line_index]
        key_points = {
            "pivot_1_date": anchors[0].date, "pivot_1_kind": "low", "pivot_1": anchors[0].price,
            "pivot_2_date": anchors[1].date, "pivot_2_kind": "low", "pivot_2": anchors[1].price,
            "pivot_scale": scale,
        }
        frozen = str(dates[11 + line_index * 2].date())
        tests = [{"pair_id": f"line-{line_index}", "group_id": f"g-{line_index}-1",
                  "pivot_date": str(dates[4 + line_index * 2].date()),
                  "pivot_index": 4 + line_index * 2, "level": line["intercept"], "valid": True},
                 {"pair_id": f"line-{line_index}", "group_id": f"g-{line_index}-2",
                  "pivot_date": frozen, "pivot_index": 11 + line_index * 2,
                  "level": line["intercept"], "valid": True}]
        evidence_rows = [
            {"code": "main_structure", "value": {"role": "main", "pivot_scale": scale,
                                                        "support_lower": 8.0}},
            {"code": "neckline_selection", "value": {
                "selected_candidate_id": f"line-{line_index}", "selected_line": line,
                "line_frozen_at": frozen, "structure_known_at": str(dates[11].date())}},
            {"code": "neckline_tests", "value": tests},
        ]
        identity = pattern_identity_metadata({"code": code, "kind": kind, "timeframe": timeframe,
            "pattern_type": family, "key_points": key_points, "neckline": line,
            "evidence": evidence_rows})
        return Pattern(pattern_id=identity["instance_id"], code=code, kind=kind, timeframe=timeframe,
            pattern_type=family, direction="bullish", state="confirmed", formed_at=anchors[0].date,
            confirmed_at=14 + line_index, confirmation_date=str(dates[14 + line_index].date()),
            last_observed_date=str(dates[-1].date()), key_points=key_points, neckline=line,
            evidence=evidence_rows, structure_key=identity["structure_key"],
            boundary_version_id=identity["boundary_version_id"])

    monkeypatch.setattr(main_bottoms, "_make_main_pattern", make_pattern)

    events = detect_main_bottoms(bars, code="000001", timeframe="1w")

    assert len(events) == 10
    assert len({event.structure_key for event in events}) == 1
    assert len({event.boundary_version_id for event in events}) == 10
    assert len({event.pattern_id for event in events}) == 10


def test_batch_detection_keeps_same_level_ihs_with_equal_outer_anchors_independent(monkeypatch):
    from caisen import main_bottoms
    from caisen.pattern_instances import pattern_identity_metadata

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    bars = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                         "low": 9.8, "close": 10.0, "volume": 1000})
    sequences = [
        [Pivot(2, str(dates[2].date()), 8.0, "low", 3, str(dates[3].date()), "2ATR|5%"),
         Pivot(12, str(dates[12].date()), 7.0, "low", 13, str(dates[13].date()), "2ATR|5%"),
         Pivot(25, str(dates[25].date()), 8.0, "low", 26, str(dates[26].date()), "2ATR|5%")],
        [Pivot(2, str(dates[2].date()), 8.0, "low", 3, str(dates[3].date()), "2ATR|5%"),
         Pivot(16, str(dates[16].date()), 7.1, "low", 17, str(dates[17].date()), "2ATR|5%"),
         Pivot(25, str(dates[25].date()), 8.0, "low", 26, str(dates[26].date()), "2ATR|5%")],
    ]
    scales = {name: [] for name, _atr_multiple, _percent in PIVOT_SCALES}
    scales[PIVOT_SCALES[0][0]] = [pivot for sequence in sequences for pivot in sequence]
    monkeypatch.setattr(main_bottoms, "detect_multiscale_pivots",
                        lambda *_args, **_kwargs: scales)
    monkeypatch.setattr(main_bottoms, "_detect_neckline_test_pivots", lambda *_args: [])
    monkeypatch.setattr(main_bottoms, "_structure_candidates",
                        lambda _pivots, *_args, **_kwargs:
                        [("inverse_head_shoulders", sequence) for sequence in sequences]
                        if _pivots is scales[PIVOT_SCALES[0][0]] else [])

    def make_pattern(_frame, code, kind, timeframe, family, anchors, _pressure,
                     _atr, _tick, scale, _horizon, **_kwargs):
        middle = anchors[1]
        key_points = {f"pivot_{number}_date": pivot.date
                      for number, pivot in enumerate(anchors, 1)}
        key_points.update({f"pivot_{number}_kind": pivot.kind
                          for number, pivot in enumerate(anchors, 1)})
        key_points["pivot_scale"] = scale
        line = {"slope": 0.0, "intercept": 10.0 + middle.index / 100}
        evidence_rows = [
            {"code": "main_structure", "value": {
                "role": "main", "pivot_scale": scale}},
            {"code": "neckline_selection", "value": {
                "selected_line": line, "line_frozen_at": str(dates[27].date()),
                "structure_known_at": str(dates[27].date())}},
        ]
        identity = pattern_identity_metadata({"code": code, "kind": kind, "timeframe": timeframe,
            "pattern_type": family, "key_points": key_points, "neckline": line,
            "evidence": evidence_rows})
        return Pattern(pattern_id=identity["instance_id"], code=code, kind=kind,
            timeframe=timeframe, pattern_type=family, direction="bullish", state="forming",
            formed_at=anchors[0].date, confirmed_at=None, last_observed_date=str(dates[-1].date()),
            key_points=key_points, neckline=line, evidence=evidence_rows,
            structure_key=identity["structure_key"],
            boundary_version_id=identity["boundary_version_id"])

    monkeypatch.setattr(main_bottoms, "_make_main_pattern", make_pattern)

    events = detect_main_bottoms(bars, code="600633", timeframe="1d")

    assert len(events) == 2
    assert all(next(item["value"] for item in event.evidence
                    if item["code"] == "main_structure")["role"] == "main"
               for event in events)


def test_nested_long_cycle_w_bottom_keeps_confirmation_but_not_an_independent_buy():
    bars = synthetic_w_bars()
    # Stop at the first qualifying breakout so the later target touch does not
    # change the event's current terminal state in this confirmation test.
    found = detect_main_bottoms(bars[bars.date <= "2024-12-31"].copy(),
                                code="fixture", timeframe="1d")
    bottoms = [item for item in found if item.pattern_type == "w_bottom"]

    assert bottoms
    selected = next(item for item in bottoms
                    if evidence(item)["main_structure"]["role"] == "internal")
    main = evidence(selected)["main_structure"]
    tests = [item for item in evidence(selected)["neckline_tests"] if item["valid"]]
    assert main["family"] == "w_bottom"
    assert main["scale"] == "5y"
    assert main["signal_eligible"] is False
    assert main["eligibility_reason"] == "nested_in_known_main_structure"
    assert selected.confirmation_date == "2024-11-20"
    assert pd.Timestamp(selected.formed_at) + pd.DateOffset(months=6) <= pd.Timestamp(selected.confirmation_date)
    assert len(tests) >= 2
    assert tests[1]["pivot_index"] - tests[0]["pivot_index"] >= 10
    assert all(item["known_date"] <= main["frozen_at"] for item in tests[:2])
    assert all(item["departure_atr"] >= 1 for item in tests[:2])
    assert evidence(selected)["volume_support"]["required"] is False


def test_pair_first_topology_enumerator_preserves_candidate_order_and_anchors():
    bars = synthetic_w_bars()
    atr_values = (bars.high - bars.low).rolling(14, min_periods=1).mean().to_numpy()
    pivots = detect_multiscale_pivots(bars, timeframe="1d", tick_size=0.01)["2atr_5pct"]
    lows = pd.to_numeric(bars.low, errors="coerce").to_numpy(dtype=float)

    reference = _structure_candidates(pivots, "1d", atr_values, 0.01, low_values=lows)
    optimized = _structure_candidates(
        pivots, "1d", atr_values, 0.01, low_values=lows, fast_candidate_enumeration=True)

    assert [(family, [(p.index, p.kind, p.price) for p in sequence])
            for family, sequence in optimized] == [
                (family, [(p.index, p.kind, p.price) for p in sequence])
                for family, sequence in reference]


def test_real_600108_shared_prefix_entries_match_their_own_frozen_geometry():
    from caisen.daily_incremental import recognize_daily_instances

    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-01-09"].reset_index(drop=True)
    result = recognize_daily_instances(bars, "600108", "stock", recognizer="boundary_first")
    target = next(item for item in result["instances"]
                  if item.pattern_type == "w_bottom"
                  and item.key_points.get("pivot_1_date") == "2024-02-06"
                  and item.key_points.get("pivot_2_date") == "2024-09-18")
    outer = next(item for item in result["instances"]
                 if item.pattern_type == "w_bottom"
                 and item.key_points.get("pivot_1_date") == "2021-02-04"
                 and item.key_points.get("pivot_2_date") == "2024-09-18")
    assert target.pattern_id != outer.pattern_id
    assert target.boundary_version_id != outer.boundary_version_id

    entries = result["entry_events"]
    assert all(event["entry_eligible"] for event in entries)
    assert len({event["instance_id"] for event in entries}) == len(entries)
    for event in entries:
        pattern = event["pattern"]
        detail = evidence(Pattern(**pattern))
        breakout = detail.get("neckline_breakout") or detail.get("flag_breakout")
        assert breakout is not None
        assert breakout["date"] == event["event_date"]
        assert breakout["close"] > breakout["required_close"]
        observed_close = float(bars.loc[bars.date.eq(pd.Timestamp(event["event_date"])),
                                        "close"].iloc[0])
        assert breakout["close"] == pytest.approx(observed_close)
        assert pattern["boundary_version_id"].startswith("boundary-")


def test_516630_weekly_ihs_necklines_do_not_exceed_the_5pct_rise_gate():
    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    daily = pd.read_csv(source, parse_dates=["date"])
    from caisen.market import aggregate_weekly, default_china_calendar

    weekly = aggregate_weekly(daily, calendar=default_china_calendar())
    found = detect_main_bottoms(weekly, code="516630", kind="etf", timeframe="1w",
                               background_bars=daily)

    # The two independent weekly resistance tests at 1.283 and 1.378 imply a
    # 7.4% cumulative relative rise; other, distinct H&S candidates may remain.
    assert (1.378 - 1.283) / 1.283 > 0.05
    accepted = [item for item in found if item.pattern_type == "inverse_head_shoulders"]
    assert accepted
    for item in accepted:
        assert evidence(item)["main_structure"]["minimum_duration_months"] == 12
        tests = [test for test in evidence(item).get("neckline_tests", []) if test.get("valid")]
        if len(tests) >= 2:
            rise = (tests[-1]["level"] - tests[0]["level"]) / tests[0]["level"]
            assert rise <= MAIN_RULES["neckline_slope"]["maximum_cumulative_relative_rise"]
    assert any(item.pattern_type == "w_bottom" for item in found)


def test_516630_shared_daily_entry_prefers_frozen_parent_ihs_over_nested_w():
    from caisen.daily_incremental import recognize_daily_instances
    from caisen.pattern_instances import pattern_instance_is_frozen

    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    daily = pd.read_csv(source, parse_dates=["date"])
    daily = daily[daily.date <= "2024-10-08"].reset_index(drop=True)
    result = recognize_daily_instances(daily, "516630", "etf", recognizer="boundary_first")
    parent_ihs = [item for item in result["instances"]
                  if item.pattern_type == "inverse_head_shoulders"
                  and [item.key_points.get(f"pivot_{number}_date") for number in (1, 2, 3)] ==
                      ["2023-10-24", "2024-02-05", "2024-04-22"]]

    assert parent_ihs
    ihs = next(item for item in parent_ihs)
    ihs_detail = evidence(ihs)
    # The older drawing labels the same two IHS shoulder/head lows as a W.
    # The shared recognizer keeps the validated three-anchor IHS and does not
    # emit an independent W instance or BUY for that child label.
    assert not any(item.pattern_type == "w_bottom"
                   and item.key_points.get("pivot_1_date") == "2024-02-05"
                   and item.key_points.get("pivot_2_date") == "2024-04-22"
                   for item in result["instances"])
    assert not any(event["entry_eligible"]
                   and event["pattern"].get("pattern_type") == "w_bottom"
                   and event["pattern"].get("key_points", {}).get("pivot_1_date") == "2024-02-05"
                   and event["pattern"].get("key_points", {}).get("pivot_2_date") == "2024-04-22"
                   for event in result["entry_events"])
    assert ihs.state == "confirmed" and ihs.confirmation_date == "2024-10-08"
    assert pattern_instance_is_frozen(ihs.to_dict())
    assert ihs.pattern_id.startswith("516630:")
    assert ihs.boundary_version_id.startswith("boundary-")
    assert ihs_detail["main_structure"]["role"] == "main"
    assert ihs_detail["neckline_selection"]["formation_coverage"]["passed"] is True
    assert ihs_detail["neckline_selection"]["structure_known_at"] == "2024-04-24"
    same_day_signals = [signal for signal in result["signals"]
                        if signal["signal_date"] == "2024-10-08"]
    assert len(same_day_signals) == 1
    assert ihs.pattern_id in same_day_signals[0]["pattern_ids"]
    assert [(event["event_date"], event["entry_eligible"])
            for event in result["entry_events"]
            if event["instance_id"] == ihs.pattern_id] == [
                ("2024-10-08", True)]


def test_159718_one_atr_reference_keeps_asymmetric_shoulders_and_first_shared_entry(monkeypatch):
    from caisen.daily_incremental import DailyRecognitionSession

    monkeypatch.setitem(MAIN_RULES["neckline_tests"], "candidate_reversal_atr_multiple", 1.0)
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-30"].reset_index(drop=True)
    session = DailyRecognitionSession("159718", "etf", tick_size=0.01)
    emitted_signals = []
    for row in bars.to_dict("records"):
        emitted_signals.extend(session.append_bar(row))
    anchors = ["2023-08-22", "2024-04-19", "2025-01-13"]
    entries = [event for event in session.instance_events()
               if event["entry_eligible"]
               and event["pattern"].get("pattern_type") == "inverse_head_shoulders"
               and [event["pattern"]["key_points"].get(f"pivot_{number}_date")
                    for number in (1, 2, 3)] == anchors]

    assert [event["event_date"] for event in entries] == [
        "2025-04-01", "2025-05-21", "2025-05-29"]
    assert len({event["boundary_version_id"] for event in entries}) == 3
    assert len({str(event["pattern"]["neckline"]) for event in entries}) == 3
    entry = entries[0]
    assert entry["event_date"] == "2025-04-01"
    assert entry["pattern"]["confirmation_date"] == "2025-04-01"
    assert entry["pattern"]["neckline"]["slope"] <= 0
    assert entry["pattern"]["pattern_id"].startswith("159718:")
    assert entry["boundary_version_id"].startswith("boundary-")
    assert not any(event["entry_eligible"] and event["instance_id"] == entry["instance_id"]
                   and event["event_date"] in {"2025-05-21", "2025-05-29"}
                   for event in session.instance_events())
    assert any(event.get("event_type") == "weak_crossing"
               and event["instance_id"] == entry["instance_id"]
               for event in session.instance_events())


def test_159718_two_atr_shared_session_allows_a_later_changed_neckline_entry():
    from caisen.daily_incremental import DailyRecognitionSession

    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-30"].reset_index(drop=True)
    session = DailyRecognitionSession("159718", "etf", tick_size=0.01)
    emitted_signals = []
    for row in bars.to_dict("records"):
        emitted_signals.extend(session.append_bar(row))
    anchors = ["2023-08-22", "2024-04-19", "2025-01-13"]
    entries = [event for event in session.instance_events()
               if event["entry_eligible"]
               and event["pattern"].get("pattern_type") == "inverse_head_shoulders"
               and [event["pattern"]["key_points"].get(f"pivot_{number}_date")
                    for number in (1, 2, 3)] == anchors]

    assert [event["event_date"] for event in entries] == [
        "2025-03-20", "2025-04-01", "2025-05-21"]
    assert len({event["boundary_version_id"] for event in entries}) == 3
    emitted_revisions = [
        (signal["signal_date"], pattern["pattern_id"])
        for signal in emitted_signals
        for pattern in signal.get("patterns", [])
        if pattern.get("pattern_type") == "inverse_head_shoulders"
        and [pattern.get("key_points", {}).get(f"pivot_{number}_date")
             for number in (1, 2, 3)] == anchors]
    assert {instance_id for _, instance_id in emitted_revisions} == {
        event["instance_id"] for event in entries}
    assert sorted(date for date, _ in emitted_revisions) == [
        "2025-03-20", "2025-04-01", "2025-05-21"]


def test_w_bottom_breakout_requires_atr_buffer_and_keeps_weak_crossings():
    bars = synthetic_w_bars()
    bars.loc[820:, "close"] = 3.34
    bars.loc[820:, "open"] = 3.33
    bars.loc[820:, "high"] = 3.36
    bars.loc[820:, "low"] = 3.32
    found = detect_main_bottoms(bars, code="fixture", timeframe="1d")
    candidates = [item for item in found if item.pattern_type == "w_bottom"
                  and evidence(item).get("weak_crossings")]
    assert candidates
    selected = candidates[0]
    detail = evidence(selected)

    assert selected.confirmation_date is None
    assert selected.state != "confirmed"
    assert detail["weak_crossings"]
    assert all(item["close"] <= item["required_close"] for item in detail["weak_crossings"])


def test_main_bottom_is_invalidated_when_support_breaks_before_later_neckline_recovery():
    bars = synthetic_w_bars()
    broken_date = pd.Timestamp("2023-06-01")
    index = bars.index[bars.date == broken_date][0]
    bars.loc[index, ["open", "high", "low", "close"]] = [2.0, 2.05, 1.95, 2.0]
    found = detect_main_bottoms(bars, code="fixture", timeframe="1d")
    candidates = [item for item in main_events(found) if item.formed_at == "2022-07-01"]

    assert candidates
    assert all(evidence(item)["main_structure"]["signal_eligible"] is False for item in candidates)
    assert all(evidence(item)["structure_invalidation"]["date"] == "2023-06-01"
               for item in candidates)
    assert all(item.confirmation_date is None for item in candidates)


def test_main_bottom_requires_two_separated_resistance_tests_and_six_months():
    bars = synthetic_w_bars().iloc[:600].copy()
    # Keep the price geometry, but compress its observed dates to under six
    # months. Even a valid swing sequence must not become a long-cycle event.
    bars["date"] = pd.to_datetime(np.linspace(pd.Timestamp("2022-01-01").value,
                                               pd.Timestamp("2022-05-01").value,
                                               len(bars))).normalize()
    found = detect_main_bottoms(bars, code="fixture", timeframe="1d")

    assert not main_events(found)


def test_appending_future_bars_does_not_rewrite_frozen_main_bottom_geometry():
    bars = synthetic_w_bars()
    prefix = bars.iloc[:1050].copy()
    before = detect_main_bottoms(prefix, code="fixture", timeframe="1d")
    after = detect_main_bottoms(bars, code="fixture", timeframe="1d")
    before_by_id = {item.pattern_id: item for item in before}
    after_by_id = {item.pattern_id: item for item in after}

    assert set(before_by_id) <= set(after_by_id)
    for pattern_id, earlier in before_by_id.items():
        later = after_by_id[pattern_id]
        assert earlier.formed_at == later.formed_at
        assert earlier.key_points == later.key_points
        assert earlier.neckline == later.neckline
        assert evidence(earlier)["main_structure"]["frozen_at"] == evidence(later)["main_structure"]["frozen_at"]


def test_multiscale_confirmed_pivots_do_not_change_after_future_same_direction_extremes():
    bars = synthetic_w_bars()
    prefix = bars.iloc[:850].copy()
    extended = bars.copy()
    extended.loc[850:, "close"] = np.linspace(extended.close.iloc[850], 5.2, len(extended) - 850)
    extended.loc[850:, "open"] = extended.loc[850:, "close"]
    extended.loc[850:, "high"] = extended.loc[850:, "close"] + .025
    extended.loc[850:, "low"] = extended.loc[850:, "close"] - .025
    before = detect_multiscale_pivots(prefix, timeframe="1d")
    after = detect_multiscale_pivots(extended, timeframe="1d")

    for scale in before:
        frozen = [pivot for pivot in before[scale] if pivot.confirmed_at < len(prefix)]
        after_by_anchor = {(pivot.index, pivot.kind): pivot for pivot in after[scale]}
        for pivot in frozen:
            assert (pivot.index, pivot.kind) in after_by_anchor
            assert after_by_anchor[(pivot.index, pivot.kind)] == pivot


def test_independent_neckline_tests_allow_a_real_slope_but_reject_no_departure_near_miss():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=40))
    touches = [Pivot(5, str(dates[5].date()), 3.0, "high", 7,
                     str(dates[7].date()), "3ATR|8%|neckline_test_2ATR"),
               Pivot(25, str(dates[25].date()), 3.12, "high", 27,
                     str(dates[27].date()), "3ATR|8%|neckline_test_2ATR")]
    trough = Pivot(30, str(dates[30].date()), 2.2, "low", 32,
                   str(dates[32].date()), "3ATR|8%")
    last_anchor = Pivot(35, str(dates[35].date()), 2.4, "low", 37,
                        str(dates[37].date()), "3ATR|8%")
    intervening_low = Pivot(15, str(dates[15].date()), 2.0, "low", 17,
                            str(dates[17].date()), "3ATR|8%")
    atr_values = np.full(len(dates), 0.5)
    valid = _independent_neckline_tests(touches + [intervening_low], trough, last_anchor,
                                        atr_values, dates, .01, "1d")
    assert len(valid) == 2
    assert valid[0]["level"] < valid[1]["level"]
    assert valid[1]["departure_atr"] >= 1

    near_miss = Pivot(15, str(dates[15].date()), 2.8, "low", 17,
                      str(dates[17].date()), "3ATR|8%")
    rejected = _independent_neckline_tests(touches + [near_miss], trough, last_anchor,
                                           atr_values, dates, .01, "1d")
    assert rejected == []


def test_minimum_neckline_tests_experiment_regenerates_candidates_without_changing_default():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=40))
    bars = pd.DataFrame({"date": dates, "high": 2.0, "low": 1.5, "close": 1.9})
    touches = [Pivot(index, str(dates[index].date()), 3.0, "high", index + 2,
                     str(dates[index + 2].date()), "3ATR|8%|neckline_test_2ATR")
               for index in (5, 25)]
    for pivot in touches:
        bars.loc[pivot.index, "high"] = pivot.price
    trough = Pivot(30, str(dates[30].date()), 2.2, "low", 32,
                   str(dates[32].date()), "3ATR|8%")
    separating_low = Pivot(15, str(dates[15].date()), 2.0, "low", 17,
                           str(dates[17].date()), "3ATR|8%")
    pivots = touches + [trough, separating_low]
    atr_values = np.full(len(dates), 0.5)

    baseline = _independent_neckline_tests(
        pivots, trough, trough, atr_values, dates, 0.01, "1d", bars=bars)
    config = RecognitionExperimentConfig(
        "neckline_min3_v1", minimum_neckline_tests=3,
        admission_types=frozenset({"w_bottom", "inverse_head_shoulders",
                                   "triangle_bottom", "bull_flag"}))
    with recognition_experiment(config):
        strict = _independent_neckline_tests(
            pivots, trough, trough, atr_values, dates, 0.01, "1d", bars=bars)

    assert len(baseline) == 2
    assert strict == []
    assert _independent_neckline_tests(
        pivots, trough, trough, atr_values, dates, 0.01, "1d", bars=bars) == baseline


def test_neckline_selection_keeps_three_independent_descending_pressure_tests():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=55))
    bars = pd.DataFrame({"date": dates, "high": 2.4, "low": 1.5, "close": 2.0})
    points = [(5, 4.0), (25, 3.4), (45, 2.8)]
    pivots = [Pivot(i, str(dates[i].date()), price, "high", i + 2,
                    str(dates[i + 2].date()), "3ATR|8%|neckline_test_2ATR")
              for i, price in points]
    for i, price in points:
        bars.loc[i, ["high", "close"]] = [price, price - 0.1]
    trough = Pivot(48, str(dates[48].date()), 1.4, "low", 50,
                   str(dates[50].date()), "3ATR|8%")
    atr_values = np.full(len(dates), 0.5)
    intervening_lows = [
        Pivot(i, str(dates[i].date()), 1.5, "low", i + 2,
              str(dates[i + 2].date()), "3ATR|8%")
        for i in (15, 35)
    ]

    tests = _independent_neckline_tests(
        pivots + intervening_lows + [trough], trough, trough,
        atr_values, dates, 0.01, "1d", bars=bars)

    assert len(tests) == 3
    assert [item["pivot_index"] for item in tests] == [5, 25, 45]
    assert tests[0]["line"]["slope"] < 0
    assert tests[0]["selection_reason"].startswith("most_independent_tests")


def test_more_touches_outrank_an_earlier_higher_horizontal_resistance_without_using_close():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=100))
    bars = pd.DataFrame({"date": dates, "high": 2.0, "low": 1.5, "close": 1.9})
    points = [(5, 4.5), (25, 4.5), (45, 3.9), (65, 3.5), (85, 3.1)]
    pivots = [Pivot(i, str(dates[i].date()), price, "high", i + 2,
                    str(dates[i + 2].date()), "3ATR|8%|neckline_test_2ATR")
              for i, price in points]
    for i, price in points:
        bars.loc[i, "high"] = price
    pivots.extend(Pivot(i, str(dates[i].date()), 1.5, "low", i + 2,
                        str(dates[i + 2].date()), "3ATR|8%")
                  for i in (15, 35, 55, 75, 95))
    trough = Pivot(98, str(dates[98].date()), 1.4, "low", 99,
                   str(dates[99].date()), "3ATR|8%")
    pivots.append(trough)
    atr_values = np.full(len(dates), 0.5)

    ranked = _independent_neckline_tests(
        pivots, trough, trough, atr_values, dates, 0.01, "1d", bars=bars,
        return_candidates=True)
    changed_close = bars.assign(close=np.linspace(0.1, 10.0, len(bars)))
    reranked = _independent_neckline_tests(
        pivots, trough, trough, atr_values, dates, 0.01, "1d", bars=changed_close,
        return_candidates=True)

    assert ranked[0]["selection_rank"] == 1
    assert len(ranked[0]["tests"]) >= 3
    assert ranked[0]["line"]["slope"] < 0
    assert any(len(item["tests"]) == 2 and item["line"]["slope"] == 0 for item in ranked)
    assert [(item["score"], item["line"]) for item in ranked] == [
        (item["score"], item["line"]) for item in reranked]


def test_two_nearby_daily_highs_are_one_touch_not_two_independent_tests():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=40))
    bars = pd.DataFrame({"date": dates, "high": 2.0, "low": 1.5, "close": 1.9})
    pivots = [Pivot(i, str(dates[i].date()), 3.0, "high", i + 1,
                    str(dates[i + 1].date()), "3ATR|8%|neckline_test_2ATR")
              for i in (5, 8)]
    pivots.extend([Pivot(i, str(dates[i].date()), 1.5, "low", i + 1,
                         str(dates[i + 1].date()), "3ATR|8%") for i in (15, 25)])
    trough = Pivot(30, str(dates[30].date()), 1.4, "low", 31,
                   str(dates[31].date()), "3ATR|8%")
    pivots.append(trough)

    tests = _independent_neckline_tests(pivots, trough, trough, np.full(len(dates), 0.5),
                                        dates, 0.01, "1d", bars=bars)

    assert tests == []


def test_neckline_candidate_is_rejected_when_any_interval_high_crosses_the_line():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=45))
    bars = pd.DataFrame({"date": dates, "high": 3.0, "low": 1.5, "close": 2.0})
    left = Pivot(5, str(dates[5].date()), 3.0, "high", 7,
                 str(dates[7].date()), "3ATR|8%|neckline_test_2ATR")
    right = Pivot(35, str(dates[35].date()), 3.0, "high", 37,
                  str(dates[37].date()), "3ATR|8%|neckline_test_2ATR")
    bars.loc[5, "high"] = bars.loc[35, "high"] = 3.0
    bars.loc[20, "high"] = 3.4
    trough = Pivot(40, str(dates[40].date()), 1.5, "low", 42,
                   str(dates[42].date()), "3ATR|8%")

    tests = _independent_neckline_tests(
        [left, right, trough], trough, trough, np.full(len(dates), 0.5), dates,
        0.01, "1d", bars=bars)

    assert tests == []


def test_anchor_only_neckline_coverage_ignores_intermediate_high_but_rejects_crossed_anchor():
    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=45))
    bars = pd.DataFrame({"date": dates, "high": 2.6, "low": 1.5, "close": 2.0})
    left = Pivot(5, str(dates[5].date()), 3.0, "high", 7,
                 str(dates[7].date()), "3ATR|8%|neckline_test_2ATR")
    right = Pivot(35, str(dates[35].date()), 3.0, "high", 37,
                  str(dates[37].date()), "3ATR|8%|neckline_test_2ATR")
    bars.loc[[5, 35], "high"] = 3.0
    bars.loc[20, "high"] = 3.4  # Not a selected structural anchor.
    trough = Pivot(40, str(dates[40].date()), 2.2, "low", 42,
                   str(dates[42].date()), "3ATR|8%")
    separating_low = Pivot(20, str(dates[20].date()), 2.0, "low", 22,
                           str(dates[22].date()), "3ATR|8%")
    atr_values = np.full(len(dates), 0.5)

    accepted = _independent_neckline_tests(
        [left, right, separating_low, trough], trough, trough, atr_values,
        dates, 0.01, "1d", bars=bars, return_candidates=True,
        coverage_mode="structure_anchors_only", structure_anchors=[trough])
    rejected = _independent_neckline_tests(
        [left, right, separating_low, trough], trough, trough, atr_values,
        dates, 0.01, "1d", bars=bars, return_candidates=True,
        coverage_mode="structure_anchors_only",
        structure_anchors=[trough, Pivot(25, str(dates[25].date()), 3.4, "high", 27,
                                         str(dates[27].date()), "3ATR|8%")])

    assert accepted
    assert accepted[0]["line"]["slope"] == 0
    assert rejected == []


def test_structure_anchor_coverage_ignores_non_anchor_wicks_and_rejects_high_anchor():
    from caisen.main_bottoms import _neckline_structure_anchor_coverage

    dates = pd.bdate_range("2022-01-03", periods=30)
    anchors = [Pivot(5, str(dates[5].date()), 2.2, "low", 7,
                     str(dates[7].date()), "2ATR|5%"),
               Pivot(25, str(dates[25].date()), 2.4, "low", 27,
                     str(dates[27].date()), "2ATR|5%")]
    assert _neckline_structure_anchor_coverage(
        {"slope": 0.0, "intercept": 3.0}, anchors, np.full(30, 0.5), 0.01)["passed"]
    anchors.append(Pivot(15, str(dates[15].date()), 3.4, "high", 17,
                         str(dates[17].date()), "2ATR|5%"))
    result = _neckline_structure_anchor_coverage(
        {"slope": 0.0, "intercept": 3.0}, anchors, np.full(30, 0.5), 0.01)
    assert not result["passed"]
    assert result["worst_index"] == 15


def test_boundary_pair_cache_keeps_anchor_valid_lines_even_if_interval_coverage_fails():
    from caisen.main_bottoms import _PotentialNecklinePairCache

    dates = pd.bdate_range("2022-01-03", periods=45)
    left = Pivot(5, str(dates[5].date()), 3.0, "high", 7,
                 str(dates[7].date()), "3ATR|neckline_test_2ATR")
    right = Pivot(35, str(dates[35].date()), 3.0, "high", 37,
                  str(dates[37].date()), "3ATR|neckline_test_2ATR")
    highs = np.full(len(dates), 2.6)
    highs[[5, 35]] = 3.0
    highs[20] = 3.4
    cache = _PotentialNecklinePairCache()
    cache.add([left, right], np.full(len(dates), 0.5), highs,
              np.full(len(dates), 0.25), 0.01, "1d")

    assert cache.matching_pairs([left, right], 10) == []
    anchor_only = cache.matching_pairs([left, right], 10, anchor_only=True)
    assert anchor_only == [(left, right)]


def test_line_coverage_prefix_index_matches_brute_force_for_each_known_prefix():
    from caisen.main_bottoms import _line_coverage_prefixes

    rng = np.random.default_rng(731)
    highs = rng.uniform(2.0, 4.0, size=64)
    tolerances = rng.uniform(0.01, 0.2, size=64)
    highs[[9, 28]] = np.nan
    starts, ends = 4, [12, 31, 58]
    actual = _line_coverage_prefixes(highs, tolerances, -0.003, 3.5, starts, ends)

    for end in ends:
        indexes = np.arange(starts, end + 1)
        excess = highs[indexes] - (-0.003 * indexes + 3.5) - tolerances[indexes]
        finite = np.isfinite(excess)
        expected_maximum = max(0.0, float(np.max(excess[finite])))
        expected_index = int(indexes[finite][np.argmax(excess[finite])])
        assert actual[end] == (expected_maximum, expected_index)


def test_line_coverage_cache_reuses_old_prefixes_when_bars_are_appended():
    from collections import OrderedDict
    from caisen.main_bottoms import _line_coverage_prefixes

    rng = np.random.default_rng(808)
    highs = rng.uniform(2.0, 4.0, size=64)
    tolerances = rng.uniform(0.01, 0.2, size=64)
    cache = OrderedDict()
    first = _line_coverage_prefixes(highs[:32], tolerances[:32], -0.003, 3.5,
                                    4, [12, 31], cache=cache)
    second = _line_coverage_prefixes(highs, tolerances, -0.003, 3.5,
                                     4, [12, 31, 58], cache=cache)

    assert second[12] == first[12]
    assert second[31] == first[31]
    for end in (12, 31, 58):
        indexes = np.arange(4, end + 1)
        excess = highs[indexes] - (-0.003 * indexes + 3.5) - tolerances[indexes]
        assert second[end] == (max(0.0, float(np.max(excess))),
                               int(indexes[np.argmax(excess)]))
    assert len(cache) == 1
    assert set(cache[next(iter(cache))]["checkpoints"]) == {12, 31, 58}


def test_touch_interval_minima_reuse_unchanged_intervals_and_apply_new_lows(monkeypatch):
    from caisen import main_bottoms as engine

    low_indices = np.asarray([15, 20, 25, 40], dtype=int)
    low_prices = np.asarray([2.0, 1.0, 3.0, 0.5], dtype=float)
    range_index = engine._build_range_minimum_index(low_prices)
    calls = []
    original = engine._range_minimum

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_range_minimum", counted)
    first, state = engine._touch_interval_minima([10, 30], low_indices[:3],
                                                  low_prices[:3], range_index)
    extended, extended_state = engine._touch_interval_minima(
        [10, 30, 50], low_indices, low_prices, range_index, previous=state)
    changed, _ = engine._touch_interval_minima(
        [10, 30, 50], low_indices, np.asarray([2.0, 0.4, 3.0, 0.5]),
        range_index, previous=extended_state)

    assert first == (1.0,)
    assert extended == (1.0, 0.5)
    assert changed == (0.4, 0.5)
    assert len(calls) == 2


def test_touch_gap_range_index_matches_open_low_intervals_for_all_touch_pairs():
    from caisen.main_bottoms import (
        _build_range_minimum_index, _build_touch_gap_minimum_index,
        _range_minimum, _touch_gap_minimum,
    )

    touches = [10, 20, 30, 40]
    low_by_index = {12: 5.0, 20: 1.0, 25: 2.0, 30: 4.0, 35: 0.5}
    interval_minima = [5.0, 2.0, 0.5]
    index = _build_touch_gap_minimum_index(touches, interval_minima, low_by_index)
    all_low_indexes = np.asarray(sorted(low_by_index), dtype=int)
    all_low_prices = np.asarray([low_by_index[value] for value in all_low_indexes])
    direct = _build_range_minimum_index(all_low_prices)

    for left in range(len(touches) - 1):
        for right in range(left + 1, len(touches)):
            start = np.searchsorted(all_low_indexes, touches[left], side='right')
            end = np.searchsorted(all_low_indexes, touches[right], side='left')
            expected = _range_minimum(direct, int(start), int(end))
            assert _touch_gap_minimum(index, left, right) == expected


def test_neckline_coverage_is_scanned_once_per_anchor_pair(monkeypatch):
    from caisen import main_bottoms as engine

    dates = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=60))
    bars = pd.DataFrame({"date": dates, "high": 2.5, "low": 1.7, "close": 2.0})
    touch_indexes = (5, 25, 45)
    touches = [Pivot(index, str(dates[index].date()), 3.0, "high", index + 2,
                     str(dates[index + 2].date()), "3ATR|8%|neckline_test_2ATR")
               for index in touch_indexes]
    for index in touch_indexes:
        bars.loc[index, "high"] = 3.0
    lows = [Pivot(index, str(dates[index].date()), 1.7, "low", index + 2,
                  str(dates[index + 2].date()), "3ATR|8%") for index in (15, 35)]
    trough = Pivot(55, str(dates[55].date()), 1.5, "low", 57,
                   str(dates[57].date()), "3ATR|8%")
    calls = []
    original = engine._line_coverage_prefixes

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_line_coverage_prefixes", counted)
    ranked = engine._independent_neckline_tests(
        touches + lows + [trough], trough, trough, np.full(len(dates), 0.5),
        dates, 0.01, "1d", bars=bars, return_candidates=True)

    assert ranked
    assert len(calls) == 3


def test_triangle_apex_uses_converging_upper_and_lower_boundaries():
    upper = {"slope": -0.01, "intercept": 5.0}
    lower = {"slope": 0.01, "intercept": 1.0}
    assert _triangle_apex_index(upper, lower) == 200.0
    assert _triangle_apex_index(lower, upper) is None


def test_main_candidates_can_skip_non_destructive_internal_swings():
    def pivot(index, price, kind):
        date = pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=index)
        return Pivot(index, str(date.date()), price, kind, index + 1,
                     str((date + pd.Timedelta(weeks=1)).date()), "3ATR|8%")

    swings = [
        pivot(0, 3.1, "low"), pivot(2, 4.8, "high"), pivot(4, 2.8, "low"),
        pivot(5, 4.5, "high"), pivot(7, 2.0, "low"), pivot(9, 4.6, "high"),
        pivot(11, 2.6, "low"), pivot(12, 4.7, "high"), pivot(14, 3.0, "low"),
        pivot(16, 5.0, "high"), pivot(18, 2.0, "low"), pivot(20, 4.9, "high"),
        pivot(22, 2.25, "low"), pivot(24, 4.6, "high"), pivot(26, 2.4, "low"),
        pivot(28, 4.4, "high"), pivot(30, 2.6, "low"), pivot(32, 4.0, "high"),
        pivot(34, 2.8, "low"),
    ]
    ihs_candidates = _structure_candidates(swings[:9], "1w")
    triangle_candidates = _structure_candidates(swings, "1w")
    ihs = [anchors for family, anchors in ihs_candidates if family == "inverse_head_shoulders"]
    triangles = [anchors for family, anchors in triangle_candidates if family == "triangle_bottom"]

    assert any([point.index for point in anchors] == [4, 7, 11] for anchors in ihs)
    assert not any([point.index for point in anchors] == [0, 7, 14] for anchors in ihs)
    assert any([point.index for point in anchors] == [18, 20, 26, 28, 34] for anchors in triangles)
    selected_triangle = next(anchors for anchors in triangles
                             if [point.index for point in anchors] == [18, 20, 26, 28, 34])
    assert _triangle_boundaries_intact(selected_triangle, swings)
    broken_swings = [Pivot(point.index, point.date,
                           1.9 if point.index == 22 and point.kind == "low" else point.price,
                           point.kind, point.confirmed_at, point.confirmed_date, point.scale)
                     for point in swings]
    assert not _triangle_boundaries_intact(selected_triangle, broken_swings)
    selected_ihs = next(anchors for anchors in ihs
                        if [point.index for point in anchors] == [4, 7, 11])
    assert _ihs_valley_intact(selected_ihs[0], selected_ihs[1], selected_ihs[2], swings)
    assert not _ihs_right_shoulder_intact(selected_ihs[1], swings[8], swings)
    raw_lows = np.full(35, 4.0)
    raw_lows[0], raw_lows[7], raw_lows[11] = 3.1, 2.0, 2.6
    assert _ihs_valley_intact(selected_ihs[0], selected_ihs[1], selected_ihs[2],
                              swings, low_values=raw_lows)
    raw_lows[8] = 1.8
    assert not _ihs_valley_intact(selected_ihs[0], selected_ihs[1], selected_ihs[2],
                                  swings, low_values=raw_lows)
    head = selected_ihs[1]
    damaged_ihs = swings + [Pivot(6, head.date, head.price - 0.2, "low", 7,
                                  head.confirmed_date, head.scale)]
    assert not _ihs_valley_intact(selected_ihs[0], selected_ihs[1], selected_ihs[2], damaged_ihs)


def test_one_atr_reference_main_events_from_real_review_fixtures_remain_json_safe_and_causal(monkeypatch):
    monkeypatch.setitem(MAIN_RULES["neckline_tests"], "candidate_reversal_atr_multiple", 1.0)
    fixture_root = Path(__file__).parent / "fixtures" / "main_bottoms"
    for source in sorted(fixture_root.glob("R0[12]-*.csv")):
        bars = pd.read_csv(source, parse_dates=["date"])
        found = detect_main_bottoms(bars, code=source.stem.split("-")[1], timeframe="1d")
        extended = detect_main_bottoms(bars.assign(volume=bars.volume.fillna(0)),
                                       code=source.stem.split("-")[1], timeframe="1d")

        assert all(item.pattern_type in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
                   for item in found)
        assert [item.pattern_id for item in found] == [item.pattern_id for item in extended]
        if source.name == "R02-000008.csv":
            assert not any(is_selected(item)
                           and evidence(item)["main_structure"]["signal_eligible"]
                           for item in found)


def test_child_linkage_suppresses_only_same_timeframe_bullish_events_inside_parent_until_breakout():
    parent = Pattern("main", "fixture", "stock", "1d", "w_bottom", "bullish", "confirmed",
                     "2024-02-06", 900, confirmation_date="2026-01-09",
                     evidence=[{"code": "main_structure", "value": {
                         "role": "main",
                         "signal_eligible": True, "first_trough_date": "2024-02-06",
                         "frozen_at": "2024-09-20", "latest_anchor_date": "2024-09-18",
                         "structure_end_date": "2026-01-09", "scale": "5y"}},
                                {"code": "main_selection", "value": {"selected": True}}])
    child = Pattern("flag", "fixture", "stock", "1d", "bull_flag", "bullish", "confirmed",
                    "2025-03-01", 800, confirmation_date="2025-04-01")
    after_breakout = Pattern("later", "fixture", "stock", "1d", "bull_flag", "bullish", "confirmed",
                             "2026-01-12", 950, confirmation_date="2026-02-01")
    weekly = Pattern("weekly", "fixture", "stock", "1w", "bull_flag", "bullish", "confirmed",
                     "2025-03-01", 80, confirmation_date="2025-04-01")
    original = deepcopy(child)

    linked = link_main_bottom_children([parent], [child, after_breakout, weekly])
    by_id = {item.pattern_id: item for item in linked}

    assert by_id["flag"].parent_id == "main"
    assert evidence(by_id["flag"])["execution_suppression"]["suppressed"] is True
    assert by_id["later"].parent_id is None
    assert by_id["weekly"].parent_id is None
    assert child.parent_id == original.parent_id
    assert "execution_suppression" not in evidence(child)


def test_child_linkage_does_not_backdate_suppression_before_parent_is_known():
    parent = Pattern("main", "fixture", "stock", "1d", "w_bottom", "bullish", "forming",
                     "2024-02-06", None, last_observed_date="2026-01-09",
                     evidence=[{"code": "main_structure", "value": {
                         "role": "main", "signal_eligible": False,
                         "first_trough_date": "2024-02-06", "frozen_at": "2025-10-01",
                         "structure_end_date": "2026-01-09"}},
                                {"code": "main_selection", "value": {"selected": True}}])
    earlier_child = Pattern("early", "fixture", "stock", "1d", "bull_flag", "bullish", "confirmed",
                            "2025-05-01", 600, confirmation_date="2025-06-01")
    forming_child = Pattern("forming", "fixture", "stock", "1d", "bull_flag", "bullish", "forming",
                            "2025-09-01", None, last_observed_date="2025-10-15")

    linked = link_main_bottom_children([parent], [earlier_child, forming_child])
    by_id = {item.pattern_id: item for item in linked}

    assert by_id["early"].parent_id is None
    assert by_id["forming"].parent_id == "main"


def test_static_recognizer_suppresses_inner_pattern_by_forming_outer_w(monkeypatch):
    import caisen.main_bottoms as main_bottoms

    dates = pd.date_range("2024-01-05", periods=105, freq="W-FRI")
    frame = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                          "low": 9.8, "close": 10.0, "volume": 1000.0})
    parent = Pattern(
        "outer-w", "600108", "stock", "1w", "w_bottom", "bullish", "forming",
        "2024-01-05", None, last_observed_date="2025-01-03",
        key_points={"pivot_1_date": "2024-01-05", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-06-07", "pivot_2_kind": "low",
                    "pivot_scale": "5atr_12pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": False, "pivot_scale": "5atr_12pct",
            "first_trough_date": "2024-01-05", "structure_end_date": "2025-01-03"}},
                  {"code": "neckline_selection", "value": {
                      "structure_known_at": "2024-06-07"}}])
    child = Pattern(
        "inner-triangle", "600108", "stock", "1w", "triangle_bottom", "bullish", "confirmed",
        "2024-03-01", 50, confirmation_date="2024-12-06", last_observed_date="2025-01-03",
        key_points={"pivot_1_date": "2024-03-01", "pivot_1_kind": "low",
                    "pivot_2_date": "2024-06-07", "pivot_2_kind": "high",
                    "pivot_3_date": "2024-09-06", "pivot_3_kind": "low",
                    "pivot_4_date": "2024-12-06", "pivot_4_kind": "high",
                    "pivot_scale": "2atr_5pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True, "pivot_scale": "2atr_5pct"}}])
    scales = {"2atr_5pct": [], "3atr_8pct": [], "5atr_12pct": []}
    parent_pivots = [Pivot(0, "2024-01-05", 8.0, "low", 1, "2024-01-12", "5ATR|12%"),
                     Pivot(52, "2025-01-03", 8.1, "low", 53, "2025-01-10", "5ATR|12%")]
    child_pivots = [Pivot(8, "2024-03-01", 8.0, "low", 9, "2024-03-08", "2ATR|5%"),
                    Pivot(22, "2024-06-07", 10.0, "high", 23, "2024-06-14", "2ATR|5%"),
                    Pivot(35, "2024-09-06", 8.1, "low", 36, "2024-09-13", "2ATR|5%"),
                    Pivot(48, "2024-12-06", 10.1, "high", 49, "2024-12-13", "2ATR|5%")]
    scales["2atr_5pct"] = child_pivots
    scales["5atr_12pct"] = parent_pivots
    sequences = {"2atr_5pct": [("triangle_bottom", child_pivots)],
                 "3atr_8pct": [], "5atr_12pct": [("w_bottom", parent_pivots)]}

    monkeypatch.setattr(main_bottoms, "detect_multiscale_pivots",
                        lambda *_args, **_kwargs: scales)
    monkeypatch.setattr(main_bottoms, "_detect_neckline_test_pivots",
                        lambda *_args, **_kwargs: [])
    scale_names = {"2ATR|5%": "2atr_5pct", "3ATR|8%": "3atr_8pct",
                   "5ATR|12%": "5atr_12pct"}
    monkeypatch.setattr(main_bottoms, "_structure_candidates",
                        lambda pivots, *_args, **_kwargs: sequences[scale_names[pivots[0].scale]]
                        if pivots else [])
    monkeypatch.setattr(main_bottoms, "_make_main_pattern",
                        lambda _frame, _code, _kind, _tf, family, *_args, **_kwargs:
                        deepcopy(parent if family == "w_bottom" else child))

    events = detect_main_bottoms(frame, code="600108", kind="stock", timeframe="1w")
    by_id = {item.pattern_id: item for item in events}

    assert evidence(by_id["inner-triangle"])["main_structure"]["role"] == "internal"


@pytest.mark.parametrize(("state", "role", "selected"), [
    ("target_reached", "main", True),
    ("confirmed", "internal", True),
    ("confirmed", "main", False),
])
def test_ineligible_or_terminal_main_event_cannot_suppress_a_legacy_child(state, role, selected):
    parent = Pattern("parent", "fixture", "stock", "1d", "w_bottom", "bullish", state,
                     "2024-01-01", 900, confirmation_date="2024-10-01",
                     last_observed_date="2025-01-01",
                     evidence=[{"code": "main_structure", "value": {
                         "role": role, "signal_eligible": True,
                         "first_trough_date": "2024-01-01", "structure_end_date": "2025-01-01"}},
                               {"code": "main_selection", "value": {"selected": selected}}])
    child = Pattern("child", "fixture", "stock", "1d", "bull_flag", "bullish", "confirmed",
                    "2024-03-01", 600, confirmation_date="2024-04-01",
                    key_points={"pivot_1_date": "2024-03-01", "pivot_1_kind": "low",
                                "pivot_2_date": "2024-04-01", "pivot_2_kind": "high"})

    linked = link_main_bottom_children([parent], [child])
    linked_child = next(item for item in linked if item.pattern_id == "child")

    assert linked_child.parent_id is None
    assert "execution_suppression" not in evidence(linked_child)


def test_materially_rising_neckline_is_rejected_and_small_rise_is_flattened():
    from caisen.main_bottoms import _qualify_neckline_slope

    assert _qualify_neckline_slope({"slope": 0.0, "intercept": 10.0}, 10, 100) == {
        "slope": 0.0, "intercept": 10.0
    }
    flattened = _qualify_neckline_slope({"slope": 0.004, "intercept": 10.0}, 0, 100)
    assert flattened == {"slope": 0.0, "intercept": 10.2}
    assert _qualify_neckline_slope({"slope": 0.01, "intercept": 10.0}, 0, 100) is None
