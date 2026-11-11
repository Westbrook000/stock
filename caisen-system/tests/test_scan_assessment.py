import pandas as pd
import pytest
from pathlib import Path

import caisen.assessment as assessment
from caisen.assessment import ASSESSMENT_VERSION, assess_pattern, assess_scan
from caisen.main_bottoms import detect_main_bottoms
from test_main_bottoms import synthetic_w_bars


def bars(closes, dates=None):
    dates = dates or pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "date": dates,
        "high": [value + 0.2 for value in closes],
        "low": [value - 0.2 for value in closes],
        "close": closes,
    })


def pattern(timeframe="1d", *, state="forming", evidence=None, kind="w_bottom", line=None):
    return {
        "pattern_id": f"p-{timeframe}", "code": "000001", "kind": "stock",
        "timeframe": timeframe, "pattern_type": kind, "direction": "bullish",
        "state": state, "formed_at": "2026-01-01", "confirmed_at": 1,
        "confirmation_date": None, "last_observed_date": "2026-01-03",
        "key_points": {"pivot_1_index": 0},
        "neckline": line or {"slope": 0.0, "intercept": 10.0},
        "evidence": evidence or [{"code": "neckline_selection", "value": {"freeze_date": "2026-01-02"}}],
    }


def unselected_main_event(*, decided_at="2026-01-03", candidate_id="candidate-a", close=10.5):
    """Small immutable legacy event for assessment proof-gate tests."""
    return pattern(state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {
            "date": "2026-01-02", "index": 1, "close": close}},
        {"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True,
            "frozen_at": "2026-01-01", "support_lower": 9.0}},
        {"code": "neckline_selection", "value": {
            "selected_candidate_id": candidate_id,
            "line_frozen_at": "2026-01-01", "structure_known_at": "2026-01-01"}},
        {"code": "main_selection", "value": {
            "selected": False, "decided_at": decided_at}},
    ], line={"slope": 0.0, "intercept": 10.0})


def events_for(candidate, *, state="forming", entry_date=None, extra_state=None):
    from caisen.pattern_instances import PatternInstanceLedger
    ledger = PatternInstanceLedger()
    ledger.append(candidate, "forming", "2026-01-01")
    if entry_date:
        ledger.append(candidate, "confirmed", entry_date, entry_eligible=True)
    if extra_state:
        ledger.append(candidate, extra_state, "2026-01-03")
    return [event.to_dict() for event in ledger.events]


def test_existing_breakout_is_buy_only_on_first_valid_breakout_date():
    candidate = pattern(state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {"date": "2026-01-02", "index": 1, "close": 10.5}}
    ])
    data = bars([10.0, 10.5, 9.8])

    on_breakout = assess_pattern(candidate, data, as_of_date="2026-01-02")
    retest = assess_pattern(candidate, data, as_of_date="2026-01-03")

    assert on_breakout["status"] == "BUY"
    assert retest["status"] == "FOCUS_NECKLINE"
    assert retest["evidence"]["first_break_close"] == 10.5


@pytest.mark.parametrize("close,expected", [(9.201, "FOCUS_NECKLINE"), (9.2, "NONE")])
def test_post_breakout_focus_requires_drawdown_strictly_below_eight_percent(close, expected):
    candidate = pattern(state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {"date": "2026-01-02", "index": 1, "close": 10.0}}
    ])

    result = assess_pattern(candidate, bars([10.0, 10.0, close]), as_of_date="2026-01-03")

    assert result["status"] == expected


def test_classic_event_uses_saved_confirmation_date_and_cutoff_bar_close():
    candidate = pattern(state="confirmed", evidence=[])
    candidate.update({"confirmation_date": "2026-01-02", "confirmed_at": 1})

    result = assess_pattern(candidate, bars([10.0, 10.5, 9.8]), as_of_date="2026-01-03")

    assert result["status"] == "FOCUS_NECKLINE"
    assert result["evidence"]["first_break_close"] == 10.5


def test_frozen_unbroken_pattern_is_focus_within_one_previous_bar_atr():
    candidate = pattern()
    data = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4, freq="D"),
        "high": [10.2, 10.2, 10.2, 10.2], "low": [9.8, 9.8, 9.8, 9.8],
        "close": [10.0, 10.0, 10.0, 9.7],
    })

    result = assess_pattern(candidate, data, as_of_date="2026-01-04")

    assert result["status"] == "FOCUS_NECKLINE"
    assert result["evidence"]["boundary"] == 10.0


def test_bull_flag_uses_upper_line_and_missing_old_snapshot_is_explicit():
    flag = pattern(kind="bull_flag", line={"slope": -0.1, "intercept": 10.0})
    data = bars([10.0, 10.0, 9.7])

    result = assess_pattern(flag, data, as_of_date="2026-01-03")
    assert result["status"] == "FOCUS_NECKLINE"
    assert result["evidence"]["boundary"] == pytest.approx(9.8)
    result = assess_pattern(flag, None, as_of_date="2026-01-03")
    assert result["status"] == "INSUFFICIENT_EVIDENCE"


def test_any_timeframe_buy_takes_priority_over_focus():
    daily = pattern("1d", state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {"date": "2026-01-02", "index": 1, "close": 10.5}}
    ])
    weekly = pattern("1w")

    result = assess_scan([daily, weekly], {"1d": bars([10.0, 10.5, 9.9]), "1w": bars([10.0, 9.9, 9.9])},
                         as_of_date="2026-01-02",
                         instance_events_by_timeframe={"1d":events_for(daily,entry_date="2026-01-02")})

    assert result["status"] == "BUY"
    assert result["selected_timeframe"] == "1d"
    assert result["timeframes"]["1w"]["status"] == "FOCUS_NECKLINE"


def test_daily_ledger_confirmation_is_buy_once_then_focus(monkeypatch):
    from caisen.pattern_instances import PatternInstanceLedger

    candidate = pattern(state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {"date": "2026-01-02", "close": 10.5}}
    ])
    ledger = PatternInstanceLedger()
    ledger.append(candidate, "forming", "2026-01-02")
    entry = ledger.append(candidate, "confirmed", "2026-01-02", entry_eligible=True)
    focus = ledger.append(candidate, "focus", "2026-01-03")
    data = {"1d": bars([10.0, 10.5, 10.4])}

    on_entry = assess_scan([candidate], data, "2026-01-02",
                           instance_events_by_timeframe={"1d": [e.to_dict() for e in ledger.events]})
    after_entry = assess_scan([candidate], data, "2026-01-03",
                              instance_events_by_timeframe={"1d": [e.to_dict() for e in ledger.events]})

    assert entry.entry_eligible is True and focus.entry_eligible is False
    assert on_entry["status"] == "BUY"
    assert on_entry["timeframes"]["1d"]["event_id"] == entry.event_id
    assert after_entry["status"] == "FOCUS_NECKLINE"
    assert after_entry["timeframes"]["1d"]["event_id"] == focus.event_id


def test_supported_daily_pattern_without_shared_events_cannot_recompute_buy():
    candidate = pattern(state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {"date": "2026-01-02", "close": 10.5}}
    ])
    result = assess_scan([candidate], {"1d": bars([10.0, 10.5, 10.4])}, "2026-01-02",
                         instance_events_by_timeframe={"1d": []})

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["timeframes"]["1d"]["reason"] == "shared_instance_event_missing"


def test_scan_prepares_ohlc_and_atr_once_per_timeframe(monkeypatch):
    patterns = [pattern("1d"), pattern("1d"), pattern("1w"), pattern("1w")]
    for number, candidate in enumerate(patterns):
        candidate["pattern_id"] = f"p{number}"
    timeframe_bars = {"1d": bars([10.0, 10.0, 9.8]), "1w": bars([10.0, 10.0, 9.8])}
    expected = {}
    for candidate in patterns:
        expected[candidate["pattern_id"]] = assess_pattern(
            candidate, timeframe_bars[candidate["timeframe"]], "2026-01-03")

    calls = {"bars": 0, "atr": 0}
    original_bars, original_atr = assessment._asof_bars, assessment.atr

    def counted_bars(*args, **kwargs):
        calls["bars"] += 1
        return original_bars(*args, **kwargs)

    def counted_atr(*args, **kwargs):
        calls["atr"] += 1
        return original_atr(*args, **kwargs)

    monkeypatch.setattr(assessment, "_asof_bars", counted_bars)
    monkeypatch.setattr(assessment, "atr", counted_atr)
    result = assess_scan(patterns, timeframe_bars, "2026-01-03",
                         instance_events_by_timeframe={
                             "1d":[event for candidate in patterns[:2]
                                   for event in events_for(candidate)]})

    assert calls == {"bars": 2, "atr": 2}
    assert result["timeframes"]["1d"] == {**expected["p0"], "pattern_id": "p0"}
    assert result["timeframes"]["1w"] == {**expected["p2"], "pattern_id": "p2"}


def test_missing_weekly_snapshot_prevents_wait_unless_buy_exists():
    daily = pattern("1d")
    weekly = pattern("1w")
    data = {"1d": bars([10.0, 8.0, 8.0])}

    incomplete = assess_scan([daily, weekly], data, "2026-01-03")
    buying = pattern("1d", state="confirmed", evidence=[
        {"code": "neckline_breakout", "value": {"date": "2026-01-02", "close": 10.5}}
    ])
    complete_action = assess_scan([buying, weekly], {"1d": bars([10.0, 10.5, 10.4])}, "2026-01-03",
                                  instance_events_by_timeframe={
                                      "1d":events_for(buying,entry_date="2026-01-02",
                                                      extra_state="focus")})

    assert ASSESSMENT_VERSION == "scan-assessment-v2"
    assert incomplete["status"] == "INSUFFICIENT_EVIDENCE"
    assert complete_action["status"] == "FOCUS_NECKLINE"


@pytest.mark.parametrize("direction,kind", [("bearish", "head_shoulders"), ("bullish", "m_top")])
def test_scan_rejects_non_buy_setups(direction, kind):
    candidate = pattern(kind=kind)
    candidate["direction"] = direction

    result = assess_scan([candidate], {"1d": bars([10.0, 10.0, 9.8])}, "2026-01-03")

    assert result["status"] == "NONE"
    assert result["timeframes"]["1d"]["reason"] == "non_buy_setup"


@pytest.mark.parametrize("mutation,reason", [
    (lambda item: item.update(flags=["ambiguous_structure"]), "ambiguous_structure"),
    (lambda item: item.update(parent_id="main-id"), "execution_suppressed"),
    (lambda item: item["evidence"].append({"code": "execution_suppression", "value": {"suppressed": True}}),
     "execution_suppressed"),
])
def test_assessment_rejects_suppressed_or_ambiguous_setups(mutation, reason):
    candidate = pattern()
    mutation(candidate)

    result = assess_pattern(candidate, bars([10.0, 10.0, 9.8]), as_of_date="2026-01-03")

    assert result["status"] == "NONE"
    assert result["reason"] == reason


def test_forming_shape_requires_freeze_and_known_at_cutoff():
    candidate = pattern(evidence=[])
    candidate["confirmed_at"] = 2
    data = bars([10.0, 10.0, 9.8])

    before_freeze = assess_pattern(candidate, data, as_of_date="2026-01-02")
    at_freeze = assess_pattern(candidate, data, as_of_date="2026-01-03")

    assert before_freeze["status"] == "INSUFFICIENT_EVIDENCE"
    assert before_freeze["reason"] == "formation_not_frozen_by_cutoff"
    assert at_freeze["status"] == "FOCUS_NECKLINE"


def test_main_bottom_primary_and_boundary_use_actual_detector_evidence():
    data = synthetic_w_bars().iloc[:600].copy()
    patterns = detect_main_bottoms(data.iloc[100:].copy(), code="fixture", timeframe="1d")
    selected = next(item for item in patterns if next(
        evidence["value"] for evidence in item.evidence if evidence["code"] == "main_selection")["selected"])
    payload = selected.to_dict()
    main = next(item["value"] for item in payload["evidence"] if item["code"] == "main_structure")

    result = assess_pattern(payload, data, as_of_date=selected.last_observed_date)

    assert result["evidence"]["boundary"] == pytest.approx(main["neckline_at_observed"])
    assert result["status"] in {"NONE", "FOCUS_NECKLINE"}


def test_real_fixture_breakout_is_buy_only_at_or_after_its_saved_event():
    data = pd.read_csv(Path(__file__).parent / "fixtures" / "main_bottoms" / "R02-000008.csv")
    data["date"] = pd.to_datetime(data.date)
    current = detect_main_bottoms(data, code="000008", timeframe="1d")
    event_bars = data[data.date <= "2025-11-11"].copy()
    as_of = detect_main_bottoms(event_bars, code="000008", timeframe="1d")
    event_primary = next(
        item for item in as_of
        if item.pattern_type == "w_bottom"
        and item.confirmation_date == "2025-11-11"
        and next(evidence["value"] for evidence in item.evidence
                 if evidence["code"] == "main_selection")["selected"]
    )

    first_break = assess_pattern(event_primary, event_bars,
                                 as_of_date=event_primary.last_observed_date)

    assert first_break["status"] == "BUY"
    assert first_break["reason"] == "first_valid_breakout_day"

    retest_index = int(data.index[data.date == "2025-11-26"][0])
    retest_bars = data.iloc[:retest_index + 1].copy()
    retrace = assess_pattern(event_primary, retest_bars, as_of_date="2025-11-26")
    assert retrace["status"] == "NONE"
    assert retrace["reason"] in {"post_breakout_drawdown_at_least_8pct",
                                  "post_breakout_structure_invalidated"}
    assert retrace["evidence"]["first_break_date"] == "2025-11-11"
    assert retrace["evidence"]["first_break_close"] == 3.30
    invalidated = event_primary.to_dict()
    invalidated["state"] = "invalidated"
    assert assess_pattern(invalidated, retest_bars, as_of_date="2025-11-26")["status"] == "NONE"

    assert not any(
        item.confirmation_date == "2025-11-24"
        and next(evidence["value"] for evidence in item.evidence
                 if evidence["code"] == "main_selection")["selected"]
        for item in current
    )

    suppressed = event_primary.to_dict()
    suppressed["parent_id"] = "parent-main"
    assert assess_pattern(suppressed, event_bars, event_primary.last_observed_date)["status"] == "NONE"


def test_unselected_event_requires_historical_primary_proof(monkeypatch):
    candidate = unselected_main_event()
    data = bars([10.0, 10.5])
    calls = []

    def proof(payload, frame, event_date, cache, tick_size=None, background_bars=None):
        candidate_id = next(item["value"]["selected_candidate_id"] for item in payload["evidence"]
                            if item["code"] == "neckline_selection")
        calls.append(candidate_id)
        verified = candidate_id == "candidate-a"
        return verified, ({"verified": True, "basis": "test replay"} if verified else None)

    monkeypatch.setattr(assessment, "_historical_primary_proof", proof)
    result = assess_pattern(candidate, data, as_of_date="2026-01-02")
    assert result["status"] == "BUY"
    assert result["evidence"]["historical_primary_proof"]["verified"] is True

    wrong_candidate = dict(candidate)
    wrong_candidate["evidence"] = [dict(item, value=dict(item["value"])) for item in candidate["evidence"]]
    selection = next(evidence["value"] for evidence in wrong_candidate["evidence"]
                     if evidence["code"] == "neckline_selection")
    selection["selected_candidate_id"] = "unmatched-candidate"
    missing_proof = assess_pattern(wrong_candidate, data, as_of_date="2026-01-02")
    assert missing_proof["status"] == "INSUFFICIENT_EVIDENCE"
    assert missing_proof["reason"] == "historical_primary_not_proven"
    assert calls == ["candidate-a", "unmatched-candidate"]


def test_late_selected_flag_cannot_replace_breakout_time_primary_proof():
    data = bars([10.0, 10.5, 10.4])
    payload = unselected_main_event(decided_at="2026-01-03")
    selection = next(item["value"] for item in payload["evidence"]
                     if item["code"] == "main_selection")
    selection.update(selected=True, decided_at="2026-01-03")

    post_breakout = assess_pattern(
        payload, data, as_of_date="2026-01-03")

    assert post_breakout["status"] == "FOCUS_NECKLINE"
    assert post_breakout["status"] != "BUY"


def test_unqualified_retrace_does_not_replay_historical_primary(monkeypatch):
    import caisen.main_bottoms as main_bottoms

    data = bars([10.0, 10.5, 9.6])
    candidate = unselected_main_event()
    called = False

    def unexpected_replay(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unqualified retrace should not replay")

    monkeypatch.setattr(main_bottoms, "detect_main_bottoms", unexpected_replay)
    result = assess_pattern(candidate, data, as_of_date="2026-01-03")

    assert result["status"] == "NONE"
    assert called is False


def test_main_internal_or_unselected_events_cannot_be_buy_candidates():
    data = synthetic_w_bars().iloc[:600].copy()
    selected = next(item for item in detect_main_bottoms(data, code="fixture", timeframe="1d")
                    if next(evidence["value"] for evidence in item.evidence
                            if evidence["code"] == "main_selection")["selected"])
    candidate = selected.to_dict()
    selection = next(item["value"] for item in candidate["evidence"] if item["code"] == "main_selection")
    structure = next(item["value"] for item in candidate["evidence"] if item["code"] == "main_structure")
    selection["selected"] = False

    unselected = assess_pattern(candidate, data, as_of_date=selected.last_observed_date)
    selection["selected"] = True
    structure["role"] = "internal"
    internal = assess_pattern(candidate, data, as_of_date=selected.last_observed_date)

    assert unselected["status"] == "NONE"
    assert internal["status"] == "NONE"


def test_future_main_selection_cannot_be_used_at_earlier_cutoff():
    data = synthetic_w_bars().iloc[:600].copy()
    selected = next(item for item in detect_main_bottoms(data, code="fixture", timeframe="1d")
                    if next(evidence["value"] for evidence in item.evidence
                            if evidence["code"] == "main_selection")["selected"])
    candidate = selected.to_dict()
    cutoff = (pd.Timestamp(selected.last_observed_date) - pd.offsets.BDay(5)).date().isoformat()

    result = assess_pattern(candidate, data, as_of_date=cutoff)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["reason"] == "primary_selection_after_cutoff"
