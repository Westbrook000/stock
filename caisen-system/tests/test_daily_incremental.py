from pathlib import Path

import pandas as pd
import numpy as np

from caisen.daily_incremental import (DailyRecognitionSession, generate_daily_signals,
                                     recognize_daily_history)
from caisen.main_bottoms import MAIN_RULES


def test_session_one_atr_reference_is_causal_and_emits_each_confirmation_once(monkeypatch):
    monkeypatch.setitem(MAIN_RULES["neckline_tests"], "candidate_reversal_atr_multiple", 1.0)
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-30"].reset_index(drop=True)
    result = recognize_daily_history(bars, "159718", "etf", tick_size=0.01)
    signals = result["signals"]
    target_anchors = ["2023-08-22", "2024-04-19", "2025-01-13"]
    target_events = [pattern for signal in signals for pattern in signal["patterns"]
                     if pattern["pattern_type"] == "inverse_head_shoulders"
                     and [pattern["key_points"].get(f"pivot_{number}_date")
                          for number in (1, 2, 3)] == target_anchors]

    assert len(target_events) == 1
    assert target_events[0]["confirmation_date"] == "2025-04-01"
    assert any(item["signal_date"] == "2025-05-29"
               and "w_bottom" in item["pattern_types"] for item in signals)
    assert not any(item["signal_date"] == "2025-05-29"
                   and "inverse_head_shoulders" in item["pattern_types"] for item in signals)


def test_session_consumes_159718_same_ihs_boundary_after_earlier_mature_cross():
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-30"].reset_index(drop=True)
    # Match production ETF ingestion explicitly; the bare-frame API fallback
    # uses 0.001 and is a separate tick-sensitivity case.
    session = DailyRecognitionSession("159718", "etf", tick_size=0.01)
    signals = []
    for bar in bars.to_dict("records"):
        signals.extend(session.append_bar(bar))

    assert any(item["signal_date"] == "2025-04-01"
               and "inverse_head_shoulders" in item["pattern_types"] for item in signals)
    assert not any(item["signal_date"] == "2025-05-21"
                   and "inverse_head_shoulders" in item["pattern_types"] for item in signals)
    assert any(item["signal_date"] == "2025-05-29"
               and "w_bottom" in item["pattern_types"] for item in signals)

    dates = bars.date.dt.date.astype(str).tolist()
    april_signal = next(item for item in signals if item["signal_date"] == "2025-04-01"
                        and "inverse_head_shoulders" in item["pattern_types"])
    ihs = next(item for item in april_signal["patterns"]
                if item["pattern_type"] == "inverse_head_shoulders")
    expected_anchors = [ihs["key_points"][f"pivot_{number}_date"] for number in (1, 2, 3)]
    consumed_key = next(key for key in session.consumed_neckline_phases
                        if key[0] == "inverse_head_shoulders"
                        and [dates[int(index)] for index, _kind in key[1]] == expected_anchors)
    earlier_phase = next(phase for phase in session.consumed_neckline_phases[consumed_key]
                         if dates[phase["consumed_index"]] == "2025-04-01")
    assert earlier_phase["line"] == ihs["neckline"]
    assert not any(item["signal_date"] == "2025-05-21"
                   and "inverse_head_shoulders" in item["pattern_types"] for item in signals)



def test_batch_final_day_signal_has_no_entry_session():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-11-06"].reset_index(drop=True)

    signals = generate_daily_signals(bars, "600108", "stock")
    signal = next(item for item in signals if item["signal_date"] == "2024-11-06")

    assert signal["entry_date"] is None
    assert signal["pattern_types"]
    assert signal["pattern_ids"]


def test_batch_trade_start_filters_emissions_without_resetting_history():
    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-10-30"].reset_index(drop=True)

    recent = generate_daily_signals(bars, "516630", "etf", trade_start="2025-01-01")

    assert all(item["signal_date"] >= "2025-01-01" for item in recent)


def test_600108_daily_session_confirms_main_w_at_the_same_october_8_boundary():
    from caisen.api import _decision_patterns_by_timeframe
    from caisen.research_config import RecognitionExperimentConfig, recognition_experiment
    source = Path("/tmp/caisen-structure-boundary-worker-inputs-20261112-v2/data/stocks/600108.csv")
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-10-09"].reset_index(drop=True)
    config = RecognitionExperimentConfig(
        "internal_rebound_coverage_g1_v1", require_internal_rebound_coverage=True)
    emitted = []
    with recognition_experiment(config):
        session = DailyRecognitionSession("600108", "stock", tick_size=0.01)
        for row in bars.to_dict("records"):
            emitted.extend(session.append_bar(row))

    matches = [pattern for signal in emitted if signal["signal_date"] == "2024-10-08"
               for pattern in signal["patterns"]
               if pattern["pattern_type"] == "w_bottom"
               and pattern["key_points"].get("pivot_1_date") == "2024-02-06"
               and pattern["key_points"].get("pivot_2_date") == "2024-09-18"]

    assert len(matches) == 1
    assert matches[0]["confirmation_date"] == "2024-10-08"
    assert np.isclose(matches[0]["neckline"]["slope"] * bars.index[-1]
                      + matches[0]["neckline"]["intercept"], 2.758, atol=0.02)

    event_pattern = matches[0]
    evidence = {item["code"]: item["value"] for item in event_pattern["evidence"]}
    lifecycle = evidence["joint_lifecycle"]
    structure = evidence["main_structure"]
    selection = evidence.get("main_selection")
    assert selection and selection["selected"] is True
    assert selection["primary_instance_id"] == lifecycle["instance_id"]
    assert lifecycle["boundary_version_id"]
    assert structure["selection_at_confirmation"]["date"] == "2024-10-08"
    assert structure["entry_eligible_at_confirmation"] is True
    assert lifecycle["selection_at_confirmation"]["primary_instance_id"] == lifecycle["instance_id"]
    assert lifecycle["entry_eligible_at_confirmation"] is True
    assert lifecycle["entry_eligibility_at_confirmation_status"] == "known"
    assert lifecycle["recognition_context"]["recognition_experiment"]["cache_key"] == \
        list(config.cache_key())

    instance_id = lifecycle["instance_id"]
    active_aliases = [item["pattern"] for item in session.active.values()
                      if item.get("pattern") is not None
                      and next((e["value"].get("instance_id") for e in item["pattern"].evidence
                                if e.get("code") == "joint_lifecycle"), None) == instance_id]
    assert active_aliases
    active = active_aliases[0]
    active_evidence = {item["code"]: item["value"] for item in active.evidence}
    active_lifecycle = active_evidence["joint_lifecycle"]
    assert active_lifecycle["instance_id"] == lifecycle["instance_id"]
    assert active_lifecycle["boundary_version_id"] == lifecycle["boundary_version_id"]
    assert active_evidence["main_selection"]["selected"] is True
    assert active_lifecycle["entry_eligible_at_confirmation"] is True
    assert active_lifecycle["entry_eligible"] is False

    from caisen.schemas import Pattern
    confirmed = Pattern(**event_pattern)
    api_candidates = _decision_patterns_by_timeframe({"1d": [confirmed]})["1d"]
    assert any(pattern.pattern_id == confirmed.pattern_id for pattern in api_candidates)
    next_day_signals = [signal for signal in emitted if signal["signal_date"] == "2024-10-09"]
    assert not any(instance_id in [entry.get("joint_lifecycle", {}).get("instance_id")
                                   for entry in signal.get("patterns", [])]
                   for signal in next_day_signals)
    assert active_lifecycle["entry_eligible_at_confirmation"] is True


def test_main_lifecycle_selection_history_is_known_only_when_recorded():
    from caisen.schemas import Pattern

    def make_pattern(pattern_id, selection_marker=None, suppression=None, parent_id=None):
        evidence = [{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": True,
            "entry_eligible_at_confirmation": True}},
            {"code": "neckline_selection", "value": {
                "selected_line": {"slope": 0.0, "intercept": 10.0},
                "line_frozen_at": "2024-01-02", "freeze_date": "2024-01-02"}},
            {"code": "neckline_breakout", "value": {
                "date": "2024-01-03", "close": 11.0,
                "frozen_neckline": 10.0, "required_close": 10.8}}]
        if selection_marker is not None:
            evidence.append({"code": "main_selection", "value": {
                "selected": selection_marker, "decided_at": "2024-01-03",
                "primary_pattern_id": "selected-main",
                "primary_instance_id": "selected-instance"}})
        if suppression:
            evidence.append({"code": "execution_suppression", "value": {
                "suppressed": True, "suppressed_at": "2024-01-03"}})
        return Pattern(pattern_id, "TEST", "stock", "1d", "w_bottom", "bullish", "confirmed",
                       "2024-01-02", 2, confirmation_date="2024-01-03",
                       key_points={"pivot_1_date": "2024-01-02", "pivot_1_kind": "low",
                                   "pivot_2_date": "2024-01-02", "pivot_2_kind": "low"},
                       neckline={"slope": 0.0, "intercept": 10.0}, evidence=evidence,
                       parent_id=parent_id)

    session = DailyRecognitionSession("TEST", "stock", tick_size=0.01)
    missing = make_pattern("missing-selection")
    missing_lifecycle = session._attach_session_lifecycle(missing)
    assert missing_lifecycle["entry_eligible_at_confirmation"] is None
    assert missing_lifecycle["entry_eligibility_at_confirmation_status"] == \
        "unknown_missing_confirmation_selection"
    assert missing_lifecycle["entry_eligible"] is False

    rejected = make_pattern("rejected-selection", selection_marker=False)
    rejected_lifecycle = session._attach_session_lifecycle(rejected)
    assert rejected_lifecycle["entry_eligible_at_confirmation"] is False
    assert rejected_lifecycle["entry_eligibility_at_confirmation_status"] == "known"
    assert rejected_lifecycle["entry_eligible"] is False

    suppressed = make_pattern("suppressed-selection", selection_marker=True, suppression=True)
    suppressed_lifecycle = session._attach_session_lifecycle(suppressed)
    assert suppressed_lifecycle["entry_eligible_at_confirmation"] is False
    assert suppressed_lifecycle["entry_eligibility_at_confirmation_status"] == "known"
    assert suppressed_lifecycle["entry_eligible"] is False

    parent_suppressed = make_pattern("parent-suppressed-selection", selection_marker=True,
                                     parent_id="parent-instance")
    parent_lifecycle = session._attach_session_lifecycle(parent_suppressed)
    assert parent_lifecycle["entry_eligible_at_confirmation"] is False
    assert parent_lifecycle["entry_eligibility_at_confirmation_status"] == "known"
    assert parent_lifecycle["entry_eligible"] is False


def test_weekly_recognition_uses_same_engine_and_twelve_month_maturity():
    from caisen.daily_incremental import recognize_weekly_history
    from caisen.schemas import Pivot

    session = DailyRecognitionSession("WEEKLY", "stock", tick_size=0.01, timeframe="1w")
    session.bars = [{"date": pd.Timestamp("2024-01-08")}]
    candidate = {"family": "w_bottom", "sequence": (
        Pivot(0, "2024-01-08", 10.0, "low", 0, "2024-01-08"),
        Pivot(1, "2024-05-06", 10.1, "low", 1, "2024-05-06"))}
    assert session._candidate_maturity_date(candidate) == "2025-01-08"

    weekly_bars = pd.DataFrame([
        ("2024-01-05", 10.0, 10.2, 9.8, 10.0, 1000),
        ("2024-01-12", 10.0, 10.3, 9.9, 10.1, 1100),
        ("2024-01-19", 10.1, 10.2, 9.9, 10.0, 900),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    result = recognize_weekly_history(weekly_bars, "WEEKLY", "stock", tick_size=0.01)
    assert result["performance"]["timeframe"] == "1w"
    assert result["performance"]["engine_version"] == \
        "caisen-joint-pattern-session-v9-flag-confirmation-history"
    assert all(pattern.timeframe == "1w" for pattern in result["patterns"])


def test_daily_signal_generation_reports_monotonic_bar_progress():
    bars = pd.DataFrame([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 1000),
        ("2026-01-06", 10.0, 10.3, 9.9, 10.1, 1100),
        ("2026-01-07", 10.1, 10.2, 9.9, 10.0, 900),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    snapshots = []

    generate_daily_signals(bars, "PROGRESS", "stock", progress_callback=snapshots.append)

    assert snapshots[0]["bars_completed"] == 0
    assert snapshots[-1]["phase"] == "recognition_complete"
    assert snapshots[-1]["bars_completed"] == snapshots[-1]["bars_total"] == len(bars)
    assert [item["bars_completed"] for item in snapshots] == sorted(
        item["bars_completed"] for item in snapshots)


def test_boundary_first_dispatch_is_explicit_and_reports_pair_reuse():
    from caisen.daily_incremental import generate_daily_signals

    bars = pd.DataFrame([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 1000),
        ("2026-01-06", 10.0, 10.3, 9.9, 10.1, 1100),
        ("2026-01-07", 10.1, 10.2, 9.9, 10.0, 900),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    performance = {}

    signals = generate_daily_signals(bars, "SMOKE", "stock", recognizer="boundary_first",
                                     performance=performance)

    assert signals == []
    assert performance["engine_version"] == "caisen-joint-pattern-session-v9-flag-confirmation-history"
    assert performance["effective_recognizer"] == performance["engine_version"]
    assert performance["bars_appended"] == len(bars)
    assert performance["prefix_frame_builds"] <= len(bars)


def test_session_prefix_and_run_summary_preserve_input_provenance_and_tick_source():
    bars = pd.DataFrame([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 1000),
        ("2026-01-06", 10.0, 10.3, 9.9, 10.1, 1100),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    bars.attrs["provenance"] = {"data_version": "freeze-A"}
    bars.attrs["quality"] = {"tick_size": 0.005}

    session = DailyRecognitionSession("PROV", "stock", tick_size=0.005,
                                      provenance=bars.attrs["provenance"])
    session.append_bar(bars.iloc[0].to_dict())

    assert session._frame_for_current_prefix().attrs["provenance"]["data_version"] == "freeze-A"
    result = recognize_daily_history(bars, "PROV", "stock")
    assert result["performance"]["data_version"] == "freeze-A"
    assert result["performance"]["tick_size"] == 0.005
    assert result["performance"]["tick_size_source"] == "input_quality"
    assert result["performance"]["effective_recognizer"] == result["performance"]["engine_version"]


def test_session_default_config_is_frozen_when_appends_run_inside_experiment():
    from caisen.research_config import RecognitionExperimentConfig, recognition_experiment

    session = DailyRecognitionSession("PROV", "stock", provenance={
        "data_version": "mtime_ns:1,size:2",
        "recognition_context": {"recognition_experiment": None},
    })
    with recognition_experiment(RecognitionExperimentConfig(
            "internal_rebound_coverage_g1_v1", require_internal_rebound_coverage=True)):
        session.append_bar({"date": "2026-01-05", "open": 10, "high": 10.2,
                            "low": 9.8, "close": 10, "volume": 1000})

    context = session._frame_for_current_prefix().attrs["provenance"]["recognition_context"]
    assert context["recognition_experiment"] is None


def test_confirmed_flag_lifecycle_invalidates_before_target_touch():
    from caisen.schemas import Pattern

    session = DailyRecognitionSession("FLAG", "stock")
    session.bars = [{"date": pd.Timestamp("2026-01-05"), "open": 15.0, "high": 15.2,
                     "low": 14.8, "close": 15.0},
                    {"date": pd.Timestamp("2026-01-06"), "open": 14.9, "high": 15.0,
                     "low": 14.7, "close": 14.9},
                    {"date": pd.Timestamp("2026-01-07"), "open": 14.7, "high": 16.0,
                     "low": 14.6, "close": 14.7}]
    pattern = Pattern(
        "flag-instance", "FLAG", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "2026-01-01", 1, confirmation_date="2026-01-06", data_version="freeze-A",
        neckline={"slope": -0.1, "intercept": 15.0}, targets=[15.5],
        evidence=[{"code": "flag_lifecycle", "value": {"state": "confirmed", "target": 15.5,
                  "events": [{"date": "2026-01-06", "kind": "confirmed"}]}}],
    )
    session.flag_instances[pattern.pattern_id] = pattern

    session._advance_flag_lifecycles(2)

    assert pattern.state == "invalidated"
    lifecycle = next(item["value"] for item in pattern.evidence if item["code"] == "flag_lifecycle")
    assert lifecycle["events"][-1]["kind"] == "invalidated"
    assert lifecycle["events"][-1]["boundary"] == 14.8


def test_confirmed_flag_reaches_first_recorded_target_when_not_invalidated():
    from caisen.schemas import Pattern

    session = DailyRecognitionSession("FLAG", "stock")
    session.bars = [{"date": pd.Timestamp("2026-01-05"), "open": 15.0, "high": 15.2,
                     "low": 14.8, "close": 15.0},
                    {"date": pd.Timestamp("2026-01-06"), "open": 14.9, "high": 15.0,
                     "low": 14.7, "close": 14.9},
                    {"date": pd.Timestamp("2026-01-07"), "open": 15.0, "high": 15.6,
                     "low": 14.9, "close": 15.4}]
    pattern = Pattern(
        "flag-target-instance", "FLAG", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "2026-01-01", 1, confirmation_date="2026-01-06", data_version="freeze-A",
        neckline={"slope": -0.1, "intercept": 15.0}, targets=[15.5],
        evidence=[{"code": "flag_lifecycle", "value": {"state": "confirmed", "target": 15.5,
                  "events": [{"date": "2026-01-06", "kind": "confirmed"}]}}],
    )
    session.flag_instances[pattern.pattern_id] = pattern

    session._advance_flag_lifecycles(2)

    assert pattern.state == "target_reached"
    lifecycle = next(item["value"] for item in pattern.evidence if item["code"] == "flag_lifecycle")
    assert lifecycle["events"][-1]["kind"] == "target_reached"
    assert lifecycle["events"][-1]["price"] == 15.5


def test_flag_freezes_forming_geometry_before_a_later_bar_can_confirm(monkeypatch):
    from caisen.schemas import Pivot

    monkeypatch.setattr("caisen.daily_incremental._neckline_formation_coverage",
                        lambda *args, **kwargs: {"passed": True})
    session = DailyRecognitionSession("FLAG", "stock", tick_size=0.01)
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    session.bars = [{"date": date, "open": 18.5, "high": 19.0, "low": 17.0,
                     "close": 18.5, "volume": 1000} for date in dates]
    session.bars[5].update(high=20.0, low=19.0, close=19.5)
    session.bars[10].update(high=17.0, low=15.0, close=15.5)
    session.bars[15].update(high=19.0, low=18.0, close=18.5)
    session.bars[20].update(high=18.0, low=16.0, close=17.0)
    session.date_strings = [str(date.date()) for date in dates[:21]]
    session.atr_values = [1.0] * 21
    seq = (Pivot(0, "2026-01-01", 10.0, "low", 20, "2026-01-21"),
           Pivot(5, "2026-01-06", 20.0, "high", 20, "2026-01-21"),
           Pivot(10, "2026-01-11", 15.0, "low", 20, "2026-01-21"),
           Pivot(15, "2026-01-16", 19.0, "high", 20, "2026-01-21"),
           Pivot(20, "2026-01-21", 16.0, "low", 20, "2026-01-21"))
    session.flag_candidates["flag-geometry"] = seq

    assert session._check_bull_flag() == []
    formed = next(iter(session.flag_formations.values()))
    assert formed.state == "forming"
    assert formed.last_observed_date == "2026-01-21"
    geometry = next(item["value"] for item in formed.evidence if item["code"] == "flag_geometry")
    formed_joint = next(item["value"] for item in formed.evidence if item["code"] == "joint_lifecycle")
    assert geometry["known_at"] == "2026-01-21"

    session.bars.append({"date": pd.Timestamp("2026-01-22"), "open": 19.4,
                         "high": 19.8, "low": 19.3, "close": 19.5, "volume": 1000})
    session.date_strings.append("2026-01-22")
    session.atr_values.append(1.0)
    signals = session._check_bull_flag()

    assert len(signals) == 1
    assert signals[0]["signal_date"] == "2026-01-22"
    assert signals[0]["pattern"]["confirmation_date"] == "2026-01-22"
    confirmed = signals[0]["pattern"]
    evidence = {item["code"]: item["value"] for item in confirmed["evidence"]}
    flag_lifecycle = evidence["flag_lifecycle"]
    joint_lifecycle = evidence["joint_lifecycle"]
    assert joint_lifecycle["instance_id"] == formed_joint["instance_id"]
    assert joint_lifecycle["paired_at"] == "2026-01-21"
    assert joint_lifecycle["selection_at_confirmation"] == flag_lifecycle["selection_at_confirmation"]
    assert joint_lifecycle["entry_eligible_at_confirmation"] is True
    assert joint_lifecycle["entry_eligibility_at_confirmation_status"] == "known"


def test_flag_confirmation_is_suppressed_by_active_main_and_not_released_retroactively(monkeypatch):
    from caisen.schemas import Pattern, Pivot

    monkeypatch.setattr("caisen.daily_incremental._neckline_formation_coverage",
                        lambda *args, **kwargs: {"passed": True})
    session = DailyRecognitionSession("FLAG", "stock", tick_size=0.01)
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    session.bars = [{"date": date, "open": 18.5, "high": 19.0, "low": 17.0,
                     "close": 18.5, "volume": 1000} for date in dates]
    session.bars[5].update(high=20.0, low=19.0, close=19.5)
    session.bars[10].update(high=17.0, low=15.0, close=15.5)
    session.bars[15].update(high=19.0, low=18.0, close=18.5)
    session.bars[20].update(high=18.0, low=16.0, close=17.0)
    session.date_strings = [str(date.date()) for date in dates]
    session.atr_values = [1.0] * 21
    seq = (Pivot(0, "2026-01-01", 10.0, "low", 20, "2026-01-21"),
           Pivot(5, "2026-01-06", 20.0, "high", 20, "2026-01-21"),
           Pivot(10, "2026-01-11", 15.0, "low", 20, "2026-01-21"),
           Pivot(15, "2026-01-16", 19.0, "high", 20, "2026-01-21"),
           Pivot(20, "2026-01-21", 16.0, "low", 20, "2026-01-21"))
    session.flag_candidates["flag-geometry"] = seq

    parent = Pattern("active-main", "FLAG", "stock", "1d", "w_bottom", "bullish", "forming",
        "2025-12-01", None,
        key_points={"pivot_1_date": "2025-12-01", "pivot_1_kind": "low", "pivot_1": 8.0,
                    "pivot_2_date": "2026-01-15", "pivot_2_kind": "high", "pivot_2": 25.0,
                    "pivot_3_date": "2026-02-01", "pivot_3_kind": "low", "pivot_3": 9.0,
                    "pivot_scale": "5atr_12pct"},
        neckline={"slope": 0.0, "intercept": 25.0},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "first_trough_date": "2025-12-01",
            "structure_end_date": "2026-03-01", "structure_known_at": "2025-12-01",
            "support_lower": 8.0}},
                  {"code": "neckline_selection", "value": {"structure_known_at": "2025-12-01"}}])
    session.active[("w_bottom", "5atr_12pct", ())] = {"pattern": parent}

    assert session._check_bull_flag() == []
    session.bars.append({"date": pd.Timestamp("2026-01-22"), "open": 19.4,
                         "high": 19.8, "low": 19.3, "close": 19.5, "volume": 1000})
    session.date_strings.append("2026-01-22")
    session.atr_values.append(1.0)

    assert session._check_bull_flag() == []
    flag = next(iter(session.flag_instances.values()))
    lifecycle = next(item["value"] for item in flag.evidence if item["code"] == "flag_lifecycle")
    assert flag.confirmation_date == "2026-01-22"
    assert flag.parent_id == "active-main"
    assert lifecycle["entry_eligible_at_confirmation"] is False
    joint_lifecycle = next(item["value"] for item in flag.evidence
                           if item["code"] == "joint_lifecycle")
    assert joint_lifecycle["entry_eligible_at_confirmation"] is False
    assert joint_lifecycle["entry_eligibility_at_confirmation_status"] == "known"
    assert joint_lifecycle["selection_at_confirmation"]["selected"] is False
    assert joint_lifecycle["selection_at_confirmation"]["parent_pattern_id"] == "active-main"
    assert next(item["value"] for item in flag.evidence
                if item["code"] == "execution_suppression")["suppressed_at"] == "2026-01-22"

    parent.state = "expired"
    assert session._main_parent_for_flag(flag, "2026-01-23") is None
    assert lifecycle["entry_eligible_at_confirmation"] is False


def test_flag_forming_snapshot_invalidates_when_price_breaks_impulse_floor(monkeypatch):
    from caisen.schemas import Pivot

    monkeypatch.setattr("caisen.daily_incremental._neckline_formation_coverage",
                        lambda *args, **kwargs: {"passed": True})
    session = DailyRecognitionSession("FLAG", "stock", tick_size=0.01)
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    session.bars = [{"date": date, "open": 18.5, "high": 19.0, "low": 17.0,
                     "close": 18.5, "volume": 1000} for date in dates]
    session.bars[5].update(high=20.0, low=19.0, close=19.5)
    session.bars[10].update(high=17.0, low=15.0, close=15.5)
    session.bars[15].update(high=19.0, low=18.0, close=18.5)
    session.bars[20].update(high=18.0, low=16.0, close=17.0)
    session.date_strings = [str(date.date()) for date in dates]
    session.atr_values = [1.0] * 21
    seq = (Pivot(0, "2026-01-01", 10.0, "low", 20, "2026-01-21"),
           Pivot(5, "2026-01-06", 20.0, "high", 20, "2026-01-21"),
           Pivot(10, "2026-01-11", 15.0, "low", 20, "2026-01-21"),
           Pivot(15, "2026-01-16", 19.0, "high", 20, "2026-01-21"),
           Pivot(20, "2026-01-21", 16.0, "low", 20, "2026-01-21"))
    session.flag_candidates["flag-geometry"] = seq
    session._check_bull_flag()
    pattern = next(iter(session.flag_formations.values()))

    session.bars.append({"date": pd.Timestamp("2026-01-22"), "open": 9.5,
                         "high": 10.0, "low": 8.9, "close": 9.0, "volume": 1000})
    session.date_strings.append("2026-01-22")
    session.atr_values.append(1.0)
    session._check_bull_flag()

    assert pattern.state == "invalidated"
    lifecycle = next(item["value"] for item in pattern.evidence if item["code"] == "flag_lifecycle")
    assert lifecycle["events"][-1]["kind"] == "invalidated"
    assert lifecycle["events"][-1]["reason"] == "close_below_impulse_floor"


def test_flag_buffered_cross_during_pairing_cannot_rearm_after_reset(monkeypatch):
    from caisen.schemas import Pivot

    monkeypatch.setattr("caisen.daily_incremental._neckline_formation_coverage",
                        lambda *args, **kwargs: {"passed": True})
    session = DailyRecognitionSession("FLAG", "stock", tick_size=0.01)
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    session.bars = [{"date": date, "open": 18.5, "high": 19.0, "low": 17.0,
                     "close": 18.5, "volume": 1000} for date in dates]
    session.bars[5].update(high=20.0, low=19.0, close=19.5)
    session.bars[10].update(high=17.0, low=15.0, close=15.5)
    session.bars[15].update(high=19.0, low=18.0, close=18.5)
    session.bars[20].update(high=20.0, low=19.0, close=19.6)
    session.date_strings = [str(date.date()) for date in dates]
    session.atr_values = [1.0] * 21
    seq = (Pivot(0, "2026-01-01", 10.0, "low", 20, "2026-01-21"),
           Pivot(5, "2026-01-06", 20.0, "high", 20, "2026-01-21"),
           Pivot(10, "2026-01-11", 15.0, "low", 20, "2026-01-21"),
           Pivot(15, "2026-01-16", 19.0, "high", 20, "2026-01-21"),
           Pivot(20, "2026-01-21", 16.0, "low", 20, "2026-01-21"))
    session.flag_candidates["flag-geometry"] = seq
    assert session._check_bull_flag() == []

    for date, close in (("2026-01-22", 17.5), ("2026-01-23", 19.5)):
        session.bars.append({"date": pd.Timestamp(date), "open": close,
                             "high": close + 0.2, "low": close - 0.2,
                             "close": close, "volume": 1000})
        session.date_strings.append(date)
        session.atr_values.append(1.0)
        assert session._check_bull_flag() == []

    formed = next(iter(session.flag_formations.values()))
    lifecycle = next(item["value"] for item in formed.evidence if item["code"] == "flag_lifecycle")
    assert lifecycle["buffered_crossing_consumed_at"] == "2026-01-21"
    assert formed.state == "expired"
    assert lifecycle["state"] == "expired"
    assert not session.flag_candidates
    assert formed.pattern_id in session.emitted_ids

    from caisen.decision import synthesize_decision
    decision = synthesize_decision(
        "FLAG", "stock", {"1d": [formed.to_dict()], "1w": []},
        {"1d": 19.5}, {"1d": "usable", "1w": "insufficient"})
    assert decision["reason_codes"] == ["no_current_pattern"]


def test_flag_forming_snapshot_invalidates_when_existing_500_bar_search_window_ends():
    from caisen.schemas import Pattern, Pivot

    session = DailyRecognitionSession("FLAG", "stock", tick_size=0.01)
    dates = pd.date_range("2024-01-01", periods=506, freq="D")
    session.bars = [{"date": date, "open": 10.0, "high": 10.2, "low": 9.8,
                     "close": 10.0, "volume": 1000} for date in dates]
    session.date_strings = [str(date.date()) for date in dates]
    session.atr_values = [1.0] * len(dates)
    seq = tuple(Pivot(index, str(dates[index].date()), 10.0 + index, kind, 1,
                      str(dates[1].date()))
                for index, kind in enumerate(("low", "high", "low", "high", "low")))
    pattern_id = f"FLAG:bull_flag:{'-'.join(f'{p.index}:{p.kind}' for p in seq)}"
    pattern = Pattern(pattern_id, "FLAG", "stock", "1d", "bull_flag", "bullish", "forming",
                      seq[0].date, None, data_version="freeze-A",
                      key_points={"pivot_1_date": seq[0].date, "pivot_1": seq[0].price,
                                  "pivot_1_kind": "low"},
                      neckline={"slope": 0.0, "intercept": 10.0},
                      evidence=[{"code": "flag_geometry", "value": {"known_at": seq[-1].date}},
                                {"code": "flag_lifecycle", "value": {"state": "forming", "events": []}}])
    session.flag_candidates["stale"] = seq
    session.flag_formations[pattern_id] = pattern

    session._check_bull_flag()

    assert pattern.state == "invalidated"
    lifecycle = next(item["value"] for item in pattern.evidence if item["code"] == "flag_lifecycle")
    assert lifecycle["events"][-1]["reason"] == "flag_search_window_exceeded"


def test_same_boundary_new_pressure_test_does_not_move_original_pairing_date():
    from caisen.schemas import Pattern
    from caisen.structure_boundary import attach_joint_lifecycle

    line = {"slope": 0.0, "intercept": 12.0}
    tests = [{"pivot_date": "2024-11-01", "level": 12.0, "group_id": "touch-a"},
             {"pivot_date": "2024-12-01", "level": 12.0, "group_id": "touch-b"}]

    def candidate(freeze_date, state="forming", breakout=None):
        evidence = [
            {"code": "main_structure", "value": {"role": "main", "structure_known_at": freeze_date,
             "signal_eligible": state == "confirmed"}},
            {"code": "neckline_tests", "value": [dict(item) for item in tests]},
            {"code": "weak_crossings", "value": ([{"date": "2024-12-03", "close": 11.9,
                 "neckline": 12.0, "required_close": 13.0}] if freeze_date == "2024-12-02" else [])},
            {"code": "neckline_selection", "value": {
                "selected_line": dict(line), "line_frozen_at": freeze_date,
                "freeze_date": freeze_date, "effective_start_date": "2024-11-01",
                "selected_candidate_id": "candidate-1",
                "independent_tests": [dict(item) for item in tests]}},
        ]
        if breakout:
            evidence.append({"code": "neckline_breakout", "value": breakout})
        return Pattern("same-boundary", "000001", "stock", "1d", "w_bottom", "bullish", state,
                       "2024-11-01", 200 if breakout else None,
                       key_points={"pivot_1_date": "2024-11-01", "pivot_1": 10.0,
                                   "pivot_1_kind": "low", "pivot_2_date": "2024-12-01",
                                   "pivot_2": 10.1, "pivot_2_kind": "low"},
                       neckline=dict(line), evidence=evidence,
                       confirmation_date=breakout.get("date") if breakout else None)

    current = candidate("2024-12-02")
    replacement = candidate("2025-02-01", "confirmed", {"date": "2025-02-03", "close": 13.5})
    before = attach_joint_lifecycle(current, discovered_at="2024-12-02")

    kept = DailyRecognitionSession._keep_frozen_boundary(current, replacement)
    after = attach_joint_lifecycle(kept)

    selection = next(item["value"] for item in kept.evidence if item["code"] == "neckline_selection")
    assert before["instance_id"] == after["instance_id"]
    assert selection["line_frozen_at"] == "2024-12-02"
    assert after["paired_at"] == "2024-12-02"
    assert after["events"][0]["date"] == "2024-12-02"
    assert after["events"][-1]["kind"] == "confirmed"
    first_weak = next(event for event in before["events"] if event["kind"] == "weak_crossing")
    repeated = attach_joint_lifecycle(kept)
    repeated_weak = next(event for event in repeated["events"] if event["kind"] == "weak_crossing")
    assert repeated_weak == first_weak


def test_session_registry_preserves_first_pair_and_terminal_across_scale_aliases():
    from copy import deepcopy
    from caisen.schemas import Pattern

    session = DailyRecognitionSession("000008", "stock", tick_size=0.01)
    line = {"slope": -0.0001910828025477723, "intercept": 4.355796178343961}
    tests = [{"pivot_date": "2024-03-14", "group_id": "old-a", "level": 2.95},
             {"pivot_date": "2024-11-07", "group_id": "old-b", "level": 2.92}]

    def make(pattern_id, state="forming", breakout=None, scale="2atr_5pct"):
        selection = {"selected_candidate_id": "neckline-candidate-old",
                     "selected_line": dict(line), "line_frozen_at": "2024-11-14",
                     "freeze_date": "2024-12-02", "independent_tests": deepcopy(tests)}
        if state == "forming":
            selection["early_breakouts"] = [{"date": "2024-12-03", "index": 7532,
                "close": 3.06, "frozen_neckline": 2.91656050955414,
                "required_close": 3.036994212307247, "candidate_id": "neckline-candidate-old",
                "line": dict(line), "tests": deepcopy(tests)}]
        evidence = [{"code": "main_structure", "value": {
                         "role": "main", "structure_known_at": "2024-12-02",
                         "maturity_date": "2025-04-18", "signal_eligible": False}},
                    {"code": "neckline_selection", "value": selection},
                    {"code": "neckline_tests", "value": deepcopy(tests)}]
        if breakout:
            evidence.append({"code": "neckline_breakout", "value": breakout})
            evidence[0]["value"].update({"signal_eligible": True,
                                          "entry_eligible_at_confirmation": True})
            evidence.append({"code": "main_selection", "value": {"selected": True}})
        return Pattern(pattern_id, "000008", "stock", "1d", "w_bottom", "bullish", state,
                       "2024-10-18", 7532 if breakout else None,
                       key_points={"pivot_1_date": "2024-10-18", "pivot_1": 2.2,
                                   "pivot_1_kind": "low", "pivot_2_date": "2024-11-27",
                                   "pivot_2": 2.36, "pivot_2_kind": "low",
                                   "pivot_scale": scale},
                       neckline=dict(line), evidence=evidence,
                       confirmation_date=breakout.get("date") if breakout else None)

    first = make("000008:scale-2")
    first_lifecycle = session._attach_session_lifecycle(first, discovered_at="2025-04-18")
    canonical_events = deepcopy(first_lifecycle["events"])
    assert first.state == "expired"

    second = make("000008:scale-3", state="confirmed",
                  breakout={"date": "2025-08-21", "close": 3.0,
                            "frozen_neckline": line["intercept"], "required_close": 2.94},
                  scale="3atr_8pct")
    second_lifecycle = session._attach_session_lifecycle(second, discovered_at="2025-07-15")

    assert second_lifecycle["instance_id"] == first_lifecycle["instance_id"]
    assert second_lifecycle["paired_at"] == "2025-04-18"
    assert second.state == "expired"
    assert second.confirmation_date is None
    assert second_lifecycle["events"] == canonical_events
    assert not any(event["kind"] == "confirmed" for event in second_lifecycle["events"])
    assert any(item["kind"] == "terminal_instance_transition_blocked"
               for item in second_lifecycle["event_conflicts"])


def test_new_main_instance_cross_on_pairing_bar_is_consumed_without_entry(monkeypatch):
    import caisen.daily_incremental as incremental
    from caisen.schemas import Pattern, Pivot

    session = DailyRecognitionSession("PAIR", "stock", tick_size=0.01)
    dates = pd.date_range("2024-07-01", periods=184, freq="D")
    session.bars = [{"date": date, "open": 10.0, "high": 10.2, "low": 9.8,
                     "close": 10.0, "volume": 1000} for date in dates]
    session.date_strings = [str(date.date()) for date in dates]
    session.high_values = [10.2] * len(dates)
    session.low_values = [9.8] * len(dates)
    session.close_values = [10.0] * len(dates)
    session.atr_values = [0.2] * len(dates)
    current_date = session.date_strings[-1]
    sequence = [Pivot(0, session.date_strings[0], 9.0, "low", 2, session.date_strings[2], "2ATR|5%"),
                Pivot(90, session.date_strings[90], 9.1, "low", 92, session.date_strings[92], "2ATR|5%")]
    candidate_key = ("w_bottom", "2atr_5pct", tuple((p.index, p.kind) for p in sequence))
    candidate = {"family": "w_bottom", "scale": "2atr_5pct", "sequence": sequence,
                 "pattern": None, "needs_initial_eval": True, "maturity_date": current_date}
    session.active[candidate_key] = candidate
    monkeypatch.setattr(session, "_has_pressure_test_pair", lambda _candidate: True)

    pattern = Pattern("pairing-day-cross", "PAIR", "stock", "1d", "w_bottom", "bullish",
        "confirmed", session.date_strings[0], len(dates) - 1,
        key_points={"pivot_1_date": session.date_strings[0], "pivot_1": 9.0,
                    "pivot_1_kind": "low", "pivot_2_date": session.date_strings[90],
                    "pivot_2": 9.1, "pivot_2_kind": "low"},
        neckline={"slope": 0.0, "intercept": 10.0}, confirmation_date=current_date,
        last_observed_date=current_date,
        evidence=[{"code": "main_structure", "value": {"role": "main", "signal_eligible": True}},
                  {"code": "main_selection", "value": {"selected": True}},
                  {"code": "neckline_selection", "value": {
                      "selected_line": {"slope": 0.0, "intercept": 10.0},
                      "line_frozen_at": current_date, "freeze_date": current_date,
                      "structure_known_at": current_date, "selected_candidate_id": "same-day-line"}},
                  {"code": "neckline_breakout", "value": {"date": current_date,
                      "close": 11.0, "frozen_neckline": 10.0, "required_close": 10.8}}])
    monkeypatch.setattr(incremental, "_make_main_pattern", lambda *args, **kwargs: [pattern])

    signals = session._evaluate()

    assert signals == []
    assert candidate["pattern"].state == "expired"
    assert candidate["pattern"].confirmation_date is None
    lifecycle = next(item["value"] for item in candidate["pattern"].evidence
                     if item.get("code") == "joint_lifecycle")
    assert lifecycle["paired_at"] == current_date
    assert not any(event["kind"] == "confirmed" for event in lifecycle["events"])
    assert any(event["kind"] == "pairing_day_cross_consumed"
               for event in lifecycle["events"])
    assert lifecycle["entry_eligible"] is False


def test_target_reached_parent_does_not_suppress_a_later_internal_candidate():
    from caisen.schemas import Pattern

    parent = Pattern("ended-main", "fixture", "stock", "1d", "w_bottom", "bullish", "target_reached",
                     "2024-02-06", 900, confirmation_date="2024-10-08",
                     evidence=[{"code": "main_structure", "value": {
                         "role": "main", "signal_eligible": True,
                         "first_trough_date": "2024-02-06", "structure_end_date": "2025-03-01"}}])
    child = Pattern("new-child", "fixture", "stock", "1d", "inverse_head_shoulders", "bullish", "forming",
                    "2025-01-01", None,
                    key_points={"pivot_1_date": "2025-01-01", "pivot_2_date": "2025-02-01",
                                "pivot_3_date": "2025-03-01"},
                    last_observed_date="2025-03-01",
                    evidence=[{"code": "main_structure", "value": {
                        "role": "main", "signal_eligible": False,
                        "first_trough_date": "2025-01-01", "structure_end_date": "2025-03-01"}}])

    assert "new-child" not in DailyRecognitionSession._nested_ids([parent, child], "2025-03-01")


def test_nested_suppression_ignores_multiscale_aliases_and_requires_price_containment():
    from caisen.schemas import Pattern

    def w_pattern(pattern_id, first, second, support, neckline, scale="2atr_5pct"):
        return Pattern(
            pattern_id, "600108", "stock", "1d", "w_bottom", "bullish", "forming",
            first, None,
            key_points={"pivot_scale": scale,
                        "first_trough_index": pd.Timestamp(first).toordinal(),
                        "latest_anchor_index": pd.Timestamp(second).toordinal(),
                        "pivot_1_date": first, "pivot_1_kind": "low",
                        "pivot_2_date": second, "pivot_2_kind": "low"},
            neckline={"slope": 0.0, "intercept": neckline},
            evidence=[{"code": "main_structure", "value": {
                "role": "main", "pivot_scale": scale,
                "first_trough_date": first, "structure_end_date": second,
                "support_lower": support}},
                      {"code": "neckline_selection", "value": {
                          "structure_known_at": second}}],
        )

    # This older low pair encloses the target in time, but its support is above
    # the target's deeper first low, so it is not a containing parent structure.
    non_containing_parent = w_pattern("jan-sep", "2024-01-23", "2024-09-18", 2.25, 2.87)
    # Same two structural anchors at another pivot scale are aliases of one
    # shape, not parent/child structures that may suppress each other.
    multiscale_alias = w_pattern("same-anchors-5atr", "2024-02-06", "2024-09-18",
                                 2.22, 2.87, "5atr_12pct")
    target = w_pattern("target-w", "2024-02-06", "2024-09-18", 2.22, 2.87)

    assert "target-w" not in DailyRecognitionSession._nested_ids(
        [non_containing_parent, multiscale_alias, target], "2024-10-08")

    valid_parent = w_pattern("outer-w", "2024-01-01", "2024-10-01", 1.90, 3.20,
                             "5atr_12pct")
    inner_child = w_pattern("inner-w", "2024-02-06", "2024-09-18", 2.22, 2.87)
    assert "inner-w" in DailyRecognitionSession._nested_ids(
        [valid_parent, inner_child], "2024-10-08")


def test_confirmed_instance_invalidates_on_its_own_frozen_neckline_and_stays_terminal():
    from caisen.schemas import Pattern, Pivot

    session = DailyRecognitionSession("FIXTURE", "stock", tick_size=0.01)
    session.bars = [
        {"date": pd.Timestamp("2025-01-01"), "open": 10.6, "high": 10.8,
         "low": 10.4, "close": 10.6, "volume": 100},
        {"date": pd.Timestamp("2025-01-02"), "open": 10.5, "high": 10.6,
         "low": 10.2, "close": 10.4, "volume": 100},
    ]
    session.date_strings = ["2025-01-01", "2025-01-02"]
    session.low_values = [10.4, 10.2]
    session.high_values = [10.8, 10.6]
    session.close_values = [10.6, 10.4]
    session.atr_values = [0.2, 0.2]
    points = (Pivot(0, "2025-01-01", 10.0, "low", 0, "2025-01-01", "2ATR|5%"),
              Pivot(1, "2025-01-02", 10.1, "low", 1, "2025-01-02", "2ATR|5%"))
    pattern = Pattern("instance", "FIXTURE", "stock", "1d", "w_bottom", "bullish", "confirmed",
                      "2025-01-01", 0, confirmation_date="2025-01-01",
                      last_observed_date="2025-01-01", neckline={"slope": 0.0, "intercept": 10.5},
                      evidence=[{"code": "main_structure", "value": {
                          "role": "main", "signal_eligible": True, "support_lower": 9.5,
                          "terminal_state": None, "structure_end_date": "2025-01-01"}},
                                {"code": "neckline_invalidation", "value": None}])
    key = ("w_bottom", "2atr_5pct", ((0, "low"), (1, "low")))
    session.active[key] = {"family": "w_bottom", "scale": "2atr_5pct",
                           "sequence": points, "pattern": pattern, "structure_known_at": 0}

    session._maintain_candidates()

    assert key not in session.active
    current = next(pattern for pattern in session.retired_patterns.values()
                   if pattern.pattern_id == "instance")
    assert current.state == "invalidated"
    detail = {item["code"]: item["value"] for item in current.evidence}
    assert detail["neckline_invalidation"]["date"] == "2025-01-02"
    assert detail["neckline_invalidation"]["level"] == 10.5
    assert detail["joint_lifecycle"]["events"][-1]["kind"] == "invalidated"


def test_frozen_candidate_keeps_its_boundary_when_later_evidence_prefers_another_line():
    from caisen.schemas import Pattern

    current = Pattern("same-structure-old-line", "FIXTURE", "stock", "1d", "w_bottom",
                      "bullish", "forming", "2024-02-06", 20,
                      key_points={"pivot_1_date": "2024-02-06", "pivot_1_kind": "low",
                                  "pivot_2_date": "2024-09-18", "pivot_2_kind": "low"},
                      neckline={"slope": 0.0, "intercept": 2.87},
                      evidence=[{"code": "main_structure", "value": {
                          "role": "main", "signal_eligible": False,
                          "structure_known_at": "2024-09-25"}},
                                {"code": "neckline_selection", "value": {
                                    "line_frozen_at": "2024-09-26",
                                    "effective_start_date": "2024-03-15"}},
                                {"code": "neckline_tests", "value": [
                                    {"group_id": "a", "pivot_date": "2024-03-15", "level": 2.87},
                                    {"group_id": "b", "pivot_date": "2024-05-10", "level": 2.84}]}])
    replacement = Pattern("same-structure-new-line", "FIXTURE", "stock", "1d", "w_bottom",
                          "bullish", "forming", "2024-02-06", 20,
                          key_points=current.key_points,
                          neckline={"slope": 0.0, "intercept": 2.758},
                          evidence=[{"code": "main_structure", "value": {
                              "role": "main", "signal_eligible": False,
                              "structure_known_at": "2024-09-25"}},
                                    {"code": "neckline_selection", "value": {
                                        "line_frozen_at": "2024-10-02",
                                        "effective_start_date": "2024-03-20"}},
                                    {"code": "neckline_tests", "value": [
                                        {"group_id": "c", "pivot_date": "2024-03-20", "level": 2.76},
                                        {"group_id": "d", "pivot_date": "2024-06-13", "level": 2.75}]}])

    kept = DailyRecognitionSession._keep_frozen_boundary(current, replacement)

    assert kept is current
    assert kept.neckline == {"slope": 0.0, "intercept": 2.87}


def test_expired_candidate_can_be_replaced_by_a_new_frozen_boundary_version():
    from caisen.schemas import Pattern

    old = Pattern("old", "FIXTURE", "stock", "1d", "w_bottom", "bullish", "expired",
                  "2024-01-01", None, key_points={"pivot_1_date": "2024-01-01",
                  "pivot_1_kind": "low", "pivot_1": 9.0},
                  neckline={"slope": 0.0, "intercept": 10.0},
                  evidence=[{"code": "main_structure", "value": {"terminal_state": "expired"}},
                            {"code": "neckline_selection", "value": {"selected_candidate_id": "old-pair",
                              "selected_line": {"slope": 0.0, "intercept": 10.0}}}])
    replacement = Pattern("new", "FIXTURE", "stock", "1d", "w_bottom", "bullish", "forming",
                          "2024-01-01", None, key_points={"pivot_1_date": "2024-01-01",
                          "pivot_1_kind": "low", "pivot_1": 9.0},
                          neckline={"slope": 0.0, "intercept": 10.2},
                          evidence=[{"code": "main_structure", "value": {"terminal_state": None}},
                                    {"code": "neckline_selection", "value": {"selected_candidate_id": "new-pair",
                                      "selected_line": {"slope": 0.0, "intercept": 10.2}}}])

    current = DailyRecognitionSession._keep_frozen_boundary(old, replacement)

    assert current is replacement
    assert current.neckline["intercept"] == 10.2


def test_batch_detail_and_daily_signal_share_frozen_instance_on_159718_confirmation():
    from caisen.daily_incremental import recognize_daily_history

    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-04-02"].reset_index(drop=True)
    bars.attrs["quality"] = {"tick_size": 0.01}
    scan = recognize_daily_history(bars, "159718", "etf", tick_size=0.01)
    backtest_signals = generate_daily_signals(bars, "159718", "etf")
    scan_event = next(item for item in scan["signals"] if item["signal_date"] == "2025-04-01")
    backtest_event = next(item for item in backtest_signals if item["signal_date"] == "2025-04-01")
    scan_pattern = next(item for item in scan_event["patterns"]
                        if item["pattern_type"] == "inverse_head_shoulders")
    backtest_pattern = next(item for item in backtest_event["patterns"]
                            if item["pattern_type"] == "inverse_head_shoulders")
    scan_lifecycle = next(item["value"] for item in scan_pattern["evidence"]
                          if item["code"] == "joint_lifecycle")
    backtest_lifecycle = next(item["value"] for item in backtest_pattern["evidence"]
                              if item["code"] == "joint_lifecycle")
    assert scan_lifecycle["instance_id"] == backtest_lifecycle["instance_id"]
    assert scan_lifecycle["boundary_version_id"] == backtest_lifecycle["boundary_version_id"]
    assert scan_pattern["confirmation_date"] == backtest_pattern["confirmation_date"] == "2025-04-01"
    assert scan_pattern["neckline"] == backtest_pattern["neckline"]


def test_boundary_first_preserves_v7_signal_payload_on_a_long_history_prefix():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"]).iloc[:1200].reset_index(drop=True)
    performance = {}

    reference_performance = {}
    reference = generate_daily_signals(bars, "600108", "stock", recognizer="v7",
                                       performance=reference_performance)
    accelerated = generate_daily_signals(bars, "600108", "stock",
                                         recognizer="boundary_first", performance=performance)

    assert accelerated == reference
    # Three v8 outputs are intentionally retired: the Mar 13 triangle repeats
    # the Feb 25 consumed neckline phase; the Jun 28 W first crossed before its
    # six-month maturity; and the Dec 7 W recrosses its Jun 21 premature break.
    # A distinct 1999 W previously got suppressed by time-only nesting; its
    # parent does not contain the lower structural support in price.
    assert [item["signal_date"] for item in accelerated] == [
        "1998-02-25", "1999-06-28", "2002-06-24"]
    assert performance["boundary_pairs_generated"] > 0
    assert performance["boundary_pairs_reused"] > 0
    assert performance["coverage_checks"] > 0
    assert performance["coverage_reused"] > 0
    assert performance["candidate_refreshes"] < reference_performance["candidate_refreshes"]
    assert reference_performance.get("boundary_pairs_reused", 0) == 0


def test_boundary_first_prefix_events_do_not_change_after_appending_future_bars():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"]).iloc[:1200].reset_index(drop=True)
    cutoff = str(bars.date.iloc[899].date())

    prefix = generate_daily_signals(bars.iloc[:900], "600108", "stock",
                                    recognizer="boundary_first")
    extended = generate_daily_signals(bars, "600108", "stock", recognizer="boundary_first")
    prefix_events = [{key: value for key, value in signal.items() if key != "entry_date"}
                     for signal in prefix]
    extended_events = [{key: value for key, value in signal.items() if key != "entry_date"}
                       for signal in extended if signal["signal_date"] <= cutoff]

    assert extended_events == prefix_events
