from pathlib import Path

import pandas as pd

from caisen.daily_incremental import DailyRecognitionSession, generate_daily_signals
from caisen.main_bottoms import MAIN_RULES


def test_session_one_atr_changed_neckline_can_emit_new_instance_entry(monkeypatch):
    # Legacy diagnostic sensitivity only; production recognition uses 2 ATR.
    monkeypatch.setitem(MAIN_RULES["neckline_tests"], "candidate_reversal_atr_multiple", 1.0)
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-30"].reset_index(drop=True)
    session = DailyRecognitionSession("159718", "etf")
    signals = []

    for bar in bars.to_dict("records"):
        emitted = session.append_bar(bar)
        signals.extend(emitted)
        assert all(item["signal_date"] <= str(pd.Timestamp(bar["date"]).date()) for item in emitted)

    # Revisions share bottom/shoulder anchors but own a distinct frozen line
    # and can confirm only after that line is known.
    revised = [item for item in signals
               if item["signal_date"] in {"2025-05-21", "2025-05-29"}
               and "inverse_head_shoulders" in item["pattern_types"]]
    assert [item["signal_date"] for item in revised] == ["2025-05-21", "2025-05-29"]
    first_signal = next(item for item in signals if item["signal_date"] == "2025-04-01"
                        and "inverse_head_shoulders" in item["pattern_types"])
    first_ids = {pattern["pattern_id"] for pattern in first_signal["patterns"]
                 if pattern["pattern_type"] == "inverse_head_shoulders"}
    assert all(not (first_ids & set(item["pattern_ids"])) for item in revised)
    revised_entries = [event for event in session.instance_events()
                       if event["entry_eligible"]
                       and event["instance_id"] not in first_ids
                       and event["pattern"].get("pattern_type") == "inverse_head_shoulders"
                       and [event["pattern"]["key_points"].get(f"pivot_{number}_date")
                            for number in (1, 2, 3)] == [
                                "2023-08-22", "2024-04-19", "2025-01-13"]]
    assert {event["event_date"] for event in revised_entries} == {
        "2025-05-21", "2025-05-29"}


def test_session_suppresses_nested_159718_ihs_and_allows_same_anchor_line_revision():
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
    revised_signal = next(item for item in signals
                          if item["signal_date"] == "2025-05-21"
                          and "inverse_head_shoulders" in item["pattern_types"])
    assert not any(item["signal_date"] == "2025-05-29"
                   and "inverse_head_shoulders" in item["pattern_types"] for item in signals)
    first_ihs_signal = next(item for item in signals
                           if item["signal_date"] == "2025-04-01"
                           and "inverse_head_shoulders" in item["pattern_types"])
    ihs_ids = {pattern["pattern_id"] for pattern in first_ihs_signal["patterns"]
               if pattern["pattern_type"] == "inverse_head_shoulders"}
    same_instance_confirmations = [event for event in session.instance_events()
        if event["instance_id"] in ihs_ids and event["state"] == "confirmed"
        and event["entry_eligible"]]
    # Each independently frozen boundary keeps its own instance. The changed
    # boundary gets fresh IDs and its own post-freeze BUY; strictly contained
    # candidates remain suppressed.
    assert len(ihs_ids) == 3
    assert {event["instance_id"] for event in same_instance_confirmations} == ihs_ids
    assert [event["event_date"] for event in same_instance_confirmations] == ["2025-04-01"] * 3
    assert [item["signal_date"] for item in signals
            if ihs_ids.intersection(item["pattern_ids"])] == ["2025-04-01"]
    revised_ids = {pattern["pattern_id"] for pattern in revised_signal["patterns"]
                   if pattern["pattern_type"] == "inverse_head_shoulders"}
    assert len(revised_ids) == 3
    assert not (ihs_ids & revised_ids)
    revised_confirmations = [event for event in session.instance_events()
        if event["instance_id"] in revised_ids and event["state"] == "confirmed"
        and event["entry_eligible"]]
    assert {event["instance_id"] for event in revised_confirmations} == revised_ids
    assert [event["event_date"] for event in revised_confirmations] == ["2025-05-21"] * 3



def test_batch_final_day_signal_has_no_entry_session():
    # Use a confirmed case independent of the pending 600108 boundary review.
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-04-01"].reset_index(drop=True)

    signals = generate_daily_signals(bars, "159718", "etf")
    signal = next(item for item in signals if item["signal_date"] == "2025-04-01")

    assert signal["entry_date"] is None
    assert signal["pattern_types"]
    assert signal["pattern_ids"]


def test_batch_trade_start_filters_emissions_without_resetting_history():
    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-10-30"].reset_index(drop=True)

    recent = generate_daily_signals(bars, "516630", "etf", trade_start="2025-01-01")

    assert all(item["signal_date"] >= "2025-01-01" for item in recent)


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
    assert performance["engine_version"] == "caisen-daily-boundary-first-v16"
    assert performance["boundary_pairs_generated"] == 0
    assert performance["boundary_pairs_reused"] == 0
    assert performance["coverage_checks"] == 0


def test_boundary_first_preserves_long_history_signal_identity_with_anchor_coverage_mode():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"]).iloc[:1200].reset_index(drop=True)
    performance = {}

    reference_performance = {}
    reference = generate_daily_signals(bars, "600108", "stock", performance=reference_performance)
    accelerated = generate_daily_signals(bars, "600108", "stock",
                                         recognizer="boundary_first", performance=performance)

    # Anchor-only coverage changes the evidence payload and may admit candidates
    # that the full-interval reference rejects. For this known 600108 prefix,
    # the signal identity and execution dates remain the same.
    assert [(item["signal_date"], item["entry_date"], item["pattern_ids"],
             item["pattern_types"]) for item in accelerated] == [
        (item["signal_date"], item["entry_date"], item["pattern_ids"],
         item["pattern_types"]) for item in reference]
    assert [item["signal_date"] for item in accelerated] == ["1999-06-28", "2002-06-24"]
    assert performance["boundary_pairs_generated"] > 0
    assert performance["boundary_pairs_reused"] > 0
    assert performance["coverage_checks"] > 0
    assert performance["coverage_reused"] > 0
    assert performance["candidate_refreshes"] < reference_performance["candidate_refreshes"]


def test_516630_w_boundary_freezes_only_after_the_structure_is_known():
    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-10-08"].reset_index(drop=True)

    signals = generate_daily_signals(
        bars, "516630", "etf", recognizer="boundary_first")

    signal = next(item for item in signals if item["signal_date"] == "2024-10-08")
    w = next(item for item in signal["patterns"] if item["pattern_type"] == "w_bottom"
             and item["key_points"].get("pivot_1_date") == "2024-02-05"
             and item["key_points"].get("pivot_2_date") == "2024-09-09")
    selection = {item["code"]: item["value"] for item in w["evidence"]
                 if isinstance(item, dict)}["neckline_selection"]

    assert selection["line_frozen_at"] == selection["structure_known_at"]
    assert selection["line_frozen_at"] == "2024-09-20"


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
