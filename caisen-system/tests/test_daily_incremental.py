from pathlib import Path

import pandas as pd

from caisen.daily_incremental import DailyRecognitionSession, generate_daily_signals
from caisen.main_bottoms import MAIN_RULES


def test_session_one_atr_reference_is_causal_and_emits_each_confirmation_once(monkeypatch):
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

    breakout = [item for item in signals if item["signal_date"] == "2025-05-29"]
    assert len(breakout) == 1
    assert breakout[0]["entry_date"] is None
    assert "inverse_head_shoulders" in breakout[0]["pattern_types"]
    assert sum(item["signal_date"] == "2025-05-29" for item in signals) == 1


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
    consumed_key = next(key for key in session.consumed_neckline_phases
                        if key[0] == "inverse_head_shoulders"
                        and [dates[int(index)] for index, _kind in key[1]] ==
                        ["2023-09-22", "2024-02-05", "2024-11-22"])
    earlier_phase = next(phase for phase in session.consumed_neckline_phases[consumed_key]
                         if dates[phase["consumed_index"]] == "2025-04-01")
    assert earlier_phase["line"] == {
        "slope": -0.00024299065420560769,
        "intercept": 0.9125607476635516,
    }
    later_boundary = {
        "line": {"slope": -0.0002826747720364741,
                 "intercept": 0.9492036474164133},
        "tests": [{"pivot_index": index, "tolerance": 0.02,
                   "coverage_start_index": 450, "coverage_end_index": 779}
                  for index in (450, 669, 779)],
    }
    from caisen.main_bottoms import _same_neckline_boundary_phase
    assert _same_neckline_boundary_phase(earlier_phase, later_boundary, tick_size=0.01)
    may_21 = dates.index("2025-05-21")
    assert bars.close.iloc[may_21] == 0.744



def test_batch_final_day_signal_has_no_entry_session():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-01-09"].reset_index(drop=True)

    signals = generate_daily_signals(bars, "600108", "stock")
    signal = next(item for item in signals if item["signal_date"] == "2026-01-09")

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
    assert performance["engine_version"] == "caisen-daily-boundary-first-v3"
    assert performance["boundary_pairs_generated"] == 0
    assert performance["boundary_pairs_reused"] == 0
    assert performance["coverage_checks"] == 0


def test_boundary_first_preserves_v7_signal_payload_on_a_long_history_prefix():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"]).iloc[:1200].reset_index(drop=True)
    performance = {}

    reference_performance = {}
    reference = generate_daily_signals(bars, "600108", "stock", performance=reference_performance)
    accelerated = generate_daily_signals(bars, "600108", "stock",
                                         recognizer="boundary_first", performance=performance)

    assert accelerated == reference
    # Three v8 outputs are intentionally retired: the Mar 13 triangle repeats
    # the Feb 25 consumed neckline phase; the Jun 28 W first crossed before its
    # six-month maturity; and the Dec 7 W recrosses its Jun 21 premature break.
    assert [item["signal_date"] for item in accelerated] == ["1998-02-25", "2002-06-24"]
    assert performance["boundary_pairs_generated"] > 0
    assert performance["boundary_pairs_reused"] > 0
    assert performance["coverage_checks"] > 0
    assert performance["coverage_reused"] > 0
    assert performance["candidate_refreshes"] < reference_performance["candidate_refreshes"]


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
