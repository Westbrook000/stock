"""Guard tests for causal lifecycles that must survive the rolling session."""

import pandas as pd

from caisen.daily_incremental import DailyRecognitionSession
from caisen.schemas import Pattern, Pivot


def _bar(day, close=10.0):
    return {"date": pd.Timestamp(day), "open": close, "high": close + 0.1,
            "low": close - 0.1, "close": close, "volume": 1000}


def _quiet_pivot_updates(session, monkeypatch):
    monkeypatch.setattr(session, "_append_scale_pivots", lambda _index: [])
    monkeypatch.setattr(session, "_append_pressure_pivot", lambda _index: None)
    monkeypatch.setattr(session, "_append_local_pivot", lambda _index: False)


def _cached_bottom(first_trough="2024-01-01", frozen="2024-06-28",
                   known="2024-06-28", support_lower=9.5):
    return Pattern(
        pattern_id="cached-bottom", code="000001", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="forming",
        formed_at=first_trough, confirmed_at=None,
        key_points={"first_trough_date": first_trough, "first_trough_index": 0,
                    "pivot_1_date": first_trough, "pivot_scale": "2atr_5pct"},
        neckline={"slope": 0.0, "intercept": 10.0}, invalidation=support_lower,
        evidence=[
            {"code": "main_structure", "value": {
                "role": "main", "eligibility_reason": "awaiting_fresh_neckline_breakout",
                "support_lower": support_lower}},
            {"code": "neckline_selection", "value": {
                "line_frozen_at": frozen, "structure_known_at": known}},
        ],
        last_observed_date=frozen,
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
    session.append_bar(_bar("2024-07-01", 9.9))
    after_maturity = session.append_bar(_bar("2024-07-02", 10.5))

    assert early == []
    assert after_maturity == []
    assert session.consumed_neckline_phases
    detail = {item["code"]: item["value"] for item in
              next(iter(session.active.values()))["pattern"].evidence}
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
    assert set(emitted[0]["pattern_ids"]) == {"same-day-w", "same-day-triangle"}
    assert set(emitted[0]["pattern_types"]) == {"w_bottom", "triangle_bottom"}
    assert emitted[0]["pattern"]["pattern_id"] == "same-day-w"
    selections = [next(item["value"] for item in pattern["evidence"]
                       if item["code"] == "main_selection") for pattern in emitted[0]["patterns"]]
    assert all(item["primary_pattern_id"] == "same-day-w" for item in selections)
    assert sum(item["selected"] for item in selections) == 1
    grouped = {pattern["pattern_id"]: pattern for pattern in emitted[0]["patterns"]}
    assert grouped["same-day-w"]["neckline"]["intercept"] == 10.5
    assert grouped["same-day-triangle"]["neckline"]["intercept"] == 9.5
    line_ids = {pattern["pattern_id"]: next(item["value"]["selected_candidate_id"]
                   for item in pattern["evidence"] if item["code"] == "neckline_selection")
                for pattern in emitted[0]["patterns"]}
    assert line_ids == {"same-day-w": "w-line", "same-day-triangle": "triangle-line"}


def test_major_w_bottom_suppresses_a_contained_smaller_triangle_buy():
    parent = Pattern(
        pattern_id="main-w", code="600108", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed",
        formed_at="2024-02-06", confirmed_at=1, confirmation_date="2026-01-09",
        key_points={"pivot_1_date": "2024-02-06", "pivot_2_date": "2024-09-18",
                    "pivot_scale": "5atr_12pct"},
        evidence=[
            {"code": "main_structure", "value": {
                "role": "main", "family": "w_bottom", "pivot_scale": "5atr_12pct",
                "first_trough_date": "2024-02-06", "structure_end_date": "2025-12-30"}},
            {"code": "neckline_selection", "value": {"structure_known_at": "2024-09-26"}},
        ],
        last_observed_date="2026-01-09",
    )
    child = Pattern(
        pattern_id="inner-triangle", code="600108", kind="stock", timeframe="1d",
        pattern_type="triangle_bottom", direction="bullish", state="confirmed",
        formed_at="2025-03-31", confirmed_at=2, confirmation_date="2026-01-09",
        key_points={"pivot_1_date": "2025-03-31", "pivot_2_date": "2025-04-09",
                    "pivot_3_date": "2025-04-30", "pivot_4_date": "2025-11-17",
                    "pivot_5_date": "2025-12-17", "pivot_scale": "3atr_8pct"},
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "family": "triangle_bottom", "pivot_scale": "3atr_8pct",
            "first_trough_date": "2025-03-31", "structure_end_date": "2025-12-19"}}],
        last_observed_date="2026-01-09",
    )

    assert DailyRecognitionSession._nested_ids([parent, child], "2026-01-09") == {"inner-triangle"}
