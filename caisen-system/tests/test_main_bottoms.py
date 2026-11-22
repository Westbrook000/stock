from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from caisen.main_bottoms import (
    MAIN_RULES,
    PIVOT_SCALES,
    _independent_neckline_tests,
    _ihs_right_shoulder_intact,
    _ihs_valley_intact,
    _structure_candidates,
    _triangle_boundaries_intact,
    _triangle_apex_index,
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
    assert MAIN_RULES["source_rule_version"] == "caisen-main-bottoms-v17-relative-neckline-height-100"
    assert MAIN_RULES["neckline_relative_height"]["maximum_ratio"] == 1.0
    assert MAIN_RULES["neckline_tests"]["candidate_reversal_atr_multiple"] == 2.0
    assert MAIN_RULES["neckline_tests"]["minimum_departure_atr"] == 1.0
    assert MAIN_RULES["timeframes"]["1d"]["structure_window_calendar_years"] == 5
    assert MAIN_RULES["timeframes"]["1w"]["structure_window_calendar_years"] == 10
    assert MAIN_RULES["minimum_duration_calendar_months_by_timeframe"] == {"1d": 6, "1w": 12}
    assert MAIN_RULES["minimum_duration_calendar_months"] == 6
    assert MAIN_RULES["neckline_tests"]["minimum_independent_groups"] == 2
    assert MAIN_RULES["neckline_tests"]["consumed_boundary_phase_rule"] == \
        "same_structure_equivalent_frozen_neckline_v1"
    assert MAIN_RULES["inverse_head_shoulders"]["maximum_leg_duration_ratio"] == 2.5
    assert MAIN_RULES["inverse_head_shoulders"][
        "minimum_intervening_right_trough_head_separation_atr_multiple"] == 2.0
    assert MAIN_RULES["breakout"]["close_buffer_previous_atr_multiple"] == 1.0
    assert tuple((name, atr_multiple, percent_multiple) for name, atr_multiple, percent_multiple in PIVOT_SCALES) == (
        ("2atr_5pct", 2.0, 0.05), ("3atr_8pct", 3.0, 0.08), ("5atr_12pct", 5.0, 0.12))


def test_long_cycle_w_bottom_uses_two_prior_independent_neckline_tests_and_confirms_causally():
    bars = synthetic_w_bars()
    # Stop at the first qualifying breakout so the later target touch does not
    # change the event's current terminal state in this confirmation test.
    found = detect_main_bottoms(bars[bars.date <= "2024-11-22"].copy(),
                                code="fixture", timeframe="1d")
    bottoms = main_events(found)

    assert bottoms
    selected = next(item for item in bottoms if is_selected(item))
    main = evidence(selected)["main_structure"]
    tests = [item for item in evidence(selected)["neckline_tests"] if item["valid"]]
    assert main["family"] == "w_bottom"
    assert main["scale"] == "5y"
    assert main["signal_eligible"] is True
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


def test_real_600108_does_not_use_external_33_pressure_as_the_2024_w_neckline():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-01-09"].reset_index(drop=True)
    found = detect_main_bottoms(bars, code="600108", timeframe="1d")
    bottoms = [item for item in main_events(found)
               if evidence(item)["main_structure"]["first_trough_date"] == "2024-02-06"
               and evidence(item)["main_structure"]["latest_anchor_date"] == "2024-09-18"]

    middle = bars.loc[(bars.date > "2024-02-06") & (bars.date < "2024-09-18")]
    rim_index = int(middle.high.idxmax())
    rim_price = float(bars.high.iloc[rim_index])
    analysis_start = bars.date.iloc[-1] - pd.DateOffset(years=5)
    analysis_window = bars[bars.date >= analysis_start].reset_index(drop=True)
    local_rim_index = int(analysis_window.index[analysis_window.date == bars.date.iloc[rim_index]][0])
    atr_values = (bars.high - bars.low).rolling(14, min_periods=1).mean()
    tolerance = max(0.02, 0.5 * float(atr_values.iloc[rim_index]))
    for item in bottoms:
        projected = item.neckline["slope"] * local_rim_index + item.neckline["intercept"]
        assert abs(projected - rim_price) <= tolerance
        assert item.confirmation_date != "2026-01-09"


def test_600108_w_neckline_must_match_the_intervening_rebound_not_external_33_pressure():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-10-08"].reset_index(drop=True)
    found = detect_main_bottoms(bars, code="600108", timeframe="1d", tick_size=0.01)

    matching = []
    for item in found:
        structure = evidence(item)["main_structure"]
        if (item.pattern_type == "w_bottom"
                and structure["first_trough_date"] == "2024-02-06"
                and structure["latest_anchor_date"] == "2024-09-18"):
            matching.append(item)
    assert found
    middle = bars.loc[(bars.date > "2024-02-06") & (bars.date < "2024-09-18")]
    rim = middle.loc[middle.high.idxmax()]
    assert np.isclose(float(rim.high), 2.87)
    analysis_start = bars.date.iloc[-1] - pd.DateOffset(years=5)
    analysis_window = bars[bars.date >= analysis_start].reset_index(drop=True)
    local_rim_index = int(analysis_window.index[analysis_window.date == rim.date][0])
    atr_values = (bars.high - bars.low).rolling(14, min_periods=1).mean()
    tolerance = max(0.02, 0.5 * float(atr_values.loc[rim.name]))
    for item in matching:
        neckline = item.neckline
        projected = neckline["slope"] * local_rim_index + neckline["intercept"]
        assert abs(projected - float(rim.high)) <= tolerance


def test_600108_main_w_uses_rim_occurrence_atr_and_confirms_on_october_8():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-10-08"].reset_index(drop=True)
    found = detect_main_bottoms(bars, code="600108", kind="stock", timeframe="1d",
                                tick_size=0.01, background_bars=bars)

    candidates = [item for item in found if item.pattern_type == "w_bottom"
                  and evidence(item)["main_structure"]["first_trough_date"] == "2024-02-06"
                  and evidence(item)["main_structure"]["latest_anchor_date"] == "2024-09-18"]

    assert candidates
    selected = next(item for item in candidates if is_selected(item))
    assert selected.confirmation_date == "2024-10-08"
    analysis_window = bars[bars.date >= bars.date.iloc[-1] - pd.DateOffset(years=5)].reset_index(drop=True)
    oct_8 = int(analysis_window.index[analysis_window.date == "2024-10-08"][0])
    assert np.isclose(selected.neckline["slope"] * oct_8 + selected.neckline["intercept"],
                      2.758, atol=0.02)


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


def test_516630_full_daily_history_uses_current_w_neckline_tests_as_of_2024_10_08():
    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    daily = pd.read_csv(source, parse_dates=["date"])
    daily = daily[daily.date <= "2024-10-08"].reset_index(drop=True)
    found = detect_main_bottoms(daily, code="516630", kind="etf", timeframe="1d",
                                background_bars=daily)

    candidates = [item for item in main_events(found)
                  if evidence(item)["main_structure"]["first_trough_date"] == "2024-02-05"]
    assert candidates
    selected = next(item for item in candidates if is_selected(item))
    detail = evidence(selected)
    structure = detail["main_structure"]
    tests = [item for item in detail["neckline_tests"] if item["valid"]]

    reference_touch = next(item for item in tests if item["pivot_date"] == "2024-03-21")
    effective_level = (structure["neckline"]["slope"] * reference_touch["pivot_index"]
                       + structure["neckline"]["intercept"])
    assert np.isclose(effective_level, 0.973, atol=0.01)
    assert detail["neckline_selection"]["rule_version"] == "joint_structure_boundary_v4_relative_height"
    height_check = detail["neckline_selection"]["neckline_relative_height"]
    assert height_check["maximum_ratio"] == 1.0
    assert 0 < height_check["relative_height"] <= 1.0
    assert tests[-1]["pivot_date"] >= "2024-02-05"
    assert all(item["level"] <= 0.973 * 1.05 for item in tests)
    assert selected.confirmation_date == "2024-10-08"


def test_159718_one_atr_session_confirms_same_ihs_once_before_later_w_signal(monkeypatch):
    monkeypatch.setitem(MAIN_RULES["neckline_tests"], "candidate_reversal_atr_multiple", 1.0)
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-30"].reset_index(drop=True)
    from caisen.daily_incremental import recognize_daily_history
    result = recognize_daily_history(bars, "159718", "etf", tick_size=0.01)
    anchor_dates = ["2023-08-22", "2024-04-19", "2025-01-13"]

    confirmed = [pattern for signal in result["signals"] for pattern in signal["patterns"]
                 if pattern["pattern_type"] == "inverse_head_shoulders"
                 and [pattern["key_points"].get(f"pivot_{number}_date") for number in (1, 2, 3)]
                 == anchor_dates]

    assert len(confirmed) == 1
    assert confirmed[0]["confirmation_date"] == "2025-04-01"
    lifecycle = next(item["value"] for item in confirmed[0]["evidence"]
                     if item["code"] == "joint_lifecycle")
    assert lifecycle["events"][-1]["kind"] == "confirmed"
    assert lifecycle["events"][-1]["date"] == "2025-04-01"
    assert confirmed[0]["neckline"]["slope"] <= 0
    assert not any(pattern["pattern_type"] == "inverse_head_shoulders"
                   and [pattern["key_points"].get(f"pivot_{number}_date") for number in (1, 2, 3)]
                   == anchor_dates
                   for signal in result["signals"] if signal["signal_date"] > "2025-04-01"
                   for pattern in signal["patterns"])
    assert any(signal["signal_date"] == "2025-05-29"
               and "w_bottom" in signal["pattern_types"] for signal in result["signals"])


def test_159718_two_atr_session_does_not_reissue_consumed_ihs_as_a_may_21_buy():
    source = Path(__file__).parents[2] / "data" / "etfs" / "159718.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2025-05-28"].reset_index(drop=True)
    from caisen.daily_incremental import recognize_daily_history
    result = recognize_daily_history(bars, "159718", "etf", tick_size=0.01)
    anchors = ["2023-08-22", "2024-04-19", "2025-01-13"]
    matching = [pattern for signal in result["signals"] for pattern in signal["patterns"]
                if pattern["pattern_type"] == "inverse_head_shoulders"
                and [pattern["key_points"].get(f"pivot_{number}_date") for number in (1, 2, 3)] == anchors]

    assert len(matching) == 1
    assert matching[0]["confirmation_date"] == "2025-04-01"
    assert not any(signal["signal_date"] == "2025-05-21"
                   and "inverse_head_shoulders" in signal["pattern_types"]
                   for signal in result["signals"])


def test_w_bottom_breakout_requires_atr_buffer_and_keeps_weak_crossings():
    bars = synthetic_w_bars()
    bars.loc[820:, "close"] = 3.34
    bars.loc[820:, "open"] = 3.33
    bars.loc[820:, "high"] = 3.36
    bars.loc[820:, "low"] = 3.32
    found = detect_main_bottoms(bars, code="fixture", timeframe="1d")
    candidates = [item for item in main_events(found) if evidence(item).get("weak_crossings")]
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


def test_two_atr_policy_rejects_known_000008_r02_no_book_pattern():
    source = Path(__file__).parent / "fixtures" / "main_bottoms" / "R02-000008.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    found = detect_main_bottoms(bars, code="000008", timeframe="1d")
    selected = [item for item in found if is_selected(item)
                and evidence(item)["main_structure"]["signal_eligible"]]

    assert selected == []


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


def test_expired_main_structure_cannot_suppress_a_later_bull_flag():
    expired_parent = Pattern(
        "expired-main", "fixture", "stock", "1d", "w_bottom", "bullish", "expired",
        "2024-02-06", None, last_observed_date="2024-09-01",
        evidence=[{"code": "main_structure", "value": {
            "role": "main", "signal_eligible": False,
            "first_trough_date": "2024-02-06", "frozen_at": "2024-03-01",
            "structure_end_date": "2024-09-01", "terminal_state": "expired"}},
                  {"code": "main_selection", "value": {"selected": True}}],
    )
    later_flag = Pattern(
        "later-flag", "fixture", "stock", "1d", "bull_flag", "bullish", "confirmed",
        "2024-10-01", 200, confirmation_date="2024-10-15",
        key_points={"pivot_1_date": "2024-10-01", "pivot_1_index": 10,
                    "pivot_2_date": "2024-10-05", "pivot_2_index": 12},
    )

    linked = link_main_bottom_children([expired_parent], [later_flag])

    assert linked[1].parent_id is None
    assert "execution_suppression" not in evidence(linked[1])


def test_parent_linking_persists_as_of_flag_suppression_in_lifecycle():
    parent = Pattern("main", "fixture", "stock", "1d", "w_bottom", "bullish", "confirmed",
                     "1994-01-03", 100, confirmation_date="1994-12-01",
                     evidence=[{"code": "main_structure", "value": {
                         "role": "main", "signal_eligible": True,
                         "first_trough_date": "1994-01-03", "frozen_at": "1994-01-10",
                         "structure_end_date": "1994-12-01"}},
                               {"code": "main_selection", "value": {"selected": True}}])
    flag = Pattern("flag", "fixture", "stock", "1d", "bull_flag", "bullish", "confirmed",
                   "1994-03-01", 50, confirmation_date="1994-03-14",
                       rule_version="caisen-joint-pattern-session-v6",
                   key_points={"pivot_1_date": "1994-03-01", "pivot_1_index": 40,
                               "pivot_2_date": "1994-03-10", "pivot_2_index": 47},
                   evidence=[{"code": "flag_geometry", "value": {"known_at": "1994-03-11"}},
                             {"code": "flag_breakout", "value": {"date": "1994-03-14", "close": 12.2,
                                   "frozen_boundary": 12.0, "required_close": 12.1}},
                                 {"code": "recognition_context", "value": {"recognition_experiment": None,
                                   "engine_version": "caisen-joint-pattern-session-v6"}}])

    linked = link_main_bottom_children([parent], [flag])
    linked_flag = next(pattern for pattern in linked if pattern.pattern_id == "flag")
    lifecycle = evidence(linked_flag)["joint_lifecycle"]
    suppression = evidence(linked_flag)["execution_suppression"]

    assert suppression["suppressed_at"] == "1994-03-14"
    assert lifecycle["entry_eligible_at_confirmation"] is False
    assert lifecycle["entry_eligibility_at_confirmation_status"] == "known_suppressed_at_confirmation"
    assert lifecycle["entry_eligible"] is False


def test_materially_rising_neckline_is_rejected_and_small_rise_is_flattened():
    from caisen.main_bottoms import _qualify_neckline_slope

    assert _qualify_neckline_slope({"slope": 0.0, "intercept": 10.0}, 10, 100) == {
        "slope": 0.0, "intercept": 10.0
    }
    flattened = _qualify_neckline_slope({"slope": 0.004, "intercept": 10.0}, 0, 100)
    assert flattened == {"slope": 0.0, "intercept": 10.2}
    assert _qualify_neckline_slope({"slope": 0.01, "intercept": 10.0}, 0, 100) is None


def test_span_first_neckline_experiment_prefers_longer_independent_test_span():
    from caisen.main_bottoms import _neckline_candidate_order

    candidates = [
        {"score": (-3, -40, 0.1, -100, 10, 50)},
        {"score": (-2, -100, 0.2, -120, 20, 120)},
    ]
    evidence_first = min(candidates, key=_neckline_candidate_order)
    config = RecognitionExperimentConfig(
        "neckline_span_first_v1", neckline_selection_order="span_first")
    with recognition_experiment(config):
        span_first = min(candidates, key=_neckline_candidate_order)

    assert evidence_first is candidates[0]
    assert span_first is candidates[1]
