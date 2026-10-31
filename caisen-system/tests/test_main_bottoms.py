from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from caisen.main_bottoms import (
    MAIN_RULES,
    PIVOT_SCALES,
    _independent_neckline_tests,
    _ihs_valley_intact,
    _structure_candidates,
    _triangle_boundaries_intact,
    _triangle_apex_index,
    detect_main_bottoms,
    detect_multiscale_pivots,
    link_main_bottom_children,
)
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
    assert MAIN_RULES["source_rule_version"] == "caisen-main-bottoms-v1"
    assert MAIN_RULES["timeframes"]["1d"]["structure_window_calendar_years"] == 5
    assert MAIN_RULES["timeframes"]["1w"]["structure_window_calendar_years"] == 10
    assert MAIN_RULES["minimum_duration_calendar_months_by_timeframe"] == {"1d": 6, "1w": 12}
    assert MAIN_RULES["minimum_duration_calendar_months"] == 6
    assert MAIN_RULES["neckline_tests"]["minimum_independent_groups"] == 2
    assert MAIN_RULES["breakout"]["close_buffer_previous_atr_multiple"] == 1.0
    assert tuple((name, atr_multiple, percent_multiple) for name, atr_multiple, percent_multiple in PIVOT_SCALES) == (
        ("2atr_5pct", 2.0, 0.05), ("3atr_8pct", 3.0, 0.08), ("5atr_12pct", 5.0, 0.12))


def test_long_cycle_w_bottom_uses_two_prior_independent_neckline_tests_and_confirms_causally():
    bars = synthetic_w_bars()
    # Stop at the first qualifying breakout so the later target touch does not
    # change the event's current terminal state in this confirmation test.
    found = detect_main_bottoms(bars[bars.date <= "2024-12-31"].copy(),
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


def test_real_600108_prefix_recognizes_the_frozen_2024_w_and_jan_2026_breakout():
    source = Path(__file__).parents[2] / "data" / "stocks" / "600108.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-01-09"].reset_index(drop=True)
    found = detect_main_bottoms(bars, code="600108", timeframe="1d")
    bottoms = [item for item in main_events(found)
               if item.formed_at == "2024-02-06" and is_selected(item)]

    assert bottoms
    selected = bottoms[0]
    detail = evidence(selected)
    structure = detail["main_structure"]
    assert selected.confirmation_date == "2026-01-09"
    assert selected.key_points["support_lower"] == 2.22
    assert selected.key_points["support_upper"] == 2.25
    breakout = detail["neckline_breakout"]
    projected_neckline = (structure["neckline"]["slope"] * breakout["index"]
                          + structure["neckline"]["intercept"])
    assert abs(projected_neckline - 3.3) <= 0.05
    tests = [item for item in detail["neckline_tests"] if item["valid"]]
    assert len(tests) >= MAIN_RULES["neckline_tests"]["minimum_independent_groups"]
    selected_tests = tests[:2]
    assert selected_tests[0]["pivot_index"] < selected_tests[1]["pivot_index"]
    assert selected_tests[1]["pivot_index"] - selected_tests[0]["pivot_index"] >= (
        MAIN_RULES["neckline_tests"]["minimum_separation_daily_sessions"]
    )
    assert selected_tests[0]["pair_id"] == selected_tests[1]["pair_id"]
    assert all(item["departure_atr"] >= MAIN_RULES["neckline_tests"]["minimum_departure_atr"]
               for item in selected_tests)
    assert structure["frozen_at"] >= structure["latest_anchor_date"]
    assert all(item["known_date"] <= structure["frozen_at"] for item in selected_tests)
    assert structure["frozen_at"] <= "2026-01-09"
    assert structure["signal_is_volume_independent"] is True


def test_516630_weekly_ihs_with_7pct_rising_neckline_is_excluded():
    source = Path(__file__).parents[2] / "data" / "etfs" / "516630.csv"
    daily = pd.read_csv(source, parse_dates=["date"])
    from caisen.market import aggregate_weekly, default_china_calendar

    weekly = aggregate_weekly(daily, calendar=default_china_calendar())
    found = detect_main_bottoms(weekly, code="516630", kind="etf", timeframe="1w",
                               background_bars=daily)

    # The two independent weekly resistance tests at 1.283 and 1.378 imply a
    # 7.4% cumulative relative rise; that IHS is outside the configured 5% gate.
    assert (1.378 - 1.283) / 1.283 > 0.05
    assert not any(item.pattern_type == "inverse_head_shoulders" for item in found)
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

    assert np.isclose(structure["neckline"]["intercept"], 0.973)
    assert tests[-1]["pivot_date"] >= "2024-02-05"
    assert all(item["level"] <= 0.973 * 1.05 for item in tests)
    assert selected.confirmation_date == "2024-10-08"


def test_w_bottom_breakout_requires_atr_buffer_and_keeps_weak_crossings():
    bars = synthetic_w_bars()
    bars.loc[820:, "close"] = 3.34
    bars.loc[820:, "open"] = 3.33
    bars.loc[820:, "high"] = 3.36
    bars.loc[820:, "low"] = 3.32
    found = detect_main_bottoms(bars, code="fixture", timeframe="1d")
    selected = next(item for item in main_events(found) if is_selected(item))
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
                     str(dates[7].date()), "3ATR|8%|neckline_test_1ATR"),
               Pivot(25, str(dates[25].date()), 3.6, "high", 27,
                     str(dates[27].date()), "3ATR|8%|neckline_test_1ATR")]
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
    candidates = _structure_candidates(swings, "1w")
    ihs = [anchors for family, anchors in candidates if family == "inverse_head_shoulders"]
    triangles = [anchors for family, anchors in candidates if family == "triangle_bottom"]

    assert any([point.index for point in anchors] == [0, 7, 14] for anchors in ihs)
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
                        if [point.index for point in anchors] == [0, 7, 14])
    assert _ihs_valley_intact(selected_ihs[0], selected_ihs[1], selected_ihs[2], swings)
    head = selected_ihs[1]
    damaged_ihs = swings + [Pivot(6, head.date, head.price - 0.2, "low", 7,
                                  head.confirmed_date, head.scale)]
    assert not _ihs_valley_intact(selected_ihs[0], selected_ihs[1], selected_ihs[2], damaged_ihs)


def test_main_events_from_real_review_fixtures_remain_json_safe_and_causal():
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


def test_materially_rising_neckline_is_rejected_and_small_rise_is_flattened():
    from caisen.main_bottoms import _qualify_neckline_slope

    assert _qualify_neckline_slope({"slope": 0.0, "intercept": 10.0}, 10, 100) == {
        "slope": 0.0, "intercept": 10.0
    }
    flattened = _qualify_neckline_slope({"slope": 0.004, "intercept": 10.0}, 0, 100)
    assert flattened == {"slope": 0.0, "intercept": 10.2}
    assert _qualify_neckline_slope({"slope": 0.01, "intercept": 10.0}, 0, 100) is None
