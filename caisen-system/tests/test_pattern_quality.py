import pandas as pd

from caisen.patterns import _duration_quality, _low_position_quality


def history_frame(dates, lows, highs):
    close = [(low + high) / 2 for low, high in zip(lows, highs)]
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": close, "high": highs,
        "low": lows, "close": close, "volume": [1000.0] * len(close),
    })


def test_structure_must_age_strictly_more_than_three_calendar_months():
    start = pd.Timestamp("2025-01-31")

    boundary = _duration_quality(start, pd.Timestamp("2025-04-30"))
    after = _duration_quality(start, pd.Timestamp("2025-05-01"))

    assert boundary["eligible"] is False
    assert after["eligible"] is True
    assert boundary["eligible_after"] == "2025-04-30"


def test_first_break_before_maturity_permanently_rejects_structure():
    result = _duration_quality("2025-01-01", "2025-06-01", "2025-03-15")

    assert result["eligible"] is False
    assert result["early_break"] is True


def test_low_position_branch_passes_using_only_history_known_at_observation():
    background = history_frame(
        ["2025-01-01", "2025-03-01", "2025-06-01"],
        [25.0, 20.0, 20.0], [100.0, 90.0, 90.0],
    )
    future = history_frame(["2026-01-01"], [0.0], [1000.0])

    observed = _low_position_quality(25.0, "2025-01-01", background, "2025-06-01")
    extended = pd.concat([background, future], ignore_index=True)
    same_cutoff = _low_position_quality(25.0, "2025-01-01", extended, "2025-06-01")

    assert observed["passed"] is True
    assert observed["position_pass"] is True
    assert observed["drawdown_pass"] is False
    assert observed["position_ratio"] < 0.30
    assert same_cutoff == observed


def test_later_extreme_high_cannot_rescue_a_low_that_failed_at_signal_time():
    at_signal = history_frame(["2025-01-01", "2025-05-01"], [50.0, 60.0], [100.0, 100.0])
    later = history_frame(["2025-06-01"], [1.0], [1000.0])

    rejected = _low_position_quality(90.0, "2025-01-01", at_signal, "2025-05-01")
    appended = pd.concat([at_signal, later], ignore_index=True)
    still_rejected = _low_position_quality(90.0, "2025-01-01", appended, "2025-05-01")

    assert rejected["position_pass"] is False
    assert rejected["drawdown_pass"] is False
    assert still_rejected == rejected


def test_prepattern_drawdown_branch_can_pass_when_range_position_fails():
    background = history_frame(
        ["2024-01-01", "2024-12-31", "2025-01-01", "2025-05-01"],
        [90.0, 40.0, 69.0, 69.0], [100.0, 70.0, 75.0, 75.0],
    )

    result = _low_position_quality(69.0, "2025-01-01", background, "2025-05-01")

    assert result["position_pass"] is False
    assert result["drawdown_pass"] is True
    assert result["passed"] is True


def test_invalid_or_empty_background_does_not_pass_low_position():
    flat = history_frame(["2025-01-01", "2025-05-01"], [10.0, 10.0], [10.0, 10.0])

    result = _low_position_quality(10.0, "2025-01-01", flat, "2025-05-01")
    empty = _low_position_quality(10.0, "2025-01-01", flat.iloc[0:0], "2025-05-01")

    assert result["passed"] is False
    assert empty["passed"] is False


def test_indexed_history_lookup_matches_naive_ranges_with_unsorted_rows_and_missing_prices():
    background = history_frame(
        ["2024-12-01", "2025-01-01", "2024-01-01", "2024-06-01", "2025-02-01"],
        [12.0, 20.0, float("nan"), 15.0, 8.0],
        [18.0, 30.0, 25.0, float("nan"), 22.0],
    )
    for asof in ("2024-12-15", "2025-01-15", "2025-02-01"):
        observed = pd.Timestamp(asof)
        start = pd.Timestamp("2025-01-01")
        known = background.copy()
        known["date"] = pd.to_datetime(known.date)
        known = known[known.date <= observed].sort_values("date")
        lows = pd.to_numeric(known.low, errors="coerce").dropna()
        highs = pd.to_numeric(known.high, errors="coerce").dropna()
        expected = {"structure_low": 20.0, "passed": False,
                    "position_pass": False, "drawdown_pass": False}
        if not lows.empty and not highs.empty:
            low, high = float(lows.min()), float(highs.max())
            expected.update({"history_start": str(known.date.min().date()),
                             "history_end": str(known.date.max().date()),
                             "range_low": low, "range_high": high,
                             "position_threshold": 0.30, "position_ratio": None,
                             "prepattern_high": None, "prepattern_drawdown": None,
                             "drawdown_threshold": 0.30})
            if high > low:
                ratio = (20.0 - low) / (high - low)
                expected["position_ratio"] = float(ratio)
                expected["position_pass"] = bool(0 <= ratio <= .30)
            prior = known[(known.date >= start - pd.DateOffset(years=2)) & (known.date < start)]
            prior_high = pd.to_numeric(prior.high, errors="coerce").dropna()
            if not prior_high.empty and float(prior_high.max()) > 0:
                top = float(prior_high.max())
                drawdown = (top - 20.0) / top
                expected["prepattern_high"] = top
                expected["prepattern_drawdown"] = float(drawdown)
                expected["drawdown_pass"] = bool(drawdown >= .30)
            expected["passed"] = bool(expected["position_pass"] or expected["drawdown_pass"])

        actual = _low_position_quality(20.0, start, background, observed)
        assert actual == expected


def test_future_new_low_cannot_remove_a_confirmed_pattern_that_passed_at_signal_time():
    from caisen.patterns import _add
    from caisen.schemas import Pivot

    count = 151
    dates = pd.date_range("2025-01-01", periods=count, freq="D")
    close = [75.0] * count
    high = [80.0] * count
    low = [70.0] * count
    for index in range(10, count):
        close[index], high[index], low[index] = 85.0, 90.0, 80.0
    for index, value in ((10, 60.0), (40, 100.0), (70, 65.0), (100, 95.0), (110, 62.0)):
        close[index] = value
        high[index] = value + 0.1
        low[index] = value - 0.1 if index in (40, 100) else value
    for index in range(140, count):
        close[index], high[index], low[index] = 101.0, 101.0, 80.0
    frame = pd.DataFrame({"date": dates, "open": close, "high": high,
                          "low": low, "close": close, "volume": [1000.0] * count})
    pivots = [Pivot(index, str(dates[index].date()), price, kind, index + 2,
                    str(dates[index + 2].date()))
              for index, kind, price in [(10, "low", 60.0), (40, "high", 100.0),
                                         (70, "low", 65.0), (100, "high", 95.0),
                                         (110, "low", 62.0)]]

    def recognize(bars):
        result = []
        _add(result, bars, "fixture", "stock", "1d", "w_bottom", pivots,
             float(bars.close.iloc[-1]), {"slope": 0.0, "intercept": 100.0},
             55.0, [2000.0], 110)
        return result

    before = recognize(frame)
    future = history_frame([dates[-1] + pd.Timedelta(days=1)], [0.0], [1000.0])
    future["close"] = 101.0
    future["open"] = 101.0
    extended = pd.concat([frame, future], ignore_index=True)
    after = recognize(extended)

    assert len(before) == 1
    assert before[0].state == "confirmed"
    assert len(after) == 1
    assert after[0].pattern_id == before[0].pattern_id
    evidence = {item["code"]: item["value"] for item in after[0].evidence}
    assert evidence["low_position"]["position_pass"] is True
    assert evidence["low_position"]["history_end"] == str(dates[140].date())


def test_early_breakout_is_not_reintroduced_after_the_structure_matures():
    from caisen.patterns import _add
    from caisen.schemas import Pivot

    count = 160
    dates = pd.date_range("2025-01-01", periods=count, freq="D")
    close = [25.0] * count
    high = [42.0] * count
    low = [8.0] * count
    for index, value in ((10, 10.0), (40, 40.0), (70, 12.0), (100, 41.0), (110, 11.0)):
        close[index] = value
        if index in (40, 100):
            high[index] = value
        else:
            low[index] = value
    close[30] = 45.0  # first neckline break is before the three-month boundary
    high[30] = 45.0
    frame = pd.DataFrame({"date": dates, "open": close, "high": high,
                          "low": low, "close": close, "volume": [1000.0] * count})
    pivots = [Pivot(index, str(dates[index].date()), price, kind, index + 2,
                    str(dates[index + 2].date()))
              for index, kind, price in ((10, "low", 10.0), (40, "high", 40.0),
                                         (70, "low", 12.0), (100, "high", 41.0),
                                         (110, "low", 11.0))]
    found = []

    _add(found, frame, "fixture", "stock", "1d", "w_bottom", pivots,
         25.0, {"slope": 0.0, "intercept": 40.0}, 8.0, [70.0], 110)

    assert found == []
