import unittest
from unittest.mock import patch

import pandas as pd

from caisen.indicators import atr
from caisen.patterns import current_pattern_snapshot, detect_patterns
from caisen.pivots import detect_pivots
from caisen.schemas import Pattern, Pivot


def swing_frame(points, spacing=5):
    closes = []
    for left, right in zip(points, points[1:]):
        closes.extend([left + (right - left) * step / spacing for step in range(spacing)])
    closes.append(points[-1])
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=len(close)),
        "open": close, "high": close + 0.15, "low": close - 0.15,
        "close": close, "volume": [1_000_000.0] * len(close),
    })


def pivot_fixture(spec, close_overrides=None, length=40):
    close = [9.0] * length
    for index, value in (close_overrides or {}).items():
        close[index] = value
    highs = [value + .2 for value in close]
    lows = [value - .2 for value in close]
    dates = pd.bdate_range("2026-01-01", periods=length)
    pivots = []
    for index, kind, price in spec:
        if kind == "high":
            highs[index] = max(highs[index], price)
        else:
            lows[index] = min(lows[index], price)
        pivots.append(Pivot(index, str(dates[index].date()), price, kind, index + 2,
                            str(dates[index + 2].date())))
    frame = pd.DataFrame({"date": dates, "open": close, "high": highs,
                          "low": lows, "close": close, "volume": [1000.0] * length})
    return frame, pivots


def flag_fixture(spec, base, terminal_close):
    length = 60
    dates = pd.bdate_range("2026-01-01", periods=length)
    close = [float(base)] * length
    high = [float(base) + .2] * length
    low = [float(base) - .2] * length
    pivots = []
    for index, kind, price in spec:
        close[index] = float(price)
        if kind == "high":
            high[index] = float(price) + .2
        else:
            low[index] = float(price) - .2
        pivots.append(Pivot(index, str(dates[index].date()), float(price), kind,
                            index + 2, str(dates[index + 2].date())))
    close[-1] = float(terminal_close)
    high[-1] = max(high[-1], close[-1] + .2)
    low[-1] = min(low[-1], close[-1] - .2)
    frame = pd.DataFrame({"date": dates, "open": close, "high": high,
                          "low": low, "close": close, "volume": [1000.0] * length})
    return frame, pivots


class PatternTests(unittest.TestCase):
    def test_688216_user_clarification_has_no_active_book_pattern_at_label_date(self):
        from pathlib import Path
        from caisen.market import (InstrumentFile, aggregate_weekly, default_china_calendar,
                                   load_daily_bars)

        root = Path(__file__).parents[2]
        terminal = {"invalidated", "target_reached", "followthrough_confirmed"}
        for cutoff in ("2026-10-16", "2026-10-23"):
            loaded = load_daily_bars(InstrumentFile(
                "688216", "stock", str(root / "data" / "stocks" / "688216.csv")), root=root,
                as_of=cutoff)
            weekly = aggregate_weekly(loaded, asof=cutoff, calendar=default_china_calendar())
            for timeframe, bars in (("1d", loaded.bars), ("1w", weekly.bars)):
                snapshot = current_pattern_snapshot(detect_patterns(
                    bars, "688216", "stock", timeframe, asof=cutoff))
                active = [pattern for pattern in snapshot if pattern.state not in terminal]
                self.assertEqual(active, [], f"{timeframe} at {cutoff} should show no active book pattern")

        positive = load_daily_bars(InstrumentFile(
            "688330", "stock", str(root / "data" / "stocks" / "688330.csv")), root=root,
            as_of="2026-10-16")
        self.assertTrue(any(pattern.pattern_type == "w_bottom" and pattern.state == "forming"
                            for pattern in current_pattern_snapshot(detect_patterns(
                                positive.bars, "688330", "stock", "1d", asof="2026-10-16"))))

    def test_current_snapshot_keeps_latest_event_per_rule(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        failure = pd.DataFrame({
            "date": [frame.date.iloc[-1] + pd.offsets.BDay(1)], "open": [7.0],
            "high": [7.1], "low": [6.9], "close": [7.0], "volume": [1_000_000.0],
        })
        frame = pd.concat([frame, failure], ignore_index=True)
        patterns = detect_patterns(frame, code="000001", kind="stock", timeframe="1d")
        snapshot = current_pattern_snapshot(patterns)
        by_type = {}
        for pattern in snapshot:
            by_type.setdefault(pattern.pattern_type, []).append(pattern)
        self.assertTrue(all(len(items) == 1 for items in by_type.values()))
        self.assertTrue(snapshot)
        self.assertTrue(any(pattern.state == "invalidated" for pattern in snapshot))
        self.assertTrue(all(pattern.source_rule_version == "caisen-rules-v5" for pattern in snapshot))
        self.assertTrue(all(pattern.last_observed_date == str(frame.date.iloc[-1].date()) for pattern in snapshot))

    def test_current_snapshot_keeps_latest_invalidated_event_for_history(self):
        older = Pattern("old", "fixture", "stock", "1d", "w_bottom", "bullish", "confirmed",
                        "2026-01-01", 5, key_points={"pivot_1_index": 5})
        newer = Pattern("new", "fixture", "stock", "1d", "w_bottom", "bullish", "invalidated",
                        "2026-02-01", 10, key_points={"pivot_1_index": 10})

        snapshot = current_pattern_snapshot([older, newer])

        self.assertEqual([pattern.pattern_id for pattern in snapshot], ["new"])
        self.assertEqual(snapshot[0].state, "invalidated")

    def test_first_measured_target_marks_terminal_state_without_erasing_shape(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        initial = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        measured = next(pattern for pattern in initial
                        if pattern.pattern_type == "w_bottom" and pattern.state == "confirmed")
        future_close = max(measured.targets[0] * 1.1, 30.0)
        append = pd.DataFrame({
            "date": pd.bdate_range(frame.date.iloc[-1] + pd.Timedelta(days=1), periods=3),
            "open": [future_close] * 3, "high": [future_close + .1] * 3,
            "low": [future_close - .1] * 3, "close": [future_close] * 3,
            "volume": [1_000_000.0] * 3,
        })
        extended = pd.concat([frame, append], ignore_index=True)
        later = detect_patterns(extended, code="fixture", kind="stock", timeframe="1d")
        same_event = next(pattern for pattern in later if pattern.pattern_id == measured.pattern_id)

        self.assertEqual(same_event.state, "target_reached")

    def test_reclaim_pattern_stops_current_state_after_target_touch(self):
        from caisen.patterns import _add_reclaim_event

        close = [10.2, 9.8, 9.4, 10.6, 11.0, 12.2]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(0, "2026-01-01", 10.2, "low", 0, "2026-01-01"),
               Pivot(1, "2026-01-02", 11.0, "high", 1, "2026-01-02"),
               Pivot(2, "2026-01-05", 9.4, "low", 2, "2026-01-05")]
        found = []
        _add_reclaim_event(found, frame, "fixture", "stock", "1d", "spring", seq,
                           {"slope": 0.0, "intercept": 10.0}, 9.4, [11.5], 2, bullish=True)

        self.assertEqual(found[0].state, "target_reached")
        self.assertEqual(next(item for item in found[0].evidence if item["code"] == "target_reached")["value"]["price"], 11.5)

    def test_spring_box_followthrough_is_terminal_after_upper_boundary_break(self):
        from caisen.patterns import _add_reclaim_event

        close = [10.2, 9.8, 9.4, 10.6, 11.0, 12.2, 12.4]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(0, "2026-01-01", 10.2, "low", 0, "2026-01-01"),
               Pivot(1, "2026-01-02", 11.0, "high", 1, "2026-01-02"),
               Pivot(2, "2026-01-05", 9.4, "low", 2, "2026-01-05")]
        found = []
        _add_reclaim_event(found, frame, "fixture", "stock", "1d", "spring", seq,
                           {"slope": 0.0, "intercept": 10.0}, 9.4, [], 2, bullish=True,
                           extra={"box_lower": 10.0, "box_upper": 12.0})

        self.assertEqual(found[0].state, "followthrough_confirmed")
        evidence = {item["code"]: item["value"] for item in found[0].evidence}
        self.assertEqual(evidence["followthrough_confirmed"]["level"], 12.0)

    def test_false_breakout_box_followthrough_is_terminal_after_lower_boundary_break(self):
        from caisen.patterns import _add_reclaim_event

        close = [9.8, 10.2, 10.6, 9.7, 9.0, 7.8, 7.6]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(0, "2026-01-01", 9.8, "high", 0, "2026-01-01"),
               Pivot(1, "2026-01-02", 8.0, "low", 1, "2026-01-02"),
               Pivot(2, "2026-01-05", 10.6, "high", 2, "2026-01-05")]
        found = []
        _add_reclaim_event(found, frame, "fixture", "stock", "1d", "false_breakout", seq,
                           {"slope": 0.0, "intercept": 10.0}, 10.6, [], 2, bullish=False,
                           extra={"box_lower": 8.0, "box_upper": 10.0})

        self.assertEqual(found[0].state, "followthrough_confirmed")
        evidence = {item["code"]: item["value"] for item in found[0].evidence}
        self.assertEqual(evidence["followthrough_confirmed"]["level"], 8.0)

    def test_rejects_neckline_extrapolated_outside_the_pattern_price_range(self):
        from caisen.patterns import _add, _line

        close = [14.0, 10.0, 15.0, 13.0, 14.5, 14.0]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i, str(frame.date.iloc[i].date()))
               for i, (kind, price) in enumerate([
                   ("high", 14.0), ("low", 10.0), ("high", 15.0),
                   ("low", 13.0), ("high", 14.5),
               ])]
        found = []
        _add(found, frame, "fixture", "stock", "1d", "m_top", seq, close[-1],
             _line((1, 10.0), (3, 13.0)), 15.0, [12.0], 4, direction="bearish")

        self.assertEqual(found, [])

    def test_neckline_that_later_extrapolates_below_pattern_lows_ends_candidate(self):
        from caisen.patterns import _add, _line

        close = [14.0, 10.0, 15.0, 9.0, 14.5] + [10.0] * 16
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i, str(frame.date.iloc[i].date()))
               for i, (kind, price) in enumerate([
                   ("high", 14.0), ("low", 10.0), ("high", 15.0),
                   ("low", 9.8), ("high", 14.5),
               ])]
        found = []
        _add(found, frame, "fixture", "stock", "1d", "m_top", seq, close[-1],
             _line((1, 10.0), (3, 9.9)), 15.0, [5.0], 4, direction="bearish")

        self.assertEqual(found[0].state, "invalidated")
        evidence = {item["code"]: item["value"] for item in found[0].evidence}
        self.assertIn("neckline_projection_invalidation", evidence)

    def test_rejects_nonpositive_first_measured_target(self):
        from caisen.patterns import _add, _line

        close = [10.0, 8.0, 12.0, 9.0, 11.0, 7.0]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i, str(frame.date.iloc[i].date()))
               for i, (kind, price) in enumerate([
                   ("high", 10.0), ("low", 8.0), ("high", 12.0),
                   ("low", 9.0), ("high", 11.0),
               ])]
        found = []
        _add(found, frame, "fixture", "stock", "1d", "m_top", seq, close[-1],
             _line((1, 8.0), (3, 9.0)), 12.0, [-1.0], 4, direction="bearish")

        self.assertEqual(found, [])

    def test_head_shoulders_is_rejected_when_first_measured_target_is_nonpositive(self):
        from caisen.patterns import _add, _line

        close = [40.0] * 65
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        anchors = [(5, "high", 41.4), (15, "low", 28.95), (25, "high", 51.3),
                   (35, "low", 23.77), (55, "high", 42.2)]
        seq = []
        for index, kind, price in anchors:
            if kind == "high":
                frame.loc[index, "high"] = price
            else:
                frame.loc[index, "low"] = price
            day = str(frame.date.iloc[index].date())
            seq.append(Pivot(index, day, price, kind, index + 2,
                             str(frame.date.iloc[index + 2].date())))
        neckline = _line((15, 28.95), (35, 23.77))
        head_height = 51.3 - (neckline["slope"] * 25 + neckline["intercept"])
        invalid_target = neckline["slope"] * 55 + neckline["intercept"] - head_height
        self.assertLess(invalid_target, 0)

        found = []
        _add(found, frame, "688216", "stock", "1d", "head_shoulders", seq,
             close[-1], neckline, 51.3, [invalid_target], 55, direction="bearish")

        self.assertEqual(found, [])

    def test_unconfirmed_bullish_structure_is_invalidated_when_close_breaks_its_low(self):
        from caisen.patterns import _add, _line

        close = [10.0, 8.0, 11.0, 8.5, 7.7]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i, str(frame.date.iloc[i].date()))
               for i, (kind, price) in enumerate([
                   ("low", 8.0), ("high", 11.0), ("low", 8.5),
               ])]
        found = []
        _add(found, frame, "fixture", "stock", "1d", "w_bottom", seq, close[-1],
             _line((1, 11.0), (1, 11.0)), 8.0, [14.0], 2)

        self.assertEqual(found[0].state, "invalidated")
        evidence = {item["code"]: item["value"] for item in found[0].evidence}
        self.assertEqual(evidence["structure_invalidation"]["level"], 8.0)
        self.assertEqual(evidence["structure_invalidation"]["date"], "2026-01-07")

    def test_atr_uses_only_current_and_prior_bars(self):
        frame = swing_frame([10, 9, 11, 9, 12])
        before = atr(frame, period=3)
        changed = frame.copy()
        changed.loc[len(changed) - 1, "high"] = 1000
        after = atr(changed, period=3)
        self.assertEqual(before.iloc[-2], after.iloc[-2])

    def test_swing_pivots_keep_formation_and_confirmation_dates(self):
        frame = swing_frame([10, 9, 12, 9, 13, 8, 14])
        pivots = detect_pivots(frame, radius=2, atr_multiple=1.0)
        self.assertTrue(pivots)
        self.assertTrue(all(p.confirmed_at >= p.index for p in pivots))
        self.assertTrue(all(p.confirmed_at < len(frame) for p in pivots))

    def test_pivot_thresholds_and_known_dates_are_stable_after_future_extreme_bars(self):
        frame = swing_frame([10, 9, 12, 9, 13, 8, 14])
        before = detect_pivots(frame, radius=2, atr_multiple=1.0)
        append = pd.concat([frame.iloc[[-1]]] * 4, ignore_index=True)
        append["date"] = pd.bdate_range(frame.date.iloc[-1] + pd.Timedelta(days=1), periods=4)
        append["high"] = 10000
        append["low"] = 1
        append["open"] = 10
        append["close"] = 10
        extended = pd.concat([frame, append], ignore_index=True)
        after = detect_pivots(extended, radius=2, atr_multiple=1.0)
        old_known = [(p.index, p.kind, p.price, p.confirmed_at, p.confirmed_date) for p in before]
        new_known = [(p.index, p.kind, p.price, p.confirmed_at, p.confirmed_date) for p in after if p.confirmed_at < len(frame)]
        self.assertEqual(old_known, new_known)

    def test_pivots_do_not_backfill_missing_atr_from_a_future_bar(self):
        from unittest.mock import patch

        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=5),
            "open": [10.0, 11.0, 20.0, 11.0, 10.0],
            "high": [11.0, 12.0, 21.0, 12.0, 11.0],
            "low": [9.0, 10.0, 10.0, 10.0, 9.0],
            "close": [10.0, 11.0, 20.0, 11.0, 10.0],
            "volume": [100.0] * 5,
        })
        atr_values = pd.Series([1.0, 1.0, float("nan"), 100.0, 100.0])

        with patch("caisen.pivots.atr", return_value=atr_values):
            pivots = detect_pivots(frame, radius=2, atr_multiple=1.0)

        self.assertEqual(pivots, [])

    def test_break_before_last_pivot_is_known_does_not_confirm_pattern(self):
        from caisen.patterns import _add

        close = [10.0, 10.5, 9.0, 10.0, 11.0, 12.0, 11.5, 10.0,
                 9.0, 13.0, 13.0, 13.0]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        seq = [
            Pivot(2, str(frame.date.iloc[2].date()), 8.0, "low", 4, str(frame.date.iloc[4].date()), "1ATR"),
            Pivot(5, str(frame.date.iloc[5].date()), 12.0, "high", 7, str(frame.date.iloc[7].date()), "1ATR"),
            Pivot(8, str(frame.date.iloc[8].date()), 9.0, "low", 10, str(frame.date.iloc[10].date()), "1ATR"),
        ]
        found = []

        _add(found, frame, "fixture", "stock", "1d", "w_bottom", seq,
             close[-1], {"slope": 0.0, "intercept": 12.0}, 8.0, [15.0], 8)

        self.assertEqual(found[0].state, "confirmed")
        self.assertEqual(found[0].confirmed_at, 10)
        self.assertEqual(next(item["value"] for item in found[0].evidence
                              if item["code"] == "confirmation_index"), 10)

    def test_w_bottom_recognizes_unequal_lows_and_frozen_neckline(self):
        frame = swing_frame([12, 8.2, 11, 7.8, 12.5])
        patterns = detect_patterns(frame, code="000001", kind="stock", timeframe="1d")
        bottoms = [p for p in patterns if p.pattern_type == "w_bottom"]
        self.assertTrue(bottoms)
        pattern = next(p for p in bottoms if p.key_points.get("second_low", 0) < p.key_points.get("first_low", 0))
        self.assertLess(pattern.key_points["second_low"], pattern.key_points["first_low"])
        self.assertIsNotNone(pattern.neckline)
        self.assertEqual(pattern.rule_version, "caisen-v14")

    def test_w_bottom_exposes_geometry_diagnostics_without_changing_candidate_state(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        patterns = detect_patterns(frame, code="000001", kind="stock", timeframe="1d")
        pattern = next(p for p in patterns if p.pattern_type == "w_bottom" and p.state == "confirmed"
                       and next(item["value"] for item in p.evidence if item["code"] == "pivot_topology")
                       == ["low", "high", "low"])
        geometry = next(item["value"] for item in pattern.evidence if item["code"] == "w_geometry_metrics")
        self.assertEqual(geometry["first_leg_bars"], 5)
        self.assertEqual(geometry["second_leg_bars"], 5)
        self.assertEqual(geometry["leg_duration_ratio"], 1.0)
        self.assertEqual(geometry["trough_gap_pct"], 2.55)

    def test_w_bottom_with_highly_unbalanced_legs_is_not_a_book_pattern_candidate(self):
        from caisen.patterns import detect_patterns

        frame = swing_frame([10, 8, 12, 8.5, 10], spacing=10)
        dates = pd.to_datetime(frame.date)
        pivots = [Pivot(5, str(dates.iloc[5].date()), 8.0, "low", 7, str(dates.iloc[7].date())),
                  Pivot(10, str(dates.iloc[10].date()), 12.0, "high", 12, str(dates.iloc[12].date())),
                  Pivot(35, str(dates.iloc[35].date()), 8.5, "low", 37, str(dates.iloc[37].date()))]
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type == "w_bottom" for pattern in patterns))

    def test_m_top_with_highly_unbalanced_legs_is_not_a_book_pattern_candidate(self):
        frame = swing_frame([8, 12, 7, 11.5, 10], spacing=10)
        dates = pd.to_datetime(frame.date)
        pivots = [Pivot(5, str(dates.iloc[5].date()), 12.0, "high", 7, str(dates.iloc[7].date())),
                  Pivot(10, str(dates.iloc[10].date()), 7.0, "low", 12, str(dates.iloc[12].date())),
                  Pivot(35, str(dates.iloc[35].date()), 11.5, "high", 37, str(dates.iloc[37].date()))]
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type == "m_top" for pattern in patterns))

    def test_m_top_recognizes_balanced_three_pivot_structure(self):
        frame, pivots = pivot_fixture([(5, "high", 12), (10, "low", 8), (15, "high", 11.8)])
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertTrue(any(pattern.pattern_type == "m_top" for pattern in patterns))

    def test_head_shoulders_recognizes_five_pivot_topology(self):
        frame, pivots = pivot_fixture([(5, "high", 10), (10, "low", 8.5), (15, "high", 13),
                                       (20, "low", 8.5), (25, "high", 10.2)])
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertTrue(any(pattern.pattern_type == "head_shoulders" for pattern in patterns))

    def test_head_shoulders_rejects_a_sequence_without_a_higher_head(self):
        frame, pivots = pivot_fixture([(5, "high", 10), (10, "low", 8.5), (15, "high", 10.1),
                                       (20, "low", 8.5), (25, "high", 10.2)])
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type == "head_shoulders" for pattern in patterns))

    def test_inverse_head_shoulders_recognizes_five_pivot_topology(self):
        frame, pivots = pivot_fixture([(5, "low", 8), (10, "high", 10.5), (15, "low", 6),
                                       (20, "high", 10.5), (25, "low", 8.1)])
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertTrue(any(pattern.pattern_type == "inverse_head_shoulders" for pattern in patterns))

    def test_converging_triangles_recognize_both_directions(self):
        bottom_frame, bottom_pivots = pivot_fixture([(5, "low", 8), (10, "high", 12), (15, "low", 9),
                                                     (20, "high", 11.5), (25, "low", 10)])
        top_frame, top_pivots = pivot_fixture([(5, "high", 12), (10, "low", 8), (15, "high", 11),
                                               (20, "low", 8.5), (25, "high", 10)])
        with patch("caisen.patterns.detect_pivots", return_value=bottom_pivots):
            bottom = detect_patterns(bottom_frame, code="fixture", kind="stock", timeframe="1d")
        with patch("caisen.patterns.detect_pivots", return_value=top_pivots):
            top = detect_patterns(top_frame, code="fixture", kind="stock", timeframe="1d")
        self.assertTrue(any(pattern.pattern_type == "triangle_bottom" for pattern in bottom))
        self.assertTrue(any(pattern.pattern_type == "triangle_top" for pattern in top))

    def test_triangle_bottom_rejects_nonconverging_boundary_lines(self):
        frame, pivots = pivot_fixture([(5, "low", 9), (10, "high", 12), (15, "low", 8.5),
                                       (20, "high", 11), (25, "low", 8)])
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type == "triangle_bottom" for pattern in patterns))

    def test_triangle_top_rejects_nonconverging_boundary_lines(self):
        frame, pivots = pivot_fixture([(5, "high", 12), (10, "low", 8), (15, "high", 11),
                                       (20, "low", 8.5), (25, "high", 13)])
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type == "triangle_top" for pattern in patterns))

    def test_false_breakout_head_shoulders_requires_a_neckline_break_and_reclaim(self):
        spec = [(5, "high", 10), (10, "low", 8.5), (15, "high", 13),
                (20, "low", 8.5), (25, "high", 10.2)]
        frame, pivots = pivot_fixture(spec, {26: 7.5, 27: 9.0})
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertTrue(any(pattern.pattern_type == "false_breakout_head_shoulders" for pattern in patterns))

        no_reclaim_frame, no_reclaim_pivots = pivot_fixture(spec)
        with patch("caisen.patterns.detect_pivots", return_value=no_reclaim_pivots):
            no_reclaim = detect_patterns(no_reclaim_frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type == "false_breakout_head_shoulders" for pattern in no_reclaim))

    def test_converging_triangle_balance_gate_checks_every_touch_interval(self):
        from caisen.patterns import _duration_balance_ratio, _passes_duration_balance

        dates = pd.bdate_range("2026-01-01", periods=70)
        pivots = [Pivot(index, str(dates[index].date()), price, kind, index + 2,
                        str(dates[index + 2].date()))
                  for index, price, kind in [(5, 100, "low"), (19, 150, "high"),
                                             (25, 120, "low"), (36, 140, "high"),
                                             (65, 130, "low")]]
        self.assertEqual(round(_duration_balance_ratio("triangle_bottom", pivots), 2), 4.83)
        self.assertFalse(_passes_duration_balance("triangle_bottom", pivots))

    def test_pattern_result_has_no_confirmation_after_asof(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        end = frame.iloc[:-2].copy()
        patterns = detect_patterns(end, code="000001", kind="stock", timeframe="1d")
        self.assertTrue(all(p.confirmed_at < len(end) for p in patterns))

    def test_confirmed_pattern_identity_and_confirmation_are_stable_after_future_bars(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        earlier = detect_patterns(frame, code="000001", kind="stock", timeframe="1d")
        confirmed = {p.pattern_id: p for p in earlier if p.state == "confirmed"}
        self.assertTrue(confirmed)
        future = frame.iloc[[-1]].loc[frame.index[-1]:].copy()
        append = pd.concat([future] * 15, ignore_index=True)
        append["date"] = pd.bdate_range(frame.date.iloc[-1] + pd.Timedelta(days=1), periods=len(append))
        extended = pd.concat([frame, append], ignore_index=True)
        later = {p.pattern_id: p for p in detect_patterns(extended, code="000001", kind="stock", timeframe="1d")}
        for pattern_id, old in confirmed.items():
            self.assertIn(pattern_id, later)
            new = later[pattern_id]
            self.assertEqual(new.formed_at, old.formed_at)
            self.assertEqual(new.confirmed_at, old.confirmed_at)
            self.assertEqual(new.key_points, old.key_points)
            self.assertEqual(new.neckline, old.neckline)

    def test_pattern_geometry_never_crosses_an_unverified_data_break(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        frame["segment_id"] = 0
        frame.loc[frame.index[15]:, "segment_id"] = 1
        patterns = detect_patterns(frame, code="000001", kind="stock", timeframe="1d")
        for pattern in patterns:
            indices = [value for key, value in pattern.key_points.items() if key.startswith("pivot_") and key.endswith("_index")]
            if indices:
                self.assertEqual(frame.segment_id.iloc[indices].nunique(), 1)

    def test_pattern_is_not_actionable_after_a_later_data_break(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        original = detect_patterns(frame, code="000001", kind="stock", timeframe="1d")
        expected = next(p for p in original if p.pattern_type == "w_bottom" and p.state == "confirmed")
        append = pd.concat([frame.iloc[[-1]]] * 4, ignore_index=True)
        append["date"] = pd.bdate_range(frame.date.iloc[-1] + pd.Timedelta(days=1), periods=len(append))
        extended = pd.concat([frame, append], ignore_index=True)
        extended["segment_id"] = [0] * len(frame) + [1] * len(append)
        later = {p.pattern_id: p for p in detect_patterns(extended, code="000001", kind="stock", timeframe="1d")}
        self.assertEqual(later[expected.pattern_id].state, "invalidated")
        prefix_again = {p.pattern_id: p for p in detect_patterns(frame, code="000001", kind="stock", timeframe="1d")}
        self.assertEqual(prefix_again[expected.pattern_id].state, "confirmed")

    def test_open_week_never_contributes_a_confirmed_weekly_pattern(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5], spacing=5)
        frame["date"] = pd.date_range("2020-01-06", periods=len(frame), freq="B")
        from caisen.market import aggregate_weekly
        weekly = aggregate_weekly(frame, asof=frame.iloc[-1]["date"])
        patterns = detect_patterns(weekly, code="000001", kind="stock", timeframe="1w") if len(weekly) >= 20 else []
        self.assertTrue(all(p.confirmed_at < len(weekly) for p in patterns))

    def test_three_day_jump_and_small_pullback_are_not_a_classical_flag(self):
        close = []
        for index in range(70):
            if index < 45:
                value = 10 + index * 0.03
            elif index < 49:
                value = 11.35 + (index - 45) * 0.30
            elif index < 63:
                value = 12.55 - (index - 49) * 0.035
            else:
                value = 12.06 + (index - 63) * 0.10
            close.append(value)
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .03 for value in close],
            "low": [value - .03 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertFalse(any(pattern.pattern_type in ("bull_flag", "bear_flag") for pattern in patterns))

    def test_flag_search_includes_valid_structures_older_than_last_ten_pivots(self):
        close = [110.0] * 501
        close[-1] = 99.0
        frame = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(close)),
            "open": close, "high": [value + .2 for value in close],
            "low": [value - .2 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        points = [(5, "high", 120), (10, "low", 100), (20, "high", 106),
                  (30, "low", 102), (40, "high", 109)]
        later = [
            (50, "low", 98), (60, "high", 112), (70, "low", 97), (80, "high", 115),
            (90, "low", 95), (100, "high", 117), (110, "low", 94), (120, "high", 118),
            (130, "low", 93), (140, "high", 119), (150, "low", 92),
        ]
        pivots = [Pivot(index, str(frame.date.iloc[index].date()), float(price), kind, index,
                        str(frame.date.iloc[index].date())) for index, kind, price in points + later]
        from caisen.patterns import _detect_flags
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            flags = [pattern for pattern in _detect_flags(frame, "fixture", "stock", "1d")
                     if pattern.pattern_type == "bear_flag"]

        self.assertTrue(flags)
        self.assertEqual(flags[0].key_points["pivot_1_index"], 5)

    def test_flag_detector_recognizes_valid_bull_and_bear_continuations(self):
        from caisen.patterns import _detect_flags

        bull_frame, bull_pivots = flag_fixture(
            [(5, "low", 10), (15, "high", 20), (25, "low", 18),
             (35, "high", 19), (45, "low", 15)], base=16.5, terminal_close=20.5)
        bear_frame, bear_pivots = flag_fixture(
            [(5, "high", 20), (15, "low", 10), (25, "high", 12),
             (35, "low", 11), (45, "high", 15)], base=12.5, terminal_close=9.5)
        with patch("caisen.patterns.detect_pivots", side_effect=[bull_pivots, bear_pivots]):
            bull = _detect_flags(bull_frame, "fixture", "stock", "1d")
            bear = _detect_flags(bear_frame, "fixture", "stock", "1d")

        self.assertTrue(any(pattern.pattern_type == "bull_flag" for pattern in bull))
        self.assertTrue(any(pattern.pattern_type == "bear_flag" for pattern in bear))

    def test_flag_detector_rejects_overdeep_countertrend_correction(self):
        from caisen.patterns import _detect_flags

        frame, pivots = flag_fixture(
            [(5, "low", 10), (15, "high", 20), (25, "low", 18),
             (35, "high", 19), (45, "low", 14)], base=16.5, terminal_close=20.5)
        with patch("caisen.patterns.detect_pivots", return_value=pivots):
            patterns = _detect_flags(frame, "fixture", "stock", "1d")

        self.assertFalse(any(pattern.pattern_type == "bull_flag" for pattern in patterns))

    def test_w_topology_does_not_automatically_create_a_spring_label(self):
        frame = swing_frame([12, 8, 11, 8.2, 12.5])
        patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")
        self.assertTrue(any(pattern.pattern_type == "w_bottom" for pattern in patterns))
        spring_events = [pattern for pattern in patterns if pattern.pattern_type == "spring"]
        self.assertFalse(spring_events)

    def test_two_bottoms_with_break_and_reclaim_do_not_create_spring_without_a_box(self):
        close = [8.0, 10.0, 8.2, 7.8, 8.4, 8.6, 8.5, 8.4]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        pivots = [Pivot(0, "2026-01-01", 8.0, "low", 0, "2026-01-01"),
                  Pivot(1, "2026-01-02", 10.0, "high", 1, "2026-01-02"),
                  Pivot(2, "2026-01-05", 8.2, "low", 2, "2026-01-05")]
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")

        self.assertFalse(any(pattern.pattern_type == "spring" for pattern in patterns))

    def test_double_top_break_and_reclaim_do_not_create_false_breakout_without_a_box(self):
        close = [10.0, 8.0, 10.2, 10.4, 9.9, 9.8, 9.7, 9.6]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [10.0, 8.1, 10.2, 10.5, 10.0, 9.9, 9.8, 9.7],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        pivots = [Pivot(0, "2026-01-01", 10.0, "high", 0, "2026-01-01"),
                  Pivot(1, "2026-01-02", 8.0, "low", 1, "2026-01-02"),
                  Pivot(2, "2026-01-05", 10.2, "high", 2, "2026-01-05")]
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")

        self.assertFalse(any(pattern.pattern_type == "false_breakout" for pattern in patterns))

    def test_nested_w_bottom_break_and_reclaim_does_not_create_w_bottom_spring_without_inner_box(self):
        close = [8.0, 10.0, 7.0, 9.5, 8.1, 7.8, 8.2, 8.4]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .1 for value in close],
            "low": [value - .1 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        pivots = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i,
                        str(frame.date.iloc[i].date()))
                  for i, (kind, price) in enumerate([
                      ("low", 8.0), ("high", 10.0), ("low", 7.0),
                      ("high", 9.5), ("low", 8.1),
                  ])]
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")

        self.assertFalse(any(pattern.pattern_type == "w_bottom_spring" for pattern in patterns))

    def test_prior_consolidation_box_breakdown_and_reclaim_creates_spring_candidate(self):
        close = [8.8, 8.5, 8.0, 8.6, 9.2, 9.8, 9.3, 8.7, 8.1, 8.8, 9.3, 9.7,
                 9.2, 8.8, 8.2, 9.0, 9.9, 9.4, 7.7, 7.9, 8.4, 8.7, 8.9]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .5 for value in close],
            "low": [value - .5 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        pivot_spec = [(2, "low", 8.0), (8, "high", 10.0),
                      (8, "low", 8.1), (16, "high", 10.1)]
        pivots = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i,
                        str(frame.date.iloc[i].date())) for i, kind, price in pivot_spec]
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")

        spring = [pattern for pattern in patterns if pattern.pattern_type == "spring"]
        self.assertEqual(len(spring), 1)
        self.assertEqual(spring[0].state, "confirmed")
        evidence = {item["code"]: item["value"] for item in spring[0].evidence}
        self.assertEqual(evidence["box_lower"], 8.05)
        self.assertEqual(evidence["box_upper"], 10.05)
        self.assertEqual(spring[0].targets, [])

    def test_box_spring_nested_inside_outer_w_bottom_creates_composite_candidate(self):
        close = [8.8, 8.5, 8.0, 8.6, 9.2, 9.8, 9.3, 8.7, 8.1, 8.8, 9.3, 9.7,
                 9.2, 8.8, 8.2, 9.0, 9.9, 9.4, 9.8, 9.4, 7.7, 7.9, 8.4, 9.2, 10.3]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .5 for value in close],
            "low": [value - .5 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        pivot_spec = [(2, "low", 8.0), (8, "high", 10.0),
                      (14, "low", 8.1), (18, "high", 10.1)]
        pivots = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i,
                        str(frame.date.iloc[i].date())) for i, kind, price in pivot_spec]
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")

        composite = [pattern for pattern in patterns if pattern.pattern_type == "w_bottom_spring"]
        self.assertEqual(len(composite), 1)
        self.assertEqual(composite[0].state, "confirmed")
        parent = next(pattern for pattern in patterns if pattern.pattern_id == composite[0].parent_id)
        self.assertEqual(parent.state, "invalidated")
        self.assertIsNotNone(composite[0].parent_id)
        self.assertIn("manual_review_required", composite[0].flags)
        composite_evidence = {item["code"] for item in composite[0].evidence}
        self.assertIn("outer_w_invalidation", composite_evidence)
        self.assertNotIn("structure_invalidation", composite_evidence)

    def test_prior_consolidation_box_breakout_and_reclaim_creates_false_breakout_candidate(self):
        close = [4.9, 4.8, 5.0, 4.9, 4.85, 5.02, 4.9, 4.85, 5.0, 4.9, 4.95,
                 5.0, 4.9, 5.02, 4.9, 4.84, 4.57, 4.7, 5.32, 5.2, 4.99, 4.9, 4.85]
        frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=len(close)),
            "open": close, "high": [value + .5 for value in close],
            "low": [value - .5 for value in close], "close": close,
            "volume": [1000.0] * len(close),
        })
        pivot_spec = [(2, "high", 5.24), (5, "low", 4.81),
                      (8, "high", 5.08), (16, "low", 4.57)]
        pivots = [Pivot(i, str(frame.date.iloc[i].date()), price, kind, i,
                        str(frame.date.iloc[i].date())) for i, kind, price in pivot_spec]
        with patch("caisen.patterns.detect_pivots", lambda *args, **kwargs: pivots):
            patterns = detect_patterns(frame, code="fixture", kind="stock", timeframe="1d")

        false_breakouts = [pattern for pattern in patterns if pattern.pattern_type == "false_breakout"]
        self.assertEqual(len(false_breakouts), 1)
        self.assertEqual(false_breakouts[0].state, "confirmed")
        evidence = {item["code"]: item["value"] for item in false_breakouts[0].evidence}
        self.assertAlmostEqual(evidence["box_lower"], 4.69)
        self.assertAlmostEqual(evidence["box_upper"], 5.16)
        self.assertEqual(false_breakouts[0].targets, [])


if __name__ == "__main__":
    unittest.main()
