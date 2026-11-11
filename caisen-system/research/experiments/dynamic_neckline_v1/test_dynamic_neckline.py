import pandas as pd

from dynamic_neckline import (build_boundary_revisions, build_breakout_signals,
                              pressure_points_2atr, structures_from_events)


def _bars(closes):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="B"),
        "open": closes,
        "high": [value + 0.1 for value in closes],
        "low": [value - 0.1 for value in closes],
        "close": closes,
        "volume": [1000] * len(closes),
    })


def test_revision_waits_for_known_structure_and_confirmed_pressure_points():
    bars = _bars([9.0, 10.0, 9.0, 11.0, 11.0, 13.5])
    structure = {
        "structure_key": "s1",
        "instance_id": "p1",
        "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[3].date().isoformat(),
        "anchors": [{"index": 0, "price": 8.0, "kind": "low"}],
    }
    pressure = [
        {"index": 1, "price": 10.1, "kind": "high", "confirmed_at": 2},
        {"index": 3, "price": 11.1, "kind": "high", "confirmed_at": 4},
    ]

    revisions = build_boundary_revisions(bars, structure, pressure, tick_size=0.01)

    assert len(revisions) == 1
    assert revisions[0]["known_at"] == bars.date.iloc[4].date().isoformat()
    assert revisions[0]["supporting_point_indices"] == [1, 3]


def test_breakout_must_follow_revision_and_cannot_be_backfilled():
    bars = _bars([9.0, 10.0, 9.0, 11.0, 11.0, 13.5])
    structure = {
        "structure_key": "s1",
        "instance_id": "p1",
        "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[3].date().isoformat(),
        "anchors": [{"index": 0, "price": 8.0, "kind": "low"}],
    }
    pressure = [
        {"index": 1, "price": 10.1, "kind": "high", "confirmed_at": 2},
        {"index": 3, "price": 11.1, "kind": "high", "confirmed_at": 4},
    ]
    revisions = build_boundary_revisions(bars, structure, pressure, tick_size=0.01)

    signals = build_breakout_signals(
        bars, structure, revisions, atr_multiple=0.0, minimum_buffer=0.01)

    assert all(signal["signal_date"] > revisions[0]["known_at"] for signal in signals)
    assert signals[0]["signal_date"] == bars.date.iloc[5].date().isoformat()


def test_revision_bar_crossing_does_not_create_a_signal():
    bars = _bars([9.0, 10.0, 9.0, 11.0, 13.5, 14.0])
    structure = {
        "structure_key": "s1",
        "instance_id": "p1",
        "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[3].date().isoformat(),
        "anchors": [{"index": 0, "price": 8.0, "kind": "low"}],
    }
    pressure = [
        {"index": 1, "price": 10.1, "kind": "high", "confirmed_at": 2},
        {"index": 3, "price": 11.1, "kind": "high", "confirmed_at": 4},
    ]
    revisions = build_boundary_revisions(bars, structure, pressure, tick_size=0.01)

    signals = build_breakout_signals(
        bars, structure, revisions, atr_multiple=0.0, minimum_buffer=0.01)

    assert signals == []


def test_distinct_new_neckline_for_same_lows_can_emit_a_later_buy_signal():
    bars = _bars([9.0, 9.0, 9.0, 10.4, 10.6, 10.5, 10.5, 12.4, 13.0])
    structure = {
        "structure_key": "same-two-lows",
        "instance_id": "same-two-lows-old-line",
        "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[2].date().isoformat(),
        "known_index": 2,
        "anchors": [
            {"index": 0, "price": 8.0, "kind": "low"},
            {"index": 2, "price": 8.1, "kind": "low"},
        ],
        "low_position_pass": True,
    }
    revisions = [
        {"boundary_version_id": "line-low", "known_index": 2,
         "known_at": bars.date.iloc[2].date().isoformat(),
         "line": {"slope": 0.0, "intercept": 10.0}},
        {"boundary_version_id": "line-revised", "known_index": 6,
         "known_at": bars.date.iloc[6].date().isoformat(),
         "line": {"slope": 0.0, "intercept": 12.0}},
    ]

    signals = build_breakout_signals(bars, structure, revisions)

    assert [signal["signal_date"] for signal in signals] == [
        bars.date.iloc[3].date().isoformat(), bars.date.iloc[7].date().isoformat()]
    assert [signal["boundary_version_id"] for signal in signals] == [
        "line-low", "line-revised"]
    assert all(signal["pattern_ids"] == ["same-two-lows-old-line"] for signal in signals)


def test_breakout_waits_until_archived_structure_maturity():
    bars = _bars([9.0, 10.0, 9.0, 11.0, 11.0, 13.5, 11.0, 16.0])
    structure = {
        "structure_key": "s1",
        "instance_id": "p1",
        "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[3].date().isoformat(),
        "maturity_date": bars.date.iloc[6].date().isoformat(),
        "anchors": [{"index": 0, "price": 8.0, "kind": "low"}],
    }
    pressure = [
        {"index": 1, "price": 10.1, "kind": "high", "confirmed_at": 2},
        {"index": 3, "price": 11.1, "kind": "high", "confirmed_at": 4},
    ]
    revisions = build_boundary_revisions(bars, structure, pressure, tick_size=0.01)

    signals = build_breakout_signals(
        bars, structure, revisions, atr_multiple=0.0, minimum_buffer=0.01)

    assert signals[0]["signal_date"] >= structure["maturity_date"]


def test_weak_cross_can_confirm_on_a_later_bar_with_atr_buffer():
    bars = _bars([9.0, 10.0, 9.0, 11.0, 12.0, 13.5])
    structure = {
        "structure_key": "s1", "instance_id": "p1", "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[3].date().isoformat(),
        "anchors": [{"index": 0, "price": 8.0, "kind": "low"}],
    }
    pressure = [
        {"index": 1, "price": 10.1, "kind": "high", "confirmed_at": 2},
        {"index": 3, "price": 11.1, "kind": "high", "confirmed_at": 4},
    ]
    revisions = build_boundary_revisions(bars, structure, pressure, tick_size=0.01)

    signals = build_breakout_signals(bars, structure, revisions)

    assert signals[0]["signal_date"] == bars.date.iloc[5].date().isoformat()


def test_pressure_point_uses_pure_two_atr_reversal_not_five_percent_floor():
    bars = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="B"),
        "open": [10.0, 10.1, 10.8, 10.4, 10.2],
        "high": [10.05, 10.2, 11.0, 10.6, 10.3],
        "low": [9.95, 10.0, 10.6, 10.2, 10.0],
        "close": [10.0, 10.1, 10.8, 10.4, 10.2],
        "volume": [1000] * 5,
    })

    pivots = pressure_points_2atr(bars, tick_size=0.01)

    assert any(pivot["kind"] == "high" and pivot["index"] == 2
               and pivot["confirmed_at"] == 3 for pivot in pivots)


def _archived_event(event_date, anchors):
    return {
        "event_date": event_date, "instance_id": "p1", "structure_key": "s1",
        "paired_at": event_date,
        "frozen_geometry": {"pattern_type": "w_bottom", "anchors": anchors,
                            "support_boundary": {"support_lower": 7.0}},
        "pattern": {
            "pattern_type": "w_bottom", "formed_at": "2019-01-01",
            "evidence": [
                {"code": "main_structure", "value": {"frozen_at": "2020-01-03",
                 "support_lower": 7.0, "low_position": {"passed": True}}},
                {"code": "neckline_selection", "value": {"structure_known_at": "2020-01-03"}},
                {"code": "boundary_available_at", "value": {"date": "2020-01-05"}},
            ],
        },
    }


def _archive_bars():
    dates = pd.date_range("2019-01-01", periods=300, freq="B")
    return pd.DataFrame({"date": dates, "open": [9.0] * len(dates),
                         "high": [9.1] * len(dates), "low": [8.9] * len(dates),
                         "close": [9.0] * len(dates), "volume": [1000] * len(dates)})


def test_structure_cannot_start_before_first_archived_event_date():
    bars = _archive_bars()
    events = [_archived_event("2020-01-10", [{"date": "2019-01-01", "price": 8.0, "role": "low"}])]

    structures = structures_from_events(events, bars, "2024-01-31")

    assert structures[0]["known_at"] == "2020-01-10"


def test_future_archive_events_do_not_add_anchors_to_seed_structure():
    bars = _archive_bars()
    events = [
        _archived_event("2020-01-10", [{"date": "2019-01-01", "price": 8.0, "role": "low"}]),
        _archived_event("2020-01-20", [{"date": "2019-01-01", "price": 8.0, "role": "low"},
                                       {"date": "2020-01-15", "price": 8.2, "role": "low"}]),
    ]

    structures = structures_from_events(events, bars, "2024-01-31")

    assert len(structures[0]["anchors"]) == 1
