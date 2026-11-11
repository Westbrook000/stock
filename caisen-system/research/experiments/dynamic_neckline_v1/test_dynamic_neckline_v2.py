import pandas as pd

from dynamic_neckline_v2 import build_boundary_revisions_v2
from dynamic_neckline_v3 import (boundary_matches_neckline_points,
                                 link_neckline_points)


def _bars(count=70):
    closes = [10.0] * count
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=count, freq="B"),
        "open": closes,
        "high": [10.3] * count,
        "low": [9.7] * count,
        "close": closes,
        "volume": [1000] * count,
    })


def _structure(bars, known_index=30):
    return {
        "structure_key": "structure-1",
        "instance_id": "pattern-1",
        "pattern_type": "w_bottom",
        "known_at": bars.date.iloc[known_index].date().isoformat(),
        "known_index": known_index,
        "anchors": [
            {"index": 20, "price": 8.0, "kind": "low"},
            {"index": 25, "price": 8.1, "kind": "low"},
        ],
        "support_lower": 7.5,
        "low_position_pass": True,
    }


def test_initial_boundary_can_use_pressure_points_before_structure_start():
    bars = _bars()
    structure = _structure(bars)
    pressure = [
        {"index": 2, "price": 12.0, "kind": "high", "confirmed_at": 4},
        {"index": 15, "price": 12.2, "kind": "high", "confirmed_at": 17},
    ]

    revisions = build_boundary_revisions_v2(bars, structure, pressure)

    assert len(revisions) == 1
    assert revisions[0]["known_index"] == 30
    assert revisions[0]["supporting_point_indices"] == [2, 15]
    assert max(revisions[0]["supporting_point_indices"]) < structure["anchors"][0]["index"]


def test_micro_line_drift_is_ignored_but_material_revision_is_accepted():
    bars = _bars()
    structure = _structure(bars)
    pressure = [
        {"index": 2, "price": 12.0, "kind": "high", "confirmed_at": 4},
        {"index": 15, "price": 12.2, "kind": "high", "confirmed_at": 17},
        {"index": 40, "price": 12.4, "kind": "high", "confirmed_at": 42},
        {"index": 55, "price": 14.0, "kind": "high", "confirmed_at": 57},
    ]

    revisions = build_boundary_revisions_v2(
        bars, structure, pressure, revision_shift_atr_multiple=1.0)

    assert [item["known_index"] for item in revisions] == [30, 57]
    assert revisions[0]["revision_reason"] == "initial_boundary_after_structure_known"
    assert revisions[1]["revision_reason"] == "material_pressure_boundary_shift"


def test_pressure_points_not_yet_confirmed_cannot_define_the_initial_boundary():
    bars = _bars()
    structure = _structure(bars, known_index=20)
    structure["anchors"] = [
        {"index": 10, "price": 8.0, "kind": "low"},
        {"index": 15, "price": 8.1, "kind": "low"},
    ]
    pressure = [
        {"index": 2, "price": 12.0, "kind": "high", "confirmed_at": 4},
        {"index": 12, "price": 12.2, "kind": "high", "confirmed_at": 22},
    ]

    revisions = build_boundary_revisions_v2(bars, structure, pressure)

    assert len(revisions) == 1
    assert revisions[0]["known_index"] == 22
    assert revisions[0]["known_at"] == bars.date.iloc[22].date().isoformat()


def test_boundary_must_cover_every_structural_anchor():
    bars = _bars()
    structure = _structure(bars)
    structure["anchors"][1]["price"] = 13.0
    pressure = [
        {"index": 2, "price": 12.0, "kind": "high", "confirmed_at": 4},
        {"index": 15, "price": 12.2, "kind": "high", "confirmed_at": 17},
    ]

    revisions = build_boundary_revisions_v2(bars, structure, pressure)

    assert revisions == []


def test_revision_is_not_activated_until_line_matches_linked_rebound():
    bars = _bars()
    structure = _structure(bars)
    structure["neckline_points"] = [
        {"index": 22, "price": 13.0, "confirmed_at": 25, "role": "neckline_rebound"}
    ]
    pressure = [
        {"index": 2, "price": 12.0, "kind": "high", "confirmed_at": 4},
        {"index": 15, "price": 12.2, "kind": "high", "confirmed_at": 17},
        {"index": 40, "price": 14.0, "kind": "high", "confirmed_at": 42},
    ]

    revisions = build_boundary_revisions_v2(
        bars, structure, pressure, atr_values=[0.2] * len(bars),
        association_points=structure["neckline_points"],
        maximum_association_gap_atr=2.0)

    assert revisions
    assert revisions[0]["known_index"] == 42
    assert revisions[0]["supporting_point_indices"] == [15, 40]
    assert revisions[0]["association_point_dates"] == [bars.date.iloc[22].date().isoformat()]


def test_w_bottom_links_its_intermediate_rebound_as_a_neckline_point():
    bars = _bars()
    structure = _structure(bars, known_index=30)
    pressure = [
        {"index": 15, "price": 12.0, "kind": "high", "confirmed_at": 17},
        {"index": 22, "price": 13.0, "kind": "high", "confirmed_at": 25},
        {"index": 35, "price": 15.0, "kind": "high", "confirmed_at": 37},
    ]

    linked = link_neckline_points(structure, pressure, bars, [0.2] * len(bars))

    assert [(point["index"], point["price"]) for point in linked["neckline_points"]] == [(22, 13.0)]
    assert linked["known_index"] == 30


def test_head_and_shoulders_links_both_rebounds_and_delays_pairing_until_known():
    bars = _bars()
    structure = _structure(bars, known_index=35)
    structure["pattern_type"] = "inverse_head_shoulders"
    structure["anchors"] = [
        {"index": 10, "price": 7.0, "kind": "low"},
        {"index": 20, "price": 5.0, "kind": "low"},
        {"index": 30, "price": 7.2, "kind": "low"},
    ]
    pressure = [
        {"index": 15, "price": 11.0, "kind": "high", "confirmed_at": 17},
        {"index": 25, "price": 12.0, "kind": "high", "confirmed_at": 38},
    ]

    linked = link_neckline_points(structure, pressure, bars, [0.2] * len(bars))

    assert [(point["index"], point["price"]) for point in linked["neckline_points"]] == [
        (15, 11.0), (25, 12.0)]
    assert linked["known_index"] == 38
    assert linked["known_at"] == bars.date.iloc[38].date().isoformat()


def test_boundary_must_match_each_confirmed_neckline_pivot_within_atr_tolerance():
    points = [
        {"index": 10, "price": 12.0, "confirmed_at": 20},
        {"index": 25, "price": 13.0, "confirmed_at": 30},
    ]
    atr_values = [0.25] * 50

    assert boundary_matches_neckline_points(
        {"slope": 1.0 / 15.0, "intercept": 12.0 - (1.0 / 15.0) * 10},
        points, atr_values, asof_index=35)
    assert not boundary_matches_neckline_points(
        {"slope": 0.0, "intercept": 15.0}, points, atr_values, asof_index=35)
    assert not boundary_matches_neckline_points(
        {"slope": 1.0 / 15.0, "intercept": 12.0 - (1.0 / 15.0) * 10},
        points, atr_values, asof_index=29)
