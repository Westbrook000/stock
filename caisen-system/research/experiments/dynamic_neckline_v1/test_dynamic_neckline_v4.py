from dynamic_neckline_v4 import (
    filter_triangle_boundary_revisions,
    filter_triangle_breakout_signals,
    triangle_breakout_in_window,
)
from dynamic_neckline import structures_from_events
import pandas as pd


def test_triangle_revision_must_keep_its_apex_after_its_freeze_date():
    structure = {
        "pattern_type": "triangle_bottom",
        "first_trough_index": 10,
        "lower_boundary": {"slope": 0.02, "intercept": 5.0},
    }
    revisions = [
        {"known_index": 100, "boundary_version_id": "future-apex",
         "line": {"slope": -0.01, "intercept": 14.0}},
        {"known_index": 100, "boundary_version_id": "apex-now",
         "line": {"slope": -0.01, "intercept": 8.0}},
        {"known_index": 100, "boundary_version_id": "diverging",
         "line": {"slope": 0.03, "intercept": 14.0}},
    ]

    kept = filter_triangle_boundary_revisions(structure, revisions)

    assert [item["boundary_version_id"] for item in kept] == ["future-apex"]
    assert kept[0]["triangle_apex_index"] == 300


def test_triangle_breakout_window_includes_half_to_three_quarters_only():
    start, apex = 10, 110

    assert not triangle_breakout_in_window(59, start, apex)
    assert triangle_breakout_in_window(60, start, apex)
    assert triangle_breakout_in_window(85, start, apex)
    assert not triangle_breakout_in_window(86, start, apex)


def test_triangle_gate_does_not_change_other_pattern_revisions():
    revisions = [{"known_index": 100, "boundary_version_id": "w-line",
                  "line": {"slope": 0.03, "intercept": 14.0}}]

    assert filter_triangle_boundary_revisions(
        {"pattern_type": "w_bottom"}, revisions) == revisions


def test_v12_triangle_event_preserves_lower_rail_for_dynamic_revision_checks():
    bars = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="B")})
    geometry = {
        "pattern_type": "triangle_bottom",
        "anchors": [
            {"date": "2024-01-01", "price": 8.0, "role": "low"},
            {"date": "2024-01-02", "price": 9.0, "role": "high"},
            {"date": "2024-01-03", "price": 8.1, "role": "low"},
        ],
        "lower_boundary": {"slope": 0.05, "intercept": 7.95},
        "support_boundary": {"support_lower": 7.5},
    }
    event = {
        "event_date": "2024-01-03", "instance_id": "triangle-1",
        "structure_key": "structure-1", "frozen_geometry": geometry,
        "pattern": {"pattern_type": "triangle_bottom", "formed_at": "2024-01-01",
                    "evidence": [{"code": "main_structure", "value": {
                        "support_lower": 7.5, "low_position": {"passed": True}}}]},
    }

    [structure] = structures_from_events([event], bars)

    assert structure["first_trough_index"] == 0
    assert structure["lower_boundary"] == geometry["lower_boundary"]


def test_late_triangle_rebreakout_near_apex_is_rejected():
    # The 000008 V2 example had a dynamic upper rail intersecting the rising
    # support around bar 845 and a buy at bar 842 (about 99% progress).
    structure = {"pattern_type": "triangle_bottom", "first_trough_index": 550,
                 "lower_boundary": {"slope": 0.00036630036630036625,
                                    "intercept": -0.011465201465201441}}
    revision = {"known_index": 836, "boundary_version_id": "late-line",
                "line": {"slope": -0.00875, "intercept": 7.69125}}
    [qualified] = filter_triangle_boundary_revisions(structure, [revision])

    assert 842 > qualified["triangle_apex_index"] * 0.75 + 550 * 0.25
    assert not triangle_breakout_in_window(
        842, structure["first_trough_index"], qualified["triangle_apex_index"])


def test_signal_filter_uses_the_breakout_window_of_its_own_boundary_version():
    bars = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=120, freq="B")})
    structure = {"pattern_type": "triangle_bottom", "first_trough_index": 10}
    revisions = [{"boundary_version_id": "line-a", "triangle_start_index": 10,
                  "triangle_apex_index": 110}]
    signals = [{"boundary_version_id": "line-a", "signal_date": bars.date.iloc[60].date().isoformat()},
               {"boundary_version_id": "line-a", "signal_date": bars.date.iloc[90].date().isoformat()}]

    kept = filter_triangle_breakout_signals(structure, revisions, signals, bars)

    assert len(kept) == 1
    assert kept[0]["signal_date"] == signals[0]["signal_date"]
    assert kept[0]["triangle_progress"] == 0.5
