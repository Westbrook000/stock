from pathlib import Path

import pandas as pd
import pytest

from caisen.assessment import _main_boundary
from caisen.main_bottoms import detect_main_bottoms


def _evidence(pattern):
    return {item["code"]: item["value"] for item in pattern.evidence}


def test_000538_legacy_daily_w_drawing_is_not_a_shared_tradable_candidate():
    source = Path(__file__).parents[2] / "data" / "stocks" / "000538.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2024-12-09"].reset_index(drop=True)
    patterns = detect_main_bottoms(bars, code="000538", timeframe="1d",
                                   background_bars=bars)

    # The historical daily drawing is no longer selected under the current
    # W structure/boundary association rule. Its prior-break consumption and
    # no-reissue behavior is covered by the cached-session regression; the
    # user's 000538 example itself is verified on the weekly path.
    assert not any(item.pattern_type == "w_bottom"
                   and item.key_points.get("first_trough_date") == "2024-01-23"
                   for item in patterns)


def test_confirmed_sloped_neckline_uses_confirmation_level_for_later_support():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-09", "2024-12-10", "2025-01-13"]),
        "close": [52.0, 51.8, 50.0], "high": [52.2, 52.0, 50.2],
        "low": [51.8, 51.6, 49.8],
    })
    target = {
        "key_points": {"pivot_1_index": 0, "pivot_1_date": "2024-12-09"},
        "neckline": {"slope": 0.001, "intercept": 51.5},
        "evidence": [{"code": "main_structure", "value": {
            "post_breakout_support_level": 51.5221739130435,
            "post_breakout_support_date": "2024-12-09",
            "neckline_at_observed": 51.5221739130435,
        }}],
    }
    line = target["neckline"]

    assert _main_boundary(target, frame, len(frame) - 1, (line["slope"], line["intercept"])) == \
        pytest.approx(51.5221739130435)


def test_unselected_cross_without_primary_breakout_does_not_consume_the_side_line():
    source = Path(__file__).parents[2] / "data" / "etfs" / "159122.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    bars = bars[bars.date <= "2026-06-22"].reset_index(drop=True)
    patterns = detect_main_bottoms(bars, code="159122", kind="etf", timeframe="1d",
                                   background_bars=bars)
    target = next(
        pattern for pattern in patterns
        if pattern.pattern_type == "w_bottom"
        and pattern.key_points.get("first_trough_date") == "2025-12-16"
    )
    phases = _evidence(target)["neckline_selection"]["consumed_boundary_phases"]
    assert not any(
        phase["consumed_date"] == "2026-04-29"
        and phase["line"] == {"slope": 0.0, "intercept": 1.101}
        for phase in phases
    )
