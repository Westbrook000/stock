from unittest.mock import patch

import pandas as pd
import pytest

from caisen.patterns import detect_patterns
from caisen.schemas import Pivot


def bottom_pattern_fixture(pattern_type, early_break=False):
    dates = pd.date_range("2025-01-01", periods=170, freq="D")
    close = [70.0] * len(dates)
    high = [95.0] * len(dates)
    low = [45.0] * len(dates)
    if pattern_type == "w_bottom":
        points = [(10, "low", 50.0), (40, "high", 90.0), (70, "low", 55.0),
                  (100, "high", 90.0), (120, "low", 52.0)]
        neckline = 90.0
        breakout = 92.0
    elif pattern_type == "inverse_head_shoulders":
        points = [(10, "low", 70.0), (40, "high", 80.0), (70, "low", 60.0),
                  (100, "high", 80.0), (120, "low", 70.0)]
        neckline = 80.0
        breakout = 82.0
    else:
        points = [(10, "low", 50.0), (40, "high", 90.0), (70, "low", 56.0),
                  (100, "high", 84.0), (120, "low", 62.0)]
        neckline = None
        breakout = 90.0
    pivots = []
    for index, kind, price in points:
        close[index] = price
        if kind == "low":
            low[index] = price
        else:
            high[index] = price
        pivots.append(Pivot(index, str(dates[index].date()), price, kind, index + 2,
                            str(dates[index + 2].date())))
    if pattern_type == "triangle_bottom":
        # With these frozen rails the upper boundary at index 150 is about 79.
        breakout = 82.0
    close[150:] = [breakout] * (len(close) - 150)
    high[150] = max(high[150], breakout)
    if early_break:
        close[80] = 105.0
        high[80] = max(high[80], close[80])
        # Keep confirmation absent at the mature signal checkpoint.
        close[150:] = [70.0] * (len(close) - 150)
    frame = pd.DataFrame({"date": dates, "open": close, "high": high,
                          "low": low, "close": close, "volume": [1000.0] * len(dates)})
    return frame, pivots


@pytest.mark.parametrize("pattern_type", ["w_bottom", "inverse_head_shoulders", "triangle_bottom"])
def test_detector_emits_each_aged_low_pattern_when_it_first_breaks_after_maturity(pattern_type):
    frame, pivots = bottom_pattern_fixture(pattern_type)

    with patch("caisen.patterns.detect_pivots", return_value=pivots):
        patterns = detect_patterns(frame, code="fixture", background_bars=frame, structure_window=500)

    confirmed = [pattern for pattern in patterns
                 if pattern.pattern_type == pattern_type and pattern.state == "confirmed"]
    assert confirmed
    evidence = {item["code"]: item["value"] for item in confirmed[0].evidence}
    assert evidence["structure_duration"]["eligible"] is True
    assert evidence["structure_duration"]["first_valid_break_date"] == str(frame.date.iloc[150].date())
    assert evidence["low_position"]["passed"] is True
    assert evidence["pattern_quality_pass"] is True


@pytest.mark.parametrize("pattern_type", ["w_bottom", "inverse_head_shoulders", "triangle_bottom"])
def test_detector_permanently_drops_each_low_pattern_after_an_early_break(pattern_type):
    frame, pivots = bottom_pattern_fixture(pattern_type, early_break=True)

    with patch("caisen.patterns.detect_pivots", return_value=pivots):
        patterns = detect_patterns(frame, code="fixture", background_bars=frame, structure_window=500)

    assert not [pattern for pattern in patterns if pattern.pattern_type == pattern_type]
