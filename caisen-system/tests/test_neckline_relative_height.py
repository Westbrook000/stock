from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caisen import main_bottoms as bottoms


@pytest.mark.parametrize("family", ["w_bottom", "inverse_head_shoulders", "triangle_bottom"])
def test_relative_height_filters_each_boundary_before_formation_checks(monkeypatch, family):
    dates = pd.bdate_range("2018-01-01", periods=240)
    frame = pd.DataFrame({"date": dates, "close": 3.5, "low": 3.0, "high": 4.0})
    frame.loc[0, ["low", "high"]] = [1.0, 40.0]
    frame.loc[100, "high"] = 30.0  # Historical highs do not affect this height gate.
    atr = np.full(len(frame), 2.0)

    def pivot(index, kind, price):
        return bottoms.Pivot(index, dates[index].strftime("%Y-%m-%d"), price, kind,
                             index+1, dates[index+1].strftime("%Y-%m-%d"), "2ATR")

    left, right = pivot(50, "low", 3.0), pivot(190, "low", 3.0)
    middle = pivot(110, "high", 5.8 if family == "w_bottom" else 4.2)
    sequence = [left, middle, right]
    if family == "inverse_head_shoulders":
        sequence = [left, middle, pivot(125, "low", 1.0), pivot(155, "high", 4.2), right]
    if family == "triangle_bottom":
        sequence = [left, middle, pivot(125, "low", 3.2), pivot(155, "high", 4.1), pivot(190, "low", 3.4)]

    candidates = [{"line":{"slope":0, "intercept":level}, "frozen_index":161,
                   "tests":[{"pair_id":str(level), "pivot_index":index, "level":level,
                             "known_index":index+1, "known_date":dates[index+1].strftime("%Y-%m-%d"),
                             "tolerance":1.0, "coverage_start_index":110} for index in (110, 155)]}
                  for level in ((6.1, 6.0) if family == "w_bottom" else (4.6, 4.5))]
    if family == "triangle_bottom":
        candidates = candidates[:1]  # Its native upper edge is 4.2 regardless of retest line.
    monkeypatch.setattr(bottoms, "_independent_neckline_tests", lambda *args, **kw: candidates)
    monkeypatch.setattr(bottoms, "_ihs_shoulders_have_depth", lambda *args, **kw: True)
    monkeypatch.setattr(bottoms, "_ihs_right_shoulder_intact", lambda *args, **kw: True)
    # Capture the boundaries passed through to the next stage of the real builder.
    seen = []
    def coverage(highs, atr, line, *args, **kwargs):
        seen.append(line["intercept"])
        return {"passed":False}
    monkeypatch.setattr(bottoms, "_neckline_formation_coverage", coverage)
    if family == "triangle_bottom":
        # Isolate this gate from upper-edge coverage and the 1 ATR support departure rule.
        frame.loc[110:156, "high"] = 4.0
        atr[126] = 0.5
    bottoms._make_main_pattern(frame, "test", "stock", "1d", family, sequence,
                              sequence, atr, .01, "2atr_5pct", "5y")
    if family == "w_bottom":
        assert seen == [6.0]  # exactly 100%; 6.1 is too high
    elif family == "inverse_head_shoulders":
        assert seen == []  # head is 1.00, not the 3.00 shoulders
    else:
        assert len(seen) == 1  # upper edge at the lowest point is within 100%


@pytest.mark.parametrize("code,folder,cutoff,left,right,expected_date", [
    ("600715", "stocks", "2019-02-13", "2014-06-09", "2019-01-31", None),
    ("600108", "stocks", "2024-10-08", "2024-02-06", "2024-09-18", "2024-10-08"),
])
def test_real_w_bottom_neckline_relative_height(code, folder, cutoff, left, right, expected_date):
    source = Path(__file__).parents[2] / "data" / folder / f"{code}.csv"
    bars = pd.read_csv(source, parse_dates=["date"])
    patterns = bottoms.detect_main_bottoms(bars[bars.date <= cutoff].reset_index(drop=True), code)
    matches = []
    for pattern in patterns:
        if pattern.pattern_type != "w_bottom" or pattern.formed_at != left:
            continue
        evidence = {item["code"]: item["value"] for item in pattern.evidence}
        if evidence["main_structure"]["latest_anchor_date"] == right:
            matches.append((pattern.confirmation_date, evidence["neckline_selection"]["neckline_relative_height"]))
    if expected_date is None:
        assert matches == []
    else:
        assert matches
        for date, check in matches:
            assert date == expected_date
            between = bars[(bars.date > left) & (bars.date < right)]
            assert check["reference_date"] == between.loc[between.high.idxmax(), "date"].date().isoformat()
            assert check["base_price"] == pytest.approx((2.22 + 2.25) / 2)
            assert check["relative_height"] == pytest.approx(
                check["neckline_price"] / check["base_price"] - 1)
            assert check["relative_height"] <= 1.0
