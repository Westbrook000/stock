import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_loss_five_cases_20261105.py"
spec = importlib.util.spec_from_file_location("loss_five_cases", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_mfe_excludes_entry_and_exit_sessions():
    bars = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
        "high": [130.0, 105.0, 112.0, 140.0],
    })

    result = report._strict_interior_mfe(bars, entry_index=1, exit_index=3, entry_price=100.0)

    assert result == {"mfe_pct": 0.12, "high": 112.0, "date": "2025-01-03"}


def test_stop_gap_requires_fill_at_open_below_8_percent_stop():
    trade = {"exit_reason": "stop_loss", "stop_price": 92.0, "exit_price": 86.0}

    result = report._stop_gap_evidence(trade, exit_open=86.0, tick_size=0.01)

    assert result["gap_through_stop"] is True
    assert result["gap_points"] == 6.0
    assert report._stop_gap_evidence(trade, exit_open=95.0, tick_size=0.01)["gap_through_stop"] is False


def test_false_break_requires_confirmed_close_then_post_entry_close_below_frozen_line():
    evidence = report._false_break_evidence(
        signal_close=101.0, required_confirmation_close=100.5,
        closes=[100.8, 99.8, 98.0], dates=["2025-02-03", "2025-02-04", "2025-02-05"],
        boundary_values=[100.0, 100.0, 100.0], entry_offset=0, exit_offset=2,
    )

    assert evidence == {"confirmation_passed": True, "first_close_below_line": "2025-02-04",
                        "close": 99.8, "frozen_line": 100.0,
                        "offset_from_entry_sessions": 1}
    assert report._false_break_evidence(100.4, 100.5, [99.0], ["2025-02-04"], [100.0], 0, 0) is None


def test_case_selector_returns_exact_five_distinct_stable_event_keys():
    rows = [
        {"kind": kind, "code": code, "signal_date": date, "archetypes": archetypes}
        for kind, code, date, archetypes in [
            ("etf", "159122", "2026-06-22", ["premature_recross"]),
            ("stock", "002117", "2019-04-30", ["rapid_false_break"]),
            ("stock", "002242", "2025-11-13", ["extended_entry"]),
            ("stock", "600281", "2017-02-15", ["mfe_then_stop"]),
            ("stock", "600359", "2022-04-12", ["gap_stop", "mfe_then_stop"]),
        ]
    ]

    selected = report._select_five_cases(rows)

    assert len(selected) == 5
    assert len({(row["kind"], row["code"], row["signal_date"]) for row in selected}) == 5
    assert [row["case_id"] for row in selected] == [
        "premature_recross", "rapid_false_break", "extended_entry", "mfe_then_stop", "gap_stop"]
