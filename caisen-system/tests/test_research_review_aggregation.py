from pathlib import Path
import sys
import math

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_research_review_20261105 as review


def test_period_aggregate_uses_daily_equity_levels_across_calendar_gaps():
    calendar = ["2025-01-01", "2025-01-02", "2025-01-03"]
    target = review._new_period(calendar)
    bars = pd.DataFrame({"date": pd.to_datetime(calendar)})
    review._record_period(target, bars, [], {
        "trades": [{"pnl": 10000, "exit_reason": "take_profit", "holding_sessions": 2,
                    "holding_days": 3}],
        "daily_equity": [{"date": calendar[0], "equity": 100000},
                         {"date": calendar[2], "equity": 110000}],
        "rejections": []})
    review._record_period(target, bars, [], {
        "trades": [{"pnl": -5000, "exit_reason": "stop_loss", "holding_sessions": 1,
                    "holding_days": 2}],
        "daily_equity": [{"date": calendar[0], "equity": 100000},
                         {"date": calendar[1], "equity": 95000},
                         {"date": calendar[2], "equity": 95000}],
        "rejections": []})

    result = review._summarize_period(target, {
        "name": "test", "start": calendar[0], "end": calendar[-1]})

    assert [item["equity"] for item in result["daily_equity"]] == [200000, 195000, 205000]
    assert result["ending_equity"] == 205000
    assert result["metrics"]["maximum_drawdown"] == 0.025


def test_stratification_uses_frozen_boundary_daily_close_and_atr_definitions():
    bars = pd.DataFrame([
        {"date": "2025-01-01", "open": 9.5, "high": 11, "low": 9, "close": 10},
        {"date": "2025-01-02", "open": 10.5, "high": 11, "low": 9, "close": 10.5},
        {"date": "2025-01-03", "open": 10, "high": 10.5, "low": 9, "close": 9.5},
        {"date": "2025-01-04", "open": 9.5, "high": 10, "low": 9, "close": 9.5},
    ])
    pattern = {"pattern_id": "p", "pattern_type": "w_bottom", "neckline": {"slope": 0.0},
               "evidence": [{"code": "main_structure", "value": {
                   "pivot_scale": "2atr_5pct", "neckline_independent_test_count": 2}},
                            {"code": "neckline_breakout", "value": {
                                "date": "2025-01-01", "close": 11,
                                "frozen_neckline": 10, "atr14_previous": 2}}]}
    signal = {"signal_date": "2025-01-01", "entry_date": "2025-01-02",
              "patterns": [pattern]}
    trade = {"pattern_ids": ["p"], "pattern_types": ["w_bottom"],
             "signal_date": "2025-01-01", "entry_date": "2025-01-02",
             "exit_date": "2025-01-04", "exit_reason": "stop_loss",
             "holding_sessions": 2, "pnl": -8000}

    rows = review._trade_stratification_rows(bars, [signal], [trade])

    assert rows[0]["pivot_scale"] == "2atr_5pct"
    assert rows[0]["independent_test_count"] == 2
    assert rows[0]["breakout_atr"] == 0.5
    assert rows[0]["opening_displacement_atr"] == 0.25
    assert rows[0]["signal_atr_price_ratio"] == 0.2
    assert rows[0]["early_stop"] is True
    assert rows[0]["neckline_close_loss"] is True
    summary = review._summarize_stratifications(rows)
    scale_group = summary["pivot_scale"]["2atr_5pct"]
    assert scale_group["early_stop_denominator"] == 1
    assert scale_group["early_stop_rate"] == 1.0
    assert scale_group["neckline_close_loss_denominator"] == 1
    assert scale_group["neckline_close_loss_rate"] == 1.0
    assert scale_group["drawdown_basis"].startswith("realized-only")


def test_mfe_excludes_entry_and_exit_sessions():
    bars = pd.DataFrame([
        {"date": "2025-01-01", "high": 10.0},
        {"date": "2025-01-02", "high": 20.0},  # entry session, excluded
        {"date": "2025-01-03", "high": 11.0},  # interior holding session
        {"date": "2025-01-04", "high": 30.0},  # exit session, excluded
    ])
    trade = {"entry_date": "2025-01-02", "exit_date": "2025-01-04", "entry_price": 10.0}

    assert math.isclose(review._trade_mfe(bars, trade), 0.1)
