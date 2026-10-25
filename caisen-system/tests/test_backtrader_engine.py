import pandas as pd
import pytest

from caisen.backtrader_engine import run_backtrader_simulation
from caisen.backtest import run_backtest


def bars(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def signal(date, stop=9.5, target=10.8, max_entry=10.2):
    return {"signal_date": "2026-01-05", "entry_date": date, "pattern_id": "fixture-pattern",
            "stop": stop, "target": target, "maximum_entry": max_entry, "lot_size": 100}


def test_backtrader_enters_next_open_exits_target_and_uses_zero_fees():
    frame = bars([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 10000),
        ("2026-01-06", 10.1, 10.4, 10.0, 10.2, 10000),
        ("2026-01-07", 10.3, 11.0, 10.2, 10.9, 10000),
    ])

    result = run_backtrader_simulation(frame, [signal("2026-01-06")], cash=100000)

    assert result["engine"] == "backtrader"
    assert result["commission"] == 0
    assert result["trades"][0]["entry_date"] == "2026-01-06"
    assert result["trades"][0]["entry_price"] == 10.1
    assert result["trades"][0]["exit_reason"] == "target"
    assert result["ending_equity"] == result["available_cash"] + result["marked_position_value"]
    assert result["ending_equity"] == pytest.approx(106930)


def test_open_above_maximum_is_rejected_once_without_position():
    frame = bars([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 10000),
        ("2026-01-06", 10.4, 10.5, 10.2, 10.3, 10000),
        ("2026-01-07", 10.0, 10.2, 9.8, 10.0, 10000),
    ])

    result = run_backtrader_simulation(frame, [signal("2026-01-06")], cash=100000)

    assert result["rejections"][0]["reason"] == "next_open_outside_plan"
    assert result["order_events"] == []
    assert result["ending_equity"] == 100000


def test_stop_precedes_target_when_both_touch_the_same_bar():
    frame = bars([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 10000),
        ("2026-01-06", 10.1, 10.4, 10.0, 10.2, 10000),
        ("2026-01-07", 10.0, 11.0, 9.0, 10.0, 10000),
    ])

    result = run_backtrader_simulation(frame, [signal("2026-01-06")], cash=100000)

    assert result["trades"][0]["exit_reason"] == "stop"
    assert result["trades"][0]["exit_price"] == 9.5


def test_end_of_data_position_is_marked_and_equity_reconciles():
    frame = bars([
        ("2026-01-05", 10.0, 10.2, 9.8, 10.0, 10000),
        ("2026-01-06", 10.1, 10.4, 10.0, 10.2, 10000),
    ])

    result = run_backtrader_simulation(frame, [signal("2026-01-06", target=12.0)], cash=100000)

    assert result["open_position"]["shares"] > 0
    assert result["marked_position_value"] == result["open_position"]["shares"] * 10.2
    assert result["ending_equity"] == result["available_cash"] + result["marked_position_value"]


def test_weekly_backtest_without_calendar_has_no_formal_signals():
    dates = pd.bdate_range("2026-01-05", periods=40)
    close = [10.0] * len(dates)
    frame = pd.DataFrame({"date": dates, "open": close, "high": [10.2] * len(dates),
                          "low": [9.8] * len(dates), "close": close, "volume": [10000] * len(dates)})

    result = run_backtest(frame, "000001", "stock", timeframe="1w")

    assert result["engine"] == "backtrader"
    assert result["signal_count"] == 0
    assert result["ending_equity"] == 100000
