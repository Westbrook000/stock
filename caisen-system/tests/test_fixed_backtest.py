import json

import pandas as pd
from types import SimpleNamespace
from caisen import patterns as patterns_module
from caisen import fixed_backtest as fixed_backtest_module
from caisen.v16_baseline import patterns as baseline_patterns_module

from caisen.fixed_backtest import (
    STRATEGY_VERSION_V1,
    STRATEGY_VERSION_V2,
    STRATEGY_VERSION,
    calculate_metrics,
    simulate_fixed_trades,
)


def bars(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def test_fixed_stop_precedes_same_day_target_and_gap_stop_uses_open():
    frame = bars([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 10.1, 9.9, 10, 1000),
        ("2024-01-03", 8.8, 12.2, 8.7, 11, 1000),
    ])
    result = simulate_fixed_trades(frame, [{
        "signal_date": "2024-01-01", "entry_date": "2024-01-02",
        "pattern_ids": ["p1"], "pattern_types": ["w_bottom"],
    }])
    trade = result["trades"][0]
    assert trade["entry_price"] == 10
    assert trade["exit_date"] == "2024-01-03"
    assert trade["exit_price"] == 8.8
    assert trade["exit_reason"] == "stop_loss"
    assert result["engine"] == "backtrader_broker"
    assert [event["role"] for event in result["order_events"][:2]] == ["entry", "stop_loss"]
    assert result["order_events"][2]["role"] == "take_profit"
    assert result["order_events"][2]["status"] == "canceled"
    assert all(event.get("commission", 0) == 0 for event in result["order_events"])
    metrics = calculate_metrics(result["trades"], 100000, 88000, result["daily_equity"])
    assert metrics["exit_reason_counts"] == {"stop_loss": 1}
    assert metrics["average_holding_sessions"] == 1


def test_last_position_is_closed_by_backtrader_at_final_close():
    frame = bars([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 10.1, 9.9, 10, 1000),
        ("2024-01-03", 11, 11.5, 10, 11.25, 1000),
    ])
    result = simulate_fixed_trades(frame, [{
        "signal_date": "2024-01-01", "entry_date": "2024-01-02",
        "pattern_ids": ["p1"], "pattern_types": ["w_bottom"],
    }])
    assert result["trades"][0]["exit_reason"] == "end_of_period"
    assert result["trades"][0]["exit_price"] == 11.25
    assert result["order_events"][-1]["role"] == "end_of_period"
    assert result["ending_equity"] == 112500


def test_same_day_stop_and_target_touch_fills_stop_first_at_stop_price():
    frame = bars([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 10.1, 9.9, 10, 1000),
        ("2024-01-03", 10, 12.2, 8.7, 11, 1000),
    ])
    result = simulate_fixed_trades(frame, [{
        "signal_date": "2024-01-01", "entry_date": "2024-01-02",
        "pattern_ids": ["p1"], "pattern_types": ["w_bottom"],
    }])
    trade = result["trades"][0]
    assert trade["exit_reason"] == "stop_loss"
    assert abs(trade["exit_price"] - 9.2) < 1e-9
    assert next(event for event in result["order_events"] if event["status"] == "completed" and
                event["role"] != "entry")["role"] == "stop_loss"


def test_exit_children_do_not_trigger_on_entry_day():
    frame = bars([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 12.2, 8.7, 10, 1000),
        ("2024-01-03", 10, 11, 9.5, 10.5, 1000),
        ("2024-01-04", 10.5, 10.7, 10.2, 10.5, 1000),
    ])
    result = simulate_fixed_trades(frame, [{
        "signal_date": "2024-01-01", "entry_date": "2024-01-02",
        "pattern_ids": ["p1"], "pattern_types": ["w_bottom"],
    }])
    assert result["trades"][0]["entry_date"] == "2024-01-02"
    assert result["trades"][0]["exit_date"] == "2024-01-04"
    assert result["trades"][0]["exit_reason"] == "end_of_period"
    assert [event["date"] for event in result["order_events"] if event["status"] == "completed"] == [
        "2024-01-02", "2024-01-04"]


def test_entry_on_last_bar_is_rejected_to_preserve_t_plus_one_exit_rule():
    frame = bars([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 10.1, 9.9, 10, 1000),
    ])
    result = simulate_fixed_trades(frame, [{
        "signal_date": "2024-01-01", "entry_date": "2024-01-02",
        "pattern_ids": ["p1"], "pattern_types": ["w_bottom"],
    }])
    assert result["trades"] == []
    assert result["rejections"][0]["reason"] == "no_next_t1_session"


def test_no_trade_and_no_loss_metrics_are_valid_nulls():
    empty = calculate_metrics([], 100000, 100000, [])
    assert empty["trade_count"] == 0
    assert empty["win_rate"] is None
    assert empty["profit_factor"] is None
    assert empty["average_pnl"] is None

    profit_only = calculate_metrics([{"pnl": 100, "holding_days": 2,
                                      "entry_date": "2024-01-01", "exit_date": "2024-01-03"}],
                                    100000, 100100, [])
    assert profit_only["profit_factor"] is None


def test_strategy_version_is_explicit():
    assert STRATEGY_VERSION == "daily_four_patterns_fixed_8_20_v1"
    assert fixed_backtest_module.SUPPORTED_STRATEGY_VERSIONS == {STRATEGY_VERSION}


def test_main_bottom_signals_require_selected_eligible_parent_and_same_day_breakout():
    def event(pattern_id, *, selected=True, eligible=True, parent_id=None,
              confirmation_date="2024-02-01", breakout_date="2024-02-01",
              decided_at="2024-02-01"):
        return SimpleNamespace(
            pattern_id=pattern_id,
            pattern_type="w_bottom",
            direction="bullish",
            state="confirmed",
            confirmation_date=confirmation_date,
            parent_id=parent_id,
            evidence=[
                {"code": "main_structure", "value": {
                    "role": "main",
                    "signal_eligible": eligible,
                    "family": "w_bottom",
                }},
                {"code": "main_selection", "value": {
                    "selected": selected,
                    "decided_at": decided_at,
                    "primary_pattern_id": pattern_id if selected else "main-1",
                }},
                {"code": "neckline_breakout", "value": {"date": breakout_date}},
            ],
        )

    selected_parent = event("main-1")
    suppressed_child = event("child-1", selected=False, eligible=False, parent_id="main-1")
    weak_crossing = event("weak-1", eligible=False, confirmation_date=None, breakout_date=None)
    stale_breakout = event("stale-1", breakout_date="2024-01-31")
    stale_selection = event("stale-selection", decided_at="2024-01-31")

    result = fixed_backtest_module._eligible_main_patterns(
        [selected_parent, suppressed_child, weak_crossing, stale_breakout, stale_selection], "2024-02-01")

    assert [pattern.pattern_id for pattern in result] == ["main-1"]


def test_new_strategy_never_calls_legacy_detector_for_bull_flag_fallback(monkeypatch):
    from caisen import daily_incremental
    frame = bars([("2024-01-01",10,10,10,10,1000)])
    monkeypatch.setattr(patterns_module,"detect_patterns",lambda *a,**k: (_ for _ in ()).throw(AssertionError("legacy")))
    monkeypatch.setattr(daily_incremental,"generate_daily_signals",lambda *a,**k: [])
    assert fixed_backtest_module._prefix_signals(frame,"1d","AAA","stock") == []


def test_retired_v1_cannot_be_replayed():
    import pytest
    with pytest.raises(ValueError,match="unsupported"):
        fixed_backtest_module._prefix_signals(pd.DataFrame(),"1d","AAA","stock",strategy_version=STRATEGY_VERSION_V1)


def test_v1_159136_archived_reports_remain_readable_after_retirement():
    from pathlib import Path

    fixtures = Path(__file__).resolve().parent / "fixtures" / "fixed_backtest_archives"
    archive_paths = [
        fixtures / "fixed-backtest-20261029T050519Z-a1f404.json",
        fixtures / "fixed-backtest-20261029T054651Z-b46fad.json",
    ]
    archives = [json.loads(path.read_text(encoding="utf-8")) for path in archive_paths]
    archived = [next(row for row in report["results"]
                     if row["code"] == "159136" and row["timeframe"] == "1d")
                for report in archives]
    fields = ("signals", "orders", "trades", "ending_equity")
    def normalized_order_events(events):
        # Backtrader assigns process-global refs; execution fields are stable,
        # but the numeric ref depends on what else ran in this Python process.
        return [{key: value for key, value in event.items() if key != "order_ref"}
                for event in events]

    for field in fields:
        source_key = "order_events" if field == "orders" else field
        if field == "orders":
            assert normalized_order_events(archived[0][source_key]) == normalized_order_events(archived[1][source_key])
        else:
            assert archived[0][source_key] == archived[1][source_key]



def test_execution_suppression_blocks_nested_bull_flag():
    nested_flag = patterns_module.Pattern(
        pattern_id="flag-child", code="AAA", kind="stock", timeframe="1d",
        pattern_type="bull_flag", direction="bullish", state="confirmed", formed_at="2024-01-01",
        confirmed_at=10, confirmation_date="2024-02-01",
        evidence=[{"code": "execution_suppression", "value": {
            "suppressed": True, "reason": "nested_in_main_bottom", "parent_id": "main-1"}}],
    )

    assert fixed_backtest_module._execution_suppressed(nested_flag)


def test_incremental_signal_reaches_backtrader_only_at_next_open(monkeypatch):
    from caisen import daily_incremental
    frame=bars([("2024-02-05",10,10,10,10,1000),
                ("2024-02-06",10.2,10.3,10.1,10.2,1000),
                ("2024-02-07",10.3,10.4,10.2,10.3,1000)])
    source=[dict(signal_date="2024-02-05",entry_date="2024-02-06",pattern_ids=["main"],
                 pattern_types=["w_bottom"],pattern_tags=[dict(pattern_id="main",neckline_selection=dict(selected_candidate_id="neck-1"))])]
    monkeypatch.setattr(daily_incremental,"generate_daily_signals",lambda *a,**k: source)
    signals=fixed_backtest_module._prefix_signals(frame,"1d","AAA","stock")
    result=simulate_fixed_trades(frame,signals)
    assert signals[0]["pattern_tags"][0]["neckline_selection"]["selected_candidate_id"]=="neck-1"
    assert result["trades"][0]["signal_date"]=="2024-02-05"
    assert result["trades"][0]["entry_date"]=="2024-02-06"
    assert result["trades"][0]["entry_price"]==10.2


def test_weekly_backtest_is_rejected_before_recognition():
    import pytest
    with pytest.raises(ValueError,match="daily"):
        fixed_backtest_module._prefix_signals(pd.DataFrame(),"1w","AAA","stock")


def test_trade_start_restricts_evaluation_curve_but_retains_full_data_start():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    frame = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                          "low": 9.8, "close": 10.0, "volume": 1000})
    start = dates[30]
    result = __import__("caisen.fixed_backtest", fromlist=["run_fixed_backtest"]).run_fixed_backtest(
        frame, "AAA", "stock", "1d", trade_start=start)
    assert result["data_start"] == "2024-01-01"
    assert result["trade_start"] == "2024-01-31"
    assert result["evaluation_start"] == "2024-01-31"
    assert result["daily_equity"][0]["date"] == "2024-01-31"
    assert result["metrics"]["annualization_period_years"] < 20 / 365.25
