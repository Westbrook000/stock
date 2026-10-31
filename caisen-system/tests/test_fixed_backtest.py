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
    assert STRATEGY_VERSION == STRATEGY_VERSION_V2 == "pattern_fixed_8_20_main_parent_v2"
    assert STRATEGY_VERSION_V1 == "pattern_fixed_8_20_v1"


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
    dates = pd.date_range("2024-01-01", periods=36, freq="D")
    frame = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                          "low": 9.8, "close": 10.0, "volume": 1000})
    def legacy_detector(*args, **kwargs):
        raise AssertionError("v2 must not call the legacy detector")

    class MainBottoms:
        @staticmethod
        def detect_main_bottoms(*args, **kwargs):
            return []
    monkeypatch.setattr(fixed_backtest_module, "_main_bottom_detector", lambda: MainBottoms)
    monkeypatch.setattr(patterns_module, "detect_patterns", legacy_detector)
    result = fixed_backtest_module._prefix_signals(frame, "1d", "AAA", "stock", object())

    assert result == []


def test_v1_replays_frozen_detector_for_all_four_legacy_entry_shapes(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=37, freq="D")
    frame = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                          "low": 9.8, "close": 10.0, "volume": 1000})
    observed_date = str(dates[-2].date())

    def recognize(window, code, kind, timeframe, background_bars, structure_window):
        return [baseline_patterns_module.Pattern(
            pattern_id=f"v1-{ptype}", code=code, kind=kind, timeframe=timeframe,
            pattern_type=ptype, direction="bullish", state="confirmed", formed_at="2024-01-01",
            confirmed_at=len(window) - 1, confirmation_date=observed_date,
                evidence=[{"code": "pattern_quality_pass", "value": True}],
        ) for ptype in ("w_bottom", "bull_flag")]

    monkeypatch.setattr(baseline_patterns_module, "detect_patterns", recognize)
    monkeypatch.setattr(fixed_backtest_module, "_main_bottom_detector",
                        lambda: (_ for _ in ()).throw(AssertionError("v1 must not call v2 engine")))
    signals = fixed_backtest_module._prefix_signals(
        frame, "1d", "AAA", "stock", object(), strategy_version=STRATEGY_VERSION_V1)

    assert len(signals) == 1
    assert signals[0]["signal_date"] == observed_date
    assert signals[0]["entry_date"] == str(dates[-1].date())
    assert signals[0]["pattern_types"] == ["bull_flag", "w_bottom"]
    assert signals[0]["pattern_tags"] == [
        {"pattern_id": "v1-bull_flag", "pattern_type": "bull_flag", "label": "下飘旗形上涨中继"},
        {"pattern_id": "v1-w_bottom", "pattern_type": "w_bottom", "label": "W底"},
    ]


def test_v1_159136_replay_matches_archived_signals_orders_trades_and_equity():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    archive_paths = [
        repo / "state/fixed_backtest_samples/fixed-backtest-20261029T050519Z-a1f404/fixed-backtest-results.json",
        repo / "state/fixed_backtest_samples/fixed-backtest-20261029T054651Z-b46fad/fixed-backtest-results.json",
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

    source_path = repo.parent / "data/etfs/159136.csv"
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    source.columns = [str(column).lower() for column in source.columns]
    replay_bars = source.loc[
        source.date.between("2025-12-25", "2026-10-26"),
        ["date", "open", "high", "low", "close", "volume"],
    ].copy()
    replay = fixed_backtest_module.run_fixed_backtest(
        replay_bars, "159136", "etf", "1d", cash=100000,
        development_cutoff="2016-05-13", trade_start="2025-12-25",
        strategy_version=STRATEGY_VERSION_V1,
    )
    assert replay["strategy_version"] == STRATEGY_VERSION_V1
    for field in fields:
        source_key = "order_events" if field == "orders" else field
        if field == "orders":
            assert normalized_order_events(replay[source_key]) == normalized_order_events(archived[0][source_key])
        else:
            assert replay[source_key] == archived[0][source_key], field


def test_execution_suppression_blocks_nested_bull_flag():
    nested_flag = patterns_module.Pattern(
        pattern_id="flag-child", code="AAA", kind="stock", timeframe="1d",
        pattern_type="bull_flag", direction="bullish", state="confirmed", formed_at="2024-01-01",
        confirmed_at=10, confirmation_date="2024-02-01",
        evidence=[{"code": "execution_suppression", "value": {
            "suppressed": True, "reason": "nested_in_main_bottom", "parent_id": "main-1"}}],
    )

    assert fixed_backtest_module._execution_suppressed(nested_flag)


def test_new_main_parent_signal_reaches_backtrader_only_at_next_open(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=38, freq="D")
    frame = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                          "low": 9.8, "close": 10.0, "volume": 1000})
    cutoff = "2024-02-05"
    main = patterns_module.Pattern(
        pattern_id="main-parent", code="AAA", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="confirmed", formed_at="2023-07-01",
        confirmed_at=35, confirmation_date=cutoff,
        evidence=[
            {"code": "main_structure", "value": {"role": "main", "selected": True,
             "signal_eligible": True, "family": "w_bottom"}},
            {"code": "main_selection", "value": {"selected": True, "decided_at": cutoff,
             "primary_pattern_id": "main-parent"}},
            {"code": "neckline_breakout", "value": {"date": cutoff, "frozen_neckline": 10.0}},
        ],
    )

    class MainBottoms:
        @staticmethod
        def detect_main_bottoms(rows, **kwargs):
            return [main] if str(pd.Timestamp(rows.date.iloc[-1]).date()) == cutoff else []

        @staticmethod
        def link_main_bottom_children(main_events, legacy_patterns, **kwargs):
            return list(main_events) + list(legacy_patterns)

    monkeypatch.setattr(fixed_backtest_module, "_main_bottom_detector", lambda: MainBottoms)
    monkeypatch.setattr(patterns_module, "detect_patterns", lambda *args, **kwargs: [])
    signals = fixed_backtest_module._prefix_signals(frame, "1d", "AAA", "stock", object())

    assert len(signals) == 1
    assert signals[0]["signal_date"] == cutoff
    assert signals[0]["entry_date"] == "2024-02-06"
    simulated = simulate_fixed_trades(frame, signals)
    assert simulated["trades"][0]["signal_date"] == cutoff
    assert simulated["trades"][0]["entry_date"] == "2024-02-06"
    assert simulated["order_events"][0]["role"] == "entry"


def test_weekly_main_engine_receives_full_daily_background_without_legacy_recognizer(monkeypatch):
    frame = pd.DataFrame([(_date, 10, 10.2, 9.8, 10, 1000)
                          for _date in pd.date_range("2024-01-01", periods=40)],
                         columns=["date", "open", "high", "low", "close", "volume"])
    received = []
    main_received = []

    def weekly(rows, calendar=None):
        return rows.iloc[::5].copy().reset_index(drop=True)

    def recognize(window, code, kind, timeframe, background_bars, structure_window):
        received.append((timeframe, len(window), len(background_bars),
                         str(background_bars.date.iloc[-1].date())))
        return []

    class MainBottoms:
        @staticmethod
        def detect_main_bottoms(rows, **kwargs):
            main_received.append(len(rows))
            return []

        @staticmethod
        def link_main_bottom_children(main_events, existing_patterns, **kwargs):
            return list(main_events) + list(existing_patterns)

    monkeypatch.setattr(fixed_backtest_module, "aggregate_weekly", weekly)
    monkeypatch.setattr(fixed_backtest_module, "_main_bottom_detector", lambda: MainBottoms)
    monkeypatch.setattr(patterns_module, "detect_patterns", recognize)
    fixed_backtest_module._prefix_signals(frame, "1w", "AAA", "stock", object())
    assert received == []
    assert main_received[-1] == 8


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
