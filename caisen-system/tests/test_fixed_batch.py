import json

import pandas as pd

from caisen.fixed_batch import run_fixed_batch, summarize_batch
from caisen.fixed_backtest import STRATEGY_VERSION_V1


def test_batch_runner_freezes_one_cutoff_and_writes_replayable_artifacts(tmp_path):
    bars = pd.DataFrame([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 10.1, 9.9, 10, 1000),
        ("2024-01-03", 10, 10.1, 9.9, 10, 1000),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    progress = []
    result = run_fixed_batch([
        {"bars": bars, "code": "AAA", "kind": "stock", "timeframe": "1d"},
        {"bars": bars, "code": "BBB", "kind": "etf", "timeframe": "1d"},
    ], tmp_path, progress_callback=progress.append)
    assert result["development_cutoff"] == "2024-01-03"
    assert result["batch_execution"]["worker_count"] == 1
    assert [item["completed"] for item in progress] == [1, 2]
    assert result["summary"]["security_timeframe_accounts"] == 2
    assert result["summary"]["note"].find("收益率不直接求和") >= 0
    assert all(path.exists() for path in map(__import__("pathlib").Path, result["output"].values()))
    saved = json.loads(__import__("pathlib").Path(result["output"]["json"]).read_text())
    assert saved["strategy_version"] == "pattern_fixed_8_20_main_parent_v2"
    report_html = __import__("pathlib").Path(result["output"]["html"]).read_text()
    assert ".account-table{min-width:1800px;white-space:nowrap}" in report_html
    assert "src=\"/static/fixed-backtest-report.js?v=20261029v3\"" in report_html
    previous_json = result["output"]["json"]
    second = run_fixed_batch([], tmp_path)
    assert second["output"]["json"] != previous_json
    assert __import__("pathlib").Path(previous_json).exists()


def test_batch_runner_preserves_selected_strategy_version_in_account_and_report(tmp_path):
    bars = pd.DataFrame([
        ("2024-01-01", 10, 10, 10, 10, 1000),
        ("2024-01-02", 10, 10.1, 9.9, 10, 1000),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    result = run_fixed_batch(
        [{"bars": bars, "code": "AAA", "kind": "stock", "timeframe": "1d"}],
        tmp_path, strategy_version=STRATEGY_VERSION_V1,
    )
    saved = json.loads(__import__("pathlib").Path(result["output"]["json"]).read_text())
    assert result["strategy_version"] == STRATEGY_VERSION_V1
    assert saved["results"][0]["strategy_version"] == STRATEGY_VERSION_V1
    assert saved["strategy_version"] == STRATEGY_VERSION_V1


def test_failed_account_preserves_completed_checkpoints(tmp_path):
    bars = pd.DataFrame([("2024-01-01", 10, 10, 10, 10, 1000)],
                        columns=["date", "open", "high", "low", "close", "volume"])
    result = run_fixed_batch([
        {"bars": bars, "code": "AAA", "kind": "stock"},
        {"bars": pd.DataFrame({"wrong": [1]}), "code": "BROKEN", "kind": "stock"},
    ], tmp_path)
    assert result["batch_execution"]["status"] == "partial_failed"
    assert result["batch_execution"]["completed_count"] == 1
    saved = json.loads(__import__("pathlib").Path(result["output"]["json"]).read_text())
    assert [row["code"] for row in saved["results"]] == ["AAA"]


def test_multilabel_trade_is_one_trade_with_deduped_shape_tags():
    result = summarize_batch([{
        "code": "AAA", "kind": "stock", "timeframe": "1d", "signal_count": 1,
        "signals": [{"pattern_ids": ["p1", "p2"]}],
        "trades": [{"entry_date": "2024-01-02", "exit_date": "2024-01-03", "pnl": 2,
                    "pattern_ids": ["p1", "p2"], "pattern_types": ["w_bottom", "bull_flag"],
                    "pattern_tags": [{"pattern_id": "p1", "pattern_type": "w_bottom"},
                                     {"pattern_id": "p2", "pattern_type": "bull_flag"}]}],
    }])
    assert result["summary"]["trade_count"] == 1
    assert result["summary"]["distinct_traded_patterns"] == 2
    assert result["summary"]["overlapping_multi_pattern_signals"] == 1
    assert {row["value"] for row in result["groups"] if row["dimension"] == "shape"} == {
        "w_bottom", "bull_flag", "multi_shape"}


def test_equal_initial_cash_portfolio_includes_zero_trade_accounts():
    accounts = [
        {"code": "AAA", "kind": "stock", "timeframe": "1d", "signal_count": 0,
         "trades": [{"entry_date": "2024-01-02", "exit_date": "2024-01-03", "pnl": 500,
                     "holding_days": 1, "holding_sessions": 1, "exit_reason": "take_profit"}],
         "signals": [], "starting_cash": 100000, "ending_equity": 100500,
         "daily_equity": [{"date": "2024-01-02", "equity": 100000, "invested_value": 0},
                          {"date": "2024-01-03", "equity": 100500, "invested_value": 0}]},
        {"code": "BBB", "kind": "etf", "timeframe": "1w", "signal_count": 0,
         "trades": [], "signals": [], "starting_cash": 100000, "ending_equity": 100000,
         "daily_equity": [{"date": "2024-01-01", "equity": 100000, "invested_value": 0},
                          {"date": "2024-01-03", "equity": 100000, "invested_value": 0}]},
    ]
    result = summarize_batch(accounts)
    portfolio = result["summary"]["equal_initial_cash_portfolio"]
    assert portfolio["account_count"] == 2
    assert portfolio["starting_capital"] == 200000
    assert portfolio["ending_equity"] == 200500
    assert abs(portfolio["metrics"]["cumulative_return"] - 0.0025) < 1e-9
    assert portfolio["metrics"]["win_rate"] == 1.0
    assert [point["date"] for point in portfolio["daily_equity"]] == [
        "2024-01-01", "2024-01-02", "2024-01-03"]
