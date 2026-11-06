from caisen.fixed_batch import summarize_batch


def test_legacy_serial_batch_executor_is_removed():
    from caisen import fixed_batch
    assert not hasattr(fixed_batch, 'run_fixed_batch')


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
