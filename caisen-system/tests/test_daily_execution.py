import pandas as pd
from caisen.fixed_backtest import simulate_fixed_trades


def test_missing_next_session_is_rejected_and_not_parsed_as_nat():
    bars = pd.DataFrame([dict(date='2024-01-02', open=10, high=10, low=10, close=10, volume=1000)])
    result = simulate_fixed_trades(bars, [dict(signal_date='2024-01-02', entry_date=None, pattern_ids=['last'])])
    assert result['trades'] == []
    assert result['rejections'][0]['reason'] == 'no_next_t1_session'
    assert result['ending_equity'] == 100000


def test_dual_touch_is_disclosed_and_final_equity_reconciles():
    bars = pd.DataFrame([
        dict(date='2024-01-02',open=10,high=10,low=10,close=10,volume=1000),
        dict(date='2024-01-03',open=10,high=10,low=10,close=10,volume=1000),
        dict(date='2024-01-04',open=10,high=13,low=9,close=11,volume=1000),
    ])
    result=simulate_fixed_trades(bars,[dict(signal_date='2024-01-02',entry_date='2024-01-03',pattern_ids=['p'])])
    trade=result['trades'][0]
    assert trade['exit_reason']=='stop_loss'
    assert trade['same_bar_stop_target_touched'] is True
    assert abs(trade['exit_price']-9.2)<1e-9
    assert result['accounting_checks']['cash_plus_position_equals_equity'] is True
    assert result['accounting_checks']['realized_pnl_matches_ending_equity'] is True


def test_retired_strategies_and_weekly_backtests_cannot_execute():
    import pytest
    from caisen.fixed_backtest import run_fixed_backtest
    for old_version in ('pattern_fixed_8_20_v1','pattern_fixed_8_20_main_parent_v2'):
        with pytest.raises(ValueError,match='unsupported'):
            run_fixed_backtest(pd.DataFrame(), '000001','stock',strategy_version=old_version)
    with pytest.raises(ValueError,match='daily'):
        run_fixed_backtest(pd.DataFrame(),'000001','stock',timeframe='1w')


def test_only_fixed_execution_plan_uses_actual_entry_not_old_structure_or_target():
    from caisen.plans import build_trade_plan
    candidate=dict(pattern_id='p',pattern_type='w_bottom',direction='bullish',state='confirmed',
                   invalidation=2,targets=[10.01],volume_verified=False,evidence=[])
    plan=build_trade_plan(candidate,10)
    assert plan['decision']=='CONSIDER_BUY'
    assert abs(plan['stop']-9.2)<1e-9
    assert plan['targets']==[12]
    assert plan['planned_risk_pct']==.08
    assert plan['reward_risk']==2.5
    assert plan['highest_allowed_entry'] is None
    assert plan['stop_basis']=='actual_entry_fixed_fraction'
    assert plan['data_warnings']


def test_confirmed_flag_can_buy_before_old_measured_target_unlock():
    from caisen.plans import build_trade_plan
    candidate=dict(pattern_id='flag',pattern_type='bull_flag',direction='bullish',state='confirmed',
                   evidence=[dict(code='measured_target_locked',value=True)])
    assert build_trade_plan(candidate,10)['decision']=='CONSIDER_BUY'
