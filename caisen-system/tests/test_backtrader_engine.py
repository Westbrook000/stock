"""Backtrader ledger invariants for the single approved daily strategy."""
import pandas as pd
import pytest
from caisen.fixed_backtest import simulate_fixed_trades
from caisen.backtest import run_backtest


def bars(rows):
    return pd.DataFrame(rows,columns=['date','open','high','low','close','volume'])


def signal(date):
    return dict(signal_date='2026-01-05',entry_date=date,pattern_ids=['fixture-pattern'])


def test_backtrader_enters_next_open_exits_fixed_target_and_uses_zero_fees():
    frame=bars([('2026-01-05',10,10.2,9.8,10,10000),
                ('2026-01-06',10.1,10.4,10,10.2,10000),
                ('2026-01-07',10.3,12.5,10.2,12.2,10000)])
    result=simulate_fixed_trades(frame,[signal('2026-01-06')])
    assert result['engine']=='backtrader_broker'
    trade=result['trades'][0]
    assert trade['entry_date']=='2026-01-06'
    assert trade['entry_price']==10.1
    assert trade['exit_reason']=='take_profit'
    assert trade['exit_price']==pytest.approx(12.12)
    assert all(event.get('commission',0)==0 for event in result['order_events'])
    assert result['ending_equity']==pytest.approx(119998)


def test_gap_open_does_not_apply_retired_maximum_entry_cap():
    frame=bars([('2026-01-05',10,10,10,10,10000),
                ('2026-01-06',10.4,10.5,10.2,10.3,10000),
                ('2026-01-07',10.3,10.4,10.2,10.3,10000)])
    result=simulate_fixed_trades(frame,[signal('2026-01-06')])
    assert result['trades'][0]['entry_price']==10.4
    assert result['trades'][0]['stop_price']==pytest.approx(10.4*.92)
    assert result['rejections']==[]


def test_stop_precedes_target_when_both_touch_same_bar():
    frame=bars([('2026-01-05',10,10,10,10,10000),
                ('2026-01-06',10,10,10,10,10000),
                ('2026-01-07',10,12.2,9,10,10000)])
    trade=simulate_fixed_trades(frame,[signal('2026-01-06')])['trades'][0]
    assert trade['exit_reason']=='stop_loss'
    assert trade['exit_price']==pytest.approx(9.2)
    assert trade['same_bar_stop_target_touched']


def test_end_of_data_liquidation_reconciles_cash_and_equity():
    frame=bars([('2026-01-05',10,10,10,10,10000),
                ('2026-01-06',10,10.1,9.9,10,10000),
                ('2026-01-07',10,10.3,9.9,10.2,10000)])
    result=simulate_fixed_trades(frame,[signal('2026-01-06')])
    assert result['trades'][0]['exit_reason']=='end_of_period'
    assert result['daily_equity'][-1]['shares']==0
    assert result['ending_equity']==result['daily_equity'][-1]['cash']
    assert all(result['accounting_checks'].values())


def test_weekly_backtest_is_retired():
    with pytest.raises(ValueError,match='daily'):
        run_backtest(pd.DataFrame(),'000001','stock',timeframe='1w')
