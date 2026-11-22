"""Regression tests for the sole fixed execution policy (old 7%/2R policy retired)."""
import pytest
import pandas as pd
from caisen.plans import build_trade_plan
from caisen.schemas import Pattern


def pattern(**overrides):
    values=dict(pattern_id='000001:w',code='000001',kind='stock',timeframe='1d',
                pattern_type='w_bottom',direction='bullish',state='confirmed',
                formed_at='2025-01-02',confirmed_at=10,confirmation_date='2025-01-10',invalidation=9.5,targets=[12],
                volume_verified=True,execution_rules_verified=True,
                evidence=[dict(code='targets', value=dict(available=True, prices=[12.0]))])
    values.update(overrides)
    return Pattern(**values)


def test_fixed_plan_has_zero_costs_and_cash_bounded_lot_size():
    candidate = pattern(evidence=[dict(code='targets', value=dict(available=True, prices=[13.0]))],
                       confirmation_date='2025-01-10')
    bars = pd.DataFrame({'date':['2025-01-09','2025-01-10'],
                         'close':[9.8,10.0], 'high':[10.0,10.1], 'low':[9.7,9.9]})
    p=build_trade_plan(candidate,10.1,bars=bars)
    assert p['decision']=='CONSIDER_BUY'
    assert p['planned_risk_pct']==.08
    assert p['targets']==[13.0]
    assert p['strategy_version']=='daily_four_patterns_measured_target_v2'
    assert p['target_method']=='signal_pattern_first_measured_target'
    assert p['measured_target_method']=='stored_frozen_pattern_measured_target'
    assert p['shares']%100==0
    assert p['shares']*10.1<=100000
    assert p['cost_model']==dict(commission=0,minimum_commission=0,slippage=0)


def test_unverified_metadata_does_not_warn_or_block_trade_plan():
    bars = pd.DataFrame({'date':['2025-01-10'], 'close':[10.0], 'high':[10.1], 'low':[9.9]})
    p=build_trade_plan(pattern(volume_verified=False, execution_rules_verified=False),10.1,bars=bars)
    assert p['decision']=='CONSIDER_BUY'
    assert p['data_warnings'] == []


@pytest.mark.parametrize('old_stop', [10.2,9,1])
def test_old_structural_stop_cannot_change_fixed_policy(old_stop):
    bars = pd.DataFrame({'date':['2025-01-10'], 'close':[10.0], 'high':[10.1], 'low':[9.9]})
    p=build_trade_plan(pattern(invalidation=old_stop),10.1,bars=bars)
    assert p['decision']=='CONSIDER_BUY'
    assert p['stop']==pytest.approx(10.1*.92)


def test_ambiguous_structure_cannot_bypass_wait():
    p=build_trade_plan(pattern(flags=['ambiguous_structure']),10.1)
    assert p['decision']=='WAIT'
    assert 'ambiguous_structure' in p['reason_codes']


def test_direct_plan_cannot_buy_outside_four_shapes():
    p=build_trade_plan(pattern(pattern_type='spring'),10.1)
    assert p['decision']=='WAIT'
    assert 'pattern_not_buy_candidate' in p['reason_codes']


def test_locked_measured_flag_target_is_explanatory_only():
    p=build_trade_plan(pattern(pattern_type='bull_flag',evidence=[dict(code='measured_target_locked',value=True)]),10.1)
    assert p['decision']=='WAIT'
    assert p['targets']==[]
    assert p['measured_target_status']=='missing'
    assert 'confirmation_bars_unavailable' in p['reason_codes']


def test_pending_flag_target_is_exposed_without_inventing_a_price():
    candidate = pattern(pattern_type='bull_flag', confirmation_date='2025-01-10',
                        key_points={'pivot_1': 5.0, 'pivot_2': 12.0, 'pivot_5': 9.0},
                        evidence=[dict(code='flag_breakout', value={'prior_high': 12.0,
                                                                     'impulse_size': 7.0})])
    bars = pd.DataFrame({'date':['2025-01-09','2025-01-10'],
                         'close':[10.0,11.0], 'high':[11.0,11.5], 'low':[9.5,10.5]})
    p = build_trade_plan(candidate, 11.0, bars=bars)
    assert p['decision'] == 'CONSIDER_BUY'
    assert p['targets'] == []
    assert p['measured_target_status'] == 'pending_prior_high'
    assert p['measured_target_price'] is None


def test_historical_confirmation_can_show_target_but_cannot_be_a_buy_plan():
    candidate = pattern(confirmation_date='2025-01-09',
                        evidence=[dict(code='targets', value=dict(available=True, prices=[13.0]))])
    bars = pd.DataFrame({'date':['2025-01-09','2025-01-10'],
                         'close':[10.0,10.2], 'high':[10.1,10.3], 'low':[9.9,10.0]})
    p = build_trade_plan(candidate, 10.2, bars=bars)
    assert p['decision'] == 'WAIT'
    assert p['targets'] == [13.0]
    assert 'historical_confirmation' in p['reason_codes']


def test_missing_confirmation_date_cannot_be_inferred_from_last_bar():
    candidate = pattern(confirmation_date=None)
    bars = pd.DataFrame({'date':['2025-01-10'], 'close':[10.0], 'high':[10.1], 'low':[9.9]})
    p = build_trade_plan(candidate, 10.0, bars=bars)
    assert p['decision'] == 'WAIT'
    assert p['measured_target_status'] == 'missing'
    assert 'missing_confirmation_date' in p['reason_codes']


def test_execution_suppression_still_blocks_nested_child():
    p=build_trade_plan(pattern(pattern_type='bull_flag',evidence=[dict(code='execution_suppression',value=dict(suppressed=True))]),10.1)
    assert p['decision']=='WAIT'
    assert 'nested_in_main_bottom' in p['reason_codes']


def test_main_shape_reference_survives_but_exit_basis_is_actual_fill():
    main=pattern(evidence=[dict(code='main_structure',value=dict(role='main',signal_eligible=True,
                  neckline_at_observed=9.7,plan_stop_level=9.5,support_lower=8.8,support_upper=9,
                  first_trough_date='2024-01-01',latest_anchor_date='2024-09-01')),
                  dict(code='main_selection',value=dict(selected=True,primary_pattern_id='000001:w')),
                  dict(code='neckline_breakout',value=dict(frozen_neckline=9.7))])
    p=build_trade_plan(main,10.1,bars=pd.DataFrame({'date':['2025-01-10'],'close':[10.0],
                                                   'high':[10.1],'low':[9.9]}))
    assert p['stop']==pytest.approx(10.1*.92)
    assert p['structure_level']==9.7
    assert p['stop_basis']=='actual_entry_fixed_fraction'
    assert p['main_bottom_reference']['support_lower']==8.8


def test_invalid_reference_price_cannot_buy():
    assert build_trade_plan(pattern(),0)['decision']=='WAIT'
