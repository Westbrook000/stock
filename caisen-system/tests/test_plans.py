"""Regression tests for the sole fixed execution policy (old 7%/2R policy retired)."""
import pytest
from caisen.plans import build_trade_plan
from caisen.schemas import Pattern


def pattern(**overrides):
    values=dict(pattern_id='000001:w',code='000001',kind='stock',timeframe='1d',
                pattern_type='w_bottom',direction='bullish',state='confirmed',
                formed_at='2025-01-02',confirmed_at=10,invalidation=9.5,targets=[12],
                volume_verified=True,execution_rules_verified=True,evidence=[])
    values.update(overrides)
    return Pattern(**values)


def test_fixed_plan_has_zero_costs_and_cash_bounded_lot_size():
    p=build_trade_plan(pattern(),10.1)
    assert p['decision']=='CONSIDER_BUY'
    assert p['planned_risk_pct']==.08
    assert p['reward_risk']==2.5
    assert p['shares']%100==0
    assert p['shares']*10.1<=100000
    assert p['cost_model']==dict(commission=0,minimum_commission=0,slippage=0)


def test_volume_is_auxiliary_and_disclosed():
    p=build_trade_plan(pattern(volume_verified=False),10.1)
    assert p['decision']=='CONSIDER_BUY'
    assert 'volume unit or corporate-action adjustment is unverified' in p['data_warnings']


@pytest.mark.parametrize('old_stop', [10.2,9,1])
def test_old_structural_stop_cannot_change_fixed_policy(old_stop):
    p=build_trade_plan(pattern(invalidation=old_stop),10.1)
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
    assert p['decision']=='CONSIDER_BUY'
    assert p['targets']==pytest.approx([12.12])


def test_execution_suppression_still_blocks_nested_child():
    p=build_trade_plan(pattern(pattern_type='bull_flag',evidence=[dict(code='execution_suppression',value=dict(suppressed=True))]),10.1)
    assert p['decision']=='WAIT'
    assert 'nested_in_main_bottom' in p['reason_codes']


def test_main_shape_reference_survives_but_exit_basis_is_actual_fill():
    main=pattern(evidence=[dict(code='main_structure',value=dict(role='main',signal_eligible=True,
                  neckline_at_observed=9.7,plan_stop_level=9.5,support_lower=8.8,support_upper=9,
                  first_trough_date='2024-01-01',latest_anchor_date='2024-09-01')),
                  dict(code='main_selection',value=dict(selected=True))])
    p=build_trade_plan(main,10.1)
    assert p['stop']==pytest.approx(10.1*.92)
    assert p['structure_level']==9.7
    assert p['stop_basis']=='actual_entry_fixed_fraction'
    assert p['main_bottom_reference']['support_lower']==8.8


def test_invalid_reference_price_cannot_buy():
    assert build_trade_plan(pattern(),0)['decision']=='WAIT'
