"""Scan plans using the fixed backtest's frozen measured-target resolver."""
import math

import pandas as pd

from .fixed_backtest import (STOP_FRACTION, STRATEGY_VERSION, TARGET_METHOD,
                             _select_measured_target, _signal_global_index)

BUY_CANDIDATE_PATTERN_TYPES = frozenset({
    'w_bottom', 'inverse_head_shoulders', 'triangle_bottom', 'bull_flag',
})


def _latest_date(bars):
    if bars is None or getattr(bars, 'empty', True):
        return None
    try:
        return str(pd.Timestamp(bars.iloc[-1]['date']).date())
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None


def _structure_level(pattern, evidence, bars, date_to_index):
    breakout = evidence.get('neckline_breakout') or {}
    frozen = breakout.get('frozen_neckline')
    if frozen is not None:
        return float(frozen)
    line = pattern.get('neckline') or (evidence.get('main_structure') or {}).get('neckline')
    if not isinstance(line, dict) or bars is None:
        return None
    signal_index = _signal_global_index(
        pattern, str(pattern.get('confirmation_date') or '')[:10], bars, date_to_index)
    if signal_index is None:
        return None
    try:
        return float(line['slope']) * signal_index + float(line['intercept'])
    except (KeyError, TypeError, ValueError):
        return None
def build_trade_plan(pattern, entry_price, cash=100000, lot_size=100, bars=None,
                     commission=0.0, min_commission=0.0, slippage=0.0,
                     adv_shares=None, tick_size=0.01):
    if any(float(value) != 0 for value in (commission, min_commission, slippage)):
        raise ValueError('the fixed strategy uses zero costs')
    p = pattern.to_dict() if hasattr(pattern, 'to_dict') else dict(pattern)
    evidence = {item['code']: item.get('value') for item in p.get('evidence', [])}
    main = evidence.get('main_structure') or {}
    selection = evidence.get('main_selection') or {}
    reasons = []
    entry = float(entry_price)
    valid_entry = math.isfinite(entry) and entry > 0
    if not valid_entry:
        reasons.append('invalid_entry_price')
    if p.get('pattern_type') not in BUY_CANDIDATE_PATTERN_TYPES:
        reasons.append('pattern_not_buy_candidate')
    if p.get('direction') != 'bullish' or p.get('state') != 'confirmed':
        reasons.append('pattern_not_confirmed_bullish')
    if 'ambiguous_structure' in (p.get('flags') or []):
        reasons.append('ambiguous_structure')
    if p.get('parent_id') or (evidence.get('execution_suppression') or {}).get('suppressed'):
        reasons.append('nested_in_main_bottom')
    if main:
        if main.get('role') != 'main' or main.get('signal_eligible') is not True:
            reasons.append('main_structure_not_signal_eligible')
        if (selection.get('selected') is not True
                or selection.get('primary_pattern_id') != p.get('pattern_id')):
            reasons.append('main_structure_not_selected_primary')

    confirmation_date = p.get('confirmation_date')
    date_to_index = {}
    if bars is not None and not getattr(bars, 'empty', True):
        date_to_index = {str(pd.Timestamp(value).date()): index
                         for index, value in enumerate(bars['date'])}
    structure_level = _structure_level(p, evidence, bars, date_to_index)
    if not confirmation_date:
        measured = {'price': None, 'status': 'missing', 'source_pattern_id': None,
                    'method': TARGET_METHOD}
        reasons.append('missing_confirmation_date')
    elif bars is None or getattr(bars, 'empty', True):
        measured = {'price': None, 'status': 'missing', 'source_pattern_id': None,
                    'method': TARGET_METHOD}
        reasons.append('confirmation_bars_unavailable')
    else:
        signal = {'signal_date': str(confirmation_date)[:10], 'pattern_id': p.get('pattern_id'),
                  'primary_pattern_id': p.get('pattern_id'), 'pattern': p, 'patterns': [p]}
        measured = _select_measured_target(signal, entry, bars, date_to_index)
        if measured['status'] == 'missing':
            reasons.append('missing_measured_target')
        elif measured['status'] == 'invalid':
            reasons.append('measured_target_not_above_reference_entry')

    latest = _latest_date(bars)
    if latest is None:
        reasons.append('confirmation_context_unavailable')
    elif str(confirmation_date)[:10] != latest:
        reasons.append('historical_confirmation')

    shares = max(0, int(float(cash) // (entry * lot_size)) * lot_size) if valid_entry else 0
    if shares == 0:
        reasons.append('insufficient_cash_or_liquidity')
    active_target = measured.get('price') if measured.get('status') == 'active' else None
    risk_per_share = entry * STOP_FRACTION if valid_entry else None
    reward_risk = ((active_target - entry) / risk_per_share
                   if active_target is not None and risk_per_share else None)
    return {
        'pattern_id': p.get('pattern_id'), 'decision': 'WAIT' if reasons else 'CONSIDER_BUY',
        'entry_price': entry if valid_entry else None,
        'entry_price_basis': 'latest_close_reference',
        'stop': entry * (1 - STOP_FRACTION) if valid_entry else None,
        'targets': [active_target] if active_target is not None else [],
        'structure_level': structure_level,
        'structure_level_basis': 'confirmation_day_frozen_boundary',
        'shape_targets': p.get('targets', []), 'stop_basis': 'actual_entry_fixed_fraction',
        'planned_risk_pct': STOP_FRACTION, 'net_reward_risk': reward_risk,
        'reward_risk': reward_risk, 'reward_risk_basis': 'latest_close_reference_indicative',
        'highest_allowed_entry': None, 'shares': shares,
        'strategy_version': STRATEGY_VERSION, 'target_method': TARGET_METHOD,
        'measured_target_method': measured.get('method') or TARGET_METHOD,
        'measured_target_status': measured.get('status', 'missing'),
        'measured_target_price': active_target,
        'measured_target_source_pattern_id': measured.get('source_pattern_id'),
        'measured_target_unlocked_date': measured.get('unlocked_date'),
        'measured_target_first_wave_high': measured.get('first_wave_high'),
        'measured_target_pullback_low': measured.get('pullback_low'),
        'measured_target_impulse': measured.get('impulse'),
        'cost_model': {'commission': 0.0, 'minimum_commission': 0.0, 'slippage': 0.0},
        'reason_codes': reasons, 'data_warnings': [],
        'main_bottom_reference': {key: main.get(key) for key in
                                  ('support_lower','support_upper','first_trough_date','latest_anchor_date')} if main else None,
        'explanation': ';'.join(reasons) if reasons else '仅首次确认日生成候选；参考收盘价止损按实际成交价−8%，首个有效等幅目标按形态冻结。',
    }
