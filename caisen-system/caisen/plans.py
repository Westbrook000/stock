"""One fixed 8%/20% execution plan, with pattern evidence kept separately."""
import math

BUY_CANDIDATE_PATTERN_TYPES = frozenset({
    'w_bottom', 'inverse_head_shoulders', 'triangle_bottom', 'bull_flag',
})


def build_trade_plan(pattern, entry_price, cash=100000, lot_size=100, bars=None,
                     commission=0.0, min_commission=0.0, slippage=0.0,
                     adv_shares=None, tick_size=0.01):
    if any(float(value) != 0 for value in (commission, min_commission, slippage)):
        raise ValueError('the fixed strategy uses zero costs')
    p = pattern.to_dict() if hasattr(pattern, 'to_dict') else dict(pattern)
    evidence = {item['code']: item.get('value') for item in p.get('evidence', [])}
    main = evidence.get('main_structure') or {}
    reasons, warnings = [], []
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
        if (evidence.get('main_selection') or {}).get('selected') is not True:
            reasons.append('main_structure_not_selected_primary')
    if not p.get('volume_verified'):
        warnings.append('volume unit or corporate-action adjustment is unverified')
    if not p.get('execution_rules_verified'):
        warnings.append('instrument trading rules are unverified')
    shares = max(0, int(float(cash) // (entry * lot_size)) * lot_size) if valid_entry else 0
    if shares == 0:
        reasons.append('insufficient_cash_or_liquidity')
    return {
        'pattern_id': p.get('pattern_id'), 'decision': 'WAIT' if reasons else 'CONSIDER_BUY',
        'entry_price': entry if valid_entry else None,
        'stop': entry * .92 if valid_entry else None, 'targets': [entry * 1.2] if valid_entry else [],
        'structure_level': main.get('neckline_at_observed', p.get('invalidation')),
        'shape_targets': p.get('targets', []), 'stop_basis': 'actual_entry_fixed_fraction',
        'planned_risk_pct': .08, 'net_reward_risk': 2.5, 'reward_risk': 2.5,
        'highest_allowed_entry': None, 'shares': shares,
        'cost_model': {'commission': 0.0, 'minimum_commission': 0.0, 'slippage': 0.0},
        'reason_codes': reasons, 'data_warnings': warnings,
        'main_bottom_reference': {key: main.get(key) for key in
                                  ('support_lower','support_upper','first_trough_date','latest_anchor_date')} if main else None,
        'explanation': ';'.join(reasons) if reasons else '有效突破后下一交易日开盘介入；实际成交价止损8%、止盈20%。',
    }
