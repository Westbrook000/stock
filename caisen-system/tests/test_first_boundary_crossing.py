"""First-crossing audits must survive late discovery and candidate reselection."""
import numpy as np
import pandas as pd

from caisen import main_bottoms as m
from caisen.daily_incremental import DailyRecognitionSession
from caisen.schemas import Pattern, Pivot


def test_first_crossing_is_not_replaced_by_a_later_recross():
    close = np.array([9., 9., 9., 9., 11., 11., 9., 9., 11., 11.])
    atr = np.full(10, .2)
    cache = {}
    line = {'slope': 0., 'intercept': 10.}
    first = m._first_frozen_boundary_crossing(close, atr, line, 2, 5, .01, cache)
    assert first['index'] == 4
    later = m._first_frozen_boundary_crossing(close, atr, line, 2, 9, .01, cache)
    assert later['index'] == 4


def test_prefix_audit_cannot_read_a_cached_future_crossing():
    close = np.array([9., 9., 9., 9., 11.])
    atr = np.full(5, .2)
    cache = {}
    line = {'slope': 0., 'intercept': 10.}
    assert m._first_frozen_boundary_crossing(close, atr, line, 2, 4, .01, cache)['index'] == 4
    assert m._first_frozen_boundary_crossing(close[:4], atr[:4], line, 2, 3, .01, cache) is None


def test_cache_without_a_crossing_only_scans_new_bars():
    close = np.array([9., 9., 9., 9., 9., 11.])
    atr = np.full(6, .2)
    cache = {}
    line = {'slope': 0., 'intercept': 10.}
    assert m._first_frozen_boundary_crossing(close[:4], atr[:4], line, 2, 3, .01, cache) is None
    assert m._first_frozen_boundary_crossing(close, atr, line, 2, 5, .01, cache)['index'] == 5
    assert len(cache) == 1


def test_economically_identical_line_with_shared_touches_remains_consumed():
    tests = [{'pivot_index': i, 'known_index': i + 1} for i in (10, 30)]
    old = {'line': {'slope': -.001, 'intercept': 10.2}, 'tests': tests,
           'consumed_index': 40}
    numerically_refitted = {'line': {'slope': -.001000001, 'intercept': 10.200001},
                           'tests': tests}
    assert m._same_neckline_boundary_phase(old, numerically_refitted, .01)
    different = {'line': {'slope': -.001, 'intercept': 10.22}, 'tests': tests}
    assert not m._same_neckline_boundary_phase(old, different, .01)


def test_boundary_identity_distinguishes_new_line_and_survives_window_offsets():
    old = {'slope': -.01, 'intercept': 20.}
    shifted = {'slope': -.01, 'intercept': 19.5}
    assert m._frozen_boundary_identity(old, 100) == m._frozen_boundary_identity(shifted, 50)
    different = {'slope': -.01, 'intercept': 20.02}
    assert m._frozen_boundary_identity(old, 100) != m._frozen_boundary_identity(different, 100)


def test_failed_boundary_does_not_reject_another_line_in_the_same_atr_zone():
    tests = [{'pivot_index': i, 'known_index': i + 1} for i in (10, 30)]
    failed = {'line': {'slope': 0., 'intercept': 10.}, 'tests': tests,
              'index': 40, 'consumed_index': 40, 'price': 10., 'tolerance': .5}
    other = {'line': {'slope': 0., 'intercept': 10.02}, 'tests': tests}
    assert not m._boundary_was_rejected(other, [failed], .01)
    same = {'line': dict(failed['line']), 'tests': tests}
    assert m._boundary_was_rejected(same, [failed], .01)


def test_price_only_rejection_has_no_proven_boundary_identity():
    other = {'line': {'slope': 0., 'intercept': 10.02}, 'tests': []}
    assert not m._boundary_was_rejected(other, [{'index': 40, 'price': 10., 'tolerance': .5}], .01)


def test_late_discovery_cannot_emit_a_000538_style_old_boundary_recross():
    dates = ['2020-02-28', '2020-03-17', '2020-04-30', '2020-07-03',
             '2020-08-28', '2021-08-18', '2021-08-19']
    close = [42.8, 39.14, 51., 59.01, 60., 51., 54.13]
    session = DailyRecognitionSession('000538', 'stock')
    session.bars = [dict(date=pd.Timestamp(d), open=p, high=p + .1,
                         low=p - .1, close=p, volume=1000) for d, p in zip(dates, close)]
    session.date_strings = dates
    session.atr_values = [1.69] * len(dates)
    pattern = Pattern(
        pattern_id='late-discovered-line', code='000538', kind='stock', timeframe='1d',
        pattern_type='w_bottom', direction='bullish', state='forming', formed_at=dates[0],
        confirmed_at=None, invalidation=35., neckline={'slope': 0., 'intercept': 52.35},
        key_points={'first_trough_date': dates[0], 'first_trough_index': 0},
        evidence=[{'code': 'main_structure', 'value': {'role': 'main', 'support_lower': 35.,
                   'eligibility_reason': 'awaiting_fresh_neckline_breakout'}},
                  {'code': 'neckline_selection', 'value': {'line_frozen_at': dates[2],
                   'structure_known_at': dates[2]}}])
    pivots = [Pivot(0, dates[0], 42.8, 'low', 1, dates[1], '2ATR'),
              Pivot(1, dates[1], 39.14, 'low', 2, dates[2], '2ATR')]
    session.active[('w_bottom', 'fixture')] = {
        'family': 'w_bottom', 'scale': '2atr_5pct', 'sequence': pivots,
        'structure_known_at': 2, 'pattern': pattern, 'needs_initial_eval': False}
    assert session._check_cached_breakout() == []
    assert pattern.state == 'invalidated'
    assert pattern.confirmation_date is None
    phases = list(session.consumed_neckline_phases.values())
    assert phases and phases[0][0]['consumed_index'] == 3
