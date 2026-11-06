"""Pressure must contain formation prices; only W/IHS have a decline cap."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from caisen.main_bottoms import _neckline_decline_check, _neckline_formation_coverage


@pytest.mark.parametrize('family', ['w_bottom', 'inverse_head_shoulders'])
def test_decline_is_measured_through_observation_and_accepts_exact_twenty_percent(family):
    line = {'slope': -.02, 'intercept': 10.0}
    assert _neckline_decline_check(family, line, 0, 100)['passed']
    assert not _neckline_decline_check(family, line, 0, 101)['passed']
    assert not _neckline_decline_check(family, line, 0, 150)['passed']


def test_triangle_and_flag_have_no_decline_cap():
    for family in ('triangle_bottom', 'bull_flag'):
        assert _neckline_decline_check(family, {'slope': -.04, 'intercept': 10}, 0, 100)['passed']


def test_formation_checks_all_highs_before_first_touch_and_after_last_touch():
    high = np.array([9.0, 10.8, 10.0, 9.0, 10.0, 10.8, 9.0, 11.2])
    atr = np.full(len(high), .2)
    result = _neckline_formation_coverage(high, atr, {'slope': 0., 'intercept': 10.}, 0, 6, .01)
    assert not result['passed']
    assert result['worst_index'] == 1
    assert result['end_index'] == 6


def test_final_weak_crossing_and_breakout_are_outside_formation_coverage():
    high = np.array([9.0, 10.08, 9.0, 10.15, 10.5])
    result = _neckline_formation_coverage(high, np.full(5, .2),
                                        {'slope': 0., 'intercept': 10.}, 0, 2, .01)
    assert result['passed']  # 0.08 wick is within 0.5ATR; later bars are breakout attempts.


@pytest.mark.parametrize('date,end', [('2014-08-04', 4012), ('2014-08-11', 4017)])
def test_reported_600108_w_neckline_is_too_steep(date, end):
    result = _neckline_decline_check('w_bottom',
        {'slope': -.020192307692307673, 'intercept': 87.15942307692299}, 3845, end)
    assert not result['passed']
    assert result['relative_decline'] > .35


def test_reported_600108_ihs_neckline_does_not_cover_the_formation():
    bars = pd.read_csv(Path(__file__).parents[2] / 'data/stocks/600108.csv')
    dates = pd.to_datetime(bars.date)
    end = int(np.flatnonzero(dates == pd.Timestamp('2021-07-26'))[0])
    h, l, c = (bars[k].to_numpy(float) for k in ('high', 'low', 'close'))
    atr = []
    for i in range(len(bars)):
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) if i else h[i]-l[i]
        atr.append(tr if not i else atr[-1]+(tr-atr[-1])/14)
    result = _neckline_formation_coverage(h, np.asarray(atr),
        {'slope': -.0015000000000000013, 'intercept': 12.021000000000008}, 5181, end, .01)
    assert not result['passed']
    assert result['maximum_excess'] > .4
