"""A failed boundary consumes its own phase, not the entire bottom geometry."""
import numpy as np
import pandas as pd

from caisen import main_bottoms as m
from caisen.schemas import Pivot


def fixture(allow_new_boundary=True, length=190):
    dates = pd.bdate_range('2024-01-01', periods=length)
    anchors = [(0, 10.8), (20, 8.1), (50, 10.4), (60, 9), (70, 10.4),
               (80, 8.3), (100, 10.8), (101, 10.3), (140, 10.3),
               (160, 10.7), (170, 11.4), (189, 11.5)]
    close = np.interp(np.arange(length), [x[0] for x in anchors],
                      [x[1] for x in anchors])
    bars = pd.DataFrame({'date': dates, 'open': close, 'close': close,
                         'high': close + .05, 'low': close - .05, 'volume': 1000})
    bars.loc[[0, 50, 70, 100], 'high'] = [11, 10.5, 10.5, 11]
    bars.loc[[20, 80], 'low'] = [8, 8.2]

    def pivot(i, price, kind, known):
        return Pivot(i, str(dates[i].date()), price, kind, known,
                     str(dates[known].date()),
                     m._neckline_candidate_scale() if kind == 'high' else '2ATR')

    pivots = [pivot(0, 11, 'high', 3), pivot(20, 8, 'low', 23),
              pivot(50, 10.5, 'high', 53), pivot(60, 9, 'low', 63),
              pivot(70, 10.5, 'high', 73), pivot(80, 8.2, 'low', 83)]
    if allow_new_boundary:
        pivots.append(pivot(100, 11, 'high', 104))
    phase = {'line': {'slope': 0., 'intercept': 10.5}, 'tests': [],
             'consumed_index': 100, 'reason': 'breakout_before_minimum_structure_duration'}
    return bars, pivots, phase


def recognize(bars, pivots, phase):
    result = m._make_main_pattern(
        bars, 'fixture', 'stock', '1d', 'w_bottom', [pivots[1], pivots[5]],
        pivots, np.full(len(bars), .2), .01, '2atr_5pct', '5y',
        background_bars=bars, consumed_boundary_phases=[phase])
    return [p for p in (result if isinstance(result, list) else [result]) if p]


def test_failed_old_line_does_not_block_new_independently_supported_line():
    bars, pivots, phase = fixture()
    patterns = recognize(bars, pivots, phase)
    confirmed = [p for p in patterns if p.confirmation_date]
    assert len(confirmed) == 1
    assert confirmed[0].neckline == {'slope': 0., 'intercept': 11.}
    assert confirmed[0].confirmation_date == '2024-08-22'


def test_consumed_old_line_cannot_rearm_when_no_new_boundary_exists():
    bars, pivots, phase = fixture(allow_new_boundary=False)
    assert not any(p.confirmation_date for p in recognize(bars, pivots, phase))


def test_new_line_confirmation_is_not_backdated_before_maturity():
    bars, pivots, phase = fixture()
    before = recognize(bars.iloc[:140].copy(), pivots, phase)
    assert not any(p.confirmation_date for p in before)
    after = recognize(bars, pivots, phase)
    assert [p.confirmation_date for p in after if p.confirmation_date] == ['2024-08-22']


def test_evidence_winner_is_selected_before_checking_which_line_breaks_today():
    bars, pivots, _ = fixture()
    # The upper 11.0 line has two tests covering 100 bars; the 10.5 line has
    # two covering 20. The weaker line crosses earlier after maturity.
    bars.loc[100, ['open', 'close', 'low']] = [10.3, 10.3, 10.25]
    result = m._make_main_pattern(
        bars, 'fixture', 'stock', '1d', 'w_bottom', [pivots[1], pivots[5]],
        pivots, np.full(len(bars), .2), .01, '2atr_5pct', '5y', background_bars=bars)
    patterns = result if isinstance(result, list) else [result]
    confirmed = [p for p in patterns if p and p.confirmation_date]
    assert confirmed
    assert confirmed[-1].neckline == {'slope': 0., 'intercept': 11.}
    assert confirmed[-1].confirmation_date == '2024-08-22'
