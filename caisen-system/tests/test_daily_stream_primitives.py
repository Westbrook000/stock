import numpy as np
import pandas as pd
import pytest
from caisen.daily_incremental import DailyRecognitionSession
from caisen.indicators import atr
from caisen.main_bottoms import _date_text
from caisen.schemas import Pattern, Pivot


def test_pressure_candidates_wait_for_two_atr_reversal_in_batch_and_stream():
    from caisen.main_bottoms import _detect_neckline_test_pivots

    bars = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=4),
        "open": [10.0] * 4,
        "high": [9.9, 10.0, 10.0, 10.0],
        "low": [9.8] * 4,
        "close": [9.9, 10.0, 8.5, 8.0],
        "volume": [1000] * 4,
    })
    atr_values = np.ones(4)

    batch = _detect_neckline_test_pivots(bars, atr_values, 0.01)

    session = DailyRecognitionSession("PRESSURE", "stock")
    session.bars = bars.to_dict("records")
    session.atr_values = atr_values.tolist()
    session.pressure["candidate"] = 1
    assert session._append_pressure_pivot(2) is None
    streamed = session._append_pressure_pivot(3)

    assert [(pivot.index, pivot.confirmed_at, pivot.scale) for pivot in batch] == [
        (1, 3, "neckline_test_2ATR")]
    assert (streamed.index, streamed.confirmed_at, streamed.scale) == (
        1, 3, "neckline_test_2ATR")


def test_continuous_atr_matches_wilder_reference_without_prefix_recalculation():
    dates=pd.bdate_range('2024-01-02',periods=180)
    close=np.linspace(10,20,len(dates))
    rows=pd.DataFrame(dict(date=dates,open=close,high=close+.1,low=close-.1,close=close,volume=1000))
    session=DailyRecognitionSession('ATR','stock')
    for row in rows.to_dict('records'):
        session.append_bar(row)
    np.testing.assert_allclose(session.atr_values,atr(rows).to_numpy(),rtol=1e-12,atol=1e-12)
    assert session.performance['atr_updates']==len(rows)
    assert session.performance['bars_appended']==len(rows)


def test_append_rejects_out_of_order_dates_before_changing_stream():
    session=DailyRecognitionSession('ORDER','stock')
    bar=dict(date='2024-01-02',open=10,high=10.2,low=9.9,close=10,volume=100)
    session.append_bar(bar)
    with pytest.raises(ValueError,match='increasing'):
        session.append_bar(bar)
    assert len(session.bars)==1
    assert len(session.atr_values)==1


def test_incremental_date_strings_stay_aligned_and_match_timestamp_rendering():
    session=DailyRecognitionSession('DATES','stock')
    for day in pd.bdate_range('2024-02-27',periods=4):
        value=10.
        session.append_bar(dict(date=day,open=value,high=value+.1,low=value-.1,
                                close=value,volume=100))
    expected=[str(pd.Timestamp(row['date']).date()) for row in session.bars]
    assert session.date_strings==expected
    assert [_date_text(pd.DatetimeIndex([row['date']]),0,[expected[index]])
            for index,row in enumerate(session.bars)]==expected


def test_pattern_evaluations_share_precomputed_neckline_context(monkeypatch):
    import caisen.daily_incremental as incremental
    from caisen.schemas import Pivot

    dates = pd.to_datetime(['2024-01-02', '2024-03-01', '2024-06-03', '2024-08-01'])
    session = DailyRecognitionSession('CONTEXT', 'stock')
    session.bars = [dict(date=day, open=10., high=10.2, low=9.8, close=10., volume=1000)
                    for day in dates]
    session.date_strings = [str(day.date()) for day in dates]
    session.atr_values = [.2] * len(dates)
    pressure = Pivot(2, session.date_strings[2], 10.2, 'high', 3,
                     session.date_strings[3], 'neckline_test_2ATR')
    session.pressure['pivots'] = [pressure]
    first = Pivot(0, session.date_strings[0], 9.0, 'low', 1,
                  session.date_strings[1], '2ATR|5%')
    candidates = {}
    for family in ('w_bottom', 'inverse_head_shoulders'):
        key = (family, '2atr_5pct', ((0, 'low'),))
        candidates[key] = {'family': family, 'scale': '2atr_5pct',
                           'sequence': [first], 'pattern': None,
                           'needs_initial_eval': False}
    session.active = candidates
    monkeypatch.setattr(session, '_has_pressure_test_pair', lambda _candidate: True)
    observed = []

    def capture(frame, *_args, **_kwargs):
        values = (frame.attrs.get('neckline_high_values'),
                  frame.attrs.get('neckline_atr_values'),
                  frame.attrs.get('neckline_date_index'),
                  frame.attrs.get('neckline_close_values'),
                  frame.attrs.get('neckline_low_values'))
        assert all(value is not None for value in values)
        observed.append(tuple(id(value) for value in values))
        assert _kwargs.get('collect_diagnostics') is False
        return None

    monkeypatch.setattr(incremental, '_make_main_pattern', capture)
    session._evaluate(pressure)

    assert len(observed) == 2
    assert len(set(observed)) == 1


def test_invalid_inverse_head_shoulders_is_rejected_before_neckline_search(monkeypatch):
    import caisen.main_bottoms as main_bottoms

    dates = pd.bdate_range("2024-01-02", periods=20)
    bars = pd.DataFrame({"date": dates, "open": 10.0, "high": 10.2,
                         "low": 9.8, "close": 10.0, "volume": 1000})
    points = [Pivot(index, str(dates[index].date()), price, kind, confirmed,
                    str(dates[confirmed].date()), "5ATR|12%")
              for index, price, kind, confirmed in (
                  (0, 10.0, "low", 1), (1, 11.0, "high", 2),
                  (2, 8.0, "low", 3), (3, 11.0, "high", 4),
                  (19, 10.0, "low", 19))]
    called = []

    def fail_if_called(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("invalid shoulder geometry should short-circuit first")

    monkeypatch.setattr(main_bottoms, "_independent_neckline_tests", fail_if_called)
    result = main_bottoms._make_main_pattern(
        bars, "GEOMETRY", "stock", "1d", "inverse_head_shoulders", points,
        points, np.full(len(bars), 0.5), 0.01, "5atr_12pct", "5y")

    assert result is None
    assert called == []


def test_identical_confirmed_structure_evaluates_only_at_coarsest_scale(monkeypatch):
    import caisen.daily_incremental as incremental
    from caisen.schemas import Pivot

    dates = pd.date_range('2024-01-02', '2024-08-01', periods=23)
    session = DailyRecognitionSession('SCALE-DEDUP', 'stock')
    session.bars = [dict(date=day, open=10., high=10.2, low=9.8, close=10., volume=1000)
                    for day in dates]
    session.date_strings = [str(day.date()) for day in dates]
    session.atr_values = [.2] * len(dates)
    sequence = [Pivot(index, session.date_strings[index], price, kind, index + 1,
                      session.date_strings[index + 1], 'same-confirmation')
                for index, price, kind in ((0, 8., 'low'), (10, 10., 'high'), (20, 8.1, 'low'))]
    pressure = Pivot(22, session.date_strings[22], 10.2, 'high', 22,
                     session.date_strings[22], 'neckline_test_2ATR')
    pressure_anchor = Pivot(12, session.date_strings[12], 10.2, 'high', 12,
                            session.date_strings[12], 'neckline_test_2ATR')
    session.pressure['pivots'] = [pressure_anchor, pressure]
    session.scale_pivots['2atr_5pct'] = [sequence[1]]
    session.active = {}
    for scale in ('2atr_5pct', '3atr_8pct', '5atr_12pct'):
        key = ('w_bottom', scale, tuple((p.index, p.kind) for p in sequence))
        session.active[key] = {'family': 'w_bottom', 'scale': scale,
                               'sequence': sequence, 'pattern': None,
                               'needs_initial_eval': False}
    monkeypatch.setattr(session, '_has_pressure_test_pair', lambda _candidate: True)
    evaluated_scales = []
    monkeypatch.setattr(incremental, '_make_main_pattern',
                        lambda _frame, _code, _kind, _tf, _family, _seq, _pivots, _atr, _tick,
                        scale, *_args, **_kwargs: evaluated_scales.append(scale))

    session._evaluate(pressure)

    assert evaluated_scales == ['5atr_12pct']


def test_main_bottom_candidate_waits_six_months_before_first_full_evaluation(monkeypatch):
    import caisen.daily_incremental as incremental

    session = DailyRecognitionSession('MATURITY', 'stock')
    sequence = [Pivot(index, day, price, kind, known, known_day, '3ATR|8%')
                for index, day, price, kind, known, known_day in (
                    (0, '2024-01-02', 8., 'low', 1, '2024-01-03'),
                    (1, '2024-02-02', 10., 'high', 2, '2024-02-05'),
                    (2, '2024-03-04', 8.1, 'low', 3, '2024-03-05'))]
    session.active[('w_bottom', '3atr_8pct', tuple((p.index, p.kind) for p in sequence))] = {
        'family': 'w_bottom', 'scale': '3atr_8pct', 'sequence': sequence,
        'structure_known_at': 3, 'pattern': None, 'needs_initial_eval': True,
    }
    monkeypatch.setattr(session, '_append_scale_pivots', lambda _index: [])
    monkeypatch.setattr(session, '_append_pressure_pivot', lambda _index: None)
    monkeypatch.setattr(session, '_append_local_pivot', lambda _index: False)
    monkeypatch.setattr(session, '_has_pressure_test_pair', lambda _candidate: True)
    calls = []
    monkeypatch.setattr(incremental, '_make_main_pattern', lambda *args, **kwargs: calls.append(1))
    session.append_bar(dict(date='2024-01-02', open=10., high=10.2, low=9.8, close=10., volume=1000))
    session.append_bar(dict(date='2024-03-05', open=10., high=10.2, low=9.8, close=10., volume=1000))
    session.append_bar(dict(date='2024-06-30', open=10., high=10.2, low=9.8, close=10., volume=1000))

    assert calls == []
    session.append_bar(dict(date='2024-07-02', open=10., high=10.2, low=9.8, close=10., volume=1000))

    assert calls == [1]


def test_neckline_low_range_index_matches_numpy_minimum_for_all_intervals():
    from caisen.main_bottoms import _build_range_minimum_index, _range_minimum

    values = np.random.default_rng(123).uniform(0.1, 10.0, size=137)
    index = _build_range_minimum_index(values)
    for start in range(len(values)):
        for end in range(start + 1, len(values) + 1):
            assert _range_minimum(index, start, end) == float(np.min(values[start:end]))


def test_touch_gap_minimum_cache_reuses_identical_intervals(monkeypatch):
    import caisen.main_bottoms as main_bottoms

    calls = []
    original = main_bottoms._touch_gap_minimum

    def counted(index, left, right):
        calls.append((left, right))
        return original(index, left, right)

    monkeypatch.setattr(main_bottoms, "_touch_gap_minimum", counted)
    index = main_bottoms._build_touch_gap_minimum_index(
        [2, 8, 14, 20], [1.5, 2.5, 3.5], {8: 1.0, 14: 1.2})
    cache = {}

    first = main_bottoms._cached_touch_gap_minimum(index, 0, 2, cache)
    second = main_bottoms._cached_touch_gap_minimum(index, 0, 2, cache)

    assert first == second == original(index, 0, 2)
    assert calls == [(0, 2)]


def test_boundary_pool_reuses_only_unchanged_confirmed_evidence(monkeypatch):
    from caisen import main_bottoms as engine
    from caisen.schemas import Pivot
    dates=pd.DatetimeIndex(pd.bdate_range('2023-01-02',periods=42))
    frame=pd.DataFrame(dict(date=dates,high=2.8,low=1.5,close=2.0))
    points=[Pivot(i,str(dates[i].date()),3.,'high',i+2,str(dates[i+2].date()),'neckline_test_2ATR') for i in (5,25)]
    points += [Pivot(15,str(dates[15].date()),1.5,'low',17,str(dates[17].date()),'2ATR|5%')]
    trough=Pivot(30,str(dates[30].date()),1.4,'low',32,str(dates[32].date()),'2ATR|5%')
    frame.loc[[5,25],'high']=3.
    frame.attrs['neckline_candidate_cache']={}
    calls=[]
    original=engine._qualify_neckline_slope
    def observed(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(engine,'_qualify_neckline_slope',observed)
    kwargs=dict(pivots=points,first_trough=trough,last_anchor=trough,atr_values=[.5]*len(dates),
                dates=dates,tick_size=.01,timeframe='1d',bars=frame,return_candidates=True,
                trusted_readonly_cache=True)
    before=engine._independent_neckline_tests(**kwargs)
    count=len(calls)
    assert before and count>0
    cached=engine._independent_neckline_tests(**kwargs)
    assert cached==before
    assert cached is before
    assert len(calls)==count
    # Newly confirmed evidence inside the covered interval invalidates the memo.
    kwargs['pivots']=points+[Pivot(18,str(dates[18].date()),1.3,'low',35,str(dates[35].date()),'2ATR|5%')]
    engine._independent_neckline_tests(**kwargs)
    assert len(calls)>count


def test_incremental_boundary_pairs_match_uncached_prefix_search():
    from caisen import main_bottoms as engine
    from caisen.schemas import Pivot
    dates=pd.DatetimeIndex(pd.bdate_range('2023-01-02',periods=78))
    source=pd.DataFrame(dict(date=dates,high=2.8,low=1.5,close=2.0))
    pressure=[Pivot(i,str(dates[i].date()),4-i*.01,'high',i+2,str(dates[i+2].date()),'neckline_test_2ATR') for i in (5,25,45,65)]
    lows=[Pivot(i,str(dates[i].date()),1.5,'low',i+2,str(dates[i+2].date()),'2ATR|5%') for i in (15,35,55)]
    # A late-confirmed lower trough changes departure evidence for old anchors.
    lows.append(Pivot(33,str(dates[33].date()),1.2,'low',72,str(dates[72].date()),'3ATR|8%'))
    for point in pressure:
        source.loc[point.index,'high']=point.price
    trough=Pivot(2,str(dates[2].date()),1.4,'low',4,str(dates[4].date()),'2ATR|5%')
    pool,formatted={},{}
    pair_sizes=[]
    original_combinations=engine.combinations
    original_qualify=engine._qualify_neckline_slope
    qualify_calls=[]
    def watched_combinations(values, count):
        values=tuple(values)
        pair_sizes.append((len(values),count))
        return original_combinations(values,count)
    def watched_qualify(*args, **kwargs):
        qualify_calls.append(1)
        return original_qualify(*args, **kwargs)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine,'combinations',watched_combinations)
    monkeypatch.setattr(engine,'_qualify_neckline_slope',watched_qualify)
    for end in (28,35,48,58,68,73,78):
        frame=source.iloc[:end].copy()
        points=[p for p in pressure+lows if p.confirmed_at<end]
        kwargs=dict(pivots=points,first_trough=trough,last_anchor=trough,atr_values=[.5]*end,
                    dates=dates[:end],tick_size=.01,timeframe='1d',return_candidates=True)
        baseline=engine._independent_neckline_tests(bars=frame,**kwargs)
        frame.attrs.update(neckline_candidate_cache=formatted,neckline_pair_cache=pool)
        pair_sizes.clear()
        if end == 73:
            qualify_calls.clear()
        result=engine._independent_neckline_tests(bars=frame,**kwargs)
        assert result==baseline, end
        if end == 48:
            current_touch_count=sum(point.confirmed_at < end for point in pressure)
            assert not any(size >= current_touch_count for size, count in pair_sizes)
        if end == 73:
            assert len(qualify_calls) <= 5
    monkeypatch.undo()


@pytest.mark.parametrize(
    "family,sequence_indices,irrelevant_index",
    [
        ("w_bottom", (1300, 1350, 1400), 10),
        ("inverse_head_shoulders", (1300, 1350, 1400), 10),
        ("triangle_bottom", (1300, 1325, 1350, 1375, 1400), 1500),
    ],
)
def test_pressure_touch_outside_pattern_search_domain_does_not_rebuild(
        family, sequence_indices, irrelevant_index, monkeypatch):
    import caisen.daily_incremental as incremental

    dates = pd.bdate_range("2017-01-02", periods=1600)
    session = DailyRecognitionSession("CACHE", "stock")
    session.bars = [dict(date=day, open=10., high=10.2, low=9.8, close=10., volume=1000)
                    for day in dates]
    session.atr_values = [.2] * len(dates)
    sequence = []
    kinds = ("low", "high", "low", "high", "low")
    for pos, index in enumerate(sequence_indices):
        kind = kinds[pos]
        price = 9. if kind == "low" else 11.
        sequence.append(Pivot(index, str(dates[index].date()), price, kind,
                              index + 2, str(dates[index + 2].date()), "2ATR|5%"))
    pressure = []
    for index in (sequence_indices[0] + 20, sequence_indices[0] + 40, irrelevant_index):
        pressure.append(Pivot(index, str(dates[index].date()), 11., "high",
                              index + 2, str(dates[index + 2].date()), "neckline_test_2ATR"))
    session.pressure["pivots"] = pressure
    key = (family, "2atr_5pct", tuple((item.index, item.kind) for item in sequence))
    session.active[key] = {"family": family, "scale": "2atr_5pct", "sequence": sequence,
                           "pattern": None, "needs_initial_eval": False}
    calls = []
    monkeypatch.setattr(incremental, "_make_main_pattern",
                        lambda *args, **kwargs: calls.append(1))

    session._evaluate(pressure[-1])

    assert calls == []


def test_pressure_touch_outside_w_bottom_neckline_level_band_does_not_rebuild():
    dates = pd.bdate_range('2020-01-02', periods=2200)
    session = DailyRecognitionSession('W-CAP', 'stock', tick_size=.01)
    session.bars = [dict(date=day, open=10., high=10.2, low=9.8, close=10., volume=1000)
                    for day in dates]
    session.atr_values = [.5] * len(dates)
    sequence = [Pivot(index, str(dates[index].date()), price, kind, index + 1,
                      str(dates[index + 1].date()), '3ATR|8%')
                for index, price, kind in ((1000, 8., 'low'), (1100, 10., 'high'), (1200, 8.1, 'low'))]
    session.scale_pivots['3atr_8pct'] = [sequence[1]]
    session.pressure['pivots'] = [Pivot(1300, str(dates[1300].date()), 10.1, 'high', 1302,
                                        str(dates[1302].date()), 'neckline_test_2ATR')]
    candidate = {'family': 'w_bottom', 'sequence': sequence}
    too_high = Pivot(1400, str(dates[1400].date()), 12.1, 'high', 1402,
                     str(dates[1402].date()), 'neckline_test_2ATR')

    too_low = Pivot(1400, str(dates[1400].date()), 9.0, 'high', 1402,
                    str(dates[1402].date()), 'neckline_test_2ATR')
    too_steep = Pivot(1400, str(dates[1400].date()), 10.9, 'high', 1402,
                      str(dates[1402].date()), 'neckline_test_2ATR')
    acceptable = Pivot(1400, str(dates[1400].date()), 10.4, 'high', 1402,
                       str(dates[1402].date()), 'neckline_test_2ATR')

    assert not session._pressure_pivot_can_change_candidate(candidate, too_high)
    assert not session._pressure_pivot_can_change_candidate(candidate, too_low)
    assert not session._pressure_pivot_can_change_candidate(candidate, too_steep)
    assert session._pressure_pivot_can_change_candidate(candidate, acceptable)


def test_interior_w_touch_below_rim_band_does_not_rebuild():
    dates = pd.bdate_range("2020-01-02", periods=1600)
    session = DailyRecognitionSession("W-INTERIOR-CAP", "stock", tick_size=0.01)
    session.bars = [dict(date=day, open=10.0, high=10.2, low=9.8, close=10.0, volume=1000)
                    for day in dates]
    session.atr_values = [0.5] * len(dates)
    sequence = [Pivot(index, str(dates[index].date()), price, kind, index + 1,
                      str(dates[index + 1].date()), "3ATR|8%")
                for index, price, kind in ((1000, 8.0, "low"), (1100, 10.0, "high"),
                                           (1200, 8.1, "low"))]
    interior = Pivot(1150, str(dates[1150].date()), 8.5, "high", 1152,
                     str(dates[1152].date()), "neckline_test_2ATR")
    session.scale_pivots["3atr_8pct"] = [sequence[1]]
    session.pressure["pivots"] = [interior]

    assert not session._pressure_pivot_can_change_candidate(
        {"family": "w_bottom", "sequence": sequence}, interior)


def test_ihs_pressure_touch_cannot_create_neckline_after_excessive_rise():
    dates = pd.bdate_range('2020-01-02', periods=1600)
    session = DailyRecognitionSession('IHS-SLOPE', 'stock')
    sequence = [Pivot(index, str(dates[index].date()), price, kind, index + 1,
                      str(dates[index + 1].date()), '3ATR|8%')
                for index, price, kind in ((1000, 8., 'low'), (1100, 6., 'low'), (1200, 8.1, 'low'))]
    session.pressure['pivots'] = [Pivot(1250, str(dates[1250].date()), 10., 'high', 1252,
                                        str(dates[1252].date()), 'neckline_test_2ATR')]
    candidate = {'family': 'inverse_head_shoulders', 'sequence': sequence}
    excessive = Pivot(1300, str(dates[1300].date()), 10.6, 'high', 1302,
                      str(dates[1302].date()), 'neckline_test_2ATR')
    acceptable = Pivot(1300, str(dates[1300].date()), 10.4, 'high', 1302,
                       str(dates[1302].date()), 'neckline_test_2ATR')

    assert not session._pressure_pivot_can_change_candidate(candidate, excessive)
    assert session._pressure_pivot_can_change_candidate(candidate, acceptable)
    too_close = Pivot(1300, str(dates[1300].date()), 10.01, 'high', 1302,
                      str(dates[1302].date()), 'neckline_test_2ATR')
    session.pressure['pivots'] = [Pivot(1295, str(dates[1295].date()), 10., 'high', 1297,
                                        str(dates[1297].date()), 'neckline_test_2ATR')]
    assert not session._pressure_pivot_can_change_candidate(candidate, too_close)


def test_triangle_rechecks_only_pressure_touches_near_its_fixed_upper_boundary():
    dates = pd.bdate_range('2020-01-02', periods=700)
    session = DailyRecognitionSession('TRIANGLE-TOUCH', 'stock', tick_size=.01)
    session.atr_values = [.5] * len(dates)
    sequence = [Pivot(index, str(dates[index].date()), price, kind, index + 1,
                      str(dates[index + 1].date()), '3ATR|8%')
                for index, price, kind in ((100, 4., 'low'), (200, 6., 'high'),
                                           (300, 4.3, 'low'), (400, 5.5, 'high'),
                                           (500, 4.6, 'low'))]
    candidate = {'family': 'triangle_bottom', 'sequence': sequence}
    off_boundary = Pivot(250, str(dates[250].date()), 5.0, 'high', 252,
                         str(dates[252].date()), 'neckline_test_2ATR')
    on_boundary = Pivot(250, str(dates[250].date()), 5.85, 'high', 252,
                        str(dates[252].date()), 'neckline_test_2ATR')

    assert not session._pressure_pivot_can_change_candidate(candidate, off_boundary)
    assert session._pressure_pivot_can_change_candidate(candidate, on_boundary)


def test_non_independent_pressure_touch_skips_cached_bottom_unless_it_crosses_line():
    from caisen.schemas import Pivot

    dates = pd.bdate_range('2020-01-02', periods=1100)
    session = DailyRecognitionSession('TOUCH-GROUP', 'stock')
    session.bars = [dict(date=day, open=10., high=10.2, low=9.8, close=10., volume=1000)
                    for day in dates]
    session.atr_values = [.2] * len(dates)
    sequence = [Pivot(index, str(dates[index].date()), price, 'low', index + 1,
                      str(dates[index + 1].date()), '3ATR|8%')
                for index, price in ((100, 8.), (200, 6.), (300, 8.1))]
    cached = Pattern(pattern_id='cached-touch', code='TOUCH-GROUP', kind='stock',
                     timeframe='1d', pattern_type='inverse_head_shoulders',
                     direction='bullish', state='forming', formed_at=str(dates[100].date()),
                     confirmed_at=None, key_points={}, neckline={'slope': 0., 'intercept': 10.},
                     invalidation=7., evidence=[], last_observed_date=str(dates[900].date()))
    candidate = {'family': 'inverse_head_shoulders', 'sequence': sequence, 'pattern': cached}
    close_touch = Pivot(1004, str(dates[1004].date()), 10.05, 'high', 1005,
                        str(dates[1005].date()), 'neckline_test_2ATR')
    crossing = Pivot(1004, str(dates[1004].date()), 10.3, 'high', 1005,
                     str(dates[1005].date()), 'neckline_test_2ATR')

    assert not session._pressure_pivot_can_change_candidate(candidate, close_touch)
    assert session._pressure_pivot_can_change_candidate(candidate, crossing)


@pytest.mark.parametrize(
    "family,sequence_points,test_points",
    [
        ("w_bottom", [(50, 4., "low"), (100, 6., "high"), (200, 4.1, "low")],
         [(80, 6.), (160, 6.)]),
        ("inverse_head_shoulders", [(20, 5., "low"), (80, 4., "low"), (140, 5.1, "low")],
         [(50, 7.), (110, 7.)]),
        ("triangle_bottom", [(50, 4., "low"), (90, 7., "high"), (130, 4.3, "low"),
                             (170, 6.8, "high"), (210, 4.6, "low")],
         [(110, 6.95), (150, 6.85)]),
    ],
)
def test_pressure_pair_gate_accepts_independent_tests_matching_pattern_geometry(
        family, sequence_points, test_points):
    dates = pd.bdate_range("2024-01-02", periods=240)
    session = DailyRecognitionSession("PAIR", "stock")
    session.bars = [dict(date=day, open=6., high=6.1, low=5.9, close=6., volume=1000)
                    for day in dates]
    session.atr_values = [.5] * len(dates)

    def pivot(index, price, kind, scale="3ATR|8%"):
        return Pivot(index, str(dates[index].date()), price, kind, index + 1,
                     str(dates[index + 1].date()), scale)

    sequence = [pivot(index, price, kind) for index, price, kind in sequence_points]
    session.scale_pivots["3atr_8pct"] = list(sequence)
    for item in sequence:
        session._index_low_pivot(item)
    if family == "w_bottom":
        inner_low = pivot(120, 4.5, "low", "2ATR|5%")
        session.scale_pivots["2atr_5pct"].append(inner_low)
        session._index_low_pivot(inner_low)
    session.pressure["pivots"] = [pivot(index, price, "high", "neckline_test_2ATR")
                                   for index, price in test_points]
    candidate = {"family": family, "sequence": sequence}

    assert session._has_pressure_test_pair(candidate)


def test_ihs_pressure_pair_gate_requires_tests_on_both_reaction_legs():
    dates = pd.bdate_range("2024-01-02", periods=240)
    session = DailyRecognitionSession("IHS-LEGS", "stock")
    session.atr_values = [.5] * len(dates)
    sequence = [Pivot(index, str(dates[index].date()), price, "low", index + 1,
                      str(dates[index + 1].date()), "3ATR|8%")
                for index, price in ((20, 5.), (80, 4.), (140, 5.1))]
    session.scale_pivots["3atr_8pct"] = list(sequence)
    for item in sequence:
        session._index_low_pivot(item)
    session.pressure["pivots"] = [Pivot(index, str(dates[index].date()), 7., "high", index + 1,
                                         str(dates[index + 1].date()), "neckline_test_2ATR")
                                   for index in (50, 65)]

    assert not session._has_pressure_test_pair(
        {"family": "inverse_head_shoulders", "sequence": sequence})


@pytest.mark.parametrize(
    "family,sequence_points,test_points",
    [
        ("w_bottom", [(50, 4.0, "low"), (100, 6.0, "high"), (200, 4.1, "low")],
         [(80, 6.0), (160, 6.7)]),
        ("inverse_head_shoulders", [(20, 5.0, "low"), (80, 4.0, "low"), (140, 5.1, "low")],
         [(50, 7.0), (110, 7.7)]),
    ],
)
def test_pressure_pair_gate_rejects_test_groups_without_allowed_neckline_slope(
        family, sequence_points, test_points):
    dates = pd.bdate_range("2024-01-02", periods=240)
    session = DailyRecognitionSession("PAIR-SLOPE", "stock")
    session.bars = [dict(date=day, open=6.0, high=6.1, low=5.9, close=6.0, volume=1000)
                    for day in dates]
    session.atr_values = [0.5] * len(dates)

    def pivot(index, price, kind, scale="3ATR|8%"):
        return Pivot(index, str(dates[index].date()), price, kind, index + 1,
                     str(dates[index + 1].date()), scale)

    sequence = [pivot(index, price, kind) for index, price, kind in sequence_points]
    session.scale_pivots["3atr_8pct"] = list(sequence)
    for item in sequence:
        session._index_low_pivot(item)
    session.pressure["pivots"] = [pivot(index, price, "high", "neckline_test_2ATR")
                                   for index, price in test_points]

    assert not session._has_pressure_test_pair({"family": family, "sequence": sequence})


def test_w_pressure_pair_gate_requires_an_atr_retreat_between_test_groups():
    dates = pd.bdate_range("2024-01-02", periods=240)
    session = DailyRecognitionSession("PAIR-RETREAT", "stock")
    session.bars = [dict(date=day, open=6.0, high=6.1, low=5.9, close=6.0, volume=1000)
                    for day in dates]
    session.atr_values = [0.5] * len(dates)
    sequence = [Pivot(index, str(dates[index].date()), price, kind, index + 1,
                      str(dates[index + 1].date()), "3ATR|8%")
                for index, price, kind in ((50, 4.0, "low"), (100, 6.0, "high"),
                                           (200, 4.1, "low"))]
    session.scale_pivots["3atr_8pct"] = list(sequence)
    for item in sequence:
        session._index_low_pivot(item)
    session.pressure["pivots"] = [
        Pivot(index, str(dates[index].date()), 6.0, "high", index + 1,
              str(dates[index + 1].date()), "neckline_test_2ATR")
        for index in (80, 160)]

    assert not session._has_pressure_test_pair({"family": "w_bottom", "sequence": sequence})


def test_confirmed_low_pivot_range_minimum_is_open_interval_and_updates_by_index():
    session = DailyRecognitionSession("LOW-RMQ", "stock")

    def pivot(index, price):
        return Pivot(index, "2024-01-02", price, "low", index + 1, "2024-01-03", "3ATR|8%")

    for index, price in ((10, 1.0), (15, 4.0), (20, 3.0), (30, 0.5), (4097, 2.0)):
        session._index_low_pivot(pivot(index, price))

    assert session._minimum_low_pivot(10, 30) == 3.0
    assert session._minimum_low_pivot(15, 20) is None
    assert session._minimum_low_pivot(4096, 4098) == 2.0
    assert session._minimum_low_pivot(10, 5000) == 0.5

    session._index_low_pivot(pivot(20, 2.5))
    assert session._minimum_low_pivot(10, 30) == 2.5
