import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd

from wyckoff.config import StrategyConfig
from wyckoff.analysis.events import Event
from wyckoff.analysis.engine import _make_signals


class Day0RefactorTests(unittest.TestCase):
    def test_local_day0_ward_returns_cluster_median(self):
        from wyckoff.analysis.day0_backend import cluster_prices
        levels = cluster_prices(np.array([100.0, 100.2, 100.4, 110.0, 110.2]), 0.8)
        self.assertEqual(sorted(x['peak_count'] for x in levels), [2, 3])
        self.assertTrue(np.allclose(sorted(x['price'] for x in levels), [100.2, 110.1]))

    def test_close_zigzag_confirms_only_when_reversal_arrives(self):
        from wyckoff.analysis.zigzag import confirmed_close_pivots
        prices = np.array([100., 98., 95., 97., 100., 104., 102., 98.])
        pivots = confirmed_close_pivots(prices, .05)
        self.assertEqual([(p.price_idx, p.confirm_idx, p.kind) for p in pivots], [(0, 2, 'high'), (2, 4, 'low'), (5, 7, 'high')])
        self.assertEqual(len(confirmed_close_pivots(prices[:4], .05)), 1)

    def test_direct_eligible_event_ignores_quality_score(self):
        for etype, kind in [('SPRING', 'ideal'), ('SC', 'confirmed'), ('ST', 'sc_test'), ('TEST', 'spring_test'), ('JOC', 'breakout')]:
            decisions = []
            for quality in [.001, 1.0]:
                n = 3
                feats = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=n), 'close': [100.]*n, 'vol_ratio': [1.]*n, 'pos_in_60': [.5]*n})
                event = Event(etype, 1, 1, 99., quality=quality, kind=kind,
                              meta={'stop_price': 97., 'entry_allowed': True, 'test_low': 98., 'break_low': 98.})
                cfg = StrategyConfig()
                cfg.signal.entry_breakouts = True  # explicit aggressive mode, still independent of scores
                signals = _make_signals(feats, [event], np.zeros(n, dtype=int), np.zeros(n, dtype=int), {k: np.zeros(n) for k in ['demand_touch','demand_quality','lts_position','demand_bottom','demand_top','supply_touch','supply_quality']}, None, cfg)
                decisions.append(signals.action.iloc[1])
            self.assertEqual(decisions, ['BUY', 'BUY'], etype)

    def test_unconfirmed_sc_and_needs_test_spring_do_not_buy(self):
        n = 3
        feats = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=n), 'close': [100.]*n, 'vol_ratio': [1.]*n, 'pos_in_60': [.5]*n})
        for etype, kind in [('SC_CANDIDATE', 'panic'), ('SPRING', 'needs_test')]:
            event = Event(etype, 1, 1, 99., quality=1., kind=kind, meta={'entry_allowed': False})
            result = _make_signals(feats, [event], np.zeros(n, dtype=int), np.zeros(n, dtype=int), {k: np.zeros(n) for k in ['demand_touch','demand_quality','lts_position','demand_bottom','demand_top','supply_touch','supply_quality']}, None, StrategyConfig())
            self.assertEqual(result.action.iloc[1], 'HOLD')

class EventPolicyRegressionTests(unittest.TestCase):
    def test_sc_requires_stopping_or_later_demand_confirmation(self):
        from test_causal_events import spring_scanner
        scanner=spring_scanner()
        scanner.trend[60]=-1
        scanner.o[61],scanner.c[61],scanner.h[61],scanner.l[61],scanner.v[61]=100.,99.,104.,98.,400.
        scanner.close_pos[61]=1/6
        first=scanner._sc_events_at(61)
        self.assertEqual([e.etype for e in first],['SC_CANDIDATE'])
        scanner.events.extend(first)
        scanner.o[62],scanner.c[62],scanner.h[62],scanner.l[62],scanner.v[62]=100.,102.,103.,99.,100.
        second=scanner._sc_events_at(62)
        confirmed=[e for e in second if e.etype=='SC']
        self.assertEqual(len(confirmed),1)
        self.assertEqual((confirmed[0].idx,confirmed[0].confirm_idx),(61,62))
        self.assertTrue(confirmed[0].meta['entry_allowed'])
        self.assertEqual(confirmed[0].meta['reference_price'],100.5)

    def test_sc_high_close_stopping_can_confirm_same_bar(self):
        from test_causal_events import spring_scanner
        scanner=spring_scanner(); scanner.trend[60]=-1
        scanner.o[61],scanner.c[61],scanner.h[61],scanner.l[61],scanner.v[61]=98.,101.,102.,97.,400.
        scanner.close_pos[61]=.8
        output=scanner._sc_events_at(61)
        self.assertEqual([e.etype for e in output],['SC_CANDIDATE','SC'])
        self.assertEqual(output[-1].meta['confirmation'],'stopping')

    def test_sc_without_day0_reference_is_not_an_entry(self):
        from test_causal_events import spring_scanner
        scanner=spring_scanner();scanner.trend[60]=-1;scanner.support[60]=np.nan
        scanner.o[61],scanner.c[61],scanner.h[61],scanner.l[61],scanner.v[61]=98.,101.,102.,97.,400.
        self.assertEqual(scanner._sc_events_at(61),[])

    def test_diagnostic_toggle_does_not_change_decisions_and_exit_has_priority(self):
        n=3
        feats=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=n),'close':[100.]*n})
        events=[Event('SC',1,1,98.,kind='confirmed',meta={'entry_allowed':True,'stop_price':97.}),
                Event('SOW',1,1,99.)]
        actions=[]
        for diagnostic in (True,False):
            cfg=StrategyConfig();cfg.signal.diagnostic_scores=diagnostic
            sig=_make_signals(feats,events,np.zeros(n,dtype=int),np.zeros(n,dtype=int),{},None,cfg)
            actions.append(sig.action.tolist())
            self.assertEqual(sig.action.iloc[1],'SELL')
        self.assertEqual(actions[0],actions[1])

    def test_real_stock_and_etf_prefix_results_are_causal(self):
        from wyckoff.data_store import load_frame
        from wyckoff.analysis.engine import run_analysis
        from pandas.testing import assert_frame_equal
        for code,kind in [('600846','stock'),('515450','etf')]:
            df=load_frame(code,kind)
            if df is None: self.skipTest('local market fixture absent')
            df=df.iloc[-850:].reset_index(drop=True)
            full=run_analysis(df,StrategyConfig())
            for cut in (300,600):
                prefix=run_analysis(df.iloc[:cut].copy(),StrategyConfig())
                assert_frame_equal(full.signals.iloc[:cut],prefix.signals)
                full_events=[e for e in full.events if e.confirm_idx<cut]
                self.assertEqual(full_events,prefix.events)
                known=[(l.line_id,l.price,l.peak_count,l.known_from,
                        min(l.valid_to if l.valid_to is not None else cut-1,cut-1))
                       for l in full.levels if l.known_from<cut]
                prior=[(l.line_id,l.price,l.peak_count,l.known_from,
                        l.valid_to if l.valid_to is not None else cut-1) for l in prefix.levels]
                self.assertEqual(known,prior)


class BreakoutEntryPolicyTests(unittest.TestCase):
    def test_book_default_waits_for_test_after_sos_or_joc(self):
        feats=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=3),'close':[100.]*3})
        for etype in ('SOS','JOC'):
            event=Event(etype,1,1,99.,kind='breakout',meta={'entry_allowed':True,'stop_price':97.})
            cfg=StrategyConfig()
            signals=_make_signals(feats,[event],np.zeros(3,dtype=int),np.zeros(3,dtype=int),{},None,cfg)
            self.assertEqual(signals.action.iloc[1],'HOLD')
            cfg.signal.entry_breakouts=True
            signals=_make_signals(feats,[event],np.zeros(3,dtype=int),np.zeros(3,dtype=int),{},None,cfg)
            self.assertEqual(signals.action.iloc[1],'BUY')
