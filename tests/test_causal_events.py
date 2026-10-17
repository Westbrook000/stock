import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from wyckoff.analysis.events import Event, EventScanner
from wyckoff.analysis.engine import _make_signals
from wyckoff.config import StrategyConfig


def spring_scanner(low=99.8, volume=40.0):
    n = 70
    df = pd.DataFrame({"open": np.full(n, 102.0), "close": np.full(n, 102.0),
                       "high": np.full(n, 102.5), "low": np.full(n, 101.5),
                       "volume": np.full(n, 100.0)})
    df.loc[61, ["open", "close", "high", "low", "volume"]] = [100.4, 100.75, 100.9, low, volume]
    df["body"] = abs(df.close-df.open)
    df["range"] = df.high-df.low
    for key, value in {"vol_ratio": 1.0, "vol_max_30": 100.0, "body_avg_14": 0.5,
                       "upper_shadow": 0.1, "close_pos": 0.8, "pos_in_60": 0.5,
                       "chg_avg_abs_14": 0.01, "ema20": 101.0, "ema60": 100.0,
                       "atr": 1.0}.items():
        df[key] = value
    ss = SimpleNamespace(support=np.full(n, 100.5), resistance=np.full(n, np.nan),
                         support_id=np.ones(n, dtype=int), support_count=np.full(n, 2))
    return EventScanner(df, ss, StrategyConfig())


class CausalEventTests(unittest.TestCase):
    def test_same_bar_break_and_reclaim_is_ideal_spring(self):
        event = spring_scanner()._spring_at(61)
        self.assertIsNotNone(event)
        self.assertEqual((event.idx, event.confirm_idx, event.kind), (61, 61, "ideal"))

    def test_waist_touch_without_break_is_not_spring(self):
        self.assertIsNone(spring_scanner(low=100.48)._spring_at(61))

    def test_future_failure_does_not_hide_spring_asof(self):
        scanner = spring_scanner()
        event = Event("SPRING", 50, 50, 100.5, kind="ideal", failed_idx=65)
        scanner.events.append(event)
        self.assertIs(scanner._last("SPRING", 61, only_active=True), event)

    def test_needs_test_spring_cannot_buy_before_test(self):
        n = 3
        feats = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n),
                              "close": [100.0]*n, "vol_ratio": [1.0]*n,
                              "pos_in_60": [0.5]*n})
        zst = {key: np.zeros(n) for key in ["demand_touch", "demand_quality", "lts_position",
               "demand_bottom", "demand_top", "supply_touch", "supply_quality"]}
        ev = Event("SPRING", 1, 1, 100.0, quality=1.0, kind="needs_test", meta={"break_low": 99.0})
        result = _make_signals(feats, [ev], np.zeros(n), np.full(n, 2), zst, None, StrategyConfig())
        self.assertNotEqual(result.action.iloc[1], "BUY")

    def test_sc_test_requires_prior_automatic_rally(self):
        scanner = spring_scanner()
        sc = Event("SC", 50, 50, 100.0, meta={"vol": 1000.0, "range": 5.0})
        scanner.events.append(sc)
        self.assertIsNone(scanner._sc_test_at(61))
        scanner.events.append(Event("AR", 52, 52, 103.0, meta={"sc_idx": 50}))
        self.assertIsNotNone(scanner._sc_test_at(61))

    def test_spring_retest_uses_frozen_parent_line(self):
        scanner = spring_scanner()
        parent = scanner._spring_at(61)
        scanner.events.append(parent)
        scanner.support[65] = 500.0
        scanner.o[65], scanner.c[65], scanner.h[65], scanner.l[65], scanner.v[65] = 100.5, 100.8, 101.0, 100.3, 10
        event = scanner._spring_test_at(65)
        self.assertIsNotNone(event)
        self.assertEqual(event.price, 100.5)

    def test_failure_is_a_new_event_and_does_not_change_parent(self):
        scanner = spring_scanner()
        parent = scanner._spring_at(61)
        scanner.events.append(parent)
        scanner.c[62] = 99.0
        scanner._failures_at(62)
        self.assertIsNone(parent.failed_idx)
        self.assertEqual(parent.kind, "ideal")
        self.assertEqual(scanner.events[-1].etype, "SPRING_FAIL")

    def test_spring_test_rejects_intervening_loss_of_original_low(self):
        scanner=spring_scanner()
        parent=scanner._spring_at(61);scanner.events.append(parent)
        scanner.c[63]=98.0
        scanner.o[65],scanner.c[65],scanner.h[65],scanner.l[65],scanner.v[65]=100.5,100.8,101.,100.3,10.
        self.assertIsNone(scanner._spring_test_at(65))

    def test_sc_test_rejects_intervening_loss_of_climax_low(self):
        scanner=spring_scanner()
        scanner.events.append(Event('SC',50,50,100.,kind='confirmed',meta={'vol':1000.,'range':5.}))
        scanner.events.append(Event('AR',52,52,103.,meta={'sc_idx':50}))
        scanner.c[55]=98.
        self.assertIsNone(scanner._sc_test_at(61))
