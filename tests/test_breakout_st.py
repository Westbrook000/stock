import numpy as np
import pandas as pd
import unittest

from wyckoff.analysis.events import Event, EventScanner
from wyckoff.config import StrategyConfig
from types import SimpleNamespace


def scanner_with_retest(*, test_volume=40.0):
    n = 70
    close = np.full(n, 10.5)
    open_ = close.copy()
    high = close + 0.2
    low = close - 0.2
    volume = np.full(n, 50.0)
    # JOC breakout bar and its quiet return to the breakout level.
    open_[50], close[50], high[50], low[50], volume[50] = 10.0, 12.0, 12.1, 9.9, 100.0
    open_[55], close[55], high[55], low[55], volume[55] = 10.3, 10.0, 10.5, 9.95, test_volume
    open_[56], close[56], high[56], low[56], volume[56] = 10.0, 10.1, 10.2, 9.99, 35.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "vol_ratio": np.ones(n), "vol_max_30": np.full(n, 100.0), "body": np.abs(close-open_),
        "body_avg_14": np.full(n, 0.5), "upper_shadow": np.zeros(n),
        "close_pos": np.full(n, 0.5), "pos_in_60": np.full(n, 0.5),
        "chg_avg_abs_14": np.full(n, 0.01), "ema20": close, "ema60": close,
    })
    na = np.full(n, np.nan)
    df['atr'] = 1.0
    df['atr'] = 1.0
    ss = SimpleNamespace(support=na, resistance=na)
    scanner = EventScanner(df, ss, StrategyConfig())
    scanner.events.append(Event("JOC", 50, 50, 10.0, meta={
        "ref": 10.0, "volume": 100.0, "range": 2.2, "body": 2.0,
    }))
    return scanner


class BreakoutSTTests(unittest.TestCase):
    def test_quiet_retest_after_joc_confirms_st(self):
        scanner = scanner_with_retest()

        self.assertIsNone(scanner._detect_breakout_test(55, StrategyConfig().spring))
        event = scanner._detect_breakout_test(56, StrategyConfig().spring)

        self.assertIsNotNone(event)
        self.assertEqual(event.etype, "ST")
        self.assertEqual(event.kind, "breakout_test")
        self.assertEqual(event.confirm_idx, 56)
        self.assertEqual(event.meta["breakout_idx"], 50)

    def test_high_volume_retest_after_joc_does_not_confirm_st(self):
        scanner = scanner_with_retest(test_volume=90.0)

        event = scanner._detect_breakout_test(55, StrategyConfig().spring)

        self.assertIsNone(event)

    def test_already_failed_breakout_cannot_later_confirm_test(self):
        scanner = scanner_with_retest()
        scanner.c[52] = 9.0
        self.assertIsNone(scanner._detect_breakout_test(56, StrategyConfig().spring))
