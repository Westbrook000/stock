import unittest

import pandas as pd

from wyckoff.backtest.engine_bt import simulate_daily
from wyckoff.config import TradeRuleConfig


class DailyExecutionTests(unittest.TestCase):
    def run_case(self, exit_open, exit_low, exit_high, stop=95.0):
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3),
                              "open": [100.0, 100.0, exit_open],
                              "high": [101.0, 101.0, exit_high],
                              "low": [99.0, 99.0, exit_low],
                              "close": [100.0, 100.0, exit_open], "volume": [1000]*3})
        signals = pd.DataFrame({"action": ["BUY", "HOLD", "HOLD"],
                                "stop": [stop, float("nan"), float("nan")],
                                "entry_type": ["spring", "", ""]})
        cfg = TradeRuleConfig(take_profit=0.1, stop_loss=0.08, commission=0.001, slippage=0.0)
        return simulate_daily(frame, signals, cfg)

    def test_tighter_structural_stop_is_preserved(self):
        result = self.run_case(100.0, 94.0, 101.0)
        self.assertEqual(result["trade_list"][0]["exit_price"], 95.0)

    def test_stop_gap_fills_at_open_and_pnl_includes_fees(self):
        result = self.run_case(90.0, 89.0, 92.0)
        trade = result["trade_list"][0]
        self.assertEqual(trade["exit_price"], 90.0)
        self.assertLess(trade["ret"], -0.1)
        self.assertLess(result["max_dd"], -0.09)

    def test_when_stop_and_target_both_hit_stop_is_first(self):
        result = self.run_case(100.0, 94.0, 112.0)
        self.assertEqual(result["trade_list"][0]["exit_reason"], "stop")

    def test_take_profit_gap_fills_at_open(self):
        result = self.run_case(115.0, 114.0, 116.0)
        self.assertEqual(result["trade_list"][0]["exit_price"], 115.0)
