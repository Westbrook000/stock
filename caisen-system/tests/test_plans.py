import unittest

from caisen.plans import build_trade_plan
from caisen.backtest import valid_next_open
from caisen.schemas import Pattern


def pattern(**overrides):
    values = dict(
        pattern_id="000001:stock:1d:w_bottom:abc", code="000001", kind="stock",
        timeframe="1d", pattern_type="w_bottom", direction="bullish",
        state="confirmed", formed_at="2025-01-02", confirmed_at="2025-03-01",
        key_points={"first_low": 8.0, "second_low": 8.5}, neckline={"slope": 0.0, "intercept": 10.0},
        invalidation=9.5, targets=[12.0], evidence=[{"code": "neckline_break", "value": 10.1}],
        volume_ratio=1.8, volume_verified=True, execution_rules_verified=True, rule_version="caisen-v2", data_version="fixture",
        parent_id=None, flags=[],
    )
    values.update(overrides)
    return Pattern(**values)


class TradePlanTests(unittest.TestCase):
    def test_next_open_must_be_above_stop_below_target_and_under_cap(self):
        self.assertTrue(valid_next_open(10.1, 9.5, 12, 10.2))
        self.assertFalse(valid_next_open(9.5, 9.5, 12, 10.2))
        self.assertFalse(valid_next_open(12, 9.5, 12, 12.5))
        self.assertFalse(valid_next_open(10.3, 9.5, 12, 10.2))

    def test_plan_uses_zero_cost_baseline_when_risk_and_reward_meet_limits(self):
        result = build_trade_plan(pattern(), entry_price=10.1, cash=100_000, lot_size=100)
        self.assertEqual(result["decision"], "CONSIDER_BUY")
        self.assertLessEqual(result["planned_risk_pct"], 0.07)
        self.assertGreaterEqual(result["net_reward_risk"], 2.0)
        self.assertLessEqual(result["shares"] * result["entry_price"], 100_000)
        self.assertEqual(result["cost_model"], {"commission": 0.0, "minimum_commission": 0.0, "slippage": 0.0})

    def test_unverified_volume_remains_visible_but_trade_plan_waits(self):
        result = build_trade_plan(pattern(volume_verified=False), entry_price=10.1, cash=100_000)
        self.assertEqual(result["decision"], "WAIT")
        self.assertIn("volume_data_unverified", result["reason_codes"])
        self.assertIn("volume unit or corporate-action adjustment is unverified", result["data_warnings"])

    def test_stop_above_entry_never_produces_buy_recommendation(self):
        result = build_trade_plan(pattern(invalidation=10.2), entry_price=10.1, cash=100_000)
        self.assertNotEqual(result["decision"], "CONSIDER_BUY")

    def test_excessive_structural_risk_is_not_repaired_by_moving_stop(self):
        result = build_trade_plan(pattern(invalidation=9.0), entry_price=10.1, cash=100_000)
        self.assertEqual(result["decision"], "WAIT")
        self.assertEqual(result["stop"], 9.0)
        self.assertIn("structural_risk_over_7pct", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
