import pandas as pd

from caisen import api, daily_incremental, fixed_backtest
from caisen.assessment import assess_scan
from caisen.pattern_instances import PatternInstanceLedger


def test_daily_scan_assessment_and_backtrader_share_the_same_entry_event(monkeypatch):
    pattern = {
        "pattern_id": "000001:w:2026-01-01",
        "code": "000001",
        "kind": "stock",
        "timeframe": "1d",
        "pattern_type": "w_bottom",
        "direction": "bullish",
        "state": "confirmed",
        "formed_at": "2026-01-01",
        "confirmed_at": 1,
        "confirmation_date": "2026-01-02",
        "last_observed_date": "2026-01-02",
        "key_points": {"pivot_1_index": 0, "pivot_1_date": "2026-01-01"},
        "neckline": {"slope": 0.0, "intercept": 10.0},
        "invalidation": 9.0,
        "evidence": [{"code": "neckline_breakout", "value": {
            "date": "2026-01-02", "close": 10.5}}],
    }
    ledger = PatternInstanceLedger()
    ledger.append(pattern, "forming", "2026-01-01")
    entry_event = ledger.append(pattern, "confirmed", "2026-01-02", entry_eligible=True)
    signal = {"code": "000001", "kind": "stock", "signal_date": "2026-01-02",
              "entry_date": "2026-01-03", "pattern_id": pattern["pattern_id"],
              "pattern_ids": [pattern["pattern_id"]], "pattern_types": ["w_bottom"],
              "patterns": [pattern], "pattern": pattern}
    shared_result = {"instances": [pattern], "instance_events": [e.to_dict() for e in ledger.events],
                     "entry_events": [entry_event.to_dict()], "signals": [signal]}

    shared_owner = daily_incremental.recognize_daily_instances
    assert api.recognize_daily_instances is shared_owner
    recognizers = []
    def recognize(*args, **kwargs):
        recognizers.append(kwargs.get("recognizer"))
        return shared_result
    monkeypatch.setattr(daily_incremental, "recognize_daily_instances", recognize)
    backtrader_signals = fixed_backtest._prefix_signals(
        pd.DataFrame([{"date": "2026-01-02", "open": 10.5, "high": 10.6,
                       "low": 10.4, "close": 10.5, "volume": 1000}]),
        "1d", "000001", "stock")
    scan = assess_scan([pattern], {"1d": pd.DataFrame([
        {"date": "2026-01-01", "high": 10.2, "low": 9.8, "close": 10.0},
        {"date": "2026-01-02", "high": 10.6, "low": 10.4, "close": 10.5},
    ])}, "2026-01-02", instance_events_by_timeframe={
        "1d": shared_result["instance_events"]})

    assert backtrader_signals[0]["pattern_ids"] == [entry_event.instance_id]
    assert recognizers == ["boundary_first"]
    assert scan["status"] == "BUY"
    assert scan["timeframes"]["1d"]["event_id"] == entry_event.event_id
