import pandas as pd

from caisen import fixed_backtest as fb
from caisen.research_config import current_recognition_experiment


def _bars():
    return pd.DataFrame([
        {"date": "2025-01-01", "open": 9.0, "high": 10.0, "low": 8.0, "close": 9.0, "volume": 1},
        {"date": "2025-01-02", "open": 9.0, "high": 10.0, "low": 8.0, "close": 9.0, "volume": 1},
        {"date": "2025-01-03", "open": 17.0, "high": 18.0, "low": 16.0, "close": 17.0, "volume": 1},
        {"date": "2025-01-04", "open": 17.0, "high": 18.0, "low": 16.0, "close": 17.0, "volume": 1},
    ])


def _signal(signal_date="2025-01-02", entry_date="2025-01-03", *, breakout=True):
    evidence = ([{"code": "neckline_breakout", "value": {
        "date": signal_date, "frozen_neckline": 10.0}}] if breakout else [])
    pattern = {"pattern_id": "p1", "pattern_type": "w_bottom", "direction": "bullish",
               "state": "confirmed", "confirmation_date": signal_date,
               "parent_id": None, "neckline": {"slope": 1.0, "intercept": 8.0},
               "evidence": evidence}
    return {"signal_date": signal_date, "entry_date": entry_date,
            "pattern_ids": ["p1"], "pattern_types": ["w_bottom"],
            "pattern_labels": ["W底"],
            "pattern_tags": [{"pattern_id": "p1", "pattern_type": "w_bottom"}],
            "patterns": [pattern], "pattern": pattern}


def test_open_admission_projects_frozen_line_and_uses_signal_close_atr():
    bars = _bars()
    accepted, decisions = fb._open_admission_gate(bars, [_signal()])

    assert len(accepted) == 1
    admission = accepted[0]["pattern_tags"][0]["admission"]
    assert admission["projected_frozen_boundary"] == 11.0
    assert admission["signal_day_atr14"] == 2.0
    assert admission["upper_open_limit"] == 17.0
    assert admission["entry_open"] == 17.0
    assert decisions[0]["admitted_pattern_ids"] == ["p1"]
    assert accepted[0]["pattern_ids"] == ["p1"]
    assert accepted[0]["patterns"][0]["evidence"][-1]["code"] == "entry_admission"


def test_open_admission_rejects_missing_emitted_breakout_evidence():
    accepted, decisions = fb._open_admission_gate(_bars(), [_signal(breakout=False)])

    assert accepted == []
    assert decisions[0]["rejected_tags"] == [{
        "pattern_id": "p1", "pattern_type": "w_bottom",
        "reason": "missing_signal_day_breakout_evidence"}]


def test_multitag_gate_admits_only_qualifying_emitted_tags():
    bars = _bars()
    signal = _signal()
    suppressed = {"pattern_id": "p2", "pattern_type": "triangle_bottom", "direction": "bullish",
                  "state": "confirmed", "confirmation_date": "2025-01-02", "parent_id": None,
                  "neckline": {"slope": 0.0, "intercept": 10.0},
                  "evidence": [{"code": "neckline_breakout", "value": {
                      "date": "2025-01-02", "frozen_neckline": 10.0}},
                               {"code": "execution_suppression", "value": {"suppressed": True}}]}
    signal["pattern_ids"].append("p2")
    signal["pattern_types"].append("triangle_bottom")
    signal["pattern_tags"].append({"pattern_id": "p2", "pattern_type": "triangle_bottom"})
    signal["patterns"].append(suppressed)

    accepted, decisions = fb._open_admission_gate(bars, [signal])

    assert accepted[0]["pattern_ids"] == ["p1"]
    assert accepted[0]["pattern_types"] == ["w_bottom"]
    assert decisions[0]["rejected_tags"] == [{
        "pattern_id": "p2", "pattern_type": "triangle_bottom", "reason": "execution_suppressed"}]


def test_experiment_replay_keeps_warmup_and_starts_fresh_period_accounts(monkeypatch):
    bars = _bars()
    observed = {"recognition_rows": None, "configs": [], "periods": []}

    def fake_signals(daily, timeframe, code, kind, calendar=None, trade_start=None,
                     strategy_version=fb.STRATEGY_VERSION):
        observed["recognition_rows"] = len(daily)
        config = current_recognition_experiment()
        observed["configs"].append(config)
        return [_signal(entry_date="2025-01-03"), _signal(
            signal_date="2025-01-03", entry_date="2025-01-04")]

    def fake_simulator(period_bars, signals, cash, lot_size):
        observed["periods"].append({"dates": list(period_bars.date.dt.strftime("%Y-%m-%d")),
                                    "signal_dates": [s["signal_date"] for s in signals],
                                    "cash": cash})
        return {"trades": [], "rejections": [], "order_events": [], "daily_equity": [],
                "ending_equity": cash, "accounting_checks": {}}

    monkeypatch.setattr(fb, "_prefix_signals", fake_signals)
    monkeypatch.setattr(fb, "simulate_fixed_trades", fake_simulator)
    result = fb.run_strategy_experiment(
        bars, "000001", "stock", fb.EXPERIMENT_NECKLINE_MIN3,
        [{"name": "early", "start": "2025-01-03", "end": "2025-01-03"},
         {"name": "late", "start": "2025-01-04", "end": "2025-01-04"}])

    assert observed["recognition_rows"] == 4
    assert observed["configs"][0].minimum_neckline_tests == 3
    assert observed["configs"][0].admission_types == fb.EXPERIMENT_PATTERN_TYPES
    assert observed["periods"] == [
        {"dates": ["2025-01-03"], "signal_dates": ["2025-01-02"], "cash": 100000.0},
        {"dates": ["2025-01-04"], "signal_dates": ["2025-01-03"], "cash": 100000.0},
    ]
    assert result["periods"][0]["starting_cash"] == result["periods"][1]["starting_cash"]
    assert current_recognition_experiment() is None
