import pandas as pd

from research.experiments.structure_boundary_losses_20261112.audit_backtest_economic_ledger import (
    audit_account,
    _history_missing,
)


def _account(exit_price=9.1):
    bars = pd.DataFrame([
        {"date": "2026-01-01", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
        {"date": "2026-01-02", "open": 10.0, "high": 10.5, "low": 9.0, "close": 10.2},
        {"date": "2026-01-05", "open": 10.0, "high": 10.6, "low": 9.5, "close": 10.5},
        {"date": "2026-01-06", "open": 9.1, "high": 9.8, "low": 8.9, "close": 9.2},
    ])
    pattern = {"pattern_id": "p1", "pattern_type": "w_bottom", "confirmation_date": "2026-01-02",
               "evidence": [{"code": "joint_lifecycle", "value": {
                   "paired_at": "2026-01-01", "entry_eligible_at_confirmation": True,
                   "entry_eligibility_at_confirmation_status": "known"}}]}
    signal = {"signal_date": "2026-01-02", "entry_date": "2026-01-05",
              "pattern_ids": ["p1"], "patterns": [pattern]}
    trade = {"signal_date": "2026-01-02", "entry_date": "2026-01-05", "entry_price": 10.0,
             "shares": 900, "pattern_ids": ["p1"], "stop_price": 9.2, "target_price": 12.0,
             "exit_date": "2026-01-06", "exit_price": exit_price, "exit_reason": "stop_loss",
             "exit_shares": 900, "pnl": (exit_price - 10.0) * 900,
             "return_pct": exit_price / 10.0 - 1}
    trade.update({"shares": 900, "exit_shares": 900, "pnl": (exit_price - 10.0) * 900})
    return bars, {"code": "600000", "kind": "stock", "starting_cash": 9500.0,
                  "ending_equity": 8690.0, "signals": [signal], "trades": [trade],
                  "daily_equity": [
                      {"date": "2026-01-01", "cash": 9500.0, "invested_value": 0.0,
                       "equity": 9500.0, "shares": 0},
                      {"date": "2026-01-02", "cash": 9500.0, "invested_value": 0.0,
                       "equity": 9500.0, "shares": 0},
                      {"date": "2026-01-05", "cash": 500.0, "invested_value": 9450.0,
                       "equity": 9950.0, "shares": 900},
                      {"date": "2026-01-06", "cash": 8690.0, "invested_value": 0.0,
                       "equity": 8690.0, "shares": 0}], "order_events": [
                      {"role": "entry", "status": "completed", "commission": 0.0},
                      {"role": "stop_loss", "status": "completed", "commission": 0.0}]}


def test_economic_ledger_verifies_next_open_t1_gap_stop_lots_and_daily_equity():
    bars, account = _account()

    result = audit_account(account, bars, cutoff="2026-01-06", lot_size=100)

    assert result["economics_passed"] is True
    assert result["historical_eligibility_passed"] is True
    assert result["checks"]["daily_equity_reconstructs"] is True
    assert result["checks"]["max_cash_lot_sizing"] is True
    assert result["checks"]["protective_exit_uses_gap_price"] is True
    assert result["checks"]["completed_order_events_match_trades"] is True


def test_economic_ledger_rejects_wrong_gap_stop_fill():
    bars, account = _account(exit_price=9.2)

    result = audit_account(account, bars, cutoff="2026-01-06", lot_size=100)

    assert result["checks"]["protective_exit_uses_gap_price"] is False
    assert result["economics_passed"] is False


def test_economic_ledger_does_not_assume_fees_are_zero_when_order_events_are_missing():
    bars, account = _account()
    account["order_events"] = []

    result = audit_account(account, bars, cutoff="2026-01-06", lot_size=100)

    assert result["checks"]["completed_order_events_match_trades"] is False
    assert result["economics_passed"] is False


def test_history_gap_counts_a_noneligible_sibling_even_when_signal_has_other_eligible_shape():
    missing_flag = {"pattern_type": "bull_flag", "evidence": [{"code": "joint_lifecycle", "value": {
        "entry_eligible_at_confirmation": None,
        "entry_eligibility_at_confirmation_status": "unknown_missing_confirmation_evidence"}}]}

    assert _history_missing(missing_flag) is True
