"""Combine the exploratory next-open gap gate with Backtrader exit variants."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
CAISEN_ROOT = HERE.parents[2]
WORKSPACE = CAISEN_ROOT.parent
SOURCE = HERE / "results/paired-100-v3-rebound-linked-20261110-attempt01"
RUN_OUT = SOURCE / "gapfilter-partial-v3.json"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(CAISEN_ROOT))

from replay_partial_profit_v17 import replay  # noqa: E402
from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades  # noqa: E402
from caisen.main_bottoms import atr  # noqa: E402


VARIANTS = (
    ("baseline_8_20", False, 0.0, False),
    ("gapfilter_8_20", True, 0.0, False),
    ("gapfilter_60_at_8_40_at_20", True, 0.6, False),
    ("gapfilter_70_at_8_30_at_20", True, 0.7, False),
    ("gapfilter_60_at_8_runner_breakeven", True, 0.6, True),
)


def _keep(signal, frame, atr_by_date, gap_filter):
    if not gap_filter:
        return True
    entry_date = signal.get("entry_date")
    if not entry_date:
        return True
    rows = frame.loc[frame.date.dt.strftime("%Y-%m-%d") == entry_date]
    if rows.empty:
        return False
    opening = float(rows.iloc[0].open)
    boundary = float(signal["boundary_value"])
    signal_atr = float(atr_by_date.get(signal["signal_date"], 0.0))
    return signal_atr <= 0 or opening <= boundary + 3.0 * signal_atr


def _portfolio_summary(account_rows, starting_capital):
    trades = [trade for row in account_rows for trade in row["trades"]]
    ending = sum(row["ending_equity"] for row in account_rows)
    metrics = calculate_metrics(trades, starting_capital, ending, [])
    curves = []
    all_dates = set()
    for row in account_rows:
        values = {point["date"]: (float(point["equity"]), float(point["invested_value"]))
                  for point in row["daily_equity"]}
        curves.append(sorted(values.items()))
        all_dates.update(values)
    equity_by_account = [100_000.0] * len(account_rows)
    invested_by_account = [0.0] * len(account_rows)
    pointers = [0] * len(curves)
    peak = starting_capital
    max_dd = 0.0
    invested_area = 0.0
    equity_area = 0.0
    for date in sorted(all_dates):
        for account_index, curve in enumerate(curves):
            pos = pointers[account_index]
            if pos < len(curve) and curve[pos][0] == date:
                equity_by_account[account_index], invested_by_account[account_index] = curve[pos][1]
                pointers[account_index] += 1
        total = sum(equity_by_account)
        invested_area += sum(invested_by_account)
        equity_area += total
        peak = max(peak, total)
        max_dd = max(max_dd, (peak - total) / peak if peak else 0.0)
    metrics.update({
        "account_count": len(account_rows), "maximum_drawdown": max_dd,
        "capital_utilization": invested_area / equity_area if equity_area else 0.0,
        "accounting_all_pass": all(all(row["accounting_checks"].values())
                                    for row in account_rows),
        "accounts_positive": sum(row["metrics"]["cumulative_return"] > 0
                                  for row in account_rows),
        "average_account_win_rate": sum(row["metrics"]["win_rate"] or 0
                                         for row in account_rows) / len(account_rows),
        "median_account_return": sorted(row["metrics"]["cumulative_return"]
                                         for row in account_rows)[len(account_rows) // 2],
    })
    return metrics


def main():
    files = sorted(SOURCE.glob("stock-*.json.gz"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 v3 account files, got {len(files)}")
    results = {name: [] for name, *_ in VARIANTS}
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        saved = json.load(gzip.open(path, "rt", encoding="utf-8"))
        code = saved["code"]
        frame = pd.read_csv(WORKSPACE / "data/stocks" / f"{code}.csv")
        frame["date"] = pd.to_datetime(frame.date)
        frame = frame[frame.date <= "2026-10-30"].reset_index(drop=True)
        atr_by_date = dict(zip(frame.date.dt.strftime("%Y-%m-%d"),
                               atr(frame, period=14).to_numpy(dtype=float)))
        signals = saved["signals_detail"]

        baseline = simulate_fixed_trades(frame, signals, cash=100_000.0)
        archived = saved["trades"]
        if len(baseline["trades"]) != len(archived):
            raise AssertionError(f"{code}: v3 baseline trade count diverges")
        for actual, previous in zip(baseline["trades"], archived):
            if (actual["entry_date"], actual["exit_date"], actual["shares"], actual["exit_reason"]) != (
                    previous["entry_date"], previous["exit_date"], previous["shares"],
                    previous["exit_reason"]):
                raise AssertionError(f"{code}: v3 baseline ledger diverges")
            if abs(float(actual["pnl"]) - float(previous["pnl"])) > 1e-7:
                raise AssertionError(f"{code}: v3 baseline PnL diverges")

        for name, gap_filter, fraction, breakeven in VARIANTS:
            admitted = [signal for signal in signals
                        if _keep(signal, frame, atr_by_date, gap_filter)]
            outcome = replay(frame, admitted, fraction=fraction, breakeven=breakeven)
            if not all(outcome["accounting_checks"].values()):
                raise AssertionError(f"{code}/{name}: ledger failed")
            metrics = calculate_metrics(outcome["trades"], 100_000.0,
                                        outcome["ending_equity"], outcome["daily_equity"])
            results[name].append({
                "code": code, "trades": outcome["trades"], "ending_equity": outcome["ending_equity"],
                "daily_equity": outcome["daily_equity"],
                "accounting_checks": outcome["accounting_checks"], "metrics": metrics,
                "admitted_signal_count": len(admitted), "filtered_signal_count": len(signals)-len(admitted),
            })
        if index % 10 == 0:
            print(f"{index}/100 accounts elapsed={time.perf_counter()-started:.1f}s", flush=True)

    report = {
        "experiment": "next_open_gap_filter_x_partial_exit_v1",
        "source": "v3 rebound-linked replay over saved v12 structures",
        "cutoff": "2026-10-30", "account_count": 100, "starting_capital": 10_000_000,
        "v8_reference": {"trades": 3780, "win_rate": 0.33756613756613757,
                          "cumulative_return": 0.6134487840000002},
        "variants": {name: _portfolio_summary(rows, 10_000_000)
                     for name, rows in results.items()},
        "account_details": results, "elapsed_seconds": time.perf_counter()-started,
        "limitations": [
            "development-only historical sample; the 100 symbols and outcomes have been repeatedly inspected",
            "reuses saved v12 structure candidates and v3 frozen boundary signals, not the production recognizer",
            "gap filter uses signal-day ATR14 and next-session opening price; no fees or slippage",
            "partial exits are an execution experiment, not the fixed 8/20 strategy",
        ],
    }
    RUN_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(RUN_OUT), "elapsed_seconds": report["elapsed_seconds"],
                      "variants": report["variants"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
