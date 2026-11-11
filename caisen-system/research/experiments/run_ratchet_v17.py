"""Backtrader experiment: ratchet the 8%/20% full-position bracket after MFE.

The trigger is detected on a completed daily bar; the revised stop is active
from the next bar. This script only consumes frozen v17 signals.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import json
import math
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "reports/daily-research/v17-directional-suppression-neckline-revisions-100-20261110-attempt02"
DATA = ROOT.parent / "data/stocks"
sys.path.insert(0, str(HERE))
from replay_partial_profit_v17 import replay  # noqa: E402
sys.path.insert(0, str(ROOT))
from caisen.fixed_backtest import simulate_fixed_trades, calculate_metrics  # noqa: E402

VARIANTS = [
    (0.08, 0.00, "trigger_8_lock_0"),
    (0.08, 0.01, "trigger_8_lock_1"),
    (0.10, 0.02, "trigger_10_lock_2"),
    (0.12, 0.04, "trigger_12_lock_4"),
]


def summarize_accounts(account_rows, curves):
    all_trades = [t for account in account_rows for t in account["trades"]]
    ending = sum(a["ending_equity"] for a in account_rows)
    metrics = calculate_metrics(all_trades, 10_000_000, ending, [])
    union_dates = set()
    rows = []
    for curve in curves:
        values = {p["date"]: (float(p["equity"]), float(p["invested_value"])) for p in curve}
        rows.append(sorted(values.items()))
        union_dates.update(values)
    dates = sorted(union_dates)
    cash_equity = [100_000.0] * len(rows)
    invested = [0.0] * len(rows)
    cursors = [0] * len(rows)
    curve = []
    for date in dates:
        for i, values in enumerate(rows):
            j = cursors[i]
            if j < len(values) and values[j][0] == date:
                cash_equity[i], invested[i] = values[j][1]
                cursors[i] += 1
        curve.append((sum(cash_equity), sum(invested)))
    peak, drawdown = 10_000_000.0, 0.0
    for equity, _ in curve:
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak if peak else 0.0)
    metrics["maximum_drawdown"] = drawdown
    metrics["capital_utilization"] = (sum(x[1] for x in curve) / sum(x[0] for x in curve)
                                      if curve and sum(x[0] for x in curve) else 0.0)
    bands = {"through_2020": [], "2021_2023": [], "2024_plus": []}
    for trade in all_trades:
        year = int(trade["signal_date"][:4])
        band = "through_2020" if year <= 2020 else "2021_2023" if year <= 2023 else "2024_plus"
        bands[band].append(trade)
    metrics["signal_date_bands"] = {
        name: {"trades": len(values), "wins": sum(t["pnl"] > 0 for t in values),
               "win_rate": sum(t["pnl"] > 0 for t in values) / len(values) if values else None,
               "net_pnl": sum(t["pnl"] for t in values)}
        for name, values in bands.items()}
    metrics.update({
        "accounts": len(account_rows),
        "accounting_all_pass": all(all(row["accounting_checks"].values()) for row in account_rows),
        "accounts_positive": sum(a["metrics"]["cumulative_return"] > 0 for a in account_rows),
        "median_account_return": sorted(a["metrics"]["cumulative_return"] for a in account_rows)[50],
        "average_account_win_rate": sum(a["metrics"]["win_rate"] or 0 for a in account_rows) / len(account_rows),
        "ratchet_activations": sum(a["ratchet_activations"] for a in account_rows),
    })
    return metrics


def main():
    files = sorted((RUN / "accounts").glob("stock-*.json"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 fixed accounts, got {len(files)}")
    account_rows = {name: [] for _, _, name in VARIANTS}
    curves = {name: [] for _, _, name in VARIANTS}
    started = time.time()
    for idx, path in enumerate(files, 1):
        saved = json.loads(path.read_text())
        code = saved["code"]
        frame = pd.read_csv(DATA / f"{code}.csv")
        frame["date"] = pd.to_datetime(frame.date)
        frame = frame[frame.date <= "2026-10-30"].reset_index(drop=True)
        baseline = simulate_fixed_trades(frame, saved["signals"], cash=100_000)
        if len(baseline["trades"]) != len(saved["trades"]):
            raise AssertionError(f"{code}: current baseline trade count mismatch")
        for current, frozen in zip(baseline["trades"], saved["trades"]):
            if (current["entry_date"], current["exit_date"], current["exit_reason"], current["shares"]) != (
                    frozen["entry_date"], frozen["exit_date"], frozen["exit_reason"], frozen["shares"]):
                raise AssertionError(f"{code}: baseline replay differs from frozen v17 trades")
            if not math.isclose(current["pnl"], frozen["pnl"], abs_tol=1e-7):
                raise AssertionError(f"{code}: baseline PnL differs from frozen v17 trades")
        for trigger, floor, name in VARIANTS:
            outcome = replay(frame, saved["signals"], fraction=0, ratchet_trigger=trigger,
                             ratchet_floor=floor)
            if not all(outcome["accounting_checks"].values()):
                raise AssertionError(f"{code}/{name}: accounting failed {outcome['accounting_checks']}")
            metrics = calculate_metrics(outcome["trades"], 100_000, outcome["ending_equity"],
                                        outcome["daily_equity"])
            account_rows[name].append({"code": code, "metrics": metrics,
                                       "ending_equity": outcome["ending_equity"],
                                       "accounting_checks": outcome["accounting_checks"],
                                       "ratchet_activations": outcome["ratchet_entries"],
                                       "trades": outcome["trades"]})
            curves[name].append(outcome["daily_equity"])
        if idx % 10 == 0:
            print(f"{idx}/100 accounts; elapsed={time.time()-started:.1f}s", flush=True)
    summary = {name: summarize_accounts(rows, curves[name]) for name, rows in account_rows.items()}
    result = {"experiment": "v17_full_position_ratchet_after_mfe_v1",
              "baseline_run_id": "7f79f305fe214de48844a9f065632774",
              "recognizer": "caisen-daily-boundary-first-v11",
              "source_rule_version": "caisen-main-bottoms-v16",
              "cutoff": "2026-10-30", "account_count": 100,
              "starting_capital": 10_000_000, "variants": summary,
              "account_details": account_rows, "elapsed_seconds": time.time()-started,
              "trigger_observation": "Intraday high is known only at close; changed stop is active from next session.",
              "interpretation": "Research only on historical development data; production rules unchanged."}
    out = RUN / "ratchet-experiment-v17.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "elapsed_seconds": result["elapsed_seconds"],
                      "variants": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
