"""Backtrader test of full-position stop ratchets after excluding >3ATR gaps."""
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
OUT = SOURCE / "gapfilter-ratchet-v3.json"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(CAISEN_ROOT))

from replay_partial_profit_v17 import replay  # noqa: E402
from caisen.fixed_backtest import calculate_metrics  # noqa: E402
from caisen.main_bottoms import atr  # noqa: E402


VARIANTS = ((0.08, 0.01, "trigger_8_lock_1"),
            (0.08, 0.02, "trigger_8_lock_2"),
            (0.08, 0.03, "trigger_8_lock_3"),
            (0.10, 0.02, "trigger_10_lock_2"))


def _summary(rows):
    trades = [trade for row in rows for trade in row["trades"]]
    pnl = sum(float(trade["pnl"]) for trade in trades)
    losses = -sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) < 0)
    account_returns = [row["ending_equity"] / 100_000.0 - 1 for row in rows]
    return {
        "accounts": len(rows), "trades": len(trades),
        "win_rate": sum(float(trade["pnl"]) > 0 for trade in trades) / len(trades)
        if trades else None,
        "net_pnl": pnl, "cumulative_return": pnl / (len(rows) * 100_000.0) if rows else None,
        "profit_factor": sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0) / losses
        if losses else None,
        "accounts_positive": sum(value > 0 for value in account_returns),
        "median_account_return": sorted(account_returns)[len(account_returns)//2],
        "accounting_all_pass": all(all(row["accounting_checks"].values()) for row in rows),
        "ratchet_activations": sum(row["ratchet_activations"] for row in rows),
    }


def main():
    files = sorted(SOURCE.glob("stock-*.json.gz"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 v3 account files, got {len(files)}")
    rows = {name: [] for _, _, name in VARIANTS}
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
        filtered = []
        for signal in signals:
            if not signal.get("entry_date"):
                continue
            entry = frame[frame.date.dt.strftime("%Y-%m-%d") == signal["entry_date"]]
            if entry.empty:
                continue
            opening = float(entry.iloc[0].open)
            boundary = float(signal["boundary_value"])
            signal_atr = float(atr_by_date.get(signal["signal_date"], 0.0))
            if signal_atr > 0 and opening > boundary + 3.0 * signal_atr:
                continue
            filtered.append(signal)

        for trigger, floor, name in VARIANTS:
            outcome = replay(frame, filtered, fraction=0.0, ratchet_trigger=trigger,
                             ratchet_floor=floor)
            if not all(outcome["accounting_checks"].values()):
                raise AssertionError(f"{code}/{name}: accounting failed")
            metrics = calculate_metrics(outcome["trades"], 100_000.0,
                                        outcome["ending_equity"], outcome["daily_equity"])
            rows[name].append({
                "code": code, "trades": outcome["trades"],
                "ending_equity": outcome["ending_equity"],
                "accounting_checks": outcome["accounting_checks"],
                "ratchet_activations": outcome["ratchet_entries"], "metrics": metrics,
            })
        if index % 10 == 0:
            print(f"{index}/100 elapsed={time.perf_counter()-started:.1f}s", flush=True)

    report = {
        "experiment": "next_open_gap_filter_x_full_position_ratchet_v1",
        "source": "v3 rebound-linked saved v12 structures, 100 stocks",
        "cutoff": "2026-10-30", "base_entry_filter": "next open <= boundary + 3 * signal-day ATR14",
        "v8_reference": {"win_rate": 0.33756613756613757,
                          "cumulative_return": 0.6134487840000002},
        "variants": {name: _summary(accounts) for name, accounts in rows.items()},
        "account_details": rows, "elapsed_seconds": time.perf_counter()-started,
        "limitations": [
            "historical development sample, not an independent holdout",
            "uses saved v12 structure candidates and dynamic v3 boundary signals",
            "ratchet is activated at close after intraday high hits trigger and applies next session",
            "zero fees and slippage; same fixed 8 percent initial stop and 20 percent target",
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "elapsed_seconds": report["elapsed_seconds"],
                      "variants": report["variants"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
