"""Research-only next-open admission filters for the saved v3 100-stock replay."""
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
OUT = SOURCE / "entry-gap-filter-v3.json"
sys.path.insert(0, str(CAISEN_ROOT))

from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades  # noqa: E402
from caisen.main_bottoms import atr  # noqa: E402


VARIANTS = ("baseline", "reject_below_boundary", "reject_above_3atr", "combined")


def _metrics(rows):
    trades = [trade for row in rows for trade in row["trades"]]
    pnl = sum(float(trade["pnl"]) for trade in trades)
    wins = sum(float(trade["pnl"]) > 0 for trade in trades)
    losses = -sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) < 0)
    return {
        "accounts": len(rows),
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else None,
        "net_pnl": pnl,
        "cumulative_return": pnl / (len(rows) * 100_000.0) if rows else None,
        "profit_factor": sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0) / losses
        if losses else None,
        "accounting_all_pass": all(all(item["accounting_checks"].values()) for item in rows),
    }


def _admit(signal, bars, atr_by_date, variant):
    if variant == "baseline":
        return True, None
    entry_date = signal.get("entry_date")
    if not entry_date or entry_date not in bars.index:
        return False, "no_next_session"
    opening = float(bars.at[entry_date, "open"])
    boundary = float(signal["boundary_value"])
    atr_value = float(atr_by_date.get(signal["signal_date"], 0.0))
    if variant in {"reject_below_boundary", "combined"} and opening <= boundary:
        return False, "next_open_at_or_below_frozen_boundary"
    if variant in {"reject_above_3atr", "combined"} and atr_value > 0 \
            and opening > boundary + 3.0 * atr_value:
        return False, "next_open_more_than_3_signal_atr_above_boundary"
    return True, None


def main():
    files = sorted(SOURCE.glob("stock-*.json.gz"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 v3 account files, got {len(files)}")
    results = {variant: [] for variant in VARIANTS}
    rejected = {variant: {} for variant in VARIANTS}
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        account = json.load(gzip.open(path, "rt", encoding="utf-8"))
        code = account["code"]
        frame = pd.read_csv(WORKSPACE / "data/stocks" / f"{code}.csv")
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame.date <= "2026-10-30"].set_index(
            pd.to_datetime(frame.loc[frame.date <= "2026-10-30", "date"]).dt.strftime("%Y-%m-%d"))
        atr_values = atr(frame.reset_index(drop=True), period=14)
        atr_by_date = dict(zip(frame.index, atr_values))
        signals = account["signals_detail"]
        baseline = simulate_fixed_trades(frame.reset_index(drop=True), signals, cash=100_000.0)
        saved = account["trades"]
        if len(baseline["trades"]) != len(saved):
            raise AssertionError(f"{code}: baseline trade count diverges")
        for actual, old in zip(baseline["trades"], saved):
            if (actual["entry_date"], actual["exit_date"], actual["shares"], actual["exit_reason"]) != (
                    old["entry_date"], old["exit_date"], old["shares"], old["exit_reason"]):
                raise AssertionError(f"{code}: baseline replay diverges")
            if abs(float(actual["pnl"]) - float(old["pnl"])) > 1e-7:
                raise AssertionError(f"{code}: baseline PnL diverges")

        for variant in VARIANTS:
            allowed = []
            for signal in signals:
                ok, reason = _admit(signal, frame, atr_by_date, variant)
                if ok:
                    allowed.append(signal)
                elif reason:
                    rejected[variant][reason] = rejected[variant].get(reason, 0) + 1
            simulation = simulate_fixed_trades(
                frame.reset_index(drop=True), allowed, cash=100_000.0)
            if not all(simulation["accounting_checks"].values()):
                raise AssertionError(f"{code}/{variant}: accounting failed")
            metrics = calculate_metrics(simulation["trades"], 100_000.0,
                                        simulation["ending_equity"], simulation["daily_equity"])
            results[variant].append({
                "code": code, "trades": simulation["trades"],
                "ending_equity": simulation["ending_equity"],
                "accounting_checks": simulation["accounting_checks"], "metrics": metrics,
            })
        if index % 10 == 0:
            print(f"{index}/100 elapsed={time.perf_counter()-started:.1f}s", flush=True)

    report = {
        "experiment": "next_open_boundary_gap_admission_v1",
        "source": "dynamic_neckline_v3_rebound_linked_saved_v12_structures_only",
        "cutoff": "2026-10-30", "account_count": 100,
        "baseline_reference_v8": {
            "trades": 3780, "win_rate": 0.33756613756613757,
            "cumulative_return": 0.6134487840000002,
        },
        "variants": {name: _metrics(rows) for name, rows in results.items()},
        "rejected_signal_counts": rejected,
        "account_details": results,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "historical exploratory experiment on a repeatedly inspected 100-stock sample",
            "reuses v3 signals from saved v12 structures; it does not compare new recognition",
            "zero fees and slippage; each account uses fixed 8 percent stop and 20 percent target",
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "elapsed_seconds": report["elapsed_seconds"],
                      "variants": report["variants"], "rejected": rejected},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
