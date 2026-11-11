"""Test volume confirmation gates on top of the v3 next-open gap filter."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAISEN_ROOT = HERE.parents[2]
WORKSPACE = CAISEN_ROOT.parent
SOURCE = HERE / "results/paired-100-v3-rebound-linked-20261110-attempt01"
OUT = SOURCE / "gap-volume-filter-v3.json"
sys.path.insert(0, str(CAISEN_ROOT))

from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades  # noqa: E402
from caisen.main_bottoms import atr  # noqa: E402

VOLUME_GATES = (None, 0.8, 1.0, 1.5, 2.0)


def _summary(rows):
    trades = [trade for row in rows for trade in row["trades"]]
    pnl = sum(float(trade["pnl"]) for trade in trades)
    wins = sum(float(trade["pnl"]) > 0 for trade in trades)
    losses = -sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) < 0)
    return {
        "accounts": len(rows), "trade_count": len(trades),
        "win_rate": wins / len(trades) if trades else None,
        "net_pnl": pnl, "cumulative_return": pnl / (len(rows) * 100_000.0),
        "profit_factor": sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0) / losses
        if losses else None,
        "positive_accounts": sum(row["metrics"]["cumulative_return"] > 0 for row in rows),
        "accounting_all_pass": all(all(row["accounting_checks"].values()) for row in rows),
    }


def main():
    files = sorted(SOURCE.glob("stock-*.json.gz"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 fixed v3 accounts, got {len(files)}")
    outputs = {str(gate): [] for gate in VOLUME_GATES}
    rejected = {str(gate): {"gap": 0, "volume": 0, "missing_volume_history": 0}
                for gate in VOLUME_GATES}
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        saved = json.load(gzip.open(path, "rt", encoding="utf-8"))
        code = saved["code"]
        bars = pd.read_csv(WORKSPACE / "data/stocks" / f"{code}.csv")
        bars["date"] = pd.to_datetime(bars.date)
        bars = bars[bars.date <= "2026-10-30"].sort_values("date").reset_index(drop=True)
        dates = bars.date.dt.strftime("%Y-%m-%d").tolist()
        index_by_date = {date: i for i, date in enumerate(dates)}
        atr_values = atr(bars, period=14).to_numpy(dtype=float)
        volume_ratios = {}
        for signal_date, i in index_by_date.items():
            if i < 20:
                volume_ratios[signal_date] = np.nan
            else:
                prior = pd.to_numeric(bars.volume.iloc[i-20:i], errors="coerce")
                average = float(prior.mean())
                current = float(bars.volume.iloc[i])
                volume_ratios[signal_date] = current / average if average > 0 else np.nan
        signals = saved["signals_detail"]
        official = simulate_fixed_trades(bars, signals, cash=100_000.0)
        archived = saved["trades"]
        if len(official["trades"]) != len(archived):
            raise AssertionError(f"{code}: baseline count mismatch")
        for actual, old in zip(official["trades"], archived):
            if (actual["entry_date"], actual["exit_date"], actual["exit_reason"], actual["shares"]) != (
                    old["entry_date"], old["exit_date"], old["exit_reason"], old["shares"]):
                raise AssertionError(f"{code}: baseline ledger differs")
            if abs(float(actual["pnl"]) - float(old["pnl"])) > 1e-7:
                raise AssertionError(f"{code}: baseline PnL differs")

        for gate in VOLUME_GATES:
            admitted = []
            gate_name = str(gate)
            for signal in signals:
                entry_date = signal.get("entry_date")
                if not entry_date or entry_date not in index_by_date:
                    continue
                open_price = float(bars.open.iloc[index_by_date[entry_date]])
                boundary = float(signal["boundary_value"])
                signal_atr = float(atr_values[index_by_date[signal["signal_date"]]])
                if signal_atr > 0 and open_price > boundary + 3.0 * signal_atr:
                    rejected[gate_name]["gap"] += 1
                    continue
                ratio = volume_ratios.get(signal["signal_date"], np.nan)
                if gate is not None and (not np.isfinite(ratio) or ratio < gate):
                    rejected[gate_name]["volume"] += 1
                    if not np.isfinite(ratio):
                        rejected[gate_name]["missing_volume_history"] += 1
                    continue
                admitted.append(signal)
            simulation = simulate_fixed_trades(bars, admitted, cash=100_000.0)
            if not all(simulation["accounting_checks"].values()):
                raise AssertionError(f"{code}/{gate}: accounting failed")
            metrics = calculate_metrics(simulation["trades"], 100_000.0,
                                        simulation["ending_equity"], simulation["daily_equity"])
            outputs[gate_name].append({
                "code": code, "trades": simulation["trades"],
                "ending_equity": simulation["ending_equity"],
                "accounting_checks": simulation["accounting_checks"], "metrics": metrics,
                "signals_admitted": len(admitted),
            })
        if index % 10 == 0:
            print(f"{index}/100 elapsed={time.perf_counter()-started:.1f}s", flush=True)

    report = {
        "experiment": "next_open_gap_filter_x_signal_volume_gate_v1",
        "source": "dynamic v3 rebound-linked saved v12 structures; fixed 100 stocks",
        "cutoff": "2026-10-30", "volume_reference": "current signal-day volume / previous 20-session mean",
        "gap_rule": "reject next open > frozen boundary + 3 * signal-day ATR14",
        "v8_reference": {"trade_count": 3780, "win_rate": 0.33756613756613757,
                          "cumulative_return": 0.6134487840000002},
        "variants": {name: _summary(rows) for name, rows in outputs.items()},
        "rejected_signal_counts": rejected, "account_details": outputs,
        "elapsed_seconds": time.perf_counter()-started,
        "limitations": [
            "development cohort, repeatedly inspected; not holdout validation",
            "reuses saved v12 structures and v3 frozen boundaries",
            "zero commissions and slippage; fixed 8 percent stop and 20 percent target",
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "elapsed_seconds": report["elapsed_seconds"],
                      "variants": report["variants"], "rejected": rejected},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
