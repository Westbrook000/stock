"""Full-history 100-stock sweep of the material neckline revision threshold."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
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
OUT = SOURCE / "revision-shift-sweep-v3.json"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAISEN_ROOT))

from dynamic_neckline import (DEFAULT_CUTOFF, SOURCE as V12_RUN, _sha256, load_bars,
                              load_structures, pressure_points_2atr, build_breakout_signals)
from dynamic_neckline_v2 import build_boundary_revisions_v2
from dynamic_neckline_v3 import link_neckline_points
from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades
from caisen.main_bottoms import atr

THRESHOLDS = (0.5, 1.0, 1.5, 2.0)
WORKERS = 8


def _one(code):
    saved_path = SOURCE / f"stock-{code}.json.gz"
    saved = json.load(gzip.open(saved_path, "rt", encoding="utf-8"))
    bars, bars_path = load_bars(code, DEFAULT_CUTOFF)
    account_path = V12_RUN / "accounts" / f"stock-{code}.json"
    structures, account = load_structures(code, bars, DEFAULT_CUTOFF, account_dir=account_path)
    pivots = pressure_points_2atr(bars, tick_size=0.01)
    atr_values = atr(bars, period=14).to_numpy(dtype=float)
    results = {}
    for threshold in THRESHOLDS:
        revisions, signals = [], []
        linked = 0
        for structure in structures:
            structure = link_neckline_points(
                structure, pivots, bars, atr_values, tick_size=0.01)
            if not structure.get("neckline_link_pass"):
                continue
            linked += 1
            current = build_boundary_revisions_v2(
                bars, structure, pivots, tick_size=0.01,
                revision_shift_atr_multiple=threshold, minimum_shift_ticks=2.0,
                minimum_pressure_span_bars=10, minimum_revision_gap_bars=10,
                atr_values=atr_values,
                association_points=structure.get("neckline_points") or None)
            for revision in current:
                revision["source_instance_id"] = structure["instance_id"]
            revisions.extend(current)
            signals.extend(build_breakout_signals(bars, structure, current))
        signals.sort(key=lambda row: (row["signal_date"], row["boundary_version_id"]))
        simulation = simulate_fixed_trades(bars, signals, cash=100_000.0)
        if not all(simulation["accounting_checks"].values()):
            raise AssertionError(f"{code}/{threshold}: ledger check failed")
        metrics = calculate_metrics(simulation["trades"], 100_000.0,
                                    simulation["ending_equity"], simulation["daily_equity"])
        results[str(threshold)] = {
            "code": code, "bars": len(bars), "first_date": str(bars.date.iloc[0].date()),
            "last_date": str(bars.date.iloc[-1].date()), "linked_structures": linked,
            "boundary_revisions": len(revisions), "signals": len(signals),
            "metrics": metrics, "trades": simulation["trades"],
            "ending_equity": simulation["ending_equity"],
            "accounting_checks": simulation["accounting_checks"],
            "rejections": simulation["rejections"],
            "provenance": {"bars_sha256": _sha256(bars_path),
                           "v12_account_sha256": _sha256(account_path)},
        }
    # The 1 ATR experimental setting must exactly recreate the archived v3 run.
    baseline = results["1.0"]
    if len(baseline["trades"]) != len(saved["trades"]):
        raise AssertionError(f"{code}: 1 ATR trade count differs from archived V3")
    for actual, archived in zip(baseline["trades"], saved["trades"]):
        for key in ("entry_date", "exit_date", "exit_reason", "shares"):
            if actual.get(key) != archived.get(key):
                raise AssertionError(f"{code}: 1 ATR {key} differs from archived V3")
        if abs(float(actual["pnl"]) - float(archived["pnl"])) > 1e-7:
            raise AssertionError(f"{code}: 1 ATR PnL differs from archived V3")
    return code, results


def _aggregate(rows):
    accounts = list(rows.values())
    trades = [trade for row in accounts for trade in row["trades"]]
    pnl = sum(float(trade["pnl"]) for trade in trades)
    wins = sum(float(trade["pnl"]) > 0 for trade in trades)
    losses = -sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) < 0)
    return {
        "account_count": len(accounts), "trade_count": len(trades),
        "win_rate": wins / len(trades) if trades else None,
        "net_pnl": pnl, "cumulative_return": pnl / (len(accounts) * 100_000.0)
        if accounts else None,
        "profit_factor": sum(float(trade["pnl"]) for trade in trades if float(trade["pnl"]) > 0) / losses
        if losses else None,
        "positive_accounts": sum(row["metrics"]["cumulative_return"] > 0 for row in accounts),
        "accounting_all_pass": all(all(row["accounting_checks"].values()) for row in accounts),
        "total_boundary_revisions": sum(row["boundary_revisions"] for row in accounts),
        "total_signals": sum(row["signals"] for row in accounts),
    }


def main():
    manifest = json.loads((SOURCE / "data-manifest.json").read_text(encoding="utf-8"))
    codes = [item["code"] for item in manifest["symbols"]]
    if len(codes) != 100:
        raise SystemExit(f"Expected fixed 100 stock cohort, got {len(codes)}")
    outputs = {str(threshold): {} for threshold in THRESHOLDS}
    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_one, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                symbol_results_code, variants = future.result()
                for threshold, row in variants.items():
                    outputs[threshold][symbol_results_code] = row
                print(f"{index}/100 {code} complete", flush=True)
            except Exception as error:
                failures.append({"code": code, "error": f"{type(error).__name__}: {error}"})
                print(f"{index}/100 {code} FAILED {error}", flush=True)
    report = {
        "experiment": "material_neckline_revision_shift_sweep_v3",
        "cutoff": DEFAULT_CUTOFF, "sample_id": "paired-100-v8-v12-v16-stock-cohort-v1",
        "sample_count": len(codes), "completed_accounts": len(outputs["1.0"]),
        "failures": failures, "workers": WORKERS,
        "baseline_v3": {threshold: _aggregate(accounts) for threshold, accounts in outputs.items()},
        "account_details": outputs, "elapsed_seconds": time.perf_counter()-started,
        "v12_source_run_manifest_sha256": _sha256(V12_RUN / "run-manifest.json"),
        "interpretation": "Research sweep on the repeatedly inspected 100-stock development cohort, not holdout evidence.",
        "fixed_execution": {"initial_cash": 100_000, "lot_size": 100,
                             "stop_fraction": 0.08, "target_fraction": 0.20,
                             "commission": 0, "slippage": 0, "end_of_period_close": True},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "elapsed_seconds": report["elapsed_seconds"],
                      "failures": len(failures), "variants": report["baseline_v3"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
