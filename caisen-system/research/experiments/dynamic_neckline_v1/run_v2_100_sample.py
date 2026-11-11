"""Run a paired 100-stock, full-history v2 dynamic-neckline research replay."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAISEN_ROOT = HERE.parents[2]
WORKSPACE_ROOT = CAISEN_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAISEN_ROOT))

from dynamic_neckline import (DEFAULT_CUTOFF, LOT_SIZE, PAIRED_FILE, SAMPLE_FILE,
                              SOURCE, STARTING_CASH, STOP_FRACTION, TARGET_FRACTION,
                              _sha256, _iso, load_bars, load_structures,
                              pressure_points_2atr, build_breakout_signals)
from dynamic_neckline_v2 import build_boundary_revisions_v2
from dynamic_neckline_v3 import link_neckline_points
from dynamic_neckline_v4 import (filter_triangle_boundary_revisions,
                                  filter_triangle_breakout_signals)
from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades
from caisen.main_bottoms import atr


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _run_symbol(code, cutoff, output_dir, association_mode=False,
                triangle_apex_gate=False):
    started = time.perf_counter()
    bars, bars_path = load_bars(code, cutoff)
    account_path = SOURCE / "accounts" / f"stock-{code}.json"
    structures, account = load_structures(code, bars, cutoff, account_dir=account_path)
    pivots = pressure_points_2atr(bars, tick_size=0.01)
    atr_values = atr(bars, period=14).to_numpy(dtype=float)
    revisions, signals = [], []
    linked_structures = 0
    association_rejections = {}
    for structure in structures:
        association_points = None
        if association_mode:
            structure = link_neckline_points(
                structure, pivots, bars, atr_values, tick_size=0.01)
            if not structure.get("neckline_link_pass"):
                reason = structure.get("neckline_link_reason", "unknown")
                association_rejections[reason] = association_rejections.get(reason, 0) + 1
                continue
            linked_structures += 1
            association_points = structure.get("neckline_points") or None
        current = build_boundary_revisions_v2(
            bars, structure, pivots, tick_size=0.01,
            revision_shift_atr_multiple=1.0, minimum_shift_ticks=2.0,
            minimum_pressure_span_bars=10, minimum_revision_gap_bars=10,
            atr_values=atr_values, association_points=association_points)
        for revision in current:
            revision["source_instance_id"] = structure["instance_id"]
        if triangle_apex_gate:
            current = filter_triangle_boundary_revisions(structure, current)
        revisions.extend(current)
        current_signals = build_breakout_signals(bars, structure, current)
        if triangle_apex_gate:
            current_signals = filter_triangle_breakout_signals(
                structure, current, current_signals, bars)
        signals.extend(current_signals)
    signals.sort(key=lambda row: (row["signal_date"], row["boundary_version_id"]))
    simulation = simulate_fixed_trades(bars, signals, cash=STARTING_CASH, lot_size=LOT_SIZE)
    metrics = calculate_metrics(simulation["trades"], STARTING_CASH,
                                simulation["ending_equity"], simulation["daily_equity"])
    result = {
        "code": code, "kind": "stock", "cutoff": cutoff, "status": "complete",
        "bars": len(bars), "first_date": _iso(bars.date.iloc[0]) if len(bars) else None,
        "last_date": _iso(bars.date.iloc[-1]) if len(bars) else None,
        "structures": len(structures), "linked_structures": linked_structures,
        "association_rejections": association_rejections,
        "pressure_points_2atr": len(pivots),
        "boundary_revisions": len(revisions), "signals": len(signals),
        "metrics": metrics, "ending_equity": simulation["ending_equity"],
        "trades": simulation["trades"], "signals_detail": signals,
        "boundary_revisions_detail": revisions, "rejections": simulation["rejections"],
        "daily_equity": simulation["daily_equity"],
        "accounting_checks": simulation["accounting_checks"],
        "engine": simulation["engine"],
        "provenance": {
            "bars_path": str(bars_path), "bars_sha256": _sha256(bars_path),
            "account_path": str(account_path), "account_sha256": _sha256(account_path),
            "source_rule_version": account.get("source_rule_version"),
            "source_engine_version": account.get("engine_version"),
            "data_cutoff": cutoff,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    path = Path(output_dir) / f"stock-{code}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
    return {"code": code, "bars": len(bars), "structures": len(structures),
            "linked_structures": linked_structures,
            "association_rejections": association_rejections,
            "pressure_points_2atr": len(pivots), "boundary_revisions": len(revisions),
            "signals": len(signals), "metrics": metrics,
            "accounting_checks": simulation["accounting_checks"],
            "runtime_seconds": result["runtime_seconds"],
            "result_path": str(path)}


def _summarize(results, failures, sample_count, cutoff, elapsed, workers, output_dir,
               association_mode=False, triangle_apex_gate=False):
    trades = []
    for result in results:
        with gzip.open(result["result_path"], "rt", encoding="utf-8") as stream:
            trades.extend(json.load(stream).get("trades", []))
    pnls = [float(trade["pnl"]) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    account_returns = [item["metrics"]["cumulative_return"] for item in results]
    account_dd = [item["metrics"]["maximum_drawdown"] for item in results]
    pnl = sum(pnls)
    family = {}
    for name in ("w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"):
        tagged = [trade for trade in trades
                  if name in (trade.get("pattern_types") or [])]
        tagged_pnl = [float(trade["pnl"]) for trade in tagged]
        gross_loss = -sum(value for value in tagged_pnl if value < 0)
        family[name] = {
            "tagged_trade_count": len(tagged),
            "tagged_win_rate": sum(value > 0 for value in tagged_pnl) / len(tagged_pnl)
            if tagged_pnl else None,
            "tagged_profit_factor": sum(value for value in tagged_pnl if value > 0) / gross_loss
            if gross_loss else None,
            "tagged_net_pnl": sum(tagged_pnl),
        }
    ranking = [{"code": row["code"], "return": row["metrics"]["cumulative_return"],
                "drawdown": row["metrics"]["maximum_drawdown"],
                "trades": row["metrics"]["trade_count"],
                "win_rate": row["metrics"]["win_rate"]}
               for row in results]
    ranking.sort(key=lambda row: row["return"])
    summary = {
        "prototype": ("dynamic_neckline_v3_rebound_linked" if association_mode else
                      "dynamic_neckline_v4_triangle_apex_window" if triangle_apex_gate else
                      "dynamic_neckline_v2"),
        "scope": "saved_v12_structures_only",
        "association_mode": bool(association_mode),
        "triangle_apex_gate": bool(triangle_apex_gate),
        "sample_count": sample_count, "completed_accounts": len(results),
        "failed_accounts": failures, "cutoff": cutoff, "workers": workers,
        "runtime_seconds": elapsed, "total_structures": sum(r["structures"] for r in results),
        "total_linked_structures": sum(r.get("linked_structures", 0) for r in results),
        "association_rejections": {
            reason: sum(r.get("association_rejections", {}).get(reason, 0) for r in results)
            for reason in sorted({reason for r in results
                                  for reason in r.get("association_rejections", {})})},
        "total_2atr_pressure_points": sum(r["pressure_points_2atr"] for r in results),
        "total_boundary_revisions": sum(r["boundary_revisions"] for r in results),
        "total_signals": sum(r["signals"] for r in results), "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else None,
        "profit_factor": sum(wins) / -sum(losses) if losses else None,
        "net_pnl": pnl, "starting_capital": sample_count * STARTING_CASH,
        "equal_start_capital_return": pnl / (sample_count * STARTING_CASH)
        if sample_count else None,
        "positive_accounts": sum(value > 0 for value in account_returns),
        "account_return_distribution": {"mean": float(np.mean(account_returns)) if account_returns else None,
                                        "median": _percentile(account_returns, 50),
                                        "p10": _percentile(account_returns, 10),
                                        "p90": _percentile(account_returns, 90),
                                        "min": min(account_returns) if account_returns else None,
                                        "max": max(account_returns) if account_returns else None},
        "account_max_drawdown_distribution": {"median": _percentile(account_dd, 50),
                                              "p90": _percentile(account_dd, 90),
                                              "max": max(account_dd) if account_dd else None},
        "pattern_families": family,
        "top_10_worst_accounts": ranking[:10], "top_10_best_accounts": ranking[-10:][::-1],
        "accounting_all_pass": (not failures and all(
            all(bool(value) for value in r["accounting_checks"].values()) for r in results)),
        "limitations": [
            "only replays structures present in the saved v12 instance_events; missed structures are not recovered",
            "candidate boundary coverage is limited to stored structural anchors; full formation candle coverage is not required",
            "zero commission and slippage; fixed 8 percent stop and 20 percent target",
            "this is a research experiment, not the production recognizer or a holdout validation",
        ],
        "accounts": [{k: row[k] for k in ("code", "bars", "structures", "pressure_points_2atr",
                                           "boundary_revisions", "signals", "metrics",
                                           "runtime_seconds", "result_path")}
                     for row in sorted(results, key=lambda item: item["code"])],
    }
    path = Path(output_dir) / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run(symbols=None, cutoff=DEFAULT_CUTOFF, output_dir=None, workers=8,
        association_mode=False, triangle_apex_gate=False):
    if association_mode and triangle_apex_gate:
        raise ValueError("run the neckline association and triangle-apex experiments separately")
    selection = json.load(open(SAMPLE_FILE, encoding="utf-8"))
    full_sample = [item["code"] for item in selection["symbols"]]
    symbols = list(symbols or full_sample)
    default_output = ("paired-100-v3-rebound-linked-20261110-attempt01" if association_mode else
                      "paired-100-v4-triangle-apex-window-20261111" if triangle_apex_gate else
                      "paired-100-v2-20261110-attempt01")
    output_dir = Path(output_dir or HERE / "results" / default_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results, failures = [], []
    print(f"START {len(symbols)} stocks cutoff={cutoff} workers={workers} "
          f"association_mode={association_mode} triangle_apex_gate={triangle_apex_gate} "
          f"output={output_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_symbol, code, cutoff, str(output_dir), association_mode,
                               triangle_apex_gate): code
                   for code in symbols}
        for index, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"{index}/{len(symbols)} {code}: bars={result['bars']} "
                      f"structures={result['structures']} revisions={result['boundary_revisions']} "
                      f"signals={result['signals']} trades={result['metrics']['trade_count']} "
                      f"{result['runtime_seconds']:.1f}s", flush=True)
            except Exception as error:
                failure = {"code": code, "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                print(f"{index}/{len(symbols)} {code}: FAILED {failure['error']}", flush=True)
    elapsed = time.perf_counter() - started
    summary = _summarize(results, failures, len(symbols), cutoff, elapsed, workers,
                         output_dir, association_mode=association_mode,
                         triangle_apex_gate=triangle_apex_gate)
    manifest = {
        "sample_selection_path": str(SAMPLE_FILE), "sample_selection_sha256": _sha256(SAMPLE_FILE),
        "sample_selection_rule": selection.get("selection_rule"),
        "cutoff": cutoff, "workers": workers,
        "source_run_manifest_path": str(SOURCE / "run-manifest.json"),
        "source_run_manifest_sha256": _sha256(SOURCE / "run-manifest.json"),
        "paired_v8_v12_comparison_path": str(PAIRED_FILE),
        "paired_v8_v12_comparison_sha256": _sha256(PAIRED_FILE),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "v2_module_sha256": _sha256(HERE / "dynamic_neckline_v2.py"),
        "v3_module_sha256": _sha256(HERE / "dynamic_neckline_v3.py"),
        "v4_module_sha256": _sha256(HERE / "dynamic_neckline_v4.py"),
        "association_mode": bool(association_mode),
        "triangle_apex_gate": bool(triangle_apex_gate),
        "symbols": [{"code": code,
                     "account_path": str(SOURCE / "accounts" / f"stock-{code}.json"),
                     "account_sha256": _sha256(SOURCE / "accounts" / f"stock-{code}.json"),
                     "bars_path": str(WORKSPACE_ROOT / "data/stocks" / f"{code}.csv"),
                     "bars_sha256": _sha256(WORKSPACE_ROOT / "data/stocks" / f"{code}.csv")}
                    for code in symbols],
        "summary": {k: v for k, v in summary.items() if k != "accounts"},
    }
    (output_dir / "data-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="", help="comma-separated stock codes; default is fixed 100")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--association-mode", action="store_true",
                        help="require necklines to match the linked rebound pivots")
    parser.add_argument("--triangle-apex-gate", action="store_true",
                        help="recompute each triangle revision's apex and keep breakouts in 50%%-75%% window")
    args = parser.parse_args()
    symbols = [part.strip() for part in args.symbols.split(",") if part.strip()] or None
    summary = run(symbols, args.cutoff, args.output_dir or None, args.workers,
                  association_mode=args.association_mode,
                  triangle_apex_gate=args.triangle_apex_gate)
    print(json.dumps({k: v for k, v in summary.items() if k != "accounts"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
