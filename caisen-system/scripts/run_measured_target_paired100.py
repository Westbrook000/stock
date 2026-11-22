#!/usr/bin/env python3
"""Replay saved G2 signals with the production 8% stop/measured-target broker."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing as mp
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "research/experiments/structure_boundary_losses_20261112/dynamic_reline/results/baseline-A-perf2-extrema-10w-attempt02"
SNAPSHOT = Path("/private/tmp/caisen-structure-boundary-snapshot-20261112-v2")
OUT = ROOT / "reports/stop8-measured-target-100-20261116"
CUTOFF = "2026-10-30"
STARTING_CASH = 100_000.0


def _load_account(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _bar_path(code: str, kind: str):
    group = "etfs" if kind == "etf" else "stocks"
    return SNAPSHOT / "data" / group / f"{code}.csv"


def _worker(task):
    code, kind, path = task
    # Late import ensures the runner uses the production helper from the
    # shared checkout, rather than duplicating its order logic here.
    from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades

    src = Path(path)
    baseline = _load_account(BASELINE / "accounts" / f"stock-{code}.json")
    frozen_csv = _bar_path(code, kind)
    file_stat = frozen_csv.stat()
    expected_revision = baseline.get("provenance", {}).get("data_revision")
    actual_revision = f"mtime_ns:{file_stat.st_mtime_ns},size:{file_stat.st_size}"
    if expected_revision != actual_revision:
        raise ValueError(f"{code}: frozen CSV revision mismatch saved={expected_revision} current={actual_revision}")
    bars = pd.read_csv(frozen_csv)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.loc[bars.date <= pd.Timestamp(CUTOFF)].copy()
    signals = [s for s in baseline["signals"] if s.get("signal_date", "") <= CUTOFF]
    # The saved signal is passed through byte-for-byte apart from serialization;
    # this run does no recognition and does not substitute another overlapping
    # pattern from the account ledger.
    for signal in signals:
        pattern = signal.get("pattern") or {}
        if not pattern.get("pattern_id") or pattern.get("pattern_id") not in signal.get("pattern_ids", []):
            raise ValueError(f"{code}: saved signal is missing its exact full pattern")
    result = simulate_fixed_trades(bars, signals, cash=STARTING_CASH)
    metrics = calculate_metrics(result["trades"], STARTING_CASH,
                                result["ending_equity"], result["daily_equity"])
    return {
        "code": code, "kind": kind, "status": "complete",
        "starting_cash": STARTING_CASH, "ending_equity": result["ending_equity"],
        "signals": signals, "trades": result["trades"],
        "daily_equity": result["daily_equity"], "rejections": result["rejections"],
        "order_events": result["order_events"],
        "accounting_checks": result["accounting_checks"],
        "metrics": metrics,
        "provenance": {"baseline_account": str(BASELINE / "accounts" / f"stock-{code}.json"),
                       "snapshot_csv": str(frozen_csv), "cutoff": CUTOFF,
                       "data_revision": actual_revision,
                       "recognition_reused": True,
                       "saved_signal_count": len(signals),
                       "saved_signal_pattern_ids": [s["pattern"]["pattern_id"] for s in signals]},
    }


def _aggregate(accounts):
    all_trades = [t for a in accounts for t in a["trades"]]
    pnls = [float(t["pnl"]) for t in all_trades]
    win_count = sum(p > 0 for p in pnls)
    losers = [p for p in pnls if p < 0]
    wins = [p for p in pnls if p > 0]
    equity = sum(float(a["ending_equity"]) for a in accounts)
    initial = sum(float(a["starting_cash"]) for a in accounts)
    reason_counts = {}
    rejection_counts = Counter()
    for account in accounts:
        for reason, count in account["metrics"].get("exit_reason_counts", {}).items():
            reason_counts[reason] = reason_counts.get(reason, 0) + count
        rejection_counts.update(str(x.get("reason", "unknown")) for x in account["rejections"])
    target_ratios = []
    target_rr = []
    target_status = {}
    for trade in all_trades:
        status = str(trade.get("measured_target_status", "none"))
        target_status[status] = target_status.get(status, 0) + 1
        target = trade.get("measured_target_price")
        entry = float(trade.get("entry_price", 0) or 0)
        if target is not None and entry > 0:
            ratio = float(target) / entry - 1
            target_ratios.append(ratio)
            target_rr.append(ratio / 0.08)
    account_returns = [float(a["ending_equity"]) / float(a["starting_cash"]) - 1 for a in accounts]
    account_drawdowns = [float(a["metrics"].get("maximum_drawdown") or 0) for a in accounts]
    date_index = sorted({p["date"] for a in accounts for p in a["daily_equity"]})
    account_points = []
    for a in accounts:
        points = sorted(a["daily_equity"], key=lambda p: p["date"])
        account_points.append((points, [p["date"] for p in points]))
    peak, max_dd = initial if accounts else 0.0, 0.0
    for date in date_index:
        combined = 0.0
        for a, (points, dates) in zip(accounts, account_points):
            i = bisect_right(dates, date) - 1
            combined += float(points[i]["equity"]) if i >= 0 else float(a["starting_cash"])
        peak = max(peak, combined)
        if peak > 0:
            max_dd = max(max_dd, (peak - combined) / peak)
    return {"account_count": len(accounts), "signal_count": sum(len(a["signals"]) for a in accounts),
            "trade_count": len(all_trades), "win_count": win_count,
            "loss_count": sum(p < 0 for p in pnls),
            "flat_count": sum(p == 0 for p in pnls),
            "win_rate": win_count / len(pnls) if pnls else None,
            "cumulative_return": equity / initial - 1 if initial else None,
            "ending_equity": equity, "starting_cash": initial,
            "profit_factor": sum(wins) / -sum(losers) if losers else None,
            "max_drawdown_combined_equity": max_dd,
            "account_return_median": sorted(account_returns)[len(account_returns)//2] if account_returns else None,
            "account_return_min": min(account_returns) if account_returns else None,
            "account_return_max": max(account_returns) if account_returns else None,
            "account_drawdown_median": sorted(account_drawdowns)[len(account_drawdowns)//2] if account_drawdowns else None,
            "measured_target_status_counts": dict(sorted(target_status.items())),
            "target_return_below_fixed20_fraction": sum(v < 0.20 for v in target_ratios) / len(target_ratios) if target_ratios else None,
            "target_reward_return_median": sorted(target_ratios)[len(target_ratios)//2] if target_ratios else None,
            "target_reward_risk_median": sorted(target_rr)[len(target_rr)//2] if target_rr else None,
            "exit_reason_counts": dict(sorted(reason_counts.items())),
            "rejection_count": sum(len(a["rejections"]) for a in accounts),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "target_source_available_count": sum(t.get("measured_target_status") == "active" for t in all_trades),
            "target_pending_prior_high_count": sum(t.get("measured_target_status") == "pending_prior_high" for t in all_trades),
            "target_missing_count": rejection_counts.get("measured_target_missing", 0),
            "target_rejected_not_above_entry_count": rejection_counts.get("measured_target_not_above_entry", 0),
            "accounting_all_pass": all(all(a["accounting_checks"].values()) for a in accounts)}


def _baseline_aggregate(files):
    accounts = [_load_account(p) for p in files]
    return _aggregate([{"signals": a["signals"], "trades": a["trades"],
                        "daily_equity": a["daily_equity"],
                        "starting_cash": a["starting_cash"],
                        "ending_equity": a["ending_equity"],
                        "rejections": a["rejections"],
                        "accounting_checks": a["accounting_checks"],
                        "metrics": a["metrics"]} for a in accounts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if not SNAPSHOT.exists() or not BASELINE.exists():
        raise SystemExit("Required frozen snapshot or saved G2 baseline is missing")
    files = sorted((BASELINE / "accounts").glob("stock-*.json"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 saved G2 accounts; found {len(files)}")
    args.output.mkdir(parents=True, exist_ok=True)
    accounts_dir = args.output / "accounts"
    accounts_dir.mkdir(exist_ok=True)
    tasks = []
    for p in files:
        old = _load_account(p)
        tasks.append((old["code"], old.get("kind", "stock"), str(p)))
    t0 = time.monotonic()
    completed = []
    errors = []
    if (args.output / "daily-research.json").exists() or any(accounts_dir.glob("stock-*.json")):
        raise SystemExit(f"Refusing to overwrite existing run artifacts: {args.output}")
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=mp.get_context("spawn")) as pool:
        future_by_code = {pool.submit(_worker, task): task[0] for task in tasks}
        for n, future in enumerate(as_completed(future_by_code), 1):
            code = future_by_code[future]
            try:
                account = future.result()
                target = accounts_dir / f"stock-{code}.json"
                target.write_text(json.dumps(account, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                completed.append(account)
            except Exception as exc:
                errors.append({"code": code, "error": f"{type(exc).__name__}: {exc}"})
            elapsed = time.monotonic() - t0
            print(f"progress {n}/100 complete={len(completed)} failed={len(errors)} elapsed={elapsed:.1f}s", flush=True)
            (args.output / "progress.json").write_text(json.dumps({
                "finished": n, "expected": 100, "completed": len(completed),
                "failed": len(errors), "elapsed_seconds": round(elapsed, 2),
                "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    completed.sort(key=lambda x: x["code"])
    from caisen.fixed_backtest import STRATEGY_VERSION
    summary = {"status": "complete" if len(completed) == 100 and not errors else "partial",
               "sample": "paired100", "run_id": args.output.name,
               "cutoff": CUTOFF, "snapshot_root": str(SNAPSHOT),
               "baseline_root": str(BASELINE), "target_method": "pattern_measured_move",
               "stop_fraction": 0.08, "starting_cash_per_account": STARTING_CASH,
               "recognition": "reused frozen saved G2 signals; no recognizer rerun",
               "recognition_engine_version": "caisen-joint-pattern-session-g2-prepressure-bridge-rank-v2-perf2-extrema1",
               "execution_engine": "Backtrader simulate_fixed_trades",
               "strategy_version": STRATEGY_VERSION,
               "target_fraction": None,
               "target_method": "signal_pattern_first_measured_target",
               "workers": args.workers, "elapsed_seconds": round(time.monotonic() - t0, 2),
               "completed_accounts": len(completed), "errors": errors,
               "frozen_revision_match_count": sum(bool(a["provenance"].get("data_revision")) for a in completed),
               "aggregate": _aggregate(completed),
               "fixed_8_20_baseline_aggregate": _baseline_aggregate(files),
               "baseline_fixed_stop_fraction": 0.08,
               "baseline_fixed_target_fraction": 0.20,
               "account_files": "accounts/stock-<code>.json"}
    (args.output / "daily-research.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "elapsed_seconds", "completed_accounts", "aggregate", "errors")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
