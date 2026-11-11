"""Research-only arbitration of same-session competing v3 neckline signals."""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
CAISEN_ROOT = HERE.parents[2]
WORKSPACE = CAISEN_ROOT.parent
RESULTS = HERE / "results/paired-100-v3-rebound-linked-20261110-attempt01"
OUT = RESULTS / "signal-arbitration-v3.json"
sys.path.insert(0, str(CAISEN_ROOT))
from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades  # noqa: E402

VARIANTS = ("margin_high", "margin_low", "main_bottoms_first")


def _structure_start_dates(account):
    result = {}
    for event in account.get("instance_events", []):
        pattern = event.get("pattern") or {}
        key = event.get("structure_key") or pattern.get("structure_key")
        formed = pattern.get("formed_at")
        if key and formed:
            old = result.get(key)
            if old is None or formed < old:
                result[key] = str(formed)
    return result


def _signal_score(signal, variant, starts):
    tag = (signal.get("pattern_tags") or [{}])[0]
    pattern_type = (signal.get("pattern_types") or [""])[0]
    margin = (float(signal["breakout_close"]) - float(signal["required_close"])) / max(
        abs(float(signal["required_close"])), 1e-12)
    if variant == "margin_high":
        return (margin,)
    if variant == "margin_low":
        return (-margin,)
    if variant == "main_bottoms_first":
        is_main_bottom = pattern_type in {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
        formed = starts.get(tag.get("structure_key"))
        duration = ((pd.Timestamp(signal["signal_date"]) - pd.Timestamp(formed)).days
                    if formed else -1)
        return (int(is_main_bottom), duration, margin)
    raise ValueError(variant)


def _metrics_for_run(account_results):
    trades = [trade for item in account_results for trade in item["trades"]]
    pnl = sum(float(trade["pnl"]) for trade in trades)
    wins = sum(float(trade["pnl"]) > 0 for trade in trades)
    starting = len(account_results) * 100_000.0
    account_return = [item["ending_equity"] / 100_000.0 - 1 for item in account_results]
    bands = {"through_2020": [], "2021_2023": [], "2024_plus": []}
    for trade in trades:
        year = int(str(trade["signal_date"])[:4])
        band = "through_2020" if year <= 2020 else "2021_2023" if year <= 2023 else "2024_plus"
        bands[band].append(trade)
    return {
        "account_count": len(account_results), "trade_count": len(trades),
        "win_rate": wins / len(trades) if trades else None,
        "net_pnl": pnl, "cumulative_return": pnl / starting if starting else None,
        "profit_factor": sum(float(t["pnl"]) for t in trades if t["pnl"] > 0) /
                         -sum(float(t["pnl"]) for t in trades if t["pnl"] < 0)
                         if any(t["pnl"] < 0 for t in trades) else None,
        "accounting_all_pass": all(all(x["accounting_checks"].values()) for x in account_results),
        "positive_accounts": sum(value > 0 for value in account_return),
        "median_account_return": sorted(account_return)[len(account_return) // 2],
        "signal_date_bands": {
            name: {"trades": len(rows), "wins": sum(t["pnl"] > 0 for t in rows),
                   "win_rate": sum(t["pnl"] > 0 for t in rows) / len(rows) if rows else None,
                   "net_pnl": sum(float(t["pnl"]) for t in rows)}
            for name, rows in bands.items()},
    }


def main():
    files = sorted(RESULTS.glob("stock-*.json.gz"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 v3 account files, got {len(files)}")
    outputs = {name: [] for name in VARIANTS}
    baseline_replay_count = 0
    collision_days = 0
    collision_rows = 0
    largest_collision = 0
    started = time.time()
    for index, path in enumerate(files, 1):
        account = json.load(gzip.open(path, "rt", encoding="utf-8"))
        code = account["code"]
        frame = pd.read_csv(WORKSPACE / "data/stocks" / f"{code}.csv")
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame.date <= "2026-10-30"].reset_index(drop=True)
        signals = account["signals_detail"]
        original = sorted(signals, key=lambda item: (item["signal_date"], item["boundary_version_id"]))
        collision_counts = defaultdict(int)
        for signal in original:
            collision_counts[signal["signal_date"]] += 1
        sizes = [count for count in collision_counts.values() if count > 1]
        collision_days += len(sizes)
        collision_rows += sum(sizes)
        largest_collision = max([largest_collision, *sizes]) if sizes else largest_collision

        baseline = simulate_fixed_trades(frame, original, cash=100_000.0)
        archived = account["trades"]
        if len(baseline["trades"]) != len(archived):
            raise AssertionError(f"{code}: v3 baseline trade count differs from saved replay")
        for actual, saved in zip(baseline["trades"], archived):
            if (actual["entry_date"], actual["exit_date"], actual["exit_reason"], actual["shares"]) != (
                    saved["entry_date"], saved["exit_date"], saved["exit_reason"], saved["shares"]):
                raise AssertionError(f"{code}: v3 baseline replay diverges")
            if abs(float(actual["pnl"]) - float(saved["pnl"])) > 1e-7:
                raise AssertionError(f"{code}: v3 baseline PnL diverges")
        baseline_replay_count += 1
        starts = _structure_start_dates(account)
        for variant in VARIANTS:
            ordered = sorted(signals, key=lambda row: (
                row["signal_date"], tuple(-value for value in _signal_score(row, variant, starts)),
                row["boundary_version_id"]))
            result = simulate_fixed_trades(frame, ordered, cash=100_000.0)
            if not all(result["accounting_checks"].values()):
                raise AssertionError(f"{code}/{variant}: accounting failed")
            metrics = calculate_metrics(result["trades"], 100_000.0,
                                        result["ending_equity"], result["daily_equity"])
            outputs[variant].append({"code": code, "trades": result["trades"],
                                     "ending_equity": result["ending_equity"],
                                     "accounting_checks": result["accounting_checks"],
                                     "metrics": metrics})
        if index % 10 == 0:
            print(f"{index}/100 accounts elapsed={time.time()-started:.1f}s", flush=True)

    summary = {name: _metrics_for_run(rows) for name, rows in outputs.items()}
    result = {
        "experiment": "same_day_signal_arbitration_v1",
        "source_prototype": "dynamic_neckline_v3_rebound_linked_saved_v12_structures_only",
        "cutoff": "2026-10-30", "accounts": 100,
        "baseline_v3": {"trades": 3147, "win_rate": 0.3371464887194153,
                        "cumulative_return": 0.5480993920000016,
                        "profit_factor": 1.2529366620495805},
        "v8_reference": {"trades": 3780, "win_rate": 0.33756613756613757,
                         "cumulative_return": 0.6134487840000002,
                         "profit_factor": 1.2356292564034943},
        "same_day_conflict_diagnostics": {"collision_symbol_dates": collision_days,
                                          "signal_rows_on_collision_dates": collision_rows,
                                          "max_signals_for_one_symbol_date": largest_collision},
        "baseline_exact_replay_accounts": baseline_replay_count,
        "variants": summary, "account_details": outputs,
        "elapsed_seconds": time.time()-started,
        "interpretation": "Historical exploratory arbitration; all orders retain original signals and 8%/20% Backtrader rules. Not an independent holdout.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "elapsed_seconds": result["elapsed_seconds"],
                      "collision_diagnostics": result["same_day_conflict_diagnostics"],
                      "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
