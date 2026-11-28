#!/usr/bin/env python3
"""Audit alignment between full-history backtest recognition and 5-year scan window."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

for _env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from caisen.api import (  # noqa: E402
    _decision_patterns_by_timeframe,
    _recognition_window,
    _same_day_signal_ids,
    _supported_main_patterns,
)
from caisen.daily_incremental import recognize_daily_history  # noqa: E402
from caisen.daily_research import load_research_bars  # noqa: E402
from caisen.decision import synthesize_decision  # noqa: E402
from caisen.market import discover_instruments, load_daily_bars  # noqa: E402
from caisen.patterns import current_pattern_snapshot  # noqa: E402


def _load_36_symbols():
    bench_path = REPO / "reports" / "daily-research" / "benchmark" / "benchmark-execution-v6-current-rules-36.json"
    if bench_path.exists():
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        inputs = data.get("inputs", [])
        if len(inputs) == 36:
            return [(item["kind"], item["code"]) for item in inputs]
    cohort_path = REPO / "reports" / "daily-research" / "benchmark" / "cohort-v3.json"
    reg_path = REPO / "reports" / "daily-research" / "benchmark" / "regression-cases-v3.json"
    symbols = []
    for path in (cohort_path, reg_path):
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            for item in doc.get("symbols", []):
                symbols.append((item["kind"], item["code"]))
    return symbols[:36]


def _extract_latest_state(recognition, bars, code, kind):
    patterns = current_pattern_snapshot(_supported_main_patterns(recognition["patterns"]))
    decision_patterns = _decision_patterns_by_timeframe({"1d": patterns})
    latest_date = str(pd.Timestamp(bars.date.iloc[-1]).date()) if not bars.empty else None
    allowed = {"1d": _same_day_signal_ids(recognition, latest_date)}
    price = float(bars.close.iloc[-1])
    quality = "usable_with_warnings"
    decision = synthesize_decision(
        code,
        kind,
        decision_patterns,
        {"1d": price},
        {"1d": quality},
        bars_by_timeframe={"1d": bars},
        allowed_pattern_ids_by_timeframe=allowed,
        watch_by_timeframe={"1d": recognition.get("latest_bar")},
        watch_patterns_by_timeframe={"1d": patterns},
    )
    selected_plan = decision.get("selected_plan") or {}
    return {
        "latest_date": latest_date,
        "action": decision.get("action"),
        "primary_pattern_id": selected_plan.get("pattern_id"),
        "structure_level": selected_plan.get("structure_level"),
        "measured_target_price": selected_plan.get("measured_target_price"),
        "watch_count": len(decision.get("watch") or []),
        "recent_signals_250": [
            {
                "signal_date": s.get("signal_date"),
                "pattern_types": s.get("pattern_types"),
            }
            for s in recognition.get("signals", [])
            if s.get("signal_date", "") >= str((pd.Timestamp(latest_date) - pd.DateOffset(days=365)).date())
        ],
    }


def _audit_one(args):
    kind, code = args
    found = {(item.kind, item.code): item for item in discover_instruments(ROOT)}
    instrument = found.get((kind, code))
    if instrument is None:
        return None
    scan_loaded = load_daily_bars(instrument, root=ROOT)
    research_loaded = load_research_bars(instrument, root=ROOT)
    scan_bars = scan_loaded.bars
    res_bars = research_loaded["bars"]
    row_mismatch = None
    if len(scan_bars) != len(res_bars):
        row_mismatch = {
            "kind": kind,
            "code": code,
            "scan_rows": len(scan_bars),
            "research_rows": len(res_bars),
        }

    tick = float(scan_bars.attrs.get("quality", {}).get("tick_size", 0.001 if kind == "etf" else 0.01))
    full_rec = recognize_daily_history(res_bars, code, kind, tick_size=tick, recognizer="boundary_first")
    win_bars = _recognition_window(scan_bars, "1d")
    win_rec = recognize_daily_history(win_bars, code, kind, tick_size=tick, recognizer="boundary_first")

    full_state = _extract_latest_state(full_rec, res_bars, code, kind)
    win_state = _extract_latest_state(win_rec, win_bars, code, kind)
    entry = {
        "kind": kind,
        "code": code,
        "full_bars": len(res_bars),
        "window_bars": len(win_bars),
        "full_state": full_state,
        "window_state": win_state,
        "latest_action_match": full_state["action"] == win_state["action"],
        "recent_signals_match": full_state["recent_signals_250"] == win_state["recent_signals_250"],
    }
    return entry, row_mismatch


def main():
    symbols = _load_36_symbols()
    results = []
    latest_action_mismatches = []
    recent_signal_mismatches = []
    bar_row_mismatches = []

    with ProcessPoolExecutor(max_workers=min(10, len(symbols)), mp_context=get_context("spawn")) as pool:
        for outcome in pool.map(_audit_one, symbols):
            if outcome is None:
                continue
            entry, row_mismatch = outcome
            if row_mismatch is not None:
                bar_row_mismatches.append(row_mismatch)
            results.append(entry)
            if not entry["latest_action_match"]:
                latest_action_mismatches.append(entry)
            if not entry["recent_signals_match"]:
                recent_signal_mismatches.append(entry)

    report = {
        "checked_symbols": len(results),
        "bar_row_mismatches": len(bar_row_mismatches),
        "latest_action_match_count": sum(1 for r in results if r["latest_action_match"]),
        "latest_action_mismatch_count": len(latest_action_mismatches),
        "recent_signals_match_count": sum(1 for r in results if r["recent_signals_match"]),
        "recent_signals_mismatch_count": len(recent_signal_mismatches),
        "bar_row_mismatch_details": bar_row_mismatches,
        "latest_action_mismatches": latest_action_mismatches,
        "recent_signal_mismatches": recent_signal_mismatches,
    }
    out_path = REPO / "reports" / "daily-research" / "alignment-audit-20261127.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("mismatches") and not k.endswith("details")},
                     ensure_ascii=False, indent=2))
    print("Report written to", out_path)


if __name__ == "__main__":
    main()
