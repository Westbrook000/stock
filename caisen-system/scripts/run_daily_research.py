#!/usr/bin/env python3
"""Freeze the daily research universe, benchmark workers, or run all 1,000 symbols."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import (  # noqa: E402
    UNIVERSE_ID,
    BENCHMARK_ENGINE_REPETITIONS,
    benchmark_current_v8_pool,
    create_universe,
    run_daily_research,
)
from caisen.market import discover_instruments  # noqa: E402


UNIVERSE_PATH = REPO / "research" / "universes" / (UNIVERSE_ID + ".json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-universe", action="store_true", help="create the frozen 800 stock / 200 ETF universe")
    parser.add_argument("--benchmark", action="store_true",
                        help="validate 36 frozen v8 ledgers, then benchmark 1, 2, 4, 6, 8 and 10 spawn workers")
    parser.add_argument("--repetitions", type=int, default=BENCHMARK_ENGINE_REPETITIONS)
    parser.add_argument("--cohort", type=Path, help="frozen 24-member benchmark cohort")
    parser.add_argument("--regression-cases", type=Path, help="three additional pinned regression inputs")
    parser.add_argument("--archive-reference", type=Path,
                        help="complete v8/2ATR 1000-account archive used for 33 parity ledgers")
    parser.add_argument("--snapshot-reference", type=Path,
                        help="frozen v8/2ATR 3-account reference for 600108/159718/516630")
    parser.add_argument("--universe", type=Path, default=UNIVERSE_PATH)
    parser.add_argument("--output", type=Path, default=REPO / "reports" / "daily-research")
    parser.add_argument("--date-to", default=None,
                        help="cutoff date; benchmark defaults to its frozen archive cutoff 2026-10-30")
    parser.add_argument("--trade-start")
    parser.add_argument("--recognizer", choices=("v7", "boundary_first"),
                        help="explicit recognizer; defaults to the validated benchmark recommendation")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8, 10), default=None,
                        help="worker count; omit to consume an exact-parity, memory-capped benchmark recommendation")
    parser.add_argument("--signal-cache-dir", type=Path,
                        help="optional directory for versioned, cutoff- and data-revision-keyed recognition results")
    parser.add_argument("--require-signal-cache", action="store_true",
                        help="fail on any missing/invalid cache entry without running recognition")
    args = parser.parse_args()
    if args.freeze_universe:
        result = create_universe(discover_instruments(REPO.parent), args.universe)
        print(json.dumps({"path": str(args.universe.resolve()), "counts": result["counts"],
                          "inventory_counts": result["inventory_counts"]}, ensure_ascii=False))
        return
    if not args.universe.exists():
        raise SystemExit("universe file is missing; run with --freeze-universe first")
    if args.benchmark:
        def benchmark_progress(snapshot):
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        result = benchmark_current_v8_pool(args.universe, args.output, root=REPO.parent,
                                           date_to=args.date_to or "2026-10-30", trade_start=args.trade_start,
                                           cohort_path=args.cohort, cases_path=args.regression_cases,
                                           repetitions=args.repetitions, progress_callback=benchmark_progress,
                                           archive_reference_path=args.archive_reference,
                                           snapshot_reference_path=args.snapshot_reference)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") != "complete":
            raise SystemExit(1)
        return

    last_print = [0.0]
    def progress(snapshot):
        now = time.monotonic()
        if now - last_print[0] >= 2.0:
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            last_print[0] = now

    payload = run_daily_research(args.universe, args.output, date_to=args.date_to,
                                 trade_start=args.trade_start, workers=args.workers,
                                 progress_callback=progress, root=REPO.parent,
                                 signal_cache_dir=args.signal_cache_dir,
                                 recognizer=args.recognizer,
                                 require_signal_cache=args.require_signal_cache)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"],
                      "artifacts": payload["artifacts"], "timings": payload["timings"],
                      "effective_configuration": {
                          "universe": payload.get("universe"),
                          "parameters": payload.get("parameters"),
                          "strategy_version": payload.get("strategy_version"),
                          "recognizer_version": payload.get("recognizer_version"),
                          "engine_version": payload.get("engine_version"),
                          "source_rule_version": payload.get("source_rule_version"),
                          "recognition_rules": payload.get("recognition_rules"),
                          "trading_rules": payload.get("trading_rules"),
                      }},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
