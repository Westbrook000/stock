#!/usr/bin/env python3
"""Cold, frozen-input A/B runner for structure-boundary daily research."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SYSTEM = REPO / "caisen-system"
DEFAULT_CODE_ROOT = SYSTEM
CODE_ROOT = Path(os.environ.get("CAISEN_CODE_ROOT", DEFAULT_CODE_ROOT)).resolve()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("A-748", "B-current", "case-A-748", "case-B-current"),
                        required=True)
    parser.add_argument("--code-root", type=Path, required=True,
                        help="caisen-system source root; use the extracted 748 source for A")
    parser.add_argument("--snapshot-root", type=Path, required=True,
                        help="frozen data root containing data/{stocks,etfs} and metadata")
    parser.add_argument("--universe", type=Path,
                        default=SYSTEM / "research" / "universes" / "daily_800_stock_200_etf_v1.json")
    parser.add_argument("--symbols", type=Path,
                        help="optional JSON object with a `symbols` list of {kind,code} rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-10-30")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8, 10), required=True)
    return parser.parse_args()


def _tree_rss_bytes(root_pid):
    import subprocess
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    except Exception:
        return 0
    parent, rss = {}, {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, resident_kb = map(int, fields)
        except ValueError:
            continue
        parent[pid], rss[pid] = ppid, resident_kb
    tree, changed = {int(root_pid)}, True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in tree) * 1024


def main():
    args = parse_args()
    code_root = args.code_root.resolve()
    snapshot_root = args.snapshot_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (code_root / "caisen" / "daily_research.py").is_file():
        raise FileNotFoundError(code_root / "caisen" / "daily_research.py")
    if not (args.universe.resolve()).is_file():
        raise FileNotFoundError(args.universe)
    if not (snapshot_root / "data" / "stocks").is_dir() or not (snapshot_root / "data" / "etfs").is_dir():
        raise FileNotFoundError("snapshot root must contain data/stocks and data/etfs")
    if (output_dir / "run-manifest.json").exists():
        raise FileExistsError(output_dir / "run-manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CAISEN_CODE_ROOT"] = str(code_root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from caisen.daily_research import (  # imported after choosing the source version
        BENCHMARK_PROCESS_TREE_RSS_CAP,
        _all_ledger_checks_pass,
        run_daily_research,
    )

    symbols = None
    if args.symbols:
        symbols_doc = json.loads(args.symbols.read_text(encoding="utf-8"))
        symbols = [(str(row["kind"]), str(row["code"])) for row in symbols_doc.get("symbols", [])]
        if not symbols or len(symbols) != len(set(symbols)):
            raise ValueError("symbols file must contain a nonempty list of unique {kind,code} rows")

    peak_holder = [0]
    from threading import Event, Thread
    stop = Event()

    def sample_tree():
        while not stop.is_set():
            peak_holder[0] = max(peak_holder[0], _tree_rss_bytes(os.getpid()))
            stop.wait(0.5)

    monitor = Thread(target=sample_tree, name="structure-study-rss", daemon=True)
    monitor.start()
    wall_started = time.monotonic()
    wall_start_epoch = time.time()
    try:
        report = run_daily_research(
            args.universe.resolve(), output_dir, date_to=args.cutoff, workers=args.workers,
            root=snapshot_root, symbols=symbols, recognizer="boundary_first",
            execution_mode="spawn", signal_cache_dir=None)
    finally:
        peak_holder[0] = max(peak_holder[0], _tree_rss_bytes(os.getpid()))
        stop.set()
        monitor.join()
    wall_seconds = time.monotonic() - wall_started
    peak = max(peak_holder[0], int(report.get("memory", {}).get("rss_peak_bytes", 0)))
    account_count = int(report.get("summary", {}).get("account_count", 0))
    checks_pass = _all_ledger_checks_pass(report, account_count)
    complete_accounts = int(report.get("summary", {}).get("successful_account_count", 0))
    failed_accounts = int(report.get("summary", {}).get("failed_account_count", 0))
    wrapper = {
        "label": args.label,
        "source_root": str(code_root),
        "snapshot_root": str(snapshot_root),
        "universe_path": str(args.universe.resolve()),
        "symbols_path": str(args.symbols.resolve()) if args.symbols else None,
        "cutoff": args.cutoff,
        "workers": args.workers,
        "cold_signal_cache": True,
        "started_at_epoch": wall_start_epoch,
        "measured_end_to_end_wall_seconds": wall_seconds,
        "process_tree_peak_bytes": peak,
        "process_tree_memory_cap_bytes": BENCHMARK_PROCESS_TREE_RSS_CAP,
        "process_tree_memory_pass": 0 < peak <= BENCHMARK_PROCESS_TREE_RSS_CAP,
        "account_count": account_count,
        "successful_account_count": complete_accounts,
        "failed_account_count": failed_accounts,
        "accounting_checks_pass": checks_pass,
        "full_run_status": report.get("status"),
        "daily_research_report": str((output_dir / "daily-research.json").resolve()),
        "timings": report.get("timings"),
        "summary": report.get("summary"),
    }
    (output_dir / "structure-boundary-run.json").write_text(
        json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(wrapper, ensure_ascii=False), flush=True)
    return 0 if report.get("status") == "complete" and checks_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
