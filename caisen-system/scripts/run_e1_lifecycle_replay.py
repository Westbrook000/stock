#!/usr/bin/env python3
"""Run the frozen E1 recognizer on the paired 100 or full 800/200 cohort.

The runner intentionally uses a cold signal cache and a single explicit G1
configuration. It records code/rule hashes and market-file stat revisions;
market CSV contents are never hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

SYSTEM = Path(__file__).resolve().parents[1]
REPO = SYSTEM.parent
if str(SYSTEM) not in sys.path:
    sys.path.insert(0, str(SYSTEM))

SOURCE_FILES = (
    "caisen/daily_incremental.py",
    "caisen/structure_boundary.py",
    "caisen/main_bottoms.py",
    "caisen/research_config.py",
    "caisen/fixed_backtest.py",
    "caisen/daily_research.py",
    "caisen/market.py",
    "caisen/api.py",
    "caisen/store.py",
    "caisen/rules/patterns.v2.json",
    "scripts/run_e1_lifecycle_replay.py",
)


def _source_hashes():
    result = {}
    for relative in SOURCE_FILES:
        path = SYSTEM / relative
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _tree_rss_bytes(root_pid):
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


def _symbols_for_sample(sample, paired_path, universe_path):
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    if sample == "all1000":
        return None, universe
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    symbols = [("stock", str(code)) for code in paired.get("stocks", [])]
    symbols += [("etf", str(code)) for code in paired.get("etfs", [])]
    expected = {(kind, str(code)) for kind, rows in (("stock", universe.get("stocks", [])),
                                                     ("etf", universe.get("etfs", [])))
                for code in rows}
    missing_from_full_cohort = [f"{kind}-{code}" for kind, code in symbols
                                if (kind, code) not in expected]
    if missing_from_full_cohort:
        raise ValueError("paired members are outside the frozen 1000 cohort: "
                         + ", ".join(missing_from_full_cohort))
    return symbols, universe


def _input_revisions(snapshot_root, selected):
    rows = []
    for kind, code in selected:
        folder = "stocks" if kind == "stock" else "etfs"
        path = snapshot_root / "data" / folder / f"{code}.csv"
        stat = path.stat()
        rows.append({"kind": kind, "code": code,
                     "relative_path": str(path.relative_to(snapshot_root)),
                     "mtime_ns": int(stat.st_mtime_ns), "size_bytes": int(stat.st_size)})
    return rows


def _validate_selected_tick_sizes(selected, tick_sizes):
    """Validate the exchange tick for each asset kind present in the cohort."""
    expected_by_kind = {"stock": 0.01, "etf": 0.001}
    selected_kinds = {kind for kind, _code in selected}
    unexpected_ticks = {
        kind: sorted(tick_sizes.get(kind, set()))
        for kind in sorted(selected_kinds)
        if kind not in expected_by_kind
        or tick_sizes.get(kind, set()) != {expected_by_kind[kind]}
    }
    if unexpected_ticks:
        raise ValueError("frozen registry tick values differ from required stock .01 / ETF .001: "
                         + json.dumps(unexpected_ticks))


def _verify_snapshot_manifest(snapshot_root, selected):
    manifest_path = snapshot_root / "freeze-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("snapshot root lacks freeze-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {(item["kind"], str(item["code"])): item.get("frozen_revision")
                for item in manifest.get("symbols", [])}
    missing, changed = [], []
    for kind, code in selected:
        key = (kind, str(code))
        if key not in expected:
            missing.append(f"{kind}-{code}")
            continue
        path = snapshot_root / "data" / ("stocks" if kind == "stock" else "etfs") / f"{code}.csv"
        actual = path.stat()
        revision = {"mtime_ns": int(actual.st_mtime_ns), "size_bytes": int(actual.st_size)}
        if revision != expected[key]:
            changed.append({"kind": kind, "code": code, "expected": expected[key], "actual": revision})
    resource_changed = []
    for item in manifest.get("resources", []):
        path = snapshot_root / item["snapshot_path"]
        actual = path.stat()
        revision = {"mtime_ns": int(actual.st_mtime_ns), "size_bytes": int(actual.st_size)}
        if revision != item.get("frozen_revision"):
            resource_changed.append({"path": item["snapshot_path"],
                                     "expected": item.get("frozen_revision"), "actual": revision})
    if missing or changed or resource_changed:
        raise RuntimeError("frozen input validation failed: " + json.dumps({
            "missing": missing, "changed": changed, "resource_changed": resource_changed}))
    return {"manifest_path": str(manifest_path), "universe_id": manifest.get("universe_id"),
            "source_revision_policy": manifest.get("revision_policy"),
            "selected_inputs_verified": len(selected),
            "metadata_resources_verified": len(manifest.get("resources", []))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("paired100", "all1000"), required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True,
                        help="Frozen root containing data/stocks, data/etfs, and metadata")
    parser.add_argument("--universe", type=Path,
                        default=SYSTEM / "research/universes/daily_800_stock_200_etf_v1.json")
    parser.add_argument("--paired-members", type=Path,
                        default=SYSTEM / "research/universes/paired_100_v8_v12_v16_stock_cohort_v1.json")
    parser.add_argument("--cutoff", default="2026-10-30")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    snapshot_root = args.snapshot_root.resolve()
    universe_path = args.universe.resolve()
    paired_path = args.paired_members.resolve()
    output_dir = args.output_dir.resolve()
    if not (snapshot_root / "data/stocks").is_dir() or not (snapshot_root / "data/etfs").is_dir():
        raise FileNotFoundError("snapshot root must contain data/stocks and data/etfs")
    instruments_path = snapshot_root / "caisen-system/config/instruments.json"
    if not instruments_path.is_file():
        raise FileNotFoundError("snapshot root lacks frozen instrument registry: " + str(instruments_path))
    os.environ["CAISEN_INSTRUMENTS"] = str(instruments_path)
    if (output_dir / "run-manifest.json").exists():
        raise FileExistsError(output_dir / "run-manifest.json")

    symbols, universe = _symbols_for_sample(args.sample, paired_path, universe_path)
    selected = ([ ("stock", str(code)) for code in universe.get("stocks", [])]
                + [("etf", str(code)) for code in universe.get("etfs", [])]
                if symbols is None else symbols)
    missing_files = []
    for kind, code in selected:
        folder = "stocks" if kind == "stock" else "etfs"
        path = snapshot_root / "data" / folder / f"{code}.csv"
        if not path.is_file():
            missing_files.append(f"{kind}-{code}")
            continue
    if missing_files:
        raise FileNotFoundError("missing frozen accounts: " + ", ".join(missing_files))
    freeze_validation = _verify_snapshot_manifest(snapshot_root, selected)
    revisions_before = _input_revisions(snapshot_root, selected)

    from caisen.market import InstrumentFile, TICK_RULE_VERSION, resolve_instrument_tick
    tick_rows = []
    tick_sizes = {"stock": set(), "etf": set()}
    for kind, code in selected:
        folder = "stocks" if kind == "stock" else "etfs"
        path = snapshot_root / "data" / folder / f"{code}.csv"
        tick_size, tick_source = resolve_instrument_tick(
            snapshot_root, InstrumentFile(code=code, kind=kind, path=str(path)))
        tick_sizes[kind].add(float(tick_size))
        tick_rows.append({"kind": kind, "code": code, "tick_size": float(tick_size),
                          "tick_size_source": tick_source})
    _validate_selected_tick_sizes(selected, tick_sizes)

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CAISEN_CODE_ROOT"] = str(SYSTEM)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    from caisen.daily_research import (BENCHMARK_PROCESS_TREE_RSS_CAP,
                                       _all_ledger_checks_pass, run_daily_research)
    from caisen.research_config import RecognitionExperimentConfig
    config = RecognitionExperimentConfig(
        experiment_id="internal_rebound_coverage_g1_v1",
        require_internal_rebound_coverage=True,
        require_minimum_anchor_span=False,
    )
    before_hashes = _source_hashes()
    peak_holder = [0]
    stop = Event()

    def sample_tree():
        while not stop.is_set():
            peak_holder[0] = max(peak_holder[0], _tree_rss_bytes(os.getpid()))
            stop.wait(0.5)

    monitor = Thread(target=sample_tree, name="e1-run-rss", daemon=True)
    monitor.start()
    started_at = time.time()
    wall_started = time.monotonic()
    report = None
    try:
        report = run_daily_research(
            universe_path, output_dir, date_to=args.cutoff, workers=args.workers,
            root=snapshot_root, symbols=symbols, recognizer="boundary_first",
            execution_mode="spawn", signal_cache_dir=None, recognition_config=config)
    finally:
        peak_holder[0] = max(peak_holder[0], _tree_rss_bytes(os.getpid()))
        stop.set()
        monitor.join()
    wall_seconds = time.monotonic() - wall_started
    after_hashes = _source_hashes()
    revisions_after = _input_revisions(snapshot_root, selected)
    account_count = int(report.get("summary", {}).get("account_count", 0))
    checks_pass = _all_ledger_checks_pass(report, account_count)
    result = {
        "status": report.get("status"),
        "sample": args.sample,
        "cohort_id": universe.get("id"),
        "cohort_counts": {"selected": len(selected), "stock": sum(k == "stock" for k, _ in selected),
                           "etf": sum(k == "etf" for k, _ in selected)},
        "paired_members_path": str(paired_path) if args.sample == "paired100" else None,
        "paired_member_validation": "all members present in fixed full cohort and frozen snapshot",
        "cutoff": args.cutoff,
        "snapshot_root": str(snapshot_root),
        "freeze_validation": freeze_validation,
        "input_file_revisions_before": revisions_before,
        "input_file_revisions_after": revisions_after,
        "input_revisions_unchanged": revisions_before == revisions_after,
        "recognition_experiment": {
            "experiment_id": config.experiment_id,
            "require_internal_rebound_coverage": True,
            "require_minimum_anchor_span": False,
            "cache_key": list(config.cache_key()),
        },
        "recognizer": "boundary_first",
        "engine_version": report.get("engine_version"),
        "source_rule_version": report.get("source_rule_version"),
        "tick_policy": "snapshot instrument registry; expected stock .01 and ETF .001",
        "tick_rule_version": TICK_RULE_VERSION,
        "resolved_ticks": tick_rows,
        "strategy": {"stop_fraction": 0.08, "target_fraction": 0.20,
                     "entry": "next-session open", "initial_cash": 100000,
                     "lot_size": 100},
        "cold_signal_cache": True,
        "workers": args.workers,
        "started_at_epoch": started_at,
        "wall_seconds": wall_seconds,
        "process_tree_peak_rss_bytes": peak_holder[0],
        "process_tree_memory_cap_bytes": BENCHMARK_PROCESS_TREE_RSS_CAP,
        "process_tree_memory_pass": 0 < peak_holder[0] <= BENCHMARK_PROCESS_TREE_RSS_CAP,
        "source_rule_sha256_before": before_hashes,
        "source_rule_sha256_after": after_hashes,
        "source_unchanged_during_run": before_hashes == after_hashes,
        "account_count": account_count,
        "successful_account_count": int(report.get("summary", {}).get("successful_account_count", 0)),
        "failed_account_count": int(report.get("summary", {}).get("failed_account_count", 0)),
        "accounting_checks_pass": checks_pass,
        "timings": report.get("timings"),
        "memory": report.get("memory"),
        "summary": report.get("summary"),
        "daily_research_json": str((output_dir / "daily-research.json").resolve()),
        "daily_research_html": str((output_dir / "daily-research.html").resolve()),
    }
    (output_dir / "e1-run-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0 if (report.get("status") == "complete" and checks_pass
                 and result["failed_account_count"] == 0 and result["source_unchanged_during_run"]
                 and result["input_revisions_unchanged"] and result["process_tree_memory_pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
