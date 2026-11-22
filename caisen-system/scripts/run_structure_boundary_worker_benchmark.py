#!/usr/bin/env python3
"""Run the frozen 36-symbol spawn-worker matrix for the structure/boundary study."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SYSTEM = REPO / "caisen-system"
if str(SYSTEM) not in sys.path:
    sys.path.insert(0, str(SYSTEM))

from caisen.daily_research import (  # noqa: E402
    BENCHMARK_PROCESS_TREE_RSS_CAP,
    _account_ledger_signature,
    _all_ledger_checks_pass,
    _process_tree_rss_bytes,
    _select_safe_worker_count,
    _version_metadata,
    run_daily_research,
)
from scripts.run_structure_boundary_study import verify_frozen_snapshot  # noqa: E402


STUDY = SYSTEM / "reports" / "daily-research" / "structure-boundary-study-20261112"
SNAPSHOT = Path("/tmp/caisen-structure-boundary-snapshot-20261112-v2")
WORKER_ROOT = Path("/tmp/caisen-structure-boundary-worker-inputs-20261112-v2")
COHORT = STUDY / "worker-benchmark-symbols.json"
UNIVERSE = SYSTEM / "research" / "universes" / "daily_800_stock_200_etf_v1.json"
OUTPUT = STUDY / "worker-benchmark-v2"
COUNTS = (1, 2, 4, 6, 8, 10)
REPETITIONS = 2
CUTOFF = "2026-10-30"


def revision(path: Path):
    stat = path.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size_bytes": int(stat.st_size)}


def prepare_worker_root():
    """Use snapshot symlinks for frozen 800/200 files; freeze only the three external cases."""
    base_check = verify_frozen_snapshot(SNAPSHOT)
    if base_check["status"] != "valid":
        raise RuntimeError("fixed 800/200 snapshot changed: " + json.dumps(base_check))
    rows = json.loads(COHORT.read_text(encoding="utf-8"))["symbols"]
    if WORKER_ROOT.exists():
        manifest_path = WORKER_ROOT / "worker-input-manifest.json"
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError("worker input root exists without a manifest: %s" % WORKER_ROOT)

    WORKER_ROOT.mkdir(parents=True)
    data_root = WORKER_ROOT / "data"
    data_root.mkdir()
    manifest_rows = []
    for kind in ("stock", "etf"):
        folder = "stocks" if kind == "stock" else "etfs"
        destination = data_root / folder
        destination.mkdir()
        frozen_dir = SNAPSHOT / "data" / folder
        for source in frozen_dir.glob("*.csv"):
            (destination / source.name).symlink_to(source)

    extra_keys = {("stock", "600108"), ("etf", "159718"), ("etf", "516630")}
    for kind, code in sorted(extra_keys):
        folder = "stocks" if kind == "stock" else "etfs"
        frozen_target = data_root / folder / (code + ".csv")
        if frozen_target.exists():
            continue
        source = REPO / "data" / folder / (code + ".csv")
        if not source.is_file():
            raise FileNotFoundError(source)
        frozen = revision(source)
        target = frozen_target
        shutil.copy2(source, target)
        if revision(source) != frozen or revision(target) != frozen:
            raise RuntimeError("case input changed while freezing: %s:%s" % (kind, code))
        manifest_rows.append({"kind": kind, "code": code, "source": str(source),
                              "source_revision": frozen, "frozen_revision": revision(target)})

    config = WORKER_ROOT / "caisen-system" / "config"
    config.mkdir(parents=True)
    (config / "instruments.json").symlink_to(
        SNAPSHOT / "caisen-system" / "config" / "instruments.json")
    basis_dir = data_root / "market_data_current"
    basis_dir.mkdir()
    (basis_dir / ".market-data-basis.json").symlink_to(
        SNAPSHOT / "data" / "market_data_current" / ".market-data-basis.json")
    doc = {"schema_version": "worker-benchmark-inputs-v1", "snapshot": str(SNAPSHOT),
           "snapshot_validation": base_check, "extra_case_inputs": manifest_rows,
           "revision_policy": "mtime_ns_and_size_no_content_hash"}
    (WORKER_ROOT / "worker-input-manifest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_manifest = prepare_worker_root()
    symbols = [(str(row["kind"]), str(row["code"]))
               for row in json.loads(COHORT.read_text(encoding="utf-8"))["symbols"]]
    if len(symbols) != 36 or len(set(symbols)) != 36:
        raise ValueError("frozen worker cohort must contain exactly 36 unique symbols")
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    os.environ.update(env)

    report = {"schema_version": "structure-boundary-worker-matrix-v1", "status": "running",
              "date_to": CUTOFF, "worker_counts": list(COUNTS), "repetitions": REPETITIONS,
              "cohort_path": str(COHORT), "cohort_count": len(symbols),
              "input_manifest": input_manifest, "versions": _version_metadata(),
              "process_tree_memory_cap_bytes": BENCHMARK_PROCESS_TREE_RSS_CAP,
              "runs": [], "started_at_epoch": time.time()}
    progress_path = OUTPUT / "progress.json"
    result_path = OUTPUT / "worker-benchmark.json"
    reference_signature = None
    ordered_runs = [(workers, repetition) for repetition, sequence in enumerate(
        (COUNTS, tuple(reversed(COUNTS))), start=1) for workers in sequence]

    for index, (workers, repetition) in enumerate(ordered_runs, start=1):
        run_dir = OUTPUT / ("workers-%02d-repeat-%02d" % (workers, repetition))
        if (run_dir / "daily-research.json").is_file():
            run = json.loads((run_dir / "daily-research.json").read_text(encoding="utf-8"))
            peak = int(run.get("memory", {}).get("rss_peak_bytes", 0))
            # Existing runs are reused only if their complete manifest matches this matrix.
            if (run.get("status") != "complete" or run.get("parameters", {}).get("workers") != workers
                    or run.get("parameters", {}).get("date_to") != CUTOFF
                    or run.get("summary", {}).get("account_count") != len(symbols)):
                raise RuntimeError("incomplete/mismatched run exists: %s" % run_dir)
        else:
            peak_holder = [0]
            from threading import Event, Thread
            stop = Event()

            def sample_tree():
                while not stop.is_set():
                    peak_holder[0] = max(peak_holder[0], _process_tree_rss_bytes(os.getpid()))
                    stop.wait(0.5)

            monitor = Thread(target=sample_tree, daemon=True)
            monitor.start()
            start = time.monotonic()
            try:
                run = run_daily_research(
                    UNIVERSE, run_dir, date_to=CUTOFF, workers=workers,
                    root=WORKER_ROOT, symbols=symbols, recognizer="boundary_first",
                    execution_mode="spawn", signal_cache_dir=None)
            finally:
                peak_holder[0] = max(peak_holder[0], _process_tree_rss_bytes(os.getpid()))
                stop.set()
                monitor.join()
            peak = max(peak_holder[0], int(run.get("memory", {}).get("rss_peak_bytes", 0)))

        signature = _account_ledger_signature(run)
        if reference_signature is None and workers == 1:
            reference_signature = signature
        parity = reference_signature == signature if reference_signature is not None else False
        record = {"workers": workers, "repetition": repetition,
                  "status": run.get("status"), "wall_seconds": run.get("timings", {}).get("total_seconds"),
                  "recognition_seconds_sum": run.get("timings", {}).get("recognition_seconds"),
                  "backtrader_seconds": run.get("timings", {}).get("backtrader_seconds"),
                  "process_tree_peak_bytes": peak, "account_count": run.get("summary", {}).get("account_count"),
                  "failed_account_count": run.get("summary", {}).get("failed_account_count"),
                  "signal_count": run.get("summary", {}).get("signal_count"),
                  "trade_count": run.get("summary", {}).get("trade_count"),
                  "accounting_pass": _all_ledger_checks_pass(run, len(symbols)),
                  "parity_with_one_worker": parity,
                  "ledger_signature_sha256": __import__("hashlib").sha256(signature.encode()).hexdigest(),
                  "report_path": str((run_dir / "daily-research.json").resolve())}
        report["runs"].append(record)
        completed_by_workers = {}
        for count in COUNTS:
            matches = [item for item in report["runs"] if item["workers"] == count]
            if len(matches) == REPETITIONS:
                completed_by_workers[str(count)] = matches
        report["completed_runs"] = len(report["runs"])
        report["total_runs"] = len(ordered_runs)
        report["last_completed"] = record
        report["workers"] = completed_by_workers
        progress_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"completed": index, "total": len(ordered_runs), **record}, ensure_ascii=False),
              flush=True)

    seconds, peaks, parity = {}, {}, {}
    for count in COUNTS:
        samples = [row["wall_seconds"] for row in report["runs"] if row["workers"] == count]
        peak_samples = [row["process_tree_peak_bytes"] for row in report["runs"]
                        if row["workers"] == count]
        parity[str(count)] = all(row["parity_with_one_worker"] for row in report["runs"]
                                 if row["workers"] == count)
        seconds[str(count)] = sum(samples) / len(samples) if samples else None
        peaks[str(count)] = max(peak_samples, default=0)
    recommendation, fastest, eligible = _select_safe_worker_count(seconds, peaks, parity)
    report.update({"status": "complete" if recommendation is not None else "parity_or_memory_failed",
                   "median_or_mean_wall_seconds_by_workers": seconds,
                   "peak_bytes_by_workers": peaks, "parity_by_workers": parity,
                   "fastest_wall_seconds": fastest,
                   "within_5_percent_fastest": eligible,
                   "recommended_workers": recommendation,
                   "completed_at_epoch": time.time()})
    progress_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "recommended_workers": recommendation,
                      "wall_seconds_by_workers": seconds, "peak_bytes_by_workers": peaks,
                      "parity_by_workers": parity}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
