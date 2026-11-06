#!/usr/bin/env python3
"""Validate the active recognition source with a same-input serial vs spawn-8 run.

This bounded execution-policy check covers 36 pinned symbols. It is not a new
1,000-account acceptance run and does not use prior-version account ledgers as
the semantic reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen import daily_research as research  # noqa: E402
from caisen.market import discover_instruments  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path):
    stat = Path(path).stat()
    return "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)


def _write(path, payload):
    research._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2,
                                              default=research._json_default))


def _source_inputs(reference_manifest_path, snapshot_root):
    reference_manifest = json.loads(Path(reference_manifest_path).read_text(encoding="utf-8"))
    snapshot_root = Path(snapshot_root).resolve()
    if reference_manifest.get("status") != "validated" or reference_manifest.get("date_to") != "2026-10-30":
        raise ValueError("the pinned 36-symbol manifest must be validated at cutoff 2026-10-30")
    references = reference_manifest.get("references", [])
    keys = {(row.get("kind"), row.get("code")) for row in references}
    if len(keys) != 36:
        raise ValueError("expected 36 unique references")
    snapshot_manifest = json.loads((snapshot_root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if (snapshot_manifest.get("status") != "complete"
            or snapshot_manifest.get("account_count") != 36
            or snapshot_manifest.get("date_to") != reference_manifest.get("date_to")):
        raise ValueError("frozen 36-input manifest does not match the pinned reference scope")
    registry = snapshot_root / "caisen-system" / "config" / "instruments.json"
    if not registry.is_file():
        raise ValueError("frozen snapshot instrument registry is missing")
    os.environ["CAISEN_INSTRUMENTS"] = str(registry.resolve())
    frozen_files = {(row.get("kind"), row.get("code")): row
                    for row in snapshot_manifest.get("files", [])}
    if set(frozen_files) != keys:
        raise ValueError("frozen input manifest symbol set differs from the pinned 36-symbol set")
    instruments = {(row.kind, row.code): row for row in discover_instruments(snapshot_root)}
    rows = []
    for kind, code in sorted(keys):
        frozen = frozen_files.get((kind, code))
        instrument = instruments.get((kind, code))
        if not frozen or instrument is None:
            raise ValueError("missing frozen input or snapshot inventory for %s:%s" % (kind, code))
        path = Path(frozen["snapshot_path"]).resolve()
        if Path(instrument.path).resolve() != path or path.stat().st_size == 0:
            raise ValueError("snapshot path mismatch for %s:%s" % (kind, code))
        digest = _sha256(path)
        if digest != frozen.get("sha256"):
            raise ValueError("snapshot hash mismatch for %s:%s" % (kind, code))
        if _revision(path) != frozen.get("snapshot_data_revision"):
            raise ValueError("snapshot revision mismatch for %s:%s" % (kind, code))
        previous_reference = next(row for row in references
                                  if (row.get("kind"), row.get("code")) == (kind, code))
        loaded = research.load_research_bars(instrument, root=snapshot_root)
        if loaded.get("data_revision") != frozen.get("snapshot_data_revision"):
            raise ValueError("normal loader revision mismatch for %s:%s" % (kind, code))
        if (frozen.get("tick_size") != loaded.get("tick_size")
                or frozen.get("tick_size_source") != loaded.get("tick_size_source")):
            raise ValueError("frozen quote settings mismatch for %s:%s" % (kind, code))
        rows.append({"kind": kind, "code": code, "snapshot_path": str(path),
                     "snapshot_revision": frozen["snapshot_data_revision"],
                     "snapshot_sha256": digest, "prior_reference_source": previous_reference.get("source"),
                     "prior_reference_account": previous_reference.get("account_path"),
                     "tick_size": loaded.get("tick_size"),
                     "tick_size_source": loaded.get("tick_size_source"),
                     "loaded_rows": loaded.get("normalized_rows")})
    return reference_manifest, snapshot_manifest, instruments, rows


def run_policy(reference_manifest_path, snapshot_root, universe_path,
               output_dir, progress_callback=None):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("execution-policy output directory must be new and empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest, snapshot_manifest, instruments, inputs = _source_inputs(
        reference_manifest_path, snapshot_root)
    expected_keys = {(item["kind"], item["code"]) for item in inputs}
    input_revisions = {(item["kind"], item["code"]): item["snapshot_revision"] for item in inputs}
    date_to = "2026-10-30"
    callback = progress_callback or (lambda item: print(json.dumps(item, ensure_ascii=False), flush=True))
    versions = research._version_metadata()
    source_hashes = {name: _sha256(REPO / name) for name in (
        "caisen/daily_incremental.py", "caisen/main_bottoms.py",
        "caisen/rules/patterns.v2.json", "caisen/fixed_backtest.py",
        "caisen/research_config.py", "caisen/daily_research.py",
        "scripts/benchmark_current_rule_workers.py")}
    runs = {}
    peaks = {}
    elapsed = {}
    report = None
    progress_path = output_dir / "worker-policy-progress.json"

    def run_one(label, worker_count, execution_mode):
        started = time.perf_counter()
        callback({"stage": label + "_started", "workers": worker_count,
                  "execution_mode": execution_mode, "account_count": len(expected_keys)})
        result, peak = research._run_with_process_tree_peak(lambda: research.run_daily_research(
            universe_path=universe_path, output_dir=output_dir / label, date_to=date_to,
            workers=worker_count, root=snapshot_root, symbols=sorted(expected_keys),
            recognizer="boundary_first", execution_mode=execution_mode,
            progress_callback=callback,
            worker_source_override=("current-rule-serial-reference" if worker_count == 1
                                    else "current-rule-spawn8-candidate")))
        elapsed[label] = time.perf_counter() - started
        peaks[label] = peak
        runs[label] = result
        _write(output_dir / (label + "-timing.json"), {
            "wall_seconds": elapsed[label], "process_tree_peak_bytes": peak,
            "runner_timings": result.get("timings"),
            "account_count": len(result.get("account_summaries", [])),
        })
        return result

    try:
        # Run one main-process reference first, then the spawn pool on identical inputs.
        reference = run_one("serial-reference", 1, "in_process")
        candidate = run_one("spawn8-candidate", 8, "spawn")
        serial_rows = {(row["kind"], row["code"]): row for row in reference.get("account_summaries", [])}
        parity, divergence = research._compare_account_ledgers(reference, candidate, expected_keys)
        accounting_ok = research._all_ledger_checks_pass(candidate, len(expected_keys))
        serial_accounting_ok = research._all_ledger_checks_pass(reference, len(expected_keys))
        revisions_ok = len(candidate.get("account_summaries", [])) == len(expected_keys) and all(
            row.get("provenance", {}).get("data_revision") == input_revisions[(row["kind"], row["code"])]
            and serial_rows.get((row["kind"], row["code"]), {}).get("provenance", {}).get("data_revision")
                == input_revisions[(row["kind"], row["code"])]
            and row.get("provenance", {}).get("used_last_date") == date_to
            and serial_rows.get((row["kind"], row["code"]), {}).get("provenance", {}).get("used_last_date")
                == date_to
            for row in candidate.get("account_summaries", []))
        peak = peaks.get("spawn8-candidate", 0)
        memory_ok = 0 < peak <= research.BENCHMARK_PROCESS_TREE_RSS_CAP
        performance_ok = elapsed["spawn8-candidate"] < elapsed["serial-reference"]
        source_hashes_after = {name: _sha256(REPO / name) for name in source_hashes}
        source_unchanged = source_hashes_after == source_hashes
        status = "complete" if (reference.get("status") == "complete"
                                and candidate.get("status") == "complete"
                                and parity and accounting_ok and serial_accounting_ok
                                and revisions_ok and memory_ok and performance_ok
                                and source_unchanged) else "failed"
        reference_signature = hashlib.sha256(
            research._account_ledger_signature(reference).encode("utf-8")).hexdigest()
        candidate_signature = hashlib.sha256(
            research._account_ledger_signature(candidate).encode("utf-8")).hexdigest()
        report = {
            "status": status,
            "benchmark_id": output_dir.name,
            "benchmark_protocol_version": research.BENCHMARK_PROTOCOL_VERSION,
            "policy_scope": research.BENCHMARK_POLICY_SCOPE,
            "reference_mode": "single_process_same_current_code",
            "validation_scope_note": "36 pinned accounts only; not a 1000-account baseline acceptance",
            "universe_id": research.UNIVERSE_ID,
            "reference_count": len(expected_keys),
            "reference_manifest_path": str(Path(reference_manifest_path).resolve()),
            "reference_manifest_protocol": reference_manifest.get("protocol"),
            "snapshot_root": str(Path(snapshot_root).resolve()),
            "snapshot_id": snapshot_manifest.get("snapshot_id"),
            "snapshot_manifest_path": str((Path(snapshot_root) / "snapshot-manifest.json").resolve()),
            "paired_source_map_path": snapshot_manifest.get("paired_source_map_path"),
            "date_to": date_to,
            "versions": versions,
            "source_sha256": source_hashes,
            "source_sha256_after": source_hashes_after,
            "source_unchanged_during_run": source_unchanged,
            "inputs": inputs,
            "worker_counts": {
                "1": {"seconds": elapsed["serial-reference"],
                      "process_tree_peak_bytes": peaks["serial-reference"],
                      "execution_mode": "in_process", "account_count": len(reference.get("account_summaries", [])),
                      "accounting_checks_passed": serial_accounting_ok,
                      "ledger_signature_sha256": reference_signature},
                "8": {"seconds": elapsed["spawn8-candidate"],
                      "process_tree_peak_bytes": peak, "execution_mode": "spawn",
                      "account_count": len(candidate.get("account_summaries", [])),
                      "parity_passed": parity, "first_divergence": divergence,
                      "accounting_checks_passed": accounting_ok,
                      "data_revisions_match_references": revisions_ok,
                      "memory_status": "pass" if memory_ok else "unverified_or_over_cap",
                      "ledger_signature_sha256": candidate_signature,
                      "runner_timings": candidate.get("timings")},
            },
            "recommended_workers": 8 if status == "complete" else None,
            "recommended_recognizer": "boundary_first" if status == "complete" else None,
            "performance_gate_passed": performance_ok,
            "speedup_vs_serial": (elapsed["serial-reference"] / elapsed["spawn8-candidate"]
                                  if elapsed["spawn8-candidate"] > 0 else None),
            "reference_ledger_signature_sha256": reference_signature,
            "candidate_ledger_signature_sha256": candidate_signature,
            "parity_passed": parity and reference_signature == candidate_signature,
            "accounting_checks_passed": accounting_ok and serial_accounting_ok,
            "data_revisions_match_references": revisions_ok,
            "process_tree_memory_cap_bytes": research.BENCHMARK_PROCESS_TREE_RSS_CAP,
            "timing_scope": "each run timed from run_daily_research call through report write",
            "elapsed_seconds": elapsed,
            "error": None if status == "complete" else "one or more current-rule worker gates failed",
        }
        report["parity_passed"] = report["parity_passed"] and status == "complete"
        report_path = output_dir / "worker-policy-result.json"
        _write(report_path, report)
        if status == "complete":
            # The ordinary runner consumes only this versioned policy artifact.
            _write(research.BENCHMARK_PATH, report)
        _write(progress_path, {"status": status, "stage": "report_written",
                               "report_path": str(report_path.resolve()),
                               "reference_count": len(expected_keys), "completed": len(expected_keys),
                               "total": len(expected_keys)})
        callback({"status": status, "stage": "report_written",
                  "report_path": str(report_path.resolve()), "parity_passed": report["parity_passed"]})
        return report
    except Exception as exc:
        failure = {"status": "failed", "policy_scope": research.BENCHMARK_POLICY_SCOPE,
                   "reference_manifest_path": str(Path(reference_manifest_path).resolve()),
                   "snapshot_root": str(Path(snapshot_root).resolve()), "versions": versions,
                   "source_sha256": source_hashes, "elapsed_seconds": elapsed,
                   "error": "%s: %s" % (type(exc).__name__, exc)}
        _write(output_dir / "worker-policy-result.json", failure)
        _write(progress_path, {"status": "failed", "stage": "failed", "error": failure["error"]})
        raise


def main():
    base = REPO / "reports" / "daily-research"
    benchmark_dir = base / "benchmark" / "1704b24a04344b81a8bafeb1c3c393a4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path,
        default=benchmark_dir / "parity-reference-manifest.json")
    parser.add_argument("--snapshot-root", type=Path,
        default=base / "benchmark" / "current-rules-36-input-snapshot-20261105-retry1")
    parser.add_argument("--universe", type=Path,
        default=REPO / "research" / "universes" / "daily_800_stock_200_etf_v1.json")
    parser.add_argument("--output", type=Path,
        default=base / "benchmark" / "current-rules-36-20261105")
    args = parser.parse_args()
    result = run_policy(args.reference_manifest, args.snapshot_root, args.universe, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=research._json_default))
    if result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
