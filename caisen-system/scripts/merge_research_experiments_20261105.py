#!/usr/bin/env python3
"""Integrate baseline/open/min3 segmented results with run and validation evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO / "reports" / "daily-research"
COMPOSITE_ROOT = REPORT_ROOT / "paired-composites-20261105-snapshot-v1"
MAIN_REPORT = REPORT_ROOT / "research-review-paired-baseline-20261105"
MIN3_REPORT = REPORT_ROOT / "research-review-paired-min3-20261105"
FINAL_REPORT = REPORT_ROOT / "research-review-combined-20261105"
COLD_RUN = REPORT_ROOT / "full-cold-20261105-v8-boundary-w8-spawnfix"
MIN3_RUN = REPORT_ROOT / "min3-20261105-8worker"
BENCHMARK = REPORT_ROOT / "benchmark" / "1704b24a04344b81a8bafeb1c3c393a4" / "benchmark-result.json"
PARITY = REPORT_ROOT / "min3-recognizer-parity-20261105.json"
BASELINE_PARITY = REPORT_ROOT / "min3-baseline-snapshot-parity-20261105.json"
WARM_CACHE_AUDIT = REPORT_ROOT / "warm-cache-20261105-8worker-retry1" / "warm-cache-audit.json"
PAIR_REPAIR = REPORT_ROOT / "paired-repairs-20261105-snapshot-v1" / "paired-repair-manifest.json"
SNAPSHOT_MANIFEST = REPORT_ROOT / "paired-input-snapshot-20261105-v1" / "snapshot-manifest.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compact_parity(payload):
    return {"status": payload.get("status"),
            "experiment_config": payload.get("experiment_config"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "cases": [{key: case.get(key) for key in (
                "code", "kind", "source", "rows", "signals_equal", "v7_signal_count",
                "boundary_first_signal_count", "first_difference", "baseline_vs_min3")}
                       for case in payload.get("cases", [])]}


def _recompute_recross_counts(archive):
    spec = importlib.util.spec_from_file_location(
        "review_builder_recross", REPO / "scripts" / "build_research_review_20261105.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARCHIVE = Path(archive)
    rows, diagnostics = [], {}
    from collections import Counter
    diagnostics = Counter()
    for account_path in module._account_paths():
        account = _read(account_path)
        rows.extend(module._early_breakout_late_recross_rows(account, diagnostics))
    return module._recross_summary(rows, diagnostics)


def main():
    main_json = MAIN_REPORT / "research-review.json"
    min3_json = MIN3_REPORT / "research-review.json"
    cold_manifest_path = COLD_RUN / "run-manifest.json"
    min3_manifest_path = MIN3_RUN / "run-manifest.json"
    min3_provenance_path = MIN3_RUN / "experiment-provenance.json"
    required = (main_json, min3_json, cold_manifest_path, min3_manifest_path,
                min3_provenance_path,
                BENCHMARK, PARITY, BASELINE_PARITY, PAIR_REPAIR, SNAPSHOT_MANIFEST,
                WARM_CACHE_AUDIT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required experiment artifacts missing: " + ", ".join(missing))

    payload = _read(main_json)
    min3_payload = _read(min3_json)
    baseline_recross = _recompute_recross_counts(COMPOSITE_ROOT / "baseline")
    min3_recross = _recompute_recross_counts(COMPOSITE_ROOT / "min3")
    payload["early_breakout_late_recross_scan"] = baseline_recross
    min3_payload["early_breakout_late_recross_scan"] = min3_recross
    cold_manifest, min3_manifest = _read(cold_manifest_path), _read(min3_manifest_path)
    min3_provenance = _read(min3_provenance_path)
    cold_progress, min3_progress = _read(COLD_RUN / "progress.json"), _read(MIN3_RUN / "progress.json")
    benchmark, parity = _read(BENCHMARK), _read(PARITY)
    baseline_parity = _read(BASELINE_PARITY)
    pair_repair, snapshot_manifest = _read(PAIR_REPAIR), _read(SNAPSHOT_MANIFEST)
    warm_cache = _read(WARM_CACHE_AUDIT)
    acceptance = _read(COLD_RUN / "acceptance-audit.json")
    benchmark_8 = benchmark.get("worker_counts", {}).get("8", {})
    r02 = next((case for case in parity.get("cases", [])
                if case.get("code") == "000008"), {})
    if min3_payload.get("account_count") != 1000 or min3_manifest.get("status") != "complete":
        raise RuntimeError("min3 full run and period report must both be complete for integration")
    if cold_manifest.get("status") != "complete" or cold_manifest.get("summary", {}).get("account_count") != 1000:
        raise RuntimeError("cold baseline is incomplete")
    if benchmark.get("status") != "complete" or benchmark.get("parity_passed") is not True:
        raise RuntimeError("worker benchmark parity is not complete")
    if parity.get("status") != "complete" or not all(
            case.get("signals_equal") is True for case in parity.get("cases", [])):
        raise RuntimeError("strict min3 engine parity failed")
    if (warm_cache.get("status") != "complete" or warm_cache.get("cache_hits") != 1000
            or warm_cache.get("full_normalized_account_ledger_parity") is not True):
        raise RuntimeError("warm-cache verification failed")
    if baseline_parity.get("status") != "complete" or not all(
            case.get("baseline_shapes_match_snapshot") is True and
            case.get("snapshot_signal_count_matches") is True
            for case in baseline_parity.get("cases", [])):
        raise RuntimeError("baseline regression snapshots do not reconcile")

    min3_periods = {row["name"]: row for row in min3_payload.get("periods", [])}
    for row in payload.get("periods", []):
        other = min3_periods.get(row["name"])
        if other is None:
            raise ValueError("min3 report is missing period: " + row["name"])
        row["variants"]["neckline_min3"] = copy.deepcopy(other["variants"]["baseline"])

    benchmark_worker_seconds = {
        count: item.get("seconds_samples", [])
        for count, item in benchmark.get("worker_counts", {}).items()
    }
    payload["experiment_integration"] = {
        "status": "complete",
        "baseline_run": {
            "run_id": cold_manifest.get("run_id"),
            "status": cold_manifest.get("status"),
            "recognizer": cold_manifest.get("recognizer"),
            "engine_version": cold_manifest.get("engine_version"),
            "source_rule_version": cold_manifest.get("source_rule_version"),
            "workers": cold_manifest.get("parameters", {}).get("workers"),
            "elapsed_seconds": acceptance.get("wall_seconds_load_through_report_write"),
            **cold_manifest.get("summary", {}),
            "archive_acceptance_audit": acceptance,
        },
        "min3_experiment": {
            "run_id": min3_manifest.get("run_id"),
            "status": min3_manifest.get("status"),
            "recognizer": min3_manifest.get("recognizer"),
            "engine_version": min3_manifest.get("engine_version"),
            "source_rule_version": min3_manifest.get("source_rule_version"),
            "recognition_config": min3_manifest.get("parameters", {}).get("recognition_config"),
            "account_and_signal_provenance": min3_provenance,
            "workers": min3_manifest.get("parameters", {}).get("workers"),
            "elapsed_seconds": min3_progress.get("elapsed_seconds"),
            "runtime_caveat": "This strict-min3 run is a research result, not the cold performance acceptance baseline; a small timing overlap with the cold run makes its elapsed time unsuitable as a controlled speed comparison.",
            **min3_manifest.get("summary", {}),
            "period_report": str(min3_json.relative_to(REPO)),
        },
        "worker_benchmark": {
            "status": benchmark.get("status"),
            "benchmark_id": benchmark.get("benchmark_id"),
            "protocol_version": benchmark.get("benchmark_protocol_version"),
            "reference_count": benchmark.get("reference_count"),
            "tested_workers": sorted(benchmark_worker_seconds, key=int),
            "wall_seconds_by_worker": benchmark_worker_seconds,
            "recommended_recognizer": benchmark.get("recommended_recognizer"),
            "recommended_workers": benchmark.get("recommended_workers"),
            "ledger_parity_passed": benchmark.get("ledger_parity_passed"),
            "accounting_checks_passed": benchmark.get("accounting_checks_passed"),
            "data_revisions_match_references": benchmark.get("data_revisions_match_references"),
            "process_tree_memory_cap_bytes": benchmark.get("process_tree_memory_cap_bytes"),
        },
        "warm_cache_verification": {
            "status": warm_cache.get("status"),
            "workers": warm_cache.get("workers"),
            "cache_hits": warm_cache.get("cache_hits"),
            "seed_source": warm_cache.get("seed_source"),
            "seed_elapsed_seconds": warm_cache.get("seed_elapsed_seconds"),
            "cache_preflight_seconds": warm_cache.get("cache_preflight_seconds"),
            "warm_wall_seconds": warm_cache.get("warm_wall_seconds"),
            "ledger_parity": warm_cache.get("full_normalized_account_ledger_parity"),
            "data_revisions_match_snapshot": warm_cache.get("data_revisions_match_snapshot"),
            "accounting_checks_pass": warm_cache.get("all_accounting_checks_pass"),
            "process_tree_peak_bytes": warm_cache.get("process_tree_peak_bytes"),
            "audit_path": str(WARM_CACHE_AUDIT.relative_to(REPO)),
            "comparison_note": "This is a verified seeded-cache replay; compare the cold 5,019.36-second full load-through-write run separately.",
        },
        "strict_recognizer_parity": _compact_parity(parity),
        "baseline_snapshot_parity": baseline_parity,
        "recognition_case_findings": {
            "R02_000008_human_negative": {
                "strict_min3_signal_count": r02.get("boundary_first_signal_count"),
                "baseline_vs_min3": r02.get("baseline_vs_min3"),
                "finding": "Strict min3 did not cure the human-negative fixture; a new three-test W signal remained."
            },
            "delayed_maturity_breakout": payload.get("delayed_maturity_breakout_audit"),
            "early_breakout_late_recross": {
                "baseline": {key: baseline_recross.get(key) for key in
                             ("delayed_recross_signal_count", "delayed_recross_pattern_count",
                              "affected_symbol_count", "by_pattern_type", "excluded_evidence_counts")},
                "strict_min3": {key: min3_recross.get(key) for key in
                                ("delayed_recross_signal_count", "delayed_recross_pattern_count",
                                 "affected_symbol_count", "by_pattern_type", "excluded_evidence_counts")},
                "counting_rule": "unique kind/code/signal_date tuples count as signals; pattern snapshots are reported separately",
            },
        },
        "runtime_acceptance": {
            "cold_run_load_through_report_write_seconds": acceptance.get("wall_seconds_load_through_report_write"),
            "cold_run_target_seconds": 1800,
            "cold_run_target_met": bool(acceptance.get("wall_seconds_load_through_report_write", 0) <= 1800),
            "cold_run_elapsed_source": "acceptance audit wall_seconds_load_through_report_write",
            "cold_run_exact_1000_archive_ledger_parity": acceptance.get("full_normalized_account_ledger_parity"),
            "cold_run_account_count": acceptance.get("candidate_accounts"),
            "cold_run_signatures_equal": acceptance.get("ledger_signatures_equal"),
            "cold_run_peak_process_tree_bytes": acceptance.get("process_tree_peak_bytes"),
            "worker_selection": {
                "recognizer": benchmark.get("recommended_recognizer"),
                "workers": benchmark.get("recommended_workers"),
                "8_worker_reference_seconds": benchmark_8.get("seconds_samples"),
                "8_worker_process_tree_peak_bytes": benchmark_8.get("process_tree_peak_bytes"),
                "8_worker_memory_status": benchmark_8.get("memory_status"),
                "benchmark_path": str(BENCHMARK.relative_to(REPO)),
                "pool_memory_cap_bytes": benchmark.get("process_tree_memory_cap_bytes"),
            },
            "full_suite_result": "340 passed in the latest reported full-suite run; current report/source-map/promotion focused tests: 4 passed",
        },
        "paired_input_hybrid": {
            "source_policy": pair_repair.get("source_policy"),
            "snapshot_id": snapshot_manifest.get("snapshot_id"),
            "snapshot_root": pair_repair.get("snapshot_path"),
            "snapshot_manifest": str(SNAPSHOT_MANIFEST.relative_to(REPO)),
            "paired_repair_manifest": str(PAIR_REPAIR.relative_to(REPO)),
            "paired_input_snapshot_html_link": "../paired-input-snapshot-20261105-v1/snapshot-manifest.json",
            "changed_since_baseline_count": pair_repair.get("changed_by_run_counts", {}).get("baseline"),
            "changed_since_min3_count": pair_repair.get("changed_by_run_counts", {}).get("min3"),
            "paired_symbol_count": pair_repair.get("paired_account_count"),
            "baseline_elapsed_seconds": pair_repair.get("baseline_elapsed_seconds"),
            "min3_elapsed_seconds": pair_repair.get("min3_elapsed_seconds"),
            "baseline_composite": str((COMPOSITE_ROOT / "baseline" / "run-manifest.json").relative_to(REPO)),
            "min3_composite": str((COMPOSITE_ROOT / "min3" / "run-manifest.json").relative_to(REPO)),
            "integrity_note": "All 1,000 current input CSVs were frozen with before/after stat checks and SHA-256. Changed symbols were rerun in baseline/min3 pairs on the same frozen copies; unchanged-symbol account artifacts were reused. Original cold and min3 archives remain untouched. This is a paired hybrid report source, not a byte-for-byte replay of the historical inputs."
        },
        "strategy_constraints": {
            "evaluation_windows": [{"name": "through_2020", "end": "2020-12-31"},
                                   {"name": "2021_2023", "start": "2021-01-01", "end": "2023-12-31"},
                                   {"name": "2024_2026", "start": "2024-01-01", "end": "2026-10-30"}],
            "period_accounting": "fresh 100000 cash account per symbol and period; no carried positions or orders",
            "recognition_history": "full prefix retained before the period start",
            "stop_fraction": 0.08,
            "target_fraction": 0.20,
            "pattern_types": ["w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"],
            "baseline_breakout_buffer": "prior completed session ATR plus at least two ticks",
            "open_admission": "emitted tag entry open within projected frozen boundary through boundary + 3 signal-day ATR, inclusive",
            "min3_admission": "at least three independent neckline tests; same emitted four-pattern types",
            "mfe": "strictly interior holding-session highs; entry and exit sessions excluded",
        },
        "report_sources": {
            "baseline_period_report": str(main_json.relative_to(REPO)),
            "min3_period_report": str(min3_json.relative_to(REPO)),
            "cold_run_manifest": str(cold_manifest_path.relative_to(REPO)),
            "min3_run_manifest": str(min3_manifest_path.relative_to(REPO)),
            "min3_experiment_provenance": str(min3_provenance_path.relative_to(REPO)),
            "worker_benchmark": str(BENCHMARK.relative_to(REPO)),
            "warm_cache_audit": str(WARM_CACHE_AUDIT.relative_to(REPO)),
            "min3_parity": str(PARITY.relative_to(REPO)),
            "baseline_snapshot_parity": str(BASELINE_PARITY.relative_to(REPO)),
            "paired_input_snapshot": str(SNAPSHOT_MANIFEST.relative_to(REPO)),
            "paired_repair_manifest": str(PAIR_REPAIR.relative_to(REPO)),
            "baseline_composite_source_map": str((COMPOSITE_ROOT / "baseline" / "data-source-map.json").relative_to(REPO)),
            "min3_composite_source_map": str((COMPOSITE_ROOT / "min3" / "data-source-map.json").relative_to(REPO)),
        },
    }

    FINAL_REPORT.mkdir(parents=True, exist_ok=True)
    charts = FINAL_REPORT / "charts"
    if not charts.exists() and (MAIN_REPORT / "charts").is_dir():
        charts.symlink_to((MAIN_REPORT / "charts").resolve(), target_is_directory=True)
    final_json = FINAL_REPORT / "research-review.json"
    final_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "review_builder", REPO / "scripts" / "build_research_review_20261105.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (FINAL_REPORT / "research-review.html").write_text(module._render_html(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(final_json),
                      "html": str(FINAL_REPORT / "research-review.html"),
                      "variants": sorted(payload["periods"][0]["variants"]),
                      "min3_run": str(MIN3_RUN)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
