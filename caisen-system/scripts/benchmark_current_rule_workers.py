#!/usr/bin/env python3
"""Validate the active recognition source with a same-input serial vs spawn-pool run.

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
from contextlib import contextmanager
from pathlib import Path

# Set before importing daily_research/numpy so the spawned pool inherits the cap.
for _thread_env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen import daily_research as research  # noqa: E402
from caisen.market import discover_instruments  # noqa: E402
from caisen.research_config import RecognitionExperimentConfig  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path):
    stat = Path(path).stat()
    return "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)


@contextmanager
def _using_registry(registry_path):
    previous = os.environ.get("CAISEN_INSTRUMENTS")
    os.environ["CAISEN_INSTRUMENTS"] = str(Path(registry_path).resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CAISEN_INSTRUMENTS", None)
        else:
            os.environ["CAISEN_INSTRUMENTS"] = previous


def _quote_settings(value):
    return {key: value.get(key) for key in
            ("tick_size", "tick_size_source", "tick_rule_version")}


def _input_fingerprint(inputs):
    return tuple(sorted((item["kind"], item["code"], item["snapshot_path"],
                         item["snapshot_revision"], item["registry_path"],
                         item["registry_revision"], item["active_quote_manifest_path"],
                         item["active_quote_manifest_revision"], item["tick_size"],
                         item["tick_size_source"], item["tick_rule_version"])
                        for item in inputs))


def _keyed_rows(items, *, label):
    rows = {(item.get("kind"), item.get("code")): item for item in items}
    if len(rows) != len(items):
        raise ValueError("%s has duplicate instrument rows" % label)
    return rows


def _scope_inputs(source_reference_manifest_path, snapshot_root):
    source_reference = json.loads(Path(source_reference_manifest_path).read_text(encoding="utf-8"))
    snapshot_root = Path(snapshot_root).resolve()
    snapshot_manifest_path = snapshot_root / "snapshot-manifest.json"
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    if (source_reference.get("status") != "validated"
            or source_reference.get("date_to") != "2026-10-30"):
        raise ValueError("the source 36-symbol manifest must be validated at cutoff 2026-10-30")
    references = source_reference.get("references", [])
    keys = {(row.get("kind"), row.get("code")) for row in references}
    if len(keys) != 36:
        raise ValueError("expected 36 unique source references")
    if (snapshot_manifest.get("status") != "complete"
            or snapshot_manifest.get("account_count") != 36
            or snapshot_manifest.get("date_to") != source_reference.get("date_to")):
        raise ValueError("frozen 36-input manifest does not match the source reference scope")
    frozen_files = _keyed_rows(snapshot_manifest.get("files", []), label="snapshot manifest")
    if set(frozen_files) != keys:
        raise ValueError("frozen input manifest symbol set differs from source references")
    registry = snapshot_root / "caisen-system" / "config" / "instruments.json"
    if not registry.is_file():
        raise ValueError("frozen snapshot instrument registry is missing")
    with _using_registry(registry):
        instruments = {(row.kind, row.code): row for row in discover_instruments(snapshot_root)}
    if set(instruments) != keys:
        raise ValueError("snapshot instrument inventory differs from the source 36-symbol scope")
    return source_reference, snapshot_manifest, keys, frozen_files, instruments, registry


def prepare_active_quote_manifest(source_reference_manifest_path, snapshot_root, output_path):
    """Freeze current loader quote rules for the same pinned 36 inputs.

    The old parity manifest and input snapshot are read-only lineage evidence;
    their historical quote metadata is recorded but never required to match
    the active loader's current tick policy.
    """
    source_reference, snapshot_manifest, keys, frozen_files, instruments, registry = _scope_inputs(
        source_reference_manifest_path, snapshot_root)
    registry_revision = _revision(registry)
    source_refs = _keyed_rows(source_reference["references"], label="source reference")
    rows, changed = [], []
    for kind, code in sorted(keys):
        frozen = frozen_files[(kind, code)]
        instrument = instruments[(kind, code)]
        path = Path(frozen["snapshot_path"]).resolve()
        if Path(instrument.path).resolve() != path or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("snapshot path mismatch for %s:%s" % (kind, code))
        revision = _revision(path)
        if revision != frozen.get("snapshot_data_revision"):
            raise ValueError("snapshot revision mismatch for %s:%s" % (kind, code))
        with _using_registry(registry):
            loaded = research.load_research_bars(instrument, root=Path(snapshot_root).resolve())
        if loaded.get("data_revision") != revision:
            raise ValueError("normal loader revision mismatch for %s:%s" % (kind, code))
        if loaded.get("tick_rule_version") != research.TICK_RULE_VERSION:
            raise ValueError("active tick rule metadata mismatch for %s:%s" % (kind, code))
        old_quote = {"tick_size": frozen.get("tick_size"),
                     "tick_size_source": frozen.get("tick_size_source"),
                     "tick_rule_version": frozen.get("tick_rule_version")}
        new_quote = _quote_settings(loaded)
        if old_quote != new_quote:
            changed.append({"kind": kind, "code": code,
                            "previous_snapshot_quote": old_quote,
                            "active_loader_quote": new_quote})
        rows.append({"kind": kind, "code": code,
                     "source": source_refs[(kind, code)].get("source"),
                     "snapshot_path": str(path), "snapshot_revision": revision,
                     "registry_path": str(registry.resolve()),
                     "registry_revision": registry_revision,
                     **new_quote,
                     "loaded_rows": loaded.get("normalized_rows"),
                     "previous_snapshot_quote": old_quote})
    tick_value_changes = sum(item["previous_snapshot_quote"].get("tick_size")
                             != item["active_loader_quote"].get("tick_size") for item in changed)
    tick_source_changes = sum(item["previous_snapshot_quote"].get("tick_size_source")
                              != item["active_loader_quote"].get("tick_size_source") for item in changed)
    tick_rule_changes = sum(item["previous_snapshot_quote"].get("tick_rule_version")
                            != item["active_loader_quote"].get("tick_rule_version") for item in changed)
    manifest = {
        "format_version": 1,
        "manifest_kind": "active_quote_reference",
        "protocol": "current-rules-active-quotes-36-v1",
        "status": "validated",
        "date_to": source_reference["date_to"],
        "snapshot_id": snapshot_manifest.get("snapshot_id"),
        "snapshot_manifest_path": str((Path(snapshot_root).resolve() / "snapshot-manifest.json")),
        "instrument_registry_path": str(registry.resolve()),
        "instrument_registry_revision": registry_revision,
        "references": [{key: value for key, value in row.items()
                        if key in ("kind", "code", "source", "account_path")}
                       for row in source_reference["references"]],
        "inputs": rows,
        "migration": {
            "kind": "active_loader_quote_rebase",
            "source_reference_manifest": str(Path(source_reference_manifest_path).resolve()),
            "source_reference_protocol": source_reference.get("protocol"),
            "source_snapshot_manifest": str((Path(snapshot_root).resolve() / "snapshot-manifest.json")),
            "source_snapshot_id": snapshot_manifest.get("snapshot_id"),
            "reason": "The legacy manifest froze prior-run quotes; reference and candidate now both use quotes loaded by the active exchange_asset_class tick rule.",
            "migrated_quote_count": len(changed),
            "tick_size_value_change_count": tick_value_changes,
            "tick_size_source_change_count": tick_source_changes,
            "tick_rule_version_change_count": tick_rule_changes,
            "changed_quotes": changed,
            "generated_by": "prepare_active_quote_manifest",
        },
    }
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("active quote manifest is frozen; choose a new path")
    _write(output_path, manifest)
    return manifest


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    research._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2,
                                              default=research._json_default))


def _source_inputs(reference_manifest_path, snapshot_root):
    reference_manifest = json.loads(Path(reference_manifest_path).read_text(encoding="utf-8"))
    snapshot_root = Path(snapshot_root).resolve()
    if (reference_manifest.get("manifest_kind") != "active_quote_reference"
            or reference_manifest.get("protocol") != "current-rules-active-quotes-36-v1"
            or reference_manifest.get("status") != "validated"
            or reference_manifest.get("date_to") != "2026-10-30"):
        raise ValueError("reference must be a validated active-quote 36 manifest; migrate old manifests first")
    migration = reference_manifest.get("migration") or {}
    if migration.get("kind") != "active_loader_quote_rebase":
        raise ValueError("active quote migration evidence is missing")
    source_reference_path = Path(migration.get("source_reference_manifest", ""))
    if not source_reference_path.is_file():
        raise ValueError("active quote migration source reference is missing")
    source_reference = json.loads(source_reference_path.read_text(encoding="utf-8"))
    if (source_reference.get("status") != "validated"
            or source_reference.get("protocol") != migration.get("source_reference_protocol")):
        raise ValueError("active quote migration source reference no longer matches its evidence")
    references = reference_manifest.get("references", [])
    keys = {(row.get("kind"), row.get("code")) for row in references}
    if len(keys) != 36:
        raise ValueError("expected 36 unique references")
    if keys != {(row.get("kind"), row.get("code")) for row in source_reference.get("references", [])}:
        raise ValueError("active quote migration changed the pinned source symbol set")
    snapshot_manifest = json.loads((snapshot_root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if (snapshot_manifest.get("status") != "complete"
            or snapshot_manifest.get("account_count") != 36
            or snapshot_manifest.get("date_to") != reference_manifest.get("date_to")):
        raise ValueError("frozen 36-input manifest does not match the pinned reference scope")
    registry = snapshot_root / "caisen-system" / "config" / "instruments.json"
    if not registry.is_file():
        raise ValueError("frozen snapshot instrument registry is missing")
    registry_revision = _revision(registry)
    if (reference_manifest.get("instrument_registry_path") != str(registry.resolve())
            or reference_manifest.get("instrument_registry_revision") != registry_revision):
        raise ValueError("active quote manifest registry revision mismatch")
    frozen_files = {(row.get("kind"), row.get("code")): row
                    for row in snapshot_manifest.get("files", [])}
    if set(frozen_files) != keys:
        raise ValueError("frozen input manifest symbol set differs from the pinned 36-symbol set")
    with _using_registry(registry):
        instruments = {(row.kind, row.code): row for row in discover_instruments(snapshot_root)}
    active_inputs = _keyed_rows(reference_manifest.get("inputs", []), label="active quote manifest")
    if set(active_inputs) != keys:
        raise ValueError("active quote manifest symbol set differs from the pinned 36-symbol set")
    manifest_path = Path(reference_manifest_path).resolve()
    manifest_revision = _revision(manifest_path)
    expected_snapshot_manifest_path = str((snapshot_root / "snapshot-manifest.json").resolve())
    if (reference_manifest.get("snapshot_id") != snapshot_manifest.get("snapshot_id")
            or reference_manifest.get("snapshot_manifest_path") != expected_snapshot_manifest_path
            or migration.get("source_snapshot_id") != snapshot_manifest.get("snapshot_id")
            or migration.get("source_snapshot_manifest") != expected_snapshot_manifest_path):
        raise ValueError("active quote manifest is bound to a different frozen snapshot")
    rows = []
    changed_quotes = []
    for kind, code in sorted(keys):
        frozen = frozen_files.get((kind, code))
        instrument = instruments.get((kind, code))
        if not frozen or instrument is None:
            raise ValueError("missing frozen input or snapshot inventory for %s:%s" % (kind, code))
        path = Path(frozen["snapshot_path"]).resolve()
        if Path(instrument.path).resolve() != path or path.stat().st_size == 0:
            raise ValueError("snapshot path mismatch for %s:%s" % (kind, code))
        if _revision(path) != frozen.get("snapshot_data_revision"):
            raise ValueError("snapshot revision mismatch for %s:%s" % (kind, code))
        with _using_registry(registry):
            loaded = research.load_research_bars(instrument, root=snapshot_root)
        if loaded.get("data_revision") != frozen.get("snapshot_data_revision"):
            raise ValueError("normal loader revision mismatch for %s:%s" % (kind, code))
        if loaded.get("tick_rule_version") != research.TICK_RULE_VERSION:
            raise ValueError("active tick rule metadata mismatch for %s:%s" % (kind, code))
        active = active_inputs[(kind, code)]
        quote = _quote_settings(loaded)
        previous_quote = {"tick_size": frozen.get("tick_size"),
                          "tick_size_source": frozen.get("tick_size_source"),
                          "tick_rule_version": frozen.get("tick_rule_version")}
        if (_quote_settings(active) != quote
                or active.get("previous_snapshot_quote") != previous_quote
                or active.get("snapshot_path") != str(path)
                or active.get("snapshot_revision") != frozen["snapshot_data_revision"]
                or active.get("registry_path") != str(registry.resolve())
                or active.get("registry_revision") != registry_revision):
            raise ValueError("active quote manifest mismatch for %s:%s" % (kind, code))
        if previous_quote != quote:
            changed_quotes.append({"kind": kind, "code": code,
                                   "previous_snapshot_quote": previous_quote,
                                   "active_loader_quote": quote})
        rows.append({"kind": kind, "code": code, "snapshot_path": str(path),
                     "snapshot_revision": frozen["snapshot_data_revision"],
                     "registry_path": str(registry.resolve()),
                     "registry_revision": registry_revision,
                     "active_quote_manifest_path": str(manifest_path),
                     "active_quote_manifest_revision": manifest_revision,
                     "prior_reference_source": active.get("source"),
                     "tick_size": quote["tick_size"],
                     "tick_size_source": quote["tick_size_source"],
                     "tick_rule_version": quote["tick_rule_version"],
                     "loaded_rows": loaded.get("normalized_rows")})
    migration_changes = migration.get("changed_quotes", [])
    expected_changes_by_key = {(item["kind"], item["code"]): item for item in changed_quotes}
    evidence_changes_by_key = {(item.get("kind"), item.get("code")): item
                               for item in migration_changes}
    if (len(changed_quotes) != migration.get("migrated_quote_count")
            or migration.get("tick_size_value_change_count")
                != sum(item["previous_snapshot_quote"].get("tick_size")
                       != item["active_loader_quote"].get("tick_size") for item in changed_quotes)
            or migration.get("tick_size_source_change_count")
                != sum(item["previous_snapshot_quote"].get("tick_size_source")
                       != item["active_loader_quote"].get("tick_size_source") for item in changed_quotes)
            or migration.get("tick_rule_version_change_count")
                != sum(item["previous_snapshot_quote"].get("tick_rule_version")
                       != item["active_loader_quote"].get("tick_rule_version") for item in changed_quotes)
            or evidence_changes_by_key != expected_changes_by_key):
        raise ValueError("active quote migration evidence does not match frozen and active quotes")
    return reference_manifest, snapshot_manifest, instruments, rows


def _assert_inputs_unchanged(reference_manifest_path, snapshot_root, expected_fingerprint):
    _, _, _, current = _source_inputs(reference_manifest_path, snapshot_root)
    if _input_fingerprint(current) != expected_fingerprint:
        raise ValueError("snapshot stat, registry revision, or active quotes changed during run")
    return current


def _account_quotes_match(result, inputs):
    expected = {(row["kind"], row["code"]): _quote_settings(row) for row in inputs}
    summaries = result.get("account_summaries", [])
    if len(summaries) != len(expected):
        return False
    for row in summaries:
        key = (row.get("kind"), row.get("code"))
        account = row.get("account_json")
        if key not in expected or not account or not Path(account).is_file():
            return False
        payload = json.loads(Path(account).read_text(encoding="utf-8"))
        provenance = payload.get("provenance") or {}
        if _quote_settings(provenance) != expected[key]:
            return False
    return True


def _candidate_worker_schedule(candidate_workers, matrix=False):
    candidate_workers = int(candidate_workers)
    allowed = tuple(worker for worker in research.BENCHMARK_WORKER_COUNTS if worker != 1)
    if candidate_workers not in allowed:
        raise ValueError("candidate_workers must be one of 2, 4, 6, 8 or 10")
    return allowed if matrix else (candidate_workers,)


def _select_matrix_worker(metrics, serial_seconds, cap_bytes):
    seconds = {str(worker): row.get("seconds") for worker, row in metrics.items()}
    peaks = {str(worker): row.get("peak", 0) for worker, row in metrics.items()}
    gates = {str(worker): row.get("passed") is True for worker, row in metrics.items()}
    return research._select_safe_worker_count(seconds, peaks, gates, cap_bytes)


def resolve_recognition_profile(profile_name):
    """Resolve the benchmark's explicitly named recognition semantics."""
    name = str(profile_name).strip().lower()
    if name == "default":
        return None, {"name": "default", "cache_key": None,
                      "recognition_experiment": None}
    if name == "g1":
        config = RecognitionExperimentConfig(
            "internal_rebound_coverage_g1_v1", require_internal_rebound_coverage=True)
        return config, {"name": "g1", "cache_key": list(config.cache_key()),
                        "recognition_experiment": research._recognition_experiment_metadata(config)}
    raise ValueError("unknown recognition profile: %s" % profile_name)


def _manifest_matches_recognition_profile(result, profile):
    manifest_path = (result.get("artifacts") or {}).get("manifest_path")
    if not manifest_path or not Path(manifest_path).is_file():
        return False
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    parameters = manifest.get("parameters") or {}
    expected_key = profile["cache_key"] if profile["cache_key"] is not None else {}
    if parameters.get("recognition_config") != expected_key:
        return False
    return manifest.get("recognition_experiment") == profile["recognition_experiment"]


def _matrix_passed(reference_passed, metrics, *, source_unchanged, selected):
    return (reference_passed and bool(metrics) and all(row.get("passed") is True
            for row in metrics.values()) and source_unchanged and selected is not None)


def run_policy(reference_manifest_path, snapshot_root, universe_path,
               output_dir, candidate_workers=10, progress_callback=None, matrix=False,
               recognition_profile="default", certificate_output=None):
    candidate_schedule = _candidate_worker_schedule(candidate_workers, matrix=matrix)
    recognition_config, profile_metadata = resolve_recognition_profile(recognition_profile)
    certificate_path = (Path(certificate_output).resolve() if certificate_output is not None
                        else Path(research.BENCHMARK_PATH).resolve())
    default_certificate_path = Path(research.BENCHMARK_PATH).resolve()
    if profile_metadata["name"] == "g1" and certificate_output is None:
        raise ValueError("G1 requires an independent certificate_output path")
    if profile_metadata["name"] == "g1" and certificate_path == default_certificate_path:
        raise ValueError("G1 certificate_output must be independent of the default certificate")
    if certificate_path.exists():
        raise FileExistsError("active benchmark certificate exists; refusing to overwrite old measurement")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("execution-policy output directory must be new and empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest, snapshot_manifest, instruments, inputs = _source_inputs(
        reference_manifest_path, snapshot_root)
    expected_keys = {(item["kind"], item["code"]) for item in inputs}
    input_revisions = {(item["kind"], item["code"]): item["snapshot_revision"] for item in inputs}
    expected_input_fingerprint = _input_fingerprint(inputs)
    date_to = "2026-10-30"
    callback = progress_callback or (lambda item: print(json.dumps(item, ensure_ascii=False), flush=True))
    versions = research._version_metadata()
    source_hashes = {name: _sha256(REPO / name) for name in (
        "caisen/daily_incremental.py", "caisen/main_bottoms.py", "caisen/structure_boundary.py",
        "caisen/market.py",
        "caisen/rules/patterns.v2.json", "caisen/fixed_backtest.py",
        "caisen/research_config.py", "caisen/daily_research.py",
        "scripts/benchmark_current_rule_workers.py")}
    runs = {}
    peaks = {}
    elapsed = {}
    input_validation_checkpoints = []
    report = None
    progress_path = output_dir / "worker-policy-progress.json"

    def run_one(label, worker_count, execution_mode):
        _assert_inputs_unchanged(reference_manifest_path, snapshot_root, expected_input_fingerprint)
        input_validation_checkpoints.append("before_" + label)
        started = time.perf_counter()
        callback({"stage": label + "_started", "workers": worker_count,
                  "execution_mode": execution_mode, "account_count": len(expected_keys)})
        with _using_registry(inputs[0]["registry_path"]):
            result, peak = research._run_with_process_tree_peak(lambda: research.run_daily_research(
                universe_path=universe_path, output_dir=output_dir / label, date_to=date_to,
                workers=worker_count, root=snapshot_root, symbols=sorted(expected_keys),
                recognizer="boundary_first", execution_mode=execution_mode,
                recognition_config=recognition_config,
                signal_cache_dir=None,
                progress_callback=callback,
                worker_source_override=("current-rule-serial-reference" if worker_count == 1
                                        else "current-rule-spawn-candidate")))
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
        # Run one main-process reference, then each requested spawn candidate on identical inputs.
        reference = run_one("serial-reference", 1, "in_process")
        serial_rows = {(row["kind"], row["code"]): row for row in reference.get("account_summaries", [])}
        serial_accounting_ok = research._all_ledger_checks_pass(reference, len(expected_keys))
        reference_event_signature = research._signal_event_signature_sha256(reference)
        serial_quote_provenance_ok = _account_quotes_match(reference, inputs)
        serial_profile_ok = _manifest_matches_recognition_profile(reference, profile_metadata)
        serial_peak = peaks.get("serial-reference", 0)
        serial_memory_ok = (0 < serial_peak <= research.BENCHMARK_PROCESS_TREE_RSS_CAP)
        reference_ok = (reference.get("status") == "complete" and serial_accounting_ok
                        and serial_quote_provenance_ok and serial_memory_ok and serial_profile_ok)
        reference_signature = hashlib.sha256(
            research._account_ledger_signature(reference).encode("utf-8")).hexdigest()
        candidates = {}
        for worker_count in candidate_schedule:
            candidate_label = "spawn%d-candidate" % worker_count
            candidate = run_one(candidate_label, worker_count, "spawn")
            parity, divergence = research._compare_account_ledgers(reference, candidate, expected_keys)
            accounting_ok = research._all_ledger_checks_pass(candidate, len(expected_keys))
            candidate_event_signature = research._signal_event_signature_sha256(candidate)
            event_parity = reference_event_signature == candidate_event_signature
            revisions_ok = len(candidate.get("account_summaries", [])) == len(expected_keys) and all(
                row.get("provenance", {}).get("data_revision") == input_revisions[(row["kind"], row["code"])]
                and serial_rows.get((row["kind"], row["code"]), {}).get("provenance", {}).get("data_revision")
                    == input_revisions[(row["kind"], row["code"])]
                and row.get("provenance", {}).get("used_last_date") == date_to
                and serial_rows.get((row["kind"], row["code"]), {}).get("provenance", {}).get("used_last_date")
                    == date_to
                for row in candidate.get("account_summaries", []))
            candidate_quote_provenance_ok = _account_quotes_match(candidate, inputs)
            candidate_profile_ok = _manifest_matches_recognition_profile(candidate, profile_metadata)
            profile_parity = serial_profile_ok and candidate_profile_ok
            peak = peaks.get(candidate_label, 0)
            memory_ok = (0 < peak <= research.BENCHMARK_PROCESS_TREE_RSS_CAP
                         and serial_memory_ok)
            performance_ok = elapsed[candidate_label] < elapsed["serial-reference"]
            candidate_signature = hashlib.sha256(
                research._account_ledger_signature(candidate).encode("utf-8")).hexdigest()
            passed = (reference_ok and candidate.get("status") == "complete" and parity
                      and event_parity and accounting_ok and revisions_ok
                      and candidate_quote_provenance_ok and memory_ok and performance_ok
                      and candidate_profile_ok and profile_parity)
            candidates[worker_count] = {
                "result": candidate, "seconds": elapsed[candidate_label], "peak": peak,
                "passed": passed, "parity_passed": parity, "first_divergence": divergence,
                "accounting_checks_passed": accounting_ok,
                "data_revisions_match_references": revisions_ok,
                "account_quote_provenance_matches_active_manifest": candidate_quote_provenance_ok,
                "memory_status": "pass" if memory_ok else "unverified_or_over_cap",
                "performance_gate_passed": performance_ok,
                "ledger_signature_sha256": candidate_signature,
                "event_signature_sha256": candidate_event_signature,
                "event_parity_passed": event_parity,
                "recognition_profile_matches_manifest": candidate_profile_ok,
                "recognition_profile_parity_passed": profile_parity,
            }

        _assert_inputs_unchanged(reference_manifest_path, snapshot_root, expected_input_fingerprint)
        input_validation_checkpoints.append("after_all_runs")
        selected_workers, fastest_candidate_seconds, eligible_workers = _select_matrix_worker(
            candidates, elapsed["serial-reference"], research.BENCHMARK_PROCESS_TREE_RSS_CAP)
        source_hashes_after = {name: _sha256(REPO / name) for name in source_hashes}
        source_unchanged = source_hashes_after == source_hashes
        status = ("complete" if _matrix_passed(reference_ok, candidates,
                    source_unchanged=source_unchanged, selected=selected_workers) else "failed")
        selected = candidates.get(selected_workers) if status == "complete" else None
        worker_counts = {
            "1": {"seconds": elapsed["serial-reference"],
                  "process_tree_peak_bytes": serial_peak, "execution_mode": "in_process",
                  "account_count": len(reference.get("account_summaries", [])),
                  "accounting_checks_passed": serial_accounting_ok,
                  "account_quote_provenance_matches_active_manifest": serial_quote_provenance_ok,
                  "recognition_profile_matches_manifest": serial_profile_ok,
                  "recognition_profile": profile_metadata,
                  "memory_status": "pass" if serial_memory_ok else "unverified_or_over_cap",
                  "ledger_signature_sha256": reference_signature,
                  "event_signature_sha256": reference_event_signature},
        }
        for worker_count, candidate in candidates.items():
            result = candidate.pop("result")
            worker_counts[str(worker_count)] = {
                "seconds": candidate["seconds"],
                "process_tree_peak_bytes": candidate["peak"], "execution_mode": "spawn",
                "account_count": len(result.get("account_summaries", [])),
                "parity_passed": candidate["parity_passed"],
                "first_divergence": candidate["first_divergence"],
                "accounting_checks_passed": candidate["accounting_checks_passed"],
                "data_revisions_match_references": candidate["data_revisions_match_references"],
                "account_quote_provenance_matches_active_manifest": candidate[
                    "account_quote_provenance_matches_active_manifest"],
                "recognition_profile_matches_manifest": candidate["recognition_profile_matches_manifest"],
                "recognition_profile_parity_passed": candidate["recognition_profile_parity_passed"],
                "recognition_profile": profile_metadata,
                "memory_status": candidate["memory_status"],
                "ledger_signature_sha256": candidate["ledger_signature_sha256"],
                "event_signature_sha256": candidate["event_signature_sha256"],
                "event_parity_passed": candidate["event_parity_passed"],
                "performance_gate_passed": candidate["performance_gate_passed"],
                "candidate_gates_passed": candidate["passed"],
                "runner_timings": result.get("timings"),
            }
        selected_worker_row = worker_counts.get(str(selected_workers), {}) if selected else {}
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
            "quote_migration": reference_manifest.get("migration"),
            "recognition_profile": profile_metadata,
            "recognition_profile_parity_passed": serial_profile_ok and bool(selected)
                and selected_worker_row.get("recognition_profile_parity_passed") is True,
            "certificate_output": str(certificate_path),
            "snapshot_root": str(Path(snapshot_root).resolve()),
            "snapshot_id": snapshot_manifest.get("snapshot_id"),
            "snapshot_manifest_path": str((Path(snapshot_root) / "snapshot-manifest.json").resolve()),
            "paired_source_map_path": snapshot_manifest.get("paired_source_map_path"),
            "date_to": date_to,
            "versions": versions,
            "source_sha256": source_hashes,
            "source_sha256_after": source_hashes_after,
            "source_unchanged_during_run": source_unchanged,
            "input_validation_checkpoints": input_validation_checkpoints,
            "inputs": inputs,
            "matrix_requested": bool(matrix),
            "worker_counts": worker_counts,
            "recommended_workers": selected_workers if status == "complete" else None,
            "recommended_recognizer": "boundary_first" if status == "complete" else None,
            "selected_worker_count": selected_workers if status == "complete" else None,
            "selected_process_tree_peak_bytes": selected_worker_row.get("process_tree_peak_bytes"),
            "performance_gate_passed": bool(selected) and all(
                candidate["performance_gate_passed"] for candidate in candidates.values()),
            "fastest_spawn_seconds": fastest_candidate_seconds,
            "eligible_workers_within_5_percent": eligible_workers,
            "speedup_vs_serial": (elapsed["serial-reference"] / selected_worker_row["seconds"]
                                  if selected and selected_worker_row.get("seconds", 0) > 0 else None),
            "reference_ledger_signature_sha256": reference_signature,
            "candidate_ledger_signature_sha256": selected_worker_row.get("ledger_signature_sha256"),
            "reference_event_signature_sha256": reference_event_signature,
            "candidate_event_signature_sha256": selected_worker_row.get("event_signature_sha256"),
            "event_parity_passed": bool(selected) and selected_worker_row.get("event_parity_passed") is True,
            "parity_passed": bool(selected) and selected_worker_row.get("parity_passed") is True
                and reference_signature == selected_worker_row.get("ledger_signature_sha256"),
            "accounting_checks_passed": serial_accounting_ok and bool(selected)
                and selected_worker_row.get("accounting_checks_passed") is True,
            "data_revisions_match_references": bool(selected)
                and selected_worker_row.get("data_revisions_match_references") is True,
            "account_quote_provenance_matches_active_manifest": serial_quote_provenance_ok
                and bool(selected)
                and selected_worker_row.get("account_quote_provenance_matches_active_manifest") is True,
            "process_tree_memory_cap_bytes": research.BENCHMARK_PROCESS_TREE_RSS_CAP,
            "timing_scope": "each run timed from run_daily_research call through report write",
            "elapsed_seconds": elapsed,
            "error": None if status == "complete" else "one or more current-rule worker gates failed",
        }
        report["parity_passed"] = report["parity_passed"] and status == "complete"
        report_path = output_dir / "worker-policy-result.json"
        _write(report_path, report)
        if status == "complete":
            # Only the default profile updates the ordinary worker-policy certificate.
            _write(certificate_path, report)
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
                   "recognition_profile": profile_metadata,
                   "certificate_output": str(certificate_path),
                   "source_sha256": source_hashes, "elapsed_seconds": elapsed,
                   "error": "%s: %s" % (type(exc).__name__, exc)}
        _write(output_dir / "worker-policy-result.json", failure)
        _write(progress_path, {"status": "failed", "stage": "failed", "error": failure["error"]})
        raise


def run_preflight(reference_manifest_path, snapshot_root, report_path=None):
    reference_manifest, snapshot_manifest, _, inputs = _source_inputs(
        reference_manifest_path, snapshot_root)
    fingerprint = _input_fingerprint(inputs)
    _assert_inputs_unchanged(reference_manifest_path, snapshot_root, fingerprint)
    report = {
        "status": "preflight_passed",
        "benchmark_started": False,
        "reference_manifest_path": str(Path(reference_manifest_path).resolve()),
        "reference_manifest_protocol": reference_manifest.get("protocol"),
        "migration": reference_manifest.get("migration"),
        "snapshot_root": str(Path(snapshot_root).resolve()),
        "snapshot_id": snapshot_manifest.get("snapshot_id"),
        "date_to": reference_manifest.get("date_to"),
        "input_count": len(inputs),
        "input_identity": [list(row) for row in fingerprint],
        "inputs": inputs,
    }
    if report_path:
        _write(report_path, report)
    return report


def _build_parser():
    base = REPO / "reports" / "daily-research"
    benchmark_dir = base / "benchmark" / "1704b24a04344b81a8bafeb1c3c393a4"
    active_quote_manifest = (base / "benchmark" / "current-rules-36-active-quotes-v1"
                             / "active-quote-manifest.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(2, 4, 6, 8, 10), default=10,
                        help="spawn candidate worker count; serial reference always uses one")
    parser.add_argument("--matrix", action="store_true",
                        help="run spawn candidates 2/4/6/8/10 once each after one serial reference")
    parser.add_argument("--reference-manifest", type=Path, default=active_quote_manifest,
                        help="frozen current-rule active-quote manifest used by both runs")
    parser.add_argument("--source-reference-manifest", type=Path,
        default=benchmark_dir / "parity-reference-manifest.json",
        help="legacy 36-symbol scope used only as read-only migration lineage")
    parser.add_argument("--prepare-active-quotes", action="store_true",
                        help="create an immutable active-quote manifest from the source scope")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate frozen file stats and active quote settings without running accounts")
    parser.add_argument("--snapshot-root", type=Path,
        default=base / "benchmark" / "current-rules-36-input-snapshot-20261105-retry1")
    parser.add_argument("--universe", type=Path,
        default=REPO / "research" / "universes" / "daily_800_stock_200_etf_v1.json")
    parser.add_argument("--output", type=Path,
        default=base / "benchmark" / "current-rules-36-joint-selection-matrix-20261113")
    parser.add_argument("--recognition-profile", choices=("default", "g1"), default="default",
                        help="recognition semantics used by both serial and spawn runs")
    parser.add_argument("--certificate-output", type=Path,
                        help="write the completed policy certificate to a new path; required for G1")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.prepare_active_quotes:
        active = prepare_active_quote_manifest(args.source_reference_manifest,
                                               args.snapshot_root,
                                               args.reference_manifest)
        print(json.dumps({"status": "active_quote_manifest_created",
                          "path": str(args.reference_manifest.resolve()),
                          "migration": active.get("migration")},
                         ensure_ascii=False, indent=2))
        if not args.preflight_only:
            return
    if args.preflight_only:
        report_path = args.reference_manifest.parent / "preflight-report.json"
        result = run_preflight(args.reference_manifest, args.snapshot_root, report_path)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=research._json_default))
        return
    result = run_policy(args.reference_manifest, args.snapshot_root, args.universe,
                        args.output, candidate_workers=args.workers, matrix=args.matrix,
                        recognition_profile=args.recognition_profile,
                        certificate_output=args.certificate_output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=research._json_default))
    if result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
