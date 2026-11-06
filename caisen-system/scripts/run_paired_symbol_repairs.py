#!/usr/bin/env python3
"""Rerun changed symbols in baseline and min3 against one frozen input snapshot."""

from __future__ import annotations

import json
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import UNIVERSE_ID, run_daily_research  # noqa: E402
from caisen.fixed_backtest import EXPERIMENT_PATTERN_TYPES  # noqa: E402
from caisen.research_config import RecognitionExperimentConfig  # noqa: E402
_annotator_spec = importlib.util.spec_from_file_location(
    "min3_annotator", REPO / "scripts" / "annotate_min3_run.py")
_annotator = importlib.util.module_from_spec(_annotator_spec)
_annotator_spec.loader.exec_module(_annotator)
annotate_run = _annotator.annotate_run


REPORT_ROOT = REPO / "reports" / "daily-research"
SNAPSHOT = REPORT_ROOT / "paired-input-snapshot-20261105-v1"
UNIVERSE = REPO / "research" / "universes" / (UNIVERSE_ID + ".json")
PAIR_ROOT = REPORT_ROOT / "paired-repairs-20261105-snapshot-v1"
BASELINE_OUT = PAIR_ROOT / "baseline-changed-52"
MIN3_OUT = PAIR_ROOT / "min3-changed-52"
DATE_TO = "2026-10-30"


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _run(name, output, symbols, recognition_config):
    last = [0.0]

    def progress(snapshot):
        now = time.monotonic()
        if now - last[0] >= 10.0:
            print(json.dumps({"repair": name, **snapshot}, ensure_ascii=False), flush=True)
            last[0] = now

    return run_daily_research(
        UNIVERSE,
        output,
        date_to=DATE_TO,
        workers=8,
        progress_callback=progress,
        root=SNAPSHOT,
        symbols=symbols,
        recognizer="boundary_first",
        execution_mode="spawn",
        signal_cache_dir=None,
        recognition_config=recognition_config,
    )


def main():
    freeze_path = SNAPSHOT / "snapshot-manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "complete" or len(frozen.get("files", [])) != 1000:
        raise RuntimeError("paired source snapshot is incomplete")
    changed = {}
    for label in ("baseline", "min3"):
        changed[label] = {(item["kind"], item["code"])
                          for item in frozen.get("changed_by_run", {}).get(label, [])}
    symbols = sorted(changed["baseline"] | changed["min3"])
    if len(symbols) != int(frozen.get("changed_union_count", -1)):
        raise RuntimeError("changed-symbol union does not match frozen manifest")
    if not symbols:
        raise RuntimeError("no changed symbols were found; no paired repair is needed")
    for output in (BASELINE_OUT, MIN3_OUT):
        if (output / "run-manifest.json").exists():
            raise FileExistsError("paired repair output already exists: %s" % output)

    PAIR_ROOT.mkdir(parents=True, exist_ok=True)
    config = RecognitionExperimentConfig(
        experiment_id="neckline_min3_v1",
        minimum_neckline_tests=3,
        admission_types=frozenset(EXPERIMENT_PATTERN_TYPES),
    )
    plan = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "date_to": DATE_TO, "workers": 8, "execution_mode": "spawn",
        "recognizer": "boundary_first", "signal_cache_used": False,
        "snapshot_path": str(SNAPSHOT.resolve()),
        "snapshot_manifest": str(freeze_path.resolve()),
        "changed_by_run_counts": {key: len(value) for key, value in changed.items()},
        "changed_union_count": len(symbols),
        "changed_symbols": [{"kind": kind, "code": code} for kind, code in symbols],
        "min3_recognition_config": config.cache_key(),
        "baseline_output": str(BASELINE_OUT.resolve()),
        "min3_output": str(MIN3_OUT.resolve()),
    }
    _write_json(PAIR_ROOT / "paired-repair-manifest.json", plan)
    started = time.perf_counter()
    baseline = _run("baseline", BASELINE_OUT, symbols, recognition_config=None)
    if baseline.get("status") != "complete" or baseline.get("summary", {}).get("account_count") != len(symbols):
        raise RuntimeError("baseline paired correction did not complete all changed accounts")
    baseline_elapsed = time.perf_counter() - started

    min3_started = time.perf_counter()
    min3 = _run("min3", MIN3_OUT, symbols, recognition_config=config)
    if min3.get("status") != "complete" or min3.get("summary", {}).get("account_count") != len(symbols):
        raise RuntimeError("min3 paired correction did not complete all changed accounts")
    min3_elapsed = time.perf_counter() - min3_started
    annotation = annotate_run(MIN3_OUT)

    account_map = []
    for kind, code in symbols:
        account_name = "%s-%s.json" % (kind, code)
        base_account = json.loads((BASELINE_OUT / "accounts" / account_name).read_text(encoding="utf-8"))
        strict_account = json.loads((MIN3_OUT / "accounts" / account_name).read_text(encoding="utf-8"))
        if base_account.get("provenance", {}).get("data_revision") != strict_account.get(
                "provenance", {}).get("data_revision"):
            raise RuntimeError("paired input revision differs between variants: %s:%s" % (kind, code))
        account_map.append({
            "kind": kind, "code": code,
            "snapshot_sha256": next(item["sha256"] for item in frozen["files"]
                                     if item["kind"] == kind and item["code"] == code),
            "snapshot_relative_path": "data/%s/%s.csv" % (
                "stocks" if kind == "stock" else "etfs", code),
            "original_baseline_revision": next(item.get("baseline_source_revision")
                                                for item in frozen["files"]
                                                if item["kind"] == kind and item["code"] == code),
            "original_min3_revision": next(item.get("min3_source_revision")
                                            for item in frozen["files"]
                                            if item["kind"] == kind and item["code"] == code),
            "paired_data_revision": base_account.get("provenance", {}).get("data_revision"),
            "baseline_account": str((BASELINE_OUT / "accounts" / account_name).resolve()),
            "min3_account": str((MIN3_OUT / "accounts" / account_name).resolve()),
            "min3_experiment_id": config.experiment_id,
        })
    plan.update({
        "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": baseline.get("run_id"),
        "min3_run_id": min3.get("run_id"),
        "baseline_elapsed_seconds": round(baseline_elapsed, 3),
        "min3_elapsed_seconds": round(min3_elapsed, 3),
        "min3_annotation": {key: annotation.get(key) for key in
                             ("account_count", "signal_count", "postprocessing_elapsed_seconds")},
        "paired_account_count": len(account_map),
        "paired_accounts": account_map,
        "source_policy": "original archives remain intact; unchanged account artifacts are reused; this pair replaces only the changed-symbol account artifacts in the report composite",
    })
    _write_json(PAIR_ROOT / "paired-repair-manifest.json", plan)
    print(json.dumps({"status": plan["status"], "changed_union_count": len(symbols),
                      "baseline_run_id": plan["baseline_run_id"],
                      "min3_run_id": plan["min3_run_id"],
                      "baseline_elapsed_seconds": plan["baseline_elapsed_seconds"],
                      "min3_elapsed_seconds": plan["min3_elapsed_seconds"],
                      "manifest": str((PAIR_ROOT / "paired-repair-manifest.json").resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
