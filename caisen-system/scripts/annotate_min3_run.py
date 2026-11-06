#!/usr/bin/env python3
"""Attach explicit min3 provenance to a completed min3 run, idempotently."""

from __future__ import annotations

import json
import html
import os
import tempfile
import time
import argparse
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "reports" / "daily-research" / "min3-20261105-8worker"
EXPERIMENT = {
    "experiment_id": "neckline_min3_v1",
    "minimum_neckline_tests": 3,
    "admission_types": ["bull_flag", "inverse_head_shoulders",
                        "triangle_bottom", "w_bottom"],
}
EXPERIMENT["cache_key"] = [EXPERIMENT["experiment_id"],
                            EXPERIMENT["minimum_neckline_tests"],
                            EXPERIMENT["admission_types"]]


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _attach(record, expected, key):
    existing = record.get(key)
    if existing is not None and existing != expected:
        raise ValueError("conflicting experiment provenance at %s" % key)
    if existing == expected:
        return False
    record[key] = expected
    return True


def annotate_run(run_path: Path) -> dict:
    run_path = Path(run_path).expanduser().resolve()
    manifest_path = run_path / "run-manifest.json"
    report_path = run_path / "daily-research.json"
    progress_path = run_path / "progress.json"
    manifest, report, progress = [json.loads(path.read_text(encoding="utf-8"))
                                  for path in (manifest_path, report_path, progress_path)]
    if manifest.get("status") != "complete" or progress.get("status") != "complete":
        raise RuntimeError("min3 run must be complete before provenance annotation")
    config = manifest.get("parameters", {}).get("recognition_config")
    if config != EXPERIMENT["cache_key"]:
        raise ValueError("run manifest has unexpected min3 recognition config: %r" % (config,))

    started = time.perf_counter()
    account_files = sorted((run_path / "accounts").glob("*.json"))
    mapping = []
    account_count = signal_count = updated_accounts = updated_signals = 0
    for path in account_files:
        account = json.loads(path.read_text(encoding="utf-8"))
        if account.get("status") not in {"complete", "no_data", "missing_inventory_file",
                                         "no_bars_before_cutoff"}:
            raise RuntimeError("unexpected account status in %s: %s" %
                               (path.name, account.get("status")))
        changed = _attach(account, EXPERIMENT, "recognition_experiment")
        account_count += 1
        updated_accounts += int(changed)
        signals = account.get("signals", [])
        changed_any = changed
        for signal in signals:
            signal_changed = _attach(signal, EXPERIMENT, "recognition_experiment")
            updated_signals += int(signal_changed)
            changed_any = changed_any or signal_changed
        signal_count += len(signals)
        if changed_any:
            _atomic_json(path, account)
        mapping.append({"account_file": path.name, "status": account.get("status"),
                        "recognition_experiment_id": EXPERIMENT["experiment_id"],
                        "signal_count": len(signals)})

    if account_count != int(manifest.get("total", 0)):
        raise RuntimeError("annotated account file count does not match run manifest")
    for row in report.get("account_summaries", []):
        row["recognition_experiment_id"] = EXPERIMENT["experiment_id"]
        row["recognition_config"] = EXPERIMENT
    report["recognition_experiment"] = EXPERIMENT
    report["experiment_provenance_path"] = "experiment-provenance.json"
    _atomic_json(report_path, report)

    elapsed = round(time.perf_counter() - started, 3)
    provenance = {
        "status": "complete",
        "run_id": manifest.get("run_id"),
        "recognition_experiment": EXPERIMENT,
        "annotation_scope": "min3 run accounts and emitted signal dictionaries only",
        "baseline_archive_modified": False,
        "account_count": account_count,
        "signal_count": signal_count,
        "newly_annotated_accounts": updated_accounts,
        "newly_annotated_signals": updated_signals,
        "postprocessing_elapsed_seconds": elapsed,
        "account_mapping": mapping,
    }
    sidecar_path = run_path / "experiment-provenance.json"
    _atomic_json(sidecar_path, provenance)
    html_path = run_path / "daily-research.html"
    if html_path.is_file():
        page = html_path.read_text(encoding="utf-8")
        if "experiment-provenance.json" not in page:
            detail = ("<section><h2>Recognition experiment</h2><p>Experiment: <code>%s</code>; "
                      "minimum independent neckline tests: 3; four pattern types enabled.</p>"
                      "<p><a href=\"experiment-provenance.json\">Per-account and per-signal provenance</a></p>"
                      "<pre>%s</pre></section>" %
                      (html.escape(EXPERIMENT["experiment_id"]),
                       html.escape(json.dumps(EXPERIMENT, ensure_ascii=False, indent=2))))
            if "</main>" in page:
                page = page.replace("</main>", detail + "</main>", 1)
            else:
                page += detail
            html_path.write_text(page, encoding="utf-8")
    manifest["postprocessing"] = {"experiment_provenance_path": str(sidecar_path.resolve()),
                                  "postprocessing_elapsed_seconds": elapsed,
                                  "annotated_account_count": account_count,
                                  "annotated_signal_count": signal_count}
    _atomic_json(manifest_path, manifest)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN,
                        help="completed min3 run directory to annotate")
    args = parser.parse_args()
    provenance = annotate_run(args.run)
    print(json.dumps({key: provenance[key] for key in (
        "status", "run_id", "account_count", "signal_count",
        "newly_annotated_accounts", "newly_annotated_signals",
        "postprocessing_elapsed_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
