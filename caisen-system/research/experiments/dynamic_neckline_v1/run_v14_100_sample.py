#!/usr/bin/env python3
"""Replay the fixed v12 100-stock cohort with the current recognizer and rules."""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from caisen.daily_research import run_daily_research  # noqa: E402


REPORT_ROOT = REPO / "reports" / "daily-research"
SOURCE_SAMPLE = REPORT_ROOT / "full-unified-v12-20261106-w4-rerun" / "partial-100-report" / "sample-selection.json"
DATA_SNAPSHOT = REPORT_ROOT / "paired-input-snapshot-20261105-v1"
OUTPUT = REPORT_ROOT / "v14-suppression-neckline-revision-100-20261110-attempt02"
CUTOFF = "2026-10-30"


def main() -> None:
    if (OUTPUT / "run-manifest.json").exists():
        raise FileExistsError(OUTPUT / "run-manifest.json")
    selection = json.loads(SOURCE_SAMPLE.read_text(encoding="utf-8"))
    stocks = sorted(item["code"] for item in selection["symbols"] if item["kind"] == "stock")
    etfs = sorted(item["code"] for item in selection["symbols"] if item["kind"] == "etf")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    universe = {
        "id": "paired-100-v12-fixed-cohort-v1",
        "source_universe_id": selection["source_universe_id"],
        "selection_rule": selection["selection_rule"],
        "date": "2026-11-06",
        "counts": {"stock": len(stocks), "etf": len(etfs), "total": len(stocks) + len(etfs)},
        "stocks": stocks,
        "etfs": etfs,
    }
    universe_path = OUTPUT / "sample-universe.json"
    universe_path.write_text(json.dumps(universe, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"phase": "start", "universe": universe["counts"],
                      "cutoff": CUTOFF, "workers": 8, "recognizer": "boundary_first",
                      "data_snapshot": str(DATA_SNAPSHOT), "output": str(OUTPUT)},
                     ensure_ascii=False), flush=True)

    last_report = [0.0]

    def progress(snapshot):
        now = time.monotonic()
        if now - last_report[0] >= 5.0:
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            last_report[0] = now

    payload = run_daily_research(
        universe_path,
        OUTPUT,
        date_to=CUTOFF,
        workers=8,
        progress_callback=progress,
        root=DATA_SNAPSHOT,
        allow_test_universe=True,
        recognizer="boundary_first",
        execution_mode="spawn",
        signal_cache_dir=None,
        require_signal_cache=False,
    )
    print(json.dumps({"phase": "complete", "status": payload.get("status"),
                      "summary": payload.get("summary"), "artifacts": payload.get("artifacts"),
                      "timings": payload.get("timings"), "engine_version": payload.get("engine_version"),
                      "source_rule_version": payload.get("source_rule_version")},
                     ensure_ascii=False, default=str, indent=2), flush=True)


if __name__ == "__main__":
    main()
