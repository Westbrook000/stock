#!/usr/bin/env python3
"""Run the opt-in causal min-three-neckline experiment on the frozen universe."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import UNIVERSE_ID, run_daily_research  # noqa: E402
from caisen.fixed_backtest import EXPERIMENT_PATTERN_TYPES  # noqa: E402
from caisen.research_config import RecognitionExperimentConfig  # noqa: E402


UNIVERSE_PATH = REPO / "research" / "universes" / (UNIVERSE_ID + ".json")
OUTPUT_PATH = REPO / "reports" / "daily-research" / "min3-20261105-8worker"
CACHE_PATH = REPO / "reports" / "daily-research" / "signal-cache-min3-v1"
DATE_TO = "2026-10-30"


def main() -> None:
    if not UNIVERSE_PATH.is_file():
        raise SystemExit("frozen universe is missing: %s" % UNIVERSE_PATH)
    if (OUTPUT_PATH / "run-manifest.json").exists():
        raise SystemExit("min3 output already exists: %s" % OUTPUT_PATH)

    config = RecognitionExperimentConfig(
        experiment_id="neckline_min3_v1",
        minimum_neckline_tests=3,
        admission_types=frozenset(EXPERIMENT_PATTERN_TYPES),
    )
    last_print = [0.0]

    def progress(snapshot):
        now = time.monotonic()
        if now - last_print[0] >= 2.0:
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            last_print[0] = now

    payload = run_daily_research(
        UNIVERSE_PATH,
        OUTPUT_PATH,
        date_to=DATE_TO,
        workers=8,
        progress_callback=progress,
        root=REPO.parent,
        recognizer="boundary_first",
        execution_mode="spawn",
        signal_cache_dir=CACHE_PATH,
        recognition_config=config,
    )
    print(json.dumps({
        "status": payload["status"],
        "summary": payload["summary"],
        "artifacts": payload["artifacts"],
        "timings": payload["timings"],
        "recognition_config": config.cache_key(),
        "output": str(OUTPUT_PATH.resolve()),
        "signal_cache": str(CACHE_PATH.resolve()),
    }, ensure_ascii=False, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
