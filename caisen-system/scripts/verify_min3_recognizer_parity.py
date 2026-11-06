#!/usr/bin/env python3
"""Compare v7 and boundary-first signals under the strict min-three experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import load_research_bars  # noqa: E402
from caisen.daily_incremental import generate_daily_signals  # noqa: E402
from caisen.fixed_backtest import EXPERIMENT_PATTERN_TYPES  # noqa: E402
from caisen.market import InstrumentFile  # noqa: E402
from caisen.research_config import RecognitionExperimentConfig, recognition_experiment  # noqa: E402


ROOT = REPO.parent
PINNED_CASES = (
    ("600108", "stock", "data"),
    ("159718", "etf", "data"),
    ("516630", "etf", "data"),
    ("000008", "stock", "negative_fixture"),
)
REFERENCE_BASELINE_SIGNAL_COUNTS = {"600108": 60, "159718": 14, "516630": 9}


def _source_path(code: str, kind: str, source: str) -> Path:
    if source == "negative_fixture":
        return REPO / "tests" / "fixtures" / "main_bottoms" / "R02-000008.csv"
    folder = "stocks" if kind == "stock" else "etfs"
    return ROOT / "data" / folder / (code + ".csv")


def _digest(signals) -> str:
    encoded = json.dumps(signals, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "v7_signal_date": a.get("signal_date"),
                    "boundary_first_signal_date": b.get("signal_date"),
                    "v7_pattern_ids": a.get("pattern_ids"),
                    "boundary_first_pattern_ids": b.get("pattern_ids")}
    if len(left) != len(right):
        longer = left if len(left) > len(right) else right
        return {"index": min(len(left), len(right)),
                "extra_signal": longer[min(len(left), len(right))]}
    return None


def _signal_shapes(signals):
    rows = []
    for signal in signals:
        for pattern in signal.get("patterns", []):
            evidence = {item.get("code"): item.get("value")
                        for item in pattern.get("evidence", [])}
            tests = evidence.get("neckline_tests") or []
            selection = evidence.get("neckline_selection") or {}
            rows.append({
                "pattern_id": pattern.get("pattern_id"),
                "pattern_type": pattern.get("pattern_type"),
                "signal_date": signal.get("signal_date"),
                "key_points": pattern.get("key_points", {}),
                "selected_main": (evidence.get("main_selection") or {}).get("selected"),
                "primary_pattern_id": (evidence.get("main_selection") or {}).get("primary_pattern_id"),
                "independent_line_test_count": len(tests),
                "line_test_dates": [item.get("pivot_date") for item in tests],
                "selected_line": selection.get("selected_line"),
                "selected_candidate_id": selection.get("selected_candidate_id"),
                "structure_known_at": selection.get("structure_known_at"),
                "line_frozen_at": selection.get("line_frozen_at"),
            })
    return rows


def _structure_key(row):
    return (row.get("pattern_type"), json.dumps(row.get("key_points", {}), sort_keys=True,
                                                 separators=(",", ":"), default=str))


def _baseline_vs_min3(baseline, strict):
    baseline_rows, strict_rows = _signal_shapes(baseline), _signal_shapes(strict)
    baseline_by_key = {_structure_key(row): row for row in baseline_rows}
    strict_by_key = {_structure_key(row): row for row in strict_rows}
    retained, moved, lost, added = [], [], [], []
    for key, before in baseline_by_key.items():
        after = strict_by_key.get(key)
        if after is None:
            lost.append(before)
        elif before["signal_date"] == after["signal_date"]:
            retained.append({"baseline": before, "min3": after})
        else:
            moved.append({"baseline": before, "min3": after})
    for key, after in strict_by_key.items():
        if key not in baseline_by_key:
            added.append(after)
    return {"baseline_confirmed_pattern_count": len(baseline_rows),
            "min3_confirmed_pattern_count": len(strict_rows),
            "retained": retained, "moved": moved, "lost": lost, "added": added}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "reports" / "daily-research" /
                        "min3-recognizer-parity-20261105.json")
    args = parser.parse_args()
    config = RecognitionExperimentConfig(
        experiment_id="neckline_min3_v1",
        minimum_neckline_tests=3,
        admission_types=frozenset(EXPERIMENT_PATTERN_TYPES),
    )
    results = []
    started = time.perf_counter()
    for code, kind, source in PINNED_CASES:
        path = _source_path(code, kind, source)
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded = load_research_bars(InstrumentFile(code, kind, str(path)), root=ROOT)
        bars = loaded["bars"]
        bars = bars.loc[bars.date <= "2026-10-30"].reset_index(drop=True)
        per_engine = {}
        per_engine["baseline_v7"] = {}
        performance = {}
        baseline = generate_daily_signals(bars, code, kind, recognizer="v7",
                                          performance=performance)
        per_engine["baseline_v7"] = {"signals": baseline, "performance": performance,
                                     "sha256": _digest(baseline)}
        expected_baseline_count = REFERENCE_BASELINE_SIGNAL_COUNTS.get(code)
        for recognizer in ("v7", "boundary_first"):
            performance = {}
            with recognition_experiment(config):
                signals = generate_daily_signals(
                    bars, code, kind, recognizer=recognizer,
                    performance=performance,
                )
            per_engine[recognizer] = {"signals": signals, "performance": performance,
                                      "sha256": _digest(signals)}
        left = per_engine["v7"]["signals"]
        right = per_engine["boundary_first"]["signals"]
        results.append({
            "code": code,
            "kind": kind,
            "source": source,
            "rows": len(bars),
            "last_date": str(bars.date.max().date()) if len(bars) else None,
            "tick_size": loaded.get("tick_size"),
            "tick_size_source": loaded.get("tick_size_source"),
            "expected_baseline_signal_count": expected_baseline_count,
            "baseline_count_matches_snapshot": (None if expected_baseline_count is None
                                                  else len(baseline) == expected_baseline_count),
            "baseline_signal_count": len(baseline),
            "v7_signal_count": len(left),
            "boundary_first_signal_count": len(right),
            "signals_equal": left == right,
            "first_difference": _first_difference(left, right),
            "v7_sha256": per_engine["v7"]["sha256"],
            "boundary_first_sha256": per_engine["boundary_first"]["sha256"],
            "v7_performance": per_engine["v7"]["performance"],
            "boundary_first_performance": per_engine["boundary_first"]["performance"],
            "baseline_vs_min3": _baseline_vs_min3(
                per_engine["baseline_v7"]["signals"], left),
        })
        del bars, per_engine
    payload = {
        "status": ("complete" if all(item["signals_equal"] and
                                     item["baseline_count_matches_snapshot"] is not False
                                     for item in results) else "diverged"),
        "experiment_config": config.cache_key(),
        "date_to": "2026-10-30",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve()),
                      "elapsed_seconds": payload["elapsed_seconds"],
                      "cases": [{key: item[key] for key in
                                 ("code", "signals_equal", "v7_signal_count",
                                  "boundary_first_signal_count", "baseline_signal_count",
                                  "expected_baseline_signal_count", "baseline_count_matches_snapshot",
                                  "tick_size", "tick_size_source", "first_difference")}
                                for item in results]}, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
