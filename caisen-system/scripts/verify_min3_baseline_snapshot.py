#!/usr/bin/env python3
"""Match baseline-v7 event summaries to the frozen three-symbol signal lists."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("/tmp/caisen-hotpath-20261105-baseline/reference-regression-3-20261030")
PARITY_PATH = REPO / "reports" / "daily-research" / "min3-recognizer-parity-20261105.json"
EXPECTED_SIGNAL_COUNTS = {"600108": 60, "159718": 14, "516630": 9}


def _shape_key(row):
    return (row.get("pattern_type"), json.dumps(row.get("key_points", {}), sort_keys=True,
                                                 separators=(",", ":"), default=str))


def main():
    parity = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    if parity.get("status") != "complete":
        raise RuntimeError("corrected registry-tick strict parity must complete first")
    spec = importlib.util.spec_from_file_location(
        "min3_parity", REPO / "scripts" / "verify_min3_recognizer_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for case in parity["cases"]:
        code = case["code"]
        if code not in EXPECTED_SIGNAL_COUNTS:
            continue
        kind = case["kind"]
        reference_path = SNAPSHOT / "accounts" / ("%s-%s.json" % (kind, code))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_signals = reference.get("signals", [])
        reference_shapes = module._signal_shapes(reference_signals)
        result = case["baseline_vs_min3"]
        baseline_shapes = []
        for item in result.get("retained", []):
            baseline_shapes.append(item["baseline"])
        for item in result.get("moved", []):
            baseline_shapes.append(item["baseline"])
        baseline_shapes.extend(result.get("lost", []))
        # Exact shape/date/line evidence must match; signal event totals remain a
        # separate field because a same-day signal may contain multiple patterns.
        signature = lambda values: sorted(
            (json.dumps(item, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), default=str) for item in values))
        rows.append({
            "code": code,
            "kind": kind,
            "tick_size": case.get("tick_size"),
            "tick_size_source": case.get("tick_size_source"),
            "expected_snapshot_signal_count": EXPECTED_SIGNAL_COUNTS[code],
            "snapshot_signal_count": len(reference_signals),
            "baseline_shape_count": len(baseline_shapes),
            "snapshot_shape_count": len(reference_shapes),
            "snapshot_signal_count_matches": len(reference_signals) == EXPECTED_SIGNAL_COUNTS[code],
            "baseline_shapes_match_snapshot": signature(baseline_shapes) == signature(reference_shapes),
            "baseline_vs_min3_signal_count": len(reference_signals),
            "baseline_signal_count_source": "frozen snapshot list, reconciled to exact baseline shape summaries",
            "min3_signal_count": case.get("v7_signal_count"),
        })
    payload = {"status": "complete" if all(
        item["snapshot_signal_count_matches"] and item["baseline_shapes_match_snapshot"]
        for item in rows) else "diverged",
        "source": "frozen reference-regression-3-20261030 signal lists",
        "method": "reconcile flattened baseline pattern snapshots, dates, key points, and line evidence; count signal events separately",
        "cases": rows}
    output = REPO / "reports" / "daily-research" / "min3-baseline-snapshot-parity-20261105.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
