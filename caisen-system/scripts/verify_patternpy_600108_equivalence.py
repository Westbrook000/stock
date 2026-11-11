#!/usr/bin/env python3
"""Independently compare report labels with pinned PatternPy outputs."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


PINNED_COMMIT = "2b1f29b17986acafffeb32c83e772417d540c100"


def load_source(root):
    path = root / "tradingpatterns" / "tradingpatterns.py"
    spec = importlib.util.spec_from_file_location("patternpy_audit_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patternpy-root", type=Path, default=Path(__file__).parents[1] / "state/external/PatternPy-2b1f29b")
    parser.add_argument("--csv", type=Path, default=Path("data/stocks/600108.csv"))
    parser.add_argument("--report-json", type=Path, default=Path("caisen-system/reports/patternpy-600108-full-history.json"))
    parser.add_argument("--output", type=Path, default=Path("caisen-system/reports/patternpy-600108-equivalence-audit.json"))
    args = parser.parse_args()

    commit = subprocess.run(["git", "-C", str(args.patternpy_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"Expected source commit {PINNED_COMMIT}, found {commit}")
    subprocess.run(["git", "-C", str(args.patternpy_root), "diff", "--quiet", commit, "--", "tradingpatterns/tradingpatterns.py", "LICENSE.md"], check=True)
    source, source_path = load_source(args.patternpy_root)
    raw = pd.read_csv(args.csv, encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.strftime("%Y-%m-%d")
    raw = raw.sort_values("date", kind="stable").reset_index(drop=True)
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    expected = {(event["date"], event["label"], event["window"]) for event in report["events"]}
    native_labels = set()
    counts = Counter()
    for window in report["metadata"]["windows"]:
        native = raw.rename(columns={"high": "High", "low": "Low", "close": "Close"})
        native = source.detect_double_top_bottom(native.copy(), window=window, threshold=0.05)
        native = source.detect_head_shoulder(native, window=window)
        for col, label in (("double_pattern", "Double Bottom"), ("head_shoulder_pattern", "Inverse Head and Shoulder")):
            indices = native.index[native[col] == label].tolist()
            counts[f"{window}:{label}"] = len(indices)
            native_labels.update((raw.at[i, "date"], label, window) for i in indices if i + 1 < len(raw))
    missing = sorted(native_labels - expected)
    extra = sorted(expected - native_labels)
    result = {
        "match": not missing and not extra,
        "patternpy_commit": commit,
        "patternpy_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "rows": len(raw), "start": raw.iloc[0]["date"], "end": raw.iloc[-1]["date"],
        "report_events": len(expected), "upstream_events": len(native_labels),
        "counts_by_window_and_label": dict(counts),
        "missing_from_report": missing, "extra_in_report": extra,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
