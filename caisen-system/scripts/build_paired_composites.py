#!/usr/bin/env python3
"""Build report-only 1,000-account composites backed by one frozen input set."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from caisen.daily_research import STARTING_CASH, _aggregate_equity_curve, _summarize  # noqa: E402

REPORTS = REPO / "reports" / "daily-research"
SNAPSHOT = REPORTS / "paired-input-snapshot-20261105-v1"
PAIR = REPORTS / "paired-repairs-20261105-snapshot-v1"
BASELINE_ORIGINAL = REPORTS / "full-cold-20261105-v8-boundary-w8-spawnfix"
MIN3_ORIGINAL = REPORTS / "min3-20261105-8worker"
OUT_ROOT = REPORTS / "paired-composites-20261105-snapshot-v1"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_variant(label, original, repaired, output):
    snapshot_manifest = _json(SNAPSHOT / "snapshot-manifest.json")
    repair_manifest = _json(PAIR / "paired-repair-manifest.json")
    if snapshot_manifest.get("status") != "complete" or repair_manifest.get("status") != "complete":
        raise RuntimeError("input snapshot and paired repairs must both be complete")
    files = {(item["kind"], str(item["code"])): item for item in snapshot_manifest["files"]}
    repaired_codes = {(item["kind"], str(item["code"])) for item in repair_manifest["paired_accounts"]}
    if output.exists():
        raise FileExistsError(output)
    account_dir = output / "accounts"
    account_dir.mkdir(parents=True)
    source_map = {"snapshot_root": str(SNAPSHOT.resolve()), "snapshot_id": snapshot_manifest["snapshot_id"],
                  "composite_variant": label, "sources": {}}
    accounts, trades, equity_events = [], [], {}
    account_rows = []
    for item in snapshot_manifest["files"]:
        kind, code = item["kind"], str(item["code"])
        key = (kind, code)
        filename = "%s-%s.json" % (kind, code)
        if key in repaired_codes:
            source_path = repaired / "accounts" / filename
            source_kind = "paired_repair"
        else:
            source_path = original / "accounts" / filename
            source_kind = "original_unchanged"
        account = _json(source_path)
        revision = account.get("provenance", {}).get("data_revision")
        source_file = SNAPSHOT / item["snapshot_relative_path"]
        stat = source_file.stat()
        snapshot_revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
        if key in repaired_codes and revision != snapshot_revision:
            raise RuntimeError("paired account/snapshot revision mismatch for %s:%s (%s != %s)" %
                               (kind, code, revision, snapshot_revision))
        if key not in repaired_codes:
            original_revision = item.get(label + "_source_revision", {}) or {}
            if revision != original_revision.get("data_revision"):
                raise RuntimeError("reused account no longer matches its frozen source record for %s:%s" %
                                   (kind, code))
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError("frozen CSV hash mismatch for %s:%s" % (kind, code))
        destination = account_dir / filename
        # Point at immutable run artifacts; the composite and source map do not rewrite either run.
        destination.symlink_to(source_path.resolve())
        accounts.append(account)
        trades.extend(dict(trade, code=code, kind=kind) for trade in account.get("trades", []))
        _aggregate_equity_curve.add_account(equity_events, STARTING_CASH, account.get("daily_equity", []))
        source_map["sources"]["%s:%s" % key] = {
            "snapshot_path": str(source_file.resolve()),
            "snapshot_root": str(SNAPSHOT.resolve()),
            "snapshot_relative_path": item["snapshot_relative_path"],
            "snapshot_data_revision": snapshot_revision,
            "snapshot_sha256": item["sha256"],
            "account_data_revision": revision,
            "account_path": str(destination.resolve()),
            "account_source_kind": source_kind,
            "original_source_revision": item.get(label + "_source_revision"),
        }
        account_rows.append({"kind": kind, "code": code, "status": account.get("status"),
                             "signal_count": len(account.get("signals", [])),
                             "trade_count": len(account.get("trades", [])),
                             "provenance": account.get("provenance"),
                             "account_json": str(destination.resolve()),
                             "account_json_relative": "accounts/" + filename})
    curve = _aggregate_equity_curve.build(equity_events, len(accounts) * STARTING_CASH)
    ending = sum(float(account.get("ending_equity", STARTING_CASH)) for account in accounts)
    summary_payload = _summarize(accounts, trades, len(accounts) * STARTING_CASH, ending, curve)
    if curve and abs(float(curve[-1]["equity"]) - ending) > 0.01:
        raise RuntimeError("hybrid aggregate equity curve does not match ending equity")
    source_map_path = output / "data-source-map.json"
    _write(source_map_path, source_map)
    original_manifest = _json(original / "run-manifest.json")
    manifest = {**{key: original_manifest.get(key) for key in
                   ("strategy_version", "recognizer_version", "engine_version", "source_rule_version",
                    "recognizer", "parameters", "universe")},
                "run_id": "paired-%s-%s" % (label, snapshot_manifest["snapshot_id"]),
                "status": "complete", "composite": True, "variant": label,
                "source_policy": "original archives preserved; changed accounts replaced with paired reruns from one frozen snapshot",
                "snapshot_id": snapshot_manifest["snapshot_id"],
                "snapshot_manifest": str((SNAPSHOT / "snapshot-manifest.json").resolve()),
                "paired_repair_manifest": str((PAIR / "paired-repair-manifest.json").resolve()),
                "account_directory": str(account_dir.resolve()), "output_dir": str(output.resolve()),
                "completed": len(accounts), "total": len(accounts),
                "summary": summary_payload["summary"], "groups": summary_payload["groups"],
                "account_summaries": account_rows,
                "data_source_map": str(source_map_path.resolve()),
                "repaired_account_count": len(repaired_codes), "reused_account_count": len(accounts)-len(repaired_codes)}
    _write(output / "run-manifest.json", manifest)
    print(json.dumps({"variant": label, "accounts": len(accounts), "trades": len(trades),
                      "summary": summary_payload["summary"], "output": str(output.resolve()),
                      "data_source_map": str(source_map_path.resolve())}, ensure_ascii=False))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    build_variant("baseline", BASELINE_ORIGINAL, PAIR / "baseline-changed-52", OUT_ROOT / "baseline")
    build_variant("min3", MIN3_ORIGINAL, PAIR / "min3-changed-52", OUT_ROOT / "min3")


if __name__ == "__main__":
    main()
