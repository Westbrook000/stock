#!/usr/bin/env python3
"""Freeze the current 1,000 source files for paired changed-symbol repairs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
RUN_ROOT = REPO / "reports" / "daily-research"
UNIVERSE = REPO / "research" / "universes" / "daily_800_stock_200_etf_v1.json"
BASELINE = RUN_ROOT / "full-cold-20261105-v8-boundary-w8-spawnfix" / "accounts"
MIN3 = RUN_ROOT / "min3-20261105-8worker" / "accounts"
SNAPSHOT = RUN_ROOT / "paired-input-snapshot-20261105-v1"
RETRIES = 3


def _read_run_revisions(directory):
    result = {}
    for path in Path(directory).glob("*.json"):
        account = json.loads(path.read_text(encoding="utf-8"))
        result[(str(account.get("kind")), str(account.get("code")))] = {
            "data_revision": account.get("provenance", {}).get("data_revision"),
            "tick_size": account.get("provenance", {}).get("tick_size"),
            "tick_size_source": account.get("provenance", {}).get("tick_size_source"),
            "first_date": account.get("provenance", {}).get("first_date"),
            "last_date": account.get("provenance", {}).get("last_date"),
            "normalized_rows": account.get("provenance", {}).get("normalized_rows"),
        }
    return result


def _stream_copy_verified(source: Path, destination: Path):
    last_error = None
    for attempt in range(1, RETRIES + 1):
        before = source.stat()
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                         dir=str(destination.parent))
        digest = hashlib.sha256()
        copied = 0
        try:
            with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            after = source.stat()
            if ((before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)
                    or copied != after.st_size):
                last_error = "source changed during copy"
                os.unlink(temporary)
                continue
            os.replace(temporary, destination)
            os.chmod(destination, 0o444)
            return {"source_stat_before": {"mtime_ns": before.st_mtime_ns,
                                           "size": before.st_size},
                    "source_stat_after": {"mtime_ns": after.st_mtime_ns,
                                          "size": after.st_size},
                    "snapshot_size": copied,
                    "sha256": digest.hexdigest(),
                    "copy_attempts": attempt}
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
    raise RuntimeError("could not freeze %s after %d attempts: %s" %
                       (source, RETRIES, last_error))


def main():
    if SNAPSHOT.exists():
        raise FileExistsError("snapshot destination already exists: %s" % SNAPSHOT)
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    source_revisions = {
        "baseline": _read_run_revisions(BASELINE),
        "min3": _read_run_revisions(MIN3),
    }
    symbols = [("stock", code) for code in universe["stocks"]]
    symbols.extend(("etf", code) for code in universe["etfs"])
    if len(symbols) != 1000:
        raise ValueError("frozen universe does not contain exactly 1,000 symbols")
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "snapshot_id": SNAPSHOT.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "date_to": "2026-10-30", "universe_id": universe.get("id"),
                "source_run_revisions": {}, "files": []}
    registry_source = REPO / "config" / "instruments.json"
    registry_destination = SNAPSHOT / "caisen-system" / "config" / "instruments.json"
    registry_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest["instrument_registry"] = {
        "source": str(registry_source.resolve()),
        "snapshot_relative_path": str(registry_destination.relative_to(SNAPSHOT)),
        **_stream_copy_verified(registry_source, registry_destination),
    }
    for label, values in source_revisions.items():
        manifest["source_run_revisions"][label] = {
            "%s:%s" % key: value for key, value in values.items()}
    for index, (kind, code) in enumerate(symbols, start=1):
        folder = "stocks" if kind == "stock" else "etfs"
        source = ROOT / "data" / folder / (code + ".csv")
        if not source.is_file():
            raise FileNotFoundError("frozen-universe source is missing: %s" % source)
        destination = SNAPSHOT / "data" / folder / (code + ".csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied = _stream_copy_verified(source, destination)
        row = {"kind": kind, "code": code,
               "snapshot_path": str(destination.resolve()),
               "snapshot_relative_path": str(destination.relative_to(SNAPSHOT)),
               **copied}
        for label, revisions in source_revisions.items():
            old = revisions.get((kind, code))
            row[label + "_source_revision"] = old
            if old:
                row[label + "_changed_since_run"] = (
                    "%s:%d,size:%d" % ("mtime_ns", copied["source_stat_after"]["mtime_ns"],
                                       copied["source_stat_after"]["size"])
                    != old.get("data_revision"))
        manifest["files"].append(row)
        if index % 100 == 0:
            print(json.dumps({"phase": "freeze", "completed": index,
                              "total": len(symbols)},
                             ensure_ascii=False), flush=True)

    # Reconcile changed sets directly from the manifest's per-run revisions.
    for row in manifest["files"]:
        key = (row["kind"], row["code"])
        for label in ("baseline", "min3"):
            old = row.get(label + "_source_revision")
            if old and row.get(label + "_changed_since_run"):
                manifest.setdefault("changed_by_run", {}).setdefault(label, []).append(
                    {"kind": key[0], "code": key[1], "old_revision": old.get("data_revision"),
                     "snapshot_sha256": row["sha256"]})
    changed_by_run = {label: len(manifest.get("changed_by_run", {}).get(label, []))
                      for label in ("baseline", "min3")}
    manifest["changed_union_count"] = len({(item["kind"], item["code"])
                                            for label in ("baseline", "min3")
                                            for item in manifest.get("changed_by_run", {}).get(label, [])})
    manifest["changed_by_run_counts"] = changed_by_run
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = SNAPSHOT / "snapshot-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    print(json.dumps({"status": "complete", "snapshot": str(SNAPSHOT.resolve()),
                      "file_count": len(manifest["files"]),
                      "changed_union_count": manifest["changed_union_count"],
                      "changed_by_run_counts": changed_by_run}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
