#!/usr/bin/env python3
"""Freeze the exact 36 benchmark CSV inputs and their effective quote settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
STOCK_ROOT = REPO.parent
DAILY = REPO / "reports" / "daily-research"
REFERENCE_MANIFEST = DAILY / "benchmark" / "1704b24a04344b81a8bafeb1c3c393a4" / "parity-reference-manifest.json"
PAIRED_SOURCE_MAP = (DAILY / "paired-composites-20261105-snapshot-v1" / "baseline" /
                     "data-source-map.json")
PAIRED_SNAPSHOT_ROOT = DAILY / "paired-input-snapshot-20261105-v1"
DEFAULT_OUTPUT = DAILY / "benchmark" / "current-rules-36-input-snapshot-20261105-retry1"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path):
    stat = Path(path).stat()
    return "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)


def _copy_frozen(source: Path, destination: Path):
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=str(destination.parent))
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if ((before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)
                or size != after.st_size):
            raise RuntimeError("input changed while being copied: %s" % source)
        os.replace(temporary, destination)
        os.utime(destination, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.chmod(destination, 0o444)
        return {"source_path": str(source.resolve()),
                "source_revision": "mtime_ns:%d,size:%d" % (before.st_mtime_ns, before.st_size),
                "snapshot_path": str(destination.resolve()),
                "snapshot_data_revision": _revision(destination),
                "sha256": digest.hexdigest(), "size": size}
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def freeze(reference_manifest_path=REFERENCE_MANIFEST, source_map_path=PAIRED_SOURCE_MAP,
           paired_snapshot_root=PAIRED_SNAPSHOT_ROOT, output_root=DEFAULT_OUTPUT):
    reference_manifest_path = Path(reference_manifest_path).resolve()
    source_map_path = Path(source_map_path).resolve()
    paired_snapshot_root = Path(paired_snapshot_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("current-rule 36 input snapshot already exists: %s" % output_root)
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    references = reference_manifest.get("references", [])
    expected = {(row.get("kind"), row.get("code")) for row in references}
    if reference_manifest.get("status") != "validated" or reference_manifest.get("date_to") != "2026-10-30" or len(expected) != 36:
        raise ValueError("source reference must be the validated 36-account cutoff 2026-10-30 manifest")
    output_root.mkdir(parents=True)

    paired_registry = paired_snapshot_root / "caisen-system" / "config" / "instruments.json"
    if not paired_registry.is_file():
        raise FileNotFoundError("paired frozen instrument registry is missing")
    registry_result = _copy_frozen(paired_registry, output_root / "caisen-system" / "config" / "instruments.json")
    os.environ["CAISEN_INSTRUMENTS"] = registry_result["snapshot_path"]

    files = []
    for kind, code in sorted(expected):
        reference = next(row for row in references
                         if (row.get("kind"), row.get("code")) == (kind, code))
        account_path = Path(reference["account_path"]).resolve()
        account = json.loads(account_path.read_text(encoding="utf-8"))
        quote = account.get("provenance", {})
        source_record = source_map.get("sources", {}).get("%s:%s" % (kind, code))
        if source_record:
            source_path = Path(source_record["snapshot_path"]).resolve()
            source_kind = "paired_snapshot"
            if _hash(source_path) != source_record.get("snapshot_sha256"):
                raise ValueError("paired snapshot source hash mismatch for %s:%s" % (kind, code))
        else:
            folder = "stocks" if kind == "stock" else "etfs"
            source_path = STOCK_ROOT / "data" / folder / (code + ".csv")
            source_kind = "revision_matched_local_regression_input"
            expected_revision = quote.get("data_revision")
            if _revision(source_path) != expected_revision:
                raise ValueError("regression source revision differs from pinned reference for %s:%s" %
                                 (kind, code))
        folder = "stocks" if kind == "stock" else "etfs"
        copied = _copy_frozen(source_path, output_root / "data" / folder / (code + ".csv"))
        row = {"kind": kind, "code": code, "source_kind": source_kind,
               "reference_source": reference.get("source"),
               "reference_account_path": str(account_path),
               "reference_data_revision": quote.get("data_revision"),
               "tick_size": quote.get("tick_size"),
               "tick_size_source": quote.get("tick_size_source"),
               **copied}
        files.append(row)

    # Verify every copied input with the regular data loader and pinned quote settings.
    from caisen.daily_research import load_research_bars
    from caisen.market import discover_instruments
    instruments = {(item.kind, item.code): item for item in discover_instruments(output_root)}
    for row in files:
        loaded = load_research_bars(instruments[(row["kind"], row["code"])], root=output_root)
        if (loaded["data_revision"] != row["snapshot_data_revision"]
                or loaded["tick_size"] != row["tick_size"]
                or loaded["tick_size_source"] != row["tick_size_source"]):
            raise ValueError("frozen loader/quote validation failed for %s:%s" % (row["kind"], row["code"]))
        row["loaded_rows"] = loaded["normalized_rows"]

    manifest = {"status": "complete", "snapshot_id": output_root.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "date_to": reference_manifest["date_to"], "account_count": len(files),
                "reference_manifest_path": str(reference_manifest_path),
                "paired_source_map_path": str(source_map_path),
                "paired_snapshot_root": str(paired_snapshot_root),
                "instrument_registry": registry_result, "files": files}
    manifest_path = output_root / "snapshot-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--source-map", type=Path, default=PAIRED_SOURCE_MAP)
    parser.add_argument("--paired-snapshot-root", type=Path, default=PAIRED_SNAPSHOT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = freeze(args.reference_manifest, args.source_map, args.paired_snapshot_root, args.output)
    print(json.dumps({"status": result["status"], "snapshot_id": result["snapshot_id"],
                      "snapshot_root": str(args.output.resolve()), "account_count": result["account_count"],
                      "source_kinds": {kind: sum(row["source_kind"] == kind for row in result["files"])
                                       for kind in sorted({row["source_kind"] for row in result["files"]})}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
