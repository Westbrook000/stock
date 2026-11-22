#!/usr/bin/env python3
"""Freeze immutable market inputs for structure/boundary paired studies.

The full study is deliberately driven from the ordinary daily-research entry
point. This helper only freezes membership and market-file revisions so all
recognizer versions read the same inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _revision(path: Path):
    stat = path.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size_bytes": int(stat.st_size)}


def _symbol_rows(universe_path: Path, paired_reference_path: Path | None = None):
    universe = _read_json(universe_path)
    rows = {("stock", str(code)) for code in universe.get("stocks", [])}
    rows.update(("etf", str(code)) for code in universe.get("etfs", []))
    if paired_reference_path is not None:
        reference = _read_json(paired_reference_path)
        rows.update((str(row["kind"]), str(row["code"]))
                    for row in reference.get("account_summaries", []))
    return universe, sorted(rows)


def freeze_inputs(source_root, snapshot_root, universe_path, paired_reference_path=None):
    """Copy the fixed research universe without replacing missing members.

    Revisions are represented by mtime and byte size, matching the local data
    provenance already used by Caisen. No content hashes are introduced.
    """
    source_root, snapshot_root = Path(source_root).resolve(), Path(snapshot_root).resolve()
    universe_path = Path(universe_path).resolve()
    if snapshot_root.exists():
        raise FileExistsError("refusing to overwrite frozen study inputs: %s" % snapshot_root)
    universe, symbols = _symbol_rows(universe_path, paired_reference_path)
    snapshot_root.mkdir(parents=True)
    records, missing = [], []
    for kind, code in symbols:
        folder = "stocks" if kind == "stock" else "etfs" if kind == "etf" else None
        if folder is None:
            raise ValueError("unsupported instrument kind: %s" % kind)
        source = source_root / "data" / folder / (code + ".csv")
        if not source.is_file():
            missing.append({"kind": kind, "code": code, "source_path": str(source)})
            continue
        target = snapshot_root / "data" / folder / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_revision, frozen_revision = _revision(source), _revision(target)
        if source_revision != frozen_revision:
            raise RuntimeError("snapshot copy revision mismatch: %s:%s" % (kind, code))
        records.append({"kind": kind, "code": code, "source_revision": source_revision,
                        "frozen_revision": frozen_revision,
                        "source_last_modified_ns": source_revision["mtime_ns"]})
    if missing:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise FileNotFoundError("required frozen members are missing: %s" %
                                json.dumps(missing, ensure_ascii=False))
    resource_pairs = (
        (source_root / "caisen-system" / "config" / "instruments.json",
         snapshot_root / "caisen-system" / "config" / "instruments.json"),
        (source_root / "data" / "market_data_current" / ".market-data-basis.json",
         snapshot_root / "data" / "market_data_current" / ".market-data-basis.json"),
    )
    resources = []
    for source, target in resource_pairs:
        if not source.is_file():
            shutil.rmtree(snapshot_root, ignore_errors=True)
            raise FileNotFoundError("required market metadata is missing: %s" % source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_revision, frozen_revision = _revision(source), _revision(target)
        if source_revision != frozen_revision:
            raise RuntimeError("snapshot metadata revision mismatch: %s" % source)
        resources.append({"source_path": str(source), "snapshot_path": str(target.relative_to(snapshot_root)),
                          "source_revision": source_revision, "frozen_revision": frozen_revision})
    stock_count = sum(kind == "stock" for kind, _ in symbols)
    etf_count = sum(kind == "etf" for kind, _ in symbols)
    manifest = {
        "schema_version": "caisen-frozen-research-inputs-v1",
        "status": "frozen",
        "universe_id": universe.get("id"),
        "universe_date": universe.get("date"),
        "universe_seed": universe.get("seed"),
        "counts": {"stock": stock_count, "etf": etf_count, "total": len(symbols)},
        "source_root": str(source_root),
        "snapshot_root": str(snapshot_root),
        "paired_reference": (str(Path(paired_reference_path).resolve())
                              if paired_reference_path else None),
        "revision_policy": "mtime_ns_and_size_no_content_hash",
        "symbols": records,
        "resources": resources,
    }
    manifest_path = snapshot_root / "freeze-manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def verify_frozen_snapshot(snapshot_root):
    snapshot_root = Path(snapshot_root).resolve()
    manifest = _read_json(snapshot_root / "freeze-manifest.json")
    changed = []
    missing = []
    for row in manifest.get("symbols", []):
        path = snapshot_root / "data" / ("stocks" if row["kind"] == "stock" else "etfs") / (
            row["code"] + ".csv")
        if not path.is_file():
            missing.append({"kind": row["kind"], "code": row["code"]})
            continue
        actual = _revision(path)
        expected = row["frozen_revision"]
        if actual != expected:
            changed.append({"kind": row["kind"], "code": row["code"],
                            "expected": expected, "actual": actual})
    for row in manifest.get("resources", []):
        path = snapshot_root / row["snapshot_path"]
        if not path.is_file():
            missing.append({"resource": row["snapshot_path"]})
            continue
        actual = _revision(path)
        expected = row["frozen_revision"]
        if actual != expected:
            changed.append({"resource": row["snapshot_path"],
                            "expected": expected, "actual": actual})
    return {"status": "valid" if not (missing or changed) else "invalid",
            "snapshot_root": str(snapshot_root), "checked": len(manifest.get("symbols", [])),
            "missing": missing, "changed": changed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="copy the fixed universe and optional paired cohort")
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--snapshot-root", type=Path, required=True)
    freeze.add_argument("--universe", type=Path, required=True)
    freeze.add_argument("--paired-reference", type=Path)
    verify = sub.add_parser("verify", help="verify a frozen snapshot has not changed")
    verify.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_inputs(args.source_root, args.snapshot_root, args.universe,
                               args.paired_reference)
    else:
        result = verify_frozen_snapshot(args.snapshot_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "invalid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
