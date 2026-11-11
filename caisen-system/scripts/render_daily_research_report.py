#!/usr/bin/env python3
"""Rebuild the full HTML report from a completed daily-research JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import _account_instance_metadata, _report_html  # noqa: E402
from caisen.store import save_pattern_instance_metadata  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json", type=Path, help="path to daily-research.json")
    args = parser.parse_args()
    source = args.report_json.resolve()
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    native_count = derived_count = metadata_count = 0
    for summary in payload.get("account_summaries", []):
        account_path = Path(summary["account_json"])
        with account_path.open(encoding="utf-8") as handle:
            account = json.load(handle)
        metadata = _account_instance_metadata(account)
        account["pattern_instance_metadata"] = metadata
        temporary_account = account_path.with_suffix(account_path.suffix + ".tmp")
        temporary_account.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_account.replace(account_path)
        save_pattern_instance_metadata(payload["run_id"], account["code"], account["kind"], metadata)
        summary["pattern_instance_metadata_count"] = len(metadata)
        metadata_count += len(metadata)
        native_count += sum(item["metadata_source"] == "native_event_snapshot" for item in metadata)
        derived_count += sum(item["metadata_source"] == "derived_from_immutable_event_snapshot" for item in metadata)
    payload["instance_metadata"] = {
        "schema_version": "caisen-pattern-instance-metadata-v1",
        "event_count": metadata_count,
        "native_event_count": native_count,
        "derived_event_count": derived_count,
        "source_event_schema_version": "legacy event snapshots" if derived_count else "native",
        "raw_events_modified": False,
    }
    manifest_path = Path(payload.get("artifacts", {}).get("manifest_path", ""))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["instance_metadata"] = payload["instance_metadata"]
        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
    temporary_json = source.with_suffix(source.suffix + ".tmp")
    temporary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(source)
    target = source.with_name("daily-research.html")
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text(_report_html(payload), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"html_path": str(target), "status": payload.get("status"),
                      "account_count": len(payload.get("account_summaries", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
