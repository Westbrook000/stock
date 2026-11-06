#!/usr/bin/env python3
"""Seed a required signal cache from paired account signals, then time warm replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import (  # noqa: E402
    BENCHMARK_PROCESS_TREE_RSS_CAP,
    BOUNDARY_FIRST_ENGINE_VERSION,
    RECOGNIZER_VERSION,
    STRATEGY_VERSION,
    _account_ledger_signature,
    _all_ledger_checks_pass,
    _atomic_write,
    _compare_account_ledgers,
    _read_signal_cache,
    _run_with_process_tree_peak,
    _signal_cache_identity,
    _version_metadata,
    _write_signal_cache,
    load_research_bars,
    run_daily_research,
)
from caisen.daily_incremental import ENGINE_VERSION as DAILY_ENGINE_VERSION  # noqa: E402
from caisen.market import discover_instruments  # noqa: E402
from caisen.main_bottoms import MAIN_RULES  # noqa: E402


def _stamp_revision(path: Path) -> str:
    stat = path.stat()
    return "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_cache(source_map_path, baseline_report_path, snapshot_root, cache_dir,
               date_to="2026-10-30", trade_start=None, progress_callback=None):
    """Seed only signals whose input, quote and semantic metadata match this runner."""
    started = time.perf_counter()
    source_map_path, baseline_report_path = Path(source_map_path), Path(baseline_report_path)
    snapshot_root, cache_dir = Path(snapshot_root).resolve(), Path(cache_dir).resolve()
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    sources = source_map.get("sources", {})
    snapshot_manifest = json.loads((snapshot_root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if snapshot_manifest.get("status") != "complete":
        raise ValueError("frozen input snapshot manifest is not complete")
    snapshot_files = {(row.get("kind"), row.get("code")): row
                      for row in snapshot_manifest.get("files", [])}
    rows = {(row.get("kind"), row.get("code")): row
            for row in report.get("account_summaries", [])}
    parameters = report.get("parameters", {})
    if (report.get("status") != "complete" or parameters.get("date_to") != date_to
            or parameters.get("trade_start") != trade_start
            or len(rows) != 1000 or len(sources) != 1000):
        raise ValueError("paired baseline must have 1,000 complete accounts at the requested cutoff")
    if source_map.get("snapshot_id") != report.get("snapshot_id"):
        raise ValueError("paired baseline and source map identify different snapshots")
    if (report.get("strategy_version") != STRATEGY_VERSION
            or report.get("recognizer_version") != RECOGNIZER_VERSION
            or report.get("source_rule_version") != str(MAIN_RULES.get("source_rule_version", "unknown"))
            or report.get("engine_version") != BOUNDARY_FIRST_ENGINE_VERSION):
        raise ValueError("paired baseline semantic versions do not match this recognizer")

    os.environ["CAISEN_INSTRUMENTS"] = str(REPO / "config" / "instruments.json")
    if Path(source_map.get("snapshot_root", "")).resolve() != snapshot_root:
        raise ValueError("source map snapshot root does not match the selected frozen input root")
    instruments = {(item.kind, item.code): item for item in discover_instruments(snapshot_root)}
    if set(instruments) != set(rows):
        raise ValueError("frozen snapshot inventory does not match paired baseline account set")
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for index, key in enumerate(sorted(rows), 1):
        kind, code = key
        source = sources.get("%s:%s" % key)
        if not source:
            raise ValueError("source map missing %s:%s" % key)
        instrument = instruments[key]
        snapshot_path = Path(source["snapshot_path"]).resolve()
        if Path(instrument.path).resolve() != snapshot_path or _stamp_revision(snapshot_path) != source.get("snapshot_data_revision"):
            raise ValueError("frozen snapshot revision/path mismatch for %s:%s" % key)
        snapshot_sha = _sha256(snapshot_path)
        if snapshot_sha != source.get("snapshot_sha256"):
            raise ValueError("frozen snapshot SHA-256 mismatch for %s:%s" % key)
        frozen_row = snapshot_files.get(key)
        if not frozen_row or frozen_row.get("sha256") != snapshot_sha:
            raise ValueError("frozen snapshot manifest/hash mismatch for %s:%s" % key)

        row = rows[key]
        account_path = Path(source["account_path"]).resolve()
        if Path(row.get("account_json", "")).resolve() != account_path:
            raise ValueError("source map account path differs from composite report for %s:%s" % key)
        account = json.loads(account_path.read_text(encoding="utf-8"))
        if (account.get("status") not in ("complete", "no_data", "no_bars_before_cutoff")
                or account.get("recognizer") != "boundary_first"
                or account.get("engine_version") != BOUNDARY_FIRST_ENGINE_VERSION
                or account.get("recognizer_version") != RECOGNIZER_VERSION
                or account.get("strategy_version") != STRATEGY_VERSION
                or account.get("source_rule_version") != str(MAIN_RULES.get("source_rule_version", "unknown"))
                or account.get("recognition_experiment") is not None):
            raise ValueError("paired account semantic metadata mismatch for %s:%s" % key)

        loaded = load_research_bars(instrument, root=snapshot_root)
        provenance = {"data_revision": loaded["data_revision"], "tick_size": loaded["tick_size"],
            "tick_size_source": loaded["tick_size_source"], "first_date": loaded["first_date"],
            "last_date": loaded["last_date"], "normalized_rows": int(loaded["normalized_rows"])}
        bars = loaded["bars"]
        import pandas as pd
        dates = pd.to_datetime(bars["date"], errors="coerce")
        used = bars.loc[dates.notna() & (dates.dt.normalize() <= pd.Timestamp(date_to)), "date"]
        provenance["used_last_date"] = used.max().date().isoformat() if not used.empty else None
        account_provenance = account.get("provenance", {})
        account_revision = account_provenance.get("data_revision")
        if source.get("account_data_revision") != account_revision:
            raise ValueError("source map account revision mismatch for %s:%s" % key)
        if source.get("account_source_kind") == "paired_repair":
            if account_revision != provenance["data_revision"]:
                raise ValueError("paired repair was not generated from the frozen snapshot for %s:%s" % key)
        else:
            original_revision = (source.get("original_source_revision") or {}).get("data_revision")
            if (original_revision != account_revision
                    or frozen_row.get("baseline_changed_since_run") is not False):
                raise ValueError("reused account is not tied to unchanged frozen input for %s:%s" % key)
        for field in ("tick_size", "tick_size_source", "first_date", "last_date", "normalized_rows", "used_last_date"):
            if account_provenance.get(field) != provenance.get(field):
                raise ValueError("account/input %s mismatch for %s:%s" % (field, kind, code))
        if (row.get("provenance", {}).get("data_revision") != account_revision
                or row.get("provenance", {}).get("used_last_date") != date_to):
            raise ValueError("composite report input provenance mismatch for %s:%s" % key)

        signals = account.get("signals")
        stats = account.get("recognition_stats")
        if not isinstance(signals, list) or not isinstance(stats, dict):
            raise ValueError("paired account lacks signal payload or recognizer stats for %s:%s" % key)
        key_digest, identity = _signal_cache_identity(
            instrument, provenance, date_to, trade_start, "boundary_first", None)
        cache_path = cache_dir / (key_digest + ".json")
        normalized_signals = _write_signal_cache(cache_path, key_digest, identity, signals, stats)
        if normalized_signals != signals:
            raise ValueError("paired signals are not already normalized JSON for %s:%s" % key)
        cached = _read_signal_cache(cache_path, key_digest)
        if cached is None or cached["signals"] != signals:
            raise RuntimeError("seeded cache failed its exact signal payload check for %s:%s" % key)
        seed_rows.append({"kind": kind, "code": code, "source_account": str(account_path),
            "snapshot_path": str(snapshot_path), "snapshot_revision": provenance["data_revision"],
            "snapshot_sha256": snapshot_sha, "tick_size": provenance["tick_size"],
            "tick_size_source": provenance["tick_size_source"], "cache_key": key_digest,
            "cache_path": str(cache_path), "signal_count": len(signals),
            "recognizer_engine_version": stats.get("engine_version"),
            "seed_source": source.get("account_source_kind")})
        if progress_callback and (index % 100 == 0 or index == len(rows)):
            progress_callback({"stage": "cache_seed", "completed": index, "total": len(rows)})

    elapsed = time.perf_counter() - started
    manifest = {"status": "complete", "method": "existing paired account signals; no recognition executed",
        "source_map_path": str(source_map_path.resolve()), "baseline_report_path": str(baseline_report_path.resolve()),
        "snapshot_id": source_map.get("snapshot_id"), "snapshot_root": str(snapshot_root),
        "date_to": date_to, "trade_start": trade_start, "recognizer": "boundary_first",
        "versions": _version_metadata(), "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown")),
        "account_count": len(seed_rows), "signal_count": sum(row["signal_count"] for row in seed_rows),
        "seed_elapsed_seconds": elapsed, "cache_dir": str(cache_dir), "accounts": seed_rows}
    return manifest


def run(args):
    from caisen.daily_research import _atomic_write, _compare_account_ledgers, _all_ledger_checks_pass

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("warm-cache output directory must be new and empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "signal-cache"
    progress = lambda snapshot: print(json.dumps(snapshot, ensure_ascii=False), flush=True)
    seed = seed_cache(args.source_map, args.baseline_report, args.snapshot_root, cache_dir,
                      date_to=args.date_to, trade_start=args.trade_start,
                      progress_callback=progress)
    seed_path = output / "warm-cache-seed-manifest.json"
    _atomic_write(seed_path, json.dumps(seed, ensure_ascii=False, indent=2))

    # Validate all keys and complete cached event payloads before starting workers.
    preflight_started = time.perf_counter()
    source_rows = {(row["kind"], row["code"]): row for row in seed["accounts"]}
    cache_validated = 0
    for row in seed["accounts"]:
        cached = _read_signal_cache(row["cache_path"], row["cache_key"])
        if cached is None or len(cached["signals"]) != row["signal_count"]:
            raise RuntimeError("cache preflight failed for %s:%s" % (row["kind"], row["code"]))
        cache_validated += 1
    preflight_seconds = time.perf_counter() - preflight_started

    warm_started = time.perf_counter()
    report, process_tree_peak = _run_with_process_tree_peak(lambda: run_daily_research(
        args.universe, output / "warm-run", date_to=args.date_to, trade_start=args.trade_start,
        workers=args.workers, root=args.snapshot_root, recognizer="boundary_first",
        execution_mode="spawn", signal_cache_dir=cache_dir, require_signal_cache=True))
    warm_wall = time.perf_counter() - warm_started

    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    expected = set(source_rows)
    parity, divergence = _compare_account_ledgers(baseline, report, expected)
    cache_hits = sum(row.get("recognition_stats", {}).get("cache_hit") is True
                     for row in report.get("account_summaries", []))
    revisions_match = len(report.get("account_summaries", [])) == len(expected) and all(
        row.get("provenance", {}).get("data_revision") == source_rows[(row["kind"], row["code"])]["snapshot_revision"]
        and row.get("provenance", {}).get("used_last_date") == args.date_to
        for row in report.get("account_summaries", []))
    accounting_ok = _all_ledger_checks_pass(report, 1000)
    audit = {"status": "complete" if report.get("status") == "complete" and cache_hits == 1000
             and parity and revisions_match and accounting_ok else "failed",
        "warm_run_report_path": report.get("artifacts", {}).get("json_path"),
        "baseline_report_path": str(Path(args.baseline_report).resolve()),
        "source_map_path": str(Path(args.source_map).resolve()),
        "date_to": args.date_to, "workers": args.workers, "execution_mode": "spawn",
        "recognizer": "boundary_first", "signal_cache_required": True,
        "cache_preflight_entries": cache_validated, "cache_hits": cache_hits,
        "seed_source": "paired composite account signals; no recognition executed",
        "seed_elapsed_seconds": seed["seed_elapsed_seconds"],
        "cache_preflight_seconds": preflight_seconds,
        "warm_wall_seconds": warm_wall,
        "warm_runner_timings": report.get("timings"),
        "process_tree_peak_bytes": process_tree_peak,
        "process_tree_memory_cap_bytes": BENCHMARK_PROCESS_TREE_RSS_CAP,
        "full_normalized_account_ledger_parity": parity, "first_divergence": divergence,
        "data_revisions_match_snapshot": revisions_match,
        "all_accounting_checks_pass": accounting_ok,
        "account_count": report.get("summary", {}).get("account_count"),
        "signal_count": report.get("summary", {}).get("signal_count"),
        "trade_count": report.get("summary", {}).get("trade_count"),
        "error": None if report.get("status") == "complete" and cache_hits == 1000
            and parity and revisions_match and accounting_ok else "warm cache or ledger acceptance gate failed"}
    _atomic_write(output / "warm-cache-audit.json", json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def main():
    default_root = REPO / "reports" / "daily-research" / "paired-input-snapshot-20261105-v1"
    default_base = REPO / "reports" / "daily-research" / "paired-composites-20261105-snapshot-v1" / "baseline"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, default=default_base / "data-source-map.json")
    parser.add_argument("--baseline-report", type=Path, default=default_base / "run-manifest.json")
    parser.add_argument("--snapshot-root", type=Path, default=default_root)
    parser.add_argument("--universe", type=Path,
        default=REPO / "research" / "universes" / "daily_800_stock_200_etf_v1.json")
    parser.add_argument("--output", type=Path,
        default=REPO / "reports" / "daily-research" / "warm-cache-20261105-8worker")
    parser.add_argument("--date-to", default="2026-10-30")
    parser.add_argument("--trade-start")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8, 10), default=8)
    args = parser.parse_args()
    result = run(args)
    if result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
