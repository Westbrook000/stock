#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a same-members E1-vs-G1 comparison from completed daily-research JSON."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from statistics import median
import re
import sys
import time


SYSTEM = Path(__file__).resolve().parents[1]
if str(SYSTEM) not in sys.path:
    sys.path.insert(0, str(SYSTEM))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_metadata(path, include_hash=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "mtime_ns": int(stat.st_mtime_ns),
              "size_bytes": int(stat.st_size)}
    if include_hash:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def audit_evidence(path):
    audit = load_json(path)
    fields = ("audit_version", "scope", "account_count", "trade_count", "signal_count",
              "signal_pattern_row_count", "trade_count_from_ledger_audit",
              "signals_without_historically_eligible_pattern_count",
              "causality_and_boundary_violation_count", "missing_historical_eligibility_record_count",
              "missing_historical_eligibility_pattern_rows_count",
              "ledger_missing_history_error_count", "order_signal_unmatched_count",
              "history_fields_changed", "new_unknown_data_version_rows", "failed_checks_by_invariant",
              "checked")
    result = {key: audit[key] for key in fields if key in audit}
    if "metadata_findings_missing_historical_eligibility" in audit:
        result["metadata_findings_missing_historical_eligibility_count"] = len(
            audit["metadata_findings_missing_historical_eligibility"])
    findings = audit.get("causality_and_boundary_violations") or []
    if findings:
        result["finding_examples"] = [{key: row.get(key) for key in
            ("code", "signal_date", "paired_at", "instance_id", "boundary_version_id", "failed_checks")
            if key in row} for row in findings]
    result["status"] = audit.get("status") or ("passed" if audit.get("passed") is True else "not_passed")
    result["source"] = file_metadata(path, include_hash=True)
    return result


def worker_matrix_evidence(path):
    matrix = load_json(path)
    fields = ("status", "benchmark_id", "benchmark_protocol_version", "policy_scope",
              "validation_scope_note", "universe_id", "reference_count", "worker_counts",
              "recommended_workers", "selected_worker_count", "elapsed_seconds",
              "speedup_vs_serial", "selected_process_tree_peak_bytes",
              "process_tree_memory_cap_bytes", "performance_gate_passed", "parity_passed",
              "accounting_checks_passed", "event_parity_passed", "source_unchanged_during_run")
    result = {key: matrix[key] for key in fields if key in matrix}
    result["timing_scope"] = "36 pinned accounts; not full1000 timing"
    result["source"] = file_metadata(path, include_hash=True)
    result["timing_artifacts"] = [file_metadata(item, include_hash=True)
                                  for item in sorted(Path(path).parent.glob("*-timing.json"))]
    return result


def _compact_pattern(pattern):
    source = pattern.to_dict() if hasattr(pattern, "to_dict") else pattern
    source_evidence = source.get("evidence", [])
    source_lifecycle = next((item.get("value") for item in source_evidence
                             if item.get("code") == "joint_lifecycle"), None)
    base_fields = ("pattern_id", "code", "kind", "timeframe", "pattern_type", "direction", "state",
                   "formed_at", "confirmed_at", "confirmation_date", "last_observed_date",
                   "rule_version", "parent_id")
    row = {key: source.get(key) for key in base_fields if key in source}
    if source_lifecycle is None and source.get("neckline"):
        row["legacy_neckline"] = source["neckline"]
    row["key_points"] = {key: value for key, value in (source.get("key_points") or {}).items()
                         if re.fullmatch(r"pivot_\d+(?:_date)?", key) or key == "pivot_scale"}
    def selected_fields(value, fields):
        return {key: value[key] for key in fields if key in value}

    def compact_tests(tests):
        fields = ("group_id", "pivot_date", "known_date", "kind", "level", "valid")
        return [selected_fields(test, fields) for test in tests if isinstance(test, dict)]

    def compact_lifecycle(value, test_evidence=()):
        fields = ("schema_version", "instance_id", "structure_key", "boundary_version_id",
                  "paired_at", "paired_at_status", "instance_discovered_at", "line_known_at",
                  "structure_known_at", "boundary_selected_at", "state_at",
                  "canonical_confirmation_date", "entry_eligible", "entry_eligible_at_confirmation",
                  "entry_eligibility_at_confirmation_status", "selection_at_confirmation",
                  "selection_reason", "boundary", "structure_anchors")
        result = selected_fields(value, fields)
        boundary_tests = []
        for anchor in value.get("boundary_anchor_tests", []):
            matches = [test for test in test_evidence
                       if ((anchor.get("group_id") is not None
                            and test.get("group_id") == anchor.get("group_id"))
                           or (test.get("pivot_date") == anchor.get("pivot_date")
                               and test.get("level") == anchor.get("level")))]
            boundary_tests.append(compact_tests(matches[:1])[0] if matches
                                   else selected_fields(anchor, ("group_id", "pivot_date", "known_date",
                                                                 "level", "kind", "valid")))
        result["boundary_anchor_tests"] = boundary_tests
        event_fields = ("sequence", "kind", "date", "occurred_at", "recorded_at", "known_at",
                        "date_semantics", "reason", "boundary", "required_close", "close",
                        "instance_id", "boundary_version_id")
        result["events"] = [selected_fields(event, event_fields) for event in value.get("events", [])]
        return result

    retained = []
    lifecycle_codes = {"joint_lifecycle", "main_structure", "main_selection", "neckline_selection",
                       "neckline_tests", "neckline_breakout", "flag_lifecycle", "flag_geometry",
                       "flag_breakout", "structure_lifecycle"}
    for item in source_evidence:
        code, value = item.get("code"), item.get("value")
        if code not in lifecycle_codes:
            continue
        if code == "joint_lifecycle" and isinstance(value, dict):
            test_evidence = next((evidence.get("value") for evidence in source_evidence
                                  if evidence.get("code") == "neckline_tests"), []) or []
            retained.append({"code": code, "value": compact_lifecycle(value, test_evidence)})
        elif code == "main_structure" and isinstance(value, dict) and source_lifecycle is None:
            fields = ("role", "family", "scale", "pivot_scale", "signal_eligible", "eligibility_reason",
                      "support_lower", "support_upper", "frozen_at", "first_trough_date",
                      "latest_anchor_date", "maturity_date", "structure_end_date",
                      "minimum_duration_months", "terminal_state", "observed_date",
                      "instance_discovered_at", "selection_at_confirmation",
                      "entry_eligible_at_confirmation")
            retained.append({"code": code, "value": selected_fields(value, fields)})
        elif code == "main_selection" and isinstance(value, dict) and source_lifecycle is None:
            retained.append({"code": code, "value": selected_fields(
                value, ("selected", "decided_at", "primary_pattern_id", "primary_instance_id", "reason"))})
        elif code == "neckline_selection" and isinstance(value, dict):
            fields = ("effective_start_date", "freeze_date", "line_frozen_at", "structure_known_at")
            compact = selected_fields(value, fields)
            if source_lifecycle is None:
                compact["independent_tests"] = compact_tests(value.get("independent_tests", []))
            retained.append({"code": code, "value": compact})
        elif code == "neckline_tests" and isinstance(value, list) and source_lifecycle is None:
            retained.append({"code": code, "value": compact_tests(value)})
        elif code in {"neckline_breakout", "flag_breakout", "flag_lifecycle", "flag_geometry",
                      "structure_lifecycle", "recognition_context"} and isinstance(value, dict):
            if code != "neckline_breakout" or source_lifecycle is None:
                retained.append({"code": code, "value": value})
    row["evidence"] = retained
    return row


def _chart_boundary(pattern):
    lifecycle = next((item.get("value") for item in pattern.get("evidence", [])
                      if item.get("code") == "joint_lifecycle"), None)
    if lifecycle is not None:
        return lifecycle.get("boundary")
    selection = next((item.get("value") for item in pattern.get("evidence", [])
                      if item.get("code") == "neckline_selection"), {})
    return pattern.get("legacy_neckline") or selection.get("selected_line")


def _matching_lifecycle_events(lifecycle):
    instance_id, boundary_id = lifecycle.get("instance_id"), lifecycle.get("boundary_version_id")
    if instance_id is None or boundary_id is None:
        return []
    return [event for event in lifecycle.get("events", [])
            if event.get("instance_id") == instance_id
            and event.get("boundary_version_id") == boundary_id]


def _compact_signals(signals, patterns):
    pattern_refs = {pattern.get("pattern_id"): pattern for pattern in patterns}

    def lifecycle_ids(pattern):
        evidence = pattern.get("evidence", []) if isinstance(pattern, dict) else []
        lifecycle = next((row.get("value") for row in evidence
                          if isinstance(row, dict) and row.get("code") == "joint_lifecycle"), {})
        return lifecycle.get("instance_id"), lifecycle.get("boundary_version_id")

    compact = []
    for signal in signals:
        ids = signal.get("pattern_ids", [])
        signal_patterns = signal.get("patterns", []) or []
        signal_patterns_by_id = {str(pattern.get("pattern_id")): pattern
                                 for pattern in signal_patterns
                                 if isinstance(pattern, dict) and pattern.get("pattern_id") is not None}
        labels = signal.get("pattern_labels", signal.get("labels", signal.get("pattern_types", [])))
        instances = []
        for index, pattern_id in enumerate(ids):
            local_pattern = signal_patterns_by_id.get(str(pattern_id))
            if local_pattern is None and index < len(signal_patterns):
                candidate = signal_patterns[index]
                wanted_type = (signal.get("pattern_types", []) or [])
                if (isinstance(candidate, dict)
                        and (index >= len(wanted_type)
                             or candidate.get("pattern_type") == wanted_type[index])):
                    local_pattern = candidate
            instance_id, boundary_id = lifecycle_ids(local_pattern or {})
            if instance_id is None or boundary_id is None:
                canonical = pattern_refs.get(pattern_id, {})
                canonical_instance, canonical_boundary = lifecycle_ids(canonical)
                instance_id = instance_id or canonical_instance
                boundary_id = boundary_id or canonical_boundary
            instances.append({"pattern_id": pattern_id, "instance_id": instance_id,
                              "boundary_version_id": boundary_id})
        compact.append({
            "signal_date": signal.get("signal_date"), "entry_date": signal.get("entry_date"),
            "pattern_labels": labels,
            "instances": instances,
        })
    return compact


def validate_case_bundle(case_bundle, candidate_engine_version):
    if case_bundle.get("schema_version") not in {
            "lifecycle-six-cases-v1", "engine8-six-cases-v1"}:
        raise ValueError("case bundle schema is unsupported")
    if case_bundle.get("engine_version") != candidate_engine_version:
        raise ValueError("case bundle engine version must match the candidate report")
    cases = case_bundle.get("cases", [])
    required = {("stock", "000001"), ("stock", "000012"), ("stock", "000538"),
                ("stock", "600108"), ("etf", "159718"), ("etf", "516630")}
    observed = {(str(row.get("kind")), str(row.get("code"))) for row in cases}
    if len(cases) != 6 or observed != required:
        raise ValueError("case bundle must contain the six requested stock/ETF cases")


def trade_diagnostics(trades):
    trades = list(trades)
    short = [trade for trade in trades
             if trade.get("holding_sessions") is not None
             and float(trade["holding_sessions"]) <= 5]
    negative = [trade for trade in trades if float(trade.get("pnl", 0.0)) < 0]
    short_negative = [trade for trade in short if float(trade.get("pnl", 0.0)) < 0]
    below_stop = [trade for trade in trades
                  if trade.get("stop_price") is not None and trade.get("exit_price") is not None
                  and float(trade["exit_price"]) < float(trade["stop_price"]) - 1e-8]
    return {"trade_count": len(trades), "negative_trade_count": len(negative),
            "negative_trade_rate": len(negative) / len(trades) if trades else None,
            "holding_le_5_count": len(short),
            "holding_le_5_negative_count": len(short_negative),
            "holding_le_5_negative_rate": len(short_negative) / len(short) if short else None,
            "holding_le_5_negative_share_of_all_losses": len(short_negative) / len(negative)
            if negative else None,
            "exit_below_stop_price_count": len(below_stop)}


def _pattern_eligibility(pattern):
    evidence = {row.get("code"): row.get("value") for row in pattern.get("evidence", [])}
    lifecycle = evidence.get("joint_lifecycle") or {}
    joint_value = lifecycle.get("entry_eligible_at_confirmation")
    joint_status = lifecycle.get("entry_eligibility_at_confirmation_status")
    flag_value = (evidence.get("flag_lifecycle") or {}).get("entry_eligible_at_confirmation")
    return joint_value, joint_status, flag_value


def audit_signal_eligibility(report, members=None, include_signals=True):
    """Read stored account outputs; never run recognition here."""
    wanted = None
    if members is not None:
        wanted = {("stock", str(code)) for code in members.get("stocks", [])}
        wanted |= {("etf", str(code)) for code in members.get("etfs", [])}
    expected_accounts = report.get("account_summaries", [])
    if wanted is not None:
        expected_accounts = [row for row in expected_accounts
                             if (str(row.get("kind")), str(row.get("code"))) in wanted]
    paths = [row.get("account_json") for row in expected_accounts if row.get("account_json")]
    if not paths:
        paths = report.get("account_files", [])
        if isinstance(paths, dict):
            paths = list(paths.values())
    signals = []
    source_files = []
    missing_paths = []
    daily_deltas = {}
    starting_capital = sum(float(row.get("starting_cash", 0.0)) for row in expected_accounts)
    for value in paths:
        path = Path(value)
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        account = load_json(path)
        key = (str(account.get("kind")), str(account.get("code")))
        if wanted is not None and key not in wanted:
            continue
        source_files.append(file_metadata(path))
        if include_signals:
            signals.extend(account.get("signals", []))
        previous = float(account.get("starting_cash", 0.0))
        for row in account.get("daily_equity", []):
            date = str(row.get("date"))
            equity = float(row.get("equity", previous))
            daily_deltas[date] = daily_deltas.get(date, 0.0) + equity - previous
            previous = equity

    raw_counts, joint_true, joint_false, joint_unknown = {}, {}, {}, {}
    flag_true = {}
    eligible_by_identity = {}
    false_with_true = 0
    multi_tag = 0
    tag_count = 0
    status_counts = {}
    for signal in signals:
        identity = (str(signal.get("kind")), str(signal.get("code")),
                    str(signal.get("signal_date")), str(signal.get("entry_date")))
        patterns = signal.get("patterns", []) or []
        by_id = {str(pattern.get("pattern_id")): pattern for pattern in patterns
                 if pattern.get("pattern_id") is not None}
        ids = signal.get("pattern_ids", []) or []
        matched = [by_id[str(pattern_id)] for pattern_id in ids if str(pattern_id) in by_id]
        if not matched:
            matched = patterns
        if len(matched) > 1:
            multi_tag += 1
        selected_eligible = set()
        has_true = has_false = False
        for pattern in matched:
            tag = str(pattern.get("pattern_type") or "unknown")
            raw_counts[tag] = raw_counts.get(tag, 0) + 1
            tag_count += 1
            joint_value, joint_status, flag_value = _pattern_eligibility(pattern)
            status_key = f"{tag}:{joint_status or 'missing'}"
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            if joint_value is True:
                joint_true[tag] = joint_true.get(tag, 0) + 1
                selected_eligible.add(tag)
                has_true = True
            elif joint_value is False:
                joint_false[tag] = joint_false.get(tag, 0) + 1
                has_false = True
            else:
                joint_unknown[tag] = joint_unknown.get(tag, 0) + 1
            if flag_value is True:
                flag_true[tag] = flag_true.get(tag, 0) + 1
        if has_true and has_false:
            false_with_true += 1
        eligible_by_identity[identity] = sorted(selected_eligible)
    expected_signals = sum(int(row.get("signal_count", 0)) for row in expected_accounts)
    equity = peak = starting_capital
    maximum_drawdown = 0.0 if starting_capital else None
    for date in sorted(daily_deltas):
        equity += daily_deltas[date]
        peak = max(peak, equity)
        if peak:
            maximum_drawdown = max(maximum_drawdown or 0.0, (peak - equity) / peak)
    audit = {"account_file_count": len(source_files),
             "expected_account_file_count": len(expected_accounts),
             "missing_account_file_count": len(missing_paths),
             "missing_account_files": missing_paths,
             "signal_count": len(signals) if include_signals else None,
             "expected_signal_count": expected_signals,
             "signal_count_matches": (len(signals) == expected_signals) if include_signals else None,
             "portfolio_maximum_drawdown": maximum_drawdown,
             "matched_pattern_tag_count": tag_count,
             "multi_tag_signal_count": multi_tag,
             "signals_with_joint_eligible_and_ineligible_siblings": false_with_true,
             "raw_pattern_tag_counts": raw_counts,
             "joint_eligible_tag_counts": joint_true,
             "joint_ineligible_tag_counts": joint_false,
             "joint_unknown_tag_counts": joint_unknown,
             "flag_lifecycle_true_side_evidence_counts": flag_true,
             "joint_eligibility_status_counts": status_counts}
    return audit, eligible_by_identity, source_files


def collect_cases(case_root, output_path, diagnostics_path=None):
    """Collect six frozen cases with the same engine/config used by the E1 run."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if diagnostics_path is not None and Path(diagnostics_path).exists():
        raise FileExistsError(diagnostics_path)
    if diagnostics_path is not None and Path(diagnostics_path).resolve() == output_path.resolve():
        raise ValueError("compact case output and full diagnostics must use different paths")
    import pandas as pd
    from caisen.daily_incremental import ENGINE_VERSION, recognize_daily_history
    from caisen.daily_research import load_research_bars
    from caisen.market import InstrumentFile, TICK_RULE_VERSION
    from caisen.research_config import RecognitionExperimentConfig, recognition_experiment

    case_root = Path(case_root).resolve()
    manifest = load_json(case_root / "case-input-manifest.json")
    expected = {f"{item['kind']}:{item['code']}": item for item in manifest.get("rows", [])}
    expected_keys = {f"{kind}:{code}" for code, kind in
                     (("000001", "stock"), ("000012", "stock"), ("000538", "stock"),
                      ("600108", "stock"), ("159718", "etf"), ("516630", "etf"))}
    if set(expected) != expected_keys:
        raise ValueError("case manifest must contain exactly the six requested symbols")
    registry = case_root / "caisen-system/config/instruments.json"
    if not registry.is_file():
        raise FileNotFoundError(f"case root lacks frozen instrument registry: {registry}")
    prior_registry = os.environ.get("CAISEN_INSTRUMENTS")
    os.environ["CAISEN_INSTRUMENTS"] = str(registry)
    cutoff = str(manifest.get("cutoff") or "2026-10-30")
    config = RecognitionExperimentConfig(
        experiment_id="internal_rebound_coverage_g1_v1",
        require_internal_rebound_coverage=True,
        require_minimum_anchor_span=False,
    )
    cases, diagnostic_cases = [], []
    try:
        for key in sorted(expected_keys):
            item = expected[key]
            kind, code = key.split(":", 1)
            path = (case_root / "data" / ("stocks" if kind == "stock" else "etfs")
                    / f"{code}.csv")
            if (not path.is_relative_to(case_root)
                    or not path.parent.resolve().is_relative_to(case_root)):
                raise ValueError(f"case input escapes frozen root: {path}")
            before = path.stat()
            frozen = item.get("revision", {})
            if (int(before.st_mtime_ns), int(before.st_size)) != (
                    int(frozen.get("mtime_ns", -1)), int(frozen.get("size_bytes", -1))):
                raise RuntimeError(f"frozen case input changed: {key}")
            instrument = InstrumentFile(code=code, kind=kind, path=str(path))
            loaded = load_research_bars(instrument, root=case_root)
            tick = float(loaded["tick_size"])
            wanted_tick = 0.01 if kind == "stock" else 0.001
            if tick != wanted_tick:
                raise ValueError(f"unexpected {kind} tick size for {code}: {tick}")
            bars = loaded["bars"].copy()
            bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
            bars = bars[bars["date"].notna() & (bars["date"].dt.strftime("%Y-%m-%d") <= cutoff)]
            bars = bars.sort_values("date", kind="mergesort").drop_duplicates(
                "date", keep="last").reset_index(drop=True)
            bars.attrs["quality"] = {"tick_size": tick,
                                      "tick_size_source": loaded["tick_size_source"],
                                      "tick_rule_version": TICK_RULE_VERSION}
            bars.attrs["provenance"] = {"data_version": loaded["data_revision"],
                                         "source": "frozen_case_snapshot",
                                         "tick_size": tick,
                                         "tick_size_source": loaded["tick_size_source"]}
            with recognition_experiment(config):
                result = recognize_daily_history(bars, code, kind, tick_size=tick)
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise RuntimeError(f"frozen case input changed during recognition: {key}")
            pattern_rows = [pattern.to_dict() for pattern in result["patterns"]]
            diagnostic_cases.append({"kind": kind, "code": code, "patterns": pattern_rows,
                                     "signals": result["signals"]})
            compact_patterns = [_compact_pattern(pattern) for pattern in result["patterns"]]
            compact_signals = _compact_signals(result["signals"], compact_patterns)
            cases.append({
                "kind": kind, "code": code, "cutoff": cutoff,
                "input_path": str(path), "source_path": item.get("source"),
                "input_revision": {"mtime_ns": int(before.st_mtime_ns),
                                    "size_bytes": int(before.st_size)},
                "tick_size": tick, "tick_size_source": loaded["tick_size_source"],
                "data_revision": loaded["data_revision"],
                "bars": [[row.date.strftime("%Y-%m-%d"), float(row.open), float(row.high),
                          float(row.low), float(row.close), float(row.volume)]
                         for row in bars.itertuples()],
                "patterns": compact_patterns,
                "signals": compact_signals,
                "recognition_stats": result["performance"],
            })
    finally:
        if prior_registry is None:
            os.environ.pop("CAISEN_INSTRUMENTS", None)
        else:
            os.environ["CAISEN_INSTRUMENTS"] = prior_registry
    payload = {"schema_version": "lifecycle-six-cases-v1", "cutoff": cutoff,
               "engine_version": ENGINE_VERSION,
               "recognition_config": {"experiment_id": config.experiment_id,
                                      "require_internal_rebound_coverage": True,
                                      "require_minimum_anchor_span": False},
               "input_manifest": manifest,
               "source_metadata": {"manifest": file_metadata(case_root / "case-input-manifest.json"),
                                   "instrument_registry": file_metadata(registry)},
               "cases": cases}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(
            {"schema_version": "lifecycle-six-cases-diagnostic-v1",
             "engine_version": ENGINE_VERSION, "cases": diagnostic_cases},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def compact_trade(trade):
    fields = ("kind", "code", "signal_date", "entry_date", "entry_price", "exit_date",
              "exit_price", "stop_price", "target_price", "exit_reason", "pnl", "return_pct",
              "holding_sessions", "holding_days", "pattern_types", "pattern_labels")
    return {field: trade[field] for field in fields if field in trade}


def summarize(report, members=None):
    rows = report.get("account_summaries", [])
    if members is not None:
        wanted = {("stock", str(code)) for code in members.get("stocks", [])}
        wanted |= {("etf", str(code)) for code in members.get("etfs", [])}
        rows = [row for row in rows if (str(row.get("kind")), str(row.get("code"))) in wanted]
    keys = {(str(row.get("kind")), str(row.get("code"))) for row in rows}
    trades = [trade for trade in report.get("trades", [])
              if (str(trade.get("kind")), str(trade.get("code"))) in keys]
    trades = [compact_trade(trade) for trade in trades]
    returns = [float(row.get("metrics", {}).get("cumulative_return", 0.0)) for row in rows]
    starting = sum(float(row.get("starting_cash", 0.0)) for row in rows)
    ending = sum(float(row.get("ending_equity", 0.0)) for row in rows)
    pnls = [float(trade.get("pnl", 0.0)) for trade in trades]
    gains = sum(pnl for pnl in pnls if pnl > 0)
    losses = -sum(pnl for pnl in pnls if pnl < 0)
    wins = sum(pnl > 0 for pnl in pnls)
    exit_reasons = {}
    shape_groups, asset_groups = {}, {}
    for trade in trades:
        reason = str(trade.get("exit_reason", "unknown"))
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        kind = str(trade.get("kind", "unknown"))
        pnl = float(trade.get("pnl", 0.0))
        asset_groups.setdefault(kind, {"trade_count": 0, "wins": 0, "pnl": 0.0})
        asset_groups[kind]["trade_count"] += 1
        asset_groups[kind]["wins"] += pnl > 0
        asset_groups[kind]["pnl"] += pnl
        for shape in trade.get("pattern_types", []) or ["unknown"]:
            group = shape_groups.setdefault(str(shape), {"trade_count": 0, "wins": 0, "pnl": 0.0})
            group["trade_count"] += 1
            group["wins"] += pnl > 0
            group["pnl"] += pnl
    revisions = {f"{row.get('kind')}:{row.get('code')}":
                 row.get("provenance", {}).get("data_revision") for row in rows}
    holding = [float(trade["holding_sessions"]) for trade in trades
               if trade.get("holding_sessions") is not None]
    drawdowns = [float(row["metrics"]["maximum_drawdown"]) for row in rows
                 if row.get("metrics", {}).get("maximum_drawdown") is not None]
    utilization = [float(row["metrics"]["capital_utilization"]) for row in rows
                   if row.get("metrics", {}).get("capital_utilization") is not None]
    failures = [row for row in rows if row.get("status") != "complete"]
    return {
        "account_count": len(rows), "failure_count": len(failures),
        "starting_capital": starting, "ending_equity": ending,
        "net_pnl": ending - starting,
        "cumulative_return": ending / starting - 1 if starting else None,
        "trade_count": len(trades), "winning_trades": wins,
        "win_rate": wins / len(trades) if trades else None,
        "profit_factor": gains / losses if losses else None,
        "average_holding_sessions": sum(holding) / len(holding) if holding else None,
        "account_max_drawdown": max(drawdowns) if drawdowns else None,
        "average_capital_utilization": sum(utilization) / len(utilization) if utilization else None,
        "signal_count": sum(int(row.get("signal_count", 0)) for row in rows),
        "positive_account_count": sum(value > 0 for value in returns),
        "account_return_mean": sum(returns) / len(returns) if returns else None,
        "account_return_median": median(returns) if returns else None,
        "account_return_min": min(returns) if returns else None,
        "account_return_max": max(returns) if returns else None,
        "account_return_distribution": sorted(returns),
        "exit_reason_counts": exit_reasons,
        "trade_diagnostics": trade_diagnostics(trades),
        "asset_groups": asset_groups,
        "shape_groups": shape_groups,
        "data_revisions": revisions,
        "accounts": rows,
        "trades": trades,
    }


def compose_hybrid_g1_baseline(stock_report, etf_report, *, expected_stock_accounts=800,
                               expected_etf_accounts=200, stock_source=None, etf_source=None):
    """Compose the historical G1 stock accounts with the corrected-tick ETF replay."""
    reports = (("stock", stock_report, 0.01, expected_stock_accounts),
               ("etf", etf_report, 0.001, expected_etf_accounts))
    accounts, account_files = [], []
    for wanted_kind, report, wanted_tick, expected_count in reports:
        if report.get("engine_version") != "caisen-joint-pattern-session-v4":
            raise ValueError(f"{wanted_kind} source must be engine-v4")
        if report.get("source_rule_version") != "caisen-main-bottoms-v12-structure-boundary":
            raise ValueError(f"{wanted_kind} source must use source rule v12")
        if report.get("date_to") != "2026-10-30":
            raise ValueError(f"{wanted_kind} source must end at 2026-10-30")
        rows = [row for row in report.get("account_summaries", [])
                if str(row.get("kind")) == wanted_kind]
        if len(rows) != expected_count:
            raise ValueError(f"{wanted_kind} source has {len(rows)} accounts; expected {expected_count}")
        keys = [(str(row.get("kind")), str(row.get("code"))) for row in rows]
        if any(kind != wanted_kind for kind, _ in keys) or len(set(keys)) != len(keys):
            raise ValueError(f"{wanted_kind} source account kinds or identities are invalid")
        if any((row.get("provenance") or {}).get("tick_size") != wanted_tick for row in rows):
            raise ValueError(f"{wanted_kind} source contains an account with the wrong tick size")
        wanted = set(keys)
        account_fields = ("kind", "code", "status", "provenance", "metrics", "signal_count",
                          "trade_count", "starting_cash", "ending_equity", "account_json",
                          "account_json_relative")
        accounts.extend({field: row[field] for field in account_fields if field in row}
                        for row in rows)
        files = report.get("account_files", [])
        for row in rows:
            path = row.get("account_json")
            if not path:
                code = str(row.get("code"))
                path = next((item for item in files
                             if Path(str(item)).name.startswith(f"{wanted_kind}-{code}")), None)
            if path and path not in account_files:
                account_files.append(path)
        if wanted_kind == "stock":
            stock_keys = wanted
        else:
            etf_keys = wanted

    if stock_report.get("source_rule_version") != etf_report.get("source_rule_version"):
        raise ValueError("hybrid sources must use the same rule version")
    trades = [trade for wanted_kind, report, keys in
              (("stock", stock_report, stock_keys), ("etf", etf_report, etf_keys))
              for trade in report.get("trades", [])
              if str(trade.get("kind")) == wanted_kind
              and (str(trade.get("kind")), str(trade.get("code"))) in keys]
    trade_fields = ("kind", "code", "signal_date", "entry_date", "entry_price", "exit_price",
                    "stop_price", "target_price", "exit_reason", "pnl", "return_pct",
                    "holding_sessions", "holding_days", "pattern_types", "pattern_labels")
    trades = [{field: trade[field] for field in trade_fields if field in trade}
              for trade in trades]
    starting = sum(float(row.get("starting_cash", 0.0)) for row in accounts)
    ending = sum(float(row.get("ending_equity", 0.0)) for row in accounts)
    wins = sum(float(trade.get("pnl", 0.0)) > 0 for trade in trades)
    pnl = sum(float(trade.get("pnl", 0.0)) for trade in trades)
    return {
        "date_to": "2026-10-30", "engine_version": "caisen-joint-pattern-session-v4",
        "source_rule_version": "caisen-main-bottoms-v12-structure-boundary",
        "account_summaries": accounts, "account_files": account_files, "trades": trades,
        "summary": {"account_count": len(accounts), "successful_account_count":
                    sum(row.get("status") == "complete" for row in accounts),
                    "failed_account_count": sum(row.get("status") != "complete" for row in accounts),
                    "trade_count": len(trades), "win_rate": wins / len(trades) if trades else None,
                    "starting_capital": starting, "ending_equity": ending,
                    "net_pnl": ending - starting,
                    "cumulative_return": ending / starting - 1 if starting else None,
                    "trade_net_pnl": pnl},
        "timings": {},
        "composition": {
            "kind": "hybrid_existing_account_reports", "timing": "N/A; composed from existing account reports",
            "tick_sizes": {"stock": 0.01, "etf": 0.001},
            "stock_source": str(stock_source) if stock_source else None,
            "etf_source": str(etf_source) if etf_source else None,
            "stock_account_count": len(stock_keys), "etf_account_count": len(etf_keys),
            "admission_metadata_note": "Preserve source admission_types serialization; effective cache admission index is None per metadata-correction-20261113.json.",
            "metadata_correction_source": "research/experiments/structure_boundary_losses_20261112/tick_study/metadata-correction-20261113.json"}}


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _account_keys(summary):
    return {(str(row.get("kind")), str(row.get("code"))) for row in summary["accounts"]}


def compare_trades(baseline, candidate, candidate_eligible_types=None):
    def identity(trade):
        return (str(trade.get("kind")), str(trade.get("code")),
                str(trade.get("signal_date")), str(trade.get("entry_date")))

    old = {identity(trade): trade for trade in baseline["trades"]}
    new = {identity(trade): trade for trade in candidate["trades"]}
    common = old.keys() & new.keys()
    changed = []
    for key in sorted(common):
        old_labels = old[key].get("pattern_types", []) or old[key].get("pattern_tags", [])
        new_labels = new[key].get("pattern_types", []) or new[key].get("pattern_tags", [])
        if old_labels != new_labels:
            changed.append({"identity": list(key), "baseline_labels": old_labels,
                            "candidate_labels": new_labels,
                            "baseline_pnl": float(old[key].get("pnl", 0.0)),
                            "candidate_pnl": float(new[key].get("pnl", 0.0))})

    def aggregate(rows):
        rows = list(rows)
        return {"trade_count": len(rows),
                "wins": sum(float(row.get("pnl", 0.0)) > 0 for row in rows),
                "win_rate": (sum(float(row.get("pnl", 0.0)) > 0 for row in rows) / len(rows)
                             if rows else None),
                "net_pnl": sum(float(row.get("pnl", 0.0)) for row in rows)}

    old_only_keys = sorted(old.keys() - new.keys())
    new_only_keys = sorted(new.keys() - old.keys())
    common_keys = sorted(common)

    def grouped(rows, get_key):
        result = {}
        for trade in rows:
            group = get_key(trade)
            if group is None:
                continue
            result.setdefault(str(group), []).append(trade)
        return result

    old_years, new_years = grouped(old.values(), lambda row:
                                   f"{int(str(row.get('entry_date', '0000'))[:4]) // 10 * 10}s"
                                   if str(row.get("entry_date", ""))[:4].isdigit() else "unknown"), \
                    grouped(new.values(), lambda row:
                            f"{int(str(row.get('entry_date', '0000'))[:4]) // 10 * 10}s"
                            if str(row.get("entry_date", ""))[:4].isdigit() else "unknown")
    decade_groups = []
    for decade in sorted(set(old_years) | set(new_years)):
        left, right = aggregate(old_years.get(decade, [])), aggregate(new_years.get(decade, []))
        decade_groups.append({"group": decade, "baseline": left, "candidate": right,
                              "trade_count_delta": right["trade_count"] - left["trade_count"],
                              "net_pnl_delta": right["net_pnl"] - left["net_pnl"]})
    old_periods, new_periods = grouped(old.values(), lambda row:
                                       "≤2020" if str(row.get("entry_date", ""))[:4].isdigit()
                                       and int(str(row["entry_date"])[:4]) <= 2020 else
                                       "2021-2023" if str(row.get("entry_date", ""))[:4].isdigit()
                                       and int(str(row["entry_date"])[:4]) <= 2023 else "2024+"), \
                        grouped(new.values(), lambda row:
                                "≤2020" if str(row.get("entry_date", ""))[:4].isdigit()
                                and int(str(row["entry_date"])[:4]) <= 2020 else
                                "2021-2023" if str(row.get("entry_date", ""))[:4].isdigit()
                                and int(str(row["entry_date"])[:4]) <= 2023 else "2024+")
    period_groups = []
    period_order = ("≤2020", "2021-2023", "2024+")
    for period in period_order:
        left, right = aggregate(old_periods.get(period, [])), aggregate(new_periods.get(period, []))
        period_groups.append({"group": period, "baseline": left, "candidate": right,
                              "trade_count_delta": right["trade_count"] - left["trade_count"],
                              "net_pnl_delta": right["net_pnl"] - left["net_pnl"]})

    old_codes = grouped(old.values(), lambda row: row.get("code"))
    new_codes = grouped(new.values(), lambda row: row.get("code"))
    stock_groups = []
    for code in sorted(set(old_codes) | set(new_codes)):
        left, right = aggregate(old_codes.get(code, [])), aggregate(new_codes.get(code, []))
        stock_groups.append({"code": code, "baseline": left, "candidate": right,
                             "trade_count_delta": right["trade_count"] - left["trade_count"],
                             "net_pnl_delta": right["net_pnl"] - left["net_pnl"]})

    def shapes(trades):
        result = {}
        for trade in trades:
            labels = trade.get("pattern_types", []) or trade.get("pattern_labels", []) or ["unknown"]
            for label in labels:
                result.setdefault(str(label), []).append(trade)
        return result
    old_shapes, new_shapes = shapes(old.values()), shapes(new.values())
    shape_groups = []
    for shape in sorted(set(old_shapes) | set(new_shapes)):
        left, right = aggregate(old_shapes.get(shape, [])), aggregate(new_shapes.get(shape, []))
        shape_groups.append({"shape": shape, "baseline": left, "candidate": right,
                             "trade_count_delta": right["trade_count"] - left["trade_count"],
                             "net_pnl_delta": right["net_pnl"] - left["net_pnl"]})

    def single_shapes(trades):
        result = {}
        for trade in trades:
            labels = trade.get("pattern_types", []) or trade.get("pattern_labels", [])
            labels = [str(label) for label in labels]
            if len(labels) == 1:
                result.setdefault(labels[0], []).append(trade)
        return result
    old_single, new_single = single_shapes(old.values()), single_shapes(new.values())
    single_shape_groups = []
    for shape in sorted(set(old_single) | set(new_single)):
        left, right = aggregate(old_single.get(shape, [])), aggregate(new_single.get(shape, []))
        single_shape_groups.append({"shape": shape, "baseline": left, "candidate": right,
                                   "trade_count_delta": right["trade_count"] - left["trade_count"],
                                   "net_pnl_delta": right["net_pnl"] - left["net_pnl"]})

    candidate_joint_shapes = {}
    for key, trade in new.items():
        for shape in (candidate_eligible_types or {}).get(key, []):
            candidate_joint_shapes.setdefault(str(shape), []).append(trade)
    candidate_joint_shape_groups = []
    for shape in sorted(set(new_shapes) | set(candidate_joint_shapes)):
        raw_group = aggregate(new_shapes.get(shape, []))
        joint_group = aggregate(candidate_joint_shapes.get(shape, []))
        candidate_joint_shape_groups.append({"shape": shape, "raw": raw_group,
                                             "joint_eligible": joint_group})

    old_only = [old[key] for key in old_only_keys]
    new_only = [new[key] for key in new_only_keys]
    examples = []

    def add_examples(category, rows, trade_side, sort_key, limit=1):
        added = 0
        for row in sorted(rows, key=sort_key, reverse=True):
            key = identity(row)
            if any(item["identity"] == list(key) for item in examples):
                continue
            left, right = old.get(key), new.get(key)
            examples.append({"category": category, "identity": list(key),
                             "baseline_labels": (left or {}).get("pattern_types", []),
                             "candidate_labels": (right or {}).get("pattern_types", []),
                             "baseline_pnl": float(left.get("pnl", 0.0)) if left else None,
                             "candidate_pnl": float(right.get("pnl", 0.0)) if right else None,
                             "pnl_delta": (float(right.get("pnl", 0.0)) - float(left.get("pnl", 0.0)))
                                          if left and right else None})
            added += 1
            if added >= limit or len(examples) >= 5:
                return

    add_examples("G1 有、E1 无", old_only, "baseline",
                 lambda row: abs(float(row.get("pnl", 0.0))))
    add_examples("E1 有、G1 无", new_only, "candidate",
                 lambda row: abs(float(row.get("pnl", 0.0))))
    changed_keys = {tuple(row["identity"]) for row in changed}
    add_examples("共同成交，形态标签变化",
                 [old[key] for key in common_keys if key in changed_keys], "baseline",
                 lambda row: abs(float(new[identity(row)].get("pnl", 0.0))
                                 - float(row.get("pnl", 0.0))))
    add_examples("共同成交，净贡献差异",
                 [old[key] for key in common_keys if key not in changed_keys], "baseline",
                 lambda row: abs(float(new[identity(row)].get("pnl", 0.0))
                                 - float(row.get("pnl", 0.0))))
    add_examples("其他最大PnL差异", list(old.values()), "baseline",
                 lambda row: abs(float((new.get(identity(row)) or {}).get("pnl", 0.0)
                                       - float(row.get("pnl", 0.0)))))

    old_only_pnl = sum(float(old[key].get("pnl", 0.0)) for key in old_only_keys)
    new_only_pnl = sum(float(new[key].get("pnl", 0.0)) for key in new_only_keys)
    common_pnl_delta = sum(float(new[key].get("pnl", 0.0)) - float(old[key].get("pnl", 0.0))
                           for key in common_keys)
    return {"baseline_only": [list(key) for key in old_only_keys],
            "candidate_only": [list(key) for key in new_only_keys],
            "labels_changed_same_identity": changed,
            "common_trade_count": len(common_keys),
            "pnl_contribution": {"baseline_only_pnl": old_only_pnl,
                                 "candidate_only_pnl": new_only_pnl,
                                 "common_trade_pnl_delta": common_pnl_delta,
                                 "total_pnl_delta": new_only_pnl - old_only_pnl + common_pnl_delta},
            "by_decade": decade_groups, "by_period": period_groups, "by_stock": stock_groups,
            "by_shape": shape_groups, "by_single_shape": single_shape_groups,
            "candidate_joint_eligible_shape": candidate_joint_shape_groups,
            "typical_examples": examples[:5]}


def render_html(payload, output_path):
    baseline, candidate = payload["baseline"], payload["candidate"]
    case_bundle = payload.get("case_bundle") or {}
    case_json = json.dumps(case_bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    case_html = r'''<h2>engine8 六案例：结构、冻结边界与事件</h2>
<p class="status">__CASE_STATUS__</p>
<div id="casePanel"><div class="controls"><label>标的 <select id="caseSymbol"></select></label><label>实例 <select id="caseInstance"></select></label><label>区间缩放 <input id="caseZoom" type="range" min="1" max="100" value="100"></label><label>平移 <input id="casePan" type="range" min="0" max="100" value="100"></label><button id="caseReset">完整区间</button></div>
<canvas class="chart" id="caseChart"></canvas><div id="caseMetadata" class="case-meta"></div></div>
<script id="casePayload" type="application/json">__CASE_JSON__</script>
<script>
const C=JSON.parse(document.getElementById('casePayload').textContent),symbol=document.getElementById('caseSymbol'),instance=document.getElementById('caseInstance');
const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cases=C.cases||[];symbol.innerHTML=cases.map((x,i)=>`<option value="${i}">${x.kind}:${x.code}</option>`).join('');
function selected(){return cases[Number(symbol.value)||0]}
function refreshInstances(){const c=selected();instance.innerHTML=(c?.patterns||[]).map((p,i)=>`<option value="${i}">${esc(p.pattern_type)} · ${esc(p.state)} · ${esc(p.formed_at)} · ${esc(p.evidence?.find(e=>e.code==='joint_lifecycle')?.value?.instance_id||p.pattern_id)}</option>`).join('');drawCase()}
function drawCase(){const c=selected(),p=c?.patterns?.[Number(instance.value)||0],el=document.getElementById('caseChart');if(!c||!c.bars?.length)return;const box=el.getBoundingClientRect(),dpr=devicePixelRatio||1;el.width=box.width*dpr;el.height=box.height*dpr;const g=el.getContext('2d');g.scale(dpr,dpr);const w=box.width,h=box.height,all=c.bars,zoom=Number(document.getElementById('caseZoom').value||100)/100,count=Math.max(1,Math.floor(all.length*zoom)),start=Math.floor((all.length-count)*Number(document.getElementById('casePan').value||100)/100),end=Math.min(all.length,start+count),bars=all.slice(start,end),dates=new Map(all.map((b,i)=>[b[0],i]));let lo=Math.min(...bars.map(x=>x[3])),hi=Math.max(...bars.map(x=>x[2]));const life=p?.evidence?.find(e=>e.code==='joint_lifecycle')?.value,boundarySelection=p?.evidence?.find(e=>e.code==='neckline_selection')?.value||{},line=life?(life.boundary||{}):(p?.legacy_neckline||boundarySelection.selected_line||{});if(Number.isFinite(Number(line.slope))&&Number.isFinite(Number(line.intercept))){for(const i of [start,end-1]){const v=Number(line.slope)*i+Number(line.intercept);lo=Math.min(lo,v);hi=Math.max(hi,v)}}const pad={l:58,r:16,t:16,b:24},cw=w-pad.l-pad.r,priceBottom=h*.76,volumeTop=h*.81,volumeBottom=h-pad.b,span=hi-lo||1,y=v=>pad.t+(hi-v)/span*(priceBottom-pad.t),step=cw/bars.length,body=Math.max(1,Math.min(5,step*.55)),maxVol=Math.max(1,...bars.map(x=>x[5]||0));g.clearRect(0,0,w,h);g.strokeStyle='#e5eaf0';g.fillStyle='#68788a';g.font='11px system-ui';for(let j=0;j<=5;j++){const v=hi-span*j/5,yy=y(v);g.beginPath();g.moveTo(pad.l,yy);g.lineTo(w-pad.r,yy);g.stroke();g.fillText(v.toFixed(2),4,yy+4)}for(let j=0;j<bars.length;j++){const b=bars[j],x=pad.l+(j+.5)*step,up=b[4]>=b[1],color=up?'#c64d4d':'#16856c';g.strokeStyle=color;g.fillStyle=color;g.beginPath();g.moveTo(x,y(b[2]));g.lineTo(x,y(b[3]));g.stroke();const top=y(Math.max(b[1],b[4])),bottom=y(Math.min(b[1],b[4]));g.fillRect(x-body/2,top,body,Math.max(1,bottom-top));const vh=(volumeBottom-volumeTop)*(b[5]||0)/maxVol;g.globalAlpha=.55;g.fillRect(x-body/2,volumeBottom-vh,body,vh);g.globalAlpha=1}
if(p){const selection=p.evidence?.find(e=>e.code==='neckline_selection')?.value||{};let a=dates.get(selection.effective_start_date);if(a==null)a=dates.get(p.formed_at)||0;let z=dates.get(p.confirmation_date);if(z==null)z=end-1;a=Math.max(a,start);z=Math.min(z,end-1);if(line.slope!=null&&z>=a){g.strokeStyle='#087f73';g.lineWidth=2;g.beginPath();for(let i=a;i<=z;i++){const x=pad.l+(i-start+.5)*step,v=Number(line.slope||0)*i+Number(line.intercept||0);if(i===a)g.moveTo(x,y(v));else g.lineTo(x,y(v))}g.stroke()}const kp=p.key_points||{};for(let n=1;kp[`pivot_${n}_date`];n++){const i=dates.get(kp[`pivot_${n}_date`]),v=Number(kp[`pivot_${n}`]);if(i==null||i<start||i>=end||!Number.isFinite(v))continue;g.fillStyle='#365eea';g.beginPath();g.arc(pad.l+(i-start+.5)*step,y(v),4,0,Math.PI*2);g.fill()}const rawTests=selection.independent_tests||[],matchedTests=life?rawTests.filter(t=>t.line&&Math.abs(Number(t.line.slope)-Number(line.slope))<1e-9&&Math.abs(Number(t.line.intercept)-Number(line.intercept))<1e-7):rawTests,chartTests=matchedTests.length?matchedTests:(life?.boundary_anchor_tests||[]);for(const test of chartTests){const i=dates.get(test.pivot_date),v=Number(test.level);if(i==null||i<start||i>=end||!Number.isFinite(v))continue;const x=pad.l+(i-start+.5)*step,yy=y(v);g.fillStyle='#d08b20';g.beginPath();g.moveTo(x,yy-5);g.lineTo(x+5,yy);g.lineTo(x,yy+5);g.lineTo(x-5,yy);g.closePath();g.fill()}const knownDate=life?life.line_known_at:(selection.line_frozen_at||selection.freeze_date||selection.effective_start_date);const knownIndex=dates.get(knownDate);if(knownIndex!=null&&knownIndex>=start&&knownIndex<end){const x=pad.l+(knownIndex-start+.5)*step;g.strokeStyle='#365eea';g.setLineDash([5,3]);g.beginPath();g.moveTo(x,pad.t);g.lineTo(x,priceBottom);g.stroke();g.setLineDash([]);g.fillStyle='#365eea';g.fillText('line_known',Math.min(w-80,x+3),pad.t+12)}const eventRows=life?(life.events||[]).filter(ev=>ev.instance_id===life.instance_id&&ev.boundary_version_id===life.boundary_version_id):(p.evidence?.flatMap(e=>e.value?.events||[])||[]),seen=new Set();for(const ev of eventRows){const kind=String(ev.kind||'').toLowerCase(),date=ev.occurred_at||ev.date;if(!date||!/(paired|confirm|terminal|expir|invalidat|retir)/.test(kind))continue;const key=kind+':'+date;if(seen.has(key))continue;seen.add(key);const i=dates.get(String(date).slice(0,10));if(i==null||i<start||i>=end)continue;const x=pad.l+(i-start+.5)*step;g.setLineDash(kind.includes('confirm')?[4,4]:[2,3]);g.strokeStyle=kind.includes('confirm')?'#c15344':kind.includes('terminal')||kind.includes('expir')||kind.includes('invalidat')?'#9959b5':'#d08b20';g.beginPath();g.moveTo(x,pad.t);g.lineTo(x,priceBottom);g.stroke();g.setLineDash([]);g.fillStyle=g.strokeStyle;g.fillText(kind,Math.min(w-100,x+3),pad.t+25)} }
g.fillStyle='#65758a';g.fillText(bars[0][0],pad.l,h-6);g.fillText(bars.at(-1)[0],w-100,h-6);g.fillText('成交量',pad.l,volumeTop-3);
const signals=c.signals||[],selection=p?.evidence?.find(e=>e.code==='neckline_selection')?.value||{},testRows=life?.boundary_anchor_tests||boundarySelection.independent_tests||[],events=life?(life.events||[]):(p?.evidence?.flatMap(e=>e.value?.events||[])||[]);document.getElementById('caseMetadata').innerHTML=p?`<pre><b>状态与身份</b>\n${esc(JSON.stringify({pattern_id:p.pattern_id,instance_id:life?.instance_id,boundary_version_id:life?.boundary_version_id,state:p.state,formed_at:p.formed_at,confirmation_date:p.confirmation_date,last_observed_date:p.last_observed_date,own_boundary:life?.boundary||p.legacy_neckline,line_known_at:life?.line_known_at},null,2))}</pre><pre><b>颈线独立测试点</b>\n${esc(JSON.stringify(testRows,null,2))}</pre><pre><b>实际信号</b>\n${esc(JSON.stringify(signals.filter(s=>(s.instances||[]).some(x=>x.pattern_id===p.pattern_id)),null,2))}</pre><pre><b>生命周期事件：occurred/date 与 recorded_at 分开保留</b>\n${esc(JSON.stringify(events,null,2))}</pre>`:'<pre>无模式实例</pre>'}
symbol.onchange=()=>{document.getElementById('caseZoom').value=100;document.getElementById('casePan').value=100;refreshInstances()};instance.onchange=drawCase;document.getElementById('caseZoom').oninput=drawCase;document.getElementById('casePan').oninput=drawCase;document.getElementById('caseReset').onclick=()=>{document.getElementById('caseZoom').value=100;document.getElementById('casePan').value=100;drawCase()};window.addEventListener('resize',drawCase);refreshInstances();
</script>'''
    case_status = ("六票独立采集已附加。每个案例来自冻结快照；展示实际信号、结构锚点和生命周期事件，"
                   "并按事件原字段区分理论已知日期与实际记录日期。" if case_bundle.get("cases") else
                   "尚未附加 engine8 六案例采集结果；未用旧 cases-B-current 或旧 G1 输出填充。")
    case_html = case_html.replace("__CASE_STATUS__", case_status).replace("__CASE_JSON__", case_json)
    revision_mismatches = payload["data_revision_mismatches"]
    cohort_mismatches = payload.get("cohort_mismatches", [])
    trade_comparison = payload.get("trade_comparison", {})
    metrics = (
        ("Accounts", "account_count", lambda x: str(x)),
        ("Trades", "trade_count", lambda x: str(x)),
        ("Signals", "signal_count", lambda x: str(x)),
        ("Trade win rate", "win_rate", _pct),
        ("Total return", "cumulative_return", _pct),
        ("Net PnL (CNY)", "net_pnl", lambda x: f"{x:,.2f}"),
        ("Profit factor", "profit_factor", lambda x: "n/a" if x is None else f"{x:.3f}"),
        ("Average holding (sessions)", "average_holding_sessions", lambda x: "n/a" if x is None else f"{x:.2f}"),
        ("Maximum drawdown (combined equity, equal starting capital)", "portfolio_maximum_drawdown", _pct),
        ("Worst single-account maximum drawdown", "account_max_drawdown", _pct),
        ("Average capital utilization", "average_capital_utilization", _pct),
        ("Positive accounts", "positive_account_count", lambda x: str(x)),
        ("Mean account return", "account_return_mean", _pct),
        ("Median account return", "account_return_median", _pct),
        ("Minimum account return", "account_return_min", _pct),
        ("Maximum account return", "account_return_max", _pct),
    )
    table_rows = []
    for label, key, formatter in metrics:
        a, b = baseline.get(key), candidate.get(key)
        delta = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        delta_text = "n/a" if delta is None else (_pct(delta) if "return" in key or key == "win_rate"
                                                   else f"{delta:,.3f}" if key == "profit_factor"
                                                   else f"{delta:,.2f}")
        table_rows.append(f"<tr><th>{html.escape(label)}</th><td>{formatter(a)}</td>"
                          f"<td>{formatter(b)}</td><td>{delta_text}</td></tr>")
    links = []
    for label, key in (("G1 baseline report", "baseline_source"),
                       ("E1 candidate report", "candidate_source")):
        target = html.escape(str(payload.get(key, "")), quote=True)
        links.append(f'<a href="{target}">{html.escape(label)}</a>')
    mismatch_rows = "".join(
        f"<tr><td>{html.escape(kind)}</td><td>{html.escape(code)}</td>"
        f"<td>{html.escape(str(old))}</td><td>{html.escape(str(new))}</td></tr>"
        for kind, code, old, new in revision_mismatches)
    group_rows = []
    for group_name, key in (("Asset", "asset_groups"), ("Pattern", "shape_groups")):
        names = sorted(set(baseline[key]) | set(candidate[key]))
        for name in names:
            values = []
            for result in (baseline, candidate):
                group = result[key].get(name, {})
                count = group.get("trade_count", 0)
                wins = group.get("wins", 0)
                rate = wins / count if count else None
                values.append(f"{count} / {_pct(rate)} / {group.get('pnl', 0.0):,.2f}")
            group_rows.append(f"<tr><td>{html.escape(group_name)} · {html.escape(name)}</td>"
                              f"<td>{values[0]}</td><td>{values[1]}</td></tr>")
    def detailed_group_rows(groups, name_key):
        rows = []
        for group in groups:
            left, right = group["baseline"], group["candidate"]
            rows.append(
                f"<tr><td>{html.escape(str(group[name_key]))}</td>"
                f"<td>{left['trade_count']} / {_pct(left['win_rate'])} / {left['net_pnl']:,.2f}</td>"
                f"<td>{right['trade_count']} / {_pct(right['win_rate'])} / {right['net_pnl']:,.2f}</td>"
                f"<td>{group['trade_count_delta']:+d} / {group['net_pnl_delta']:+,.2f}</td></tr>")
        return "".join(rows)
    shape_by_name = {row["shape"]: row for row in trade_comparison.get("by_shape", [])}
    eligible_shape_rows = []
    for row in trade_comparison.get("candidate_joint_eligible_shape", []):
        baseline_raw = shape_by_name.get(row["shape"], {}).get("baseline", {})
        raw, eligible = row["raw"], row["joint_eligible"]
        eligible_shape_rows.append(
            f"<tr><td>{html.escape(row['shape'])}</td>"
            f"<td>{baseline_raw.get('trade_count', 0)} / {baseline_raw.get('net_pnl', 0.0):,.2f}</td>"
            f"<td>{raw['trade_count']} / {raw['net_pnl']:,.2f}</td>"
            f"<td>{eligible['trade_count']} / {_pct(eligible['win_rate'])} / {eligible['net_pnl']:,.2f}</td></tr>")
    single_shape_rows = detailed_group_rows(trade_comparison.get("by_single_shape", []), "shape")
    signal_audit = payload.get("candidate_signal_eligibility_audit", {})
    eligibility_types = sorted(set(signal_audit.get("raw_pattern_tag_counts", {})) |
                               set(signal_audit.get("joint_eligible_tag_counts", {})) |
                               set(signal_audit.get("joint_ineligible_tag_counts", {})) |
                               set(signal_audit.get("joint_unknown_tag_counts", {})))
    signal_eligibility_rows = "".join(
        f"<tr><td>{html.escape(shape)}</td>"
        f"<td>{signal_audit.get('raw_pattern_tag_counts', {}).get(shape, 0)}</td>"
        f"<td>{signal_audit.get('joint_eligible_tag_counts', {}).get(shape, 0)}</td>"
        f"<td>{signal_audit.get('joint_ineligible_tag_counts', {}).get(shape, 0)}</td>"
        f"<td>{signal_audit.get('joint_unknown_tag_counts', {}).get(shape, 0)}</td>"
        f"<td>{signal_audit.get('flag_lifecycle_true_side_evidence_counts', {}).get(shape, 0)}</td></tr>"
        for shape in eligibility_types)
    pnl_contribution = trade_comparison.get("pnl_contribution", {})
    contribution_html = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{pnl_contribution.get(key, 0.0):+,.2f}</td></tr>"
        for label, key in (("G1-only executed trade PnL", "baseline_only_pnl"),
                           ("E1-only executed trade PnL", "candidate_only_pnl"),
                           ("Common identity PnL change, E1−G1", "common_trade_pnl_delta"),
                           ("Net PnL difference, E1−G1", "total_pnl_delta")))
    def execution_metric(summary, key):
        value = summary["trade_diagnostics"].get(key)
        return _pct(value) if key.endswith("rate") else str(value)
    execution_rows = "".join(
        f"<tr><th>{label}</th><td>{execution_metric(baseline, key)}</td>"
        f"<td>{execution_metric(candidate, key)}</td></tr>"
        for label, key in (("Negative PnL trades", "negative_trade_count"),
                           ("Negative PnL rate", "negative_trade_rate"),
                           ("Holding ≤5 sessions", "holding_le_5_count"),
                           ("≤5 sessions and negative PnL", "holding_le_5_negative_count"),
                           ("Negative rate among ≤5 sessions", "holding_le_5_negative_rate"),
                           ("≤5-session losses / all negative trades", "holding_le_5_negative_share_of_all_losses"),
                           ("Exit price below stored stop price", "exit_below_stop_price_count")))
    example_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value if value is not None else '—'))}</td>" for value in (
            example["category"], " / ".join(example["identity"]),
            ", ".join(example.get("baseline_labels", [])),
            "n/a" if example.get("baseline_pnl") is None else f"{example['baseline_pnl']:+,.2f}",
            ", ".join(example.get("candidate_labels", [])),
            "n/a" if example.get("candidate_pnl") is None else f"{example['candidate_pnl']:+,.2f}",
            "n/a" if example.get("pnl_delta") is None else f"{example['pnl_delta']:+,.2f}")) + "</tr>"
        for example in trade_comparison.get("typical_examples", []))
    status = "PASS" if not revision_mismatches and not cohort_mismatches \
        and not baseline["failure_count"] \
        and not candidate["failure_count"] else "REVIEW REQUIRED"
    acceptance_gaps = payload.get("acceptance_gaps", [])
    acceptance_status = "INCOMPLETE" if acceptance_gaps else status
    acceptance_rows = "".join(f"<li>{html.escape(str(gap))}</li>" for gap in acceptance_gaps)
    comparison_notes = payload.get("comparison_notes", [])
    comparison_note_rows = "".join(f"<li>{html.escape(str(note))}</li>" for note in comparison_notes)
    document = f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E1 matched comparison</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;color:#18212f}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border-bottom:1px solid #d9e0e8;padding:9px;text-align:left}}th{{background:#f4f6f8}}.status{{padding:12px;background:#eef4ff}}code{{overflow-wrap:anywhere}}.chart{{width:100%;height:410px;border:1px solid #d9e0e8;border-radius:8px}}.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}select{{padding:8px}}.case-meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}}.case-meta pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:10px;border-radius:6px}}</style>
<h1>E1 对照：{html.escape(payload['sample'])}</h1>
<p>截止日 {html.escape(payload['cutoff'])}；固定G1=true、S1=false；入场次日开盘；止损8%、止盈20%；冷信号缓存。</p>
<p>引擎版本：G1 {html.escape(str(payload.get('baseline_engine_version')))}；E1 {html.escape(str(payload.get('candidate_engine_version')))}。engine session-v8 是联合结构引擎版本，不是历史蔡森策略 v8。</p>
<p>E1运行：{payload.get('candidate_run_summary', {}).get('successful_account_count', 'n/a')}账户成功 / {payload.get('candidate_run_summary', {}).get('failed_account_count', 'n/a')}失败；总耗时 {payload.get('candidate_runtime', {}).get('total_seconds', '未记录')} 秒。</p>
<p>Full-run CLI wall / report post-processing / combined active wall (seconds): {html.escape(json.dumps(payload.get('full_run_measurement', {}), ensure_ascii=False))}。signal_cache_dir=null；OS page cache was not cleared。</p>
<p>36-account G1 worker matrix (separate from full1000 timing): {html.escape(json.dumps(payload.get('worker_matrix', {}), ensure_ascii=False))}</p>
<p>目标进度：胜率目标 &gt;50%，当前 {_pct(candidate.get('win_rate'))}；汇总收益目标 &gt;100%，当前 {_pct(candidate.get('cumulative_return'))}。两项目标均未达成。</p>
<p>拒单/未成交的逐信号原因未包含在 daily-research.json，本报告不从信号数与成交数差额推算原因。</p>
<p class="status">同成员/数据/账本比较：{status}。正式验收：{acceptance_status}。数据版本差异 {len(revision_mismatches)}；样本成员差异 {len(cohort_mismatches)}；baseline失败账户 {baseline['failure_count']}，candidate失败账户 {candidate['failure_count']}。</p>
<h2>未完成的验收项</h2>{f'<ul>{acceptance_rows}</ul>' if acceptance_gaps else '<p>无</p>'}
{'<h2>样本范围说明</h2><ul>' + comparison_note_rows + '</ul>' if comparison_notes else ''}
<p>{' · '.join(links)}</p>
<table><thead><tr><th>指标</th><th>G1 baseline</th><th>E1 candidate</th><th>差值 E1−G1</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>退场原因</h2><pre>{html.escape(json.dumps({'G1': baseline['exit_reason_counts'], 'E1': candidate['exit_reason_counts']}, ensure_ascii=False, indent=2))}</pre>
<h3>已成交交易风险诊断</h3><p>仅描述已成交交易；短持有和低于stop price的退出是结果统计，不证明可用过滤条件改善反事实表现。低于stop计数使用 exit_price &lt; stop_price − 1e−8。</p><table><thead><tr><th>指标</th><th>G1 matched 100</th><th>E1 paired100</th></tr></thead><tbody>{execution_rows}</tbody></table>
<h2>信号日与真实成交身份对照</h2><p>身份键为 kind / code / signal_date / entry_date；标签变化单独统计，避免把形态标签变化误算成新成交。</p><pre>{html.escape(json.dumps({'G1-only': trade_comparison.get('baseline_only', []), 'E1-only': trade_comparison.get('candidate_only', []), 'same identity, label changed': trade_comparison.get('labels_changed_same_identity', [])}, ensure_ascii=False, indent=2))}</pre>
<h3>成交身份差异与净贡献</h3><p>共同成交身份 {trade_comparison.get('common_trade_count', 0)} 笔；下表将新增/删除交易与共同身份的PnL变化分开，标签变化仍归入共同成交。</p><table><thead><tr><th>贡献项</th><th>PnL</th></tr></thead><tbody>{contribution_html}</tbody></table>
<h3>5个典型差异样本</h3><table><thead><tr><th>差异类型</th><th>身份</th><th>G1形态</th><th>G1 PnL</th><th>E1形态</th><th>E1 PnL</th><th>差值</th></tr></thead><tbody>{example_rows or '<tr><td colspan="7">无可列示成交</td></tr>'}</tbody></table>
<h2>资产与形态成交分组</h2><table><thead><tr><th>分组</th><th>G1 笔数 / 胜率 / PnL</th><th>E1 笔数 / 胜率 / PnL</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table>
<p>下列形态分组允许同一笔多标签成交计入多个形态，因此形态笔数与PnL不能相加为总值。</p>
<h2>按年代分组</h2><table><thead><tr><th>入场年代</th><th>G1笔数 / 胜率 / PnL</th><th>E1笔数 / 胜率 / PnL</th><th>交易差 / PnL差</th></tr></thead><tbody>{detailed_group_rows(trade_comparison.get('by_decade', []), 'group')}</tbody></table>
<h2>按研究期分组</h2><table><thead><tr><th>入场时期</th><th>G1笔数 / 胜率 / PnL</th><th>E1笔数 / 胜率 / PnL</th><th>交易差 / PnL差</th></tr></thead><tbody>{detailed_group_rows(trade_comparison.get('by_period', []), 'group')}</tbody></table>
<h2>按形态分组</h2><table><thead><tr><th>形态</th><th>G1笔数 / 胜率 / PnL</th><th>E1笔数 / 胜率 / PnL</th><th>交易差 / PnL差</th></tr></thead><tbody>{detailed_group_rows(trade_comparison.get('by_shape', []), 'shape')}</tbody></table>
<h3>单一raw标签成交（仅pattern_types恰为一种）</h3><p>单标签组不含多标签交易；它与上方all-tag分组口径不同，二者不可互换。</p><table><thead><tr><th>单一标签</th><th>G1笔数 / 胜率 / PnL</th><th>E1笔数 / 胜率 / PnL</th><th>交易差 / PnL差</th></tr></thead><tbody>{single_shape_rows}</tbody></table>
<h3>raw标签与候选E1自身joint eligibility对照</h3><p>G1 v4来源没有可用的历史 entry_eligible_at_confirmation 字段，因此只列raw标签。E1 eligible组仅统计该signal同一pattern自身 joint_lifecycle.entry_eligible_at_confirmation=true；标签未知或false不计入。flag_lifecycle=true只作为旁证，不补成joint eligible。</p><table><thead><tr><th>形态</th><th>G1 raw all-tag笔数 / PnL</th><th>E1 raw all-tag笔数 / PnL</th><th>E1 joint-eligible成交数 / 胜率 / PnL</th></tr></thead><tbody>{''.join(eligible_shape_rows)}</tbody></table>
<h3>E1原始signal标签资格审计</h3><p>从候选100账户文件读取 {signal_audit.get('account_file_count', 0)} / {signal_audit.get('expected_account_file_count', 0)} 个账户、{signal_audit.get('signal_count', 0)} / {signal_audit.get('expected_signal_count', 0)} 条signals；未重新识别。raw同signal多标签 {signal_audit.get('multi_tag_signal_count', 0)} 条；含joint eligible和ineligible sibling的signal {signal_audit.get('signals_with_joint_eligible_and_ineligible_siblings', 0)} 条。</p><table><thead><tr><th>形态</th><th>Raw pattern标签</th><th>自身 joint eligible=true</th><th>自身 joint eligible=false</th><th>自身 joint eligibility未知</th><th>flag lifecycle true旁证</th></tr></thead><tbody>{signal_eligibility_rows}</tbody></table>
<details><summary>按股票分组（100只）</summary><table><thead><tr><th>代码</th><th>G1笔数 / 胜率 / PnL</th><th>E1笔数 / 胜率 / PnL</th><th>交易差 / PnL差</th></tr></thead><tbody>{detailed_group_rows(trade_comparison.get('by_stock', []), 'code')}</tbody></table></details>
<h2>账户数据版本不匹配</h2><table><thead><tr><th>类别</th><th>代码</th><th>G1</th><th>E1</th></tr></thead><tbody>{mismatch_rows or '<tr><td colspan="4">无</td></tr>'}</tbody></table>
<h2>报告输入来源</h2><pre>{html.escape(json.dumps(payload.get('source_artifacts', {}), ensure_ascii=False, indent=2))}</pre>
{case_html}
<p>回报目标仍需按账户账目与完整样本解读；本报告只描述同成员、同输入快照的识别版本差异。</p>
</html>"""
    Path(output_path).write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-hybrid-baseline", action="store_true",
                        help="Compose existing G1 stock and corrected-tick ETF reports")
    parser.add_argument("--hybrid-stock-report", type=Path)
    parser.add_argument("--hybrid-etf-report", type=Path)
    parser.add_argument("--hybrid-output", type=Path)
    parser.add_argument("--collect-cases", action="store_true",
                        help="Collect six frozen engine8 cases and exit without running a backtest")
    parser.add_argument("--case-root", type=Path,
                        help="Frozen six-case root for --collect-cases")
    parser.add_argument("--case-output", type=Path,
                        help="New compact JSON output path for --collect-cases")
    parser.add_argument("--case-diagnostics-output", type=Path,
                        help="Optional separate full patterns/signals JSON for --collect-cases")
    parser.add_argument("--cases-json", type=Path,
                        help="Lifecycle cases bundle matching the candidate engine version")
    parser.add_argument("--sample", choices=("paired100", "all1000"))
    parser.add_argument("--baseline", type=Path,
                        help="Completed G1 daily-research.json from the same frozen snapshot")
    parser.add_argument("--candidate", type=Path,
                        help="Completed E1 daily-research.json")
    parser.add_argument("--paired-members", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "research/universes/paired_100_v8_v12_v16_stock_cohort_v1.json")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--acceptance-gap", action="append", default=[],
                        help="Acceptance gate still incomplete; repeat for multiple gaps")
    parser.add_argument("--comparison-note", action="append", default=[],
                        help="Scope note to preserve in the generated report")
    parser.add_argument("--audit-evidence", action="append", type=Path, default=[],
                        help="Independent JSON audit artifact to preserve in the output JSON")
    parser.add_argument("--worker-matrix", type=Path,
                        help="Completed 36-account worker-policy-result.json")
    parser.add_argument("--full-run-cli-wall-seconds", type=float,
                        help="Externally measured full E1 CLI wall time")
    parser.add_argument("--full-run-process-tree-peak-bytes", type=int,
                        help="Externally measured full E1 process-tree peak RSS")
    args = parser.parse_args()
    if args.build_hybrid_baseline:
        missing = [name for name, value in (("--hybrid-stock-report", args.hybrid_stock_report),
                                            ("--hybrid-etf-report", args.hybrid_etf_report),
                                            ("--hybrid-output", args.hybrid_output)) if value is None]
        if missing:
            parser.error("--build-hybrid-baseline requires " + " and ".join(missing))
        payload = compose_hybrid_g1_baseline(
            load_json(args.hybrid_stock_report), load_json(args.hybrid_etf_report),
            stock_source=args.hybrid_stock_report.resolve(),
            etf_source=args.hybrid_etf_report.resolve())
        payload["source_artifacts"] = {
            "stock_report": file_metadata(args.hybrid_stock_report),
            "etf_report": file_metadata(args.hybrid_etf_report),
            "renderer": file_metadata(Path(__file__), include_hash=True)}
        args.hybrid_output.parent.mkdir(parents=True, exist_ok=True)
        args.hybrid_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        print(json.dumps({"output": str(args.hybrid_output.resolve()),
                          "account_count": len(payload["account_summaries"]),
                          "trade_count": len(payload["trades"]),
                          "timing": payload["composition"]["timing"]}, ensure_ascii=False))
        return
    if args.collect_cases:
        missing = [name for name, value in (("--case-root", args.case_root),
                                            ("--case-output", args.case_output)) if value is None]
        if missing:
            parser.error("--collect-cases requires " + " and ".join(missing))
        payload = collect_cases(args.case_root, args.case_output, args.case_diagnostics_output)
        print(json.dumps({"status": "collected", "output": str(args.case_output.resolve()),
                          "case_count": len(payload["cases"]),
                          "engine_version": payload["engine_version"]}, ensure_ascii=False))
        return
    missing = [name for name, value in (("--sample", args.sample), ("--baseline", args.baseline),
                                        ("--candidate", args.candidate), ("--output-json", args.output_json),
                                        ("--output-html", args.output_html)) if value is None]
    if missing:
        parser.error("comparison mode requires " + ", ".join(missing))
    postprocess_started = time.perf_counter()
    baseline_report, candidate_report = load_json(args.baseline), load_json(args.candidate)
    members = load_json(args.paired_members) if args.sample == "paired100" else None
    baseline = summarize(baseline_report, members)
    candidate = summarize(candidate_report, members)
    baseline_evidence, _, baseline_account_files = audit_signal_eligibility(
        baseline_report, members, include_signals=False)
    signal_eligibility, candidate_eligible_types, candidate_account_files = \
        audit_signal_eligibility(candidate_report, members, include_signals=True)
    baseline["portfolio_maximum_drawdown"] = baseline_evidence["portfolio_maximum_drawdown"]
    candidate["portfolio_maximum_drawdown"] = signal_eligibility["portfolio_maximum_drawdown"]
    mismatches = []
    for key, old_revision in baseline["data_revisions"].items():
        new_revision = candidate["data_revisions"].get(key)
        if old_revision != new_revision:
            kind, code = key.split(":", 1)
            mismatches.append((kind, code, old_revision, new_revision))
    if members is not None:
        expected_keys = {("stock", str(code)) for code in members.get("stocks", [])}
        expected_keys |= {("etf", str(code)) for code in members.get("etfs", [])}
    else:
        expected_keys = _account_keys(baseline)
    cohort_mismatches = []
    for version, observed in (("baseline", _account_keys(baseline)),
                              ("candidate", _account_keys(candidate))):
        cohort_mismatches.extend((version, kind, code, "missing")
                                 for kind, code in sorted(expected_keys - observed))
        cohort_mismatches.extend((version, kind, code, "unexpected")
                                 for kind, code in sorted(observed - expected_keys))
    for version, result in (("baseline", baseline), ("candidate", candidate)):
        if result["account_count"] != len(expected_keys):
            cohort_mismatches.append((version, "account_count", result["account_count"],
                                      "expected", len(expected_keys)))
    trade_comparison = compare_trades(baseline, candidate, candidate_eligible_types)
    payload = {
        "sample": args.sample,
        "cutoff": str(candidate_report.get("date_to") or candidate_report.get("parameters", {}).get("date_to")),
        "baseline_source": str(args.baseline.resolve()),
        "candidate_source": str(args.candidate.resolve()),
        "baseline_engine_version": baseline_report.get("engine_version"),
        "candidate_engine_version": candidate_report.get("engine_version"),
        "baseline_recognition_experiment": baseline_report.get("recognition_experiment"),
        "candidate_recognition_experiment": candidate_report.get("recognition_experiment"),
        "candidate_runtime": candidate_report.get("timings", {}),
        "candidate_run_identity": {
            "run_id": candidate_report.get("run_id"), "status": candidate_report.get("status"),
            "universe": candidate_report.get("universe"), "parameters": candidate_report.get("parameters"),
            "versions": candidate_report.get("versions"), "memory": candidate_report.get("memory"),
            "source_rule_version": candidate_report.get("source_rule_version"),
            "account_indices": [{"kind": str(row.get("kind")), "code": str(row.get("code"))}
                                for row in candidate.get("accounts", [])],
        },
        "full_run_measurement": {
            "full_run_cli_wall_seconds": args.full_run_cli_wall_seconds,
            "process_tree_peak_bytes": args.full_run_process_tree_peak_bytes,
            "candidate_reported_total_seconds": candidate_report.get("timings", {}).get("total_seconds"),
            "signal_cache_dir": candidate_report.get("parameters", {}).get("signal_cache_dir"),
            "os_page_cache_cleared": False,
            "cold_run_definition": "signal_cache_dir is None; OS page cache was not cleared",
        },
        "candidate_run_summary": {key: candidate_report.get("summary", {}).get(key)
                                  for key in ("successful_account_count", "failed_account_count",
                                              "account_count", "trade_count", "win_rate",
                                              "cumulative_return", "profit_factor",
                                              "maximum_drawdown", "capital_utilization")},
        "goal_progress": {
            "win_rate_threshold_exclusive": 0.5,
            "total_return_threshold_exclusive": 1.0,
            "candidate_win_rate": candidate.get("win_rate"),
            "candidate_total_return": candidate.get("cumulative_return"),
            "win_rate_met": candidate.get("win_rate") is not None and candidate["win_rate"] > 0.5,
            "total_return_met": candidate.get("cumulative_return") is not None
            and candidate["cumulative_return"] > 1.0,
        },
        "data_revision_mismatches": mismatches,
        "cohort_mismatches": cohort_mismatches,
        "trade_comparison": trade_comparison,
        "candidate_signal_eligibility_audit": signal_eligibility,
        "acceptance_gaps": list(args.acceptance_gap),
        "comparison_notes": args.comparison_note,
        "baseline": baseline, "candidate": candidate,
    }
    if args.audit_evidence:
        payload["independent_audits"] = [audit_evidence(path) for path in args.audit_evidence]
    if (signal_eligibility["missing_account_file_count"]
            or signal_eligibility["account_file_count"] != signal_eligibility["expected_account_file_count"]
            or not signal_eligibility["signal_count_matches"]):
        payload["acceptance_gaps"].append(
            "Stored candidate account signal evidence is incomplete or does not match report signal counts.")
    unknown_tag_count = sum(signal_eligibility["joint_unknown_tag_counts"].values())
    if unknown_tag_count:
        payload["acceptance_gaps"].append(
            f"{unknown_tag_count} candidate signal pattern tags lack known own joint eligibility; they are not inferred as eligible.")
    payload["source_artifacts"] = {
        "baseline_report": file_metadata(args.baseline),
        "candidate_report": file_metadata(args.candidate),
        "candidate_account_evidence": candidate_account_files,
        "baseline_account_evidence": baseline_account_files,
        "renderer": file_metadata(Path(__file__), include_hash=True),
    }
    if args.worker_matrix:
        payload["worker_matrix"] = worker_matrix_evidence(args.worker_matrix)
        payload["source_artifacts"]["worker_matrix"] = file_metadata(args.worker_matrix, include_hash=True)
    if args.cases_json:
        case_bundle = load_json(args.cases_json)
        try:
            validate_case_bundle(case_bundle, candidate_report.get("engine_version"))
        except ValueError as exc:
            parser.error(f"--cases-json {exc}")
        payload["case_bundle"] = case_bundle
        payload["case_source"] = str(args.cases_json.resolve())
        payload["source_artifacts"]["case_bundle"] = file_metadata(args.cases_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(payload, args.output_html)
    report_postprocess_seconds = time.perf_counter() - postprocess_started
    full_run_seconds = args.full_run_cli_wall_seconds or \
        float(candidate_report.get("timings", {}).get("total_seconds", 0.0))
    payload["full_run_measurement"]["report_postprocess_wall_seconds"] = report_postprocess_seconds
    payload["full_run_measurement"]["full_run_plus_report_active_wall_seconds"] = \
        full_run_seconds + report_postprocess_seconds
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(payload, args.output_html)
    report_postprocess_seconds = time.perf_counter() - postprocess_started
    payload["full_run_measurement"]["report_postprocess_wall_seconds"] = report_postprocess_seconds
    payload["full_run_measurement"]["full_run_plus_report_active_wall_seconds"] = \
        full_run_seconds + report_postprocess_seconds
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_html": str(args.output_html),
                      "data_revision_mismatches": len(mismatches),
                      "cohort_mismatches": len(cohort_mismatches),
                      "baseline_accounts": baseline["account_count"],
                      "candidate_accounts": candidate["account_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
