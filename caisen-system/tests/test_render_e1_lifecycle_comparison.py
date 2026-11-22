import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.render_e1_lifecycle_comparison import summarize
from scripts import render_e1_lifecycle_comparison as renderer


def test_summarize_applies_same_paired_members_and_serializes_revisions():
    report = {
        "account_summaries": [
            {"kind": "stock", "code": "000001", "status": "complete",
             "provenance": {"data_revision": "r1"}, "starting_cash": 100,
             "ending_equity": 110, "signal_count": 2,
             "metrics": {"cumulative_return": 0.1, "maximum_drawdown": 0.2,
                         "capital_utilization": 0.3}},
            {"kind": "stock", "code": "000002", "status": "complete",
             "provenance": {"data_revision": "r2"}, "starting_cash": 100,
             "ending_equity": 90, "signal_count": 5,
             "metrics": {"cumulative_return": -0.1}},
        ],
        "trades": [
            {"kind": "stock", "code": "000001", "signal_date": "2024-01-01",
             "entry_date": "2024-01-02", "pnl": 10, "holding_sessions": 4,
             "pattern_types": ["w_bottom"]},
            {"kind": "stock", "code": "000002", "signal_date": "2024-01-03",
             "entry_date": "2024-01-04", "pnl": -10, "holding_sessions": 2,
             "pattern_types": ["bull_flag"]},
        ],
    }

    result = summarize(report, {"stocks": ["000001"], "etfs": []})

    assert result["account_count"] == 1
    assert result["trade_count"] == 1
    assert result["data_revisions"] == {"stock:000001": "r1"}
    assert result["average_holding_sessions"] == 4
    assert result["account_max_drawdown"] == 0.2
    json.dumps(result)


def test_summarize_keeps_trade_attribution_without_large_raw_tag_payload():
    report = {"account_summaries": [{"kind": "stock", "code": "000001",
        "status": "complete", "starting_cash": 100, "ending_equity": 101,
        "metrics": {}, "provenance": {}}],
        "trades": [{"kind": "stock", "code": "000001", "signal_date": "2020-01-01",
            "entry_date": "2020-01-02", "pnl": 1, "pattern_types": ["w_bottom"],
            "pattern_tags": [{"evidence": "very large"}]}]}

    result = summarize(report)

    assert result["trades"][0]["pattern_types"] == ["w_bottom"]
    assert "pattern_tags" not in result["trades"][0]


def test_case_bundle_accepts_generic_and_legacy_schema_only_for_matching_engine():
    cases = [{"kind": kind, "code": code} for kind, code in
             [("stock", "000001"), ("stock", "000012"), ("stock", "000538"),
              ("stock", "600108"), ("etf", "159718"), ("etf", "516630")]]
    bundle = {"schema_version": "lifecycle-six-cases-v1",
              "engine_version": "caisen-joint-pattern-session-v9-flag-confirmation-history",
              "cases": cases}

    renderer.validate_case_bundle(bundle, bundle["engine_version"])
    bundle["schema_version"] = "engine8-six-cases-v1"
    legacy_engine = "caisen-joint-pattern-session-v8-confirmation-selection-freeze"
    bundle["engine_version"] = legacy_engine
    renderer.validate_case_bundle(bundle, legacy_engine)
    try:
        renderer.validate_case_bundle(bundle,
            "caisen-joint-pattern-session-v9-flag-confirmation-history")
    except ValueError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError("case bundle from a different engine version was accepted")


def test_audit_evidence_keeps_status_scope_and_source_metadata(tmp_path):
    source = tmp_path / "audit.json"
    source.write_text(json.dumps({"status": "passed", "audit_version": "ledger-v1",
                                  "account_count": 100, "trade_count": 1800,
                                  "signals_without_historically_eligible_pattern_count": 0}),
                       encoding="utf-8")

    evidence = renderer.audit_evidence(source)

    assert evidence["status"] == "passed"
    assert evidence["audit_version"] == "ledger-v1"
    assert evidence["account_count"] == 100
    assert evidence["signals_without_historically_eligible_pattern_count"] == 0
    assert evidence["source"]["size_bytes"] == source.stat().st_size


def test_worker_matrix_evidence_keeps_selected_policy_and_all_timings(tmp_path):
    result = tmp_path / "worker-policy-result.json"
    result.write_text(json.dumps({"status": "complete", "benchmark_id": "g1-matrix",
        "reference_count": 36, "selected_worker_count": 10,
        "selected_process_tree_peak_bytes": 2818539520, "speedup_vs_serial": 4.71,
        "elapsed_seconds": {"serial-reference": 624.78, "spawn10-candidate": 132.65},
        "parity_passed": True, "accounting_checks_passed": True,
        "event_parity_passed": True}), encoding="utf-8")

    evidence = renderer.worker_matrix_evidence(result)

    assert evidence["selected_worker_count"] == 10
    assert evidence["reference_count"] == 36
    assert evidence["elapsed_seconds"]["spawn10-candidate"] == 132.65
    assert evidence["timing_scope"] == "36 pinned accounts; not full1000 timing"


def test_hybrid_g1_baseline_uses_stock_rows_and_correct_tick_etf_rows():
    def account(kind, code, tick):
        return {"kind": kind, "code": code, "status": "complete",
                "provenance": {"tick_size": tick, "data_revision": f"{kind}-{code}"},
                "starting_cash": 100, "ending_equity": 110,
                "metrics": {"cumulative_return": 0.1}, "signal_count": 1}

    def report(kind, code, tick, trade_code):
        return {"date_to": "2026-10-30", "engine_version": "caisen-joint-pattern-session-v4",
                "source_rule_version": "caisen-main-bottoms-v12-structure-boundary",
                "account_summaries": [account(kind, code, tick)],
                "account_files": [f"{kind}-{code}.json"],
                "trades": [{"kind": kind, "code": trade_code, "pnl": 10}],
                "summary": {"starting_capital": 100, "ending_equity": 110}}

    hybrid = renderer.compose_hybrid_g1_baseline(
        report("stock", "000001", 0.01, "000001"),
        report("etf", "159718", 0.001, "159718"),
        expected_stock_accounts=1, expected_etf_accounts=1)

    assert {(row["kind"], row["code"]) for row in hybrid["account_summaries"]} == {
        ("stock", "000001"), ("etf", "159718")}
    assert {(row["kind"], row["code"]) for row in hybrid["trades"]} == {
        ("stock", "000001"), ("etf", "159718")}
    assert hybrid["composition"]["timing"] == "N/A; composed from existing account reports"
    assert hybrid["composition"]["tick_sizes"] == {"stock": 0.01, "etf": 0.001}


def test_hybrid_g1_baseline_rejects_wrong_etf_tick():
    def report(kind, code, tick):
        return {"date_to": "2026-10-30", "engine_version": "caisen-joint-pattern-session-v4",
                "source_rule_version": "caisen-main-bottoms-v12-structure-boundary",
                "account_summaries": [{"kind": kind, "code": code,
                    "provenance": {"tick_size": tick}}], "trades": []}

    try:
        renderer.compose_hybrid_g1_baseline(
            report("stock", "000001", 0.01), report("etf", "159718", 0.01),
            expected_stock_accounts=1, expected_etf_accounts=1)
    except ValueError as exc:
        assert "tick" in str(exc).lower()
    else:
        raise AssertionError("hybrid accepted an ETF replay with the wrong tick size")


def test_hybrid_g1_baseline_filters_old_etfs_out_of_full_g1_source():
    def account(kind, code, tick):
        return {"kind": kind, "code": code,
                "provenance": {"tick_size": tick}, "account_json": f"{kind}-{code}.json"}

    stock_report = {"date_to": "2026-10-30", "engine_version": "caisen-joint-pattern-session-v4",
        "source_rule_version": "caisen-main-bottoms-v12-structure-boundary",
        "account_summaries": [account("stock", "000001", 0.01), account("etf", "159718", 0.01)],
        "account_files": ["stock-000001.json", "etf-159718.json"],
        "trades": [{"kind": "stock", "code": "000001", "pnl": 1},
                   {"kind": "etf", "code": "159718", "pnl": 99}]}
    etf_report = {"date_to": "2026-10-30", "engine_version": "caisen-joint-pattern-session-v4",
        "source_rule_version": "caisen-main-bottoms-v12-structure-boundary",
        "account_summaries": [account("etf", "159718", 0.001)],
        "account_files": ["etf-159718-new.json"],
        "trades": [{"kind": "etf", "code": "159718", "pnl": 2}]}

    hybrid = renderer.compose_hybrid_g1_baseline(stock_report, etf_report,
        expected_stock_accounts=1, expected_etf_accounts=1)

    assert [(row["kind"], row["code"]) for row in hybrid["account_summaries"]] == [
        ("stock", "000001"), ("etf", "159718")]
    assert [row["pnl"] for row in hybrid["trades"]] == [1, 2]
    assert hybrid["account_files"] == ["stock-000001.json", "etf-159718.json"]


def test_hybrid_g1_baseline_drops_unneeded_equity_and_trade_payloads():
    def source(kind, code, tick, pnl):
        return {"date_to": "2026-10-30", "engine_version": "caisen-joint-pattern-session-v4",
            "source_rule_version": "caisen-main-bottoms-v12-structure-boundary",
            "account_summaries": [{"kind": kind, "code": code,
                "provenance": {"tick_size": tick}, "status": "complete",
                "starting_cash": 100, "ending_equity": 110, "signal_count": 1,
                "metrics": {"maximum_drawdown": 0.1}, "recognition_stats": {"large": "x"}}],
            "trades": [{"kind": kind, "code": code, "signal_date": "2020-01-01",
                "entry_date": "2020-01-02", "pnl": pnl, "pattern_types": ["w_bottom"],
                "pattern_tags": [{"evidence": "large"}], "extra_payload": "large"}],
            "daily_equity": [{"large": "x"}]}

    hybrid = renderer.compose_hybrid_g1_baseline(source("stock", "000001", 0.01, 10),
        source("etf", "159718", 0.001, 5), expected_stock_accounts=1, expected_etf_accounts=1)

    assert "daily_equity" not in hybrid
    assert "recognition_stats" not in hybrid["account_summaries"][0]
    assert hybrid["trades"][0]["pattern_types"] == ["w_bottom"]
    assert "extra_payload" not in hybrid["trades"][0]
    assert "pattern_tags" not in hybrid["trades"][0]


def test_paired100_cli_filters_candidate_to_same_members(tmp_path, monkeypatch):
    report = {
        "date_to": "2026-10-30",
        "engine_version": "caisen-joint-pattern-session-v8-confirmation-selection-freeze",
        "account_files": [str(tmp_path / "stock-000001.json")],
        "account_summaries": [
            {"kind": "stock", "code": code, "status": "complete",
             "provenance": {"data_revision": "same"}, "starting_cash": 100,
             "ending_equity": 100, "account_json": str(tmp_path / "stock-000001.json"),
             "signal_count": 0, "metrics": {}}
            for code in ("000001", "000002")
        ],
        "trades": [],
    }
    account_evidence = tmp_path / "stock-000001.json"
    account_evidence.write_text(json.dumps({"kind": "stock", "code": "000001", "signals": [],
                                            "daily_equity": [{"date": "2024-01-02", "equity": 100}],
                                            "starting_cash": 100}), encoding="utf-8")
    baseline, candidate = tmp_path / "base.json", tmp_path / "new.json"
    worker_matrix = tmp_path / "worker-policy-result.json"
    worker_matrix.write_text(json.dumps({"status": "complete", "benchmark_id": "test",
        "reference_count": 36, "selected_worker_count": 10,
        "elapsed_seconds": {"serial-reference": 100.0, "spawn10-candidate": 20.0},
        "speedup_vs_serial": 5.0}), encoding="utf-8")
    members, cases, output_json, output_html = (tmp_path / name for name in
                                               ("members.json", "cases.json", "out.json", "out.html"))
    for path in (baseline, candidate):
        path.write_text(json.dumps(report), encoding="utf-8")
    members.write_text(json.dumps({"stocks": ["000001"], "etfs": []}), encoding="utf-8")
    cases.write_text(json.dumps({"schema_version": "engine8-six-cases-v1",
                                 "engine_version": report["engine_version"], "cases": [
        {"kind": kind, "code": code, "bars": [["2024-01-02", 1, 2, 1, 2, 10]],
         "patterns": [{"pattern_id": "p", "pattern_type": "w_bottom", "state": "forming",
                       "neckline": {"slope": 0.1, "intercept": 1}, "key_points": {}}],
         "signals": [], "recognition_stats": {}}
        for kind, code in [("stock", "000001"), ("stock", "000012"), ("stock", "000538"),
                           ("stock", "600108"), ("etf", "159718"), ("etf", "516630")]]}),
        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["render", "--sample", "paired100", "--baseline", str(baseline),
                                      "--candidate", str(candidate), "--paired-members", str(members),
                                      "--cases-json", str(cases), "--output-json", str(output_json),
                                      "--output-html", str(output_html),
                                      "--worker-matrix", str(worker_matrix),
                                      "--acceptance-gap", "missing_historical_entry_eligibility",
                                      "--comparison-note", "stocks only; no ETF tick mixed"])

    renderer.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["baseline"]["account_count"] == 1
    assert payload["candidate"]["account_count"] == 1
    assert payload["baseline"]["portfolio_maximum_drawdown"] == 0.0
    assert len(payload["source_artifacts"]["candidate_account_evidence"]) == 1
    assert payload["data_revision_mismatches"] == []
    assert payload["source_artifacts"]["baseline_report"]["size_bytes"] == baseline.stat().st_size
    assert payload["worker_matrix"]["selected_worker_count"] == 10
    assert payload["source_artifacts"]["worker_matrix"]["path"] == str(worker_matrix.resolve())
    assert len(payload["source_artifacts"]["renderer"]["sha256"]) == 64
    assert payload["acceptance_gaps"] == ["missing_historical_entry_eligibility"]
    html = output_html.read_text(encoding="utf-8")
    assert "Average holding" in html
    assert "caseZoom" in html
    assert "line=life?(life.boundary||{})" in html
    assert "g.moveTo(x,y(b[2]));g.lineTo(x,y(b[3]))" in html
    assert "ev.instance_id===life.instance_id&&ev.boundary_version_id===life.boundary_version_id" in html
    assert "正式验收：INCOMPLETE" in html
    assert "Maximum drawdown (combined equity, equal starting capital)" in html
    assert "Net PnL (CNY)" in html
    assert "missing_historical_entry_eligibility" in html
    assert "stocks only; no ETF tick mixed" in html
    assert "5个典型差异样本" in html
    assert "两项目标均未达成" in html
    assert "按年代分组" in html
    assert "按股票分组（100只）" in html
    if shutil.which("node"):
        script = re.search(r"<script>\s*(.*?)</script>", html, re.S).group(1)
        script_path = tmp_path / "report.js"
        script_path.write_text(script, encoding="utf-8")
        subprocess.run(["node", "--check", str(script_path)], check=True, capture_output=True, text=True)


def test_compare_trades_decomposes_pnl_and_keeps_label_change_as_common_trade():
    def trade(code, signal, pnl, label="w_bottom", entry="2020-01-02"):
        return {"kind": "stock", "code": code, "signal_date": signal,
                "entry_date": entry, "pnl": pnl, "pattern_types": [label]}

    result = renderer.compare_trades(
        {"trades": [trade("000001", "2019-12-31", -10, entry="2019-12-31"),
                    trade("000002", "2020-01-01", 40, "bull_flag")]},
        {"trades": [trade("000003", "2020-01-01", 25),
                    trade("000002", "2020-01-01", 50, "triangle_bottom")]})

    assert result["common_trade_count"] == 1
    assert result["baseline_only"] == [["stock", "000001", "2019-12-31", "2019-12-31"]]
    assert result["candidate_only"] == [["stock", "000003", "2020-01-01", "2020-01-02"]]
    assert result["labels_changed_same_identity"][0]["identity"] == [
        "stock", "000002", "2020-01-01", "2020-01-02"]
    assert result["pnl_contribution"] == {
        "baseline_only_pnl": -10, "candidate_only_pnl": 25,
        "common_trade_pnl_delta": 10, "total_pnl_delta": 45}
    assert {row["group"] for row in result["by_decade"]} == {"2010s", "2020s"}
    assert {row["shape"] for row in result["by_shape"]} == {
        "bull_flag", "triangle_bottom", "w_bottom"}


def test_trade_diagnostics_counts_short_losses_and_exit_below_stop():
    result = renderer.trade_diagnostics([
        {"pnl": -5, "holding_sessions": 3, "exit_price": 91, "stop_price": 92},
        {"pnl": 10, "holding_sessions": 4, "exit_price": 94, "stop_price": 92},
        {"pnl": -2, "holding_sessions": 7, "exit_price": 90, "stop_price": 92},
    ])
    assert result["negative_trade_count"] == 2
    assert result["holding_le_5_count"] == 2
    assert result["holding_le_5_negative_count"] == 1
    assert result["holding_le_5_negative_rate"] == 0.5
    assert result["holding_le_5_negative_share_of_all_losses"] == 0.5
    assert result["exit_below_stop_price_count"] == 2


def test_compact_signals_uses_signal_local_lifecycle_for_alias_pattern_ids():
    canonical_patterns = [{
        "pattern_id": "source-canonical", "pattern_type": "w_bottom",
        "evidence": [{"code": "joint_lifecycle", "value": {
            "instance_id": "canonical-instance", "boundary_version_id": "canonical-boundary"}}],
    }]
    signal = {
        "signal_date": "2024-10-08", "entry_date": "2024-10-09",
        "pattern_ids": ["signal-alias"], "pattern_types": ["w_bottom"],
        "patterns": [{
            "pattern_id": "signal-alias", "pattern_type": "w_bottom",
            "evidence": [{"code": "joint_lifecycle", "value": {
                "instance_id": "alias-instance", "boundary_version_id": "alias-boundary"}}],
        }],
    }

    compact = renderer._compact_signals([signal], canonical_patterns)[0]

    assert compact["instances"] == [{"pattern_id": "signal-alias",
                                     "instance_id": "alias-instance",
                                     "boundary_version_id": "alias-boundary"}]


def test_signal_eligibility_audit_keeps_unknown_flags_out_of_joint_eligible_set(tmp_path):
    account = tmp_path / "stock-000001.json"
    def pattern(pattern_id, kind, joint, flag=None):
        evidence = [{"code": "joint_lifecycle", "value": joint}]
        if flag is not None:
            evidence.append({"code": "flag_lifecycle", "value": flag})
        return {"pattern_id": pattern_id, "pattern_type": kind, "evidence": evidence}
    account.write_text(json.dumps({
        "kind": "stock", "code": "000001", "signals": [{
            "kind": "stock", "code": "000001", "signal_date": "2024-01-01",
            "entry_date": "2024-01-02", "pattern_ids": ["w", "ihs", "flag"],
            "patterns": [pattern("w", "w_bottom", {"entry_eligible_at_confirmation": True}),
                         pattern("ihs", "inverse_head_shoulders",
                                 {"entry_eligible_at_confirmation": False}),
                         pattern("flag", "bull_flag",
                                 {"entry_eligible_at_confirmation": None},
                                 {"entry_eligible_at_confirmation": True})]
        }]}, ensure_ascii=False), encoding="utf-8")
    report = {"account_files": [str(account)], "account_summaries": [
        {"kind": "stock", "code": "000001", "signal_count": 1}]}

    audit, eligible, files = renderer.audit_signal_eligibility(
        report, {"stocks": ["000001"], "etfs": []})

    assert audit["account_file_count"] == audit["expected_account_file_count"] == 1
    assert audit["signal_count_matches"] is True
    assert audit["signals_with_joint_eligible_and_ineligible_siblings"] == 1
    assert audit["joint_ineligible_tag_counts"] == {"inverse_head_shoulders": 1}
    assert audit["joint_unknown_tag_counts"] == {"bull_flag": 1}
    assert audit["flag_lifecycle_true_side_evidence_counts"] == {"bull_flag": 1}
    assert eligible[("stock", "000001", "2024-01-01", "2024-01-02")] == ["w_bottom"]
    assert len(files) == 1


def test_collect_cases_calls_recognizer_once_per_frozen_symbol_and_keeps_tick(tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    (root / "caisen-system/config").mkdir(parents=True)
    (root / "caisen-system/config/instruments.json").write_text(
        json.dumps({"instruments": {}}), encoding="utf-8")
    rows = []
    symbols = [("stock", code) for code in ("000001", "000012", "000538", "600108")]
    symbols += [("etf", code) for code in ("159718", "516630")]
    for kind, code in symbols:
        folder = "stocks" if kind == "stock" else "etfs"
        path = root / "data" / folder / f"{code}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "date,open,high,low,close,volume\n2024-01-02,1,2,1,2,10\n"
        if kind == "etf" and code == "159718":
            external = tmp_path / "worker-snapshot" / f"{code}.csv"
            external.parent.mkdir(parents=True, exist_ok=True)
            external.write_text(content, encoding="utf-8")
            path.symlink_to(external)
        else:
            path.write_text(content, encoding="utf-8")
        stat = path.stat()
        rows.append({"kind": kind, "code": code, "path": str(path), "source": "snapshot",
                     "revision": {"mtime_ns": stat.st_mtime_ns, "size_bytes": stat.st_size}})
    (root / "case-input-manifest.json").write_text(
        json.dumps({"cutoff": "2026-10-30", "rows": rows}), encoding="utf-8")

    bars = pd.DataFrame([{ "date": pd.Timestamp("2024-01-02"), "open": 1,
                          "high": 2, "low": 1, "close": 2, "volume": 10 }])
    calls = []
    contexts = []
    def fake_load(instrument, root):
        tick = 0.01 if instrument.kind == "stock" else 0.001
        return {"bars": bars.copy(), "data_revision": "stat-revision", "tick_size": tick,
                "tick_size_source": "exchange_asset_class"}
    def fake_recognize(frame, code, kind, tick_size=None):
        calls.append((kind, code, tick_size))
        from caisen.research_config import current_recognition_experiment
        contexts.append(current_recognition_experiment().experiment_id)
        pattern = SimpleNamespace(to_dict=lambda: {
            "pattern_id": code, "pattern_type": "w_bottom", "state": "confirmed",
            "neckline": {"slope": 9.0, "intercept": 9.0},
            "key_points": {"pivot_1_date": "2024-01-02", "pivot_1": 1.0},
            "evidence": [{"code": "joint_lifecycle", "value": {
                             "instance_id": "instance-A", "boundary_version_id": "boundary-A",
                                 "boundary": {"slope": 0.01, "intercept": 2.0},
                                 "boundary_anchor_tests": [
                                     {"pivot_date": "2024-01-02", "known_date": "2024-01-03",
                                      "level": 1.5, "line": {"slope": 0.01, "intercept": 2.0}},
                                 ], "events": [
                                 {"kind": "paired", "date": "2024-01-02",
                                  "instance_id": "instance-A", "boundary_version_id": "boundary-A"},
                                 {"kind": "confirmed", "date": "2024-01-02",
                                  "instance_id": "other", "boundary_version_id": "boundary-A"},
                             ]}},
                             {"code": "neckline_selection", "value": {
                                 "selected_line": {"slope": 9.0, "intercept": 9.0},
                                 "effective_start_date": "2024-01-02",
                                 "independent_tests": [
                                     {"pivot_date": "2024-01-02", "known_date": "2024-01-03",
                                      "level": 1.5, "line": {"slope": 0.01, "intercept": 2.0}},
                                     {"pivot_date": "2024-01-02", "known_date": "2024-01-03",
                                      "level": 1.8, "line": {"slope": 9.0, "intercept": 9.0}},
                                 ], "rejected_candidates": [{"huge": "discard"}] }},
                             {"code": "neckline_tests", "value": [
                                 {"group_id": "selected", "pivot_date": "2024-01-02",
                                  "known_date": "2024-01-03", "level": 1.5,
                                  "line": {"slope": 0.01, "intercept": 2.0}},
                             ]},
                         {"code": "unneeded", "value": {"large": "x"}}]})
        return {"patterns": [pattern], "signals": [{"code": code, "kind": kind,
                "signal_date": "2024-01-02", "entry_date": "2024-01-03",
                "pattern_ids": [code], "pattern_types": ["w_bottom"],
                "pattern_tags": [{"pattern_id": code, "pattern_type": "w_bottom",
                                   "main_structure": {"huge": "discard"}}],
                "pattern": {"large": "x"}}], "performance": {}}
    monkeypatch.setattr("caisen.daily_research.load_research_bars", fake_load)
    monkeypatch.setattr("caisen.daily_incremental.recognize_daily_history", fake_recognize)
    monkeypatch.setenv("CAISEN_INSTRUMENTS", "restore-this-value")

    output = root / "collected.json"
    payload = renderer.collect_cases(root, output)

    assert len(calls) == 6
    assert contexts == ["internal_rebound_coverage_g1_v1"] * 6
    assert os.environ["CAISEN_INSTRUMENTS"] == "restore-this-value"
    assert {tick for kind, _, tick in calls if kind == "stock"} == {0.01}
    assert {tick for kind, _, tick in calls if kind == "etf"} == {0.001}
    assert len(payload["cases"]) == 6
    assert "pattern" not in payload["cases"][0]["signals"][0]
    assert all(item["code"] != "unneeded" for item in payload["cases"][0]["patterns"][0]["evidence"])
    case = payload["cases"][0]
    assert len(case["bars"][0]) == 6  # date, open, high, low, close, volume
    assert case["bars"][0][2:4] == [2.0, 1.0]  # candle wick endpoints survive export
    assert case["bars"][0][0] == "2024-01-02"  # real candle date survives export
    assert case["patterns"][0]["key_points"]["pivot_1_date"] == "2024-01-02"
    assert "neckline" not in case["patterns"][0]
    pattern = case["patterns"][0]
    lifecycle = next(item["value"] for item in pattern["evidence"]
                     if item["code"] == "joint_lifecycle")
    assert renderer._chart_boundary(pattern) == {"slope": 0.01, "intercept": 2.0}
    conflict = {"neckline": {"slope": 9.0, "intercept": 9.0},
                "evidence": [{"code": "joint_lifecycle", "value": lifecycle}]}
    assert renderer._chart_boundary(conflict) == {"slope": 0.01, "intercept": 2.0}
    assert [event["kind"] for event in renderer._matching_lifecycle_events(lifecycle)] == ["paired"]
    assert lifecycle["boundary_anchor_tests"] == [
        {"group_id": "selected", "pivot_date": "2024-01-02", "known_date": "2024-01-03",
         "level": 1.5}]
    selection = next(item["value"] for item in pattern["evidence"]
                     if item["code"] == "neckline_selection")
    assert "independent_tests" not in selection
    assert "data_version" not in case["patterns"][0]
    assert "rejected_candidates" not in case["patterns"][0]["evidence"][0]["value"]
    assert "main_structure" not in case["signals"][0]["instances"][0]
    assert case["signals"][0]["instances"][0]["instance_id"] == "instance-A"
    assert case["signals"][0]["pattern_labels"] == ["w_bottom"]
    assert "chart_events" not in pattern and "chart_tests" not in pattern
    assert Path(case["input_path"]).is_symlink()


def test_collect_cases_cli_imports_project_package_from_direct_script(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(renderer.__file__).resolve()), "--collect-cases",
         "--case-root", str(tmp_path), "--case-output", str(tmp_path / "out.json")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError: No module named 'caisen'" not in result.stderr
    assert "case-input-manifest.json" in result.stderr
