import json
import sys
import types
from pathlib import Path

import pandas as pd

from caisen.market import InstrumentFile
from caisen.daily_research import create_universe, run_daily_research, _load_universe
from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades


def test_universe_sampling_is_deterministic_sorted_and_without_replacement(tmp_path):
    instruments = [InstrumentFile(code=str(i), kind=kind, path=f"{kind}/{i}.csv")
                   for kind, count in (("stock", 810), ("etf", 205))
                   for i in range(count)]
    path = tmp_path / "universe.json"

    result = create_universe(instruments, path)
    saved = json.loads(path.read_text())

    assert len(result["stocks"]) == 800
    assert len(result["etfs"]) == 200
    assert result["stocks"] == sorted(set(result["stocks"]))
    assert result["etfs"] == sorted(set(result["etfs"]))
    assert saved["seed"] == 20261101
    assert saved["date"] == "2026-11-01"
    assert create_universe([], path) == saved


def test_universe_creation_rejects_insufficient_inventory(tmp_path):
    instruments = [InstrumentFile(code="only-one", kind="stock", path="x")]
    try:
        create_universe(instruments, tmp_path / "universe.json")
    except ValueError as exc:
        assert "inventory" in str(exc)
    else:
        raise AssertionError("insufficient inventory must not produce a universe")


def test_production_universe_rejects_wrong_id_or_sample_counts(tmp_path):
    path = tmp_path / "universe.json"
    path.write_text(json.dumps({"id": "fixture", "stocks": ["600001"], "etfs": []}))
    try:
        _load_universe(path)
    except ValueError as exc:
        assert "universe" in str(exc)
    else:
        raise AssertionError("production universe must have the fixed ID and 800/200 counts")


def test_runner_cuts_off_future_bars_and_writes_once_with_grouped_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "caisen.sqlite3"))
    universe = {"id": "fixture", "seed": 20261101, "date": "2026-11-01",
                "stocks": ["600001"], "etfs": ["510001"]}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe))
    frame = pd.DataFrame([
        ("2026-10-28", 10, 10, 10, 10, 1000),
        ("2026-10-29", 10, 10, 10, 10, 1000),
        ("2026-10-30", 10, 10.1, 9.9, 10, 1000),
        ("2026-11-02", 11, 11, 11, 11, 1000),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    loaded = []

    def fake_load(instrument, root):
        loaded.append(instrument.code)
        return {"bars": frame.copy(), "source": "local_csv", "data_revision": "mtime_ns:1,size:2",
                "first_date": "2026-10-29", "last_date": "2026-11-02", "normalized_rows": len(frame),
                "tick_size": 0.01, "tick_size_source": "default"}

    incremental = types.ModuleType("caisen.daily_incremental")
    class FakeSignals(list):
        instance_events = []
        entry_events = []
    def generate(bars, code, kind, trade_start=None, progress_callback=None,
                 recognizer="v7", performance=None):
        assert bars["date"].max().strftime("%Y-%m-%d") == "2026-10-30"
        if progress_callback:
            progress_callback({"phase": "recognition", "bars_completed": len(bars),
                               "bars_total": len(bars), "current_date": "2026-10-30"})
        result = FakeSignals([{"signal_date": "2026-10-28", "entry_date": "2026-10-29",
                 "pattern_ids": ["p1", "p2"], "pattern_types": ["w_bottom", "triangle_bottom"],
                 "pattern": {"evidence": [{"code": "neckline_selection", "value": {"frozen": True}}]},
                 "pattern_tags": [{"pattern_id": "p1", "pattern_type": "w_bottom"},
                                  {"pattern_id": "p2", "pattern_type": "triangle_bottom"}]}])
        result.instance_events = [{"instance_id": code + ":p1", "event_id": code + ":e1",
                                   "event_seq": 1, "event_date": "2026-10-28",
                                   "state": "confirmed", "entry_eligible": True,
                                   "pattern": {"pattern_id": code + ":p1", "pattern_type": "w_bottom"}}]
        result.entry_events = list(result.instance_events)
        return result
    incremental.generate_daily_signals = generate
    monkeypatch.setitem(sys.modules, "caisen.daily_incremental", incremental)
    monkeypatch.setattr("caisen.daily_research.discover_instruments", lambda root: [
        InstrumentFile("600001", "stock", str(tmp_path / "600001.csv")),
        InstrumentFile("510001", "etf", str(tmp_path / "510001.csv")),
    ])
    monkeypatch.setattr("caisen.daily_research.load_research_bars", fake_load)
    writes = []
    from caisen import daily_research
    atomic_write = daily_research._atomic_write
    def counted_write(path, content):
        writes.append(str(path))
        atomic_write(path, content)
    monkeypatch.setattr(daily_research, "_atomic_write", counted_write)

    payload = run_daily_research(universe_path, tmp_path / "out", date_to="2026-10-30", workers=2,
                                 allow_test_universe=True, worker_source_override="fixture-policy")

    assert loaded == ["600001", "510001"]
    assert payload["summary"]["trade_count"] == 2
    assert payload["parameters"]["workers"] == 2
    assert payload["parameters"]["worker_source"] == "fixture-policy"
    assert payload["parameters"]["recognizer"] == "boundary_first"
    assert payload["parameters"]["execution_mode"] == "in_process"
    assert payload["recognition_rules"]["candidate_reversal_atr_multiple"] == 2.0
    assert payload["recognition_rules"]["breakout_close_buffer_previous_atr_multiple"] == 1.0
    assert payload["trading_rules"]["lot_size"] == 100
    assert payload["summary"]["distinct_traded_patterns"] == 4
    assert payload["groups"]["kind"]["stock"]["trade_count"] == 1
    assert payload["groups"]["kind"]["etf"]["trade_count"] == 1
    assert payload["summary"]["account_count"] == 2
    assert payload["summary"]["pattern_instance_event_count"] == 2
    assert payload["summary"]["entry_event_count"] == 2
    assert payload["artifacts"]["json_path"]
    assert (tmp_path / "out" / "daily-research.json").exists()
    assert (tmp_path / "out" / "daily-research.html").exists()
    assert writes.count(str(tmp_path / "out" / "daily-research.json")) == 1
    assert len(json.loads((tmp_path / "out" / "daily-research.json").read_text())["account_summaries"]) == 2
    progress = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert progress["completed"] == progress["total"] == 2
    assert progress["active_symbols"] == []
    assert len(list((tmp_path / "out" / "accounts").glob("*.json"))) == 2
    assert all(a["accounting_checks"][key] for a in payload["account_summaries"]
               for key in ("cash_plus_position_equals_equity", "realized_pnl_matches_ending_equity"))
    for row in payload["account_summaries"]:
        account_path = row["account_json"]
        assert writes.count(account_path) == 1
        with open(account_path, encoding="utf-8") as handle:
            account = json.load(handle)
        assert account["signals"][0]["pattern"]["evidence"][0]["value"]["frozen"] is True
        assert account["instance_events"][0]["entry_eligible"] is True
        assert account["engine_version"] == payload["engine_version"]
    from caisen.store import get_pattern_instance_events
    stored_events = get_pattern_instance_events(payload["run_id"], path=tmp_path / "caisen.sqlite3")
    assert len(stored_events) == 2
    assert all(event["entry_eligible"] for event in stored_events)
    html_report = (tmp_path / "out" / "daily-research.html").read_text(encoding="utf-8")
    assert "形态状态事件" in html_report and "首次突破买入事件" in html_report
    assert payload["strategy_version"]
    assert payload["recognizer_version"]
    assert payload["source_rule_version"]
    assert payload["date_to"] <= payload["parameters"]["client_date"]
    assert payload["daily_equity"]


def test_full_html_report_embeds_versions_failures_and_instance_timelines(tmp_path, monkeypatch):
    from html.parser import HTMLParser
    from caisen import daily_research
    from caisen.daily_research import _report_html
    monkeypatch.setattr(daily_research, "discover_instruments",
                        lambda root: [InstrumentFile("000538", "stock", "fixture.csv")])
    case_bars = pd.DataFrame({"date": pd.date_range("2025-12-01", periods=100),
                              "open": [9.5] * 100, "high": [10.5] * 100,
                              "low": [9.0] * 100, "close": [10.0] * 100,
                              "volume": [1000] * 100})
    monkeypatch.setattr(daily_research, "load_research_bars",
                        lambda instrument, root: {"bars": case_bars, "data_revision": "r1"})

    account_path = tmp_path / "accounts" / "stock-000538.json"
    account_path.parent.mkdir(parents=True)
    pattern = {"pattern_type": "w_bottom", "neckline": {"slope": 0.0, "intercept": 10.0},
               "key_points": {"pivot_1_index": 20, "pivot_1": 9.0,
                              "pivot_2_index": 40, "pivot_2": 10.0,
                              "pivot_3_index": 60, "pivot_3": 9.2},
               "evidence": [{"code": "neckline_selection", "value": {
                   "selected_candidate_id": "paired-42"}}]}
    events = [
        {"instance_id": "000538:w:1", "event_id": "event-forming", "event_seq": 1,
         "event_date": "2026-01-02", "state": "forming", "entry_eligible": False,
         "pattern": pattern, "frozen_geometry": {"neckline": pattern["neckline"]}},
        {"instance_id": "000538:w:1", "event_id": "event-confirmed", "event_seq": 2,
         "event_date": "2026-01-08", "state": "confirmed", "entry_eligible": True,
         "pattern": pattern, "frozen_geometry": {"neckline": pattern["neckline"]}},
    ]
    account_path.write_text(json.dumps({"instance_events": events}), encoding="utf-8")
    payload = {
        "run_id": "fixture-run", "status": "complete", "date_to": "2026-01-10",
        "strategy_version": "strategy-v1", "recognizer_version": "recognizer-v2",
        "engine_version": "boundary-v3", "source_rule_version": "rules-v4",
        "versions": {"market_data": "fixture-data"},
        "universe": {"id": "daily_800_stock_200_etf_v1", "stock_count": 800,
                     "etf_count": 200, "selected_account_count": 1000},
        "parameters": {"date_to": "2026-01-10", "recognizer": "boundary_first", "workers": 8},
        "summary": {"account_count": 1000, "pattern_instance_event_count": 2,
                    "entry_event_count": 1, "trade_count": 1, "cumulative_return": 0.02,
                    "maximum_drawdown": -0.01, "successful_account_count": 999,
                    "failed_account_count": 1, "no_data_account_count": 0},
        "groups": {"kind": {"stock": {"trade_count": 1, "net_pnl": 10,
                    "win_rate": 1.0, "profit_factor": None}}, "shape": {}},
        "account_summaries": [{"code": "000538", "kind": "stock", "status": "complete",
                    "account_json": str(account_path), "account_json_relative": "accounts/stock-000538.json",
                    "signal_count": 1, "trade_count": 1, "pattern_instance_event_count": 2,
                    "entry_event_count": 1, "recognizer": "boundary_first", "engine_version": "boundary-v3",
                    "source_rule_version": "rules-v4", "provenance": {"data_revision": "r1"}}],
        "timings": {"total_seconds": 12.5, "load_seconds": 1, "recognition_seconds": 8,
                    "backtrader_seconds": 3.5},
        "memory": {"rss_peak_bytes": 123456}, "errors": [{"code": "600001", "kind": "stock", "error": "fixture failure"}],
    }
    report = _report_html(payload)

    class IDs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()
        def handle_starttag(self, tag, attrs):
            self.ids.update(value for key, value in attrs if key == "id")
    parser = IDs()
    parser.feed(report)
    assert {"data", "meta", "groups", "cases", "accounts", "runtime", "failures"} <= parser.ids
    assert "000538" in report and "event-confirmed" in report and "paired-42" in report
    assert "event_id" in report and "冻结线" in report and "accounts/stock-000538.json" in report
    assert "case-chart" in report and "P1" in report and "冻结边界" in report
    assert "derived_from_immutable_event_snapshot" in report and "boundary_version_id" in report
    assert "strategy-v1" in report and "2026-01-10" in report and "123456" in report
    assert "fixture failure" in report and "失败明细" in report
    stale = {**payload, "account_summaries": [{**payload["account_summaries"][0],
             "provenance": {"data_revision": "old-revision"}}]}
    stale_case = daily_research._report_event_cases(stale)[0]
    assert stale_case["chart_svg"] is None
    assert "版本不一致" in stale_case["chart_notice"]


def test_report_renderer_adds_legacy_metadata_as_separate_sidecar(tmp_path):
    import os
    import subprocess
    from caisen.store import get_pattern_instance_metadata

    output = tmp_path / "run"
    account_dir = output / "accounts"
    account_dir.mkdir(parents=True)
    account_path = account_dir / "stock-TEST.json"
    db_path = tmp_path / "metadata.sqlite3"
    raw_events = [{"instance_id": "TEST:w1", "event_id": "legacy-event-1", "event_seq": 1,
                   "event_date": "2026-01-05", "state": "forming", "entry_eligible": False,
                   "pattern": {"pattern_id": "TEST:w1", "pattern_type": "w_bottom", "timeframe": "1d",
                               "key_points": {"pivot_1_index": 10, "pivot_1_date": "2025-12-01"},
                               "neckline": {"slope": 0.0, "intercept": 10.0},
                               "evidence": [{"code": "neckline_selection", "value": {
                                   "selected_candidate_id": "pair-1", "selected_line": {"slope": 0.0, "intercept": 10.0},
                                   "line_frozen_at": "2026-01-02", "structure_known_at": "2026-01-04"}}]},
                   "frozen_geometry": {"neckline": {"slope": 0.0, "intercept": 10.0}}}]
    account_path.write_text(json.dumps({"run_id": "legacy-run", "code": "TEST", "kind": "stock",
                                       "instance_events": raw_events}), encoding="utf-8")
    report_path = output / "daily-research.json"
    manifest_path = output / "run-manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "legacy-run", "status": "complete"}), encoding="utf-8")
    payload = {"run_id": "legacy-run", "status": "complete", "date_to": "2026-01-10",
        "summary": {"account_count": 1, "pattern_instance_event_count": 1, "entry_event_count": 0,
                    "trade_count": 0, "net_pnl": 0, "expectancy": None, "profit_factor": None,
                    "average_holding_days": None, "cumulative_return": 0, "maximum_drawdown": 0,
                    "successful_account_count": 1, "failed_account_count": 0, "no_data_account_count": 0},
        "universe": {"id": "fixture", "stock_count": 1, "etf_count": 0, "selected_account_count": 1},
        "parameters": {"date_to": "2026-01-10", "recognizer": "boundary_first", "workers": 1},
        "groups": {"kind": {"stock": {}}, "shape": {}}, "timings": {}, "memory": {}, "errors": [],
        "artifacts": {"manifest_path": str(manifest_path)},
        "account_summaries": [{"code": "TEST", "kind": "stock", "status": "complete",
            "account_json": str(account_path), "account_json_relative": "accounts/stock-TEST.json",
            "signal_count": 0, "trade_count": 0, "pattern_instance_event_count": 1,
            "entry_event_count": 0, "recognizer": "boundary_first", "engine_version": "engine-v1",
            "source_rule_version": "rules-v1", "provenance": {"data_revision": "r1"}}]}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "render_daily_research_report.py"
    env = dict(os.environ, CAISEN_DB=str(db_path))
    subprocess.run([sys.executable, str(script), str(report_path)], check=True,
                   capture_output=True, text=True, env=env)

    updated_account = json.loads(account_path.read_text(encoding="utf-8"))
    assert updated_account["instance_events"] == raw_events
    metadata = updated_account["pattern_instance_metadata"]
    assert metadata[0]["metadata_source"] == "derived_from_immutable_event_snapshot"
    assert metadata[0]["source_event_schema_version"] == "legacy-event-v1"
    assert metadata[0]["paired_at"] == "2026-01-04"
    assert get_pattern_instance_metadata("legacy-run", path=db_path) == metadata
    updated_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated_report["instance_metadata"]["raw_events_modified"] is False
    assert updated_report["instance_metadata"]["derived_event_count"] == 1
    rendered = (output / "daily-research.html").read_text(encoding="utf-8")
    assert "derived_event_count" in rendered and "legacy-event-1" in rendered


def test_missing_and_failed_symbols_are_saved_and_counted_in_progress(tmp_path, monkeypatch):
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps({"id": "fixture", "stocks": ["600001", "600002"], "etfs": []}))
    monkeypatch.setattr("caisen.daily_research.discover_instruments", lambda root: [
        InstrumentFile("600001", "stock", str(tmp_path / "600001.csv")),
    ])
    monkeypatch.setattr("caisen.daily_research.load_research_bars", lambda instrument, root: {
        "bars": pd.DataFrame([("2026-01-01", 10, 10, 10, 10, 100)],
                              columns=["date", "open", "high", "low", "close", "volume"]),
        "source": "local_csv", "data_revision": "mtime_ns:broken,size:1", "first_date": "2026-01-01",
        "last_date": "2026-01-01", "normalized_rows": 1, "tick_size": 0.01, "tick_size_source": "default"})
    incremental = types.ModuleType("caisen.daily_incremental")
    incremental.generate_daily_signals = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("recognizer failed"))
    monkeypatch.setitem(sys.modules, "caisen.daily_incremental", incremental)
    snapshots = []
    payload = run_daily_research(universe_path, tmp_path / "run", workers=1,
                                 progress_callback=snapshots.append, root=tmp_path,
                                 allow_test_universe=True)
    accounts = {row["code"]: row for row in payload["account_summaries"]}
    assert accounts["600001"]["status"] == "failed"
    assert accounts["600002"]["status"] == "missing_inventory_file"
    assert payload["summary"]["account_count"] == 2
    assert payload["summary"]["failed_account_count"] == 2
    assert json.loads((tmp_path / "run" / "progress.json").read_text())["completed"] == 2
    assert json.loads((tmp_path / "run" / "run-manifest.json").read_text())["status"] == "partial_failed"
    assert json.loads((tmp_path / "run" / "progress.json").read_text())["active_symbols"] == []
    saved = sorted((tmp_path / "run" / "accounts").glob("*.json"))
    assert len(saved) == 2
    assert {json.loads(path.read_text())["code"] for path in saved} == {"600001", "600002"}
    assert snapshots[-1]["completed"] == 2


def test_equity_delta_sweep_matches_forward_filled_account_curves():
    from caisen.daily_research import _aggregate_equity_curve
    events = {}
    _aggregate_equity_curve.add_account(events, 100000, [
        {"date": "2026-01-01", "equity": 100000, "invested_value": 0},
        {"date": "2026-01-02", "equity": 101000, "invested_value": 50000},
    ])
    _aggregate_equity_curve.add_account(events, 100000, [
        {"date": "2026-01-02", "equity": 100000, "invested_value": 0},
        {"date": "2026-01-03", "equity": 99000, "invested_value": 60000},
    ])
    assert _aggregate_equity_curve.build(events, 200000) == [
        {"date": "2026-01-01", "equity": 200000.0, "invested_value": 0.0},
        {"date": "2026-01-02", "equity": 201000.0, "invested_value": 50000.0},
        {"date": "2026-01-03", "equity": 200000.0, "invested_value": 110000.0},
    ]


def test_benchmark_worker_selection_uses_one_when_threads_do_not_pay_off():
    from caisen.daily_research import _select_benchmark_workers
    assert _select_benchmark_workers(100, 104, 103) == 1
    assert _select_benchmark_workers(110, 100, 104) == 2
    assert _select_benchmark_workers(110, 105.1, 100) == 4


def test_signal_cache_is_keyed_to_data_cutoff_quote_and_rules(tmp_path):
    from caisen.daily_research import (_read_signal_cache, _signal_cache_identity,
                                       _write_signal_cache)

    instrument = InstrumentFile("600001", "stock", str(tmp_path / "600001.csv"))
    provenance = {"data_revision": "mtime_ns:1,size:2", "normalized_rows": 12,
                  "used_last_date": "2026-10-30", "tick_size": 0.01,
                  "tick_size_source": "registry"}
    key, identity = _signal_cache_identity(instrument, provenance, "2026-10-30", None, "boundary_first")
    path = tmp_path / (key + ".json")
    signals = [{"signal_date": "2026-10-29", "entry_date": "2026-10-30"}]
    stats = {"engine_version": "boundary-v2", "boundary_pairs_generated": 12}
    _write_signal_cache(path, key, identity, signals, stats)

    assert _read_signal_cache(path, key)["signals"] == signals
    assert _read_signal_cache(path, "wrong-key") is None
    assert _signal_cache_identity(instrument, provenance, "2026-10-29", None,
                                  "boundary_first")[0] != key
    revised = dict(provenance, data_revision="mtime_ns:2,size:2")
    assert _signal_cache_identity(instrument, revised, "2026-10-30", None,
                                  "boundary_first")[0] != key
    alternate = InstrumentFile("600001", "stock", str(tmp_path / "mirror" / "600001.csv"))
    assert _signal_cache_identity(alternate, provenance, "2026-10-30", None,
                                  "boundary_first")[0] != key
    alternate = InstrumentFile("600001", "stock", str(tmp_path / "mirror" / "600001.csv"))
    assert _signal_cache_identity(alternate, provenance, "2026-10-30", None,
                                  "boundary_first")[0] != key
    quoted = dict(provenance, tick_size=0.05)
    assert _signal_cache_identity(instrument, quoted, "2026-10-30", None,
                                  "boundary_first")[0] != key
    assert _signal_cache_identity(instrument, provenance, "2026-10-30", None,
                                  "boundary_first", {"experiment_id": "neckline_min3_v1"})[0] != key
    from caisen.research_config import RecognitionExperimentConfig
    strict = RecognitionExperimentConfig("neckline_min3_v1", 3,
        frozenset({"bull_flag", "inverse_head_shoulders", "triangle_bottom", "w_bottom"}))
    assert _signal_cache_identity(instrument, provenance, "2026-10-30", None,
                                  "boundary_first", strict)[0] != key
    from caisen import daily_research
    pure_execution_key, _ = _signal_cache_identity(instrument, provenance, "2026-10-30", None,
                                                   "boundary_first")
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(daily_research, "BENCHMARK_PROTOCOL_VERSION",
                        daily_research.BENCHMARK_PROTOCOL_VERSION + 1)
    monkeypatch.setattr(daily_research, "STRATEGY_VERSION", "execution-only-change")
    assert _signal_cache_identity(instrument, provenance, "2026-10-30", None,
                                  "boundary_first")[0] == pure_execution_key
    monkeypatch.undo()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]")
    assert _read_signal_cache(malformed, key) is None


def test_prepare_account_reuses_valid_signal_cache(tmp_path, monkeypatch):
    import types
    from caisen import daily_research

    instrument = InstrumentFile("600001", "stock", str(tmp_path / "600001.csv"))
    bars = pd.DataFrame([("2026-10-29", 10, 10, 10, 10, 100),
                         ("2026-10-30", 10, 10, 10, 10, 100)],
                        columns=["date", "open", "high", "low", "close", "volume"])
    monkeypatch.setattr(daily_research, "load_research_bars", lambda *args, **kwargs: {
        "bars": bars.copy(), "source": "local_csv", "data_revision": "mtime_ns:1,size:2",
        "first_date": "2026-10-29", "last_date": "2026-10-30", "normalized_rows": 2,
        "tick_size": 0.01, "tick_size_source": "registry"})
    calls = []
    incremental = types.ModuleType("caisen.daily_incremental")
    def generate(frame, code, kind, **kwargs):
        calls.append(code)
        return [{"signal_date": pd.Timestamp("2026-10-29"), "entry_date": "2026-10-30"}]
    incremental.generate_daily_signals = generate
    monkeypatch.setitem(sys.modules, "caisen.daily_incremental", incremental)

    first = daily_research._prepare_account(instrument, tmp_path, "2026-10-30", None,
                                            recognizer="v7", signal_cache_dir=tmp_path / "cache")
    second = daily_research._prepare_account(instrument, tmp_path, "2026-10-30", None,
                                             recognizer="v7", signal_cache_dir=tmp_path / "cache")

    assert calls == ["600001"]
    assert first["signals"] == second["signals"]
    assert first["signals"][0]["signal_date"] == "2026-10-29T00:00:00"
    assert second["recognition_stats"]["cache_hit"] is True
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_prepare_account_can_require_cache_and_fails_before_recognition(tmp_path, monkeypatch):
    import types
    from caisen import daily_research

    instrument = InstrumentFile("600001", "stock", str(tmp_path / "600001.csv"))
    bars = pd.DataFrame([("2026-10-30", 10, 10, 10, 10, 100)],
                        columns=["date", "open", "high", "low", "close", "volume"])
    monkeypatch.setattr(daily_research, "load_research_bars", lambda *args, **kwargs: {
        "bars": bars.copy(), "source": "local_csv", "data_revision": "mtime_ns:1,size:2",
        "first_date": "2026-10-30", "last_date": "2026-10-30", "normalized_rows": 1,
        "tick_size": 0.01, "tick_size_source": "registry"})
    calls = []
    incremental = types.ModuleType("caisen.daily_incremental")
    incremental.generate_daily_signals = lambda *args, **kwargs: calls.append(True)
    monkeypatch.setitem(sys.modules, "caisen.daily_incremental", incremental)

    try:
        daily_research._prepare_account(instrument, tmp_path, "2026-10-30", None,
            recognizer="boundary_first", signal_cache_dir=tmp_path / "cache",
            require_signal_cache=True)
    except RuntimeError as exc:
        assert "required signal cache miss" in str(exc)
    else:
        raise AssertionError("a required cache miss must fail before recognition")
    assert calls == []


def test_account_payload_stamps_experiment_on_account_and_signals_only_when_configured(tmp_path):
    from caisen.daily_research import _account_from_prepared, _account_summary_row
    from caisen.research_config import RecognitionExperimentConfig

    config = RecognitionExperimentConfig("neckline_min3_v1", 3,
        frozenset({"bull_flag", "inverse_head_shoulders", "triangle_bottom", "w_bottom"}))
    signal = {"signal_date": "2026-01-01", "pattern_ids": ["p1"]}
    prepared = {"status": "no_data", "recognizer": "boundary_first", "signals": [signal],
        "provenance": {}, "load_seconds": 0.0, "recognition_seconds": 0.0,
        "recognition_stats": {}, "recognition_config": config}
    account = _account_from_prepared(prepared, "600001", "stock")
    expected = {"experiment_id": "neckline_min3_v1", "minimum_neckline_tests": 3,
        "admission_types": ["bull_flag", "inverse_head_shoulders", "triangle_bottom", "w_bottom"],
        "cache_key": ["neckline_min3_v1", 3,
                      ["bull_flag", "inverse_head_shoulders", "triangle_bottom", "w_bottom"]]}

    assert account["recognition_experiment"] == expected
    assert account["signals"][0]["recognition_experiment"] == expected
    summary = _account_summary_row(account, tmp_path / "account.json", tmp_path)
    assert summary["recognition_experiment_id"] == "neckline_min3_v1"
    assert summary["recognition_config"] == expected

    baseline = _account_from_prepared({**prepared, "recognition_config": None,
        "signals": [{"signal_date": "2026-01-01", "pattern_ids": ["p1"]}]}, "600001", "stock")
    assert "recognition_experiment" not in baseline
    assert "recognition_experiment" not in baseline["signals"][0]


def test_spawn_worker_selection_excludes_over_limit_and_unverified_process_trees():
    from caisen.daily_research import _select_safe_worker_count
    cap = 12 * 1024 ** 3
    selected, fastest, eligible = _select_safe_worker_count(
        {"1": 130, "2": 120, "4": 110, "6": 100, "8": 95},
        {"1": cap - 1, "2": cap - 1, "4": cap - 1, "6": cap - 1, "8": cap + 1},
        {str(workers): True for workers in (1, 2, 4, 6, 8)}, cap)
    assert (selected, fastest, eligible) == (6, 100, [6])
    selected, _, eligible = _select_safe_worker_count(
        {"1": 103, "2": 100, "4": 99}, {"1": 0, "2": 1024, "4": 1024},
        {"1": True, "2": True, "4": True}, cap)
    assert (selected, eligible) == (2, [2, 4])


def test_current_benchmark_requires_every_expected_account_ledger_check():
    from caisen.daily_research import _all_ledger_checks_pass
    assert not _all_ledger_checks_pass({"account_summaries": []}, 0)
    assert _all_ledger_checks_pass({"account_summaries": [{
        "status": "complete", "accounting_checks": {
            "cash_plus_position_equals_equity": True,
            "realized_pnl_matches_ending_equity": True}}]}, 1)
    assert not _all_ledger_checks_pass({"account_summaries": [{
        "status": "complete", "accounting_checks": {
            "cash_plus_position_equals_equity": True,
            "realized_pnl_matches_ending_equity": False}}]}, 1)


def test_v8_benchmark_uses_36_frozen_ledgers_then_all_worker_counts(tmp_path, monkeypatch):
    from caisen import daily_research

    stocks = ["%06d" % (100000 + index) for index in range(800)]
    etfs = ["%06d" % (500000 + index) for index in range(200)]
    universe = {"id": daily_research.UNIVERSE_ID, "stocks": stocks, "etfs": etfs,
                "counts": {"stock": 800, "etf": 200, "total": 1000}}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe))
    cohort_keys = [("stock", code) for code in stocks[:24]]
    top_keys = [cohort_keys[0]] + [("stock", code) for code in stocks[24:33]]
    case_keys = [("stock", "600108"), ("etf", "159718"), ("etf", "516630")]
    all_keys = set(cohort_keys) | set(top_keys) | set(case_keys)
    archive_dir, snapshot_dir = tmp_path / "archive", tmp_path / "snapshot"
    (archive_dir / "accounts").mkdir(parents=True)
    (snapshot_dir / "accounts").mkdir(parents=True)
    rows = {"archive": [], "snapshot": []}
    instruments = []
    for kind, code in sorted(all_keys):
        data_path = tmp_path / (kind + "-" + code + ".csv")
        data_path.write_text("date,open,high,low,close,volume\n2026-01-01,1,1,1,1,1\n")
        stat = data_path.stat()
        revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
        instruments.append(InstrumentFile(code, kind, str(data_path)))
        source, base = ("snapshot", snapshot_dir) if (kind, code) in case_keys else ("archive", archive_dir)
        account_path = base / "accounts" / ("%s-%s.json" % (kind, code))
        account = {"code": code, "kind": kind, "status": "complete", "signals": [], "trades": [],
                   "daily_equity": [], "rejections": [], "order_events": [], "ending_equity": 100000,
                   "metrics": {}, "accounting_checks": {"cash_plus_position_equals_equity": True,
                     "realized_pnl_matches_ending_equity": True}}
        account_path.write_text(json.dumps(account))
        recognition_seconds = (10000 - top_keys.index((kind, code)) if (kind, code) in top_keys
                               else 100 - len(rows[source]))
        row = {"code": code, "kind": kind, "status": "complete", "account_json": str(account_path),
               "recognizer": "boundary_first",
               "strategy_version": daily_research.STRATEGY_VERSION,
               "recognizer_version": daily_research.RECOGNIZER_VERSION,
               "engine_version": daily_research.BOUNDARY_FIRST_ENGINE_VERSION,
               "source_rule_version": str(daily_research.MAIN_RULES["source_rule_version"]),
               "timings": {"recognition_seconds": float(recognition_seconds)},
               "provenance": {"data_revision": revision, "used_last_date": "2026-10-30",
                              "normalized_rows": 1},
               "accounting_checks": account["accounting_checks"], "recognition_stats": {}}
        rows[source].append(row)
    version_fields = {"strategy_version": daily_research.STRATEGY_VERSION,
                      "recognizer_version": daily_research.RECOGNIZER_VERSION,
                      "engine_version": daily_research.BOUNDARY_FIRST_ENGINE_VERSION,
                      "source_rule_version": str(daily_research.MAIN_RULES["source_rule_version"])}
    for folder, label in ((archive_dir, "archive"), (snapshot_dir, "snapshot")):
        payload = {"status": "complete", "date_to": "2026-10-30", "parameters": {"trade_start": None},
                   **version_fields, "account_summaries": rows[label]}
        (folder / "daily-research.json").write_text(json.dumps(payload))
    cohort_path = tmp_path / "cohort.json"
    cases_path = tmp_path / "cases.json"
    cohort_path.write_text(json.dumps({"universe_id": daily_research.UNIVERSE_ID,
                                       "selection_version": "fixed-universe-stratified-v1",
                                       "symbols": [{"kind": kind, "code": code}
                                                 for kind, code in cohort_keys]}))
    cases_path.write_text(json.dumps({"universe_id": daily_research.UNIVERSE_ID,
                                      "symbols": [{"kind": kind, "code": code}
                                                for kind, code in case_keys]}))
    monkeypatch.setattr(daily_research, "discover_instruments", lambda root: instruments)
    calls = []
    clock = [0.0]
    monkeypatch.setattr(daily_research.time, "perf_counter", lambda: clock[0])
    def fake_run(universe_path, output_dir, *, workers, symbols, **kwargs):
        calls.append((workers, kwargs["execution_mode"], kwargs["signal_cache_dir"], len(symbols),
                      set(kwargs["dispatch_sessions"].values())))
        clock[0] += 20 - workers
        return {"account_summaries": rows["archive"] + rows["snapshot"],
                "summary": {"account_count": 36, "signal_count": 0, "trade_count": 0},
                "timings": {"total_seconds": 20 - workers}}
    monkeypatch.setattr(daily_research, "run_daily_research", fake_run)
    monkeypatch.setattr(daily_research, "_run_with_process_tree_peak",
                        lambda callable_run: (callable_run(), 1024))
    active_policy = tmp_path / "active-v5-policy.json"
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", active_policy)
    monkeypatch.setattr(daily_research, "LEGACY_BENCHMARK_PATH", tmp_path / "missing-legacy-benchmark.json")

    result = daily_research.benchmark_current_v8_pool(universe_path, tmp_path / "reports", root=tmp_path,
        cohort_path=cohort_path, cases_path=cases_path, archive_reference_path=archive_dir,
        snapshot_reference_path=snapshot_dir)

    assert result["status"] == "complete"
    assert result["reference_count"] == 36
    assert result["archive_reference_count"] == 33
    assert result["snapshot_reference_count"] == 3
    assert [call[0] for call in calls] == [1, 2, 4, 6, 8, 10]
    assert all(call[1:] == ("spawn", None, 36, {1}) for call in calls)
    assert result["recommended_workers"] == 10
    assert not active_policy.exists()
    assert daily_research._resolve_execution_config() == (1, "default_no_benchmark", "boundary_first")


def test_benchmark_progress_report_written_follows_readable_report(tmp_path):
    from caisen.daily_research import _write_benchmark_progress
    report = tmp_path / "benchmark-result.json"
    report.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    progress_path = tmp_path / "progress.json"
    emitted = []
    snapshot = {"status": "complete", "stage": "report_written",
                "report_path": str(report.resolve())}
    _write_benchmark_progress(progress_path, snapshot, emitted.append)
    assert json.loads(Path(snapshot["report_path"]).read_text())["status"] == "complete"
    assert json.loads(progress_path.read_text()) == snapshot
    assert emitted == [snapshot]


def test_benchmark_cohort_prefers_target_length_and_balances_stocks_with_etfs():
    from caisen.daily_research import _select_benchmark_members

    records = [
        {"kind": "stock", "code": "S-short-far", "history_band": "short", "sessions": 30},
        {"kind": "stock", "code": "S-short-near", "history_band": "short", "sessions": 218},
        {"kind": "etf", "code": "E-short-near", "history_band": "short", "sessions": 224},
        {"kind": "stock", "code": "S-medium-near", "history_band": "medium", "sessions": 518},
        {"kind": "etf", "code": "E-medium-near", "history_band": "medium", "sessions": 522},
        {"kind": "stock", "code": "S-long-huge", "history_band": "long", "sessions": 8000},
        {"kind": "stock", "code": "S-long-near", "history_band": "long", "sessions": 998},
        {"kind": "etf", "code": "E-long-near", "history_band": "long", "sessions": 1002},
    ]

    selected = _select_benchmark_members(records, per_history=2)

    assert [(item["history_band"], item["kind"], item["code"]) for item in selected] == [
        ("short", "stock", "S-short-near"), ("short", "etf", "E-short-near"),
        ("medium", "stock", "S-medium-near"), ("medium", "etf", "E-medium-near"),
        ("long", "stock", "S-long-near"), ("long", "etf", "E-long-near"),
    ]


def test_benchmark_parity_signature_includes_account_trades_and_equity(tmp_path):
    from caisen.daily_research import _result_signature
    account_path = tmp_path / "account.json"
    account = {"code": "600001", "kind": "stock", "status": "complete",
               "signals": [{"signal_date": "2026-01-01"}],
               "trades": [{"entry_date": "2026-01-02", "pnl": 10}],
               "daily_equity": [{"date": "2026-01-02", "equity": 100010, "invested_value": 100000}],
               "rejections": [], "order_events": [], "ending_equity": 100010,
               "metrics": {"trade_count": 1}, "accounting_checks": {
                   "cash_plus_position_equals_equity": True,
                   "realized_pnl_matches_ending_equity": True}}
    account_path.write_text(json.dumps(account))
    report = {"account_summaries": [{"account_json": str(account_path)}],
              "trades": [{"code": "600001", "pnl": 10}],
              "daily_equity": [{"date": "2026-01-02", "equity": 100010, "invested_value": 100000}],
              "summary": {"trade_count": 1}}
    baseline = _result_signature(report)
    account["daily_equity"][0]["equity"] = 100009
    account_path.write_text(json.dumps(account))
    assert _result_signature(report) != baseline


def test_account_ledger_signature_is_stable_across_report_row_order(tmp_path):
    from caisen.daily_research import _account_ledger_signature
    paths = []
    for kind, code in (("stock", "600001"), ("etf", "510001")):
        path = tmp_path / (code + ".json")
        path.write_text(json.dumps({"code": code, "kind": kind, "status": "complete",
            "signals": [], "trades": [], "daily_equity": [], "rejections": [],
            "order_events": [], "ending_equity": 100000, "metrics": {},
            "accounting_checks": {"cash_plus_position_equals_equity": True,
                                   "realized_pnl_matches_ending_equity": True}}))
        paths.append(str(path))
    first = {"account_summaries": [{"account_json": path} for path in paths]}
    second = {"account_summaries": list(reversed(first["account_summaries"]))}
    assert _account_ledger_signature(first) == _account_ledger_signature(second)


def test_benchmark_normalizes_real_backtrader_refs_but_detects_fill_or_equity_changes(tmp_path):
    from caisen.daily_research import _result_signature
    bars = pd.DataFrame([
        ("2026-01-01", 10, 10, 10, 10, 1000),
        ("2026-01-02", 10, 10.1, 9.9, 10, 1000),
        ("2026-01-03", 10, 10.5, 9.5, 10.2, 1000),
        ("2026-01-04", 10.2, 10.5, 10, 10.4, 1000),
    ], columns=["date", "open", "high", "low", "close", "volume"])
    signal = {"signal_date": "2026-01-01", "entry_date": "2026-01-02",
              "pattern_ids": ["p1"], "pattern_types": ["w_bottom"]}
    first = simulate_fixed_trades(bars, [signal])
    second = simulate_fixed_trades(bars, [signal])
    refs = [event["order_ref"] for event in first["order_events"]]
    refs_again = [event["order_ref"] for event in second["order_events"]]
    assert refs != refs_again

    def report_for(sim, path):
        account = {"code": "600001", "kind": "stock", "status": "complete", "signals": [signal],
                   "trades": sim["trades"], "daily_equity": sim["daily_equity"], "rejections": sim["rejections"],
                   "order_events": sim["order_events"], "ending_equity": sim["ending_equity"],
                   "metrics": calculate_metrics(sim["trades"], 100000, sim["ending_equity"], sim["daily_equity"]),
                   "accounting_checks": sim["accounting_checks"]}
        path.write_text(json.dumps(account))
        return {"account_summaries": [{"account_json": str(path)}],
                "trades": [dict(trade, code="600001", kind="stock") for trade in sim["trades"]],
                "daily_equity": sim["daily_equity"],
                "summary": {"trade_count": len(sim["trades"]), "ending_equity": sim["ending_equity"]}}

    first_report = report_for(first, tmp_path / "first.json")
    second_report = report_for(second, tmp_path / "second.json")
    baseline = _result_signature(first_report)
    assert _result_signature(second_report) == baseline
    second["trades"][0]["exit_price"] += 0.01
    altered_report = report_for(second, tmp_path / "altered.json")
    assert _result_signature(altered_report) != baseline


def test_benchmark_uses_one_frozen_mixed_history_cohort_for_all_worker_counts(tmp_path, monkeypatch):
    from caisen import daily_research
    universe_path = tmp_path / "universe.json"
    stocks = ["00000%d" % index for index in range(6)] + ["1%05d" % index for index in range(794)]
    stocks.sort()
    etfs = ["5%05d" % index for index in range(200)]
    universe_path.write_text(json.dumps({"id": "daily_800_stock_200_etf_v1", "stocks": stocks,
                                         "etfs": etfs, "counts": {"stock": 800, "etf": 200, "total": 1000}}))
    instruments = [InstrumentFile(code, "stock", str(tmp_path / (code + ".csv"))) for code in stocks[:6]]
    monkeypatch.setattr(daily_research, "discover_instruments", lambda root: instruments)
    loads = []
    history = {"000000": 10, "000001": 20, "000002": 500,
               "000003": 600, "000004": 900, "000005": 1000}
    def fake_load(instrument, root):
        loads.append(instrument.code)
        dates = pd.date_range("2020-01-01", periods=history[instrument.code])
        frame = pd.DataFrame({"date": dates, "open": 10, "high": 10, "low": 10,
                              "close": 10, "volume": 100})
        return {"bars": frame, "data_revision": "mtime_ns:%d,size:%d" % (history[instrument.code], 1)}
    monkeypatch.setattr(daily_research, "load_research_bars", fake_load)
    calls = []
    times = {1: 10.0, 2: 8.0, 4: 7.5}
    def fake_run(universe_path, output_dir, workers, symbols, **kwargs):
        calls.append((workers, list(symbols)))
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries, trades = [], []
        for kind, code in symbols:
            account_path = output_dir / (code + ".json")
            account = {"code": code, "kind": kind, "status": "complete", "signals": [], "trades": [],
                       "daily_equity": [], "rejections": [], "order_events": [], "ending_equity": 100000,
                       "metrics": {}, "accounting_checks": {"cash_plus_position_equals_equity": True,
                                                                "realized_pnl_matches_ending_equity": True}}
            account_path.write_text(json.dumps(account))
            summaries.append({"code": code, "kind": kind, "status": "complete",
                              "account_json": str(account_path), "accounting_checks": account["accounting_checks"],
                              "provenance": {"data_revision": "mtime_ns:%d,size:1" % history[code]},
                              "timings": {"recognition_seconds": 1.0, "backtrader_seconds": .1}})
        return {"account_summaries": summaries, "trades": trades, "daily_equity": [], "summary": {
            "signal_count": 0, "trade_count": 0, "account_count": len(summaries)},
                "timings": {"total_seconds": times[workers]}}
    monkeypatch.setattr(daily_research, "run_daily_research", fake_run)

    first = daily_research.benchmark_daily_research(universe_path, tmp_path / "reports", root=tmp_path)
    reads_after_first = len(loads)
    second = daily_research.benchmark_daily_research(universe_path, tmp_path / "reports", root=tmp_path)

    assert first["status"] == "complete"
    assert first["recommended_workers"] == 4
    assert [workers for workers, _ in calls] == [1, 2, 4] * 6
    assert first["repetitions"] == 3
    assert first["runs"]["2"]["seconds"] == 8.0
    assert first["median_seconds_by_worker_count"] == {"1": 10.0, "2": 8.0, "4": 7.5}
    assert all(cohort == calls[0][1] for _, cohort in calls)
    assert {member["history_band"] for member in first["cohort"]} == {"short", "medium", "long"}
    assert len(first["cohort"]) == 6
    assert len(loads) == reads_after_first
    assert second["cohort"] == first["cohort"]
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", tmp_path / "missing-benchmark.json")
    monkeypatch.setattr(daily_research, "LEGACY_BENCHMARK_PATH", tmp_path / "missing-legacy-benchmark.json")
    assert daily_research._resolve_workers(None) == (1, "default_no_benchmark")


def test_direct_research_loader_records_stat_revision_without_hashing_or_quality_gates(tmp_path):
    from caisen.daily_research import load_research_bars
    path = tmp_path / "600001.csv"
    path.write_text("date,open,high,low,close,volume\n2026-01-01,10,11,9,10.5,0\n", encoding="utf-8")
    result = load_research_bars(InstrumentFile("600001", "stock", str(path)), root=tmp_path)
    assert len(result["bars"]) == 1
    assert result["data_revision"].startswith("mtime_ns:")
    assert "sha" not in result["data_revision"]
    assert result["tick_size"] == 0.01


def test_rule_revision_is_part_of_the_daily_benchmark_cache_identity():
    from caisen import daily_research

    versions = daily_research._version_metadata()

    assert versions["engine_version"] == "caisen-daily-continuous-v15"
    assert versions["boundary_first_engine_version"] == "caisen-daily-boundary-first-v16"
    assert versions["source_rule_version"] == "caisen-main-bottoms-v17"
    assert versions["main_pattern_rule_version"] == "caisen-v30-main-bottom-v1"


def test_workers_none_consumes_only_a_completed_parity_checked_recommendation(tmp_path, monkeypatch):
    from caisen import daily_research
    assert daily_research.BENCHMARK_PATH.name == "benchmark-execution-v5-current-rules-36.json"
    path = tmp_path / "benchmark.json"
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", path)
    path.write_text(json.dumps({"status": "complete", "benchmark_protocol_version": 5,
                                "policy_scope": daily_research.BENCHMARK_POLICY_SCOPE,
                                "reference_count": 36,
                                "reference_mode": "single_process_same_current_code",
                                "source_unchanged_during_run": True,
                                "source_sha256": daily_research._current_execution_source_hashes(),
                                "source_sha256_after": daily_research._current_execution_source_hashes(),
                                "reference_ledger_signature_sha256": "same",
                                "candidate_ledger_signature_sha256": "same",
                                "accounting_checks_passed": True,
                                "data_revisions_match_references": True,
                                "performance_gate_passed": True,
                                "speedup_vs_serial": 2.0,
                                "universe_id": "daily_800_stock_200_etf_v1",
                                "parity_passed": True, "versions": daily_research._version_metadata(),
                                "recommended_workers": 8,
                                "recommended_recognizer": "boundary_first",
                                "process_tree_peak_bytes": 1024,
                                "process_tree_memory_cap_bytes": daily_research.BENCHMARK_PROCESS_TREE_RSS_CAP,
                                "worker_counts": {"8": {"process_tree_peak_bytes": 1024,
                                    "account_count": 36, "execution_mode": "spawn",
                                    "accounting_checks_passed": True,
                                    "data_revisions_match_references": True,
                                    "parity_passed": True, "memory_status": "pass"}}}))
    assert daily_research._resolve_workers(None) == (8, str(path))
    assert daily_research._resolve_execution_config() == (8, str(path), "boundary_first")
    assert daily_research._resolve_execution_config(workers=2) == (2, "explicit", "boundary_first")
    assert daily_research._resolve_execution_config(recognizer="v7") == (1, "default_ineligible_benchmark", "v7")
    corrupted_hash = json.loads(path.read_text())
    corrupted_hash["source_sha256_after"] = {"caisen/main_bottoms.py": "stale"}
    path.write_text(json.dumps(corrupted_hash))
    assert daily_research._resolve_workers(None) == (1, "default_ineligible_benchmark")
    stale = daily_research._version_metadata()
    stale.update({"recognizer_version": "caisen-daily-continuous-v3",
                  "engine_version": "caisen-daily-continuous-v3",
                  "boundary_first_engine_version": "caisen-daily-boundary-first-v1",
                  "source_rule_version": "caisen-main-bottoms-v7"})
    path.write_text(json.dumps({"status": "complete", "universe_id": "daily_800_stock_200_etf_v1",
                                "parity_passed": True, "versions": stale, "recommended_workers": 8,
                                "recommended_recognizer": "boundary_first",
                                "process_tree_peak_bytes": 1024,
                                "process_tree_memory_cap_bytes": daily_research.BENCHMARK_PROCESS_TREE_RSS_CAP,
                                "benchmark_protocol_version": 5,
                                "policy_scope": daily_research.BENCHMARK_POLICY_SCOPE,
                                "reference_count": 36,
                                "reference_mode": "single_process_same_current_code",
                                "source_unchanged_during_run": True,
                                "source_sha256": daily_research._current_execution_source_hashes(),
                                "source_sha256_after": daily_research._current_execution_source_hashes(),
                                "reference_ledger_signature_sha256": "same",
                                "candidate_ledger_signature_sha256": "same",
                                "accounting_checks_passed": True,
                                "data_revisions_match_references": True,
                                "performance_gate_passed": True,
                                "speedup_vs_serial": 2.0,
                                "worker_counts": {"8": {"process_tree_peak_bytes": 1024,
                                    "account_count": 36, "execution_mode": "spawn",
                                    "accounting_checks_passed": True,
                                    "data_revisions_match_references": True,
                                    "parity_passed": True, "memory_status": "pass"}}}))
    assert daily_research._resolve_workers(None) == (1, "default_ineligible_benchmark")
    path.write_text(json.dumps({"status": "parity_failed", "parity_passed": False,
                                "universe_id": "daily_800_stock_200_etf_v1",
                                "versions": daily_research._version_metadata(), "recommended_workers": None,
                                "recommended_recognizer": None,
                                "process_tree_peak_bytes": 1024,
                                "process_tree_memory_cap_bytes": daily_research.BENCHMARK_PROCESS_TREE_RSS_CAP}))
    assert daily_research._resolve_workers(None) == (1, "default_ineligible_benchmark")
    path.unlink()
    monkeypatch.setattr(daily_research, "LEGACY_BENCHMARK_PATH", tmp_path / "missing-legacy-benchmark.json")
    assert daily_research._resolve_workers(None) == (1, "default_no_benchmark")


def test_default_daily_config_uses_current_recognizer_and_exposes_effective_rules(tmp_path, monkeypatch):
    from caisen import daily_research
    stale_path = tmp_path / "stale-benchmark.json"
    stale_path.write_text("{}")
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", stale_path)
    monkeypatch.setattr(daily_research, "_validated_benchmark_config", lambda **_: None)
    config = daily_research.resolve_daily_research_config()
    assert config["workers"] == 1
    assert config["worker_source"] == "default_ineligible_benchmark"
    assert config["recognizer"] == "boundary_first"
    assert config["execution_mode"] == "spawn"
    assert config["engine_version"] == daily_research.BOUNDARY_FIRST_ENGINE_VERSION
    assert config["strategy_version"] == daily_research.STRATEGY_VERSION
    assert config["recognition_rules"]["neckline_selection_rule_version"] == daily_research.MAIN_RULES[
        "neckline_selection_rule_version"]
    assert config["recognition_rules"]["main_pattern_rule_version"] == daily_research.MAIN_RULES[
        "main_pattern_rule_version"]
    assert config["recognition_rules"]["consumed_boundary_phase_rule"] == daily_research.MAIN_RULES[
        "neckline_tests"]["consumed_boundary_phase_rule"]
    assert config["recognition_rules"]["candidate_reversal_atr_multiple"] == 2.0
    assert config["recognition_rules"]["breakout_close_buffer_previous_atr_multiple"] == 1.0
    assert config["trading_rules"]["lot_size"] == 100
    assert config["versions"]["recognition_rules_sha256"]


def test_spawn_pool_smoke_writes_one_independent_account(tmp_path):
    from caisen.daily_research import run_daily_research
    import json

    market = tmp_path / "data" / "stocks"
    market.mkdir(parents=True)
    (market / "600001.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")
    (market / "600002.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")
    universe = tmp_path / "universe.json"
    universe.write_text(json.dumps({"id": "fixture", "stocks": ["600001", "600002"], "etfs": []}))
    result = run_daily_research(universe, tmp_path / "spawn-run", workers=2,
                                root=tmp_path, allow_test_universe=True,
                                execution_mode="spawn")
    accounts = [json.loads(Path(row["account_json"]).read_text()) for row in result["account_summaries"]]
    assert result["status"] == "complete"
    assert {account["code"] for account in accounts} == {"600001", "600002"}
    assert all(account["status"] == "no_data" for account in accounts)
    progress = json.loads((tmp_path / "spawn-run" / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["stage"] == "report_written"
    assert progress["active_symbols"] == []
    assert Path(progress["report_path"]).is_file()
