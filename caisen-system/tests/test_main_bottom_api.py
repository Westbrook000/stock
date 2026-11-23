from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd

from caisen import api
from caisen.market import InstrumentFile, MarketDataResult
from caisen.schemas import Pattern
from caisen.main_bottoms import link_main_bottom_children
from caisen.api import _decision_patterns_by_timeframe
from caisen.patterns import current_pattern_snapshot


def _bars(dates):
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates), "open": 10.0, "high": 10.5, "low": 9.5,
        "close": 10.0, "volume": 1000.0, "name": "样本标的",
    })
    frame.attrs["quality"] = {"tick_size": 0.01, "lot_size": 100, "category": "equity"}
    frame.attrs["provenance"] = {"data_version": "fixture-v1"}
    return frame


def test_scan_uses_full_prefixes_for_main_path_without_legacy_fallback(monkeypatch):
    instrument = InstrumentFile("000001", "stock", "one.csv")
    daily = _bars(pd.bdate_range("2010-01-04", "2026-10-28"))
    weekly = _bars(pd.date_range("2014-01-03", "2026-10-23", freq="W-FRI"))
    main = Pattern(
        pattern_id="main", code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
        direction="bullish", state="confirmed", formed_at="2023-01-01", confirmed_at=300,
        confirmation_date="2025-01-01", invalidation=8.0, targets=[14.0], volume_verified=False,
        evidence=[{"code":"main_structure","value":{"role":"main","selected":True,
                   "signal_eligible":True,"scale":"5y","first_trough_date":"2023-01-01",
                   "frozen_at":"2023-01-01","structure_end_date":"2026-10-28"}},
                  {"code":"main_selection","value":{"selected":True,"decided_at":"2026-10-28"}}],
    )
    ended_main = Pattern(
        pattern_id="main-ended", code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
        direction="bullish", state="invalidated", formed_at="2021-01-01", confirmed_at=None,
        evidence=[{"code":"main_structure","value":{"role":"main","signal_eligible":False}},
                  {"code":"main_selection","value":{"selected":False,"decided_at":"2026-10-28"}}],
    )
    child = Pattern(
        pattern_id="child", code="000001", kind="stock", timeframe="1d", pattern_type="bull_flag",
        direction="bullish", state="confirmed", formed_at="2025-01-01", confirmed_at=480,
        confirmation_date="2026-09-01", invalidation=9.0, targets=[13.0], volume_verified=True,
        last_observed_date="2026-10-28",
        key_points={"pivot_1_index":480,"pivot_1":9.8,"pivot_2_index":495,"pivot_2":10.2},
    )
    seen = {"daily_session": None, "weekly_session": None, "aggregate_calls": []}
    monkeypatch.setattr(api, "discover_instruments", lambda root: [instrument])
    monkeypatch.setattr(api, "load_daily_bars", lambda item, root: MarketDataResult(item, bars=daily))
    def aggregate(result, **kwargs):
        frame = result.bars if isinstance(result, MarketDataResult) else result
        seen["aggregate_calls"].append((frame.date.min(), frame.date.max(), len(frame)))
        if isinstance(result, MarketDataResult):
            return MarketDataResult(result.instrument, bars=weekly, quality_status="usable")
        return weekly
    monkeypatch.setattr(api, "aggregate_weekly", aggregate)
    def recognize_daily(frame, code, kind, **kwargs):
        seen["daily_session"] = frame.date.min(), frame.date.max(), len(frame)
        return {"patterns": [main, ended_main, child], "signals": [], "performance": {}}
    def recognize_weekly(frame, code, kind, **kwargs):
        seen["weekly_session"] = frame.date.min(), frame.date.max(), len(frame)
        return {"patterns": [], "signals": [], "performance": {}}
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("scan must not call the legacy recognizer")))
    monkeypatch.setattr(api, "recognize_daily_history", recognize_daily)
    monkeypatch.setattr(api, "recognize_weekly_history", recognize_weekly)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)

    api._scan_worker("main-bottom-fixture", {
        "timeframes": ["1d", "1w"], "date_from": "2014-01-01", "date_to": "2026-10-28",
    }, executor_factory=ThreadPoolExecutor)

    result = api._jobs["main-bottom-fixture"]
    assert [entry[:2] for entry in seen["aggregate_calls"]] == [
        (pd.Timestamp("2010-01-04"), pd.Timestamp("2026-10-28"))]
    daily_window = daily.loc[daily.date >= pd.Timestamp("2021-10-28")]
    weekly_window = weekly.loc[weekly.date >= weekly.date.max() - pd.DateOffset(years=10)]
    assert seen["daily_session"] == (daily_window.date.min(), daily_window.date.max(), len(daily_window))
    assert seen["weekly_session"] == (weekly_window.date.min(), weekly_window.date.max(), len(weekly_window))
    row = result["results"][0]
    assert result["versions"]["engine_version"] == api.DAILY_ENGINE_VERSION
    assert result["versions"]["source_rule_version"] == api.DAILY_SOURCE_RULE_VERSION
    assert result["versions"]["lifecycle_schema_version"] == api.LIFECYCLE_SCHEMA_VERSION
    assert "versions" not in row
    assert [pattern["pattern_type"] for pattern in row["patterns"]] == [main.pattern_type]
    assert row["patterns"][0]["formed_at"] == main.formed_at
    assert not ({"main_structure_events", "selected_main_structures", "ended_patterns"} & row.keys())


def test_main_bottom_chart_offset_uses_calendar_cutoff():
    bars = _bars(pd.bdate_range("2010-01-04", "2026-10-28"))
    offset = api._main_bottom_chart_offset(bars, "1d")
    assert bars.date.iloc[offset] >= pd.Timestamp("2021-10-28")
    assert bars.date.iloc[offset - 1] < pd.Timestamp("2021-10-28")


def test_new_scan_task_uses_v17_without_rewriting_v16_baseline(monkeypatch):
    from caisen import RULE_VERSION
    from caisen.v16_baseline import RULE_VERSION as V16_BASELINE_RULE_VERSION
    from caisen.store import get_scan_run

    class DeferredThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass

    monkeypatch.setattr(api, "discover_instruments", lambda root: [])
    monkeypatch.setattr(api.threading, "Thread", DeferredThread)
    response = api.start_scan({"timeframes": ["1d"], "kinds": ["stock"]})
    saved = get_scan_run(response["job_id"])
    assert RULE_VERSION == "caisen-v17"
    assert saved["rule_version"] == "caisen-v17"
    assert api._jobs[response["job_id"]]["rule_version"] == "caisen-v17"
    assert saved["params"]["versions"]["engine_version"] == api.DAILY_ENGINE_VERSION
    assert saved["params"]["versions"]["source_rule_version"] == api.DAILY_SOURCE_RULE_VERSION
    assert saved["params"]["versions"]["lifecycle_schema_version"] == api.LIFECYCLE_SCHEMA_VERSION
    assert saved["params"]["versions"]["tick_rule_version"] == api.TICK_RULE_VERSION
    assert saved["versions"] == saved["params"]["versions"]
    assert api._jobs[response["job_id"]]["versions"] == saved["params"]["versions"]
    assert saved["rule_version"] == "caisen-v17"  # legacy application label remains explicit
    assert V16_BASELINE_RULE_VERSION == "caisen-v16"


def test_scan_progress_keeps_version_metadata():
    job_id = "version-progress-test"
    previous = api._jobs.get(job_id)
    expected_versions = api._scan_version_metadata()
    api._jobs[job_id] = {"status": "queued", "rule_version": "caisen-v17",
                         "versions": expected_versions}
    try:
        api._scan_progress(job_id, 0, 2)
        assert api._jobs[job_id]["versions"] == expected_versions
        assert api._jobs[job_id]["rule_version"] == "caisen-v17"
    finally:
        if previous is None:
            api._jobs.pop(job_id, None)
        else:
            api._jobs[job_id] = previous


def test_scan_job_status_is_small_and_reads_count_from_paged_result_store(monkeypatch, tmp_path):
    from caisen import store

    monkeypatch.setattr(store, "database_path", lambda: str(tmp_path / "scan.sqlite3"))
    store.create_scan_run("large-scan", {"kinds": ["stock"], "versions": {"engine_version": "v-test"}},
                          total=1, rule_version="caisen-v17", app_version="0.9.0")
    store.save_scan_result("large-scan", 0, {"code": "000001", "kind": "stock", "action": "WAIT",
                                               "patterns": [], "mini_bars": []})
    store.update_scan_run("large-scan", status="completed", progress=1, scanned=1, total=1,
                          inventory={"stock_count": 1, "etf_count": 0})
    api._jobs["large-scan"] = {"status": "completed", "results": [{"payload": "x" * 1_000_000}]}
    try:
        status = api.get_job("large-scan")
        assert status["status"] == "completed"
        assert status["result_count"] == 1
        assert status["versions"] == {"engine_version": "v-test"}
        assert "results" not in status
        assert len(json.dumps(status)) < 2_000
    finally:
        api._jobs.pop("large-scan", None)


def test_completed_scan_job_payload_can_be_compacted_only_when_results_match(monkeypatch, tmp_path):
    import json
    from caisen import store

    monkeypatch.setattr(store, "database_path", lambda: str(tmp_path / "scan.sqlite3"))
    store.create_scan_run("compact-ok", {"kinds": ["stock"], "versions": {"engine_version": "v-test"}},
                          total=1, rule_version="caisen-v17", app_version="0.9.0")
    store.save_scan_result("compact-ok", 0, {"code": "000001", "kind": "stock", "action": "WAIT", "patterns": []})
    store.update_scan_run("compact-ok", status="completed", progress=1, scanned=1, total=1,
                          inventory={"stock_count": 1})
    with store.connect() as db:
        db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   ("compact-ok", "completed", json.dumps({"status": "completed", "results": [{"large": "x" * 10000}]})))
    assert store.compact_completed_scan_job("compact-ok") is True
    with store.connect() as db:
        payload = json.loads(db.execute("SELECT payload FROM jobs WHERE job_id='compact-ok'").fetchone()[0])
    assert "results" not in payload
    assert payload["result_count"] == 1

    store.create_scan_run("compact-mismatch", {"kinds": ["stock"]}, total=2,
                          rule_version="caisen-v17", app_version="0.9.0")
    store.update_scan_run("compact-mismatch", status="completed", progress=1, scanned=2, total=2)
    with store.connect() as db:
        original = json.dumps({"status": "completed", "results": [{"preserve": True}]})
        db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   ("compact-mismatch", "completed", original))
    assert store.compact_completed_scan_job("compact-mismatch") is False
    with store.connect() as db:
        preserved = db.execute("SELECT payload FROM jobs WHERE job_id='compact-mismatch'").fetchone()[0]
    assert preserved == original


def test_scan_exposes_and_uses_exact_same_day_emitted_signal_ids(monkeypatch):
    instrument = InstrumentFile("000001", "stock", "one.csv")
    daily = _bars(pd.bdate_range("2026-10-12", "2026-10-16"))
    weekly = _bars(pd.to_datetime(["2026-10-16"]))
    event_date = "2026-10-16"
    signal = {"signal_date": event_date, "entry_date": None,
              "pattern_ids": ["primary", "secondary-flag"],
              "pattern_types": ["w_bottom", "bull_flag"],
              "pattern": {"pattern_id": "primary"},
              "patterns": [{"pattern_id": "primary"}, {"pattern_id": "secondary-flag"}]}
    recognition = {"patterns": [], "signals": [signal], "performance": {}}
    seen = {}
    monkeypatch.setattr(api, "load_daily_bars", lambda item, root: MarketDataResult(item, bars=daily))
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, **kwargs: MarketDataResult(
        instrument, bars=weekly, quality_status="usable"))
    monkeypatch.setattr(api, "recognize_daily_history", lambda *args, **kwargs: recognition)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "synthesize_decision", lambda code, kind, patterns, *args, **kwargs: (
        seen.update(kwargs) or {"code": code, "kind": kind, "action": "WAIT",
                                "rank": {"action_priority": 1, "confirmed_timeframes": 0,
                                         "reward_risk": 0, "risk_pct": 1,
                                         "confirmation_ordinal": 0,
                                         "tie_break": {"kind": kind, "code": code}}}))

    row, _ = api._analyze_scan_instrument(instrument, {"timeframes": ["1d"]}, root="/tmp")

    assert seen["allowed_pattern_ids_by_timeframe"] == {"1d": ["primary"]}
    assert "strategy_events" not in row


def test_detail_plan_suppresses_same_day_secondary_bull_flag():
    flag = Pattern(pattern_id="secondary-flag", code="000001", kind="stock", timeframe="1d",
                   pattern_type="bull_flag", direction="bullish", state="confirmed",
                   formed_at="2026-10-01", confirmed_at=3, confirmation_date="2026-10-16")
    plan = {"decision":"CONSIDER_BUY","reason_codes":[]}

    result = api._gate_detail_plan_to_primary_signal(
        plan, flag, "1d", "2026-10-16", {"1d":["primary"]})

    assert result["decision"] == "WAIT"
    assert result["reason_codes"] == ["not_backtest_primary_event"]


def test_analysis_maps_window_indices_to_full_chart(monkeypatch, tmp_path):
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "state.sqlite3"))
    instrument = InstrumentFile("000001", "stock", "one.csv")
    daily = _bars(pd.bdate_range("2010-01-04", "2026-10-28"))
    weekly = _bars(pd.date_range("2010-01-08", "2026-10-23", freq="W-FRI"))
    main = Pattern(
        pattern_id="main-chart", code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
        direction="bullish", state="forming", formed_at="2022-01-01", confirmed_at=None,
        key_points={"pivot_1_index":2,"pivot_1_date":"2021-10-29","pivot_1":8.0,
                    "pivot_2_index":20,"pivot_2_date":"2021-11-24","pivot_2":12.0,
                    "pivot_3_index":40,"pivot_3_date":"2021-12-22","pivot_3":8.5},
        evidence=[{"code":"main_structure","value":{"role":"main","scale":"5y",
                   "signal_eligible":False,"support_lower":8.0,"support_upper":8.5,"neckline":{"slope":0,"intercept":12}}},
                  {"code":"main_selection","value":{"selected":True,"decided_at":"2026-10-28"}},
                  {"code":"neckline_tests","value":[{"group_id":"touch-a","pivot_date":"2021-11-24",
                   "known_date":"2021-11-26","level":12.0,"valid":True}]}],
    )
    monkeypatch.setattr(api, "_market_data", lambda code, kind: MarketDataResult(instrument, bars=daily, quality_status="usable"))
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, **kwargs: MarketDataResult(instrument, bars=weekly, quality_status="usable"))
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: [])
    seen_background={}
    def recognize_daily(frame, code, kind, **kwargs):
        seen_background["1d"]=(frame.date.min(),frame.date.max(),len(frame))
        return {"patterns": [main], "signals": [], "performance": {}}
    def recognize_weekly(frame, code, kind, **kwargs):
        seen_background["1w"]=(frame.date.min(),frame.date.max(),len(frame))
        return {"patterns": [], "signals": [], "performance": {}}
    monkeypatch.setattr(api, "recognize_daily_history", recognize_daily)
    monkeypatch.setattr(api, "recognize_weekly_history", recognize_weekly)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)

    result = api.analysis("000001", "stock", "1d")

    daily_offset = api._main_bottom_chart_offset(daily, "1d")
    weekly_offset = api._main_bottom_chart_offset(weekly, "1w")
    assert seen_background == {
        "1d":(daily.date.iloc[daily_offset],daily.date.max(),len(daily)-daily_offset),
        "1w":(weekly.date.iloc[weekly_offset],weekly.date.max(),len(weekly)-weekly_offset),
    }
    assert result["patterns"][0]["chart_offset"] == daily_offset
    assert result["decision"]["patterns"][0]["chart_offset"] == daily_offset
    assert result["_chart_offset"] == daily_offset
    assert result["_bars"][daily_offset]["date"] == str(daily.date.iloc[daily_offset].date())
    assert api.analysis("000001", "stock", "1d")["patterns"] == result["patterns"]
    with api.connect() as db:
        for table in ("patterns", "pattern_snapshots", "pattern_lifecycle_events"):
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_snapshot_keeps_all_main_events_of_one_existing_family():
    def make_event(pattern_id, formed_at, is_main):
        return Pattern(
            pattern_id=pattern_id, code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
            direction="bullish", state="forming", formed_at=formed_at, confirmed_at=None,
            evidence=[{"code":"main_structure","value":{"role":"main"}}] if is_main else [],
        )

    older_main = make_event("main-old", "2021-01-01", True)
    newer_main = make_event("main-new", "2025-01-01", True)
    older_legacy = make_event("legacy-old", "2021-01-01", False)
    newer_legacy = make_event("legacy-new", "2025-01-01", False)

    result = current_pattern_snapshot([older_main, newer_main, older_legacy, newer_legacy])

    assert {pattern.pattern_id for pattern in result} == {"main-old", "main-new", "legacy-new"}


def test_current_decision_uses_selected_main_not_historical_confirmations():
    def event(pattern_id, direction, state, selected, role="main"):
        return Pattern(
            pattern_id=pattern_id, code="000001", kind="stock", timeframe="1d", pattern_type="w_bottom",
            direction=direction, state=state, formed_at="2020-01-01", confirmed_at=10,
            evidence=([{"code":"main_structure","value":{"role":role,"signal_eligible":True}},
                      {"code":"main_selection","value":{"selected":selected}}] if role else []),
        )
    historical_bear = event("old-main", "bearish", "confirmed", False)
    selected_main = event("selected-main", "bullish", "forming", True)
    legacy = event("legacy", "bearish", "confirmed", False, role=None)

    selected = _decision_patterns_by_timeframe({"1d":[historical_bear,selected_main,legacy]})

    assert [item.pattern_id for item in selected["1d"]] == ["selected-main"]


def test_current_decision_does_not_reselect_an_expired_structure():
    expired = Pattern(
        pattern_id="expired", code="000008", kind="stock", timeframe="1d",
        pattern_type="w_bottom", direction="bullish", state="expired",
        formed_at="2024-10-18", confirmed_at=None, evidence=[
            {"code": "main_structure", "value": {"role": "main", "signal_eligible": False,
                "terminal_state": "expired"}},
            {"code": "main_selection", "value": {"selected": True}},
        ])

    selected = _decision_patterns_by_timeframe({"1d": [expired]})

    assert selected["1d"] == []
