import pandas as pd
import pytest
from fastapi import HTTPException

from caisen.market import InstrumentFile, MarketDataResult
from caisen import api
from caisen.schemas import Pattern


def make_bars():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16"]),
        "open": [10.0, 10.0, 10.0], "high": [10.02, 10.04, 10.05],
        "low": [9.98, 9.99, 10.0], "close": [10.0, 10.02, 10.03],
        "volume": [1000, 1200, 1500], "name": ["样本公司"] * 3,
    })
    frame.attrs["quality"] = {"volume_unit_verified": False, "corporate_actions_verified": False,
                               "trading_rules_verified": False, "category": "unclassified"}
    frame.attrs["provenance"] = {"data_version": "fixture-v1"}
    return frame


def test_scan_worker_merges_one_row_per_symbol_sorts_buy_first_and_keeps_failures(monkeypatch):
    instruments = [InstrumentFile("000001", "stock", "one.csv"),
                   InstrumentFile("000002", "etf", "two.csv"),
                   InstrumentFile("000003", "stock", "bad.csv")]
    bullish = Pattern(pattern_id="pattern-1", code="000001", kind="stock", timeframe="1d",
               pattern_type="w_bottom", direction="bullish", state="confirmed",
               formed_at="2026-10-01", confirmed_at=10, confirmation_date="2026-10-15",
               invalidation=9.6, targets=[12.0], volume_verified=True,
               execution_rules_verified=False)

    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    def load(instrument, root):
        if instrument.code == "000003":
            raise ValueError("unreadable")
        return MarketDataResult(instrument, bars=make_bars(), quality_status="usable_with_warnings",
                                warnings=["fixture provenance warning"], data_version="fixture-v1")
    def weekly(result, **kwargs):
        return MarketDataResult(result.instrument, quality_status="insufficient", warnings=["calendar missing"])
    monkeypatch.setattr(api, "load_daily_bars", load)
    monkeypatch.setattr(api, "aggregate_weekly", weekly)
    monkeypatch.setattr(api, "detect_patterns", lambda bars, code, kind, tf: [bullish] if code == "000001" and tf == "1d" else [])
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)

    api._scan_worker("fixture-job", {"kind": "all", "limit": 10, "timeframes": ["1d", "1w"]})
    result = api._jobs["fixture-job"]

    assert result["status"] == "completed"
    assert result["total"] == 3
    assert len(result["results"]) == 3
    assert [row["code"] for row in result["results"]] == ["000001", "000002", "000003"]
    assert result["results"][0]["action"] == "CONSIDER_BUY"
    assert len(result["results"][0]["mini_bars"]) == 3
    assert result["results"][-1]["action"] == "INSUFFICIENT_DATA"
    assert result["inventory"]["failed_count"] == 1
    assert result["inventory"]["inventory_at"]


def test_workbench_html_contains_one_click_scan_and_detail_navigation():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert "全量扫描" in html
    assert "miniCandles" in js
    assert "openTicker" in js
    assert "decisionMarkup" in js


def test_scan_filters_use_chinese_category_labels_and_explain_weekly_preview():
    from pathlib import Path
    html = (Path(__file__).parents[1] / "caisen" / "static" / "index.html").read_text(encoding="utf-8")

    assert '<option value="equity">股票型</option>' in html
    assert '<option value="bond">债券型</option>' in html
    assert '<option value="money">货币型</option>' in html
    assert '<option value="gold">黄金</option>' in html
    assert '<option value="commodity">商品</option>' in html
    assert '<option value="crossborder">跨境</option>' in html
    assert "周线只有在交易日历核实完整周后才参与正式判断" in html


def test_scan_results_filter_after_completion_without_resubmitting_the_scan():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert 'id="scanResultFilters"' in html
    assert 'id="scanActionFilter"' in html
    assert 'id="scanShapeFilter"' in html
    assert 'id="buyOnlyBtn"' in html
    assert 'id="resetScanFilters"' in html
    assert "全部十招" in html
    assert "全部 12 招" not in html
    assert "CaisenScanFilters.filterScanRows(scanRows" in js
    assert "筛选后 ${visibleScanRows.length} / ${scanRows.length}" in js
    assert "没有标的符合当前筛选" in js
    assert "pattern_type:$('scanPattern')" not in js
    assert "/static/scan-filters.js" in html
    filtering = js.split("function applyScanFilters", 1)[1].split("function renderScan", 1)[0]
    assert "fetch(" not in filtering


def test_pattern_card_distinguishes_engine_and_source_rule_versions():
    from pathlib import Path
    js = (Path(__file__).parents[1] / "caisen" / "static" / "app.js").read_text(encoding="utf-8")

    assert "引擎 ${escapeHtml(pattern.rule_version||'—')}" in js
    assert "规则集 ${escapeHtml(pattern.source_rule_version||'—')}" in js


def test_pattern_card_distinguishes_pivot_confirmation_from_price_breakout():
    from pathlib import Path
    js = (Path(__file__).parents[1] / "caisen" / "static" / "app.js").read_text(encoding="utf-8")

    assert "const confirmationLabel=pattern.confirmation_date?'价格确认':'结构点确认'" in js
    assert "${confirmationLabel} ${escapeHtml(confirmed)}" in js


def test_detail_chart_returns_full_history_and_preview_weekly_bars(monkeypatch):
    from pathlib import Path

    daily = make_bars()
    daily = pd.concat([daily] * 174, ignore_index=True)
    daily["date"] = pd.bdate_range("2025-01-01", periods=len(daily))
    daily_result = MarketDataResult(
        InstrumentFile("000001", "stock", "one.csv"),
        bars=daily,
        quality_status="usable_with_warnings",
    )
    weekly_preview = daily.tail(52).copy()
    weekly_result = MarketDataResult(
        daily_result.instrument,
        bars=pd.DataFrame(columns=daily.columns),
        provisional_bars=weekly_preview,
        quality_status="insufficient",
        warnings=["calendar missing"],
    )
    monkeypatch.setattr(api, "_market_data", lambda code, kind, asof: daily_result)
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, asof=None, calendar=None: weekly_result)
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)

    daily_payload = api.analysis("000001", timeframe="1d")
    weekly_payload = api.analysis("000001", timeframe="1w")

    assert len(daily_payload["_bars"]) == len(daily)
    assert daily_payload["_chart_offset"] == len(daily) - 500
    assert weekly_payload["chart_is_preview"] is True
    assert len(weekly_payload["_bars"]) == len(weekly_preview)


def test_detail_chart_exposes_range_controls_and_both_zoom_modes():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert "chartRange" in html
    assert "type:'inside'" in js
    assert "type:'slider'" in js
    assert "resize" in js
    assert "renderDetailChart(lastAnalysis" in js
    assert "显示 ${bars.length.toLocaleString()} / ${allBars.length.toLocaleString()}" in js
    assert "grid:[{left:120,right:70,top:66,height:'55%'},{left:120,right:70,top:'74%',height:'14%'}]" in js


def test_single_symbol_refresh_keeps_success_message_after_reanalysis():
    from pathlib import Path
    js = (Path(__file__).parents[1] / "caisen" / "static" / "app.js").read_text(encoding="utf-8")
    refresh = js.split("function watchRefreshJob", 1)[1].split("async function startRefresh", 1)[0]

    assert refresh.index("await analyze()") < refresh.index("formatRefreshSummary(current,meta.one)")


def test_refresh_report_shows_provenance_dates_attempts_and_safe_failures():
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).parents[2]
    html = (root / "caisen-system/caisen/static/index.html").read_text(encoding="utf-8")
    app = (root / "caisen-system/caisen/static/app.js").read_text(encoding="utf-8")
    job = {
        "job_id": "fixture-job",
        "status": "completed",
        "success_count": 1,
        "failure_count": 1,
        "results": [{"code": "000001", "kind": "stock", "first_date": "1991-04-03",
                     "last_date": "2026-10-23", "source": "AkShare Sina daily history",
                     "provider_attempts": 2, "previous_sha256": "old", "file_sha256": "new"}],
        "failures": [{"code": "510300", "kind": "etf", "error": "<script>alert(1)</script>"}],
    }
    script = """const report=require('./caisen-system/caisen/static/refresh_report.js');
let input='';process.stdin.on('data',chunk=>input+=chunk);process.stdin.on('end',()=>{
const job=JSON.parse(input);process.stdout.write(JSON.stringify({summary:report.formatRefreshSummary(job,true),markup:report.renderRefreshReport(job),interrupted:report.renderRefreshReport({...job,status:'interrupted'})}));});"""
    result = subprocess.run(["node", "-e", script], cwd=root, input=json.dumps(job),
                           text=True, capture_output=True, check=True)
    output = json.loads(result.stdout)

    assert "2026-10-23" in output["summary"]
    assert "2 次" in output["summary"]
    assert "AkShare Sina daily history" in output["markup"]
    assert "1991-04-03" in output["markup"]
    assert "失败 1" in output["markup"]
    assert "fixture-job" in output["markup"]
    assert 'data-resume-refresh="fixture-job"' in output["markup"]
    assert "重试失败标的" in output["markup"]
    assert 'data-resume-refresh="fixture-job"' in output["interrupted"]
    assert "&lt;script&gt;" in output["markup"]
    assert "<script>" not in output["markup"]
    assert "/static/refresh_report.js" in html
    assert "renderRefreshReport({...current,job_id:jobId})" in app
    assert "caisen-refresh-job" in app
    assert "/resume" in app
    assert "/api/v1/data/refresh/jobs" in app
    assert "renderRefreshRecoveryList" in app


def test_recoverable_refresh_jobs_can_be_discovered_without_browser_session(monkeypatch, tmp_path):
    import json
    from caisen.store import connect

    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "recoverable-list.sqlite3"))
    api._jobs.clear()
    inventory = [{"code": "999999", "kind": "stock", "path": "/missing/999999.csv"}]
    fixtures = [
        ("interrupted-job", "interrupted", 0, 0),
        ("partial-job", "completed", 1, 1),
        ("successful-job", "completed", 1, 0),
    ]
    for job_id, status, successes, failures in fixtures:
        payload = {"job_type": "data_refresh", "status": status, "scope": "single",
                   "inventory": inventory, "results": [], "failures": [],
                   "total": 1, "completed": successes + failures,
                   "success_count": successes, "failure_count": failures}
        api._persist_refresh_job(job_id, payload, initialize=True)
        with connect() as db:
            db.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, job_id))
    with connect() as db:
        db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   ("scan-job", "failed", json.dumps({"job_type": "scan"})))

    found = api.recoverable_refresh_jobs()

    assert {job["job_id"] for job in found["jobs"]} == {"interrupted-job", "partial-job"}
    assert all("inventory" not in job and "results" not in job for job in found["jobs"])
    assert next(job for job in found["jobs"] if job["job_id"] == "partial-job")["failure_count"] == 1


def test_refresh_recovery_formatter_renders_restart_resume_actions():
    import subprocess
    from pathlib import Path

    script = """const report=require('./caisen-system/caisen/static/refresh_report.js');
const jobs=[{job_id:'restart-job',scope:'single',status:'interrupted',success_count:1,failure_count:0,total:2,completed:1},
{job_id:'partial-job',scope:'all',status:'completed',success_count:4,failure_count:1,total:5,completed:5}];
process.stdout.write(report.renderRefreshRecoveryList(jobs));"""
    result = subprocess.run(["node", "-e", script], cwd=Path(__file__).parents[2],
                            text=True, capture_output=True, check=True)
    markup = result.stdout

    assert 'data-resume-refresh="restart-job"' in markup
    assert 'data-resume-refresh="partial-job"' in markup
    assert "恢复未完成的更新" in markup
    assert "重试失败标的" in markup


def test_refresh_worker_reports_partial_successes_and_failure_counts(monkeypatch):
    instruments = [InstrumentFile("000001", "stock", "one.csv"),
                   InstrumentFile("510300", "etf", "two.csv")]
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    monkeypatch.setattr(api, "_persist_refresh_job", lambda *args, **kwargs: None)
    def refresh(root, code, kind):
        if code == "510300":
            raise RuntimeError("provider throttled")
        return {"code": code, "kind": kind, "rows": 100}
    monkeypatch.setattr(api, "refresh_one", refresh)
    monkeypatch.setattr(api.time, "sleep", lambda duration: None)

    api._refresh_worker("refresh-fixture", {"scope": "all"})
    result = api._jobs["refresh-fixture"]

    assert result["status"] == "completed"
    assert result["total"] == 2
    assert result["success_count"] == 1
    assert result["failure_count"] == 1
    assert result["failures"][0]["error"] == "provider throttled"


def test_refresh_worker_persists_each_symbol_checkpoint(monkeypatch):
    import json

    instruments = [InstrumentFile("000001", "stock", "one.csv"),
                   InstrumentFile("510300", "etf", "two.csv")]
    checkpoints = []
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    monkeypatch.setattr(api, "_persist_refresh_job", lambda job_id, payload, **kwargs: checkpoints.append(
        ({key: value for key, value in json.loads(json.dumps(payload)).items()
         if key not in ("inventory", "results", "failures", "latest_results", "latest_failures")}, kwargs)))
    monkeypatch.setattr(api, "refresh_one", lambda root, code, kind: {
        "code": code, "kind": kind, "last_date": "2026-10-23"})
    monkeypatch.setattr(api.time, "sleep", lambda duration: None)

    api._refresh_worker("checkpoint-fixture", {"scope": "all"})

    running = [item for item, _ in checkpoints if item["status"] == "running"]
    assert len(running) == 3  # initial state, then one durable checkpoint per symbol
    assert running[1]["completed"] == 1
    assert "results" not in running[1] and "inventory" not in running[1]
    assert checkpoints[1][1]["item_update"]["record"]["code"] == "000001"
    assert checkpoints[-1][0]["status"] == "completed"


def test_refresh_worker_fatal_error_retains_last_durable_success(monkeypatch):
    instruments = [InstrumentFile("000001", "stock", "one.csv"),
                   InstrumentFile("510300", "etf", "two.csv")]
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    monkeypatch.setattr(api, "_persist_refresh_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "refresh_one", lambda root, code, kind: {"code": code, "kind": kind})
    monkeypatch.setattr(api.time, "sleep", lambda duration: (_ for _ in ()).throw(RuntimeError("worker interruption")))

    api._refresh_worker("fatal-fixture", {"scope": "all"})

    result = api._jobs["fatal-fixture"]
    assert result["status"] == "failed"
    assert result["error"] == "worker interruption"
    assert result["success_count"] == 1
    assert result["results"][0]["code"] == "000001"


def test_startup_marks_interrupted_refresh_jobs_and_resume_reuses_inventory(monkeypatch, tmp_path):
    import json
    from caisen.store import connect

    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "resume.sqlite3"))
    api._jobs.clear()
    payload = {"job_type": "data_refresh", "scope": "all", "status": "running",
               "inventory": [{"code": "000001", "kind": "stock", "path": "one.csv"},
                             {"code": "510300", "kind": "etf", "path": "two.csv"}],
               "results": [{"code": "000001", "kind": "stock", "last_date": "2026-10-23"}],
               "failures": [{"code": "510300", "kind": "etf", "error": "temporary failure"}],
               "success_count": 1, "failure_count": 1, "completed": 2, "total": 2}
    api._persist_refresh_job("resume-fixture", payload, initialize=True)
    with connect() as db:
        db.execute("UPDATE jobs SET status='running' WHERE job_id='resume-fixture'")

    assert api._recover_interrupted_refresh_jobs() == 1
    with connect() as db:
        recovered = json.loads(db.execute(
            "SELECT payload FROM jobs WHERE job_id=?", ("resume-fixture",)).fetchone()[0])
        saved_items = db.execute(
            "SELECT status FROM refresh_job_items WHERE job_id=? ORDER BY instrument_key",
            ("resume-fixture",)).fetchall()
    assert recovered["status"] == "interrupted"
    assert not {"inventory", "results", "failures"}.intersection(recovered)
    assert sorted(item["status"] for item in saved_items) == ["failure", "success"]

    captured = {}
    class FakeThread:
        def __init__(self, target, args, daemon):
            captured.update(target=target, args=args, daemon=daemon)
        def start(self):
            captured["started"] = True
    monkeypatch.setattr(api.threading, "Thread", FakeThread)

    result = api.resume_data_refresh("resume-fixture")

    assert result["status"] == "queued"
    assert result["job_id"] == "resume-fixture"
    assert captured["started"] is True
    resumed = captured["args"][1]
    assert len(resumed["inventory"]) == 2
    assert [item["code"] for item in resumed["results"]] == ["000001"]
    assert [item["code"] for item in resumed["failures"]] == ["510300"]


def test_refresh_job_checkpoint_reconstructs_inventory_and_results_after_memory_loss(monkeypatch, tmp_path):
    import json
    from caisen.store import connect

    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "normalized-refresh.sqlite3"))
    api._jobs.clear()
    payload = {"job_type": "data_refresh", "status": "running", "scope": "all",
               "inventory": [{"code": "000001", "kind": "stock", "path": "one.csv"},
                             {"code": "510300", "kind": "etf", "path": "two.csv"}],
               "results": [], "failures": [], "completed": 0, "total": 2}
    api._persist_refresh_job("normalized-fixture", payload, initialize=True)
    first = {"code": "000001", "kind": "stock", "rows": 100}
    second = {"code": "510300", "kind": "etf", "error": "HTTP 429"}
    api._persist_refresh_job("normalized-fixture", {**payload, "completed": 1},
                             item_update={"instrument": payload["inventory"][0],
                                          "status": "success", "record": first})
    api._persist_refresh_job("normalized-fixture", {**payload, "completed": 2},
                             item_update={"instrument": payload["inventory"][1],
                                          "status": "failure", "record": second})

    with connect() as db:
        summary = json.loads(db.execute(
            "SELECT payload FROM jobs WHERE job_id='normalized-fixture'").fetchone()[0])
    api._jobs.clear()  # model a process restart: only SQLite remains
    restored = api._load_job_payload("normalized-fixture")

    assert not {"inventory", "results", "failures"}.intersection(summary)
    assert len(json.dumps(summary)) < 300
    assert [item["code"] for item in restored["inventory"]] == ["000001", "510300"]
    assert restored["results"] == [first]
    assert restored["failures"] == [second]


def test_startup_migrates_legacy_inline_refresh_checkpoints(monkeypatch, tmp_path):
    import json
    from caisen.store import connect

    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "legacy-refresh.sqlite3"))
    api._jobs.clear()
    legacy = {"job_type": "data_refresh", "status": "running", "scope": "all",
              "inventory": [{"code": "000001", "kind": "stock", "path": "one.csv"},
                            {"code": "510300", "kind": "etf", "path": "two.csv"}],
              "results": [{"code": "000001", "kind": "stock", "rows": 100}],
              "failures": [{"code": "510300", "kind": "etf", "error": "HTTP 429"}],
              "success_count": 1, "failure_count": 1, "completed": 2, "total": 2}
    with connect() as db:
        db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   ("legacy-refresh-fixture", "running", json.dumps(legacy)))

    assert api._recover_interrupted_refresh_jobs() == 1
    with connect() as db:
        compact = json.loads(db.execute(
            "SELECT payload FROM jobs WHERE job_id='legacy-refresh-fixture'").fetchone()[0])
        count = db.execute(
            "SELECT count(*) FROM refresh_job_items WHERE job_id='legacy-refresh-fixture'").fetchone()[0]
    restored = api._load_job_payload("legacy-refresh-fixture")

    assert compact["status"] == "interrupted"
    assert not {"inventory", "results", "failures"}.intersection(compact)
    assert count == 2
    assert [item["code"] for item in restored["inventory"]] == ["000001", "510300"]
    assert restored["results"][0]["code"] == "000001"
    assert restored["failures"][0]["code"] == "510300"


def test_resumed_refresh_skips_successful_symbols_and_retries_failures(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "_persist_refresh_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "refresh_one", lambda root, code, kind: calls.append((code, kind)) or {
        "code": code, "kind": kind, "last_date": "2026-10-23"})
    monkeypatch.setattr(api.time, "sleep", lambda duration: None)
    inventory = [{"code": "000001", "kind": "stock", "path": "one.csv"},
                 {"code": "510300", "kind": "etf", "path": "two.csv"}]

    api._refresh_worker("resume-worker-fixture", {
        "job_type": "data_refresh", "scope": "all", "inventory": inventory,
        "results": [{"code": "000001", "kind": "stock", "last_date": "2026-10-23"}],
        "failures": [{"code": "510300", "kind": "etf", "error": "previous failure"}],
    })

    result = api._jobs["resume-worker-fixture"]
    assert calls == [("510300", "etf")]
    assert result["status"] == "completed"
    assert result["success_count"] == 2
    assert result["failure_count"] == 0


def test_completed_partial_refresh_can_be_resumed_from_persisted_job(monkeypatch, tmp_path):
    import json
    from caisen.store import connect

    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "partial-resume.sqlite3"))
    api._jobs.clear()
    payload = {"job_type": "data_refresh", "scope": "all", "status": "completed",
               "inventory": [{"code": "000001", "kind": "stock", "path": "one.csv"},
                             {"code": "510300", "kind": "etf", "path": "two.csv"}],
               "results": [{"code": "000001", "kind": "stock", "last_date": "2026-10-23"}],
               "failures": [{"code": "510300", "kind": "etf", "error": "HTTP 429"}],
               "success_count": 1, "failure_count": 1, "completed": 2, "total": 2}
    api._persist_refresh_job("partial-resume-fixture", payload, initialize=True)
    with connect() as db:
        db.execute("UPDATE jobs SET status='completed' WHERE job_id='partial-resume-fixture'")

    captured = {}
    class FakeThread:
        def __init__(self, target, args, daemon):
            captured.update(target=target, args=args, daemon=daemon)
        def start(self):
            captured["started"] = True
    monkeypatch.setattr(api.threading, "Thread", FakeThread)

    result = api.resume_data_refresh("partial-resume-fixture")

    assert result["resumed"] is True
    assert captured["started"] is True
    assert captured["args"][1]["results"][0]["code"] == "000001"
    assert captured["args"][1]["failures"][0]["code"] == "510300"


def test_refresh_endpoint_requires_single_symbol_or_explicit_all_scope(monkeypatch, tmp_path):
    captured = {}
    from pathlib import Path
    root = tmp_path
    path = root / "data" / "stocks" / "000001.csv"
    path.parent.mkdir(parents=True)
    path.write_text("date,open,high,low,close,volume\n2026-01-01,1,1,1,1,1\n", encoding="utf-8")
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "_persist_refresh_job", lambda *args, **kwargs: None)
    class FakeThread:
        def __init__(self, target, args, daemon):
            captured.update(target=target, args=args, daemon=daemon)
        def start(self):
            captured["started"] = True
    monkeypatch.setattr(api.threading, "Thread", FakeThread)

    result = api.start_data_refresh({"code": "000001", "kind": "stock"})

    assert result["status"] == "queued"
    assert result["scope"] == "single"
    assert captured["started"] is True
    payload = captured["args"][1]
    assert payload["job_type"] == "data_refresh"
    assert payload["scope"] == "single"
    assert payload["inventory"][0]["code"] == "000001"
    assert payload["inventory"][0]["kind"] == "stock"


def test_refresh_endpoint_rejects_symbol_missing_from_local_inventory(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "_persist_refresh_job", lambda *args, **kwargs: None)
    class FakeThread:
        def __init__(self, target, args, daemon):
            captured["created"] = True
        def start(self):
            captured["started"] = True
    monkeypatch.setattr(api.threading, "Thread", FakeThread)

    with pytest.raises(HTTPException) as error:
        api.start_data_refresh({"code": "000001", "kind": "stock"})

    assert error.value.status_code == 404
    assert not captured
