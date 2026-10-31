import pandas as pd
import pytest
from concurrent.futures import ThreadPoolExecutor
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
        execution_rules_verified=False,
        evidence=[{"code":"main_structure", "value":{"role":"main","signal_eligible":True}},
                  {"code":"main_selection", "value":{"selected":True}}])

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
    monkeypatch.setattr(api, "detect_main_bottoms", lambda bars, code, kind, tf, **kwargs: [bullish] if code == "000001" and tf == "1d" else [])
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)

    api._scan_worker("fixture-job", {"timeframes":["1d","1w"]}, executor_factory=ThreadPoolExecutor)
    result = api._jobs["fixture-job"]

    assert result["status"] == "partial_failed"
    assert result["total"] == 3
    assert len(result["results"]) == 3
    assert [row["code"] for row in result["results"]] == ["000001", "000002", "000003"]
    assert result["results"][0]["action"] == "CONSIDER_BUY"
    assert len(result["results"][0]["mini_bars"]) == 3
    assert result["results"][-1]["action"] == "INSUFFICIENT_DATA"
    assert result["inventory"]["failed_count"] == 1
    assert result["inventory"]["inventory_at"]


def test_scan_worker_applies_kind_and_limit_before_loading(monkeypatch):
    instruments = [InstrumentFile("000001", "stock", "one.csv"),
                   InstrumentFile("000002", "etf", "two.csv"),
                   InstrumentFile("000003", "stock", "three.csv")]
    loaded = []
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    def load(instrument, root):
        loaded.append(instrument.code)
        return MarketDataResult(instrument, bars=make_bars(), quality_status="usable_with_warnings")
    monkeypatch.setattr(api, "load_daily_bars", load)
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, **kwargs: MarketDataResult(result.instrument, quality_status="insufficient"))
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)

    api._scan_worker("prefilter-job", {"kinds": ["etf"], "limit": 1, "timeframes": ["1d"]}, executor_factory=ThreadPoolExecutor)

    result = api._jobs["prefilter-job"]
    assert loaded == ["000002"]
    assert result["total"] == 1
    assert result["inventory"]["stock_count"] == 0
    assert result["inventory"]["etf_count"] == 1


def test_scan_request_rejects_invalid_prefilters():
    with pytest.raises(HTTPException):
        api.start_scan({"timeframes": ["1d"], "kinds": ["crypto"]})
    with pytest.raises(HTTPException):
        api.start_scan({"timeframes": ["1d"], "limit": 0})
    with pytest.raises(HTTPException):
        api.start_scan({"timeframes": ["1d"], "date_from": "2026-10-20", "date_to": "2026-10-01"})
    with pytest.raises(HTTPException):
        api.start_scan({"timeframes": ["1d"], "date_from": "not-a-date"})


def test_scan_date_window_slices_before_weekly_aggregation_and_pattern_detection(monkeypatch):
    result = MarketDataResult(InstrumentFile("000001", "stock", "one.csv"), bars=make_bars())
    weekly_dates = []
    detected_dates = []
    background_dates = []
    monkeypatch.setattr(api, "discover_instruments", lambda root: [result.instrument])
    monkeypatch.setattr(api, "load_daily_bars", lambda instrument, root: result)
    def aggregate(sliced, **kwargs):
        weekly_dates.extend(sliced.bars.date.dt.strftime("%Y-%m-%d").tolist())
        return MarketDataResult(sliced.instrument, bars=sliced.bars.tail(1).copy(), quality_status="usable")
    def detect(frame, *args, **kwargs):
        detected_dates.extend(frame.date.dt.strftime("%Y-%m-%d").tolist())
        background_dates.extend(kwargs["background_bars"].date.dt.strftime("%Y-%m-%d").tolist())
        return []
    monkeypatch.setattr(api, "aggregate_weekly", aggregate)
    monkeypatch.setattr(api, "detect_main_bottoms", detect)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)

    api._scan_worker("window-job", {"timeframes": ["1d", "1w"], "date_from": "2026-10-15", "date_to": "2026-10-15"}, executor_factory=ThreadPoolExecutor)

    row = api._jobs["window-job"]["results"][0]
    assert weekly_dates == ["2026-10-15"]
    assert detected_dates == ["2026-10-15", "2026-10-15"]
    assert background_dates == ["2026-10-14", "2026-10-15", "2026-10-14", "2026-10-15"]
    assert row["latest_date"] == "2026-10-15"
    assert row["analysis_window"] == {"requested_from":"2026-10-15", "requested_to":"2026-10-15",
                                       "from":"2026-10-15", "to":"2026-10-15",
                                       "latest_available":"2026-10-16", "historical":True}


def test_scan_date_window_with_no_rows_keeps_insufficient_data_row(monkeypatch):
    result = MarketDataResult(InstrumentFile("000001", "stock", "one.csv"), bars=make_bars())
    monkeypatch.setattr(api, "discover_instruments", lambda root: [result.instrument])
    monkeypatch.setattr(api, "load_daily_bars", lambda instrument, root: result)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)

    api._scan_worker("empty-window-job", {"timeframes": ["1d"], "date_from": "2027-01-01"}, executor_factory=ThreadPoolExecutor)

    row = api._jobs["empty-window-job"]["results"][0]
    assert row["action"] == "INSUFFICIENT_DATA"
    assert row["reason_codes"] == ["no_market_data_in_selected_date_range"]
    assert row["analysis_window"]["from"] is None


def test_scan_pattern_display_prefilter_does_not_narrow_scan_or_decision_inputs(monkeypatch):
    instrument = InstrumentFile("000001", "stock", "one.csv")
    w_bottom = Pattern(pattern_id="w", code="000001", kind="stock", timeframe="1d",
                       pattern_type="w_bottom", direction="bullish", state="confirmed",
                       formed_at="2026-10-01", confirmed_at=2, confirmation_date="2026-10-15",
                       evidence=[{"code":"main_structure", "value":{"role":"main"}},
                                 {"code":"main_selection", "value":{"selected":True}}])
    m_top = Pattern(pattern_id="m", code="000001", kind="stock", timeframe="1d",
                    pattern_type="m_top", direction="bearish", state="confirmed",
                    formed_at="2026-10-01", confirmed_at=2, confirmation_date="2026-10-15")
    monkeypatch.setattr(api, "discover_instruments", lambda root: [instrument])
    monkeypatch.setattr(api, "load_daily_bars", lambda item, root: MarketDataResult(item, bars=make_bars()))
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, **kwargs: MarketDataResult(result.instrument, quality_status="insufficient"))
    monkeypatch.setattr(api, "detect_main_bottoms", lambda *args, **kwargs: [w_bottom, m_top])
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("scan must not call the legacy recognizer")))
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)
    decision_inputs = []
    def synthesize(code, kind, patterns_by_tf, *args, **kwargs):
        decision_inputs.append(patterns_by_tf)
        return {"code":code,"kind":kind,"action":"WAIT","rank":{"action_priority":1,
                "confirmed_timeframes":0,"reward_risk":0,"risk_pct":1,"confirmation_ordinal":0,
                "tie_break":{"kind":kind,"code":code}}}
    monkeypatch.setattr(api, "synthesize_decision", synthesize)

    api._scan_worker("shape-job", {"timeframes": ["1d"]}, executor_factory=ThreadPoolExecutor)

    assert {pattern["pattern_type"] for pattern in api._jobs["shape-job"]["results"][0]["patterns"]} == {"w_bottom"}
    assert {pattern.pattern_type for pattern in decision_inputs[0]["1d"]} == {"w_bottom"}
    assert api._jobs["shape-job"]["results"][0]["pattern_migration"]["families"]["m_top"]["status"] == "unsupported"


def test_backtest_api_uses_date_window_and_returns_auditable_metadata(monkeypatch):
    source = MarketDataResult(InstrumentFile("000001", "stock", "one.csv"), bars=make_bars())
    captured = {}
    class FakeConnection:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, *args): pass
    monkeypatch.setattr(api, "_market_data", lambda code, kind: source)
    def run(bars, code, kind, timeframe, cash, start=None):
        captured["dates"] = bars.date.dt.strftime("%Y-%m-%d").tolist()
        return {"engine":"backtrader","commission":0,"slippage":0,"starting_cash":cash,
                "available_cash":cash,"marked_position_value":0,"ending_equity":cash,
                "trades":[],"order_events":[],"rejections":[],
                "accounting_invariants":{"equity_equals_cash_plus_marked_position":True,"zero_commission":True},
                "strategy_effectiveness":"deferred","limitations":["strategy not validated"]}
    monkeypatch.setattr(api, "run_backtest", run)
    monkeypatch.setattr(api, "connect", FakeConnection)

    result = api.backtest(api.BacktestRequest(code="000001", start="2026-10-15", end="2026-10-15", cash=50000))

    assert captured["dates"] == ["2026-10-15"]
    assert result["starting_cash"] == 50000
    assert result["analysis_window"] == {"from":"2026-10-15", "to":"2026-10-15",
                                          "latest_available":"2026-10-16", "historical":True}
    assert result["strategy_effectiveness"] == "deferred"


def test_backtrader_page_exposes_inputs_progress_and_accounting_ledgers():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for element_id in ("backtestCode", "backtestKind", "backtestTimeframe", "backtestStart", "backtestCash", "backtestBtn"):
        assert f'id="{element_id}"' in html
    assert 'data-panel="simulation"' in html
    assert "/api/v1/backtests" in js
    assert "正在运行本地模拟，请稍候" in js
    for field in ("available_cash", "marked_position_value", "ending_equity", "order_events", "rejections"):
        assert field in js
    assert "策略效果未验证" in js


def test_modified_workbench_assets_use_current_cache_version():
    from pathlib import Path
    html = (Path(__file__).parents[1] / "caisen" / "static" / "index.html").read_text(encoding="utf-8")
    version = "20261029v7"
    for asset in ("app.css", "refresh_report.js", "scan-filters.js", "pattern-chart.js", "http-client.js", "fixed-backtest-report.js", "app.js"):
        assert f"/static/{asset}?v={version}" in html
    assert "20261027v6" not in html


def test_workbench_html_contains_one_click_scan_and_detail_navigation():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert "开始扫描" in html
    assert "miniCandles" in js
    assert "openTicker" in js
    assert "decisionMarkup" in js


def test_scan_filters_use_chinese_category_labels_and_explain_weekly_preview():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert 'id="scanKind"' not in html
    assert 'id="scanShapeFilter"' in html
    assert 'id="scanDateFrom"' in html
    assert 'id="scanDateTo"' in html
    assert 'id="scanPatternTypes"' in html
    assert 'id="scanStock"' in html and 'id="scanEtf"' in html
    assert 'id="scanLimit"' in html
    assert "仅用于扫描完成后的结果展示筛选" in html
    assert "不会缩短扫描时间" in html
    assert "scanSettings.pattern_types" in js
    assert "打开详情将加载最新行情" in js
    assert "交易日历核实完整的周线才参与正式判断" in html


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
    monkeypatch.setattr(api, "_market_data", lambda code, kind: daily_result)
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, calendar=None: weekly_result)
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)

    daily_payload = api.analysis("000001", timeframe="1d")
    weekly_payload = api.analysis("000001", timeframe="1w")

    assert len(daily_payload["_bars"]) == len(daily)
    assert daily_payload["_chart_offset"] == len(daily) - 500
    assert weekly_payload["chart_is_preview"] is True
    assert len(weekly_payload["_bars"]) == len(weekly_preview)


def test_weekly_analysis_uses_full_daily_history_as_quality_background(monkeypatch):
    daily = MarketDataResult(InstrumentFile("000001", "stock", "one.csv"), bars=make_bars())
    weekly = MarketDataResult(daily.instrument, bars=daily.bars.tail(1).copy(), quality_status="usable")
    seen = {}
    monkeypatch.setattr(api, "_market_data", lambda code, kind: daily)
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, calendar=None: weekly)
    def detect(frame, code, kind, timeframe, **kwargs):
        seen[timeframe] = kwargs["background_bars"].date.dt.strftime("%Y-%m-%d").tolist()
        return []
    monkeypatch.setattr(api, "detect_main_bottoms", detect)
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "synthesize_decision", lambda *args, **kwargs: {"action": "WAIT"})

    api.analysis("000001", timeframe="1w")

    expected = daily.bars.date.dt.strftime("%Y-%m-%d").tolist()
    assert seen["1d"] == expected
    assert seen["1w"] == expected


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
    assert "data-resume-refresh" not in output["markup"]
    assert "重试失败标的" not in output["markup"]
    assert "data-resume-refresh" not in output["interrupted"]
    assert "&lt;script&gt;" in output["markup"]
    assert "<script>" not in output["markup"]
    assert "/static/refresh_report.js" in html
    assert "renderRefreshReport({...current,job_id:jobId})" in app
    assert "caisen-refresh-job" not in app
    assert "/resume" not in app
    assert "/api/v1/data/refresh/jobs" not in app
    assert "renderRefreshRecoveryList" not in app






def test_refresh_worker_reports_partial_successes_and_failure_counts(monkeypatch):
    instruments = [InstrumentFile("000001", "stock", "one.csv"),
                   InstrumentFile("510300", "etf", "two.csv")]
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
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
















def test_refresh_endpoint_requires_single_symbol_or_explicit_all_scope(monkeypatch, tmp_path):
    captured = {}
    from pathlib import Path
    root = tmp_path
    path = root / "data" / "stocks" / "000001.csv"
    path.parent.mkdir(parents=True)
    path.write_text("date,open,high,low,close,volume\n2026-01-01,1,1,1,1,1\n", encoding="utf-8")
    monkeypatch.setattr(api, "ROOT", root)
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


def test_scan_history_routes_filter_persisted_results_without_rescanning(monkeypatch, tmp_path):
    from caisen.store import create_scan_run, save_scan_result, update_scan_run
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "history.sqlite3"))
    create_scan_run("history-1", {"kinds": ["stock"], "date_to": "2026-10-15"}, total=1,
                    rule_version="rules-v1", app_version="app-v1")
    save_scan_result("history-1", 0, {
        "code": "000001", "kind": "stock", "name": "样本公司", "category": "银行",
        "action": "WAIT", "patterns": [{"pattern_type": "w_bottom", "timeframe": "1d"}],
        "latest_date": "2026-10-15",
    })
    update_scan_run("history-1", status="completed", progress=1, scanned=1)

    history = api.list_scans(page=1)
    result = api.scan_results("history-1", q="样本", kind="stock", category="银行", action="WAIT",
                              pattern_types="w_bottom", timeframes="1d", page=1, sort="code")

    assert history["items"][0]["run_id"] == "history-1"
    assert history["items"][0]["params"]["date_to"] == "2026-10-15"
    assert result["total"] == 1
    assert result["items"][0]["code"] == "000001"


def test_history_ui_separates_new_scan_and_reads_saved_results_with_kind_and_rank_defaults():
    root = api.WEB
    html = open(f"{root}/index.html", encoding="utf-8").read()
    js = open(f"{root}/app.js", encoding="utf-8").read()
    http = open(f"{root}/http-client.js", encoding="utf-8").read()
    assert 'data-scan-view="newScan"' in html and 'data-scan-view="scanHistory"' in html
    assert 'id="historyKind"' in html and 'value="rank" selected' in html
    assert "/api/v1/scans/${encodeURIComponent(historyRunId)}/results" in js
    assert "历史区间" in js and "打开详情将加载最新行情" in js
    assert "旧任务参数未记录" in js and "失败数未记录" in js and "规则版本未记录" in js
    assert "fetchJson('/api/v1/health')" in js and ".json()" not in js
    assert "JSON.parse(body)" in http and "response.status" in http


def test_fixed_batch_worker_builds_report_link_and_validates_inputs(monkeypatch, tmp_path):
    from pathlib import Path
    instrument = InstrumentFile("000001", "stock", "one.csv")
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "fixed.sqlite3"))
    monkeypatch.setattr(api, "ROOT", str(tmp_path))
    monkeypatch.setattr(api, "discover_instruments", lambda root: [instrument])
    monkeypatch.setattr(api, "load_daily_bars", lambda item, root: MarketDataResult(item, bars=make_bars()))

    def fake_batch(items, output_dir, development_cutoff=None, max_workers=1, progress_callback=None,
                   strategy_version="pattern_fixed_8_20_main_parent_v2"):
        assert strategy_version == "pattern_fixed_8_20_v1"
        assert len(items) == 1 and items[0]["timeframe"] == "1d"
        assert len(items[0]["bars"]) == 3 and items[0]["trade_start"] == "2026-10-15"
        report_path = Path(output_dir) / "fixed-backtest-report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text('<script src="fixed-backtest-report.js"></script>', encoding="utf-8")
        if progress_callback:
            progress_callback({"completed": 1, "total": 1, "current": "000001", "timeframe": "1d"})
        return {"strategy_version": "pattern_fixed_8_20_v1", "development_cutoff": "2026-10-15",
                "summary": {"trade_count": 0}, "groups": [],
                "output": {"html": str(report_path), "json": str(report_path.with_suffix(".json"))}}

    monkeypatch.setattr(api, "run_fixed_batch", fake_batch)
    api._fixed_backtest_jobs["fixed-fixture"] = {"job_id": "fixed-fixture", "job_type": "fixed_backtest",
        "status": "queued", "output_dir": str(tmp_path / "state" / "fixed_backtests" / "fixed-fixture"),
        "params": {"timeframes": ["1d"]}, "created_at": "2026-10-29T00:00:00+00:00"}
    api._fixed_backtest_worker("fixed-fixture", {"kinds": ["stock"], "timeframes": ["1d"],
                              "cash": 100000, "max_workers": 1,"date_from":"2026-10-15",
                              "strategy_version":"pattern_fixed_8_20_v1"})

    job = api.get_fixed_backtest_job("fixed-fixture")
    report = api.fixed_backtest_report("fixed-fixture")
    history = api.list_fixed_backtests(page=1)
    assert job["status"] == "completed"
    assert job["strategy_version"] == "pattern_fixed_8_20_v1"
    assert job["report_url"] == "/api/v1/fixed-backtests/fixed-fixture/report"
    assert history["total"] == 1 and history["items"][0]["job_id"] == "fixed-fixture"
    assert b"/static/fixed-backtest-report.js" in report.body
    api._fixed_backtest_jobs.pop("fixed-fixture")
    assert api.get_fixed_backtest_job("fixed-fixture")["status"] == "completed"
    assert b"/static/fixed-backtest-report.js" in api.fixed_backtest_report("fixed-fixture").body
    with pytest.raises(HTTPException):
        api.start_fixed_backtest({"kinds": ["crypto"]})
    with pytest.raises(HTTPException):
        api.start_fixed_backtest({"cash": 50000})
    with pytest.raises(HTTPException):
        api.start_fixed_backtest({"max_workers": 2})
    with pytest.raises(HTTPException):
        api.start_fixed_backtest({"scope": "recent100"})


def test_fixed_backtest_request_selects_new_default_or_explicit_v1(monkeypatch, tmp_path):
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "strategy-version.sqlite3"))
    monkeypatch.setattr(api, "ROOT", str(tmp_path))
    monkeypatch.setattr(api, "discover_instruments", lambda root: [])
    class NoStartThread:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
    monkeypatch.setattr(api.threading, "Thread", NoStartThread)

    latest = api.start_fixed_backtest({})
    baseline = api.start_fixed_backtest({"strategy_version":"pattern_fixed_8_20_v1"})

    assert api._fixed_backtest_jobs[latest["job_id"]]["params"]["strategy_version"] == "pattern_fixed_8_20_main_parent_v2"
    assert api._fixed_backtest_jobs[baseline["job_id"]]["params"]["strategy_version"] == "pattern_fixed_8_20_v1"
    with pytest.raises(HTTPException):
        api.start_fixed_backtest({"strategy_version":"unknown"})


def test_fixed_backtest_page_posts_to_async_batch_and_opens_report():
    from pathlib import Path
    html = (Path(api.WEB) / "index.html").read_text(encoding="utf-8")
    js = (Path(api.WEB) / "app.js").read_text(encoding="utf-8")
    assert 'data-panel="fixed-simulation"' in html
    assert 'id="fixedRunBtn"' in html and 'id="fixedCutoff"' in html and 'id="fixedScope"' in html
    assert 'id="fixedStrategyVersion"' in html
    assert 'strategy_version:$(' in js and "fixedStrategyVersion" in js
    assert "fetchJson('/api/v1/fixed-backtests'" in js
    assert "CaisenFixedBacktestReport" not in js
    assert "fixed-backtests/${encodeURIComponent(jobId)}" in js
    assert "job.report_url" in js and "打开完整回测报告" in js
    assert 'id="fixedHistory"' in html and "/api/v1/fixed-backtests?page=" in js
    assert "scope=$('fixedScope').value" in js and "全部可用历史" in html
    assert "params.sample_scope_detail" in js and "混合范围样本" in js


def test_fixed_backtest_recovery_marks_active_task_interrupted_and_keeps_checkpoint(monkeypatch, tmp_path):
    import json
    from caisen.store import connect
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "recovery.sqlite3"))
    output_dir = tmp_path / "fixed-output"
    report_dir = output_dir / "fixed-backtest-20261029T000000Z-ab12cd"
    report_dir.mkdir(parents=True)
    (report_dir / "fixed-backtest-report.html").write_text('<script src="fixed-backtest-report.js"></script>', encoding="utf-8")
    job = {"job_id": "running-fixed", "job_type": "fixed_backtest", "status": "running",
           "progress": .5, "params": {"date_from": "2026-10-01"}, "output_dir": str(output_dir)}
    with connect() as db:
        db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   (job["job_id"], "running", json.dumps(job)))

    api._fixed_backtest_jobs.pop("running-fixed", None)
    api._recover_fixed_backtests()

    recovered = api.get_fixed_backtest_job("running-fixed")
    report = api.fixed_backtest_report("running-fixed")
    assert recovered["status"] == "interrupted"
    assert recovered["params"]["date_from"] == "2026-10-01"
    assert b"/static/fixed-backtest-report.js" in report.body


def test_scan_worker_marks_all_instrument_failures_failed_and_keeps_history(monkeypatch, tmp_path):
    from caisen.store import create_scan_run, get_scan_run, query_scan_results
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "all-failed.sqlite3"))
    instruments = [InstrumentFile("000001", "stock", "bad.csv")]
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    monkeypatch.setattr(api, "load_daily_bars", lambda instrument, root: (_ for _ in ()).throw(ValueError("bad csv")))
    create_scan_run("all-failed", {"kinds": ["stock"], "timeframes": ["1d"]}, total=1,
                    rule_version="rules-v1", app_version="app-v1")

    api._scan_worker("all-failed", {"kinds": ["stock"], "timeframes": ["1d"]}, executor_factory=ThreadPoolExecutor)

    run = get_scan_run("all-failed")
    rows, count = query_scan_results("all-failed")
    assert run["status"] == "failed"
    assert count == 1
    assert rows[0]["state"] == "data_error"


def test_main_bottom_detail_groups_selected_parent_children_and_volume_support():
    from pathlib import Path
    root = Path(__file__).parents[1] / "caisen" / "static"
    js = (root / "app.js").read_text(encoding="utf-8")
    chart = (root / "pattern-chart.js").read_text(encoding="utf-8")
    assert "已选主结构" in js
    assert "已纳入主结构的子形态" in js
    assert "量能仅作辅助" in js
    assert "不作买入门槛" in js
    assert "neckline_tests" in chart
    assert "neckline_breakout" in chart


def test_scan_worker_uses_at_most_ten_workers_and_persists_ranked_progress(monkeypatch, tmp_path):
    import threading
    from caisen.store import create_scan_run, get_scan_run, query_scan_results

    instruments = [InstrumentFile(f"{index:06d}", "stock", f"{index}.csv") for index in range(12)]
    active = 0
    maximum = 0
    lock = threading.Lock()
    first_batch = threading.Barrier(10)
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "concurrent.sqlite3"))
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)

    def load(instrument, root):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if int(instrument.code) < 10:
                first_batch.wait(timeout=3)
            if instrument.code == "000011":
                raise ValueError("unreadable")
            return MarketDataResult(instrument, bars=make_bars(), quality_status="usable_with_warnings")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(api, "load_daily_bars", load)
    monkeypatch.setattr(api, "aggregate_weekly", lambda result, **kwargs: MarketDataResult(result.instrument, quality_status="insufficient"))
    monkeypatch.setattr(api, "detect_patterns", lambda *args, **kwargs: [])
    monkeypatch.setattr(api, "current_pattern_snapshot", lambda patterns: patterns)
    monkeypatch.setattr(api, "save_patterns", lambda payloads: None)
    monkeypatch.setattr(api, "synthesize_decision", lambda code, kind, *args, **kwargs: {
        "code": code, "kind": kind, "action": "CONSIDER_BUY" if code == "000009" else "WAIT",
        "patterns": [], "rank": {"action_priority": 0 if code == "000009" else 1,
                                   "confirmed_timeframes": 0, "reward_risk": 0, "risk_pct": 1,
                                   "confirmation_ordinal": 0, "tie_break": {"kind": kind, "code": code}},
    })
    original_update = api.update_scan_run
    progress_updates = []
    def record_progress(run_id, **fields):
        if "scanned" in fields:
            progress_updates.append(fields["scanned"])
        return original_update(run_id, **fields)
    monkeypatch.setattr(api, "update_scan_run", record_progress)
    create_scan_run("concurrent-scan", {"kinds": ["stock"], "timeframes": ["1d"]}, total=len(instruments))

    api._scan_worker("concurrent-scan", {"kinds": ["stock"], "timeframes": ["1d"]}, executor_factory=ThreadPoolExecutor)

    result = api._jobs["concurrent-scan"]
    saved, total = query_scan_results("concurrent-scan", limit=20)
    run = get_scan_run("concurrent-scan")
    assert maximum == 10
    assert result["status"] == "partial_failed"
    assert result["scanned"] == result["total"] == 12
    assert progress_updates == [*range(1, 13), 12]
    assert result["results"][0]["code"] == "000009"
    assert result["results"][-1]["state"] == "data_error"
    assert total == 12
    assert [row["code"] for row in saved] == [row["code"] for row in result["results"]]
    assert saved == result["results"]
    assert run["status"] == "partial_failed"
    assert run["scanned"] == 12 and run["progress"] == 1
