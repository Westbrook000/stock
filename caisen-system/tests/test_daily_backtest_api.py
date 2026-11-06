from pathlib import Path
from types import SimpleNamespace

import pytest

from caisen import api
from fastapi import HTTPException


def test_daily_backtest_routes_use_fixed_universe_and_persist_report(monkeypatch, tmp_path):
    from caisen import daily_research
    calls = {}
    persisted = []
    report_path = tmp_path / "daily-report.html"
    report_path.write_text("<h1>Daily research report</h1>", encoding="utf-8")
    effective = {"workers": 8, "worker_source": "benchmark-current", "recognizer": "boundary_first",
                 "execution_mode": "spawn", "strategy_version": daily_research.STRATEGY_VERSION,
                 "recognizer_version": daily_research.RECOGNIZER_VERSION,
                 "engine_version": daily_research.BOUNDARY_FIRST_ENGINE_VERSION,
                 "source_rule_version": daily_research.MAIN_RULES.get("source_rule_version"),
                 "versions": daily_research._version_metadata(),
                 "recognition_rules": {"candidate_reversal_atr_multiple": 2.0,
                                        "breakout_close_buffer_previous_atr_multiple": 1.0,
                                        "recognition_experiment": None},
                 "trading_rules": {"initial_cash_per_symbol": 100000, "lot_size": 100,
                                   "stop_fraction": 0.08, "target_fraction": 0.2,
                                   "commission": 0.0, "slippage": 0.0,
                                   "max_hold_sessions": None, "end_of_period_close": True}}

    def fake_runner(**kwargs):
        calls.update(kwargs)
        kwargs["progress_callback"]({"stage": "symbols", "completed": 2, "total": 1000,
                                     "symbol": "000001", "elapsed_seconds": 0.2,
                                     "active_symbols": [{"kind": "stock", "code": "000001",
                                                         "phase": "recognition", "bars_completed": 520,
                                                         "bars_total": 800, "current_date": "2025-04-03"}]})
        return {"status": "partial_failed", "run_id": "run-fixture",
                "strategy_version": effective["strategy_version"],
                "recognizer_version": effective["recognizer_version"],
                "engine_version": effective["engine_version"],
                "source_rule_version": effective["source_rule_version"],
                "versions": effective["versions"],
                "account_files": [],
                "universe": {"id": "daily_800_stock_200_etf_v1", "total": 1000,
                             "selected_account_count": 1000},
                "parameters": {"initial_cash": 100000, "stop_pct": 0.08, "target_pct": 0.2,
                               "workers": 8, "worker_source": "benchmark-current",
                               "recognizer": "boundary_first", "execution_mode": "spawn"},
                "recognition_rules": effective["recognition_rules"],
                "trading_rules": effective["trading_rules"],
                "summary": {"trade_count": 3, "account_count": 997},
                "artifacts": {"html_path": str(report_path), "json_path": str(tmp_path / "daily.json")},
                "timings": {"elapsed_seconds": 1.0}}

    import sys
    monkeypatch.setitem(sys.modules, "caisen.daily_research", SimpleNamespace(
        run_daily_research=fake_runner,
        resolve_daily_research_config=lambda execution_mode: {**effective, "execution_mode": execution_mode}))
    persist_job = api._persist_job
    def capture_persisted_job(job_id, job):
        persisted.append({**job, "current": dict(job.get("current") or {})})
        persist_job(job_id, job)
    monkeypatch.setattr(api, "_persist_job", capture_persisted_job)
    class InlineThread:
        def __init__(self, target, args=(), kwargs=None, **_):
            self.target, self.args, self.kwargs = target, args, kwargs or {}
        def start(self):
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(api.threading, "Thread", InlineThread)
    queued = api.start_daily_backtest({})
    assert queued["status"] == "queued"
    job_id = queued["job_id"]
    assert calls["universe_path"].name == "daily_800_stock_200_etf_v1.json"
    assert calls["workers"] == 8
    assert calls["recognizer"] == "boundary_first"
    assert calls["execution_mode"] == "spawn"
    assert calls["worker_source_override"] == "benchmark-current"
    task_params = api._daily_backtest_jobs[job_id]["params"]
    assert task_params["universe_id"] == "daily_800_stock_200_etf_v1"
    assert task_params["strategy"] == daily_research.STRATEGY_VERSION
    assert task_params["date_to"]
    assert task_params["workers"] == 8
    assert task_params["worker_source"] == "benchmark-current"
    assert task_params["recognizer"] == "boundary_first"
    assert task_params["recognition_rules"]["candidate_reversal_atr_multiple"] == 2.0
    assert task_params["recognition_rules"]["breakout_close_buffer_previous_atr_multiple"] == 1.0
    assert task_params["trading_rules"]["lot_size"] == 100
    assert task_params["output_dir"].endswith(job_id)

    routes = {(route.path, frozenset(getattr(route, "methods", None) or [])) for route in api.app.routes}
    assert ("/api/v1/backtests", frozenset({"POST"})) in routes
    assert ("/api/v1/backtests", frozenset({"GET"})) in routes
    detail = api.get_daily_backtest(job_id)
    history = api.list_daily_backtests(page=1)
    report = api.daily_backtest_report(job_id)
    with pytest.raises(HTTPException) as retired:
        api.start_fixed_backtest({})
    assert detail["summary"]["trade_count"] == 3
    assert detail["status"] == "partial_failed"
    assert detail["completed"] == 997 and detail["total"] == 1000
    assert any(item.get("current", {}).get("symbol") == "000001"
               and item["current"].get("bars_completed") == 520
               and item["current"].get("bars_total") == 800 for item in persisted)
    assert detail["active_symbols"] == []
    assert detail["strategy_version"] == daily_research.STRATEGY_VERSION
    assert detail["engine_version"] == daily_research.BOUNDARY_FIRST_ENGINE_VERSION
    assert detail["report_url"] == f"/api/v1/backtests/{job_id}/report"
    assert history["total"] == 1
    assert history["items"][0]["strategy_version"] == daily_research.STRATEGY_VERSION
    assert history["items"][0]["engine_version"] == daily_research.BOUNDARY_FIRST_ENGINE_VERSION
    assert history["items"][0]["recognition_rules"] == effective["recognition_rules"]
    assert "Daily research report" in report.body.decode()
    assert retired.value.status_code == 410


def test_daily_backtest_rejects_runtime_configuration(monkeypatch):
    with pytest.raises(HTTPException) as error:
        api.start_daily_backtest({"cash": 50000})
    assert error.value.status_code == 400


def test_daily_backtest_reports_configuration_resolution_failure(monkeypatch):
    import sys
    def fail_to_resolve(**_kwargs):
        raise ValueError("no current engine policy")
    monkeypatch.setitem(sys.modules, "caisen.daily_research", SimpleNamespace(
        resolve_daily_research_config=fail_to_resolve))
    with pytest.raises(HTTPException) as error:
        api.start_daily_backtest({})
    assert error.value.status_code == 503
    assert "no current engine policy" in error.value.detail


def test_daily_report_account_links_are_confined_to_job_artifacts(monkeypatch, tmp_path):
    job_id = "account-fixture"
    project_root = tmp_path / "project"
    account_dir = project_root / "state" / "daily_backtests" / job_id / "accounts"
    account_dir.mkdir(parents=True)
    account_file = account_dir / "stock-000008.json"
    account_file.write_text('{"code":"000008","kind":"stock"}', encoding="utf-8")
    (account_dir / "stock-000009.json").write_text('{"code":"000009"}', encoding="utf-8")
    report_path = project_root / "state" / "daily_backtests" / job_id / "daily-research.html"
    report_path.write_text('<a href="accounts/stock-000008.json">account</a>', encoding="utf-8")
    monkeypatch.setattr(api, "PROJECT_ROOT", project_root)
    api._daily_backtest_jobs[job_id] = {
        "job_id": job_id, "job_type": "daily_backtest", "status": "completed",
        "account_files": [str(account_file.resolve())],
        "artifacts": {"account_directory": str(account_dir.resolve()),
                      "html_path": str(report_path.resolve())},
    }

    routes = {(route.path, frozenset(getattr(route, "methods", None) or [])) for route in api.app.routes}
    assert ("/api/v1/backtests/{job_id}/accounts/{filename}", frozenset({"GET"})) in routes
    report = api.daily_backtest_report(job_id)
    valid = api.daily_backtest_account(job_id, "stock-000008.json")
    assert 'href="accounts/stock-000008.json"' in report.body.decode()
    assert valid.status_code == 200
    assert Path(valid.path).read_text(encoding="utf-8") == '{"code":"000008","kind":"stock"}'

    with pytest.raises(HTTPException) as unknown:
        api.daily_backtest_account(job_id, "stock-000009.json")
    assert unknown.value.status_code == 404
    with pytest.raises(HTTPException) as traversal:
        api.daily_backtest_account(job_id, "../stock-000008.json")
    assert traversal.value.status_code == 404
    with pytest.raises(HTTPException) as missing_job:
        api.daily_backtest_account("unknown-job", "stock-000008.json")
    assert missing_job.value.status_code == 404
