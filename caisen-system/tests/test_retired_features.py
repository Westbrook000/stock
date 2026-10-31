import inspect
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from caisen import api
from caisen.market import load_daily_bars, aggregate_weekly
from caisen.patterns import detect_patterns
from caisen.store import connect


def test_legacy_review_and_refresh_checkpoints_are_removed_without_losing_patterns(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE reviews(id INTEGER);INSERT INTO reviews VALUES(1);CREATE TABLE refresh_job_items(job_id TEXT);CREATE TABLE jobs(job_id TEXT PRIMARY KEY,status TEXT,payload TEXT,created_at TEXT);INSERT INTO jobs VALUES('old','running','{\"job_type\":\"data_refresh\"}','today');CREATE TABLE patterns(pattern_id TEXT PRIMARY KEY,payload TEXT,updated_at TEXT);INSERT INTO patterns VALUES('keep','{}','today');")
    with connect(str(path)) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert 'reviews' not in tables
        assert 'refresh_job_items' not in tables
        assert db.execute('SELECT count(*) FROM patterns').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0


def test_retired_routes_and_cutoff_interfaces_are_absent():
    routes=api.app.openapi()['paths']
    assert '/api/v1/patterns/{pattern_id}/review' not in routes
    assert '/api/v1/jobs/{job_id}/resume' not in routes
    assert '/api/v1/data/refresh/jobs' not in routes
    for function in (api.analysis,load_daily_bars,aggregate_weekly,detect_patterns):
        assert not ({'asof','as_of'} & set(inspect.signature(function).parameters))
    with pytest.raises(HTTPException) as error:
        api.analysis('000001',request=Request({'type':'http','query_string':b'asof=2020-01-01'}))
    assert error.value.status_code==422


def test_product_only_exposes_scan_and_detail_and_no_calibration_modules():
    root=Path(__file__).parents[1]
    html=(root/'caisen/static/index.html').read_text()
    for fragment in ('data-panel="review"','data-panel="replay"','id="asof"','id="scanKind"','id="scanPatternOptions"'):
        assert fragment not in html
    assert not (root/'caisen/calibration.py').exists()
    assert not (root/'caisen/calibration_pack.py').exists()
    assert not (root/'calibration').exists()


def test_refresh_progress_is_visible_only_for_the_live_process(monkeypatch,tmp_path):
    from caisen.market import InstrumentFile
    monkeypatch.setenv('CAISEN_DB',str(tmp_path/'jobs.sqlite3'))
    monkeypatch.setattr(api,'discover_instruments',lambda root:[InstrumentFile('000001','stock','one.csv')])
    monkeypatch.setattr(api,'refresh_one',lambda *args:{'code':'000001','kind':'stock','rows':2})
    api._refresh_worker('transient-refresh',{'scope':'all'})
    assert api.get_job('transient-refresh')['success_count']==1
    with connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0
    api._jobs.pop('transient-refresh')
    with pytest.raises(HTTPException) as error:
        api.get_job('transient-refresh')
    assert error.value.status_code==404
