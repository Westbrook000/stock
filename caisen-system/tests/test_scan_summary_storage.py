from caisen.store import create_scan_run, query_scan_results, save_scan_result, connect


def test_scan_storage_only_keeps_visible_patterns(monkeypatch, tmp_path):
    monkeypatch.setenv("CAISEN_DB", str(tmp_path / "state.sqlite3"))
    create_scan_run("scan", {}, total=1)
    daily = {"pattern_id":"signal","pattern_type":"w_bottom","timeframe":"1d","state":"confirmed"}
    weekly = {"pattern_id":"weekly","pattern_type":"triangle_bottom","timeframe":"1w","state":"forming"}
    save_scan_result("scan", 0, {"code":"000123","kind":"stock","action":"CONSIDER_BUY",
        "selected_plan":{"pattern_id":"signal","timeframe":"1d"},
        "patterns":[daily,weekly,dict(daily,state="invalidated",pattern_type="bull_flag")],
        "main_structure_events":[{"evidence":"x"*100000}],
        "mini_bars":[{"open":1,"close":2,"low":1,"high":2}]*48})
    rows, total = query_scan_results("scan", pattern_types=["w_bottom"],timeframes=["1d"])
    assert total == 1 and len(rows[0]["patterns"]) == 2
    assert query_scan_results("scan",pattern_types=["bull_flag"])[1] == 0
    assert query_scan_results("scan",pattern_types=["w_bottom"], timeframes=["1w"])[1] == 0
    with connect() as db:
        pats, payload = db.execute("SELECT patterns_json,payload_json FROM scan_results").fetchone()
    assert len(payload) < 1000 and "patterns" not in payload and "mini_bars" not in payload
    assert "evidence" not in pats and "pattern_id" not in pats


def test_offline_migration_preserves_rank_and_is_repeatable(tmp_path, monkeypatch):
    import json
    import sqlite3

    from scripts.migrate_scan_summaries import migrate
    from caisen.store import update_scan_run

    source, target, second = (tmp_path / name for name in ("old.db", "new.db", "second.db"))
    monkeypatch.setenv("CAISEN_DB", str(source))
    create_scan_run("scan", {"timeframes": ["1d", "1w"]}, total=2)
    update_scan_run("scan", status="completed", scanned=2)
    old = {"code": "100002", "name": "样本", "kind": "etf", "category": "equity",
           "action": "WATCH", "latest_date": "2026-11-20", "latest_close": 1.23,
           "patterns": [{"pattern_type": "w_bottom", "state": "forming", "timeframe": "1d",
                         "pattern_id": "p", "formed_at": "2026-10-01"}],
           "watch": [{"timeframe": "1d", "pattern_id": "p", "distance_atr": 2.1}],
           "main_structure_events": [{"evidence": "x" * 10000}]}
    with connect() as db:
        db.execute("""INSERT INTO scan_results(run_id,ordinal,code,kind,name,category,action,payload_json)
                      VALUES(?,?,?,?,?,?,?,?)""",
                   ("scan", 0, old["code"], old["kind"], old["name"], old["category"],
                    old["action"], json.dumps(old)))
        db.execute("INSERT INTO backtests(backtest_id,payload) VALUES('b','{}')")
    save_scan_result("scan", 1, dict(old, code="100001", action="WAIT"))
    original = source.read_bytes()
    first = migrate(source, target)
    assert migrate(target, second)["scan"] == first["scan"]
    assert source.read_bytes() == original
    monkeypatch.setenv("CAISEN_DB", str(target))
    rows, total = query_scan_results("scan")
    assert total == 2
    assert [row["code"] for row in rows] == ["100002", "100001"]
    assert query_scan_results("scan", action="WATCH", kind="etf", pattern_types=["w_bottom"],
                              timeframes=["1d"])[1] == 1
    assert "main_structure_events" not in rows[0]
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT backtest_id FROM backtests").fetchall() == [("b",)]


def test_offline_migration_refuses_active_source_and_existing_target(tmp_path, monkeypatch):
    import pytest

    from scripts.migrate_scan_summaries import migrate

    source, target = tmp_path / "old.db", tmp_path / "new.db"
    monkeypatch.setenv("CAISEN_DB", str(source))
    create_scan_run("active", {}, total=2)
    before = source.read_bytes()
    with pytest.raises(RuntimeError, match="active task"):
        migrate(source, target)
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        migrate(source, target)
