import json
import os
import tempfile
import time
import unittest

from caisen.store import connect, get_pattern, save_pattern, save_patterns


class StoreTests(unittest.TestCase):
    def test_store_uses_current_assessment_module_version(self):
        from caisen.assessment import ASSESSMENT_VERSION
        from caisen.store import SCAN_ASSESSMENT_VERSION

        assert SCAN_ASSESSMENT_VERSION == ASSESSMENT_VERSION

    def test_batch_saves_patterns_and_snapshots_in_one_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                save_patterns([
                    {"pattern_id": "a", "data_version": "v1", "state": "forming"},
                    {"pattern_id": "b", "data_version": "v1", "state": "confirmed"},
                ])
                self.assertEqual(get_pattern("a")["state"], "forming")
                with connect() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 2)
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_data_versions_are_kept_as_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                base = {"pattern_id":"abc", "data_version":"cutoff-a", "state":"confirmed"}
                save_pattern(base)
                newer = dict(base, data_version="cutoff-b", state="invalidated")
                save_pattern(newer)
                with connect() as db:
                    rows=db.execute("SELECT data_version,payload FROM pattern_snapshots ORDER BY data_version").fetchall()
                self.assertEqual([r[0] for r in rows], ["cutoff-a","cutoff-b"])
                self.assertEqual(json.loads(rows[0][1])["state"],"confirmed")
                self.assertEqual(get_pattern("abc")["state"],"invalidated")
            finally:
                if previous is None: os.environ.pop("CAISEN_DB",None)
                else: os.environ["CAISEN_DB"] = previous


    def test_scan_runs_and_normalized_results_persist_and_filter(self):
        from caisen.store import (
                    create_scan_run, interrupt_active_scan_runs, list_scan_runs, query_scan_results, save_scan_result,
            update_scan_run,
        )
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                create_scan_run("scan-1", {"kinds": ["stock"], "date_to": "2026-10-01"},
                                total=2, rule_version="rules-v1", app_version="app-v1")
                save_scan_result("scan-1", 0, {
                    "code": "000001", "kind": "stock", "name": "样本公司", "category": "银行",
                    "action": "WAIT", "patterns": [{"pattern_id": "main-1", "pattern_type": "w_bottom",
                        "timeframe": "1d", "formed_at": "2026-09-01", "state": "forming",
                        "is_main_structure": True, "selected_main": True}],
                    "latest_date": "2026-10-01",
                })
                save_scan_result("scan-1", 1, {
                    "code": "000000", "kind": "stock", "name": "次序公司", "category": "银行",
                    "action": "CONSIDER_BUY", "patterns": [], "latest_date": "2026-10-01",
                })
                update_scan_run("scan-1", status="partial_failed", progress=1, scanned=2,
                                inventory={"failed_count": 1})

                runs, total = list_scan_runs(limit=50, offset=0)
                results, count = query_scan_results(
                    "scan-1", search="样本", category="银行", action="WAIT",
                    pattern_types=["w_bottom"], timeframes=["1d"], sort_by="code",
                    sort_order="asc", limit=50, offset=0,
                )
                ranked, ranked_count = query_scan_results("scan-1")
                etfs, etf_count = query_scan_results("scan-1", kind="etf")

                assert total == 1
                assert runs[0]["status"] == "partial_failed"
                assert runs[0]["params"] == {"kinds": ["stock"], "date_to": "2026-10-01"}
                assert count == 1
                assert results[0]["name"] == "样本公司"
                assert results[0]["patterns"][0]["timeframe"] == "1d"
                assert "mini_bars" not in results[0] and "main_structure_events" not in results[0]
                assert ranked_count == 2 and [row["code"] for row in ranked] == ["000000", "000001"]
                assert etf_count == 0 and etfs == []
                create_scan_run("scan-2", {"kinds": ["etf"]}, total=1)
                update_scan_run("scan-2", status="running", progress=.5, scanned=0)
                with connect() as db:
                    db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                               ("scan-2", "running", json.dumps({"status": "running", "progress": .5})))
                assert interrupt_active_scan_runs() == 1
                interrupted, _ = list_scan_runs(limit=50, offset=0)
                assert next(run for run in interrupted if run["run_id"] == "scan-2")["status"] == "interrupted"
                with connect() as db:
                    old_job = json.loads(db.execute("SELECT payload FROM jobs WHERE job_id='scan-2'").fetchone()[0])
                assert old_job["status"] == "interrupted" and "params" not in old_job
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_scan_rows_are_immutable_and_assessments_are_saved_separately(self):
        from caisen.store import (connect, create_scan_run, query_scan_results,
                                  save_scan_assessment, save_scan_result)
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                create_scan_run("immutable", {"kinds": ["stock"]}, total=2)
                save_scan_result("immutable", 0, {"code": "000001", "kind": "stock", "action": "WAIT",
                                  "rank": {"action_priority": 3, "confirmed_timeframes": 0,
                                           "reward_risk": 0, "risk_pct": 1, "confirmation_ordinal": 0}})
                save_scan_result("immutable", 1, {"code": "000002", "kind": "stock", "action": "BUY",
                                  "rank": {"action_priority": 0, "confirmed_timeframes": 1,
                                           "reward_risk": 2, "risk_pct": .1, "confirmation_ordinal": 1}})
                save_scan_result("immutable", 0, {"code": "replacement", "kind": "stock"})
                save_scan_assessment("immutable", 1, {"status": "FOCUS_NECKLINE"})
                save_scan_assessment("immutable", 0, {"status": "BUY"})

                rows, total = query_scan_results("immutable")
                assert total == 2
                assert [row["code"] for row in rows] == ["000001", "000002"]
                assert rows[0]["assessment"] == {"status": "BUY"}
                assert rows[0]["action"] == "CONSIDER_BUY"
                buys, count = query_scan_results("immutable", action="CONSIDER_BUY")
                assert count == 1 and buys[0]["action"] == "CONSIDER_BUY"
                focused, count = query_scan_results("immutable", action="FOCUS_NECKLINE")
                assert count == 1 and focused[0]["assessment"]["status"] == "FOCUS_NECKLINE"
                with connect() as db:
                    original = db.execute("SELECT payload_json FROM scan_results WHERE run_id='immutable' AND ordinal=0").fetchone()[0]
                    assert '"code": "000001"' in original
                    assert db.execute("SELECT COUNT(*) FROM scan_assessments").fetchone()[0] == 2
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_pattern_filter_uses_only_latest_visible_main_per_timeframe(self):
        from caisen.store import create_scan_run, query_scan_results, save_scan_result
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                create_scan_run("latest-main", {}, total=1)
                patterns = [
                    {"pattern_id": "old-d", "pattern_type": "w_bottom", "timeframe": "1d",
                     "formed_at": "2026-09-01", "state": "forming", "is_main_structure": True, "selected_main": True},
                    {"pattern_id": "new-d", "pattern_type": "bull_flag", "timeframe": "1d",
                     "formed_at": "2026-10-01", "state": "forming", "is_main_structure": True, "selected_main": True},
                    {"pattern_id": "new-w", "pattern_type": "w_bottom", "timeframe": "1w",
                     "formed_at": "2026-08-01", "state": "forming", "is_main_structure": True, "selected_main": True},
                ]
                save_scan_result("latest-main", 0, {"code": "000001", "kind": "stock", "action": "WAIT",
                                                       "patterns": patterns})
                daily_old, count = query_scan_results("latest-main", pattern_types=["w_bottom"], timeframes=["1d"])
                weekly, weekly_count = query_scan_results("latest-main", pattern_types=["w_bottom"], timeframes=["1w"])
                all_rows, _ = query_scan_results("latest-main")
                assert daily_old == [] and count == 0
                assert weekly_count == 1
                assert {(row["timeframe"], row["pattern_type"]) for row in all_rows[0]["patterns"]} == {
                    ("1d", "bull_flag"), ("1w", "w_bottom")}
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_scan_assessment_schema_migrates_to_versioned_key_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                import sqlite3
                raw = sqlite3.connect(os.environ["CAISEN_DB"])
                raw.executescript("""
                    CREATE TABLE scan_runs(run_id TEXT PRIMARY KEY,status TEXT,params_json TEXT,
                        progress REAL DEFAULT 0,scanned INTEGER DEFAULT 0,total INTEGER DEFAULT 0,
                        inventory_json TEXT DEFAULT '{}',rule_version TEXT,app_version TEXT,error TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
                    CREATE TABLE scan_results(run_id TEXT,ordinal INTEGER,code TEXT,kind TEXT,name TEXT,
                        category TEXT,action TEXT,patterns_json TEXT,payload_json TEXT,
                        PRIMARY KEY(run_id,ordinal));
                    CREATE TABLE scan_assessments(run_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,PRIMARY KEY(run_id,ordinal));
                    INSERT INTO scan_runs(run_id,status,params_json) VALUES('old-schema','completed','{}');
                    INSERT INTO scan_results(run_id,ordinal,code,kind,name,category,action,patterns_json,payload_json)
                        VALUES('old-schema',0,'000001','stock','','unclassified','WAIT','[]','{}');
                    INSERT INTO scan_assessments(run_id,ordinal,payload_json) VALUES('old-schema',0,'{"status":"NONE"}');
                """)
                raw.commit()
                raw.close()

                with connect() as db:
                    info = db.execute("PRAGMA table_info(scan_assessments)").fetchall()
                    key = [row[1] for row in sorted((row for row in info if row[5]), key=lambda row: row[5])]
                    old = db.execute("SELECT assessment_version,payload_json FROM scan_assessments WHERE run_id='old-schema'").fetchone()
                assert key == ["run_id", "ordinal", "assessment_version"]
                assert old[0] == "scan-assessment-v1" and json.loads(old[1]) == {"status": "NONE"}
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_legacy_scan_jobs_import_idempotently_and_keep_missing_metadata_explicit(self):
        from caisen.store import (get_scan_assessment_progress, import_legacy_scan_jobs,
                                  interrupt_active_scan_runs, list_scan_runs, query_scan_results,
                                  start_saved_scan_assessment)
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                failed_results = [
                    {"code": "000001", "kind": "stock", "name": "样本一", "state": "data_error",
                     "action": "INSUFFICIENT_DATA", "reason_codes": ["file_analysis_failed"],
                     "error": "fixture failure"},
                    {"code": "000002", "kind": "etf", "name": "样本二", "state": "data_error",
                     "action": "INSUFFICIENT_DATA", "reason_codes": ["file_analysis_failed"],
                     "error": "fixture failure"},
                ]
                success_results = [{"code": "000003", "kind": "stock", "name": "成功样本",
                                    "action": "WAIT", "patterns": []}]
                with connect() as db:
                    db.execute("INSERT INTO jobs(job_id,status,payload,created_at) VALUES(?,?,?,?)",
                               ("legacy-all-failed", "completed", json.dumps({
                                   "status": "completed", "progress": 1, "scanned": 2, "total": 2,
                                   "results": failed_results,
                                   "inventory": {"source": "local_csv", "failed_count": 2},
                               }), "2026-10-20 09:00:00"))
                    db.execute("INSERT INTO jobs(job_id,status,payload,created_at) VALUES(?,?,?,?)",
                               ("legacy-missing-failure-count", "completed", json.dumps({
                                   "status": "completed", "progress": 1, "scanned": 1, "total": 2,
                                   "results": success_results, "inventory": {"source": "local_csv"},
                               }), "2026-10-21 09:00:00"))
                    db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                               ("legacy-refresh", "completed", json.dumps({"job_type": "data_refresh",
                                   "completed": 1, "results": []})))
                    db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                               ("legacy-fixed", "completed", json.dumps({"job_type": "fixed_backtest",
                                   "results": []})))
                    db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                               ("unrelated-job", "completed", json.dumps({"results": success_results})))
                    db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                               ("legacy-running", "running", json.dumps({
                                   "status": "running", "progress": .5, "scanned": 1, "total": 2,
                                   "results": success_results, "inventory": {"source": "local_csv"},
                               })))

                assert import_legacy_scan_jobs() == 3
                assert import_legacy_scan_jobs() == 0
                with connect() as db:
                    db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                               ("after-migration", "completed", json.dumps({
                                   "status": "completed", "progress": 1, "scanned": 1, "total": 1,
                                   "results": success_results, "inventory": {"source": "local_csv"},
                               })))
                assert import_legacy_scan_jobs() == 0
                runs, total = list_scan_runs(limit=50, offset=0)
                assert total == 3
                failed = next(run for run in runs if run["run_id"] == "legacy-all-failed")
                assert failed["status"] == "failed"
                assert failed["params"]["_not_recorded"] == ["scan parameters"]
                assert failed["rule_version"] is None and failed["app_version"] is None
                assert failed["inventory"]["failed_count"] == 2
                assert failed["inventory"]["failure_count_known"] is True
                start_saved_scan_assessment("legacy-all-failed")
                deadline = time.monotonic() + 2
                while get_scan_assessment_progress("legacy-all-failed")["status"] != "completed" and time.monotonic() < deadline:
                    time.sleep(.01)
                old_rows, old_count = query_scan_results("legacy-all-failed")
                assert old_count == 2 and all(row["assessment"]["status"] == "INSUFFICIENT_EVIDENCE" for row in old_rows)
                assert get_scan_assessment_progress("legacy-all-failed")["processed"] == 2
                with connect() as db:
                    from caisen.assessment import ASSESSMENT_VERSION
                    assert {row[0] for row in db.execute("SELECT assessment_version FROM scan_assessments WHERE run_id='legacy-all-failed'")} == {ASSESSMENT_VERSION}
                with connect() as db:
                    assert db.execute("SELECT json_extract(payload_json,'$.state') FROM scan_results WHERE run_id='legacy-all-failed' LIMIT 1").fetchone()[0] == "data_error"
                unknown = next(run for run in runs if run["run_id"] == "legacy-missing-failure-count")
                assert unknown["inventory"]["failure_count_known"] is False
                assert interrupt_active_scan_runs() == 1
                interrupted, _ = list_scan_runs(limit=50, offset=0)
                assert next(run for run in interrupted if run["run_id"] == "legacy-running")["status"] == "interrupted"
                with connect() as db:
                    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 7
                    assert db.execute("SELECT status FROM jobs WHERE job_id='legacy-all-failed'").fetchone()[0] == "completed"
                    imported_ids = {row[0] for row in db.execute("SELECT run_id FROM scan_runs")}
                assert imported_ids == {"legacy-all-failed", "legacy-missing-failure-count", "legacy-running"}
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous


    def test_pattern_instance_events_are_persisted_idempotently_by_run_and_sequence(self):
        from caisen.store import get_pattern_instance_events, save_pattern_instance_events
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.sqlite3")
            events = [
                {"instance_id": "p1", "event_id": "e1", "event_seq": 1,
                 "event_date": "2026-01-01", "state": "forming", "entry_eligible": False},
                {"instance_id": "p1", "event_id": "e2", "event_seq": 2,
                 "event_date": "2026-01-02", "state": "confirmed", "entry_eligible": True},
            ]
            self.assertEqual(save_pattern_instance_events("run-a", "000001", "stock", events,
                                                           path=path), 2)
            self.assertEqual(save_pattern_instance_events("run-a", "000001", "stock", events,
                                                           path=path), 0)
            self.assertEqual([item["event_id"] for item in get_pattern_instance_events("run-a", path=path)],
                             ["e1", "e2"])
            self.assertEqual(get_pattern_instance_events("run-b", path=path), [])

    def test_pattern_instance_event_sequence_cannot_be_rewritten(self):
        from caisen.store import save_pattern_instance_events
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.sqlite3")
            original = {"instance_id": "p1", "event_id": "e1", "event_seq": 1,
                        "event_date": "2026-01-01", "state": "forming", "entry_eligible": False}
            changed = dict(original, event_id="different", state="confirmed", entry_eligible=True)
            save_pattern_instance_events("run-a", "000001", "stock", [original], path=path)
            with self.assertRaisesRegex(ValueError, "immutable"):
                save_pattern_instance_events("run-a", "000001", "stock", [changed], path=path)

    def test_pattern_instance_metadata_is_persisted_separately_and_idempotently(self):
        from caisen.store import get_pattern_instance_metadata, save_pattern_instance_metadata
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metadata.sqlite3")
            row = {"instance_id": "p1", "event_id": "e1", "event_seq": 1,
                   "structure_key": "structure-a", "boundary_version_id": "boundary-b",
                   "paired_at": "2026-01-02", "state_at": "2026-01-03",
                   "selection_reason": "paired pressure touches",
                   "recognition_schema_version": "caisen-pattern-instance-v2",
                   "state_schema_version": "caisen-pattern-state-event-v2",
                   "metadata_source": "derived_from_immutable_event_snapshot"}
            self.assertEqual(save_pattern_instance_metadata("run-a", "000001", "stock", [row], path=path), 1)
            self.assertEqual(save_pattern_instance_metadata("run-a", "000001", "stock", [row], path=path), 0)
            self.assertEqual(get_pattern_instance_metadata("run-a", path=path), [row])
            self.assertEqual(get_pattern_instance_metadata("run-b", path=path), [])

    def test_pattern_instance_metadata_sequence_cannot_be_rewritten(self):
        from caisen.store import save_pattern_instance_metadata
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metadata.sqlite3")
            original = {"instance_id": "p1", "event_id": "e1", "event_seq": 1,
                        "structure_key": "structure-a", "boundary_version_id": "boundary-b"}
            changed = dict(original, boundary_version_id="boundary-changed")
            save_pattern_instance_metadata("run-a", "000001", "stock", [original], path=path)
            with self.assertRaisesRegex(ValueError, "immutable"):
                save_pattern_instance_metadata("run-a", "000001", "stock", [changed], path=path)


if __name__ == "__main__": unittest.main()
