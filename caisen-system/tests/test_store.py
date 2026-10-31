import json
import os
import tempfile
import unittest

from caisen.store import connect, get_pattern, save_pattern, save_patterns


class StoreTests(unittest.TestCase):
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
                    "action": "WAIT", "patterns": [{"pattern_type": "w_bottom", "timeframe": "1d"}],
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
                assert ranked_count == 2 and [row["code"] for row in ranked] == ["000001", "000000"]
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

    def test_legacy_scan_jobs_import_idempotently_and_keep_missing_metadata_explicit(self):
        from caisen.store import (import_legacy_scan_jobs, interrupt_active_scan_runs,
                                  list_scan_runs, query_scan_results)
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
                old_rows, old_count = query_scan_results("legacy-all-failed")
                assert old_count == 2 and all(row["state"] == "data_error" for row in old_rows)
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


if __name__ == "__main__": unittest.main()
