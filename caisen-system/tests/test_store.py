import json
import os
import sqlite3
import tempfile
import unittest

from caisen import store
from caisen.daily_incremental import DailyRecognitionSession
from caisen.schemas import Pattern, Pivot
from caisen.store import connect, get_pattern, save_pattern, save_patterns


class StoreTests(unittest.TestCase):
    @staticmethod
    def _lifecycle_pattern(data_version="bars-v1", event_kind="paired"):
        return {
            "pattern_id": "stock:W:one", "code": "000008", "kind": "stock",
            "timeframe": "1d", "pattern_type": "w_bottom", "data_version": data_version,
            "rule_version": "pattern-v1", "source_rule_version": "source-v1", "state": "forming",
            "evidence": [{"code": "joint_lifecycle", "value": {
                "schema_version": "joint-pattern-instance-v2", "instance_id": "instance-1",
                "boundary_version_id": "boundary-1",
                "recognition_context": {"experiment_id": "g1-v1"}, "events": [
                    {"instance_id": "instance-1", "boundary_version_id": "boundary-1",
                     "sequence": 1, "date": "2024-01-10", "kind": event_kind,
                     "reason": "fixture"},
                    {"instance_id": "instance-1", "boundary_version_id": "boundary-1",
                     "sequence": 2, "date": "2024-05-10", "kind": "invalidated",
                     "reason": "close_below_instance_boundary"},
                ]}}],
        }

    def test_lifecycle_events_are_idempotent_and_isolated_by_data_version(self):
        self.assertTrue(callable(getattr(store, "get_lifecycle_events", None)))
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                first = self._lifecycle_pattern("bars-v1")
                second = self._lifecycle_pattern("bars-v2")
                save_pattern(first)
                save_pattern(first)
                save_pattern(second)
                s1 = self._lifecycle_pattern("bars-v1")
                s1["evidence"][0]["value"]["recognition_context"] = {"experiment_id": "s1-v1"}
                save_pattern(s1)

                v1 = store.get_lifecycle_events(instance_id="instance-1", data_version="bars-v1")
                v2 = store.get_lifecycle_events(instance_id="instance-1", data_version="bars-v2")

                g1_batch = next(item["analysis_batch_id"] for item in v1
                                if item["recognition_context"] == {"experiment_id": "g1-v1"})
                s1_batch = next(item["analysis_batch_id"] for item in v1
                                if item["recognition_context"] == {"experiment_id": "s1-v1"})
                g1_v1 = store.get_lifecycle_events(analysis_batch_id=g1_batch)
                self.assertEqual([item["sequence"] for item in g1_v1], [1, 2])
                self.assertEqual([item["kind"] for item in g1_v1], ["paired", "invalidated"])
                self.assertNotEqual(g1_batch, s1_batch)
                self.assertEqual(g1_v1[0]["instance_id"], v1[0]["instance_id"])
                self.assertEqual(len(v2), 2)
                self.assertNotEqual(g1_batch, v2[0]["analysis_batch_id"])
                self.assertEqual(g1_v1[1]["payload"]["reason"], "close_below_instance_boundary")
                with connect() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 2)
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_lifecycle_events").fetchone()[0], 6)
                    snapshot = db.execute(
                        "SELECT payload FROM pattern_snapshots WHERE pattern_id=? AND data_version=?",
                        ("stock:W:one", "bars-v1")).fetchone()[0]
                    saved_snapshot = json.loads(snapshot)
                    saved_context = next(item["value"] for item in saved_snapshot["evidence"]
                                         if item["code"] == "joint_lifecycle")["recognition_context"]
                    self.assertEqual(saved_context, {"experiment_id": "g1-v1"})
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_lifecycle_event_conflict_rejects_rewrite_atomically(self):
        self.assertTrue(hasattr(store, "LifecycleEventConflict"))
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                original = self._lifecycle_pattern()
                save_pattern(original, recognition_context={"experiment_id": "g1-v1"})
                changed = self._lifecycle_pattern(event_kind="weak_crossing")
                changed["state"] = "confirmed"

                with self.assertRaises(store.LifecycleEventConflict):
                    save_pattern(changed, recognition_context={"experiment_id": "g1-v1"})

                self.assertEqual(get_pattern("stock:W:one").get("state"), "forming")
                saved = store.get_lifecycle_events(instance_id="instance-1", data_version="bars-v1")
                self.assertEqual([item["kind"] for item in saved], ["paired", "invalidated"])
                with connect() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 1)
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_legacy_or_unknown_patterns_do_not_create_invented_lifecycle_events(self):
        self.assertTrue(callable(getattr(store, "get_lifecycle_events", None)))
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                save_pattern({"pattern_id": "legacy", "data_version": "unknown",
                              "evidence": [{"code": "joint_lifecycle", "value": {
                                  "schema_version": "joint-pattern-instance-v2",
                                  "instance_id": "missing-schema", "boundary_version_id": "boundary",
                                  "events": [{"kind": "confirmed"}]}}]})
                save_pattern({"pattern_id": "legacy-schema", "data_version": "bars-v1",
                              "evidence": [{"code": "joint_lifecycle", "value": {
                                  "instance_id": "old-instance", "boundary_version_id": "old-boundary",
                                  "events": [{"kind": "confirmed"}]}}]})
                save_pattern({"pattern_id": "no-lifecycle", "data_version": "bars-v1"})
                self.assertEqual(store.get_lifecycle_events(), [])
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_lifecycle_rows_must_already_be_in_event_sequence_order(self):
        self.assertTrue(callable(getattr(store, "get_lifecycle_events", None)))
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                pattern = self._lifecycle_pattern()
                lifecycle = pattern["evidence"][0]["value"]
                lifecycle["events"].reverse()
                with self.assertRaises(ValueError):
                    save_pattern(pattern, recognition_context={"experiment_id": "g1-v1"})
                self.assertEqual(store.get_lifecycle_events(), [])
                with connect() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 0)
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_batch_conflict_is_rejected_before_any_pattern_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                first = self._lifecycle_pattern("bars-v1")
                second = self._lifecycle_pattern("bars-v2")
                second["pattern_id"] = "stock:W:two"
                with self.assertRaises(store.LifecycleEventConflict):
                    save_patterns([first, second], analysis_batch_id="same-batch")
                with connect() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 0)
                    self.assertEqual(db.execute("SELECT count(*) FROM patterns").fetchone()[0], 0)
                    self.assertEqual(db.execute("SELECT count(*) FROM pattern_lifecycle_events").fetchone()[0], 0)
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_external_plain_sqlite_connection_supports_atomic_lifecycle_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.sqlite3")
            with connect(path):
                pass
            db = sqlite3.connect(path)
            try:
                pattern = self._lifecycle_pattern()
                save_pattern(pattern, db=db, analysis_batch_id="external-batch")
                save_pattern(pattern, db=db, analysis_batch_id="external-batch")
                history = store.get_lifecycle_events(analysis_batch_id="external-batch", db=db)
                self.assertEqual([item["sequence"] for item in history], [1, 2])

                changed = self._lifecycle_pattern(event_kind="weak_crossing")
                with self.assertRaises(store.LifecycleEventConflict):
                    save_pattern(changed, db=db, analysis_batch_id="external-batch")
                self.assertEqual([item["kind"] for item in
                                  store.get_lifecycle_events(analysis_batch_id="external-batch", db=db)],
                                 ["paired", "invalidated"])
                self.assertEqual(db.execute("SELECT count(*) FROM pattern_snapshots").fetchone()[0], 1)
            finally:
                db.close()

    def test_same_instance_source_pattern_ids_are_append_only_aliases(self):
        self.assertTrue(callable(getattr(store, "get_lifecycle_event_aliases", None)))
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                first = self._lifecycle_pattern()
                second = self._lifecycle_pattern()
                second["pattern_id"] = "stock:W:alternate-source-id"
                save_patterns([first, second], analysis_batch_id="same-instance-batch")

                events = store.get_lifecycle_events(analysis_batch_id="same-instance-batch")
                aliases = store.get_lifecycle_event_aliases(
                    analysis_batch_id="same-instance-batch", instance_id="instance-1")
                self.assertEqual([event["pattern_id"] for event in events],
                                 ["stock:W:one", "stock:W:one"])
                self.assertEqual({row["pattern_id"] for row in aliases},
                                 {"stock:W:one", "stock:W:alternate-source-id"})
                self.assertEqual(len(aliases), 4)

                changed = self._lifecycle_pattern()
                changed["pattern_id"] = "stock:W:third-alias"
                changed["evidence"][0]["value"]["events"][1]["reason"] = "changed_history"
                with self.assertRaises(store.LifecycleEventConflict):
                    save_pattern(changed, analysis_batch_id="same-instance-batch")
                aliases_after_conflict = store.get_lifecycle_event_aliases(
                    analysis_batch_id="same-instance-batch", instance_id="instance-1")
                self.assertEqual({row["pattern_id"] for row in aliases_after_conflict},
                                 {"stock:W:one", "stock:W:alternate-source-id"})
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

    def test_session_prefix_future_prefix_and_terminal_lifecycle_are_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("CAISEN_DB")
            os.environ["CAISEN_DB"] = os.path.join(tmp, "state.sqlite3")
            try:
                session = DailyRecognitionSession(
                    "FIXTURE", "stock", tick_size=0.01,
                    provenance={"data_version": "prefix-freeze-1"})
                session.append_bar({"date": "2025-01-01", "open": 10.6, "high": 10.8,
                                    "low": 10.4, "close": 10.6, "volume": 100})
                session.append_bar({"date": "2025-01-02", "open": 10.5, "high": 10.6,
                                    "low": 10.2, "close": 10.4, "volume": 100})
                points = (Pivot(0, "2025-01-01", 10.0, "low", 0, "2025-01-01", "2ATR|5%"),
                          Pivot(1, "2025-01-02", 10.1, "low", 1, "2025-01-02", "2ATR|5%"))
                pattern = Pattern(
                    "session-instance", "FIXTURE", "stock", "1d", "w_bottom", "bullish", "forming",
                    "2025-01-01", None,
                    key_points={"pivot_1_date": "2025-01-01", "pivot_1_kind": "low",
                                "pivot_1": 10.0, "pivot_2_date": "2025-01-02",
                                "pivot_2_kind": "low", "pivot_2": 10.1},
                    neckline={"slope": 0.0, "intercept": 10.5}, invalidation=9.0,
                    evidence=[{"code": "main_structure", "value": {
                        "role": "main", "support_lower": 9.0,
                        "structure_known_at": "2025-01-02", "structure_end_date": "2025-01-02"}},
                              {"code": "neckline_selection", "value": {
                                  "line_frozen_at": "2025-01-02",
                                  "effective_start_date": "2025-01-01"}},
                              {"code": "recognition_context", "value": dict(session.recognition_context)}],
                    data_version="prefix-freeze-1", last_observed_date="2025-01-02",
                    rule_version="engine-v1", source_rule_version="source-v1")
                candidate_key = ("w_bottom", "2atr_5pct", ((0, "low"), (1, "low")))
                session.active[candidate_key] = {
                    "family": "w_bottom", "scale": "2atr_5pct", "sequence": points,
                    "pattern": pattern, "structure_known_at": 1}

                from caisen.structure_boundary import attach_joint_lifecycle
                attach_joint_lifecycle(pattern, discovered_at="2025-01-02",
                                       discovery_kind="session_observation")
                prefix = pattern.to_dict()
                save_pattern(prefix, analysis_batch_id="prefix-run-1")
                save_pattern(prefix, analysis_batch_id="prefix-run-1")
                original_history = store.get_lifecycle_events(analysis_batch_id="prefix-run-1")
                self.assertEqual([event["kind"] for event in original_history], ["paired"])

                session.append_bar({"date": "2025-01-03", "open": 8.9, "high": 9.0,
                                    "low": 8.7, "close": 8.8, "volume": 120})
                terminal = session.retired_patterns[next(iter(session.retired_patterns))]
                terminal_payload = terminal.to_dict()
                save_pattern(terminal_payload, analysis_batch_id="prefix-run-1")

                final_history = store.get_lifecycle_events(analysis_batch_id="prefix-run-1")
                self.assertEqual([event["sequence"] for event in final_history], [1, 2])
                self.assertEqual([event["kind"] for event in final_history],
                                 ["paired", "invalidated"])
                self.assertEqual(final_history[0]["payload"], original_history[0]["payload"])
                self.assertEqual(final_history[1]["data_version"], "prefix-freeze-1")
            finally:
                if previous is None: os.environ.pop("CAISEN_DB", None)
                else: os.environ["CAISEN_DB"] = previous

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
