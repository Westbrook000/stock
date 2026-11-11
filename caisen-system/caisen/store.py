import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from .assessment import ASSESSMENT_VERSION

SCAN_ASSESSMENT_VERSION = ASSESSMENT_VERSION


def database_path():
    return os.environ.get("CAISEN_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "state", "caisen.sqlite3"))


@contextmanager
def connect(path=None):
    path = path or database_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    db = sqlite3.connect(path, timeout=20)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS patterns(pattern_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS pattern_snapshots(pattern_id TEXT NOT NULL, data_version TEXT NOT NULL, payload TEXT NOT NULL, observed_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(pattern_id,data_version));
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS backtests(backtest_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS scan_runs(
            run_id TEXT PRIMARY KEY, status TEXT NOT NULL, params_json TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0, scanned INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0, inventory_json TEXT NOT NULL DEFAULT '{}',
            rule_version TEXT, app_version TEXT, error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS scan_results(
            run_id TEXT NOT NULL REFERENCES scan_runs(run_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, code TEXT NOT NULL, kind TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '', category TEXT NOT NULL DEFAULT 'unclassified',
            action TEXT NOT NULL DEFAULT 'INSUFFICIENT_DATA', patterns_json TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL, PRIMARY KEY(run_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS scan_assessments(
            run_id TEXT NOT NULL REFERENCES scan_runs(run_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, assessment_version TEXT NOT NULL DEFAULT 'scan-assessment-v2', payload_json TEXT NOT NULL,
            PRIMARY KEY(run_id, ordinal, assessment_version)
        );
        CREATE TABLE IF NOT EXISTS scan_result_evidence(
            run_id TEXT NOT NULL REFERENCES scan_runs(run_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, pattern_id TEXT NOT NULL, payload_json TEXT NOT NULL,
            PRIMARY KEY(run_id, ordinal, pattern_id)
        );
        CREATE TABLE IF NOT EXISTS pattern_instance_events(
            run_id TEXT NOT NULL, code TEXT NOT NULL, kind TEXT NOT NULL,
            instance_id TEXT NOT NULL, event_seq INTEGER NOT NULL,
            event_id TEXT NOT NULL, event_date TEXT NOT NULL, state TEXT NOT NULL,
            entry_eligible INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id,kind,code,instance_id,event_seq),
            UNIQUE(run_id,event_id)
        );
        CREATE TABLE IF NOT EXISTS pattern_instance_metadata(
            run_id TEXT NOT NULL, code TEXT NOT NULL, kind TEXT NOT NULL,
            instance_id TEXT NOT NULL, event_seq INTEGER NOT NULL, event_id TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_id,kind,code,instance_id,event_seq),
            UNIQUE(run_id,event_id)
        );
        CREATE TABLE IF NOT EXISTS scan_assessment_jobs(
            run_id TEXT PRIMARY KEY REFERENCES scan_runs(run_id) ON DELETE CASCADE,
            status TEXT NOT NULL, processed INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0, error TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS scan_results_run_code ON scan_results(run_id, code);
        CREATE INDEX IF NOT EXISTS scan_results_run_action ON scan_results(run_id, action);
        CREATE INDEX IF NOT EXISTS pattern_instance_events_run_instance
            ON pattern_instance_events(run_id,kind,code,instance_id,event_seq);
        CREATE INDEX IF NOT EXISTS pattern_instance_metadata_run_instance
            ON pattern_instance_metadata(run_id,kind,code,instance_id,event_seq);
        CREATE TABLE IF NOT EXISTS migration_runs(
            migration_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)
        assessment_info = db.execute("PRAGMA table_info(scan_assessments)").fetchall()
        assessment_columns = {row[1] for row in assessment_info}
        assessment_pk = [row[1] for row in sorted((row for row in assessment_info if row[5]), key=lambda row: row[5])]
        if "assessment_version" not in assessment_columns or assessment_pk != ["run_id", "ordinal", "assessment_version"]:
            db.execute("ALTER TABLE scan_assessments RENAME TO scan_assessments_old")
            db.execute("""CREATE TABLE scan_assessments(
                run_id TEXT NOT NULL REFERENCES scan_runs(run_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, assessment_version TEXT NOT NULL DEFAULT 'scan-assessment-v2',
                payload_json TEXT NOT NULL, PRIMARY KEY(run_id,ordinal,assessment_version))""")
            version_expr = "assessment_version" if "assessment_version" in assessment_columns else "'scan-assessment-v1'"
            db.execute("INSERT INTO scan_assessments(run_id,ordinal,assessment_version,payload_json) "
                       "SELECT run_id,ordinal," + version_expr + ",payload_json FROM scan_assessments_old")
            db.execute("DROP TABLE scan_assessments_old")
        legacy=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name IN ('reviews','refresh_job_items')").fetchone()
        if legacy or db.execute("PRAGMA user_version").fetchone()[0] < 1:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("DROP TABLE IF EXISTS reviews")
            db.execute("DROP TABLE IF EXISTS refresh_job_items")
            db.execute("DELETE FROM jobs WHERE json_valid(payload) AND json_extract(payload,'$.job_type')='data_refresh'")
            db.execute("PRAGMA user_version=1")
        yield db
        db.commit()
    finally:
        db.close()


def save_pattern(pattern, db=None):
    return save_patterns([pattern], db=db)


def save_patterns(patterns, db=None):
    payloads = [pattern.to_dict() if hasattr(pattern, "to_dict") else pattern for pattern in patterns]
    if not payloads:
        return
    with connect() if db is None else _yield(db) as conn:
        snapshots = [(payload["pattern_id"], payload.get("data_version", "unknown"),
                      json.dumps(payload, ensure_ascii=False)) for payload in payloads]
        current = [(payload["pattern_id"], json.dumps(payload, ensure_ascii=False)) for payload in payloads]
        conn.executemany("INSERT OR IGNORE INTO pattern_snapshots(pattern_id,data_version,payload) VALUES(?,?,?)", snapshots)
        conn.executemany("INSERT OR REPLACE INTO patterns(pattern_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)", current)


def get_pattern(pattern_id, db=None):
    with connect() if db is None else _yield(db) as conn:
        row = conn.execute("SELECT payload FROM patterns WHERE pattern_id=?", (pattern_id,)).fetchone()
        return json.loads(row[0]) if row else None


def save_pattern_instance_events(run_id, code, kind, events, db=None, path=None):
    """Persist immutable lifecycle events; exact replays are idempotent."""
    if db is not None and path is not None:
        raise ValueError("pass either db or path, not both")
    rows = []
    for event in events or []:
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        required = ("instance_id", "event_id", "event_seq", "event_date", "state")
        missing = [field for field in required if payload.get(field) is None]
        if missing:
            raise ValueError("pattern instance event missing fields: " + ", ".join(missing))
        rows.append((str(run_id), str(code), str(kind), str(payload["instance_id"]),
                     int(payload["event_seq"]), str(payload["event_id"]),
                     str(payload["event_date"]), str(payload["state"]),
                     int(bool(payload.get("entry_eligible"))),
                     json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"))))
    if not rows:
        return 0
    inserted = 0
    context = connect(path=path) if db is None else _yield(db)
    with context as conn:
        for row in rows:
            prior = conn.execute(
                "SELECT event_id,payload_json FROM pattern_instance_events "
                "WHERE run_id=? AND kind=? AND code=? AND instance_id=? AND event_seq=?",
                row[:5]).fetchone()
            if prior is not None:
                if prior["event_id"] != row[5] or prior["payload_json"] != row[9]:
                    raise ValueError("pattern instance event is immutable for its run and sequence")
                continue
            try:
                result = conn.execute(
                    "INSERT INTO pattern_instance_events "
                    "(run_id,code,kind,instance_id,event_seq,event_id,event_date,state,entry_eligible,payload_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)", row)
            except sqlite3.IntegrityError as exc:
                existing = conn.execute(
                    "SELECT 1 FROM pattern_instance_events WHERE run_id=? AND event_id=?",
                    (str(run_id), str(row[5]))).fetchone()
                if existing:
                    continue
                raise ValueError("pattern instance event sequence is immutable") from exc
            inserted += max(int(result.rowcount), 0)
    return inserted


def get_pattern_instance_events(run_id, code=None, kind=None, instance_id=None, db=None, path=None):
    """Read a run's ordered immutable lifecycle events."""
    if db is not None and path is not None:
        raise ValueError("pass either db or path, not both")
    clauses, params = ["run_id=?"], [str(run_id)]
    for field, value in (("code", code), ("kind", kind), ("instance_id", instance_id)):
        if value is not None:
            clauses.append(field + "=?")
            params.append(str(value))
    context = connect(path=path) if db is None else _yield(db)
    with context as conn:
        rows = conn.execute("SELECT payload_json FROM pattern_instance_events WHERE "
                            + " AND ".join(clauses)
                            + " ORDER BY kind,code,instance_id,event_seq", params).fetchall()
        return [json.loads(row[0]) for row in rows]


def save_pattern_instance_metadata(run_id, code, kind, metadata_rows, db=None, path=None):
    """Persist derived contract metadata beside, but never inside, raw events."""
    if db is not None and path is not None:
        raise ValueError("pass either db or path, not both")
    context = connect(path=path) if db is None else _yield(db)
    inserted = 0
    with context as conn:
        for metadata in metadata_rows or []:
            row = dict(metadata)
            required = ("instance_id", "event_id", "event_seq")
            missing = [field for field in required if row.get(field) is None]
            if missing:
                raise ValueError("pattern instance metadata missing fields: " + ", ".join(missing))
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"))
            key = (str(run_id), str(kind), str(code), str(row["instance_id"]), int(row["event_seq"]))
            prior = conn.execute(
                "SELECT event_id,metadata_json FROM pattern_instance_metadata "
                "WHERE run_id=? AND kind=? AND code=? AND instance_id=? AND event_seq=?", key).fetchone()
            if prior is not None:
                if prior["event_id"] != str(row["event_id"]) or prior["metadata_json"] != serialized:
                    raise ValueError("pattern instance metadata is immutable for its run and sequence")
                continue
            try:
                conn.execute("INSERT INTO pattern_instance_metadata "
                             "(run_id,code,kind,instance_id,event_seq,event_id,metadata_json) "
                             "VALUES(?,?,?,?,?,?,?)",
                             (key[0], key[2], key[1], key[3], key[4], str(row["event_id"]), serialized))
            except sqlite3.IntegrityError as exc:
                existing = conn.execute("SELECT 1 FROM pattern_instance_metadata "
                                        "WHERE run_id=? AND event_id=?",
                                        (str(run_id), str(row["event_id"]))).fetchone()
                if existing:
                    continue
                raise ValueError("pattern instance metadata sequence is immutable") from exc
            inserted += 1
    return inserted


def get_pattern_instance_metadata(run_id, code=None, kind=None, instance_id=None, db=None, path=None):
    """Read instance contract metadata by run and optional account/instance."""
    if db is not None and path is not None:
        raise ValueError("pass either db or path, not both")
    clauses, params = ["run_id=?"], [str(run_id)]
    for field, value in (("code", code), ("kind", kind), ("instance_id", instance_id)):
        if value is not None:
            clauses.append(field + "=?")
            params.append(str(value))
    context = connect(path=path) if db is None else _yield(db)
    with context as conn:
        rows = conn.execute("SELECT metadata_json FROM pattern_instance_metadata WHERE "
                            + " AND ".join(clauses)
                            + " ORDER BY kind,code,instance_id,event_seq", params).fetchall()
        return [json.loads(row[0]) for row in rows]


class _yield:
    def __init__(self, value): self.value=value
    def __enter__(self): return self.value
    def __exit__(self,*args): return False


def create_scan_run(run_id, params, total=0, rule_version=None, app_version=None):
    with connect() as db:
        db.execute(
            "INSERT INTO scan_runs(run_id,status,params_json,total,rule_version,app_version) VALUES(?,?,?,?,?,?)",
            (run_id, "queued", json.dumps(params, ensure_ascii=False, allow_nan=False),
             int(total), rule_version, app_version),
        )


def update_scan_run(run_id, **fields):
    allowed = {"status", "progress", "scanned", "total", "inventory", "error"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError("unsupported scan run field: " + ", ".join(sorted(unknown)))
    if not fields:
        return
    values = {}
    for key, value in fields.items():
        values["inventory_json" if key == "inventory" else key] = (
            json.dumps(value, ensure_ascii=False, allow_nan=False) if key == "inventory"
            else value
        )
    assignments = ",".join(f"{key}=?" for key in values)
    with connect() as db:
        db.execute(
            f"UPDATE scan_runs SET {assignments},updated_at=CURRENT_TIMESTAMP WHERE run_id=?",
            (*values.values(), run_id),
        )


def save_scan_result(run_id, ordinal, result, assessment=None):
    payload = dict(result)
    patterns = payload.get("patterns") or []
    with connect() as db:
        _save_scan_result(db, run_id, ordinal, payload, patterns, preserve_snapshot=assessment is None)
        shared_events = payload.get("pattern_instance_events", [])
        if shared_events:
            save_pattern_instance_events(run_id, payload.get("code", ""),
                                         payload.get("kind", ""), shared_events, db=db)
        if assessment is not None:
            db.execute("INSERT OR IGNORE INTO scan_assessments(run_id,ordinal,assessment_version,payload_json) VALUES(?,?,?,?)",
                       (run_id, int(ordinal), SCAN_ASSESSMENT_VERSION, json.dumps(assessment, ensure_ascii=False, allow_nan=False)))


def _save_scan_result(db, run_id, ordinal, payload, patterns=None, preserve_snapshot=True):
    payload = dict(payload)
    patterns = payload.get("patterns") or [] if patterns is None else patterns
    events = payload.pop("main_structure_events", [])
    mini_bars = payload.get("mini_bars") if preserve_snapshot else None
    if mini_bars:
        events.append({"pattern_id": "__cutoff_snapshot__", "mini_bars": mini_bars,
                       "analysis_window": payload.get("analysis_window"),
                       "latest_date": payload.get("latest_date")})
    for key in ("selected_main_structures", "main_children", "ended_patterns", "mini_bars", "pattern_migration"):
        payload.pop(key, None)
    payload.pop("selected_plan", None)
    payload.pop("candidate_plans", None)
    db.executemany(
        "INSERT OR IGNORE INTO scan_result_evidence(run_id,ordinal,pattern_id,payload_json) VALUES(?,?,?,?)",
        [(run_id, int(ordinal), str(event.get("pattern_id") or f"event-{index}"),
          json.dumps(event, ensure_ascii=False, allow_nan=False))
         for index, event in enumerate(events)],
    )
    db.execute(
        """INSERT OR IGNORE INTO scan_results
           (run_id,ordinal,code,kind,name,category,action,patterns_json,payload_json)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (run_id, int(ordinal), str(payload.get("code", "")), str(payload.get("kind", "")),
         str(payload.get("name") or payload.get("code") or ""),
         str(payload.get("category", "unclassified")), str(payload.get("action", "INSUFFICIENT_DATA")),
         json.dumps(patterns, ensure_ascii=False, allow_nan=False),
         json.dumps(payload, ensure_ascii=False, allow_nan=False)),
    )


def save_scan_assessment(run_id, ordinal, assessment):
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO scan_assessments(run_id,ordinal,assessment_version,payload_json) VALUES(?,?,?,?)",
                   (run_id, int(ordinal), SCAN_ASSESSMENT_VERSION, json.dumps(assessment, ensure_ascii=False, allow_nan=False)))


_assessment_threads = set()
_assessment_threads_lock = threading.Lock()


def _assess_saved_result(payload, events, requested_timeframes):
    import pandas as pd
    from .assessment import assess_scan
    snapshot = next((item for item in events if item.get("pattern_id") == "__cutoff_snapshot__"), {})
    bars = payload.get("mini_bars") or snapshot.get("mini_bars") or []
    window = payload.get("analysis_window") or snapshot.get("analysis_window") or {}
    as_of = window.get("to") or payload.get("latest_date") or snapshot.get("latest_date")
    patterns = [item for item in events if item.get("pattern_id") != "__cutoff_snapshot__"]
    patterns = patterns or payload.get("main_structure_events") or []
    patterns = [pattern for pattern in patterns
                if pattern.get("state") == "confirmed"
                or any(item.get("code") == "neckline_breakout" for item in pattern.get("evidence", []))]
    result = assess_scan(patterns, {"1d": pd.DataFrame(bars)}, as_of)
    needs_weekly = "1w" in requested_timeframes or any(pattern.get("timeframe") == "1w" for pattern in patterns)
    if needs_weekly and result.get("status") != "BUY":
        result["evidence_gaps"] = ["weekly_cutoff_ohlc_snapshot_missing"]
        result["timeframes"].setdefault("1w", {"status": "INSUFFICIENT_EVIDENCE",
                                                 "reason": "weekly_cutoff_ohlc_snapshot_missing"})
        if result.get("status") == "NONE":
            result["status"] = "INSUFFICIENT_EVIDENCE"
            result["reason"] = "weekly_cutoff_ohlc_snapshot_missing"
            result["selected_timeframe"] = "1w"
    return result


def _run_saved_scan_assessment(run_id):
    batch_size = 25
    try:
        with connect() as db:
            scan = db.execute("SELECT params_json FROM scan_runs WHERE run_id=?", (run_id,)).fetchone()
        requested_timeframes = json.loads(scan["params_json"]).get("timeframes", []) if scan else []
        while True:
            with connect() as db:
                rows = db.execute(
                    "SELECT scan_results.ordinal,scan_results.payload_json FROM scan_results "
                    "LEFT JOIN scan_assessments ON scan_assessments.run_id=scan_results.run_id "
                    "AND scan_assessments.ordinal=scan_results.ordinal "
                    "AND scan_assessments.assessment_version=? "
                    "WHERE scan_results.run_id=? AND scan_assessments.ordinal IS NULL "
                    "ORDER BY scan_results.ordinal LIMIT ?", (SCAN_ASSESSMENT_VERSION, run_id, batch_size),
                ).fetchall()
                total = db.execute("SELECT COUNT(*) FROM scan_results WHERE run_id=?", (run_id,)).fetchone()[0]
            if not rows:
                break
            ordinals = [row["ordinal"] for row in rows]
            marks = ",".join("?" for _ in ordinals)
            with connect() as db:
                events = {}
                for item in db.execute(
                    f"SELECT ordinal,payload_json FROM scan_result_evidence WHERE run_id=? AND ordinal IN ({marks})",
                    (run_id, *ordinals),
                ).fetchall():
                    events.setdefault(item["ordinal"], []).append(json.loads(item["payload_json"]))
            assessed = [(run_id, row["ordinal"], json.dumps(
                _assess_saved_result(json.loads(row["payload_json"]), events.get(row["ordinal"], []), requested_timeframes),
                ensure_ascii=False, allow_nan=False)) for row in rows]
            with connect() as db:
                db.executemany("INSERT OR IGNORE INTO scan_assessments(run_id,ordinal,assessment_version,payload_json) VALUES(?,?,?,?)",
                               [(run, ordinal, SCAN_ASSESSMENT_VERSION, payload) for run, ordinal, payload in assessed])
                processed = db.execute("SELECT COUNT(*) FROM scan_assessments WHERE run_id=? AND assessment_version=?",
                                       (run_id, SCAN_ASSESSMENT_VERSION)).fetchone()[0]
                db.execute("UPDATE scan_assessment_jobs SET status=?,processed=?,total=?,updated_at=CURRENT_TIMESTAMP WHERE run_id=?",
                           ("running", processed, total, run_id))
        with connect() as db:
            processed = db.execute("SELECT COUNT(*) FROM scan_assessments WHERE run_id=? AND assessment_version=?",
                                   (run_id, SCAN_ASSESSMENT_VERSION)).fetchone()[0]
            total = db.execute("SELECT COUNT(*) FROM scan_results WHERE run_id=?", (run_id,)).fetchone()[0]
            db.execute("UPDATE scan_assessment_jobs SET status='completed',processed=?,total=?,updated_at=CURRENT_TIMESTAMP WHERE run_id=?",
                       (processed, total, run_id))
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE scan_assessment_jobs SET status='failed',error=?,updated_at=CURRENT_TIMESTAMP WHERE run_id=?",
                       (str(exc), run_id))
    finally:
        with _assessment_threads_lock:
            _assessment_threads.discard(run_id)


def start_saved_scan_assessment(run_id):
    with connect() as db:
        run = db.execute("SELECT status FROM scan_runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            return {"status": "not_found", "processed": 0, "total": 0, "error": None}
        if run["status"] in ("queued", "running"):
            return {"status": "scan_running", "processed": 0, "total": 0, "error": None}
        total = db.execute("SELECT COUNT(*) FROM scan_results WHERE run_id=?", (run_id,)).fetchone()[0]
        assessed = db.execute("SELECT COUNT(*) FROM scan_assessments WHERE run_id=? AND assessment_version=?",
                              (run_id, SCAN_ASSESSMENT_VERSION)).fetchone()[0]
        if assessed >= total:
            db.execute("INSERT INTO scan_assessment_jobs(run_id,status,processed,total) VALUES(?,?,?,?) "
                       "ON CONFLICT(run_id) DO UPDATE SET status='completed',processed=excluded.processed,total=excluded.total,updated_at=CURRENT_TIMESTAMP",
                       (run_id, "completed", assessed, total))
        else:
            db.execute("INSERT INTO scan_assessment_jobs(run_id,status,processed,total) VALUES(?,?,?,?) "
                       "ON CONFLICT(run_id) DO UPDATE SET status='running',processed=excluded.processed,total=excluded.total,error=NULL,updated_at=CURRENT_TIMESTAMP",
                       (run_id, "running", assessed, total))
    with _assessment_threads_lock:
        if assessed < total and run_id not in _assessment_threads:
            _assessment_threads.add(run_id)
            threading.Thread(target=_run_saved_scan_assessment, args=(run_id,), daemon=True).start()
    return get_scan_assessment_progress(run_id)


def get_scan_assessment_progress(run_id):
    with connect() as db:
        row = db.execute("SELECT status,processed,total,error,updated_at FROM scan_assessment_jobs WHERE run_id=?",
                         (run_id,)).fetchone()
    return ({"status": row["status"], "processed": row["processed"], "total": row["total"],
             "error": row["error"], "updated_at": row["updated_at"]} if row else
            {"status": "not_started", "processed": 0, "total": 0, "error": None})


def _is_legacy_scan_job(payload):
    if not isinstance(payload, dict) or payload.get("job_type") in ("fixed_backtest", "daily_backtest", "data_refresh"):
        return False
    return (isinstance(payload.get("inventory"), dict)
            and isinstance(payload.get("results"), list)
            and "scanned" in payload and "total" in payload)


def _is_scan_failure(result):
    if not isinstance(result, dict):
        return False
    return (result.get("state") == "data_error" or bool(result.get("error"))
            or "file_analysis_failed" in (result.get("reason_codes") or []))


def import_legacy_scan_jobs():
    """Idempotently copy old scan job snapshots into normalized scan history."""
    imported = 0
    with connect() as db:
        migration_id = "legacy_scan_jobs_v1"
        if db.execute("SELECT 1 FROM migration_runs WHERE migration_id=?", (migration_id,)).fetchone():
            return 0
        rows = db.execute(
            """SELECT jobs.job_id,jobs.status,jobs.payload,jobs.created_at
               FROM jobs LEFT JOIN scan_runs ON scan_runs.run_id=jobs.job_id
               WHERE scan_runs.run_id IS NULL
                 AND instr(jobs.payload,'\"inventory\"')>0
                 AND instr(jobs.payload,'\"scanned\"')>0
                 AND instr(jobs.payload,'\"total\"')>0
                 AND instr(jobs.payload,'\"results\"')>0
                 AND instr(jobs.payload,'\"job_type\":\"fixed_backtest\"')=0
                 AND instr(jobs.payload,'\"job_type\":\"daily_backtest\"')=0
                 AND instr(jobs.payload,'\"job_type\":\"data_refresh\"')=0
               ORDER BY jobs.created_at,jobs.job_id"""
        )
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError):
                continue
            if not _is_legacy_scan_job(payload):
                continue

            raw_results = payload.get("results") or []
            results = [(ordinal, item) for ordinal, item in enumerate(raw_results)
                       if isinstance(item, dict)]
            inventory = dict(payload.get("inventory") or {})
            explicit_failures = inventory.get("failed_count")
            if isinstance(explicit_failures, bool) or not isinstance(explicit_failures, int):
                explicit_failures = None
            inferred_failures = sum(_is_scan_failure(item) for _, item in results)
            scanned = max(0, int(payload.get("scanned") or 0))
            total = max(0, int(payload.get("total") or 0))
            result_snapshot_complete = (len(results) == scanned == total and total > 0)
            failure_count_known = explicit_failures is not None or result_snapshot_complete
            if failure_count_known:
                failure_count = max(explicit_failures or 0, inferred_failures)
                inventory["failed_count"] = failure_count
            else:
                inventory.pop("failed_count", None)
            inventory["failure_count_known"] = failure_count_known
            inventory["legacy_import"] = True
            inventory["unavailable_metadata"] = ["scan parameters", "rule version", "application version"]

            params = payload.get("params")
            if not isinstance(params, dict):
                recognized = ("timeframes", "kinds", "limit", "date_from", "date_to")
                params = {key: payload[key] for key in recognized if key in payload}
                missing = [key for key in recognized if key not in params]
                params["_legacy_import"] = True
                params["_not_recorded"] = ["scan parameters"] if missing else []
            params_json = json.dumps(params, ensure_ascii=False, allow_nan=False)

            status = payload.get("status", row["status"])
            error_count = inventory.get("failed_count")
            if status == "completed" and failure_count_known and error_count:
                status = "failed" if total and error_count >= total else "partial_failed"
            progress = payload.get("progress")
            if not isinstance(progress, (int, float)):
                progress = 1.0 if status in ("completed", "failed", "partial_failed") else 0.0
            rule_version = payload.get("rule_version")
            app_version = payload.get("app_version")
            inventory_json = json.dumps(inventory, ensure_ascii=False, allow_nan=False)
            db.execute(
                """INSERT OR IGNORE INTO scan_runs
                   (run_id,status,params_json,progress,scanned,total,inventory_json,rule_version,app_version,error,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row["job_id"], status, params_json, float(progress), scanned, total, inventory_json,
                 rule_version, app_version, payload.get("error"), row["created_at"], row["created_at"]),
            )
            if db.execute("SELECT changes()").fetchone()[0] == 0:
                continue
            for ordinal, result in results:
                _save_scan_result(db, row["job_id"], ordinal, result)
            imported += 1
        db.execute("INSERT INTO migration_runs(migration_id) VALUES(?)", (migration_id,))
    return imported


def interrupt_active_scan_runs():
    with connect() as db:
        run_ids = [row[0] for row in db.execute(
            "SELECT run_id FROM scan_runs WHERE status IN ('queued','running')"
        ).fetchall()]
        cursor = db.execute(
            "UPDATE scan_runs SET status='interrupted',error='服务重启时任务仍在运行，未自动续跑',updated_at=CURRENT_TIMESTAMP "
            "WHERE status IN ('queued','running')"
        )
        for run_id in run_ids:
            legacy = db.execute("SELECT payload FROM jobs WHERE job_id=?", (run_id,)).fetchone()
            if legacy:
                payload = json.loads(legacy[0])
                payload.update(status="interrupted", error="服务重启时任务仍在运行，未自动续跑")
                db.execute("UPDATE jobs SET status='interrupted',payload=? WHERE job_id=?",
                           (json.dumps(payload, ensure_ascii=False, allow_nan=False), run_id))
        return cursor.rowcount


def list_scan_runs(limit=50, offset=0):
    with connect() as db:
        total = db.execute("SELECT COUNT(*) FROM scan_runs").fetchone()[0]
        rows = db.execute(
            "SELECT * FROM scan_runs ORDER BY created_at DESC,run_id DESC LIMIT ? OFFSET ?",
            (int(limit), int(offset)),
        ).fetchall()
    runs = []
    for row in rows:
        runs.append({
            "run_id": row["run_id"], "status": row["status"],
            "params": json.loads(row["params_json"]), "progress": row["progress"],
            "scanned": row["scanned"], "total": row["total"],
            "inventory": json.loads(row["inventory_json"]), "rule_version": row["rule_version"],
            "app_version": row["app_version"], "error": row["error"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        })
    return runs, total


def get_scan_run(run_id):
    with connect() as db:
        row = db.execute("SELECT * FROM scan_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    return {
        "run_id": row["run_id"], "status": row["status"],
        "params": json.loads(row["params_json"]), "progress": row["progress"],
        "scanned": row["scanned"], "total": row["total"],
        "inventory": json.loads(row["inventory_json"]), "rule_version": row["rule_version"],
        "app_version": row["app_version"], "error": row["error"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def query_scan_results(run_id, search=None, category=None, action=None, kind=None, pattern_types=None,
                       timeframes=None, sort_by="rank", sort_order="asc", limit=50, offset=0):
    order_columns = {"code": "scan_results.code", "name": "scan_results.name",
                     "category": "scan_results.category", "action": "scan_results.action"}
    if sort_by not in (*order_columns, "rank") or sort_order not in ("asc", "desc"):
        raise ValueError("invalid scan result sort")
    clauses = ["scan_results.run_id=?"]
    params = [run_id]
    if search:
        clauses.append("(scan_results.code LIKE ? COLLATE NOCASE OR scan_results.name LIKE ? COLLATE NOCASE)")
        params.extend([f"%{search}%", f"%{search}%"])
    if category:
        clauses.append("scan_results.category=?")
        params.append(category)
    if action:
        clauses.append("CASE json_extract(scan_assessments.payload_json,'$.status') "
                       "WHEN 'BUY' THEN 'CONSIDER_BUY' WHEN 'FOCUS_NECKLINE' THEN 'FOCUS_NECKLINE' "
                       "WHEN 'NONE' THEN CASE WHEN scan_results.action='CONSIDER_BUY' THEN 'WAIT' ELSE scan_results.action END "
                       "ELSE scan_results.action END=?")
        params.append(action)
    if kind:
        clauses.append("scan_results.kind=?")
        params.append(kind)
    patterns = list(dict.fromkeys(pattern_types or []))
    periods = list(dict.fromkeys(timeframes or []))
    if patterns or periods:
        pattern_clauses = [
            "json_extract(pattern.value,'$.is_main_structure')=1",
            "json_extract(pattern.value,'$.selected_main')=1",
            "COALESCE(json_extract(pattern.value,'$.ambiguous_structure'),0)=0",
            "COALESCE(json_extract(pattern.value,'$.execution_suppressed'),0)=0",
            "((json_extract(scan_assessments.payload_json,'$.timeframes.' || json_extract(pattern.value,'$.timeframe') || '.status') IN ('BUY','FOCUS_NECKLINE') "
            "AND json_extract(scan_assessments.payload_json,'$.timeframes.' || json_extract(pattern.value,'$.timeframe') || '.pattern_id')=json_extract(pattern.value,'$.pattern_id')) "
            "OR (COALESCE(json_extract(scan_assessments.payload_json,'$.timeframes.' || json_extract(pattern.value,'$.timeframe') || '.status'),'NONE') NOT IN ('BUY','FOCUS_NECKLINE') "
            "AND json_extract(pattern.value,'$.state') NOT IN ('invalidated','target_reached','followthrough_confirmed'))) ",
            "NOT EXISTS (SELECT 1 FROM json_each(scan_results.patterns_json) newer "
            "WHERE json_extract(newer.value,'$.timeframe')=json_extract(pattern.value,'$.timeframe') "
            "AND json_extract(newer.value,'$.is_main_structure')=1 "
            "AND json_extract(newer.value,'$.selected_main')=1 "
            "AND COALESCE(json_extract(newer.value,'$.ambiguous_structure'),0)=0 "
            "AND COALESCE(json_extract(newer.value,'$.execution_suppressed'),0)=0 "
            "AND ((json_extract(scan_assessments.payload_json,'$.timeframes.' || json_extract(newer.value,'$.timeframe') || '.status') IN ('BUY','FOCUS_NECKLINE') "
            "AND json_extract(scan_assessments.payload_json,'$.timeframes.' || json_extract(newer.value,'$.timeframe') || '.pattern_id')=json_extract(newer.value,'$.pattern_id')) "
            "OR (COALESCE(json_extract(scan_assessments.payload_json,'$.timeframes.' || json_extract(newer.value,'$.timeframe') || '.status'),'NONE') NOT IN ('BUY','FOCUS_NECKLINE') "
            "AND json_extract(newer.value,'$.state') NOT IN ('invalidated','target_reached','followthrough_confirmed'))) "
            "AND (COALESCE(json_extract(newer.value,'$.formed_at'),'')>COALESCE(json_extract(pattern.value,'$.formed_at'),'') "
            "OR (COALESCE(json_extract(newer.value,'$.formed_at'),'')=COALESCE(json_extract(pattern.value,'$.formed_at'),'') "
            "AND COALESCE(json_extract(newer.value,'$.pattern_id'),'')>COALESCE(json_extract(pattern.value,'$.pattern_id'),''))))",
        ]
        if patterns:
            pattern_clauses.append("json_extract(pattern.value,'$.pattern_type') IN (%s)" % ",".join("?" for _ in patterns))
            params.extend(patterns)
        if periods:
            pattern_clauses.append("json_extract(pattern.value,'$.timeframe') IN (%s)" % ",".join("?" for _ in periods))
            params.extend(periods)
        clauses.append("EXISTS (SELECT 1 FROM json_each(scan_results.patterns_json) pattern WHERE %s)" % " AND ".join(pattern_clauses))
    where = " AND ".join(clauses)
    direction = sort_order.upper()
    if sort_by == "rank":
        priority = "COALESCE(json_extract(scan_results.payload_json,'$.rank.action_priority'),CASE scan_results.action WHEN 'CONSIDER_BUY' THEN 0 WHEN 'WAIT' THEN 1 WHEN 'HOLD' THEN 2 WHEN 'SELL_IF_HELD' THEN 3 ELSE 4 END)"
        reverse = "DESC" if direction == "ASC" else "ASC"
        assessed_priority = "CASE json_extract(scan_assessments.payload_json,'$.status') WHEN 'BUY' THEN 0 WHEN 'FOCUS_NECKLINE' THEN 1 WHEN 'NONE' THEN 2 WHEN 'INSUFFICIENT_EVIDENCE' THEN 3 END"
        priority = f"COALESCE({assessed_priority},json_extract(scan_results.payload_json,'$.rank.action_priority'),CASE scan_results.action WHEN 'CONSIDER_BUY' THEN 0 WHEN 'WAIT' THEN 1 WHEN 'HOLD' THEN 2 WHEN 'SELL_IF_HELD' THEN 3 ELSE 4 END)"
        order_by = (f"{priority} {direction}, "
                    f"COALESCE(json_extract(scan_results.payload_json,'$.rank.confirmed_timeframes'),0) {reverse}, "
                    f"COALESCE(json_extract(scan_results.payload_json,'$.rank.reward_risk'),0) {reverse}, "
                    f"COALESCE(json_extract(scan_results.payload_json,'$.rank.risk_pct'),1) {direction}, "
                    f"COALESCE(json_extract(scan_results.payload_json,'$.rank.confirmation_ordinal'),0) {reverse}, "
                    f"scan_results.kind {direction}, scan_results.code {direction}, scan_results.ordinal ASC")
    else:
        column = ("CASE json_extract(scan_assessments.payload_json,'$.status') "
                  "WHEN 'BUY' THEN 'CONSIDER_BUY' WHEN 'FOCUS_NECKLINE' THEN 'FOCUS_NECKLINE' "
                  "WHEN 'NONE' THEN CASE WHEN scan_results.action='CONSIDER_BUY' THEN 'WAIT' ELSE scan_results.action END "
                  "ELSE scan_results.action END" if sort_by == "action" else order_columns[sort_by])
        order_by = f"{column} {direction},scan_results.ordinal ASC"
    with connect() as db:
        join = ("LEFT JOIN scan_assessments ON scan_assessments.run_id=scan_results.run_id "
                "AND scan_assessments.ordinal=scan_results.ordinal AND scan_assessments.assessment_version=? ")
        total = db.execute(f"SELECT COUNT(*) FROM scan_results {join} WHERE {where}",
                           (SCAN_ASSESSMENT_VERSION, *params)).fetchone()[0]
        rows = db.execute(
            f"SELECT scan_results.code,scan_results.kind,scan_results.name,scan_results.category,scan_results.action,"
            f"scan_results.patterns_json,scan_assessments.payload_json AS assessment_json,"
            f"json_extract(scan_results.payload_json,'$.state') AS state,"
            f"json_extract(scan_results.payload_json,'$.latest_date') AS latest_date,"
            f"json_extract(scan_results.payload_json,'$.latest_close') AS latest_close,"
            f"json_extract(scan_results.payload_json,'$.analysis_window') AS analysis_window,"
            f"json_extract(scan_results.payload_json,'$.reason_codes') AS reason_codes,"
            f"json_extract(scan_results.payload_json,'$.explanation') AS explanation "
            f"FROM scan_results {join} WHERE {where} "
            f"ORDER BY {order_by} LIMIT ? OFFSET ?",
            (SCAN_ASSESSMENT_VERSION, *params, int(limit), int(offset)),
        ).fetchall()
    results = []
    for row in rows:
        patterns = json.loads(row["patterns_json"] or "[]")
        latest_mains = {}
        assessment = json.loads(row["assessment_json"]) if row["assessment_json"] is not None else None
        for pattern in patterns:
            if not (pattern.get("is_main_structure") and pattern.get("selected_main")
                    and not pattern.get("ambiguous_structure") and not pattern.get("execution_suppressed")):
                continue
            timeframe = pattern.get("timeframe")
            per_tf = (assessment or {}).get("timeframes", {}).get(timeframe, {})
            triggered = (per_tf.get("status") in {"BUY", "FOCUS_NECKLINE"}
                         and per_tf.get("pattern_id") == pattern.get("pattern_id"))
            active = pattern.get("state") not in {"invalidated", "target_reached", "followthrough_confirmed"}
            if not (triggered or (active and per_tf.get("status") not in {"BUY", "FOCUS_NECKLINE"})):
                continue
            current = latest_mains.get(timeframe)
            if current is None or (str(pattern.get("formed_at") or ""), str(pattern.get("pattern_id") or "")) > (
                    str(current.get("formed_at") or ""), str(current.get("pattern_id") or "")):
                latest_mains[timeframe] = pattern
        result = {key: row[key] for key in ("code", "kind", "name", "category", "action", "state", "latest_date", "latest_close", "analysis_window", "reason_codes", "explanation")}
        result["patterns"] = list(latest_mains.values())
        if row["reason_codes"]:
            result["reason_codes"] = json.loads(row["reason_codes"])
        if row["analysis_window"]:
            result["analysis_window"] = json.loads(row["analysis_window"])
        if row["assessment_json"] is not None:
            result["assessment"] = assessment
            status = result["assessment"].get("status")
            if status == "BUY":
                result["action"] = "CONSIDER_BUY"
            elif status == "FOCUS_NECKLINE":
                result["action"] = "FOCUS_NECKLINE"
            elif status == "NONE":
                if result["action"] == "CONSIDER_BUY":
                    result["action"] = "WAIT"
            elif status == "INSUFFICIENT_EVIDENCE":
                result["assessment_notice"] = "缺少扫描时点行情或完整ATR快照，保留原始行动，无法重算。"
        else:
            result["assessment_notice"] = "正在依据已保存证据评估；当前保留原始行动。"
        results.append(result)
    return results, total
