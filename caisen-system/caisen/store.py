import json
import hashlib
import os
import sqlite3
from contextlib import contextmanager

from .scan_summary import scan_summary


class LifecycleEventConflict(ValueError):
    """An append-only lifecycle event key already has a different payload."""


_UNSET = object()


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
        CREATE TABLE IF NOT EXISTS pattern_lifecycle_events(
            analysis_batch_id TEXT NOT NULL,
            instance_id TEXT NOT NULL,
            event_sequence INTEGER NOT NULL CHECK(event_sequence > 0),
            data_version TEXT NOT NULL,
            pattern_id TEXT NOT NULL,
            boundary_version_id TEXT NOT NULL,
            event_date TEXT,
            event_kind TEXT NOT NULL,
            lifecycle_schema_version TEXT NOT NULL,
            recognition_context_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(analysis_batch_id,instance_id,event_sequence)
        );
        CREATE INDEX IF NOT EXISTS lifecycle_events_instance_version
            ON pattern_lifecycle_events(instance_id,data_version,event_sequence);
        CREATE TABLE IF NOT EXISTS pattern_lifecycle_event_aliases(
            analysis_batch_id TEXT NOT NULL,
            instance_id TEXT NOT NULL,
            event_sequence INTEGER NOT NULL CHECK(event_sequence > 0),
            pattern_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(analysis_batch_id,instance_id,event_sequence,pattern_id)
        );
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
        CREATE INDEX IF NOT EXISTS scan_results_run_code ON scan_results(run_id, code);
        CREATE INDEX IF NOT EXISTS scan_results_run_action ON scan_results(run_id, action);
        CREATE TABLE IF NOT EXISTS migration_runs(
            migration_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)
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


def save_pattern(pattern, db=None, analysis_batch_id=None, recognition_context=_UNSET):
    return save_patterns([pattern], db=db, analysis_batch_id=analysis_batch_id,
                         recognition_context=recognition_context)


def save_patterns(patterns, db=None, analysis_batch_id=None, recognition_context=_UNSET):
    payloads = [pattern.to_dict() if hasattr(pattern, "to_dict") else pattern for pattern in patterns]
    if not payloads:
        return
    lifecycle_rows = []
    for payload in payloads:
        lifecycle_rows.extend(_lifecycle_event_rows(
            payload, analysis_batch_id=analysis_batch_id,
            recognition_context=recognition_context))
    _validate_event_batch(lifecycle_rows)
    with connect() if db is None else _yield(db) as conn:
        savepoint = "save_pattern_batch"
        conn.execute("SAVEPOINT " + savepoint)
        try:
            _assert_existing_events_compatible(conn, lifecycle_rows)
            snapshots = [(payload["pattern_id"], payload.get("data_version", "unknown"),
                          json.dumps(payload, ensure_ascii=False)) for payload in payloads]
            current = [(payload["pattern_id"], json.dumps(payload, ensure_ascii=False)) for payload in payloads]
            conn.executemany("INSERT OR IGNORE INTO pattern_snapshots(pattern_id,data_version,payload) VALUES(?,?,?)", snapshots)
            conn.executemany("INSERT OR REPLACE INTO patterns(pattern_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)", current)
            _append_lifecycle_events(conn, lifecycle_rows)
        except Exception:
            conn.execute("ROLLBACK TO SAVEPOINT " + savepoint)
            conn.execute("RELEASE SAVEPOINT " + savepoint)
            raise
        else:
            conn.execute("RELEASE SAVEPOINT " + savepoint)


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _is_real_data_version(value):
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().lower() not in {"unknown", "none", "null", "unavailable"}


def _joint_lifecycle(payload):
    for item in payload.get("evidence", []) or []:
        if isinstance(item, dict) and item.get("code") == "joint_lifecycle":
            value = item.get("value")
            return value if isinstance(value, dict) else None
    return None


def _recognition_context(payload, lifecycle, explicit_context):
    if explicit_context is not _UNSET:
        return explicit_context
    if "recognition_context" in lifecycle:
        return lifecycle["recognition_context"]
    if "recognition_context" in payload:
        return payload["recognition_context"]
    if "recognition_experiment" in payload:
        return payload["recognition_experiment"]
    return None


def _analysis_batch_id(payload, data_version, lifecycle_schema_version,
                       recognition_context, explicit_batch_id):
    selected = explicit_batch_id
    if selected is None:
        selected = payload.get("analysis_batch_id")
    if selected is not None:
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("analysis_batch_id must be a non-empty string")
        return selected.strip()
    identity = {
        "code": payload.get("code"), "kind": payload.get("kind"),
        "timeframe": payload.get("timeframe"), "data_version": data_version,
        "rule_version": payload.get("rule_version"),
        "source_rule_version": payload.get("source_rule_version"),
        "lifecycle_schema_version": lifecycle_schema_version,
        "recognition_context": recognition_context,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _lifecycle_event_rows(payload, analysis_batch_id=None, recognition_context=_UNSET):
    if not isinstance(payload, dict):
        return []
    data_version = payload.get("data_version")
    if not _is_real_data_version(data_version):
        return []
    lifecycle = _joint_lifecycle(payload)
    if lifecycle is None:
        return []
    schema_version = lifecycle.get("schema_version")
    instance_id = lifecycle.get("instance_id")
    boundary_version_id = lifecycle.get("boundary_version_id")
    pattern_id = payload.get("pattern_id")
    events = lifecycle.get("events")
    if (not isinstance(schema_version, str) or not schema_version.strip()
            or not isinstance(instance_id, str) or not instance_id.strip()
            or not isinstance(boundary_version_id, str) or not boundary_version_id.strip()
            or not isinstance(pattern_id, str) or not pattern_id.strip()
            or not isinstance(events, list)):
        return []
    context = _recognition_context(payload, lifecycle, recognition_context)
    context_json = _canonical_json(context)
    batch_id = _analysis_batch_id(payload, data_version, schema_version, context,
                                  analysis_batch_id)
    rows = []
    previous_sequence = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("joint_lifecycle.events entries must be objects")
        sequence = event.get("sequence")
        if (not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1
                or sequence <= previous_sequence):
            raise ValueError("joint_lifecycle.events must be in strictly increasing sequence order")
        if event.get("instance_id") != instance_id:
            raise ValueError("lifecycle event instance_id does not match its lifecycle")
        if event.get("boundary_version_id") != boundary_version_id:
            raise ValueError("lifecycle event boundary_version_id does not match its lifecycle")
        event_kind = event.get("kind")
        if not isinstance(event_kind, str) or not event_kind.strip():
            raise ValueError("lifecycle event kind must be a non-empty string")
        rows.append({
            "analysis_batch_id": batch_id,
            "instance_id": instance_id,
            "event_sequence": sequence,
            "data_version": data_version,
            "pattern_id": pattern_id,
            "boundary_version_id": boundary_version_id,
            "event_date": event.get("date"),
            "event_kind": event_kind,
            "lifecycle_schema_version": schema_version,
            "recognition_context_json": context_json,
            "payload_json": _canonical_json(event),
        })
        previous_sequence = sequence
    return rows


def _validate_event_batch(rows):
    seen = {}
    immutable_fields = ("data_version", "boundary_version_id", "event_date",
                        "event_kind", "lifecycle_schema_version", "recognition_context_json",
                        "payload_json")
    for row in rows:
        key = (row["analysis_batch_id"], row["instance_id"], row["event_sequence"])
        previous = seen.get(key)
        if previous is not None and any(previous[field] != row[field] for field in immutable_fields):
            raise LifecycleEventConflict("conflicting lifecycle event payload in one save batch for %s" % (key,))
        seen[key] = row


def _assert_existing_events_compatible(conn, rows):
    keys = set((row["analysis_batch_id"], row["instance_id"], row["event_sequence"])
               for row in rows)
    for batch_id, instance_id, sequence in keys:
        existing = _fetchone_dict(conn,
            "SELECT data_version,boundary_version_id,event_date,event_kind,"
            "lifecycle_schema_version,recognition_context_json,payload_json "
            "FROM pattern_lifecycle_events "
            "WHERE analysis_batch_id=? AND instance_id=? AND event_sequence=?",
            (batch_id, instance_id, sequence))
        if existing is None:
            continue
        incoming = next(row for row in rows
                        if (row["analysis_batch_id"], row["instance_id"], row["event_sequence"])
                        == (batch_id, instance_id, sequence))
        immutable_fields = ("data_version", "boundary_version_id", "event_date",
                            "event_kind", "lifecycle_schema_version", "recognition_context_json",
                            "payload_json")
        if any(existing[field] != incoming[field] for field in immutable_fields):
            raise LifecycleEventConflict(
                "lifecycle event key already has a different immutable payload: %s" %
                ((batch_id, instance_id, sequence),))


def _append_lifecycle_events(conn, rows):
    fields = ("analysis_batch_id", "instance_id", "event_sequence", "data_version",
              "pattern_id", "boundary_version_id", "event_date", "event_kind",
              "lifecycle_schema_version", "recognition_context_json", "payload_json")
    if not rows:
        return
    placeholders = ",".join("?" for _ in fields)
    columns = ",".join(fields)
    statement = "INSERT OR IGNORE INTO pattern_lifecycle_events (%s) VALUES (%s)" % (columns, placeholders)
    for row in rows:
        conn.execute(statement, tuple(row[field] for field in fields))
        existing = _fetchone_dict(conn,
            "SELECT data_version,boundary_version_id,event_date,event_kind,"
            "lifecycle_schema_version,recognition_context_json,payload_json "
            "FROM pattern_lifecycle_events "
            "WHERE analysis_batch_id=? AND instance_id=? AND event_sequence=?",
            (row["analysis_batch_id"], row["instance_id"], row["event_sequence"]))
        immutable_fields = ("data_version", "boundary_version_id", "event_date",
                            "event_kind", "lifecycle_schema_version", "recognition_context_json",
                            "payload_json")
        if existing is None or any(existing[field] != row[field] for field in immutable_fields):
            raise LifecycleEventConflict("lifecycle event changed while being appended")
        conn.execute(
            "INSERT OR IGNORE INTO pattern_lifecycle_event_aliases "
            "(analysis_batch_id,instance_id,event_sequence,pattern_id) VALUES(?,?,?,?)",
            (row["analysis_batch_id"], row["instance_id"], row["event_sequence"],
             row["pattern_id"]))


def get_lifecycle_events(analysis_batch_id=None, instance_id=None, data_version=None, db=None):
    clauses, params = [], []
    for column, value in (("analysis_batch_id", analysis_batch_id),
                          ("instance_id", instance_id), ("data_version", data_version)):
        if value is not None:
            clauses.append(column + "=?")
            params.append(value)
    query = "SELECT * FROM pattern_lifecycle_events"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY analysis_batch_id,instance_id,event_sequence"
    with connect() if db is None else _yield(db) as conn:
        rows = _fetchall_dicts(conn, query, params)
    result = []
    for row in rows:
        item = row
        item["payload"] = json.loads(item.pop("payload_json"))
        item["recognition_context"] = json.loads(item.pop("recognition_context_json"))
        item["sequence"] = item["event_sequence"]
        item["date"] = item["event_date"]
        item["kind"] = item["event_kind"]
        result.append(item)
    return result


def get_lifecycle_event_aliases(analysis_batch_id=None, instance_id=None,
                                event_sequence=None, db=None):
    clauses, params = [], []
    for column, value in (("analysis_batch_id", analysis_batch_id),
                          ("instance_id", instance_id),
                          ("event_sequence", event_sequence)):
        if value is not None:
            clauses.append(column + "=?")
            params.append(value)
    query = "SELECT * FROM pattern_lifecycle_event_aliases"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY analysis_batch_id,instance_id,event_sequence,pattern_id"
    with connect() if db is None else _yield(db) as conn:
        rows = _fetchall_dicts(conn, query, params)
    return rows


def _fetchone_dict(conn, query, params=()):
    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return dict(zip((column[0] for column in cursor.description), row))


def _fetchall_dicts(conn, query, params=()):
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    if not rows:
        return []
    if isinstance(rows[0], sqlite3.Row):
        return [dict(row) for row in rows]
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def get_pattern(pattern_id, db=None):
    with connect() if db is None else _yield(db) as conn:
        row = conn.execute("SELECT payload FROM patterns WHERE pattern_id=?", (pattern_id,)).fetchone()
        return json.loads(row[0]) if row else None


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


def save_scan_result(run_id, ordinal, result):
    with connect() as db:
        _save_scan_result(db, run_id, ordinal, result)


def _save_scan_result(db, run_id, ordinal, row):
    row = scan_summary(row)
    patterns = row.get("patterns") or []
    payload = {k:v for k,v in row.items()
               if k not in {"patterns", "code", "kind", "name", "category", "action"}}
    db.execute(
        """INSERT OR REPLACE INTO scan_results
           (run_id,ordinal,code,kind,name,category,action,patterns_json,payload_json)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (run_id, int(ordinal), str(row.get("code", "")), str(row.get("kind", "")),
         str(row.get("name") or row.get("code") or ""),
         str(row.get("category", "unclassified")), str(row.get("action", "INSUFFICIENT_DATA")),
         json.dumps(patterns, ensure_ascii=False, allow_nan=False),
         json.dumps(payload, ensure_ascii=False, allow_nan=False)),
    )


def rank_saved_scan_results(run_id, ordinals):
    """Permute persisted rows without encoding or replacing result payloads."""
    with connect() as db:
        db.execute("UPDATE scan_results SET ordinal=-ordinal-1 WHERE run_id=?", (run_id,))
        db.executemany("UPDATE scan_results SET ordinal=? WHERE run_id=? AND ordinal=?",
                       ((rank, run_id, -original-1) for rank, original in enumerate(ordinals)))


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
    params = json.loads(row["params_json"])
    return {
        "run_id": row["run_id"], "status": row["status"],
        "params": params, "versions": params.get("versions"), "progress": row["progress"],
        "scanned": row["scanned"], "total": row["total"],
        "inventory": json.loads(row["inventory_json"]), "rule_version": row["rule_version"],
        "app_version": row["app_version"], "error": row["error"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def scan_result_count(run_id):
    with connect() as db:
        return int(db.execute("SELECT COUNT(*) FROM scan_results WHERE run_id=?", (run_id,)).fetchone()[0])


def compact_completed_scan_job(run_id):
    """Drop the duplicated job.results blob only when the paged snapshot is complete."""
    with connect() as db:
        run = db.execute("SELECT * FROM scan_runs WHERE run_id=?", (run_id,)).fetchone()
        job = db.execute("SELECT status,payload FROM jobs WHERE job_id=?", (run_id,)).fetchone()
        if run is None or job is None or run["status"] not in ("completed", "partial_failed"):
            return False
        result_count = int(db.execute(
            "SELECT COUNT(*) FROM scan_results WHERE run_id=?", (run_id,)
        ).fetchone()[0])
        if result_count != int(run["scanned"]):
            return False
        valid_array = db.execute(
            "SELECT json_valid(payload),json_type(payload,'$.results'),json_array_length(payload,'$.results') "
            "FROM jobs WHERE job_id=?", (run_id,)
        ).fetchone()
        if (not valid_array or valid_array[0] != 1 or valid_array[1] != "array"
                or int(valid_array[2] or 0) != result_count):
            return False
        params = json.loads(run["params_json"])
        compact = {
            "job_type": "scan", "status": run["status"], "progress": run["progress"],
            "scanned": run["scanned"], "total": run["total"], "result_count": result_count,
            "rule_version": run["rule_version"], "versions": params.get("versions"),
            "inventory": json.loads(run["inventory_json"]), "error": run["error"],
        }
        db.execute("UPDATE jobs SET status=?,payload=? WHERE job_id=?",
                   (run["status"], json.dumps(compact, ensure_ascii=False, allow_nan=False), run_id))
        return True


def query_scan_results(run_id, search=None, category=None, action=None, kind=None, pattern_types=None,
                       timeframes=None, sort_by="rank", sort_order="asc", limit=50, offset=0):
    order_columns = {"rank": "ordinal", "code": "code", "name": "name", "category": "category", "action": "action"}
    if sort_by not in order_columns or sort_order not in ("asc", "desc"):
        raise ValueError("invalid scan result sort")
    clauses = ["run_id=?"]
    params = [run_id]
    if search:
        clauses.append("(code LIKE ? COLLATE NOCASE OR name LIKE ? COLLATE NOCASE)")
        params.extend([f"%{search}%", f"%{search}%"])
    if category:
        clauses.append("category=?")
        params.append(category)
    if action:
        clauses.append("action=?")
        params.append(action)
    if kind:
        clauses.append("kind=?")
        params.append(kind)
    patterns = list(dict.fromkeys(pattern_types or []))
    periods = list(dict.fromkeys(timeframes or []))
    if patterns or periods:
        pattern_clauses = []
        if patterns:
            pattern_clauses.append("json_extract(pattern.value,'$.pattern_type') IN (%s)" % ",".join("?" for _ in patterns))
            params.extend(patterns)
        if periods:
            pattern_clauses.append("json_extract(pattern.value,'$.timeframe') IN (%s)" % ",".join("?" for _ in periods))
            params.extend(periods)
        clauses.append("EXISTS (SELECT 1 FROM json_each(scan_results.patterns_json) pattern WHERE %s)" % " AND ".join(pattern_clauses))
    where = " AND ".join(clauses)
    column = order_columns[sort_by]
    with connect() as db:
        total = db.execute(f"SELECT COUNT(*) FROM scan_results WHERE {where}", params).fetchone()[0]
        rows = db.execute(
            f"SELECT payload_json,patterns_json,code,kind,name,category,action FROM scan_results WHERE {where} ORDER BY {column} {sort_order.upper()},ordinal ASC LIMIT ? OFFSET ?",
            (*params, int(limit), int(offset)),
        ).fetchall()
    results = []
    for row in rows:
        payload = json.loads(row[0])
        payload.update({key:row[key] for key in ("code","kind","name","category","action")})
        payload["patterns"] = json.loads(row["patterns_json"])
        results.append(payload)
    return results, total
