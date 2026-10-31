import json
import os
import sqlite3
from contextlib import contextmanager


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
    payload = dict(result)
    patterns = payload.get("patterns") or []
    with connect() as db:
        _save_scan_result(db, run_id, ordinal, payload, patterns)


def _save_scan_result(db, run_id, ordinal, payload, patterns=None):
    patterns = payload.get("patterns") or [] if patterns is None else patterns
    db.execute(
        """INSERT OR REPLACE INTO scan_results
           (run_id,ordinal,code,kind,name,category,action,patterns_json,payload_json)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (run_id, int(ordinal), str(payload.get("code", "")), str(payload.get("kind", "")),
         str(payload.get("name") or payload.get("code") or ""),
         str(payload.get("category", "unclassified")), str(payload.get("action", "INSUFFICIENT_DATA")),
         json.dumps(patterns, ensure_ascii=False, allow_nan=False),
         json.dumps(payload, ensure_ascii=False, allow_nan=False)),
    )


def _is_legacy_scan_job(payload):
    if not isinstance(payload, dict) or payload.get("job_type") in ("fixed_backtest", "data_refresh"):
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
            f"SELECT payload_json FROM scan_results WHERE {where} ORDER BY {column} {sort_order.upper()},ordinal ASC LIMIT ? OFFSET ?",
            (*params, int(limit), int(offset)),
        ).fetchall()
    return [json.loads(row[0]) for row in rows], total
