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
        CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY, pattern_id TEXT NOT NULL, verdict TEXT NOT NULL, corrected TEXT, note TEXT, reviewer_version TEXT DEFAULT 'manual-v1', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS refresh_job_items(
            job_id TEXT NOT NULL, instrument_key TEXT NOT NULL, code TEXT NOT NULL,
            kind TEXT NOT NULL, path TEXT NOT NULL, status TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(job_id,instrument_key));
        CREATE TABLE IF NOT EXISTS backtests(backtest_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
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
