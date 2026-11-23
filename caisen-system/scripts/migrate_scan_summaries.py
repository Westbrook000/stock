"""Build and validate a separate compact SQLite database; never modify the source."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caisen.scan_summary import scan_summary
from caisen.store import _is_legacy_scan_job, _save_scan_result, connect

MIB = 1024 * 1024


def update_digest(digest, ordinal, data):
    keys = ("code", "kind", "name", "category", "action", "latest_date", "latest_close", "patterns")
    digest.update(json.dumps([ordinal, [data.get(key) for key in keys]],
                             sort_keys=True, ensure_ascii=False).encode())


def decoded(row):
    payload = json.loads(row["payload_json"])
    payload.update({key:row[key] for key in ("code","kind","name","category","action")})
    payload["patterns"] = json.loads(row["patterns_json"])
    return payload


def migrate(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or not source.is_file():
        raise ValueError("Source must be an existing database distinct from the target")
    target.touch(exist_ok=False)
    report = {}
    src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        src.execute("BEGIN")
        for table in ("jobs", "scan_runs"):
            if src.execute(f"SELECT 1 FROM {table} WHERE status IN ('running','queued') LIMIT 1").fetchone():
                raise RuntimeError("An active task prevents migration")
        runs = list(src.execute("SELECT * FROM scan_runs ORDER BY created_at"))
        run_ids = {run["run_id"] for run in runs}
        with connect(str(target)) as dst:
            for table in ("scan_runs", "backtests", "migration_runs"):
                for row in src.execute(f"SELECT * FROM {table}"):
                    dst.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", tuple(row))
            for job in src.execute("SELECT * FROM jobs"):
                doc = json.loads(job["payload"])
                if job["job_id"] in run_ids:
                    doc.pop("results", None)
                elif _is_legacy_scan_job(doc):
                    raise RuntimeError("Found an unimported legacy scan; migrate only normalized history")
                dst.execute("INSERT INTO jobs(job_id,status,payload,created_at) VALUES(?,?,?,?)",
                            (job["job_id"], job["status"], json.dumps(doc, ensure_ascii=False), job["created_at"]))
            for run in runs:
                run_id = run["run_id"]
                before, after = hashlib.sha256(), hashlib.sha256()
                count = 0
                for row in src.execute("SELECT * FROM scan_results WHERE run_id=? ORDER BY ordinal", (run_id,)):
                    # Historical payload includes identities and patterns; read it only once.
                    raw = json.loads(row["payload_json"])
                    data = scan_summary(decoded(row) if raw.get("summary_version") == 1 else raw)
                    update_digest(before, row["ordinal"], data)
                    _save_scan_result(dst, run_id, row["ordinal"], data)
                    count += 1
                if run["status"] in {"completed", "partial_failed"} and count != run["scanned"]:
                    raise RuntimeError(f"{run_id}: saved result count {count} != scanned {run['scanned']}")
                check_count = 0
                for row in dst.execute("SELECT * FROM scan_results WHERE run_id=? ORDER BY ordinal", (run_id,)):
                    update_digest(after, row["ordinal"], decoded(row))
                    check_count += 1
                if count != check_count or before.digest() != after.digest():
                    raise RuntimeError(f"{run_id}: result verification failed")
                report[run_id] = {"rows":count, "summary_sha256":after.hexdigest()}
                print(run_id, report[run_id], flush=True)
            if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite integrity check failed")
            if dst.execute("PRAGMA foreign_key_check").fetchone():
                raise RuntimeError("SQLite foreign key check failed")
        report["old_bytes"] = source.stat().st_size
        report["new_bytes"] = target.stat().st_size
        return report
    finally:
        src.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target", help="A new, nonexistent file; production switch is separate")
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.target), indent=2))
