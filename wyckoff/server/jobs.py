"""内存任务管理：扫描/回测/数据更新等异步作业。"""
from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional

JOBS: Dict[str, Dict] = {}
_LOCK = threading.Lock()


def submit(kind: str, fn: Callable[[], object], meta: Optional[Dict] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "status": "running",
           "created": datetime.now().isoformat(timespec="seconds"),
           "meta": meta or {}, "result": None, "error": None,
           "progress": 0.0}
    with _LOCK:
        JOBS[job_id] = job

    def wrap():
        try:
            result = fn()
            job["result"] = result
            job["status"] = "done"
            job["progress"] = 1.0
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = f"{exc}"[:400]
            traceback.print_exc()

    threading.Thread(target=wrap, daemon=True).start()
    return job_id


def get(job_id: str) -> Optional[Dict]:
    with _LOCK:
        return JOBS.get(job_id)


def list_jobs(kind: Optional[str] = None) -> list:
    with _LOCK:
        rows = list(JOBS.values())
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    return sorted(rows, key=lambda r: r["created"], reverse=True)[:50]
