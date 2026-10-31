from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import active_children, get_context
import asyncio
import threading
import time
from pathlib import Path

import pandas as pd

from caisen.market import InstrumentFile, discover_instruments
from caisen import api
from caisen.store import create_scan_run, get_scan_run, query_scan_results


def _make_market_root(root):
    stock_dir = root / "data" / "stocks"
    etf_dir = root / "data" / "etfs"
    stock_dir.mkdir(parents=True)
    etf_dir.mkdir(parents=True)
    dates = pd.date_range("2026-10-01", periods=8, freq="B")
    frame = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": [10.0 + index * 0.01 for index in range(8)],
        "high": [10.1 + index * 0.01 for index in range(8)],
        "low": [9.9 + index * 0.01 for index in range(8)],
        "close": [10.02 + index * 0.01 for index in range(8)],
        "volume": [1000 + index * 10 for index in range(8)],
        "name": ["进程测试标的"] * 8,
    })
    frame.to_csv(stock_dir / "000001.csv", index=False)
    (stock_dir / "000002.csv").write_text("date,close\n2026-10-01,10\n", encoding="utf-8")
    frame.to_csv(etf_dir / "510001.csv", index=False)


def _shutdown_probe(seconds):
    time.sleep(seconds)
    return "finished"


def _slow_scan_analysis(instrument, params, root):
    if int(instrument.code) in {0, 5, 9}:
        time.sleep(0.1)
    else:
        release = Path(root) / "release-workers"
        while not release.exists():
            time.sleep(0.05)
    with open(instrument.path + ".done", "w", encoding="utf-8") as marker:
        marker.write("done")
    row = {"code": instrument.code, "kind": instrument.kind, "action": "WAIT", "state": "ready",
           "rank": {"action_priority": 1, "confirmed_timeframes": 0, "reward_risk": 0,
                    "risk_pct": 1, "confirmation_ordinal": 0}}
    return row, [{"pattern_id": "pattern-" + instrument.code}]


def test_spawn_scan_worker_uses_explicit_root_and_matches_serial_results(tmp_path):
    root = tmp_path / "market"
    _make_market_root(root)
    instruments = discover_instruments(root)
    params = {"timeframes": ["1d"]}
    expected = [api._analyze_scan_instrument(item, params, str(root)) for item in instruments]

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        actual = list(pool.map(api._analyze_scan_instrument,
                               instruments,
                               [params] * len(instruments),
                               [str(root)] * len(instruments)))

    assert actual == expected


def test_scan_worker_persists_spawn_results_and_progress_from_explicit_root(tmp_path):
    root = tmp_path / "market"
    _make_market_root(root)
    params = {"kinds": ["stock", "etf"], "timeframes": ["1d"]}
    create_scan_run("spawn-scan", params, total=3)

    api._scan_worker("spawn-scan", params, root=str(root))

    run = get_scan_run("spawn-scan")
    rows, total = query_scan_results("spawn-scan", limit=10)
    assert run["status"] == "completed"
    assert run["scanned"] == 3 and run["progress"] == 1
    assert total == 3
    assert {row["code"] for row in rows} == {"000001", "000002", "510001"}
    assert api._jobs["spawn-scan"]["results"] == rows


def test_service_shutdown_cancels_queued_spawn_scan_work():
    executor = api._create_scan_executor(2)
    api._active_scan_executors[executor] = threading.Event()
    futures = [executor.submit(_shutdown_probe, 0.5) for _ in range(30)]

    async def stop_service():
        async with api._lifespan(api.app):
            pass

    asyncio.run(stop_service())
    assert any(future.cancelled() for future in futures)
    deadline = time.monotonic() + 3
    while active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not active_children()
    with api._scan_executors_lock:
        api._active_scan_executors.pop(executor, None)


def test_scan_submits_only_a_bounded_window_and_still_processes_every_instrument(monkeypatch):
    instruments = [InstrumentFile("%06d" % index, "stock", "%d.csv" % index) for index in range(40)]
    start_ten_workers = threading.Barrier(10)
    release_workers = threading.Event()
    analyzed = {}
    analyzed_lock = threading.Lock()
    pools = []
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    monkeypatch.setattr(api, "save_patterns", lambda rows: None)

    def analyze(instrument, params, root):
        start_ten_workers.wait(timeout=3)
        assert release_workers.wait(timeout=3)
        with analyzed_lock:
            analyzed[instrument.code] = analyzed.get(instrument.code, 0) + 1
        return ({"code": instrument.code, "kind": instrument.kind, "action": "WAIT",
                 "rank": {"action_priority": 1, "confirmed_timeframes": 0, "reward_risk": 0,
                          "risk_pct": 1, "confirmation_ordinal": 0}}, [])

    def make_pool(max_workers):
        pool = ThreadPoolExecutor(max_workers=max_workers)
        pools.append(pool)
        return pool

    monkeypatch.setattr(api, "_analyze_scan_instrument", analyze)
    worker = threading.Thread(target=api._scan_worker,
                              args=("bounded-submit", {"timeframes": ["1d"]}),
                              kwargs={"executor_factory": make_pool})
    worker.start()
    assert start_ten_workers.parties == 10
    deadline = time.monotonic() + 3
    while not analyzed and not pools and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pools
    pool = pools[0]
    deadline = time.monotonic() + 1
    while pool._work_queue.qsize() > 10 and time.monotonic() < deadline:
        time.sleep(0.01)
    queued_while_first_workers_block = pool._work_queue.qsize()
    release_workers.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert queued_while_first_workers_block <= 10
    assert len(analyzed) == 40
    assert set(analyzed.values()) == {1}
    assert api._jobs["bounded-submit"]["scanned"] == 40


def test_running_scan_shutdown_saves_finished_rows_and_marks_run_interrupted(tmp_path, monkeypatch):
    root = tmp_path / "market"
    stock_dir = root / "data" / "stocks"
    stock_dir.mkdir(parents=True)
    for index in range(30):
        (stock_dir / ("%06d.csv" % index)).write_text("test", encoding="utf-8")
    params = {"kinds": ["stock"], "timeframes": ["1d"]}
    create_scan_run("shutdown-scan", params, total=30)
    saved_patterns = []
    monkeypatch.setattr(api, "save_patterns", lambda patterns: saved_patterns.extend(patterns))
    worker = threading.Thread(target=api._scan_worker,
                              args=("shutdown-scan", params),
                              kwargs={"root": str(root), "analysis_fn": _slow_scan_analysis})
    worker.start()

    expected_done = {"000000", "000005", "000009"}
    deadline = time.monotonic() + 10
    while {path.name[:6] for path in stock_dir.glob("*.done")} != expected_done and time.monotonic() < deadline:
        time.sleep(0.05)
    assert {path.name[:6] for path in stock_dir.glob("*.done")} == expected_done
    time.sleep(0.1)

    async def stop_service():
        async with api._lifespan(api.app):
            pass

    asyncio.run(stop_service())
    (root / "release-workers").touch()
    worker.join(timeout=10)

    run = get_scan_run("shutdown-scan")
    rows, total = query_scan_results("shutdown-scan", limit=50)
    assert not worker.is_alive()
    assert run["status"] == "interrupted"
    assert 3 <= run["scanned"] < 30
    assert total == run["scanned"]
    codes = [row["code"] for row in rows]
    assert len(codes) == len(set(codes))
    assert rows == api._jobs["shutdown-scan"]["results"]
    assert saved_patterns
    assert api._jobs["shutdown-scan"]["status"] == "interrupted"
    deadline = time.monotonic() + 3
    while active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not active_children()


def test_interrupted_scan_keeps_sparse_completion_ordinals_without_duplicates(monkeypatch):
    instruments = [InstrumentFile("%06d" % index, "stock", "%d.csv" % index) for index in range(30)]
    executor_box = []
    monkeypatch.setattr(api, "discover_instruments", lambda root: instruments)
    monkeypatch.setattr(api, "save_patterns", lambda rows: None)

    class ControlledExecutor:
        def __init__(self, max_workers):
            self.futures = {}
            executor_box.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, instrument, params, root):
            future = Future()
            self.futures[instrument.code] = future
            return future

        def shutdown(self, wait=False, cancel_futures=False):
            if cancel_futures:
                for future in self.futures.values():
                    future.cancel()

    create_scan_run("sparse-shutdown", {"kinds": ["stock"], "timeframes": ["1d"]}, total=30)
    worker = threading.Thread(target=api._scan_worker,
                              args=("sparse-shutdown", {"kinds": ["stock"], "timeframes": ["1d"]}),
                              kwargs={"executor_factory": ControlledExecutor})
    worker.start()
    deadline = time.monotonic() + 2
    while (not executor_box or len(executor_box[0].futures) < 20) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(executor_box[0].futures) == 20
    with api._scan_executors_lock:
        stop_event = api._active_scan_executors[executor_box[0]]
    stop_event.set()
    for code in ("000000", "000005", "000009"):
        instrument = next(item for item in instruments if item.code == code)
        executor_box[0].futures[code].set_result((
            {"code": code, "kind": "stock", "action": "WAIT", "state": "ready",
             "rank": {"action_priority": 1, "confirmed_timeframes": 0, "reward_risk": 0,
                      "risk_pct": 1, "confirmation_ordinal": 0}},
            [{"pattern_id": "pattern-" + code}],
        ))
    executor_box[0].shutdown(wait=False, cancel_futures=True)
    worker.join(timeout=3)

    rows, total = query_scan_results("sparse-shutdown", limit=50)
    run = get_scan_run("sparse-shutdown")
    assert not worker.is_alive()
    assert run["status"] == "interrupted"
    assert run["scanned"] == total == 3
    assert [row["code"] for row in rows] == ["000000", "000005", "000009"]
    assert rows == api._jobs["sparse-shutdown"]["results"]


def test_shutdown_during_initial_submission_marks_scan_interrupted(tmp_path):
    root = tmp_path / "market"
    stock_dir = root / "data" / "stocks"
    stock_dir.mkdir(parents=True)
    (stock_dir / "000001.csv").write_text("test", encoding="utf-8")
    create_scan_run("initial-submit-shutdown", {"kinds": ["stock"]}, total=1)

    class StopDuringSubmit:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args):
            api._stop_scan_executors()
            raise RuntimeError("cannot schedule new futures after shutdown")

        def shutdown(self, wait=False, cancel_futures=False):
            pass

    api._scan_worker("initial-submit-shutdown", {"kinds": ["stock"]}, root=str(root),
                     executor_factory=StopDuringSubmit)

    run = get_scan_run("initial-submit-shutdown")
    assert run["status"] == "interrupted"
    assert api._jobs["initial-submit-shutdown"]["status"] == "interrupted"
