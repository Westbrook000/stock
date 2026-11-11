"""Fixed-universe daily recognition and serial broker replay."""

from __future__ import annotations

import html
import hashlib
import inspect
import json
import functools
import math
import multiprocessing
import os
import queue
import random
import re
import resource
import subprocess
import time
from contextlib import nullcontext
from threading import Lock
from zoneinfo import ZoneInfo
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from uuid import uuid4

import pandas as pd

from .fixed_backtest import (
    LOT_SIZE,
    STARTING_CASH,
    STRATEGY_VERSION,
    STOP_FRACTION,
    TARGET_FRACTION,
    calculate_metrics,
    simulate_fixed_trades,
)
from .market import ALIASES, discover_instruments
from .main_bottoms import MAIN_RULES
from .daily_incremental import ENGINE_VERSION as DAILY_ENGINE_VERSION
from .daily_incremental import BOUNDARY_FIRST_ENGINE_VERSION


UNIVERSE_ID = "daily_800_stock_200_etf_v1"
UNIVERSE_DATE = "2026-11-01"
UNIVERSE_SEED = 20261101
DEFAULT_DATE_TO = None
DEFAULT_RECOGNIZER = "boundary_first"


class _CachedSignalStream(list):
    """Preserve recognition audit events when signals pass through the cache."""

    def __init__(self, signals=(), instance_events=(), entry_events=()):
        super().__init__(signals)
        self.instance_events = list(instance_events)
        self.entry_events = list(entry_events)
RECOGNIZER_VERSION = DAILY_ENGINE_VERSION
ENGINE_VERSION = DAILY_ENGINE_VERSION
BENCHMARK_PATH = (Path(__file__).resolve().parents[1] / "reports" / "daily-research" /
                  "benchmark" / "benchmark-execution-v5-current-rules-36.json")
LEGACY_BENCHMARK_PATH = (Path(__file__).resolve().parents[1] / "reports" / "daily-research" /
                         "benchmark" / "benchmark-execution-v4.json")
BENCHMARK_SELECTION_VERSION = "same-current-rules-36-v1"
BENCHMARK_POLICY_SCOPE = "same-current-code-36-account-serial-vs-spawn8"
BENCHMARK_TARGET_SESSIONS = {"short": 220, "medium": 520, "long": 1000}
BENCHMARK_PROTOCOL_VERSION = 5
BENCHMARK_REPETITIONS = 3
BENCHMARK_ENGINE_REPETITIONS = 1
BENCHMARK_WORKER_COUNTS = (1, 2, 4, 6, 8, 10)
BENCHMARK_PROCESS_TREE_RSS_CAP = 12 * 1024 ** 3
_WORKER_PROGRESS_QUEUE = None


def _set_worker_progress_queue(progress_queue):
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def _worker_progress_callback(kind, code):
    def send(snapshot):
        if _WORKER_PROGRESS_QUEUE is None:
            return
        keys = ("phase", "bars_completed", "bars_total", "current_date", "candidate_count",
                "candidate_evaluations", "elapsed_seconds")
        message = {"kind": str(kind), "code": str(code)}
        message.update({key: snapshot.get(key) for key in keys if key in snapshot})
        try:
            _WORKER_PROGRESS_QUEUE.put_nowait(message)
        except (queue.Full, OSError, ValueError):
            pass
    return send


def create_universe(instruments, path, seed=UNIVERSE_SEED, sampled_on=UNIVERSE_DATE):
    """Freeze a reproducible 800-stock/200-ETF sample from local files."""
    path = Path(path)
    if path.exists():
        return _load_universe(path)
    by_kind = {
        kind: sorted({str(item.code) for item in instruments if item.kind == kind})
        for kind in ("stock", "etf")
    }
    if len(by_kind["stock"]) < 800 or len(by_kind["etf"]) < 200:
        raise ValueError("local inventory is insufficient for the fixed sample")
    rng = random.Random(int(seed))
    stocks = sorted(rng.sample(by_kind["stock"], 800))
    etfs = sorted(rng.sample(by_kind["etf"], 200))
    payload = {
        "id": UNIVERSE_ID,
        "date": str(sampled_on),
        "seed": int(seed),
        "source": "local data/stocks and data/etfs CSV inventory",
        "inventory_counts": {"stock": len(by_kind["stock"]), "etf": len(by_kind["etf"])},
        "counts": {"stock": len(stocks), "etf": len(etfs), "total": len(stocks) + len(etfs)},
        "stocks": stocks,
        "etfs": etfs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _load_universe(path, allow_test=False):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for kind in ("stocks", "etfs"):
        values = payload.get(kind)
        if not isinstance(values, list) or any(not isinstance(code, str) for code in values):
            raise ValueError("universe %s must be a list of symbol strings" % kind)
        if len(values) != len(set(values)):
            raise ValueError("universe %s contains duplicate symbols" % kind)
        if values != sorted(values):
            raise ValueError("universe %s must be sorted" % kind)
    counts = payload.get("counts", {})
    if not allow_test and (payload.get("id") != UNIVERSE_ID or
                           len(payload["stocks"]) != 800 or len(payload["etfs"]) != 200 or
                           (counts and counts != {"stock": 800, "etf": 200, "total": 1000})):
        raise ValueError("universe must be %s with exactly 800 stocks and 200 ETFs" % UNIVERSE_ID)
    return payload


def _client_date():
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _instrument_tick(root, instrument):
    registry = Path(os.environ.get("CAISEN_INSTRUMENTS", Path(root) / "caisen-system" / "config" / "instruments.json"))
    entry = _instrument_registry(str(registry)).get("%s:%s" % (instrument.kind, instrument.code), {})
    if entry.get("tick_size") is not None:
        return float(entry["tick_size"]), "registry"
    return 0.01, "default"


@functools.lru_cache(maxsize=8)
def _instrument_registry(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("instruments", {})
    except (OSError, ValueError, TypeError):
        return {}


def _current_execution_source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("caisen/daily_incremental.py", "caisen/main_bottoms.py",
             "caisen/rules/patterns.v2.json", "caisen/fixed_backtest.py",
             "caisen/research_config.py", "caisen/daily_research.py",
             "scripts/benchmark_current_rule_workers.py")
    hashes = {}
    for name in names:
        digest = hashlib.sha256()
        with (root / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def load_research_bars(instrument, root="."):
    """Read one local CSV directly; record stat revision without hash gates."""
    path = Path(instrument.path)
    before = path.stat()
    raw = pd.read_csv(path, encoding="utf-8-sig")
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError("market file changed while being read")
    names = {str(column).lstrip("\ufeff").lower(): column for column in raw.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        found = next((names[alias.lower()] for alias in aliases if alias.lower() in names), None)
        if found is not None:
            rename[found] = canonical
    bars = raw.rename(columns=rename).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))
    bars = bars[required]
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    for column in required[1:]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=required).sort_values("date", kind="mergesort")
    bars = bars.drop_duplicates("date", keep="last").reset_index(drop=True)
    tick_size, tick_source = _instrument_tick(root, instrument)
    bars.attrs["quality"] = {"tick_size": tick_size}
    data_revision = "mtime_ns:%d,size:%d" % (after.st_mtime_ns, after.st_size)
    return {"bars": bars, "data_revision": data_revision,
            "first_date": bars.date.min().date().isoformat() if not bars.empty else None,
            "last_date": bars.date.max().date().isoformat() if not bars.empty else None,
            "normalized_rows": len(bars), "tick_size": tick_size, "tick_size_source": tick_source,
            "source": "local_csv"}


def _validated_benchmark_config(expected_recognizer=None):
    if not BENCHMARK_PATH.exists():
        return None
    try:
        benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        recognizer = benchmark.get("recommended_recognizer")
        selected = int(benchmark.get("recommended_workers", 0))
        selected_run = benchmark.get("worker_counts", {}).get(str(selected), {})
        selected_peak = int(selected_run.get("process_tree_peak_bytes", 0))
        if (benchmark.get("status") != "complete" or benchmark.get("parity_passed") is not True
                or benchmark.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION
                or benchmark.get("policy_scope") != BENCHMARK_POLICY_SCOPE
                or benchmark.get("reference_count") != 36
                or benchmark.get("reference_mode") != "single_process_same_current_code"
                or benchmark.get("source_unchanged_during_run") is not True
                or benchmark.get("source_sha256") != _current_execution_source_hashes()
                or benchmark.get("source_sha256_after") != _current_execution_source_hashes()
                or benchmark.get("reference_ledger_signature_sha256") !=
                    benchmark.get("candidate_ledger_signature_sha256")
                or benchmark.get("accounting_checks_passed") is not True
                or benchmark.get("data_revisions_match_references") is not True
                or benchmark.get("performance_gate_passed") is not True
                or float(benchmark.get("speedup_vs_serial", 0)) <= 1.0
                or benchmark.get("universe_id") != UNIVERSE_ID
                or benchmark.get("versions") != _version_metadata()
                or recognizer not in ("v7", "boundary_first")
                or (expected_recognizer is not None and recognizer != expected_recognizer)
                or selected not in BENCHMARK_WORKER_COUNTS
                or selected != 8
                or selected_run.get("parity_passed") is not True
                or selected_run.get("accounting_checks_passed") is not True
                or selected_run.get("data_revisions_match_references") is not True
                or selected_run.get("execution_mode") != "spawn"
                or selected_run.get("account_count") != 36
                or selected_run.get("memory_status") != "pass"
                or selected_peak <= 0 or selected_peak > BENCHMARK_PROCESS_TREE_RSS_CAP
                or benchmark.get("process_tree_memory_cap_bytes") != BENCHMARK_PROCESS_TREE_RSS_CAP):
            return None
        return {"workers": selected, "recognizer": recognizer, "source": str(BENCHMARK_PATH),
                "payload": benchmark}
    except (OSError, ValueError, TypeError):
        return None


def _resolve_workers(workers, recognizer=None):
    if workers is not None:
        workers = int(workers)
        if workers not in BENCHMARK_WORKER_COUNTS:
            raise ValueError("workers must be 1, 2, 4, 6, 8 or 10")
        return workers, "explicit"
    config = _validated_benchmark_config(expected_recognizer=recognizer)
    if config:
        return config["workers"], config["source"]
    return 1, "default_ineligible_benchmark" if _has_benchmark_artifact() else "default_no_benchmark"


def _has_benchmark_artifact():
    return BENCHMARK_PATH.exists() or LEGACY_BENCHMARK_PATH.exists()


def _resolve_execution_config(workers=None, recognizer=None):
    if recognizer not in (None, "v7", "boundary_first"):
        raise ValueError("recognizer must be v7 or boundary_first")
    if workers is not None:
        selected, source = _resolve_workers(workers)
        return selected, source, recognizer or DEFAULT_RECOGNIZER
    if recognizer is not None:
        selected, source = _resolve_workers(None, recognizer=recognizer)
        return selected, source, recognizer
    config = _validated_benchmark_config()
    if config:
        return config["workers"], config["source"], config["recognizer"]
    source = "default_ineligible_benchmark" if _has_benchmark_artifact() else "default_no_benchmark"
    return 1, source, DEFAULT_RECOGNIZER


def _effective_rule_metadata():
    neckline = MAIN_RULES.get("neckline_tests", {})
    breakout = MAIN_RULES.get("breakout", {})
    return {
        "neckline_selection_rule_version": MAIN_RULES.get("neckline_selection_rule_version"),
        "main_pattern_rule_version": MAIN_RULES.get("main_pattern_rule_version"),
        "consumed_boundary_phase_rule": neckline.get("consumed_boundary_phase_rule"),
        "candidate_reversal_atr_multiple": float(neckline.get("candidate_reversal_atr_multiple", 2.0)),
        "candidate_reversal_scope": neckline.get("candidate_reversal_scope"),
        "minimum_independent_neckline_groups": int(neckline.get("minimum_independent_groups", 2)),
        "breakout_close_buffer_previous_atr_multiple": float(
            breakout.get("close_buffer_previous_atr_multiple", 1.0)),
        "breakout_minimum_buffer_ticks": int(breakout.get("minimum_buffer_ticks", 2)),
        "recognition_experiment": None,
    }


def resolve_daily_research_config(workers=None, recognizer=None, execution_mode="spawn"):
    """Return the exact production configuration used for a regular daily run."""
    selected_workers, worker_source, selected_recognizer = _resolve_execution_config(
        workers, recognizer)
    if execution_mode not in ("spawn", "in_process"):
        raise ValueError("execution_mode must be spawn or in_process")
    engine_version = (BOUNDARY_FIRST_ENGINE_VERSION if selected_recognizer == "boundary_first"
                      else ENGINE_VERSION)
    return {
        "workers": selected_workers,
        "worker_source": worker_source,
        "recognizer": selected_recognizer,
        "execution_mode": execution_mode,
        "strategy_version": STRATEGY_VERSION,
        "recognizer_version": RECOGNIZER_VERSION,
        "engine_version": engine_version,
        "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown")),
        "recognition_rules": _effective_rule_metadata(),
        "versions": _version_metadata(),
        "trading_rules": {"initial_cash_per_symbol": STARTING_CASH, "lot_size": LOT_SIZE,
                           "stop_fraction": STOP_FRACTION, "target_fraction": TARGET_FRACTION,
                           "commission": 0.0, "slippage": 0.0, "max_hold_sessions": None,
                           "end_of_period_close": True},
    }


def _select_benchmark_members(records, per_history=2):
    """Choose near-target histories while balancing stock and ETF members."""
    count = max(1, int(per_history))
    kind_order = {"stock": 0, "etf": 1}
    selected = []
    for band in ("short", "medium", "long"):
        target = BENCHMARK_TARGET_SESSIONS[band]
        eligible = [dict(item) for item in records if item.get("history_band") == band]
        eligible.sort(key=lambda item: (abs(int(item.get("sessions", 0)) - target),
                                        kind_order.get(item.get("kind"), 2),
                                        str(item.get("code", ""))))
        band_selected = []
        if count >= 2:
            per_kind = count // 2
            for kind in ("stock", "etf"):
                same_kind = [item for item in eligible if item.get("kind") == kind]
                band_selected.extend(same_kind[:per_kind])
        selected_keys = {(item.get("kind"), item.get("code")) for item in band_selected}
        if len(band_selected) < count:
            for item in eligible:
                key = (item.get("kind"), item.get("code"))
                if key in selected_keys:
                    continue
                band_selected.append(item)
                selected_keys.add(key)
                if len(band_selected) >= count:
                    break
        band_selected.sort(key=lambda item: (kind_order.get(item.get("kind"), 2),
                                             abs(int(item.get("sessions", 0)) - target),
                                             str(item.get("code", ""))))
        selected.extend(band_selected[:count])
    return selected


def _version_metadata():
    return {"strategy_version": STRATEGY_VERSION,
            "recognizer_version": RECOGNIZER_VERSION,
            "engine_version": ENGINE_VERSION,
            "boundary_first_engine_version": BOUNDARY_FIRST_ENGINE_VERSION,
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown")),
            "neckline_selection_rule_version": MAIN_RULES.get("neckline_selection_rule_version"),
            "main_pattern_rule_version": MAIN_RULES.get("main_pattern_rule_version"),
            "consumed_boundary_phase_rule": MAIN_RULES.get("neckline_tests", {}).get(
                "consumed_boundary_phase_rule"),
            "recognition_rules_sha256": hashlib.sha256(json.dumps(
                MAIN_RULES, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")).hexdigest()}


def _recognition_config_value(recognition_config):
    if recognition_config is None:
        return None
    cache_key = getattr(recognition_config, "cache_key", None)
    return cache_key() if callable(cache_key) else recognition_config


def _recognition_experiment_metadata(recognition_config):
    if recognition_config is None:
        return None
    config_value = _recognition_config_value(recognition_config)
    if not config_value:
        return None
    experiment_id = getattr(recognition_config, "experiment_id", None)
    if isinstance(recognition_config, dict):
        experiment_id = recognition_config.get("experiment_id", experiment_id)
        minimum_tests = recognition_config.get("minimum_neckline_tests")
        admission_types = recognition_config.get("admission_types")
    else:
        minimum_tests = getattr(recognition_config, "minimum_neckline_tests", None)
        admission_types = getattr(recognition_config, "admission_types", None)
    if experiment_id is None and isinstance(config_value, (tuple, list)) and config_value:
        experiment_id = config_value[0]
        if len(config_value) > 1:
            minimum_tests = config_value[1]
        if len(config_value) > 2:
            admission_types = config_value[2]
    if experiment_id is None:
        return None
    cache_key = json.loads(json.dumps(config_value, ensure_ascii=False, default=_json_default))
    if admission_types is None and isinstance(cache_key, (tuple, list)) and len(cache_key) > 2:
        admission_types = cache_key[2]
    result = {"experiment_id": str(experiment_id)}
    if minimum_tests is not None:
        result["minimum_neckline_tests"] = int(minimum_tests)
    if admission_types is not None:
        result["admission_types"] = sorted(str(value) for value in admission_types)
    result["cache_key"] = cache_key
    return result


def _stamp_recognition_experiment(account, recognition_config):
    experiment = _recognition_experiment_metadata(recognition_config)
    if experiment is None:
        return account
    existing = account.get("recognition_experiment")
    if existing is not None and existing != experiment:
        raise ValueError("account recognition experiment conflicts with the active config")
    account["recognition_experiment"] = experiment
    for signal in account.get("signals", []):
        if not isinstance(signal, dict):
            continue
        existing = signal.get("recognition_experiment")
        if existing is not None and existing != experiment:
            raise ValueError("signal recognition experiment conflicts with the active config")
        signal["recognition_experiment"] = experiment
    return account


def _signal_cache_identity(instrument, provenance, date_to, trade_start, recognizer,
                           recognition_config=None):
    """Return a stable key for one recognition result and its validation fields."""
    rules = json.dumps(MAIN_RULES, sort_keys=True, separators=(",", ":"), default=str)
    config_key = _recognition_config_value(recognition_config)
    identity = {
        "cache_schema": 1,
        "instrument": {"kind": str(instrument.kind), "code": str(instrument.code)},
        "source_path": str(Path(instrument.path).resolve()),
        "data_revision": provenance.get("data_revision"),
        "normalized_rows": provenance.get("normalized_rows"),
        "used_last_date": provenance.get("used_last_date"),
        "date_to": str(date_to),
        "trade_start": str(trade_start) if trade_start is not None else None,
        "tick_size": provenance.get("tick_size"),
        "tick_size_source": provenance.get("tick_size_source"),
        "recognizer": str(recognizer),
        "versions": {"recognizer_version": RECOGNIZER_VERSION,
                     "engine_version": (BOUNDARY_FIRST_ENGINE_VERSION if recognizer == "boundary_first"
                                        else ENGINE_VERSION),
                     "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown"))},
        "recognition_config": config_key,
        "rules_sha256": hashlib.sha256(rules.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), identity


def _read_signal_cache(path, expected_key):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (not isinstance(payload, dict) or payload.get("cache_schema") != 2
            or payload.get("key") != expected_key
            or not isinstance(payload.get("signals"), list)
            or not isinstance(payload.get("instance_events"), list)
            or not isinstance(payload.get("recognition_stats"), dict)):
        return None
    return payload


def _write_signal_cache(path, key, identity, signals, recognition_stats):
    instance_events = list(getattr(signals, "instance_events", []))
    entry_events = list(getattr(signals, "entry_events", []))
    payload = {"cache_schema": 2, "key": key, "identity": identity,
               "signals": list(signals), "instance_events": instance_events,
               "entry_events": entry_events, "recognition_stats": recognition_stats}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    _atomic_write(path, encoded)
    saved = json.loads(encoded)
    return _CachedSignalStream(saved["signals"], saved["instance_events"], saved["entry_events"])


def _rss_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _prepare_account(instrument, root, date_to, trade_start, progress_callback=None,
                     recognizer="v7", signal_cache_dir=None, recognition_config=None,
                     require_signal_cache=False):
    if require_signal_cache and signal_cache_dir is None:
        raise ValueError("signal_cache_dir is required when require_signal_cache is true")
    if recognition_config is None:
        from .research_config import current_recognition_experiment
        recognition_config = current_recognition_experiment()
    started = time.perf_counter()
    if progress_callback:
        progress_callback({"phase": "loading", "bars_completed": 0, "bars_total": None})
    loaded = load_research_bars(instrument, root=root)
    load_seconds = time.perf_counter() - started
    provenance = {
        "source": loaded["source"],
        "data_revision": loaded["data_revision"],
        "tick_size": loaded["tick_size"],
        "tick_size_source": loaded["tick_size_source"],
        "first_date": loaded["first_date"],
        "last_date": loaded["last_date"],
        "normalized_rows": int(loaded["normalized_rows"]),
        "warnings": [] if loaded["tick_size_source"] == "registry" else [
            "tick size unresolved; baseline 0.01 fallback used"],
    }
    bars = loaded["bars"].copy()
    if bars.empty:
        if progress_callback:
            progress_callback({"phase": "no_data", "bars_completed": 0, "bars_total": 0})
        return {"code": instrument.code, "kind": instrument.kind, "status": "no_data",
                "bars": bars, "signals": [], "instance_events": [], "entry_events": [], "provenance": provenance,
                "load_seconds": load_seconds, "recognition_seconds": 0.0,
                "recognition_stats": {}, "recognizer": recognizer,
                "recognition_config": recognition_config}
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    cutoff = pd.Timestamp(date_to)
    bars = bars.loc[bars["date"].notna() & (bars["date"].dt.normalize() <= cutoff)].copy()
    bars = bars.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    provenance["used_last_date"] = bars.date.max().date().isoformat() if not bars.empty else None
    if bars.empty:
        if progress_callback:
            progress_callback({"phase": "no_bars_before_cutoff", "bars_completed": 0, "bars_total": 0})
        return {"code": instrument.code, "kind": instrument.kind, "status": "no_bars_before_cutoff",
                "bars": bars, "signals": [], "instance_events": [], "entry_events": [], "provenance": provenance,
                "load_seconds": load_seconds, "recognition_seconds": 0.0,
                "recognition_stats": {}, "recognizer": recognizer,
                "recognition_config": recognition_config}
    from .daily_incremental import generate_daily_signals
    if progress_callback:
        progress_callback({"phase": "recognition", "bars_completed": 0, "bars_total": len(bars)})
    recognized = time.perf_counter()
    recognition_stats = {}
    cache_path = None
    cache_identity = None
    cache_key = None
    if signal_cache_dir is not None:
        cache_key, cache_identity = _signal_cache_identity(
            instrument, provenance, date_to, trade_start, recognizer, recognition_config)
        cache_path = Path(signal_cache_dir) / (cache_key + ".json")
    cached = _read_signal_cache(cache_path, cache_key) if cache_path is not None else None
    if cached is not None:
        signals = _CachedSignalStream(cached["signals"], cached["instance_events"],
                                      cached.get("entry_events", []))
        recognition_stats.update(cached["recognition_stats"])
        recognition_stats["cache_hit"] = True
    else:
        if require_signal_cache:
            raise RuntimeError("required signal cache miss for %s:%s" %
                               (instrument.kind, instrument.code))
        from .research_config import recognition_experiment
        config_context = recognition_experiment(recognition_config) if recognition_config is not None else nullcontext()
        with config_context:
            if recognizer == "v7":
                options = {"trade_start": trade_start, "progress_callback": progress_callback}
                if "performance" in inspect.signature(generate_daily_signals).parameters:
                    options["performance"] = recognition_stats
                signals = generate_daily_signals(bars, instrument.code, instrument.kind, **options)
            else:
                signals = generate_daily_signals(bars, instrument.code, instrument.kind, trade_start=trade_start,
                                                 progress_callback=progress_callback, recognizer=recognizer,
                                                 performance=recognition_stats)
    if cache_path is not None and cached is None:
        recognition_stats["cache_hit"] = False
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        signals = _write_signal_cache(cache_path, cache_key, cache_identity, signals, recognition_stats)
    recognition_seconds = time.perf_counter() - recognized
    return {"code": instrument.code, "kind": instrument.kind, "status": "ready",
            "bars": bars, "signals": signals,
            "instance_events": list(getattr(signals, "instance_events", [])),
            "entry_events": list(getattr(signals, "entry_events", [])), "provenance": provenance,
            "load_seconds": load_seconds, "recognition_seconds": recognition_seconds,
            "recognition_stats": recognition_stats, "recognizer": recognizer,
            "recognition_config": recognition_config}


def _finalize_account(account):
    account["metrics"] = calculate_metrics(account["trades"], STARTING_CASH,
                                            account["ending_equity"], account["daily_equity"])
    account.setdefault("recognizer", "v7")
    engine_version = account.get("recognition_stats", {}).get("engine_version")
    if not engine_version:
        engine_version = BOUNDARY_FIRST_ENGINE_VERSION if account["recognizer"] == "boundary_first" else ENGINE_VERSION
    account.update({"strategy_version": STRATEGY_VERSION, "recognizer_version": RECOGNIZER_VERSION,
                    "recognizer": account["recognizer"], "engine_version": engine_version,
                    "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown"))})
    checks = account.get("accounting_checks", {})
    account["accounting_checks"] = {key: checks.get(key) is True for key in
        ("cash_plus_position_equals_equity", "realized_pnl_matches_ending_equity")}
    failures = [key for key, valid in account["accounting_checks"].items() if valid is not True]
    if failures:
        account["status"] = "accounting_failed"
        account["error"] = "accounting invariant failed: " + ", ".join(failures)
    return account


def _account_from_prepared(prepared, code, kind):
    """Build the same complete account payload for serial and spawned workers."""
    account = {"code": str(code), "kind": str(kind), "status": prepared["status"],
               "recognizer": prepared.get("recognizer", "v7"),
               "starting_cash": STARTING_CASH, "ending_equity": STARTING_CASH,
               "signals": prepared["signals"], "trades": [], "daily_equity": [],
               "instance_events": list(prepared.get("instance_events", [])),
               "entry_events": list(prepared.get("entry_events", [])),
               "rejections": [], "order_events": [], "provenance": prepared["provenance"],
               "timings": {"load_seconds": prepared["load_seconds"],
                           "recognition_seconds": prepared["recognition_seconds"],
                           "backtrader_seconds": 0.0},
               "recognition_stats": prepared.get("recognition_stats", {}),
               "accounting_checks": {"cash_plus_position_equals_equity": True,
                                     "realized_pnl_matches_ending_equity": True}}
    _stamp_recognition_experiment(account, prepared.get("recognition_config"))
    if prepared["status"] == "ready":
        replay_started = time.perf_counter()
        replay = simulate_fixed_trades(prepared["bars"], prepared["signals"],
                                       cash=STARTING_CASH, lot_size=LOT_SIZE)
        account["timings"]["backtrader_seconds"] = time.perf_counter() - replay_started
        account.update({"status": "complete", "trades": replay["trades"],
                        "rejections": replay["rejections"], "daily_equity": replay["daily_equity"],
                        "ending_equity": replay["ending_equity"],
                        "order_events": replay["order_events"],
                        "accounting_checks": replay.get("accounting_checks", {})})
    return _finalize_account(account)


def _account_instance_metadata(account):
    from .pattern_instances import derive_instance_metadata
    return [derive_instance_metadata(event) for event in account.get("instance_events", [])]


def _account_summary_row(account, account_path, output_dir):
    row = {key: account.get(key) for key in ("code", "kind", "status", "provenance", "metrics", "timings",
        "accounting_checks", "error", "strategy_version", "recognizer_version", "engine_version",
        "source_rule_version", "recognition_stats", "recognizer")}
    state_counts = {}
    for event in account.get("instance_events", []):
        state = event.get("state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
    row.update({"signal_count": len(account["signals"]), "trade_count": len(account["trades"]),
                "run_id": account.get("run_id"),
                "pattern_instance_event_count": len(account.get("instance_events", [])),
                "pattern_instance_metadata_count": len(account.get("pattern_instance_metadata", [])),
                "entry_event_count": sum(bool(event.get("entry_eligible"))
                                          for event in account.get("instance_events", [])),
                "pattern_instance_state_counts": state_counts,
                "starting_cash": float(account["starting_cash"]), "ending_equity": float(account["ending_equity"]),
                "account_json": str(Path(account_path).resolve()),
                "account_json_relative": Path(account_path).resolve().relative_to(
                    Path(output_dir).resolve()).as_posix()})
    experiment = account.get("recognition_experiment")
    if experiment:
        row["recognition_experiment_id"] = experiment["experiment_id"]
        row["recognition_config"] = experiment
    return row


def _complete_account_task(instrument, root, date_to, trade_start, account_path, recognizer="v7",
                           signal_cache_dir=None, recognition_config=None,
                           require_signal_cache=False, run_id=None):
    """Run and atomically persist one account; return only its compact summary over IPC."""
    prepared = _prepare_account(instrument, root, date_to, trade_start,
                                progress_callback=_worker_progress_callback(instrument.kind, instrument.code),
                                recognizer=recognizer, signal_cache_dir=signal_cache_dir,
                                recognition_config=recognition_config,
                                require_signal_cache=require_signal_cache)
    account = _account_from_prepared(prepared, instrument.code, instrument.kind)
    if run_id:
        account["run_id"] = str(run_id)
        account["pattern_instance_metadata"] = _account_instance_metadata(account)
        from .store import save_pattern_instance_events, save_pattern_instance_metadata
        save_pattern_instance_events(run_id, instrument.code, instrument.kind,
                                     account.get("instance_events", []))
        save_pattern_instance_metadata(run_id, instrument.code, instrument.kind,
                                       account["pattern_instance_metadata"])
    _atomic_write(account_path, json.dumps(account, ensure_ascii=False, indent=2, default=_json_default))
    return _account_summary_row(account, account_path, Path(account_path).parent.parent)


class _aggregate_equity_curve:
    """Combine account curves with date change events in linear time."""

    @staticmethod
    def add_account(events, initial_cash, points):
        previous_equity = float(initial_cash)
        previous_invested = 0.0
        for point in points:
            day = str(point["date"])
            event = events.setdefault(day, [0.0, 0.0])
            equity = float(point["equity"])
            invested = float(point.get("invested_value", 0.0))
            event[0] += equity - previous_equity
            event[1] += invested - previous_invested
            previous_equity, previous_invested = equity, invested

    @staticmethod
    def build(events, starting_capital):
        equity, invested = float(starting_capital), 0.0
        curve = []
        if not events:
            return curve
        day, final_day = date.fromisoformat(min(events)), date.fromisoformat(max(events))
        while day <= final_day:
            day_key = day.isoformat()
            if day_key in events:
                equity_delta, invested_delta = events[day_key]
                equity += equity_delta
                invested += invested_delta
                curve.append({"date": day_key, "equity": equity, "invested_value": invested})
            day += timedelta(days=1)
        return curve


def _summarize(accounts, trades, capital, ending, curve):
    metrics = calculate_metrics(trades, capital, ending, curve)
    groups = {"kind": {}, "shape": {}}
    for kind in ("stock", "etf"):
        rows = [trade for trade in trades if trade["kind"] == kind]
        groups["kind"][kind] = _group_metrics(rows)
    shapes = defaultdict(list)
    distinct_patterns = set()
    for trade in trades:
        tags = trade.get("pattern_tags") or [
            {"pattern_id": pid, "pattern_type": ptype}
            for pid, ptype in zip(trade.get("pattern_ids", []), trade.get("pattern_types", []))
        ]
        types = sorted({str(tag.get("pattern_type", "unknown")) for tag in tags}) or ["unknown"]
        for shape in types:
            shapes[shape].append(trade)
        for tag in tags:
            distinct_patterns.add((trade["code"], str(tag.get("pattern_id", ""))))
    for shape, rows in sorted(shapes.items()):
        groups["shape"][shape] = _group_metrics(rows)
    summary = {
        **metrics,
        "account_count": len(accounts),
        "successful_account_count": sum(account.get("status") == "complete" for account in accounts),
        "failed_account_count": sum(account.get("status") not in ("complete", "no_data", "no_bars_before_cutoff") for account in accounts),
        "no_data_account_count": sum(account.get("status") in ("no_data", "no_bars_before_cutoff") for account in accounts),
        "signal_count": sum(int(account.get("signal_count", 0)) for account in accounts),
        "pattern_instance_event_count": sum(int(account.get("pattern_instance_event_count", 0)) for account in accounts),
        "entry_event_count": sum(int(account.get("entry_event_count", 0)) for account in accounts),
        "pattern_instance_state_counts": {
            state: sum(int((account.get("pattern_instance_state_counts") or {}).get(state, 0))
                       for account in accounts)
            for state in sorted({state for account in accounts
                                 for state in (account.get("pattern_instance_state_counts") or {})})},
        "distinct_traded_patterns": len(distinct_patterns),
        "multi_tag_trade_count": sum(len(set(trade.get("pattern_types", []))) > 1 for trade in trades),
        "starting_capital": capital,
        "ending_equity": ending,
        "net_pnl": ending - capital,
        "note": "Overall trade totals count each executed trade once; shape groups may include a multi-tag trade under each matching shape.",
    }
    return {"summary": summary, "groups": groups, "trades": trades, "daily_equity": curve}


def _group_metrics(rows):
    pnls = [float(row.get("pnl", 0)) for row in rows]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = -sum(pnl for pnl in pnls if pnl < 0)
    exits = defaultdict(int)
    for row in rows:
        exits[str(row.get("exit_reason", "unknown"))] += 1
    return {"trade_count": len(rows), "net_pnl": sum(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else None,
            "profit_factor": sum(wins) / losses if losses else None,
            "expectancy": sum(pnls) / len(pnls) if pnls else None,
            "average_holding_days": (sum(int(row.get("holding_days", 0)) for row in rows) / len(rows)
                                     if rows else None),
            "exit_reason_counts": dict(sorted(exits.items()))}


def _case_chart_svg(bars, pattern, event, code):
    """Render a small actual-OHLC case chart with frozen line and pivot anchors."""
    if bars is None or bars.empty:
        return None
    bars = bars.reset_index(drop=True)
    offset = int(pattern.get("chart_offset", 0) or 0)
    points = pattern.get("key_points") or {}
    anchors = []
    for key, value in points.items():
        if not re.fullmatch(r"pivot_\d+_index", key) or not isinstance(value, int):
            continue
        ordinal = int(re.search(r"\d+", key).group())
        date_value = points.get(key.replace("_index", "_date"))
        index = next((i for i, item in enumerate(bars.date)
                      if date_value and pd.Timestamp(item).date().isoformat() == str(date_value)), value + offset)
        if 0 <= index < len(bars):
            price_value = points.get("pivot_%d" % ordinal, points.get("pivot_%d_price" % ordinal))
            price = price_value.get("price") if isinstance(price_value, dict) else price_value
            try:
                anchors.append({"index": int(index), "price": float(price), "label": "P%d" % ordinal})
            except (TypeError, ValueError):
                continue
    if not anchors:
        return None
    event_index = next((i for i, item in enumerate(bars.date)
                        if pd.Timestamp(item).date().isoformat() == str(event.get("event_date"))), None)
    first = max(0, min(point["index"] for point in anchors) - 18)
    last = min(len(bars) - 1, max(max(point["index"] for point in anchors) + 20,
                                  event_index if event_index is not None else 0))
    if last - first > 100:
        first = max(0, last - 100)
    visible = bars.iloc[first:last + 1]
    low = float(visible.low.min()); high = float(visible.high.max())
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    spread = high - low or max(abs(high) * 0.02, 0.01)
    low -= spread * 0.08; high += spread * 0.08
    width, height, left, top = 760, 210, 28, 20
    plot_w, plot_h = width - 48, height - 50
    x = lambda index: left + (index - first) * plot_w / max(last - first, 1)
    y = lambda price: top + (high - price) * plot_h / (high - low)
    pitch = plot_w / max(len(visible), 1)
    candles = []
    for index, row in visible.iterrows():
        color = "#c84a46" if float(row.close) >= float(row.open) else "#168366"
        xx = x(index); yo, yc = y(float(row.open)), y(float(row.close))
        body_y = min(yo, yc); body_h = max(1.2, abs(yc - yo)); body_w = max(1.4, pitch * 0.56)
        candles.append(f'<line x1="{xx:.2f}" y1="{y(float(row.high)):.2f}" x2="{xx:.2f}" y2="{y(float(row.low)):.2f}" stroke="{color}"/><rect x="{xx-body_w/2:.2f}" y="{body_y:.2f}" width="{body_w:.2f}" height="{body_h:.2f}" fill="{color}"/>')
    geometry = event.get("frozen_geometry") or {}
    line = geometry.get("neckline") or pattern.get("neckline") or {}
    neckline = ""
    if line and line.get("slope") is not None and line.get("intercept") is not None:
        def line_y(index):
            return y(float(line["slope"]) * (index - offset) + float(line["intercept"]))
        neckline = f'<line x1="{x(first):.2f}" y1="{line_y(first):.2f}" x2="{x(last):.2f}" y2="{line_y(last):.2f}" stroke="#6756a5" stroke-width="2.5"/><text x="{left+4}" y="{max(12,line_y(first)-5):.2f}" fill="#6756a5" font-size="11">冻结边界</text>'
    pivot_svg = "".join(f'<circle cx="{x(point["index"]):.2f}" cy="{y(point["price"]):.2f}" r="4" fill="#e1a12c"/><text x="{x(point["index"])+4:.2f}" y="{y(point["price"])-6:.2f}" font-size="10" fill="#684a12">{point["label"]}</text>' for point in anchors)
    confirm = ""
    if event_index is not None:
        confirm = f'<line x1="{x(event_index):.2f}" y1="{top}" x2="{x(event_index):.2f}" y2="{top+plot_h}" stroke="#a53d62" stroke-dasharray="4 3"/><text x="{x(event_index)+3:.2f}" y="{top+12}" fill="#a53d62" font-size="10">{html.escape(str(event.get("state")))}</text>'
    start_date = visible.date.iloc[0].date().isoformat(); end_date = visible.date.iloc[-1].date().isoformat()
    return (f'<svg class="case-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(str(code))} K线、冻结边界和结构点">'
            f'<rect width="100%" height="100%" fill="#fbfcfd"/><line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#cbd5dc"/>'
            f'{"".join(candles)}{neckline}{pivot_svg}{confirm}<text x="{left}" y="{height-10}" font-size="10" fill="#61717d">{start_date} — {end_date}</text></svg>')


def _report_event_cases(payload):
    """Read a small, traceable set of account event timelines for the HTML report."""
    accounts = list(payload.get("account_summaries", []))
    by_key = {(row.get("kind"), str(row.get("code"))): row for row in accounts}
    selected = []
    for key in (("stock", "000538"), ("stock", "600108"), ("etf", "159718"), ("etf", "516630")):
        row = by_key.get(key)
        if row:
            selected.append(row)
    for row in sorted(accounts, key=lambda item: (-int(item.get("pattern_instance_event_count", 0)),
                                                   item.get("kind", ""), item.get("code", ""))):
        if row not in selected and int(row.get("pattern_instance_event_count", 0)):
            selected.append(row)
        if len(selected) >= 8:
            break
    cases = []
    root = Path(__file__).resolve().parents[2]
    instruments = {(item.kind, str(item.code)): item for item in discover_instruments(root)}
    bars_by_key = {}
    for row in selected[:8]:
        try:
            with open(row["account_json"], encoding="utf-8") as handle:
                account = json.load(handle)
        except (OSError, ValueError, KeyError):
            continue
        events = account.get("instance_events", [])
        metadata_by_event = {(str(item.get("instance_id")), int(item.get("event_seq", 0))): item
                             for item in account.get("pattern_instance_metadata", [])}
        timeline = []
        chart_svg = None
        chart_notice = None
        for event in events[:200]:
            pattern = event.get("pattern") or {}
            from .pattern_instances import derive_instance_metadata
            contract = metadata_by_event.get((str(event.get("instance_id")), int(event.get("event_seq", 0))))
            if contract is None:
                contract = derive_instance_metadata(event)
            if chart_svg is None and chart_notice is None and pattern.get("key_points"):
                key = (row.get("kind"), str(row.get("code")))
                try:
                    if key not in bars_by_key:
                        instrument = instruments.get(key)
                        loaded = load_research_bars(instrument, root=root) if instrument else None
                        frame = loaded["bars"] if loaded else pd.DataFrame()
                        cutoff = payload.get("date_to") or (payload.get("parameters") or {}).get("date_to")
                        if cutoff and not frame.empty:
                            frame = frame.loc[frame.date.dt.date <= date.fromisoformat(str(cutoff))].copy()
                        bars_by_key[key] = (frame, loaded.get("data_revision") if loaded else None)
                    frame, current_revision = bars_by_key[key]
                    expected_revision = (row.get("provenance") or {}).get("data_revision")
                    if expected_revision and current_revision and expected_revision == current_revision:
                        chart_svg = _case_chart_svg(frame, pattern, event, row.get("code"))
                        if chart_svg is None:
                            chart_notice = "账户行情修订一致，但该事件缺少可绘制的结构锚点。"
                    else:
                        chart_notice = ("行情版本不一致，未将当前 CSV 绘成回测证据。账户修订：%s；当前修订：%s。" %
                                        (expected_revision or "未记录", current_revision or "不可用"))
                except (OSError, ValueError, KeyError, TypeError):
                    chart_svg = None
                    chart_notice = "无法核对回测时点行情版本，已隐藏当前行情图。"
            evidence = {item.get("code"): item.get("value") for item in pattern.get("evidence", [])
                        if isinstance(item, dict)}
            selection = evidence.get("neckline_selection") or {}
            geometry = event.get("frozen_geometry") or {}
            timeline.append({"instance_id": event.get("instance_id"),
                             "event_id": event.get("event_id"),
                             "event_seq": event.get("event_seq"),
                             "event_date": event.get("event_date"),
                             "state": event.get("state"),
                             "entry_eligible": bool(event.get("entry_eligible")),
                             "pattern_type": pattern.get("pattern_type"),
                             "structure_key": contract.get("structure_key"),
                             "boundary_version_id": contract.get("boundary_version_id"),
                             "paired_at": contract.get("paired_at"),
                             "state_at": contract.get("state_at"),
                             "selection_reason": contract.get("selection_reason"),
                             "metadata_source": contract.get("metadata_source"),
                             "boundary_id": selection.get("selected_candidate_id"),
                             "frozen_line": geometry.get("neckline") or pattern.get("neckline")})
        cases.append({"code": row.get("code"), "kind": row.get("kind"),
                      "account_json_relative": row.get("account_json_relative"),
                      "events": timeline, "chart_svg": chart_svg,
                      "chart_notice": chart_notice})
    return cases


def _report_html(payload):
    data = dict(payload)
    data["event_cases"] = _report_event_cases(payload)
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=_json_default).replace("</", "<\\/")
    summary = payload["summary"]
    cards = "".join("<article><b>%s</b><span>%s</span></article>" %
                    (html.escape(str(summary.get(key, "—"))), html.escape(label))
                    for key, label in (("account_count", "账户数"),
                                       ("pattern_instance_event_count", "形态状态事件"),
                                       ("entry_event_count", "首次突破买入事件"),
                                       ("trade_count", "成交数"), ("net_pnl", "净盈亏"),
                                       ("expectancy", "净期望 / 平均单笔"),
                                       ("profit_factor", "盈利因子"),
                                       ("average_holding_days", "平均持有天数"),
                                       ("cumulative_return", "累计收益率"), ("maximum_drawdown", "最大回撤")))
    template = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定日线研究报告</title><style>
      body{font:15px/1.5 system-ui,sans-serif;max-width:1280px;margin:28px auto;padding:0 16px;color:#182230;background:#f5f7fa}h1{font-size:25px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}article,section{padding:14px;background:white;border:1px solid #dde3e9;border-radius:8px}article b{display:block;font-size:22px}section{margin-top:14px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:7px;border-bottom:1px solid #e7eaee;vertical-align:top}th{background:#f4f7f9;position:sticky;top:0}td{font-size:13px}.scroll{overflow:auto;max-height:560px}.muted{color:#62717c;font-size:13px}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}.meta div{padding:7px;background:#f6f9fa;border-radius:5px}.case{margin:10px 0;border:1px solid #dce3e9;border-radius:7px;padding:10px}.case summary{cursor:pointer;font-weight:700}.state{font-weight:700}.BUY{color:#176848}.invalidated{color:#a23a38}.focus{color:#58488f}.fail{color:#a23a38}.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.case-chart{display:block;width:min(100%,760px);height:auto;margin:10px 0;border:1px solid #e0e6eb;border-radius:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#176b87}</style></head><body><main>
      <h1>固定日线研究报告</h1><p>冻结股票/ETF样本，逐标的独立账户回放。实例事件保存了结构快照、配对冻结边界、首次确认和后续状态。</p>
      <div class="cards">__CARDS__</div><section><h2>版本、数据范围与运行方式</h2><div id="meta" class="meta"></div></section>
      <section><h2>股票 / ETF 与形态分组</h2><div id="groups" class="groups"></div></section>
      <section><h2>实例事件时间线（点击账户看完整事件）</h2><div id="cases"></div></section>
      <section><h2>账户明细与版本</h2><div id="accounts" class="scroll"></div></section>
      <section><h2>运行耗时、内存和失败明细</h2><div id="runtime"></div><div id="failures"></div></section>
      </main><script type="application/json" id="data">__DATA__</script><script>
      const d=JSON.parse(document.getElementById('data').textContent),esc=v=>String(v??'—').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c])),pct=v=>v==null?'—':(Number(v)*100).toFixed(2)+'%';
      const meta=d.universe||{},p=d.parameters||{},versions=d.versions||{};
      const im=d.instance_metadata||{};
      document.getElementById('meta').innerHTML=[['样本',`${esc(meta.id)} · 股票 ${esc(meta.stock_count)} / ETF ${esc(meta.etf_count)} · 账号 ${esc(meta.selected_account_count)}`],['数据截止',`${esc(p.date_to||d.date_to)}${p.trade_start?' · 交易起点 '+esc(p.trade_start):''}`],['识别引擎',`${esc(p.recognizer)} · ${esc(d.engine_version)} · 规则 ${esc(d.source_rule_version)}`],['策略 / 识别版本',`${esc(d.strategy_version)} / ${esc(d.recognizer_version)}`],['数据版本',esc(JSON.stringify(versions))],['事件元数据',`${esc(im.schema_version||'run完成后生成')} · ${esc(im.event_count||0)}条 · 原始schema ${esc(im.source_event_schema_version||'native')} · 派生 ${esc(im.derived_event_count||0)} / 原生 ${esc(im.native_event_count||0)} · 原始事件未改写`],['运行状态',`${esc(d.status)} · run ${esc(d.run_id)}`]].map(([k,v])=>`<div><b>${k}</b><br>${v}</div>`).join('');
      const exitCounts=m=>Object.entries(m.exit_reason_counts||{}).map(([reason,count])=>`${esc(reason)} ${esc(count)}`).join('、')||'—';
      const metric=(m)=>`交易 ${esc(m.trade_count)} · 净盈亏 ${esc(m.net_pnl)} · 净期望/平均单笔 ${esc(m.expectancy)} · 胜率 ${pct(m.win_rate)} · 盈利因子 ${esc(m.profit_factor)} · 平均持有 ${esc(m.average_holding_days)} 天 · 退出 ${exitCounts(m)}`;
      const kindNames={stock:'股票',etf:'ETF'},shapeNames={w_bottom:'W底',inverse_head_shoulders:'头肩底',triangle_bottom:'三角形底部',bull_flag:'上升旗形'};
      document.getElementById('groups').innerHTML=Object.entries(d.groups||{}).map(([group,values])=>`<div><h3>${group==='kind'?'标的类型':'形态类型'}</h3><table><thead><tr><th>分组</th><th>结果</th></tr></thead><tbody>${Object.entries(values||{}).map(([key,value])=>`<tr><td>${esc(group==='kind'?(kindNames[key]||key):(shapeNames[key]||key))}</td><td>${metric(value)}</td></tr>`).join('')}</tbody></table></div>`).join('');
      document.getElementById('cases').innerHTML=(d.event_cases||[]).map(c=>`<details class="case"><summary>${esc(c.code)} · ${esc(kindNames[c.kind]||c.kind)} · ${c.events.length} 事件 <a href="${esc(c.account_json_relative)}">账户完整记录</a></summary>${c.chart_svg||''}${c.chart_notice?`<p class="muted">${esc(c.chart_notice)}</p>`:''}<div class="scroll"><table><thead><tr><th>序号 / 日期 / event ID</th><th>实例 / 结构键</th><th>形态 / 状态</th><th>BUY</th><th>边界版本 / 配对时间</th><th>选择原因 / 元数据来源</th><th>冻结线</th></tr></thead><tbody>${c.events.map(e=>`<tr><td>${esc(e.event_seq)} · ${esc(e.event_date)}<div title="${esc(e.event_id)}">${esc(e.event_id)}</div></td><td>${esc(e.instance_id)}<div>${esc(e.structure_key)}</div></td><td>${esc(shapeNames[e.pattern_type]||e.pattern_type)} · <span class="state ${esc(e.state)}">${esc(e.state)}</span></td><td>${e.entry_eligible?'BUY':'—'}</td><td>${esc(e.boundary_version_id)}<div>${esc(e.paired_at)} → ${esc(e.state_at)}</div></td><td>${esc(e.selection_reason)}<div>${esc(e.metadata_source)}</div></td><td>${esc(JSON.stringify(e.frozen_line))}</td></tr>`).join('')}</tbody></table></div></details>`).join('')||'<p class="muted">当前没有可展示的账户事件记录。</p>';
      document.getElementById('accounts').innerHTML='<table><thead><tr><th>标的</th><th>类型</th><th>状态 / 失败原因</th><th>信号 / 交易</th><th>实例事件 / 首次买入</th><th>旁路元数据</th><th>识别引擎 / 规则</th><th>数据修订</th><th>账户事件和交易明细</th></tr></thead><tbody>'+d.account_summaries.map(a=>`<tr><td>${esc(a.code)}</td><td>${esc(kindNames[a.kind]||a.kind)}</td><td>${esc(a.status)}${a.error?'<div class="fail">'+esc(a.error)+'</div>':''}</td><td>${esc(a.signal_count)} / ${esc(a.trade_count)}</td><td>${esc(a.pattern_instance_event_count)} / ${esc(a.entry_event_count)}</td><td>${esc(a.pattern_instance_metadata_count)}</td><td>${esc(a.recognizer)}<div>${esc(a.engine_version)} · ${esc(a.source_rule_version)}</div></td><td>${esc(a.provenance?.data_revision)}</td><td><a href="${esc(a.account_json_relative)}">JSON</a></td></tr>`).join('')+'</tbody></table>';
      document.getElementById('runtime').innerHTML=`<div class="meta"><div>总耗时：${esc(d.timings?.total_seconds)} 秒</div><div>读数据：${esc(d.timings?.load_seconds)} 秒</div><div>识别：${esc(d.timings?.recognition_seconds)} 秒</div><div>回放：${esc(d.timings?.backtrader_seconds)} 秒</div><div>峰值内存：${esc(d.memory?.rss_peak_bytes)} bytes</div><div>完成账号：${esc(d.summary?.successful_account_count)} / ${esc(d.summary?.account_count)}</div><div>失败账号：${esc(d.summary?.failed_account_count)} · 无数据：${esc(d.summary?.no_data_account_count)}</div></div>`;
      document.getElementById('failures').innerHTML=(d.errors||[]).length?'<h3>失败明细</h3><table><thead><tr><th>标的</th><th>类型</th><th>原因</th></tr></thead><tbody>'+d.errors.map(e=>`<tr><td>${esc(e.code)}</td><td>${esc(kindNames[e.kind]||e.kind)}</td><td class="fail">${esc(e.error)}</td></tr>`).join('')+'</tbody></table>':'<p>没有失败账号。</p>';
      </script></body></html>"""
    return template.replace("__CARDS__", cards).replace("__DATA__", embedded)


def run_daily_research(universe_path, output_dir, date_to=DEFAULT_DATE_TO, trade_start=None,
                       workers=None, progress_callback=None, root=None, symbols=None,
                       allow_test_universe=False, recognizer=None, execution_mode=None,
                       dispatch_sessions=None, signal_cache_dir=None, recognition_config=None,
                       require_signal_cache=False, worker_source_override=None):
    """Recognize and replay isolated accounts with a bounded worker pool."""
    actual_execution_mode = execution_mode or ("in_process" if allow_test_universe else "spawn")
    effective_config = resolve_daily_research_config(
        workers=workers, recognizer=recognizer, execution_mode=actual_execution_mode)
    workers = effective_config["workers"]
    worker_source = worker_source_override or effective_config["worker_source"]
    recognizer = effective_config["recognizer"]
    if execution_mode is None and not allow_test_universe:
        execution_mode = actual_execution_mode
    if recognition_config is None:
        from .research_config import current_recognition_experiment
        recognition_config = current_recognition_experiment()
    if require_signal_cache and signal_cache_dir is None:
        raise ValueError("signal_cache_dir is required when require_signal_cache is true")
    universe = _load_universe(universe_path, allow_test=allow_test_universe)
    client_date = _client_date()
    date_to = min(pd.Timestamp(date_to or client_date).date().isoformat(), client_date)
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    found = {(item.kind, item.code): item for item in discover_instruments(root)}
    selected = [("stock", code) for code in universe["stocks"]] + [("etf", code) for code in universe["etfs"]]
    if symbols is not None:
        wanted = {(str(item[0]), str(item[1])) if isinstance(item, (tuple, list)) and len(item) == 2
                  else (None, str(item)) for item in symbols}
        selected = [entry for entry in selected if entry in wanted or (None, entry[1]) in wanted]
        selected_keys = set(selected)
        # Explicit benchmark selectors may include special symbols outside the frozen production sample.
        for kind, code in sorted((kind, code) for kind, code in wanted if kind is not None):
            if (kind, code) not in selected_keys:
                selected.append((kind, code))
                selected_keys.add((kind, code))
    items = [(kind, code, found.get((kind, code))) for kind, code in selected]
    total = len(items)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run-manifest.json"
    if manifest_path.exists():
        raise FileExistsError("research output already has a run manifest: %s" % manifest_path)
    account_dir = output_dir / "accounts"
    account_dir.mkdir(exist_ok=True)
    json_path, html_path = output_dir / "daily-research.json", output_dir / "daily-research.html"
    progress_path = output_dir / "progress.json"
    run_id = uuid4().hex
    selected_engine_version = (BOUNDARY_FIRST_ENGINE_VERSION if recognizer == "boundary_first"
                               else ENGINE_VERSION)
    metadata = {
        "run_id": run_id,
        "strategy_version": STRATEGY_VERSION,
        "recognizer_version": RECOGNIZER_VERSION,
        "engine_version": selected_engine_version,
        "versions": _version_metadata(),
        "recognizer": recognizer,
        "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown")),
        "recognition_rules": effective_config["recognition_rules"],
        "trading_rules": effective_config["trading_rules"],
        "universe": {"id": universe.get("id"), "date": universe.get("date"), "seed": universe.get("seed"),
                     "stock_count": len(universe["stocks"]), "etf_count": len(universe["etfs"]),
                     "selected_account_count": total},
        "parameters": {"date_to": date_to, "client_date": client_date, "trade_start": trade_start,
                       "workers": workers, "worker_source": worker_source,
                       "recognizer": recognizer,
                       "execution_mode": actual_execution_mode,
                       "signal_cache_dir": (str(Path(signal_cache_dir).resolve())
                                            if signal_cache_dir is not None else None),
                       "recognition_config": _recognition_config_value(recognition_config) or {},
                       "initial_cash": STARTING_CASH, "lot_size": LOT_SIZE,
                       "stop_pct": STOP_FRACTION, "target_pct": TARGET_FRACTION,
                       "end_of_period_close": True},
    }
    recognition_experiment = _recognition_experiment_metadata(recognition_config)
    if recognition_experiment is not None:
        metadata["recognition_experiment"] = recognition_experiment
    if require_signal_cache:
        metadata["parameters"]["require_signal_cache"] = True
    manifest = {**metadata, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(output_dir.resolve()), "account_directory": str(account_dir.resolve())}
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    progress_state = {"run_id": run_id, "status": "running", "stage": "recognition_and_replay",
                      "completed": 0, "total": total, "elapsed_seconds": 0.0,
                      "active_symbols": []}
    _atomic_write(progress_path, json.dumps(progress_state, ensure_ascii=False, indent=2))
    started = time.perf_counter()
    last_progress_write = time.monotonic()
    account_summaries, trades, errors = [], [], []
    equity_events = {}
    starting_capital = total * STARTING_CASH
    ending_equity = 0.0
    timings = {"load_seconds": 0.0, "recognition_seconds": 0.0, "backtrader_seconds": 0.0}
    rss_start = _rss_peak_bytes()
    completed = 0
    completed_symbols = set()
    worker_progress = {}
    worker_progress_lock = Lock()
    progress_queue = None

    def worker_progress_callback(kind, code):
        key = "%s:%s" % (kind, code)
        def update(snapshot):
            with worker_progress_lock:
                worker_progress[key] = {"kind": kind, "code": str(code), **dict(snapshot)}
        return update

    def drain_process_progress():
        if progress_queue is None:
            return
        while True:
            try:
                snapshot = progress_queue.get_nowait()
            except queue.Empty:
                return
            key = "%s:%s" % (snapshot.get("kind", ""), snapshot.get("code", ""))
            if key not in completed_symbols:
                with worker_progress_lock:
                    worker_progress[key] = dict(snapshot)

    def update_progress(symbol=None, force=False, status="running"):
        nonlocal last_progress_write
        drain_process_progress()
        now = time.monotonic()
        if not force and now - last_progress_write < 2.0:
            return
        progress_state.update(status=status, completed=completed, total=total,
                              elapsed_seconds=time.perf_counter() - started)
        if symbol:
            progress_state["symbol"] = symbol
        with worker_progress_lock:
            progress_state["active_symbols"] = sorted(
                (dict(item) for item in worker_progress.values()),
                key=lambda item: (item.get("kind", ""), item.get("code", "")))
        _atomic_write(progress_path, json.dumps(progress_state, ensure_ascii=False, indent=2))
        if progress_callback:
            progress_callback(dict(progress_state))
        last_progress_write = now

    def accept_summary(row):
        nonlocal completed
        account_summaries.append(row)
        code, kind = str(row["code"]), str(row["kind"])
        completed_symbols.add("%s:%s" % (kind, code))
        if row.get("error"):
            errors.append({"code": code, "kind": kind, "error": row["error"]})
        completed += 1
        with worker_progress_lock:
            worker_progress.pop("%s:%s" % (kind, code), None)
        update_progress(symbol=code)

    def persist_account(account):
        code, kind = str(account["code"]), str(account["kind"])
        safe_code = re.sub(r"[^A-Za-z0-9_-]", "_", code)
        account_path = account_dir / ("%s-%s.json" % (kind, safe_code))
        account["run_id"] = run_id
        account.setdefault("recognizer", recognizer)
        _stamp_recognition_experiment(account, recognition_config)
        _finalize_account(account)
        account["pattern_instance_metadata"] = _account_instance_metadata(account)
        from .store import save_pattern_instance_events, save_pattern_instance_metadata
        save_pattern_instance_events(run_id, code, kind, account.get("instance_events", []))
        save_pattern_instance_metadata(run_id, code, kind, account["pattern_instance_metadata"])
        _atomic_write(account_path, json.dumps(account, ensure_ascii=False, indent=2, default=_json_default))
        accept_summary(_account_summary_row(account, account_path, output_dir))

    def empty_account(code, kind, status, error=None, provenance=None, signals=None):
        return {"code": code, "kind": kind, "status": status,
                "starting_cash": STARTING_CASH, "ending_equity": STARTING_CASH,
                "signals": list(signals or []), "trades": [], "daily_equity": [],
                "instance_events": [], "entry_events": [],
                "rejections": [], "order_events": [], "provenance": provenance or {},
                "accounting_checks": {"cash_plus_position_equals_equity": True,
                                      "realized_pnl_matches_ending_equity": True},
                "error": error}

    max_inflight = max(1, workers * 2)
    if execution_mode not in (None, "spawn", "in_process"):
        raise ValueError("execution_mode must be spawn or in_process")
    # Test fixtures use patched module globals that intentionally do not cross a spawn boundary.
    process_mode = execution_mode == "spawn" or (execution_mode is None and not allow_test_universe)
    if execution_mode == "in_process":
        process_mode = False
    if process_mode:
        process_context = multiprocessing.get_context("spawn")
        progress_queue = process_context.Queue(maxsize=max(16, workers * 4))
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=process_context,
            initializer=_set_worker_progress_queue, initargs=(progress_queue,))
    else:
        executor = ThreadPoolExecutor(max_workers=workers,
                                      thread_name_prefix="daily-research")
    next_item = 0
    futures = {}
    # Larger files tend to contain longer histories, so they start first and keep all workers useful.
    def estimated_input_size(instrument):
        try:
            return Path(instrument.path).stat().st_size
        except OSError:
            return -1
    dispatch_sessions = dispatch_sessions or {}
    items.sort(key=lambda row: (int(dispatch_sessions.get((row[0], row[1]),
                    estimated_input_size(row[2]) if row[2] is not None else -1)), row[0], row[1]),
               reverse=True)
    try:
        while next_item < total or futures:
            while next_item < total and len(futures) < max_inflight:
                kind, code, instrument = items[next_item]
                next_item += 1
                if instrument is None:
                    persist_account(empty_account(code, kind, "missing_inventory_file",
                                                   "symbol is absent from current local inventory"))
                else:
                    callback = worker_progress_callback(kind, code)
                    callback({"phase": "running" if process_mode else "queued",
                              "bars_completed": 0, "bars_total": None})
                    if process_mode:
                        account_path = account_dir / ("%s-%s.json" % (kind, re.sub(r"[^A-Za-z0-9_-]", "_", code)))
                        future = executor.submit(_complete_account_task, instrument, str(root), date_to,
                                                 trade_start, str(account_path), recognizer,
                                                 str(signal_cache_dir) if signal_cache_dir is not None else None,
                                                 recognition_config, require_signal_cache, run_id)
                    else:
                        future = executor.submit(_prepare_account, instrument, root, date_to, trade_start,
                                                 callback, recognizer, signal_cache_dir, recognition_config,
                                                 require_signal_cache)
                    futures[future] = (kind, code)
            if not futures:
                continue
            done, _ = wait(tuple(futures), timeout=2.0, return_when=FIRST_COMPLETED)
            if not done:
                update_progress()
                continue
            for future in done:
                kind, code = futures.pop(future)
                prepared = None
                account = None
                try:
                    result = future.result()
                    if process_mode:
                        row = result
                        prepared = None
                        timings["load_seconds"] += row["timings"]["load_seconds"]
                        timings["recognition_seconds"] += row["timings"]["recognition_seconds"]
                        timings["backtrader_seconds"] += row["timings"]["backtrader_seconds"]
                        worker_progress_callback(kind, code)({"phase": "saving", "bars_completed": 0,
                                                              "bars_total": 0})
                        update_progress(symbol=code)
                        accept_summary(row)
                        continue
                    prepared = result
                    worker_progress_callback(kind, code)({
                        "phase": "backtrader" if prepared.get("status") == "ready" else "saving",
                        "bars_completed": len(prepared.get("bars", [])),
                        "bars_total": len(prepared.get("bars", [])),
                    })
                    update_progress(symbol=code)
                    timings["load_seconds"] += prepared["load_seconds"]
                    timings["recognition_seconds"] += prepared["recognition_seconds"]
                    account = _account_from_prepared(prepared, code, kind)
                    timings["backtrader_seconds"] += account["timings"]["backtrader_seconds"]
                except Exception as exc:
                    account = empty_account(code, kind, "failed",
                        "%s: %s" % (type(exc).__name__, exc),
                        provenance=(prepared or {}).get("provenance", {}),
                        signals=(prepared or {}).get("signals", []))
                persist_account(account)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    progress_state["stage"] = "reporting"
    update_progress(force=True, status="reporting")
    if progress_queue is not None:
        progress_queue.close()
        progress_queue.join_thread()
    account_summaries.sort(key=lambda account: (0 if account["kind"] == "stock" else 1, account["code"]))
    for summary in account_summaries:
        with open(summary["account_json"], encoding="utf-8") as handle:
            account = json.load(handle)
        _aggregate_equity_curve.add_account(equity_events, STARTING_CASH, account["daily_equity"])
        trades.extend(dict(trade, code=account["code"], kind=account["kind"])
                      for trade in account["trades"])
    trades.sort(key=_trade_sort_key)
    ending_equity = math.fsum(float(account["ending_equity"]) for account in account_summaries)
    equity_curve = _aggregate_equity_curve.build(equity_events, starting_capital)
    report = _summarize(account_summaries, trades, starting_capital, ending_equity, equity_curve)
    report.update(metadata)
    report.update({
        "date_to": date_to,
        "status": "partial_failed" if errors else "complete",
        "account_summaries": account_summaries,
        "account_files": [account["account_json"] for account in account_summaries],
        "timings": {**timings, "total_seconds": time.perf_counter() - started},
        "memory": {"rss_peak_bytes": _rss_peak_bytes(), "rss_peak_at_start_bytes": rss_start},
        "errors": errors,
    })
    report["artifacts"] = {"json_path": str(json_path.resolve()), "html_path": str(html_path.resolve()),
                           "manifest_path": str(manifest_path.resolve()), "progress_path": str(progress_path.resolve()),
                           "account_directory": str(account_dir.resolve())}
    write_started = time.perf_counter()
    _atomic_write(html_path, _report_html(report))
    report["timings"]["write_seconds"] = time.perf_counter() - write_started
    _atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    manifest.update(status=report["status"], completed=completed, total=total,
                    finished_at=datetime.now(timezone.utc).isoformat(), artifacts=report["artifacts"],
                    summary=report["summary"])
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
    progress_state.update(status=report["status"], stage="report_written", completed=completed,
                          total=total, elapsed_seconds=time.perf_counter() - started,
                          report_path=str(json_path.resolve()), active_symbols=[])
    _atomic_write(progress_path, json.dumps(progress_state, ensure_ascii=False, indent=2))
    if progress_callback:
        progress_callback(dict(progress_state))
    return report


def benchmark_daily_research(universe_path, output_dir, root=None, date_to=DEFAULT_DATE_TO,
                             trade_start=None, per_history=2,
                             repetitions=BENCHMARK_REPETITIONS):
    """Benchmark the same short/medium/long history cohort with 1/2/4 workers."""
    universe = _load_universe(universe_path)
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    found = {(item.kind, item.code): item for item in discover_instruments(root)}
    candidates = [("stock", code) for code in universe["stocks"]] + [("etf", code) for code in universe["etfs"]]
    date_to = min(pd.Timestamp(date_to or _client_date()).date().isoformat(), _client_date())
    output_dir = Path(output_dir) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "cohort-v2.json"
    scan_started = time.perf_counter()
    if cohort_path.exists():
        cohort_doc = json.loads(cohort_path.read_text(encoding="utf-8"))
        if (cohort_doc.get("universe_id") != universe.get("id") or
                cohort_doc.get("selection_version") != BENCHMARK_SELECTION_VERSION):
            raise ValueError("saved benchmark cohort belongs to another universe")
        cohort = cohort_doc["symbols"]
        cohort_size = int(cohort_doc.get("per_history", per_history))
    else:
        records = []
        for kind, code in candidates:
            instrument = found.get((kind, code))
            if instrument is None:
                continue
            data = load_research_bars(instrument, root=root)
            bars = data["bars"]
            if bars.empty:
                continue
            dates = pd.to_datetime(bars["date"], errors="coerce").dropna()
            count = int((dates.dt.normalize() <= pd.Timestamp(date_to)).sum())
            band = "short" if count <= 260 else ("medium" if count <= 780 else "long")
            records.append({"kind": kind, "code": code, "history_band": band,
                            "sessions": count, "data_revision": data["data_revision"]})
        cohort = _select_benchmark_members(records, per_history=per_history)
        cohort_size = int(per_history)
        cohort_doc = {"universe_id": universe.get("id"), "per_history": int(per_history),
                      "selection_version": BENCHMARK_SELECTION_VERSION,
                      "target_sessions": BENCHMARK_TARGET_SESSIONS,
                      "history_bands_sessions": {"short": "1-260", "medium": "261-780", "long": "781+"},
                      "symbols": cohort}
        _atomic_write(cohort_path, json.dumps(cohort_doc, ensure_ascii=False, indent=2))
    missing_bands = [band for band in ("short", "medium", "long")
                     if not any(member["history_band"] == band for member in cohort)]
    benchmark_id = uuid4().hex
    codes = [(member["kind"], member["code"]) for member in cohort]
    repetitions = max(2, int(repetitions))
    runs = {worker_count: [] for worker_count in (1, 2, 4)}
    parity_by_repetition = {}
    for repetition in range(1, repetitions + 1):
        round_runs = {}
        for worker_count in (1, 2, 4):
            run = run_daily_research(
                universe_path,
                output_dir / benchmark_id / ("repeat-%02d" % repetition) / ("workers-%d" % worker_count),
                date_to=date_to, trade_start=trade_start, workers=worker_count,
                root=root, symbols=codes,
            )
            runs[worker_count].append(run)
            round_runs[worker_count] = run
        baseline = _result_signature(round_runs[1])
        parity_by_repetition[str(repetition)] = {
            str(worker_count): _result_signature(round_runs[worker_count]) == baseline
            for worker_count in (2, 4)
        }
    parity = {str(worker_count): all(
        checks[str(worker_count)] for checks in parity_by_repetition.values())
        for worker_count in (2, 4)}
    accounting_ok = all(
        account.get("status") in ("complete", "no_data", "no_bars_before_cutoff") and
        check.get("cash_plus_position_equals_equity") is True and
        check.get("realized_pnl_matches_ending_equity") is True
        for worker_runs in runs.values() for run in worker_runs for account in run["account_summaries"]
        for check in [account.get("accounting_checks", {})]
    )
    frozen_revisions = {(member["kind"], member["code"]): member.get("data_revision") for member in cohort}
    revisions_match = all(
        account.get("provenance", {}).get("data_revision") == frozen_revisions.get((account["kind"], account["code"]))
        for worker_runs in runs.values() for run in worker_runs for account in run["account_summaries"]
    )
    run_summaries = {}
    per_band_seconds = {str(count): {} for count in (1, 2, 4)}
    for worker_count, worker_runs in runs.items():
        seconds_samples = [float(run["timings"]["total_seconds"]) for run in worker_runs]
        last_run = worker_runs[-1]
        run_summaries[str(worker_count)] = {
            "seconds": median(seconds_samples),
            "seconds_samples": seconds_samples,
            "signal_count": last_run["summary"]["signal_count"],
            "trade_count": last_run["summary"]["trade_count"],
            "account_count": last_run["summary"]["account_count"],
        }
        for band in ("short", "medium", "long"):
            codes_in_band = {(member["kind"], member["code"]) for member in cohort if member["history_band"] == band}
            band_samples = [sum(
                account.get("timings", {}).get("recognition_seconds", 0.0)
                + account.get("timings", {}).get("backtrader_seconds", 0.0)
                for account in run["account_summaries"] if (account["kind"], account["code"]) in codes_in_band
            ) for run in worker_runs]
            per_band_seconds[str(worker_count)][band] = median(band_samples)
        run_summaries[str(worker_count)]["per_history_seconds"] = per_band_seconds[str(worker_count)]
    totals = {str(count): run_summaries[str(count)]["seconds"] for count in (1, 2, 4)}
    parity_passed = not missing_bands and all(parity.values()) and accounting_ok and revisions_match
    result = {"status": "complete" if parity_passed else "insufficient_cohorts" if missing_bands else "parity_failed",
              "parity_passed": parity_passed, "universe_id": universe.get("id"), "date_to": date_to,
              "versions": _version_metadata(),
              "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
              "repetitions": repetitions,
              "history_bands_sessions": {"short": "1-260", "medium": "261-780", "long": "781+"},
              "per_history_cohort": cohort_size, "cohort": cohort,
              "history_scan_seconds": time.perf_counter() - scan_started,
              "runs": run_summaries, "median_seconds_by_worker_count": totals,
              "parity_by_worker_count": parity, "accounting_checks_passed": accounting_ok,
              "parity_by_repetition": parity_by_repetition,
              "data_revisions_match_frozen_cohort": revisions_match,
              "recommended_workers": _select_benchmark_workers(totals["1"], totals["2"], totals["4"])
              if parity_passed else None,
              "error": ("benchmark cohorts missing: " + ", ".join(missing_bands)) if missing_bands else
                       (None if parity_passed else "worker outputs, frozen revisions or accounting checks differ")}
    _atomic_write(output_dir / "benchmark.json", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _process_tree_rss_bytes(root_pid):
    """Return resident bytes for a process and every live descendant using ps(1)."""
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True,
                                         stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return 0
    parents, rss = {}, {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, parent, resident_kb = map(int, fields)
        except ValueError:
            continue
        parents[pid], rss[pid] = parent, resident_kb
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in descendants) * 1024


def _select_safe_worker_count(seconds_by_workers, peak_bytes_by_workers,
                             parity_by_workers, cap_bytes=BENCHMARK_PROCESS_TREE_RSS_CAP):
    safe = {int(workers): float(seconds) for workers, seconds in seconds_by_workers.items()
            if seconds is not None and parity_by_workers.get(str(workers)) is True
            and 0 < int(peak_bytes_by_workers.get(str(workers), 0)) <= int(cap_bytes)}
    if not safe:
        return None, None, []
    fastest = min(safe.values())
    eligible = sorted(workers for workers, seconds in safe.items() if seconds <= fastest * 1.05)
    return min(eligible), fastest, eligible


def _run_with_process_tree_peak(callable_run):
    from threading import Event, Thread
    stop = Event()
    peak = [0]

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], _process_tree_rss_bytes(os.getpid()))
            stop.wait(0.5)

    monitor = Thread(target=sample, name="process-tree-rss", daemon=True)
    monitor.start()
    try:
        result = callable_run()
    finally:
        peak[0] = max(peak[0], _process_tree_rss_bytes(os.getpid()))
        stop.set()
        monitor.join()
    return result, peak[0]


def _write_benchmark_progress(path, snapshot, progress_callback=None):
    _atomic_write(path, json.dumps(snapshot, ensure_ascii=False, indent=2, default=_json_default))
    if progress_callback:
        progress_callback(dict(snapshot))


def _account_ledger_payloads(report):
    """Full stable per-account signal and Backtrader ledger, with ephemeral refs normalized."""
    accounts = []
    for row in report.get("account_summaries", []):
        account = json.loads(Path(row["account_json"]).read_text(encoding="utf-8"))
        refs, events = {}, []
        for event in account.get("order_events", []):
            item = dict(event)
            if "order_ref" in item:
                raw = item["order_ref"]
                if raw not in refs:
                    refs[raw] = len(refs) + 1
                item["order_ref"] = refs[raw]
            events.append(item)
        accounts.append({"code": account.get("code"), "kind": account.get("kind"),
                         "status": account.get("status"), "signals": account.get("signals"),
                         "trades": account.get("trades"), "daily_equity": account.get("daily_equity"),
                         "rejections": account.get("rejections"), "order_events": events,
                         "ending_equity": account.get("ending_equity"),
                         "metrics": account.get("metrics"),
                         "accounting_checks": account.get("accounting_checks")})
    accounts.sort(key=lambda account: (str(account.get("kind", "")), str(account.get("code", ""))))
    return accounts


def _account_ledger_signature(report):
    return json.dumps(_account_ledger_payloads(report), sort_keys=True,
                      ensure_ascii=False, default=_json_default)


def _first_value_difference(reference, candidate, path="$"):
    if type(reference) is not type(candidate):
        return {"path": path, "reference": repr(reference)[:500], "candidate": repr(candidate)[:500]}
    if isinstance(reference, dict):
        for key in sorted(set(reference) | set(candidate)):
            if key not in reference or key not in candidate:
                return {"path": path + "." + str(key), "reference": "<missing>" if key not in reference else "<present>",
                        "candidate": "<missing>" if key not in candidate else "<present>"}
            result = _first_value_difference(reference[key], candidate[key], path + "." + str(key))
            if result:
                return result
    elif isinstance(reference, list):
        if len(reference) != len(candidate):
            return {"path": path + ".length", "reference": len(reference), "candidate": len(candidate)}
        for index, (left, right) in enumerate(zip(reference, candidate)):
            result = _first_value_difference(left, right, "%s[%d]" % (path, index))
            if result:
                return result
    elif reference != candidate:
        return {"path": path, "reference": repr(reference)[:500], "candidate": repr(candidate)[:500]}
    return None


def _first_account_ledger_divergence(reference_report, candidate_report):
    reference = {(item["kind"], item["code"]): item for item in _account_ledger_payloads(reference_report)}
    candidate = {(item["kind"], item["code"]): item for item in _account_ledger_payloads(candidate_report)}
    for key in sorted(set(reference) | set(candidate)):
        if key not in reference or key not in candidate:
            return {"kind": key[0], "code": key[1], "difference": "account_missing"}
        diff = _first_value_difference(reference[key], candidate[key])
        if diff:
            return {"kind": key[0], "code": key[1], "difference": diff}
    return None


def _account_reference_map(report):
    return {(row.get("kind"), row.get("code")): row for row in report.get("account_summaries", [])}


def _compare_account_ledgers(reference_report, candidate_report, expected_keys):
    reference = _account_reference_map(reference_report)
    candidate = _account_reference_map(candidate_report)
    expected = set(expected_keys)
    if set(reference) != expected or set(candidate) != expected:
        missing_reference = sorted(expected - set(reference))
        missing_candidate = sorted(expected - set(candidate))
        extra_reference = sorted(set(reference) - expected)
        extra_candidate = sorted(set(candidate) - expected)
        return False, {"difference": "account_set_mismatch", "missing_reference": missing_reference,
                       "missing_candidate": missing_candidate, "extra_reference": extra_reference,
                       "extra_candidate": extra_candidate}
    for key in sorted(expected):
        diff = _first_account_ledger_divergence(
            {"account_summaries": [reference[key]]}, {"account_summaries": [candidate[key]]})
        if diff:
            return False, diff
    return True, None


def _all_ledger_checks_pass(report, expected_count):
    rows = report.get("account_summaries", [])
    return expected_count > 0 and len(rows) == expected_count and all(
        row.get("status") in ("complete", "no_data", "no_bars_before_cutoff")
        and row.get("accounting_checks", {}).get("cash_plus_position_equals_equity") is True
        and row.get("accounting_checks", {}).get("realized_pnl_matches_ending_equity") is True
        for row in rows)


def _sum_recognition_stats(rows):
    fields = ("boundary_pairs_generated", "boundary_pairs_reused", "coverage_checks",
              "coverage_reused", "candidate_evaluations", "candidate_refreshes")
    return {field: sum(int(row.get("recognition_stats", {}).get(field, 0) or 0) for row in rows)
            for field in fields}


def benchmark_current_v8_pool(universe_path, output_dir, root=None, date_to="2026-10-30",
                              trade_start=None, cohort_path=None, cases_path=None,
                              archive_reference_path=None, snapshot_reference_path=None,
                              repetitions=1, progress_callback=None):
    """Benchmark boundary-first against 33 archived and 3 snapshot reference ledgers."""
    universe = _load_universe(universe_path)
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    base = Path(output_dir) / "benchmark"
    base.mkdir(parents=True, exist_ok=True)
    cohort_path = Path(cohort_path) if cohort_path else base / "cohort-v3.json"
    cases_path = Path(cases_path) if cases_path else base / "regression-cases-v3.json"
    archive_reference_path = (Path(archive_reference_path) if archive_reference_path else
        Path(__file__).resolve().parents[1] / "reports" / "daily-research" /
        "full-20261104-v8-thread2-boundary")
    if snapshot_reference_path is None:
        raise ValueError("snapshot reference report for 600108/159718/516630 is required")
    snapshot_reference_path = Path(snapshot_reference_path)
    cohort_doc, cases_doc = (json.loads(cohort_path.read_text(encoding="utf-8")),
                             json.loads(cases_path.read_text(encoding="utf-8")))
    if (cohort_doc.get("universe_id") != universe.get("id")
            or cohort_doc.get("selection_version") != "fixed-universe-stratified-v1"
            or cases_doc.get("universe_id") != universe.get("id")):
        raise ValueError("cohort and regression manifests must match the production universe")
    archive_report = json.loads((archive_reference_path / "daily-research.json").read_text(encoding="utf-8"))
    snapshot_report = json.loads((snapshot_reference_path / "daily-research.json").read_text(encoding="utf-8"))
    date_to = min(pd.Timestamp(date_to).date().isoformat(), _client_date())
    expected_version = {"strategy_version": STRATEGY_VERSION,
                        "recognizer_version": RECOGNIZER_VERSION,
                        "source_rule_version": str(MAIN_RULES.get("source_rule_version", "unknown"))}
    for label, report in (("archive", archive_report), ("snapshot", snapshot_report)):
        if report.get("status") != "complete" or report.get("date_to") != date_to:
            raise ValueError("%s reference must be complete at cutoff %s" % (label, date_to))
        if report.get("parameters", {}).get("trade_start") != trade_start:
            raise ValueError("%s reference trade_start does not match the benchmark" % label)
        if any(report.get(key) != value for key, value in expected_version.items()):
            raise ValueError("%s reference versions do not match the current strategy/rules" % label)
    cohort, cases = cohort_doc.get("symbols", []), cases_doc.get("symbols", [])
    if len(cohort) != 24 or len(cases) != 3:
        raise ValueError("benchmark requires the frozen 24-member cohort and three external cases")
    fixed = {("stock", code) for code in universe["stocks"]} | {("etf", code) for code in universe["etfs"]}
    cohort_keys = {(row["kind"], row["code"]) for row in cohort}
    case_keys = {(row["kind"], row["code"]) for row in cases}
    if len(cohort_keys) != 24 or not cohort_keys <= fixed:
        raise ValueError("cohort must contain 24 unique symbols from the fixed universe")
    if case_keys != {("stock", "600108"), ("etf", "159718"), ("etf", "516630")}:
        raise ValueError("external regression set must be 600108/159718/516630")
    frozen_revisions = {(row["kind"], row["code"]): row.get("data_revision")
                        for row in cohort + cases if row.get("data_revision")}
    archive_rows = _account_reference_map(archive_report)
    snapshot_rows = _account_reference_map(snapshot_report)
    top10 = sorted(archive_report.get("account_summaries", []),
                   key=lambda row: (-float(row.get("timings", {}).get("recognition_seconds", 0.0)),
                                    row.get("kind", ""), row.get("code", "")))[:10]
    top10_keys = {(row["kind"], row["code"]) for row in top10}
    expected_keys = cohort_keys | case_keys | top10_keys
    if len(expected_keys) != 36 or len(expected_keys - case_keys) != 33:
        raise ValueError("expected 36 unique records from cohort, top-10 and external cases")
    if not (expected_keys - case_keys) <= set(archive_rows) or not case_keys <= set(snapshot_rows):
        raise ValueError("reference reports do not cover the exact 36-symbol set")
    reference_rows = {}
    for key in sorted(expected_keys):
        row = (snapshot_rows if key in case_keys else archive_rows)[key]
        source_report = snapshot_report if key in case_keys else archive_report
        if (row.get("recognizer") != "boundary_first"
                or any(row.get(name) != value for name, value in expected_version.items())
                or row.get("engine_version") != source_report.get("engine_version")):
            raise ValueError("reference account semantic versions are inconsistent for %s:%s" % key)
        frozen_revision = frozen_revisions.get(key)
        if frozen_revision and row.get("provenance", {}).get("data_revision") != frozen_revision:
            raise ValueError("reference revision does not match its frozen manifest for %s:%s" % key)
        account_path = Path(row.get("account_json", ""))
        expected_dir = snapshot_reference_path if key in case_keys else archive_reference_path
        expected_path = expected_dir / "accounts" / ("%s-%s.json" %
            (key[0], re.sub(r"[^A-Za-z0-9_-]", "_", key[1])))
        if account_path.resolve() != expected_path.resolve() or not expected_path.is_file():
            raise ValueError("reference account path is not the expected frozen artifact: %s:%s" % key)
        if row.get("provenance", {}).get("used_last_date") != date_to:
            raise ValueError("reference account cutoff mismatch for %s:%s" % key)
        reference_rows[key] = dict(row)
    instruments = {(item.kind, item.code): item for item in discover_instruments(root)}
    missing, changed = [], []
    for key, row in reference_rows.items():
        instrument = instruments.get(key)
        if instrument is None:
            missing.append(key)
            continue
        stat = Path(instrument.path).stat()
        current = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
        frozen = row.get("provenance", {}).get("data_revision")
        if frozen != current:
            changed.append({"kind": key[0], "code": key[1], "frozen": frozen, "current": current})
    if missing or changed:
        raise ValueError("reference inputs changed: " + json.dumps({"missing": missing, "changed": changed}))
    reference_manifest = {"protocol": "daily-v8-2atr-36-reference-v1", "status": "validated",
        "date_to": date_to, "references": [{"kind": key[0], "code": key[1],
            "source": "snapshot" if key in case_keys else "archive",
            "account_path": reference_rows[key]["account_json"],
            "data_revision": reference_rows[key]["provenance"]["data_revision"],
            "used_last_date": reference_rows[key]["provenance"]["used_last_date"],
            "recognition_seconds": reference_rows[key].get("timings", {}).get("recognition_seconds")}
            for key in sorted(expected_keys)],
        "archive_versions": {key: archive_report.get(key) for key in
            ("strategy_version", "recognizer_version", "engine_version", "source_rule_version")},
        "snapshot_versions": {key: snapshot_report.get(key) for key in
            ("strategy_version", "recognizer_version", "engine_version", "source_rule_version")},
        "top10_by_archive_recognition_seconds": [{"kind": row["kind"], "code": row["code"],
            "recognition_seconds": row.get("timings", {}).get("recognition_seconds")} for row in top10]}
    reference_report = {"account_summaries": [reference_rows[key] for key in sorted(expected_keys)]}
    if not _all_ledger_checks_pass(reference_report, len(expected_keys)):
        raise ValueError("one or more frozen reference account ledgers failed their accounting checks")
    symbols = sorted(expected_keys)
    dispatch_sessions = {}
    cutoff = pd.Timestamp(date_to)
    for key in expected_keys:
        bars = load_research_bars(instruments[key], root=root)["bars"]
        cutoff_dates = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
        dispatch_sessions[key] = int((cutoff_dates <= cutoff).sum())
    reference_manifest["dispatch_sessions_by_account"] = {
        "%s:%s" % key: dispatch_sessions[key] for key in sorted(dispatch_sessions)}
    benchmark_id = uuid4().hex
    run_root = base / benchmark_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / "parity-reference-manifest.json"
    progress_path = run_root / "progress.json"
    def report_progress(snapshot):
        _write_benchmark_progress(progress_path,
            {"benchmark_id": benchmark_id, "status": "running", **snapshot}, progress_callback)
    _atomic_write(manifest_path, json.dumps(reference_manifest, ensure_ascii=False, indent=2))
    repetition_count = max(1, int(repetitions))
    worker_runs, details = {}, {}
    worker_parity, worker_divergence, peaks, wall_samples = {}, {}, {}, {}
    candidate_parity = True
    run_error = None
    reference_ledger_sha256 = __import__("hashlib").sha256(
        _account_ledger_signature(reference_report).encode("utf-8")).hexdigest()
    worker_counts_to_run = BENCHMARK_WORKER_COUNTS
    for workers in worker_counts_to_run:
        worker_runs[str(workers)] = []
        worker_parity[str(workers)] = []
        worker_divergence[str(workers)] = []
        peaks[str(workers)] = []
        wall_samples[str(workers)] = []
        for repetition in range(1, repetition_count + 1):
            report_progress({"stage": "worker_benchmark", "completed": sum(len(v) for v in worker_runs.values()),
                             "total": repetition_count * len(worker_counts_to_run),
                             "workers": workers, "repetition": repetition})
            label = "boundary-first-workers-%d-repeat-%02d" % (workers, repetition)
            started = time.perf_counter()
            try:
                run, peak = _run_with_process_tree_peak(lambda: run_daily_research(
                    universe_path, run_root / label, date_to=date_to, trade_start=trade_start,
                    workers=workers, root=root, symbols=symbols, recognizer="boundary_first",
                    execution_mode="spawn", dispatch_sessions=dispatch_sessions, signal_cache_dir=None,
                    recognition_config=None))
            except Exception as exc:
                run_error = "%s: %s" % (type(exc).__name__, exc)
                candidate_parity = False
                report_progress({"stage": "worker_failed", "completed": sum(len(v) for v in worker_runs.values()),
                                 "total": repetition_count * len(worker_counts_to_run),
                                 "workers": workers, "repetition": repetition, "error": run_error})
                break
            wall = time.perf_counter() - started
            parity, divergence = _compare_account_ledgers(reference_report, run, expected_keys)
            checks = _all_ledger_checks_pass(run, len(expected_keys))
            revisions_ok = all(row.get("provenance", {}).get("data_revision") ==
                reference_rows[(row["kind"], row["code"])].get("provenance", {}).get("data_revision")
                and row.get("provenance", {}).get("used_last_date") == date_to
                for row in run.get("account_summaries", [])) and len(run.get("account_summaries", [])) == len(expected_keys)
            worker_runs[str(workers)].append(run)
            worker_parity[str(workers)].append(parity)
            worker_divergence[str(workers)].append(divergence)
            peaks[str(workers)].append(peak)
            wall_samples[str(workers)].append(wall)
            details[str(workers)] = {"seconds_samples": wall_samples[str(workers)],
                "reported_total_seconds_samples": [float(item["timings"]["total_seconds"])
                                                     for item in worker_runs[str(workers)]],
                "sample_count": len(worker_runs[str(workers)]), "process_tree_peak_bytes_samples": peaks[str(workers)],
                "process_tree_peak_bytes": max(peaks[str(workers)], default=0),
                "memory_status": "unverified" if max(peaks[str(workers)], default=0) <= 0 else
                    "pass" if max(peaks[str(workers)]) <= BENCHMARK_PROCESS_TREE_RSS_CAP else "exceeds_limit",
                "ledger_parity_passed": all(worker_parity[str(workers)]),
                "parity_passed": all(worker_parity[str(workers)]),
                "first_divergence": next((item for item in worker_divergence[str(workers)] if item), None),
                "accounting_checks_passed": checks,
                "data_revisions_match_references": revisions_ok,
                "account_count": run.get("summary", {}).get("account_count", 0),
                "signal_count": run.get("summary", {}).get("signal_count", 0),
                "trade_count": run.get("summary", {}).get("trade_count", 0),
                "recognition_stats_totals": _sum_recognition_stats(run.get("account_summaries", [])),
                "recognition_stats_by_account": {"%s:%s" % (row["kind"], row["code"]):
                    row.get("recognition_stats", {}) for row in run.get("account_summaries", [])},
                "ledger_signature_sha256": __import__("hashlib").sha256(
                    _account_ledger_signature(run).encode("utf-8")).hexdigest(),
                "run_report_path": run.get("artifacts", {}).get("json_path")}
            report_progress({"stage": "worker_complete", "completed": sum(len(v) for v in worker_runs.values()),
                             "total": repetition_count * len(worker_counts_to_run),
                             "workers": workers, "repetition": repetition,
                             "ledger_parity_passed": parity, "memory_status": details[str(workers)]["memory_status"]})
            if workers == BENCHMARK_WORKER_COUNTS[0] and not (parity and checks and revisions_ok):
                candidate_parity = False
                break
        if not candidate_parity:
            break
    seconds = {key: median(value) if value else None for key, value in wall_samples.items()}
    parity_by_worker = {key: bool(worker_parity[key]) and all(worker_parity[key]) for key in worker_parity}
    accounting_by_worker = {key: bool(details.get(key, {}).get("accounting_checks_passed"))
                            for key in worker_parity}
    revisions_by_worker = {key: bool(details.get(key, {}).get("data_revisions_match_references"))
                           for key in worker_parity}
    eligible_peaks = {key: details[key]["process_tree_peak_bytes"] for key in details}
    selected, fastest, eligible = _select_safe_worker_count(seconds, eligible_peaks,
        {key: parity_by_worker[key] and accounting_by_worker[key] and revisions_by_worker[key]
         for key in worker_parity})
    all_runs_pass = all(parity_by_worker.get(str(w), False) and accounting_by_worker.get(str(w), False)
                        and revisions_by_worker.get(str(w), False)
                        for w in worker_counts_to_run)
    status = ("complete" if selected is not None and all_runs_pass else (
        "run_failed" if run_error else
        "parity_failed" if not all(parity_by_worker.values()) else
        "memory_unverified" if any(details.get(str(w), {}).get("memory_status") == "unverified"
                                    for w in worker_counts_to_run) else "incomplete"))
    result = {"status": status, "benchmark_id": benchmark_id,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "universe_id": universe["id"], "date_to": date_to,
        "versions": _version_metadata(), "reference_manifest_path": str(manifest_path.resolve()),
        "reference_count": len(expected_keys), "archive_reference_count": 33,
        "snapshot_reference_count": 3, "top10_selection": "archive recognition_seconds descending",
        "reference_ledger_signature_sha256": reference_ledger_sha256,
        "worker_counts": details, "worker_wall_seconds": seconds,
        "candidate_parity_passed": all(parity_by_worker.values()) and all_runs_pass,
        "repetitions_per_configuration": repetition_count, "fastest_worker_seconds": fastest,
        "fastest_plus_5_percent_workers": eligible, "recommended_recognizer": "boundary_first" if status == "complete" else None,
        "recommended_workers": selected if status == "complete" else None,
        "process_tree_memory_cap_bytes": BENCHMARK_PROCESS_TREE_RSS_CAP,
        "archive_versions": {key: archive_report.get(key) for key in
            ("strategy_version", "recognizer_version", "engine_version", "source_rule_version")},
        "data_revisions_match_references": all(revisions_by_worker.values()),
        "ledger_parity_passed": all(parity_by_worker.values()),
        "parity_passed": status == "complete" and all(parity_by_worker.values()),
        "accounting_checks_passed": all(accounting_by_worker.values()),
        "error": None if status == "complete" else run_error or
            "one or more parity, input, accounting or memory gates failed"}
    result_path = run_root / "benchmark-result.json"
    result["report_path"] = str(result_path.resolve())
    _atomic_write(result_path, json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    _write_benchmark_progress(progress_path, {"benchmark_id": benchmark_id, "status": status,
        "stage": "report_written", "completed": sum(len(value) for value in worker_runs.values()),
        "total": repetition_count * len(worker_counts_to_run), "report_path": str(result_path.resolve())},
        progress_callback)
    return result


def benchmark_engine_pool(universe_path, output_dir, root=None, date_to="2026-11-01",
                          trade_start=None, cohort_path=None, cases_path=None,
                          repetitions=BENCHMARK_ENGINE_REPETITIONS, progress_callback=None,
                          reference_path=None):
    """Check v7/boundary-first parity, then measure the exact frozen 24-member cohort."""
    universe = _load_universe(universe_path)
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    base = Path(output_dir) / "benchmark"
    base.mkdir(parents=True, exist_ok=True)
    cohort_path = Path(cohort_path) if cohort_path else base / "cohort-v3.json"
    cases_path = Path(cases_path) if cases_path else base / "regression-cases-v3.json"
    cohort_doc = json.loads(cohort_path.read_text(encoding="utf-8"))
    cases_doc = json.loads(cases_path.read_text(encoding="utf-8"))
    if (cohort_doc.get("universe_id") != universe.get("id") or
            cohort_doc.get("selection_version") != "fixed-universe-stratified-v1"):
        raise ValueError("fixed 24-member benchmark cohort does not match the production universe")
    if cases_doc.get("universe_id") != universe.get("id"):
        raise ValueError("regression cases do not match the production universe")
    cohort = cohort_doc.get("symbols", [])
    cases = cases_doc.get("symbols", [])
    if len(cohort) != 24 or len({(x["kind"], x["code"]) for x in cohort}) != 24 or len(cases) != 3:
        raise ValueError("benchmark protocol requires 24 cohort symbols and three regression cases")
    fixed_symbols = ({("stock", code) for code in universe["stocks"]} |
                     {("etf", code) for code in universe["etfs"]})
    cohort_symbols = {(x["kind"], x["code"]) for x in cohort}
    if not cohort_symbols <= fixed_symbols:
        raise ValueError("all 24 benchmark cohort members must come from the frozen 800/200 universe")
    band_counts = {band: sum(item.get("history_band") == band for item in cohort)
                   for band in ("short", "medium", "long")}
    longest = cohort_doc.get("rules", {}).get("longest_in_fixed_universe", {})
    if (band_counts != {"short": 8, "medium": 8, "long": 8}
            or any((kind, longest.get(kind, {}).get("code")) not in cohort_symbols
                   for kind in ("stock", "etf"))):
        raise ValueError("cohort must be stratified 8/8/8 and include the longest stock and ETF")
    required_cases = {("stock", "600108"), ("etf", "159718"), ("etf", "516630")}
    case_symbols = {(x["kind"], x["code"]) for x in cases}
    if case_symbols != required_cases or case_symbols & fixed_symbols or case_symbols & cohort_symbols:
        raise ValueError("regression cases must be 600108, 159718 and 516630")
    date_to = min(pd.Timestamp(date_to).date().isoformat(), _client_date())
    if date_to != cohort_doc.get("date_to") or date_to != cases_doc.get("date_to"):
        raise ValueError("benchmark cutoff must match the frozen cohort and regression cases")
    found = {(item.kind, item.code): item for item in discover_instruments(root)}
    frozen = {(item["kind"], item["code"]): item.get("data_revision")
              for item in cohort + cases}
    missing = [key for key in frozen if key not in found]
    changed = []
    for key, revision in frozen.items():
        instrument = found.get(key)
        if instrument is None:
            continue
        stat = Path(instrument.path).stat()
        current_revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
        if revision and current_revision != revision:
            changed.append({"kind": key[0], "code": key[1], "frozen": revision,
                            "current": current_revision})
    if missing or changed:
        raise ValueError("frozen benchmark inputs changed or disappeared: %s" %
                         json.dumps({"missing": missing, "changed": changed}, ensure_ascii=False))
    benchmark_id = uuid4().hex
    run_root = base / benchmark_id
    run_root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "progress.json"
    def report_progress(snapshot):
        _write_benchmark_progress(progress_path,
            {"benchmark_id": benchmark_id, "status": "running", **snapshot}, progress_callback)
    all_symbols = [(x["kind"], x["code"]) for x in cohort + cases]
    reference_symbols = [(x["kind"], x["code"]) for x in cohort]
    dispatch_sessions = {(x["kind"], x["code"]): int(x["sessions"]) for x in cohort + cases}
    report_progress({"stage": "v7_serial_reference", "completed": 0, "total": 2})
    reference_path = (Path(reference_path) if reference_path else
                      base / ("reference-v7-27-%s-rerun" % date_to.replace("-", "")))
    reference_report_path = reference_path / "daily-research.json"
    reused_reference = False
    if reference_report_path.exists():
        candidate = json.loads(reference_report_path.read_text(encoding="utf-8"))
        manifest_path = reference_path / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rows = candidate.get("account_summaries", [])
        revisions_ok = all(row.get("provenance", {}).get("data_revision") ==
                           frozen.get((row.get("kind"), row.get("code"))) for row in rows)
        expected, present = set(all_symbols), {(row.get("kind"), row.get("code")) for row in rows}
        expected_source_rule = str(MAIN_RULES.get("source_rule_version", "unknown"))
        paths_ok = all(
            Path(row.get("account_json", "")).resolve() ==
            (reference_path / "accounts" /
             ("%s-%s.json" % (row.get("kind"), re.sub(r"[^A-Za-z0-9_-]", "_", row.get("code", ""))))).resolve()
            and Path(row.get("account_json", "")).is_file()
            for row in rows)
        if (candidate.get("status") == "complete" and manifest.get("status") == "complete"
                and candidate.get("date_to") == date_to
                and candidate.get("recognizer") == "v7" and present == expected and revisions_ok
                and candidate.get("summary", {}).get("account_count") == len(expected)
                and paths_ok and candidate.get("parameters", {}).get("trade_start") == trade_start
                and candidate.get("strategy_version") == STRATEGY_VERSION
                and candidate.get("recognizer_version") == RECOGNIZER_VERSION
                and candidate.get("engine_version") == ENGINE_VERSION
                and candidate.get("source_rule_version") == expected_source_rule
                and manifest.get("parameters", {}).get("date_to") == date_to
                and manifest.get("parameters", {}).get("workers") == 1
                and manifest.get("parameters", {}).get("recognizer") == "v7"
                and manifest.get("parameters", {}).get("trade_start") == trade_start
                and manifest.get("strategy_version") == STRATEGY_VERSION
                and manifest.get("recognizer_version") == RECOGNIZER_VERSION
                and manifest.get("engine_version") == ENGINE_VERSION
                and manifest.get("source_rule_version") == expected_source_rule
                and Path(candidate.get("artifacts", {}).get("account_directory", "")).resolve() ==
                    (reference_path / "accounts").resolve()):
            v7 = candidate
            v7_peak = int(candidate.get("memory", {}).get("rss_peak_bytes", 0))
            reused_reference = True
        else:
            raise ValueError("existing v7 reference report does not match frozen cohort, revisions or cutoff")
    else:
        v7, v7_peak = _run_with_process_tree_peak(lambda: run_daily_research(
            universe_path, run_root / "v7-serial-27", date_to=date_to, trade_start=trade_start,
            workers=1, root=root, symbols=all_symbols, recognizer="v7", execution_mode="in_process",
            dispatch_sessions=dispatch_sessions))
    report_progress({"stage": "boundary_first_serial_parity", "completed": 1, "total": 2})
    boundary_serial, boundary_peak = _run_with_process_tree_peak(lambda: run_daily_research(
        universe_path, run_root / "boundary-first-serial-27", date_to=date_to, trade_start=trade_start,
        workers=1, root=root, symbols=all_symbols, recognizer="boundary_first", execution_mode="in_process",
        dispatch_sessions=dispatch_sessions))
    parity_27 = (_result_signature(v7) == _result_signature(boundary_serial))
    first_divergence_27 = None
    if not parity_27:
        first_divergence_27 = _first_account_ledger_divergence(v7, boundary_serial)
        if first_divergence_27 is None:
            diff = (_first_value_difference(v7.get("trades"), boundary_serial.get("trades"), "$.trades")
                    or _first_value_difference(v7.get("daily_equity"), boundary_serial.get("daily_equity"),
                                               "$.daily_equity")
                    or _first_value_difference(v7.get("summary"), boundary_serial.get("summary"), "$.summary"))
            first_divergence_27 = {"scope": "aggregate", "difference": diff}
    reference_ledger = _account_ledger_signature(v7)
    cohort_v7_ledger = _account_ledger_signature({
        **v7, "account_summaries": [row for row in v7["account_summaries"]
                                     if (row["kind"], row["code"]) in set(reference_symbols)]})
    v7_rows = {(row["kind"], row["code"]): row for row in v7["account_summaries"]}
    boundary_rows = {(row["kind"], row["code"]): row for row in boundary_serial["account_summaries"]}
    recognition_comparison = {}
    cache_fields = ("boundary_pairs_generated", "boundary_pairs_reused", "coverage_checks",
                    "coverage_reused", "candidate_evaluations", "candidate_refreshes")
    cache_totals = {field: 0 for field in cache_fields}
    for key in sorted(set(v7_rows) | set(boundary_rows)):
        reference_row, boundary_row = v7_rows.get(key, {}), boundary_rows.get(key, {})
        diff = _first_account_ledger_divergence(
            {"account_summaries": [reference_row]} if reference_row else {"account_summaries": []},
            {"account_summaries": [boundary_row]} if boundary_row else {"account_summaries": []})
        v7_seconds = float(reference_row.get("timings", {}).get("recognition_seconds", 0.0))
        boundary_seconds = float(boundary_row.get("timings", {}).get("recognition_seconds", 0.0))
        stats = boundary_row.get("recognition_stats", {})
        recognition_comparison["%s:%s" % key] = {
            "v7_recognition_seconds": v7_seconds,
            "boundary_first_recognition_seconds": boundary_seconds,
            "recognition_speedup": v7_seconds / boundary_seconds if boundary_seconds else None,
            "v7_signal_count": reference_row.get("signal_count"),
            "boundary_first_signal_count": boundary_row.get("signal_count"),
            "v7_trade_count": reference_row.get("trade_count"),
            "boundary_first_trade_count": boundary_row.get("trade_count"),
            "ledger_parity": diff is None,
            "first_difference": diff,
            "boundary_first_engine_stats": stats}
        for field in cache_fields:
            cache_totals[field] += int(stats.get(field, 0) or 0)
    # The same source revision and ending cutoff are checked again from emitted account provenance.
    revision_matches = all(
        row.get("provenance", {}).get("data_revision") == frozen.get((row["kind"], row["code"]))
        for row in v7["account_summaries"] + boundary_serial["account_summaries"])
    def ledger_checks_ok(report):
        return all(account.get("status") in ("complete", "no_data", "no_bars_before_cutoff")
                   and account.get("accounting_checks", {}).get("cash_plus_position_equals_equity") is True
                   and account.get("accounting_checks", {}).get("realized_pnl_matches_ending_equity") is True
                   for account in report["account_summaries"])
    accounting_ok = ledger_checks_ok(v7) and ledger_checks_ok(boundary_serial)
    repetition_count = max(1, int(repetitions))
    worker_runs = {workers: [] for workers in BENCHMARK_WORKER_COUNTS}
    worker_parity = {str(workers): [] for workers in BENCHMARK_WORKER_COUNTS}
    worker_divergences = {str(workers): [] for workers in BENCHMARK_WORKER_COUNTS}
    peak_by_workers = {str(workers): [] for workers in BENCHMARK_WORKER_COUNTS}
    if parity_27 and revision_matches and accounting_ok:
        for repetition in range(1, repetition_count + 1):
            for workers in BENCHMARK_WORKER_COUNTS:
                report_progress({"stage": "worker_benchmark", "completed": len(
                    sum(worker_parity.values(), [])), "total": repetition_count * len(BENCHMARK_WORKER_COUNTS),
                    "workers": workers, "repetition": repetition})
                run, peak = _run_with_process_tree_peak(lambda workers=workers, repetition=repetition:
                    run_daily_research(universe_path,
                        run_root / ("boundary-first-workers-%d-repeat-%02d" % (workers, repetition)),
                        date_to=date_to, trade_start=trade_start, workers=workers, root=root,
                        symbols=reference_symbols, recognizer="boundary_first", execution_mode="spawn",
                        dispatch_sessions=dispatch_sessions))
                worker_runs[workers].append(run)
                peak_by_workers[str(workers)].append(peak)
                parity = (_account_ledger_signature({
                    **run, "account_summaries": [row for row in run["account_summaries"]
                        if (row["kind"], row["code"]) in set(reference_symbols)]}) == cohort_v7_ledger)
                worker_parity[str(workers)].append(parity)
                if not parity:
                    worker_divergences[str(workers)].append(_first_account_ledger_divergence(
                        {**v7, "account_summaries": [row for row in v7["account_summaries"]
                            if (row["kind"], row["code"]) in set(reference_symbols)]},
                        {**run, "account_summaries": [row for row in run["account_summaries"]
                            if (row["kind"], row["code"]) in set(reference_symbols)]}))
                revision_matches = revision_matches and all(
                    row.get("provenance", {}).get("data_revision") == frozen.get((row["kind"], row["code"]))
                    for row in run["account_summaries"])
                accounting_ok = accounting_ok and ledger_checks_ok(run)
    seconds_by_workers = {}
    details = {}
    for workers, runs in worker_runs.items():
        samples = [float(run["timings"]["total_seconds"]) for run in runs]
        seconds_by_workers[str(workers)] = median(samples) if samples else None
        details[str(workers)] = {"seconds_samples": samples,
            "sample_count": len(samples),
            "median_seconds": seconds_by_workers[str(workers)],
            "process_tree_peak_bytes_samples": peak_by_workers[str(workers)],
            "process_tree_peak_bytes": max(peak_by_workers[str(workers)], default=0),
            "parity_passed": bool(worker_parity[str(workers)]) and all(worker_parity[str(workers)]),
            "first_divergence": next((item for item in worker_divergences[str(workers)] if item), None),
            "memory_status": "unverified" if max(peak_by_workers[str(workers)], default=0) <= 0 else
                "pass" if max(peak_by_workers[str(workers)]) <= BENCHMARK_PROCESS_TREE_RSS_CAP else "exceeds_limit",
            "account_count": (runs[-1]["summary"]["account_count"] if runs else 0),
            "signal_count": (runs[-1]["summary"]["signal_count"] if runs else 0),
            "trade_count": (runs[-1]["summary"]["trade_count"] if runs else 0),
            "recognition_stats": ({row["code"]: row.get("recognition_stats", {})
                                   for row in runs[-1]["account_summaries"]} if runs else {})}
    all_peaks = [v7_peak, boundary_peak] + [peak for peaks in peak_by_workers.values() for peak in peaks]
    reference_memory_ok = (v7_peak > 0 and boundary_peak > 0 and
                           max(v7_peak, boundary_peak) <= BENCHMARK_PROCESS_TREE_RSS_CAP)
    parity_ok = parity_27 and all(details[str(w)]["parity_passed"] for w in BENCHMARK_WORKER_COUNTS)
    recommended, fastest, eligible = _select_safe_worker_count(
        seconds_by_workers,
        {key: value["process_tree_peak_bytes"] for key, value in details.items()},
        {key: value["parity_passed"] for key, value in details.items()})
    if not (recommended is not None and reference_memory_ok and parity_ok and revision_matches and accounting_ok):
        recommended = None
    any_memory_unverified = v7_peak <= 0 or boundary_peak <= 0 or any(
        detail["memory_status"] == "unverified" for detail in details.values())
    result_status = ("complete" if recommended is not None else "parity_failed" if not parity_ok else
              "memory_unverified" if any_memory_unverified else
              "memory_limit_exceeded" if not reference_memory_ok or not eligible else "incomplete")
    result = {"status": result_status,
              "parity_passed": parity_ok, "parity_27_v7_serial_vs_boundary_first": parity_27,
              "first_divergence_27": first_divergence_27,
              "universe_id": universe["id"], "date_to": date_to, "versions": _version_metadata(),
              "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
              "benchmark_id": benchmark_id, "cohort_path": str(cohort_path.resolve()),
              "regression_cases_path": str(cases_path.resolve()),
              "reference_path": str(reference_path.resolve()), "reference_reused": reused_reference,
              "cohort_count": len(cohort), "regression_case_count": len(cases),
              "v7_serial_reference": {"total_seconds": v7["timings"]["total_seconds"],
                  "process_tree_peak_bytes": v7_peak, "signal_count": v7["summary"]["signal_count"],
                  "trade_count": v7["summary"]["trade_count"], "account_count": v7["summary"]["account_count"],
                  "ledger_signature_sha256": __import__("hashlib").sha256(reference_ledger.encode()).hexdigest()},
              "boundary_first_serial": {"total_seconds": boundary_serial["timings"]["total_seconds"],
                  "process_tree_peak_bytes": boundary_peak,
                  "signal_count": boundary_serial["summary"]["signal_count"],
                  "trade_count": boundary_serial["summary"]["trade_count"],
                  "account_count": boundary_serial["summary"]["account_count"]},
              "recognition_comparison_27": recognition_comparison,
              "boundary_first_engine_stats_totals_27": cache_totals,
              "worker_counts": details, "worker_median_seconds": seconds_by_workers,
              "repetitions": repetition_count,
              "fastest_worker_seconds": fastest, "fastest_plus_5_percent_workers": sorted(eligible),
              "recommended_recognizer": "boundary_first" if recommended is not None else None,
              "recommended_workers": recommended,
              "process_tree_peak_bytes": max(all_peaks, default=0),
              "process_tree_memory_cap_bytes": BENCHMARK_PROCESS_TREE_RSS_CAP,
              "process_tree_sample_interval_seconds": 0.5,
              "reference_process_tree_memory_verified": reference_memory_ok,
              "data_revisions_match_frozen_inputs": revision_matches,
              "accounting_checks_passed": accounting_ok,
              "error": None if recommended is not None else "parity, frozen-input, accounting or memory gate failed"}
    result_path = run_root / "benchmark-result.json"
    result["report_path"] = str(result_path.resolve())
    _atomic_write(result_path, json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    _write_benchmark_progress(progress_path, {"benchmark_id": benchmark_id,
        "status": result["status"], "stage": "report_written",
        "completed": repetition_count * len(BENCHMARK_WORKER_COUNTS) if worker_runs and parity_27 else 0,
        "total": repetition_count * len(BENCHMARK_WORKER_COUNTS),
        "report_path": str(result_path.resolve())}, progress_callback)
    if result["status"] == "complete":
        _atomic_write(base / "benchmark-execution-v3.json", json.dumps(result, ensure_ascii=False, indent=2,
                                                              default=_json_default))
    return result


def _select_benchmark_workers(seconds_1, seconds_2, seconds_4):
    timings = {1: float(seconds_1), 2: float(seconds_2), 4: float(seconds_4)}
    fastest = min(timings.values())
    if timings[1] <= fastest * 1.05:
        return 1
    if timings[2] <= timings[4] * 1.05:
        return 2
    return 4


def _result_signature(report):
    accounts = []
    for row in report.get("account_summaries", []):
        account = json.loads(Path(row["account_json"]).read_text(encoding="utf-8"))
        order_refs = {}
        normalized_events = []
        for event in account.get("order_events", []):
            stable_event = dict(event)
            if "order_ref" in stable_event:
                raw_ref = stable_event["order_ref"]
                if raw_ref not in order_refs:
                    order_refs[raw_ref] = len(order_refs) + 1
                stable_event["order_ref"] = order_refs[raw_ref]
            normalized_events.append(stable_event)
        accounts.append({key: account.get(key) for key in
                         ("code", "kind", "status", "signals", "trades", "daily_equity",
                          "rejections", "ending_equity", "metrics", "accounting_checks")})
        accounts[-1]["order_events"] = normalized_events
    return json.dumps({"accounts": accounts, "trades": report["trades"],
                       "daily_equity": report["daily_equity"], "summary": report["summary"]},
                      sort_keys=True, ensure_ascii=False, default=_json_default)


def _trade_sort_key(trade):
    return (str(trade.get("kind", "")), str(trade.get("code", "")),
            str(trade.get("entry_date", "")), str(trade.get("exit_date", "")),
            str(trade.get("signal_date", "")),
            tuple(str(value) for value in trade.get("pattern_ids", [])),
            float(trade.get("pnl", 0.0)))


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("not JSON serializable: %r" % (value,))


def _atomic_write(path, content):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
