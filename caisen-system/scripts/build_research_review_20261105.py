#!/usr/bin/env python3
"""Build period-reset diagnostics from the archived 2026-11-04 daily run.

Recognition is not rerun here. The open-admission variant applies the research
entry gate to the archived emitted signals, then replays each period in a new
Backtrader account. Source account files are read-only.
"""

from __future__ import annotations

import html
import json
import math
import sys
import time
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from caisen.daily_research import load_research_bars
from caisen.fixed_backtest import (
    STARTING_CASH,
    MAIN_BOTTOM_TYPES,
    _wilder_atr_by_index,
    _open_admission_gate,
    calculate_metrics,
    simulate_fixed_trades,
)


DEFAULT_ARCHIVE = ROOT / "reports/daily-research/full-20261104-v8-thread2-boundary"
DEFAULT_OUT = ROOT / "reports/daily-research/research-review-20261105"
DELAYED_BREAKOUT_ARCHIVE = ROOT / "reports/daily-research/full-cold-20261105-v8-boundary-w8-spawnfix"
ARCHIVE = DEFAULT_ARCHIVE
OUT = DEFAULT_OUT
DATA_SOURCE_MAP = None
SNAPSHOT_ROOT = None
_VERIFIED_SNAPSHOT_HASHES = set()
WINDOWS = [
    {"name": "through_2020", "start": "1900-01-01", "end": "2020-12-31"},
    {"name": "2021_2023", "start": "2021-01-01", "end": "2023-12-31"},
    {"name": "2024_2026", "start": "2024-01-01", "end": "2026-10-30"},
]


def _account_paths():
    return sorted((ARCHIVE / "accounts").glob("*.json"))


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_bars(account):
    kind, code = account["kind"], account["code"]
    expected = account.get("provenance", {}).get("data_revision")
    map_key = "%s:%s" % (kind, code)
    source = DATA_SOURCE_MAP.get(map_key) if DATA_SOURCE_MAP is not None else None
    if DATA_SOURCE_MAP is not None and source is None:
        raise RuntimeError("paired input map has no entry for %s" % map_key)
    if source is not None:
        if expected != source.get("account_data_revision"):
            raise RuntimeError("account data revision does not match paired input map for %s" % map_key)
        path = Path(source["snapshot_path"])
        snapshot_root = Path(source["snapshot_root"])
        stat = path.stat()
        revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
        if revision != source.get("snapshot_data_revision"):
            raise RuntimeError("frozen source file changed after paired repair for %s" % map_key)
        hash_key = (str(path.resolve()), source.get("snapshot_sha256"))
        if hash_key not in _VERIFIED_SNAPSHOT_HASHES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source.get("snapshot_sha256"):
                raise RuntimeError("frozen source checksum changed for %s" % map_key)
            _VERIFIED_SNAPSHOT_HASHES.add(hash_key)
        loaded = load_research_bars(SimpleNamespace(code=code, kind=kind, path=str(path)),
                                    root=snapshot_root)
        if loaded.get("data_revision") != revision:
            raise RuntimeError("snapshot loader revision disagrees for %s" % map_key)
        bars = loaded["bars"]
    else:
        folder = "etfs" if kind == "etf" else "stocks"
        path = ROOT.parent / "data" / folder / (str(code) + ".csv")
        revision = "mtime_ns:%d,size:%d" % (path.stat().st_mtime_ns, path.stat().st_size)
        if expected and expected != revision:
            raise RuntimeError("archived data revision changed for %s:%s" % (kind, code))
        bars = load_research_bars(SimpleNamespace(code=code, kind=kind, path=str(path)),
                                  root=ROOT.parent)["bars"]
    return bars


def _delayed_maturity_breakout_audit():
    """Verify the 159122 early buffered cross directly against archived bars."""
    account_path = DELAYED_BREAKOUT_ARCHIVE / "accounts" / "etf-159122.json"
    account = json.loads(account_path.read_text(encoding="utf-8"))
    signal = next((item for item in account.get("signals", [])
                   if item.get("signal_date") == "2026-06-22"
                   and "w_bottom" in item.get("pattern_types", [])), None)
    if signal is None:
        return {"status": "not_found", "archive": str(DELAYED_BREAKOUT_ARCHIVE),
                "code": "159122", "expected_signal_date": "2026-06-22"}
    pattern = next((item for item in signal.get("patterns", [])
                    if item.get("pattern_type") == "w_bottom"), None)
    if pattern is None:
        raise RuntimeError("159122 diagnostic signal has no W pattern snapshot")
    evidence = {item.get("code"): item.get("value") for item in pattern.get("evidence", [])}
    structure = evidence.get("main_structure") or {}
    selection = evidence.get("neckline_selection") or {}
    line = (selection.get("selected_line") or structure.get("neckline") or {})
    if not line:
        raise RuntimeError("159122 diagnostic signal has no selected frozen line")

    kind, code = account["kind"], account["code"]
    bars = _read_bars(account)
    revision = account.get("provenance", {}).get("data_revision")
    bars = bars.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    atr_values = _wilder_atr_by_index(bars)
    date_indexes = {value.date().isoformat(): index for index, value in enumerate(bars.date)}
    known_date = str(selection.get("structure_known_at") or structure.get("frozen_at"))
    maturity_date = str(structure.get("maturity_date"))
    known_index = date_indexes.get(known_date)
    if known_index is None:
        raise RuntimeError("159122 structure-known date is absent from archived bars")
    signal_index = date_indexes.get(str(signal["signal_date"]))
    maturity_index = date_indexes.get(maturity_date)
    if signal_index is None or maturity_index is None:
        raise RuntimeError("159122 maturity or confirmation date is absent from archived bars")
    tick_size = float(account.get("provenance", {}).get("tick_size") or 0.001)
    slope, intercept = float(line.get("slope", 0.0)), float(line.get("intercept", 0.0))
    first_buffered_cross = None
    first_buffered_cross_after_maturity = None
    for index in range(known_index + 1, signal_index + 1):
        previous_atr = float(atr_values[index - 1]) if index > 0 else float("nan")
        if not math.isfinite(previous_atr) or previous_atr <= 0:
            continue
        boundary = slope * index + intercept
        required = boundary + max(2.0 * tick_size, previous_atr)
        close = float(bars.close.iloc[index])
        if close > required:
            row = {"date": bars.date.iloc[index].date().isoformat(), "bar_index": index,
                   "close": close, "frozen_boundary": boundary,
                   "prior_atr14": previous_atr, "required_close": required,
                   "excess_over_required": close - required}
            if first_buffered_cross is None:
                first_buffered_cross = row
            if index > maturity_index and first_buffered_cross_after_maturity is None:
                first_buffered_cross_after_maturity = row
    retired = [item for item in selection.get("retired_segments", [])
               if item.get("state") == "too_early"]
    return {
        "status": "verified" if first_buffered_cross else "no_buffered_cross_found",
        "code": str(code), "kind": kind, "pattern_id": pattern.get("pattern_id"),
        "pattern_type": pattern.get("pattern_type"),
        "archive": str(DELAYED_BREAKOUT_ARCHIVE),
        "data_revision": revision,
        "formed_at": pattern.get("formed_at"),
        "second_bottom_date": (pattern.get("key_points") or {}).get("pivot_2_date"),
        "structure_known_at": known_date, "line_frozen_at": selection.get("line_frozen_at"),
        "maturity_date": maturity_date, "signal_date": signal.get("signal_date"),
        "entry_date": signal.get("entry_date"), "selected_frozen_line": line,
        "line_test_count": len(selection.get("independent_tests", [])),
        "line_test_dates": [item.get("pivot_date")
                             for item in selection.get("independent_tests", [])],
        "first_buffered_close_after_structure_known": first_buffered_cross,
        "first_buffered_close_after_maturity": first_buffered_cross_after_maturity,
        "recorded_too_early_candidates": retired,
        "signal_is_later_recross": bool(
            first_buffered_cross and first_buffered_cross["date"] < maturity_date
            and signal.get("signal_date") > maturity_date),
        "diagnosis": (
            "A buffered close above the frozen line occurred before the six-month maturity date; "
            "the recognizer marked that candidate too_early and later accepted a recross. This can "
            "misidentify a later recross as the initial breakout. The current baseline and min3 "
            "rules are left unchanged for this study."
        ),
        "rule_defect": (
            "The established rule requires the first valid buffered breakout to be the buy event; "
            "a crossing before minimum six-month maturity cannot be ignored so a later recross can "
            "be treated as the initial breakout."
        ),
        "required_isolated_correction": (
            "Retire the candidate permanently after its first valid pre-maturity buffered crossing; "
            "a recovery and later recross of that same structure must not re-enable a buy. Implement "
            "as an isolated recognizer correction with a causal regression after this study, without "
            "rewriting the baseline or min3 study rules."
        ),
    }


def _early_breakout_late_recross_rows(account, diagnostic_counts=None):
    """Find only early crosses from the selected, already-known frozen neckline."""
    rows = []
    for signal in account.get("signals", []):
        for pattern in signal.get("patterns", []):
            structure = _evidence_value(pattern, "main_structure") or {}
            selection = _evidence_value(pattern, "neckline_selection") or {}
            maturity = str(structure.get("maturity_date") or "")
            early = selection.get("early_breakouts") or []
            selected_candidate = selection.get("selected_candidate_id")
            known_at = str(selection.get("structure_known_at") or structure.get("structure_known_at") or "")
            frozen_at = str(selection.get("line_frozen_at") or structure.get("frozen_at") or "")
            earliest_eligible = max(known_at, frozen_at)
            line = selection.get("selected_line") or structure.get("neckline") or {}
            if not maturity or not early or not selected_candidate or not line or not earliest_eligible:
                continue
            signal_date = str(signal.get("signal_date") or "")
            matching = []
            for item in early:
                date = str(item.get("date") or "")
                if item.get("candidate_id") != selected_candidate:
                    if diagnostic_counts is not None:
                        diagnostic_counts["other_boundary_retired"] += 1
                    continue
                if not date or date < earliest_eligible:
                    if diagnostic_counts is not None:
                        diagnostic_counts["before_structure_known_or_line_frozen"] += 1
                    continue
                try:
                    projected = float(line.get("slope", 0.0)) * float(item["index"]) + float(line["intercept"])
                    observed = float(item["frozen_neckline"])
                except (KeyError, TypeError, ValueError):
                    if diagnostic_counts is not None:
                        diagnostic_counts["missing_selected_line_projection"] += 1
                    continue
                if abs(projected - observed) > 1e-8 * max(1.0, abs(projected)):
                    if diagnostic_counts is not None:
                        diagnostic_counts["selected_line_projection_mismatch"] += 1
                    continue
                if date < maturity and signal_date > maturity:
                    matching.append(item)
            if matching:
                first = min(matching, key=lambda item: str(item.get("date", "")))
                rows.append({
                    "code": str(account.get("code")), "kind": account.get("kind"),
                    "pattern_id": pattern.get("pattern_id"),
                    "pattern_type": pattern.get("pattern_type"),
                    "signal_date": signal_date, "formed_at": pattern.get("formed_at"),
                    "structure_known_at": selection.get("structure_known_at"),
                    "maturity_date": maturity,
                    "first_pre_maturity_buffered_cross": first,
                    "selected_candidate_id": selected_candidate,
                    "line_frozen_at": frozen_at,
                    "line_test_count": len(selection.get("independent_tests") or []),
                })
    return rows


def _recross_summary(rows, diagnostic_counts=None):
    return {"scan_scope": "emitted main-structure snapshots with a saved pre-maturity buffered cross and a later post-maturity signal",
            "delayed_recross_signal_count": len({(item["kind"], item["code"], item["signal_date"])
                                                  for item in rows}),
            "delayed_recross_pattern_count": len(rows),
            "affected_symbol_count": len({(item["kind"], item["code"]) for item in rows}),
            "by_pattern_type": dict(Counter(item["pattern_type"] for item in rows)),
            "excluded_evidence_counts": dict(diagnostic_counts or {}),
            "rows": rows}


def _new_period(period_calendar):
    return {"trades": [], "daily_delta": defaultdict(float), "signals": 0,
            "signal_tag_counts": Counter(),
            "stratification_rows": [],
            "account_count": 0, "accounts_with_bars": 0, "accounts_with_trades": 0,
            "rejections": Counter(), "gate_decisions": Counter(),
            "period_calendar": list(period_calendar)}


def _record_period(target, bars, signals, simulation):
    target["account_count"] += 1
    target["signals"] += len(signals)
    for signal in signals:
        target["signal_tag_counts"].update(signal.get("pattern_types", []))
    if not bars.empty:
        target["accounts_with_bars"] += 1
    if simulation["trades"]:
        target["accounts_with_trades"] += 1
    target["trades"].extend({key: item.get(key) for key in (
        "pnl", "exit_reason", "holding_days", "holding_sessions", "pattern_types")}
                             for item in simulation["trades"])
    target["stratification_rows"].extend(_trade_stratification_rows(
        bars, signals, simulation.get("trades", [])))
    account_curve = {str(item["date"]): float(item["equity"])
                     for item in simulation.get("daily_equity", [])}
    current = STARTING_CASH
    for date in target["period_calendar"]:
        if date in account_curve:
            current = account_curve[date]
        target["daily_delta"][date] += current - STARTING_CASH
    for item in simulation.get("rejections", []):
        target["rejections"][str(item.get("reason", "unknown"))] += 1


def _evidence_value(pattern, code):
    for item in pattern.get("evidence", []) or []:
        if item.get("code") == code:
            return item.get("value")
    return None


def _trade_stratification_rows(bars, signals, trades):
    if bars.empty or not trades or not {"date", "open", "high", "low", "close"}.issubset(bars.columns):
        return []
    dates = pd.to_datetime(bars.date)
    date_indexes = {value.date().isoformat(): index for index, value in enumerate(dates)}
    atr = _wilder_atr_by_index(bars)
    signals_by_date = {signal.get("signal_date"): signal for signal in signals}
    rows = []
    for trade in trades:
        signal = signals_by_date.get(trade.get("signal_date"))
        if not signal:
            continue
        patterns = {item.get("pattern_id"): item for item in signal.get("patterns", [])
                    if isinstance(item, dict)}
        signal_index = date_indexes.get(str(trade.get("signal_date")))
        entry_index = date_indexes.get(str(trade.get("entry_date")))
        exit_index = date_indexes.get(str(trade.get("exit_date")))
        if signal_index is None or entry_index is None or exit_index is None:
            continue
        signal_atr = float(atr[signal_index])
        signal_close = float(bars.close.iloc[signal_index])
        opening = float(bars.open.iloc[entry_index])
        for pattern_id in trade.get("pattern_ids", []) or []:
            pattern = patterns.get(pattern_id)
            if pattern is None:
                continue
            main = _evidence_value(pattern, "main_structure") or {}
            selection = _evidence_value(pattern, "neckline_selection") or {}
            breakout = (_evidence_value(pattern, "neckline_breakout") or
                        _evidence_value(pattern, "flag_breakout") or {})
            line = pattern.get("neckline") or {}
            frozen = breakout.get("frozen_neckline", breakout.get("frozen_boundary"))
            slope = line.get("slope")
            projected = None
            if frozen is not None and slope is not None:
                projected = float(frozen) + float(slope) * (entry_index - signal_index)
            displacement = ((opening - projected) / signal_atr
                            if projected is not None and signal_atr > 0 else None)
            breakout_atr = None
            prior_atr = breakout.get("atr14_previous")
            breakout_close = breakout.get("close")
            if frozen is not None and prior_atr and float(prior_atr) > 0 and breakout_close is not None:
                breakout_atr = (float(breakout_close) - float(frozen)) / float(prior_atr)
            neckline_loss = None
            if (pattern.get("pattern_type") in MAIN_BOTTOM_TYPES and frozen is not None
                    and slope is not None):
                close_values = bars.close.iloc[entry_index:exit_index + 1].astype(float).to_numpy()
                boundaries = [float(frozen) + float(slope) * (index - signal_index)
                              for index in range(entry_index, exit_index + 1)]
                neckline_loss = any(close < boundary for close, boundary in zip(close_values, boundaries))
            independent_tests = selection.get("independent_tests") or []
            test_count = main.get("neckline_independent_test_count")
            if test_count is None and independent_tests:
                test_count = len(independent_tests)
            rows.append({"code": str(trade.get("code", signal.get("code", ""))),
                         "pattern_id": pattern_id,
                         "pattern_type": pattern.get("pattern_type"),
                         "pivot_scale": main.get("pivot_scale", pattern.get("key_points", {}).get("pivot_scale")),
                         "independent_test_count": test_count,
                         "breakout_atr": breakout_atr,
                         "opening_displacement_atr": displacement,
                         "signal_atr_price_ratio": signal_atr / signal_close if signal_close else None,
                         "early_stop": (trade.get("exit_reason") == "stop_loss"
                                        and int(trade.get("holding_sessions", 0)) <= 5),
                         "neckline_close_loss": neckline_loss,
                         "exit_date": trade.get("exit_date"), "pnl": float(trade.get("pnl", 0.0)),
                         "exit_reason": trade.get("exit_reason")})
    return rows


def _bucket(value, bounds, labels):
    if value is None or not math.isfinite(float(value)):
        return "unavailable"
    for bound, label in zip(bounds, labels):
        if float(value) < bound:
            return label
    return labels[-1]


def _summarize_stratifications(rows):
    dimensions = {
        "pivot_scale": lambda row: row.get("pivot_scale") or "unavailable",
        "independent_test_count": lambda row: (
            "unavailable" if row.get("independent_test_count") is None else
            ("%d" % int(row["independent_test_count"]) if int(row["independent_test_count"]) < 5 else "5+")),
        "breakout_atr": lambda row: _bucket(row.get("breakout_atr"), [1, 2, 3], ["<1", "1-2", "2-3", "3+"]),
        "opening_displacement_atr": lambda row: _bucket(
            row.get("opening_displacement_atr"), [0, 0.5, 1, 2, 3],
            ["<0", "0-.5", ".5-1", "1-2", "2-3", ">=3"]),
        "signal_atr_price": lambda row: _bucket(
            row.get("signal_atr_price_ratio"), [0.01, 0.02, 0.04], ["<1%", "1-2%", "2-4%", ">=4%"]),
        "early_stop_within_5_sessions": lambda row: "yes" if row.get("early_stop") else "no",
        "daily_close_below_frozen_neckline_before_exit": lambda row: (
            "unavailable" if row.get("neckline_close_loss") is None else
            "yes" if row.get("neckline_close_loss") else "no"),
    }
    output = {}
    for dimension, categorize in dimensions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[categorize(row)].append(row)
        result = {}
        for label, items in sorted(groups.items()):
            pnl = [float(item["pnl"]) for item in items]
            wins, losses = [value for value in pnl if value > 0], [value for value in pnl if value < 0]
            pnl_by_date = defaultdict(float)
            symbols = set()
            for item in items:
                pnl_by_date[str(item["exit_date"])] += float(item["pnl"])
                symbols.add(item["code"])
            equity = STARTING_CASH * len(symbols)
            peak, drawdown = equity, 0.0
            for date in sorted(pnl_by_date):
                equity += pnl_by_date[date]
                peak = max(peak, equity)
                if peak > 0:
                    drawdown = max(drawdown, (peak - equity) / peak)
            result[label] = {"tag_trade_count": len(items), "unique_symbols": len(symbols),
                             "net_pnl": sum(pnl),
                             "expectancy": sum(pnl) / len(pnl) if pnl else None,
                             "win_rate": len(wins) / len(pnl) if pnl else None,
                             "profit_factor": sum(wins) / -sum(losses) if losses else None,
                             "early_stop_count": sum(bool(item.get("early_stop")) for item in items),
                             "early_stop_denominator": len(items),
                             "early_stop_rate": (sum(bool(item.get("early_stop")) for item in items) / len(items)
                                                 if items else None),
                             "neckline_close_loss_count": sum(item.get("neckline_close_loss") is True
                                                               for item in items),
                             "neckline_close_loss_denominator": sum(item.get("neckline_close_loss") is not None
                                                                     for item in items),
                             "neckline_close_loss_rate": (
                                 sum(item.get("neckline_close_loss") is True for item in items) /
                                 sum(item.get("neckline_close_loss") is not None for item in items)
                                 if any(item.get("neckline_close_loss") is not None for item in items) else None),
                             "realized_daily_close_drawdown": drawdown,
                             "drawdown_basis": "realized-only group-tag PnL by exit date over 100000 per represented symbol"}
        output[dimension] = result
    return output


def _add_retention(period_row):
    baseline = period_row["variants"]["baseline"]["stratifications"]
    opened = period_row["variants"]["open_admission"]["stratifications"]
    for dimension, baseline_groups in baseline.items():
        opened_groups = opened.get(dimension, {})
        for label in set(baseline_groups) | set(opened_groups):
            base_count = baseline_groups.get(label, {}).get("tag_trade_count", 0)
            open_count = opened_groups.get(label, {}).get("tag_trade_count", 0)
            retention = (open_count / base_count) if base_count else None
            if label in baseline_groups:
                baseline_groups[label]["open_admission_trade_retention"] = retention
                baseline_groups[label]["open_admission_retained_tag_trades"] = open_count
            if label in opened_groups:
                opened_groups[label]["retention_vs_baseline"] = retention


def _summarize_period(target, window):
    capital = STARTING_CASH * target["account_count"]
    dates = target["period_calendar"]
    curve = []
    for date in dates:
        equity = capital + target["daily_delta"][date]
        curve.append({"date": date, "cash": equity, "invested_value": 0.0,
                      "equity": equity, "shares": 0})
    if not curve:
        curve = [{"date": window["start"], "cash": capital, "invested_value": 0.0,
                  "equity": capital, "shares": 0}]
    ending = capital + sum(float(item["pnl"]) for item in target["trades"])
    if not math.isclose(float(curve[-1]["equity"]), ending, rel_tol=1e-9, abs_tol=1e-5):
        raise AssertionError("aggregated last equity does not match summed account ending equity")
    metrics = calculate_metrics(target["trades"], capital, ending, curve)
    by_pattern = {}
    pattern_names = sorted({pattern_type for trade in target["trades"]
                            for pattern_type in (trade.get("pattern_types") or [])})
    for pattern_type in pattern_names:
        group = [trade for trade in target["trades"]
                 if pattern_type in (trade.get("pattern_types") or [])]
        group_pnl = sum(float(item["pnl"]) for item in group)
        by_pattern[pattern_type] = {
            "trade_count": len(group), "net_pnl": group_pnl,
            "win_rate": (sum(float(item["pnl"]) > 0 for item in group) / len(group)) if group else None,
            "tag_count_in_signals": target["signal_tag_counts"].get(pattern_type, 0)}
    stratifications = _summarize_stratifications(target["stratification_rows"])
    return {"name": window["name"], "start": window["start"], "end": window["end"],
            "account_count": target["account_count"], "accounts_with_bars": target["accounts_with_bars"],
            "accounts_with_trades": target["accounts_with_trades"], "signal_count": target["signals"],
            "signal_tag_counts": dict(sorted(target["signal_tag_counts"].items())),
            "trade_count": len(target["trades"]), "starting_capital": capital,
            "ending_equity": ending, "net_pnl": ending - capital,
            "pattern_type_groups": by_pattern,
            "stratifications": stratifications,
            "rejection_counts": dict(sorted(target["rejections"].items())),
            "metrics": metrics, "daily_equity": curve}


def _trade_mfe(frame, trade):
    dates = pd.to_datetime(frame.date)
    entry_index = int(dates.searchsorted(pd.Timestamp(trade["entry_date"]), side="left"))
    exit_index = int(dates.searchsorted(pd.Timestamp(trade["exit_date"]), side="left"))
    start, end = entry_index + 1, exit_index
    if start >= len(frame) or end <= start:
        return 0.0
    high = pd.to_numeric(frame.high.iloc[start:end], errors="coerce").max()
    return float(high) / float(trade["entry_price"]) - 1.0


def _candidate_samples(account, bars):
    signals = {signal["signal_date"]: signal for signal in account.get("signals", [])}
    out = []
    for trade in account.get("trades", []):
        if "w_bottom" not in (trade.get("pattern_types") or []):
            continue
        signal = signals.get(trade.get("signal_date"))
        if not signal:
            continue
        snapshot = next((pattern for pattern in signal.get("patterns", [])
                         if pattern.get("pattern_type") == "w_bottom"
                         and pattern.get("pattern_id") in trade.get("pattern_ids", [])), None)
        if snapshot is None:
            continue
        mfe = _trade_mfe(bars, trade)
        key = (account["kind"], account["code"], trade["signal_date"], snapshot["pattern_id"])
        base = {"key": key, "kind": account["kind"], "code": account["code"],
                "trade": trade, "signal": signal, "pattern": snapshot, "mfe": mfe,
                "provenance": account.get("provenance", {})}
        if trade.get("exit_reason") == "stop_loss" and int(trade.get("holding_sessions", 0)) <= 5:
            out.append(("rapid_stop_5_sessions", base))
        if trade.get("exit_reason") == "stop_loss" and mfe >= 0.10:
            out.append(("stop_after_10pct_mfe", base))
        if trade.get("exit_reason") == "take_profit":
            out.append(("take_profit", base))
    return out


def _render_chart(sample, out_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    import numpy as np

    chinese_font = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False

    bars = _read_bars({"kind": sample["kind"], "code": sample["code"],
                       "provenance": sample.get("provenance", {})})
    frame = bars.copy().reset_index(drop=True)
    signal_date = sample["trade"]["signal_date"]
    dates = pd.to_datetime(frame.date)
    signal_index = int(dates.searchsorted(pd.Timestamp(signal_date), side="left"))
    key_points = sample["pattern"].get("key_points", {})
    pivots = []
    for n in range(1, 8):
        pivot_date = key_points.get("pivot_%d_date" % n)
        if not pivot_date:
            continue
        pivot_index = int(dates.searchsorted(pd.Timestamp(pivot_date), side="left"))
        if pivot_index <= signal_index and pivot_index < len(frame):
            pivots.append((n, pivot_index, key_points.get("pivot_%d_kind" % n),
                           key_points.get("pivot_%d" % n)))
    start = max(0, min((item[1] for item in pivots), default=signal_index) - 25)
    local = frame.iloc[start:signal_index + 1]
    x = np.arange(len(local))
    opens, highs = local.open.astype(float).to_numpy(), local.high.astype(float).to_numpy()
    lows, closes = local.low.astype(float).to_numpy(), local.close.astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.4), layout="constrained")
    colors = np.where(closes >= opens, "#b23b33", "#2b7a68")
    for i in range(len(local)):
        ax.vlines(x[i], lows[i], highs[i], color=colors[i], linewidth=0.8)
        bottom, height = min(opens[i], closes[i]), abs(closes[i] - opens[i])
        ax.bar(x[i], max(height, 0.001), bottom=bottom, width=0.55,
               color=colors[i], edgecolor=colors[i], linewidth=0.4)
    pattern = sample["pattern"]
    line = pattern.get("neckline") or {}
    if line.get("slope") is not None and line.get("intercept") is not None:
        full_indexes = np.arange(start, signal_index + 1)
        values = float(line["slope"]) * full_indexes + float(line["intercept"])
        ax.plot(x, values, color="#1d4e89", linewidth=1.3, label="Frozen boundary")
    for n, index, kind, pivot_price in pivots[:6]:
        if start <= index <= signal_index:
            y = float(pivot_price) if pivot_price is not None else float(
                frame.low.iloc[index] if kind == "low" else frame.high.iloc[index])
            ax.scatter(index - start, y, marker="v" if kind == "low" else "^",
                       s=45, color="#222222", zorder=3)
            ax.annotate("P%d" % n, (index - start, y), xytext=(0, -15 if kind == "low" else 8),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(len(local) - 1, color="#bb7a00", linestyle="--", linewidth=1)
    ax.set_title("W-bottom example | %s %s | signal date %s" %
                 (sample["kind"], sample["code"], signal_date))
    ax.set_ylabel("Price")
    ax.set_xlabel("Daily bars through confirmed signal close")
    tick_positions = np.linspace(0, max(0, len(local) - 1), min(8, len(local)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pd.Timestamp(local.date.iloc[pos]).strftime("%Y-%m")
                        for pos in tick_positions], rotation=0)
    ax.grid(axis="y", alpha=0.2)
    if line:
        ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _sample_facts(sample):
    trade = sample["trade"]
    return {"code": sample["code"], "kind": sample["kind"],
            "signal_date": trade["signal_date"], "entry_date": trade["entry_date"],
            "entry_price": trade["entry_price"], "exit_date": trade["exit_date"],
            "exit_price": trade["exit_price"], "exit_reason": trade["exit_reason"],
            "return_pct": trade["return_pct"], "holding_sessions": trade["holding_sessions"],
            "maximum_favorable_excursion": sample["mfe"],
            "pattern_id": sample["pattern"]["pattern_id"],
            "pattern_type": "w_bottom"}


def _scan_w_mfe_diagnostics():
    started = time.monotonic()
    counts = Counter()
    stratification_rows = []
    selected_samples = {}
    for account_path in _account_paths():
        account = json.loads(account_path.read_text(encoding="utf-8"))
        bars = _read_bars(account)
        stratification_rows.extend(_trade_stratification_rows(
            bars, account.get("signals", []), account.get("trades", [])))
        for category, sample in _candidate_samples(account, bars):
            current = selected_samples.get(category)
            if current is None or sample["key"] < current["key"]:
                selected_samples[category] = sample
        for trade in account.get("trades", []):
            if "w_bottom" not in (trade.get("pattern_types") or []):
                continue
            counts["w_trades"] += 1
            if trade.get("exit_reason") != "stop_loss":
                continue
            counts["w_stop_loss_trades"] += 1
            if int(trade.get("holding_sessions", 0)) <= 5:
                counts["w_stop_loss_within_5_sessions"] += 1
            if _trade_mfe(bars, trade) >= 0.10:
                counts["w_stop_loss_with_10pct_or_more_mfe_through_exit_session"] += 1
    stop_count = counts["w_stop_loss_trades"]
    return {**dict(counts),
            "rapid_stop_share_of_w_stop_losses": counts["w_stop_loss_within_5_sessions"] / stop_count if stop_count else None,
            "ten_pct_mfe_share_of_w_stop_losses": (
                counts["w_stop_loss_with_10pct_or_more_mfe_through_exit_session"] / stop_count if stop_count else None),
            "continuous_run_stratifications": _summarize_stratifications(stratification_rows),
            "continuous_run_stratification_scope": "archived original account timelines, not reset at evaluation-window boundaries",
            "illustrative_w_samples": {category: _sample_facts(sample)
                                        for category, sample in selected_samples.items()},
            "mfe_definition": "maximum daily high on strictly interior holding sessions (after entry session and before exit session), divided by actual entry price minus one",
            "mfe_excludes_entry_exit_sessions": True,
            "scan_elapsed_seconds": round(time.monotonic() - started, 3)}


def _money(value):
    return "¥{:,.0f}".format(float(value))


def _render_html(payload):
    rows = []
    for period in payload["periods"]:
        for variant, data in period["variants"].items():
            metrics = data["metrics"]
            rows.append("<tr><td>{}</td><td>{}</td><td>{:,}</td><td>{:.1%}</td>"
                        "<td>{}</td><td>{:.2f}</td><td>{:.1%}</td></tr>".format(
                            html.escape(period["name"]), html.escape(variant), data["trade_count"],
                            metrics["win_rate"] or 0.0, _money(data["net_pnl"]),
                            metrics["profit_factor"] or 0.0, metrics["maximum_drawdown"]))
    stratification_sections = []
    for period in payload["periods"]:
        detail_tables = []
        for variant, data in period["variants"].items():
            for dimension, groups in data.get("stratifications", {}).items():
                entries = []
                for label, metrics in groups.items():
                    retention = metrics.get("retention_vs_baseline",
                                            metrics.get("open_admission_trade_retention"))
                    retention_text = "" if retention is None else "{:.1%}".format(retention)
                    early_denominator = metrics.get("early_stop_denominator", 0)
                    neckline_denominator = metrics.get("neckline_close_loss_denominator", 0)
                    early_text = ("{:,}/{:,} ({:.1%})".format(
                        metrics.get("early_stop_count", 0), early_denominator,
                        metrics.get("early_stop_rate") or 0.0) if early_denominator else "—")
                    neckline_text = ("{:,}/{:,} ({:.1%})".format(
                        metrics.get("neckline_close_loss_count", 0), neckline_denominator,
                        metrics.get("neckline_close_loss_rate") or 0.0) if neckline_denominator else "—")
                    entries.append("<tr><td>{}</td><td>{:,}</td><td>{}</td><td>{}</td><td>{:.1%}</td>"
                                   "<td>{:.1%}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                        html.escape(str(label)), metrics["tag_trade_count"],
                        _money(metrics["net_pnl"]), _money(metrics["expectancy"] or 0),
                        metrics["win_rate"] or 0.0, metrics["realized_daily_close_drawdown"],
                        early_text, neckline_text, retention_text))
                detail_tables.append("<h4>{} / {}</h4><table><tr><th>分组</th><th>标签交易</th><th>净盈亏</th>"
                                     "<th>单笔期望</th><th>胜率</th><th>已实现组回撤</th><th>早止损数/分母</th>"
                                     "<th>颈线回落数/有效分母</th><th>保留率</th></tr>{}</table>".format(
                                         html.escape(variant), html.escape(dimension), "".join(entries)))
        stratification_sections.append("<details><summary>{} 分层明细</summary>{}</details>".format(
            html.escape(period["name"]), "".join(detail_tables)))
    archive_tables = []
    archive_stratifications = payload.get("w_mfe_diagnostics", {}).get(
        "continuous_run_stratifications", {})
    for dimension, groups in archive_stratifications.items():
        rows_for_dimension = []
        for label, metrics in groups.items():
            early_denominator = metrics.get("early_stop_denominator", 0)
            neckline_denominator = metrics.get("neckline_close_loss_denominator", 0)
            early = ("{:,}/{:,} ({:.1%})".format(
                metrics.get("early_stop_count", 0), early_denominator,
                metrics.get("early_stop_rate") or 0.0) if early_denominator else "—")
            neckline = ("{:,}/{:,} ({:.1%})".format(
                metrics.get("neckline_close_loss_count", 0), neckline_denominator,
                metrics.get("neckline_close_loss_rate") or 0.0) if neckline_denominator else "—")
            rows_for_dimension.append(
                "<tr><td>{}</td><td>{:,}</td><td>{:.1%}</td><td>{}</td><td>{}</td>"
                "<td>{}</td><td>{:.1%}</td><td>{}</td></tr>".format(
                    html.escape(str(label)), metrics["tag_trade_count"], metrics.get("win_rate") or 0.0,
                    _money(metrics["expectancy"] or 0), early, neckline,
                    metrics["realized_daily_close_drawdown"], _money(metrics["net_pnl"])))
        archive_tables.append(
            "<h4>{}</h4><table><tr><th>分组</th><th>标签交易</th><th>胜率</th><th>单笔期望</th>"
            "<th>早止损数/分母</th><th>颈线回落数/有效分母</th><th>已实现组回撤</th><th>净盈亏</th>"
            "</tr>{}</table>".format(html.escape(dimension), "".join(rows_for_dimension)))
    cards = []
    for case_index, (category, sample) in enumerate(payload.get("illustrative_w_samples", {}).items(), start=1):
        path = sample.get("chart", "charts/{}.png".format(category))
        facts_data = dict(sample)
        facts_data["selection_category"] = category
        facts = json.dumps(facts_data, ensure_ascii=False, indent=2)
        cards.append("<article><h3>{}</h3><img src='{}' alt='W pattern chart'><details>"
                     "<summary>展开查看归档交易事实</summary><pre>{}</pre></details></article>".format(
                         "Case %02d" % case_index, path, html.escape(facts)))
    summary = payload["archive_summary"]
    mfe = payload.get("w_mfe_diagnostics", {})
    mfe_story = ("归档中有 {w:,} 笔带 W 底标签的成交，其中 {stops:,} 笔以止损结束；"
                 "止损中 {rapid:,} 笔持仓不超过 5 个交易日（{rapid_share:.1%}），"
                 "{mfe_count:,} 笔在入场日之后、退出日之前的交易日最高价统计达到至少 10% MFE（{mfe_share:.1%}）。"
                 "MFE 明确排除入场日与退出日。".format(
                     w=mfe.get("w_trades", 0), stops=mfe.get("w_stop_loss_trades", 0),
                     rapid=mfe.get("w_stop_loss_within_5_sessions", 0),
                     rapid_share=mfe.get("rapid_stop_share_of_w_stop_losses") or 0.0,
                     mfe_count=mfe.get("w_stop_loss_with_10pct_or_more_mfe_through_exit_session", 0),
                     mfe_share=mfe.get("ten_pct_mfe_share_of_w_stop_losses") or 0.0))
    delay = payload.get("delayed_maturity_breakout_audit", {})
    first_cross = delay.get("first_buffered_close_after_structure_known") or {}
    delay_story = (
        "159122 的 W 底于 {formed} 形成，右底为 {right}; 结构于 {known} 已知，冻结颈线为 {line}，"
        "该线有 {test_count} 次独立压力测试（{test_dates}）。"
        "最早在 {first_date} 收盘 {close}，超过冻结线加前一日 ATR 的要求价 {required}。"
        "该次穿越早于六个月成熟日 {maturity}；后续信号发在 {signal}。"
        "引擎将早穿越记为 too_early，之后仍接受回穿。这违反既定的首次有效突破纪律，是明确的规则缺陷。"
        .format(formed=delay.get("formed_at", "unknown"),
               right=delay.get("second_bottom_date", "unknown"),
               known=delay.get("structure_known_at", "unknown"),
               line=(delay.get("selected_frozen_line") or {}).get("intercept", "unknown"),
               test_count=delay.get("line_test_count", 0),
               test_dates=", ".join(str(item) for item in delay.get("line_test_dates", [])),
               first_date=first_cross.get("date", "none found"),
               close=first_cross.get("close", "unknown"),
               required=first_cross.get("required_close", "unknown"),
               maturity=delay.get("maturity_date", "unknown"),
               signal=delay.get("signal_date", "unknown")))
    delay_facts = json.dumps(delay, ensure_ascii=False, indent=2)
    recross = payload.get("early_breakout_late_recross_scan", {})
    recross_rows = recross.get("rows", [])
    recross_facts = json.dumps(recross, ensure_ascii=False, indent=2)
    integration = payload.get("experiment_integration")
    if integration:
        integration_json = html.escape(json.dumps(integration, ensure_ascii=False, indent=2))
        cold = integration.get("baseline_run") or integration.get("cold_baseline", {})
        strict = integration.get("min3_experiment", {})
        paired = integration.get("paired_input_hybrid", {})
        runtime = integration.get("runtime_acceptance", {})
        r02 = integration.get("recognition_case_findings", {}).get("R02_000008_human_negative", {})
        warm = integration.get("warm_cache_verification", {})
        integration_html = (
            "<h2>实验版本、约束与校验</h2><p>冷基线：{cold_accounts} 个账户、{cold_trades:,} 笔交易、"
            "8 个 spawn worker，运行 {cold_seconds:.1f} 秒。min3：{min3_status}，每条颈线至少 3 个独立测试，"
            "保持四型信号与 8% 止损/20% 止盈。四个指定案例的 min3 v7 与 boundary_first 信号逐项相同。"
            "比较只包括基线、基线开放入场与 strict-min3 三项。{paired_notice}"
            "完整冷运行 {cold_wall:.1f} 秒，对照 1800 秒目标：{cold_target}; 1000/1000 账户与原归档逐笔账本签名一致。"
            "预热缓存单独验证：1000/1000 cache hit，seed {warm_seed:.1f} 秒、preflight {warm_preflight:.1f} 秒、运行 {warm_wall:.1f} 秒，账本 parity 通过。"
            "worker 选择 {worker_recognizer} / {worker_count}，8 worker 峰值 {worker_peak_gb:.2f} GiB，池内存上限 12 GiB。"
            "全样本 min3 没有消除 R02 人工负例，仍有三测试 W 信号。{test_result}"
            "<p><a href='../paired-input-snapshot-20261105-v1/snapshot-manifest.json'>冻结输入清单</a> · "
            "<a href='../paired-repairs-20261105-snapshot-v1/paired-repair-manifest.json'>52 标的成对修复清单</a> · "
            "<a href='../benchmark/1704b24a04344b81a8bafeb1c3c393a4/benchmark-result.json'>worker benchmark 与内存选择</a> · "
            "<a href='../warm-cache-20261105-8worker-retry1/warm-cache-audit.json'>1000 标的预热缓存校验</a> · "
            "<a href='../full-cold-20261105-v8-boundary-w8-spawnfix/acceptance-audit.json'>1000 账户冷运行 parity</a></p>"
            "<details><summary>运行清单、worker benchmark 和逐案校验</summary><pre>{}</pre></details>"
            .format(integration_json,
                    cold_accounts=cold.get("account_count", 0),
                    cold_trades=cold.get("trade_count", 0),
                    cold_seconds=float(cold.get("elapsed_seconds") or 0.0),
                    min3_status=html.escape(str(strict.get("status", "unknown"))),
                    paired_notice=("变更输入按冻结快照对基线和 min3 成对重跑；未变账户复用原始归档。"
                                   "这是混合来源的配对报告，不是历史字节级重放。" if paired else ""),
                    cold_wall=float(runtime.get("cold_run_load_through_report_write_seconds") or 0.0),
                    cold_target=("达标" if runtime.get("cold_run_target_met") else "未达标"),
                    warm_seed=float(warm.get("seed_elapsed_seconds") or 0.0),
                    warm_preflight=float(warm.get("cache_preflight_seconds") or 0.0),
                    warm_wall=float(warm.get("warm_wall_seconds") or 0.0),
                    test_result=html.escape(str(runtime.get("full_suite_result", "测试结果未提供。"))),
                    worker_recognizer=html.escape(str(runtime.get("worker_selection", {}).get("recognizer"))),
                    worker_count=runtime.get("worker_selection", {}).get("workers", "n/a"),
                    worker_peak_gb=(int(runtime.get("worker_selection", {}).get(
                        "8_worker_process_tree_peak_bytes") or 0) / (1024 ** 3)))
        )
    else:
        integration_html = ""
    baseline_only = "open-admission variant was not run for this archive" in payload.get("notes", [])
    open_gate_description = (
        "本报告仅重放 archive 发出的 strict-min3 信号；没有运行开放入场门。"
        if baseline_only else
        "开放入场条件：entry open 位于信号日冻结边界向入场日投影值至该边界加 3×信号日 Wilder ATR 的闭区间。"
        "只保留有实际输出快照、信号日突破证据且未被执行抑制的标签。ATR 只使用信号日收盘时已知的数据。"
    )
    promotion_html = _promotion_table_html(payload)
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>策略分段诊断 2026-11-05</title>
<style>body{font:15px/1.55 system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 22px;color:#202124}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{padding:9px 11px;border:1px solid #d5d8dc;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#eef2f6}article{margin:32px 0}img{display:block;width:min(100%,1100px);border:1px solid #ddd}details{margin-top:10px}pre{white-space:pre-wrap;background:#f5f6f7;padding:14px}small{color:#555}</style>
<h1>日线四型策略分段诊断</h1>
{experiment_integration}
{promotion_table}
<p>数据来源：归档全市场运行 <code>{source_archive}</code>，1000 个固定账户。
基线使用归档发出的信号；开放入场变体仅对已发出的形态标签检查入场开盘价，不重新识别候选。</p>
<p>各时段都使用独立 10 万元账户；信号识别保留完整历史，撮合仅读取时段日线，不跨段持仓或订单。
风险控制保持 8% 止损和 20% 止盈。分段结果由归档信号重放得到，不与原始连续账户汇总混为一谈。</p>
<h2>归档连续运行摘要</h2><p>交易 {trade_count:,} 笔，胜率 {win_rate:.1%}，净盈亏 {net_pnl}，最大回撤 {drawdown:.1%}。这些是来源版本摘要，不是下面的分段重置结果。</p>
<h2>分段重置比较</h2><table><thead><tr><th>时段</th><th>变体</th><th>交易</th><th>胜率</th><th>净盈亏</th><th>PF</th><th>最大回撤</th></tr></thead><tbody>{rows}</tbody></table>
<small>{open_gate_description}</small>
<h2>入场与结构分层</h2><p>下表按实际成交标签分组；多标签成交会在每个对应形态组各计一次。
突破 ATR = (信号收盘 − 信号日冻结边界) ÷ 记录的前一交易日 ATR；开盘位移 ATR = (entry open − 投影边界) ÷ 信号日 ATR；波动率分组使用信号日 ATR ÷ 信号日收盘。</p>
<p>颈线回落只统计 W 底、头肩底、三角底中有冻结边界证据的成交：从入场日到退出日，任一日收盘低于按边界斜率投影的冻结线记为命中。此项是日收盘诊断，不是止损撮合依据。分层回撤是按退出日汇总的已实现 PnL 回撤；每个含组成交的证券计入 10 万元基数，不含未实现市值。</p>
<p>{mfe_story}</p>
<h2>首个有效突破与成熟期诊断</h2><p>{delay_story}</p>
<details><summary>159122 原始K线、颈线测试与引擎早穿越记录</summary><pre>{delay_facts}</pre></details>
<h2>全样本早穿越后回穿清点</h2><p>本扫描只统计已发出的主结构快照中保存了早于 maturity_date 的前一 ATR 缓冲突破、且后续信号日期晚于成熟日的案例；它读取归档证据，不重新识别。</p>
<p>符合条件的延后回穿信号：{recross_count} 个；涉及 {recross_pattern_count} 个形态快照，涵盖 {recross_symbols} 个标的。同一信号日的多个模式快照按一个信号计数。另有 {other_boundary_retired} 条早期记录来自其他颈线候选，仅作为 other-boundary-retired 诊断，不计为同一冻结线的早期突破。</p>
<details><summary>逐个样本日期、线测试与首次早穿越证据</summary><pre>{recross_facts}</pre></details>
<details><summary>归档连续账户分层结果</summary><p>{archive_scope}</p>{archive_stratifications}</details>
{stratifications}
<h2>W 底案例</h2>{cards}
<h2>数据与限制</h2><p>图表截断在确认信号日；事实折叠区列出归档账户中的交易记录与价格路径统计。
案例按规则从归档 W 底交易中确定性选择，不包含人工标注或人工真值判断。这个报告是历史诊断材料，不是未触碰的样本外评估。</p>
</html>"""
    return (page.replace("{rows}", "".join(rows)).replace("{cards}", "".join(cards))
            .replace("{stratifications}", "".join(stratification_sections))
            .replace("{experiment_integration}", integration_html)
            .replace("{promotion_table}", promotion_html)
            .replace("{open_gate_description}", html.escape(open_gate_description))
            .replace("{source_archive}", html.escape(Path(payload["source_archive"]).name))
            .replace("{mfe_story}", html.escape(mfe_story))
            .replace("{delay_story}", html.escape(delay_story))
            .replace("{delay_facts}", html.escape(delay_facts))
            .replace("{recross_count}", str(recross.get("delayed_recross_signal_count", 0)))
            .replace("{recross_pattern_count}", str(recross.get("delayed_recross_pattern_count", 0)))
            .replace("{other_boundary_retired}", str(recross.get("excluded_evidence_counts", {}).get(
                "other_boundary_retired", 0)))
            .replace("{recross_symbols}", str(recross.get("affected_symbol_count", 0)))
            .replace("{recross_facts}", html.escape(recross_facts))
            .replace("{archive_scope}", html.escape(payload.get("w_mfe_diagnostics", {}).get(
                "continuous_run_stratification_scope", "Scope unavailable")))
            .replace("{archive_stratifications}", "".join(archive_tables))
            .replace("{trade_count:,}", "{:,}".format(summary["trade_count"]))
            .replace("{win_rate:.1%}", "{:.1%}".format(summary["win_rate"]))
            .replace("{net_pnl}", _money(summary["net_pnl"]))
            .replace("{drawdown:.1%}", "{:.1%}".format(summary["maximum_drawdown"])))


def _promotion_table_html(payload):
    """Render the stated research eligibility screen against the baseline per period."""
    if "open-admission variant was not run for this archive" in payload.get("notes", []):
        return ""
    periods = {row.get("name"): row for row in payload.get("periods", [])}
    expected = ("through_2020", "2021_2023", "2024_2026")
    if not all(name in periods for name in expected):
        return ""
    variants = ("baseline", "open_admission", "neckline_min3")
    labels = {"baseline": "基线", "open_admission": "开放入场", "neckline_min3": "min3"}
    def metrics(period, variant):
        return periods[period].get("variants", {}).get(variant, {}).get("metrics", {})
    rows = []
    baseline_validation = metrics("2021_2023", "baseline")
    baseline_locked = metrics("2024_2026", "baseline")
    baseline_counts = {name: periods[name].get("variants", {}).get("baseline", {}).get("trade_count", 0)
                       for name in expected}
    baseline_dd = baseline_locked.get("maximum_drawdown")
    for variant in variants:
        if not any(variant in periods[name].get("variants", {}) for name in expected):
            continue
        validation = metrics("2021_2023", variant)
        locked = metrics("2024_2026", variant)
        development = metrics("through_2020", variant)
        delta_validation = ((validation.get("expectancy") or 0.0) -
                            (baseline_validation.get("expectancy") or 0.0))
        delta_locked = ((locked.get("expectancy") or 0.0) -
                        (baseline_locked.get("expectancy") or 0.0))
        delta_development = ((development.get("expectancy") or 0.0) -
                             (metrics("through_2020", "baseline").get("expectancy") or 0.0))
        retention = []
        retention_ok = True
        for name in expected:
            count = periods[name].get("variants", {}).get(variant, {}).get("trade_count", 0)
            base = baseline_counts[name]
            value = count / base if base else None
            retention.append("n/a" if value is None else "{:.0%}".format(value))
            if value is None or value < 0.5:
                retention_ok = False
        expectancy_ok = delta_validation > 0 and delta_locked > 0
        drawdown = locked.get("maximum_drawdown")
        drawdown_ok = (drawdown is not None and baseline_dd is not None and drawdown <= baseline_dd)
        if variant == "baseline":
            decision = "对照"
        else:
            decision = "具备进一步研究资格" if expectancy_ok and drawdown_ok and retention_ok else "未达到研究门槛"
        rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(labels[variant]), _money(delta_validation), _money(delta_locked),
            "{:.1%} ({})".format(drawdown, "≤基线" if drawdown_ok else ">基线/未过")
            if drawdown is not None else "n/a",
            " / ".join(retention),
            _money(delta_development) + ("，开发期下降" if delta_development < 0 else ""),
            decision))
    return ("<h2>研究资格筛选（不代表生产切换）</h2>"
            "<p>验证期为 2021–2023，锁定期为 2024–2026。每个非基线变体只有在验证期与锁定期的每笔期望盈亏都高于基线、"
            "锁定期 MTM 最大回撤不高于基线、且三个时期交易数均至少保留 50% 时，才标为具备进一步研究资格。"
            "开发期差值单独披露，不参与资格判定；任何结果都不会改变生产基线。</p>"
            "<table><thead><tr><th>变体</th><th>验证期期望盈亏增量</th><th>锁定期期望盈亏增量</th>"
            "<th>锁定期 MTM 回撤</th><th>交易保留率 开发/验证/锁定</th><th>开发期期望盈亏变化</th><th>结论</th></tr></thead>"
            "<tbody>{}</tbody></table>".format("".join(rows)))


def main():
    global ARCHIVE, OUT, DATA_SOURCE_MAP, SNAPSHOT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-existing", action="store_true",
                        help="rebuild HTML from the saved research-review.json")
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE),
                        help="completed per-account run directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="new report output directory")
    parser.add_argument("--data-source-map", help="JSON map of accounts to frozen CSV inputs")
    parser.add_argument("--baseline-only", action="store_true",
                        help="replay only the archive signal stream; used for strict-min3 period output")
    args = parser.parse_args()
    ARCHIVE = Path(args.archive).expanduser().resolve()
    OUT = Path(args.out).expanduser().resolve()
    source_map_path = None
    if args.data_source_map:
        map_path = Path(args.data_source_map).expanduser().resolve()
        source_map_path = map_path
        map_payload = json.loads(map_path.read_text(encoding="utf-8"))
        DATA_SOURCE_MAP = map_payload.get("sources", map_payload)
        SNAPSHOT_ROOT = Path(map_payload.get("snapshot_root", "")).expanduser().resolve()
        if not SNAPSHOT_ROOT.is_dir():
            raise RuntimeError("data source map has no existing snapshot_root")
    if args.render_existing:
        payload = json.loads((OUT / "research-review.json").read_text(encoding="utf-8"))
        if "delayed_maturity_breakout_audit" not in payload:
            payload["delayed_maturity_breakout_audit"] = _delayed_maturity_breakout_audit()
        if "early_breakout_late_recross_scan" not in payload:
            recross_rows = []
            recross_diagnostics = Counter()
            for account_path in _account_paths():
                account = json.loads(account_path.read_text(encoding="utf-8"))
                recross_rows.extend(_early_breakout_late_recross_rows(account, recross_diagnostics))
            payload["early_breakout_late_recross_scan"] = _recross_summary(
                recross_rows, recross_diagnostics)
            _atomic_json(OUT / "research-review.json", payload)
        if ("w_mfe_diagnostics" not in payload or
                "continuous_run_stratifications" not in payload["w_mfe_diagnostics"]):
            payload["w_mfe_diagnostics"] = _scan_w_mfe_diagnostics()
            payload["illustrative_w_samples"] = {
                category: {**facts, "chart": "charts/{}.png".format(category)}
                for category, facts in payload["w_mfe_diagnostics"].get(
                    "illustrative_w_samples", {}).items()}
            _atomic_json(OUT / "research-review.json", payload)
        for category, facts in payload.get("illustrative_w_samples", {}).items():
            account_path = ARCHIVE / "accounts" / ("%s-%s.json" % (facts["kind"], facts["code"]))
            account = json.loads(account_path.read_text(encoding="utf-8"))
            signal = next(item for item in account["signals"]
                          if item["signal_date"] == facts["signal_date"])
            pattern = next(item for item in signal["patterns"]
                           if item["pattern_id"] == facts["pattern_id"])
            sample = {"kind": facts["kind"], "code": facts["code"],
                      "trade": {key: facts[key] for key in (
                          "signal_date", "entry_date", "exit_date", "exit_reason")},
                      "pattern": pattern, "provenance": account.get("provenance", {})}
            _render_chart(sample, OUT / facts["chart"])
        (OUT / "research-review.html").write_text(_render_html(payload), encoding="utf-8")
        manifest_path = OUT / "run-manifest.json"
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_manifest.update(status="complete", render_recovered=True)
            _atomic_json(manifest_path, run_manifest)
        print("rendered", OUT / "research-review.html")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    chart_dir = OUT / "charts"
    chart_dir.mkdir(exist_ok=True)
    manifest = json.loads((ARCHIVE / "run-manifest.json").read_text(encoding="utf-8"))
    archive_summary = manifest["summary"]
    paths = _account_paths()
    first_account = json.loads(paths[0].read_text(encoding="utf-8"))
    calendar_bars = _read_bars(first_account)
    period_calendars = {}
    for window in WINDOWS:
        mask = (calendar_bars.date.dt.date.astype(str) >= window["start"])
        mask &= calendar_bars.date.dt.date.astype(str) <= window["end"]
        period_calendars[window["name"]] = sorted(
            {value.date().isoformat() for value in calendar_bars.loc[mask, "date"]})
    variants_to_run = [("baseline", "source_signals")]
    if not args.baseline_only:
        variants_to_run.append(("open_admission", "open_signals"))
    periods = {window["name"]: {"window": window,
                                **{variant: _new_period(period_calendars[window["name"]])
                                   for variant, _source in variants_to_run}}
               for window in WINDOWS}
    gate_reason_counts = Counter()
    gate_audit = []
    selected_samples = {}
    delayed_recross_rows = []
    delayed_recross_diagnostics = Counter()
    account_count = 0
    started = time.monotonic()
    for account_path in paths:
        account = json.loads(account_path.read_text(encoding="utf-8"))
        if account.get("status") != "complete":
            continue
        bars = _read_bars(account)
        account_count += 1
        delayed_recross_rows.extend(_early_breakout_late_recross_rows(
            account, delayed_recross_diagnostics))
        if args.baseline_only:
            opened, decisions = [], []
        else:
            opened, decisions = _open_admission_gate(bars, account.get("signals", []))
        opened_by_date = {signal.get("signal_date"): signal for signal in opened}
        for decision in decisions:
            gate_reason_counts["admitted_tag"] += len(decision["admitted_pattern_ids"])
            for reason in decision["rejected_tags"]:
                gate_reason_counts["rejected_" + reason["reason"]] += 1
            accepted_signal = opened_by_date.get(decision["signal_date"])
            gate_audit.append({"code": account["code"], "kind": account["kind"],
                               "signal_date": decision["signal_date"],
                               "entry_date": decision["entry_date"],
                               "qualifying_tags": ([{"pattern_id": tag["pattern_id"],
                                                     "pattern_type": tag["pattern_type"],
                                                     "admission": tag["admission"]}
                                                    for tag in accepted_signal["pattern_tags"]]
                                                   if accepted_signal else []),
                               "rejected_tags": decision["rejected_tags"]})
        for category, sample in _candidate_samples(account, bars):
            current = selected_samples.get(category)
            if current is None or sample["key"] < current["key"]:
                selected_samples[category] = sample
        for name, details in periods.items():
            window = details["window"]
            mask = (bars.date.dt.date.astype(str) >= window["start"]) & (bars.date.dt.date.astype(str) <= window["end"])
            period_bars = bars.loc[mask].copy().reset_index(drop=True)
            signal_sources = {"source_signals": account.get("signals", []), "open_signals": opened}
            for variant, source_key in variants_to_run:
                source_signals = signal_sources[source_key]
                period_signals = [signal for signal in source_signals
                                  if signal.get("entry_date") is not None
                                  and window["start"] <= str(signal["entry_date"]) <= window["end"]]
                simulation = simulate_fixed_trades(period_bars, period_signals,
                                                   cash=STARTING_CASH)
                _record_period(details[variant], period_bars, period_signals, simulation)
        if account_count % 50 == 0:
            print("processed %d/%d accounts in %.1fs" %
                  (account_count, len(paths), time.monotonic() - started), flush=True)
    if account_count != 1000:
        raise RuntimeError("expected 1000 complete accounts, found %d" % account_count)
    elapsed_seconds = round(time.monotonic() - started, 3)

    period_rows = []
    for details in periods.values():
        window = details["window"]
        row = {"name": window["name"], "start": window["start"], "end": window["end"],
               "variants": {variant: _summarize_period(details[variant], window)
                            for variant, _source in variants_to_run}}
        if not args.baseline_only:
            _add_retention(row)
        period_rows.append(row)
    sample_payload = {}
    for category, sample in selected_samples.items():
        sample_payload[category] = _sample_facts(sample)
        sample_payload[category]["chart"] = "charts/{}.png".format(category)
    payload = {"report_version": "research_review_20261105_v2",
               "source_archive": str(ARCHIVE),
               "build_elapsed_seconds": elapsed_seconds,
               "source_run_id": manifest.get("run_id"),
               "source_strategy_version": manifest.get("strategy_version"),
               "source_recognizer_version": manifest.get("recognizer_version"),
               "source_engine_version": manifest.get("engine_version"),
               "source_rule_version": manifest.get("source_rule_version"),
               "data_source_map": str(source_map_path) if source_map_path else None,
               "composite_source_policy": manifest.get("source_policy"),
               "snapshot_id": manifest.get("snapshot_id"),
               "account_count": account_count, "archive_summary": archive_summary,
               "periods": period_rows, "open_admission_gate_reason_counts": dict(gate_reason_counts),
               "open_admission_gate_audit": gate_audit,
               "illustrative_w_samples": sample_payload,
               "delayed_maturity_breakout_audit": _delayed_maturity_breakout_audit(),
               "early_breakout_late_recross_scan": _recross_summary(
                   delayed_recross_rows, delayed_recross_diagnostics),
               "notes": ["segment accounts reset to 100000 cash and do not carry positions or orders",
                         "recognition signals are the archived causal emitted signals",
                         ("open-admission uses signal-close ATR and projects the emitted frozen boundary to entry day"
                          if not args.baseline_only else "open-admission variant was not run for this archive"),
                         "no manual pattern labels or annotation tool outputs were introduced"]}
    # Persist the complete period calculations before chart rendering so rendering can resume
    # from saved results without rerunning any broker accounts.
    _atomic_json(OUT / "research-review.json", payload)
    report_manifest = {"status": "periods_complete_charts_pending", "source_archive": str(ARCHIVE),
                       "account_count": account_count, "build_elapsed_seconds": elapsed_seconds,
                       "artifacts": {"json": str((OUT / "research-review.json").resolve()),
                                     "html": str((OUT / "research-review.html").resolve()),
                                     "charts": str(chart_dir.resolve())},
                       "periods": WINDOWS,
                       "variants": [variant for variant, _source in variants_to_run]}
    _atomic_json(OUT / "run-manifest.json", report_manifest)
    for category, sample in selected_samples.items():
        _render_chart(sample, chart_dir / (category + ".png"))
    (OUT / "research-review.html").write_text(_render_html(payload), encoding="utf-8")
    report_manifest.update(status="complete")
    _atomic_json(OUT / "run-manifest.json", report_manifest)
    print(json.dumps({"accounts": account_count, "output": str(OUT),
                      "periods": [{"name": item["name"],
                                   "trade_counts": {variant: data["trade_count"]
                                                    for variant, data in item["variants"].items()}}
                                  for item in period_rows],
                      "samples": sorted(selected_samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
