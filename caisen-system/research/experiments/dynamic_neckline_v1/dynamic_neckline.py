"""Research-only causal replay of versioned pressure boundaries on saved v12 structures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CAISEN_ROOT = HERE.parents[2]
WORKSPACE_ROOT = CAISEN_ROOT.parent
sys.path.insert(0, str(CAISEN_ROOT))

from caisen.fixed_backtest import calculate_metrics, simulate_fixed_trades  # noqa: E402
from caisen.main_bottoms import atr  # noqa: E402


SOURCE = CAISEN_ROOT / "reports/daily-research/full-unified-v12-20261106-w4-rerun"
SAMPLE_FILE = SOURCE / "partial-100-report/sample-selection.json"
PAIRED_FILE = CAISEN_ROOT / "reports/daily-research/paired-100-v8-v12-20261109/comparison.json"
DEFAULT_CUTOFF = "2026-10-30"
STARTING_CASH = 100_000.0
LOT_SIZE = 100
STOP_FRACTION = 0.08
TARGET_FRACTION = 0.20


def _iso(value):
    return pd.Timestamp(value).date().isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(pattern, code):
    for item in pattern.get("evidence", []) or []:
        if item.get("code") == code:
            return item.get("value") or {}
    return {}


def load_bars(code, cutoff=DEFAULT_CUTOFF):
    path = WORKSPACE_ROOT / "data/stocks" / f"{code}.csv"
    bars = pd.read_csv(path, encoding="utf-8-sig")
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = (bars.dropna(subset=["date", "open", "high", "low", "close", "volume"])
            .sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
            .loc[lambda x: x.date <= pd.Timestamp(cutoff)]
            .reset_index(drop=True))
    return bars, path


def structures_from_events(instance_events, bars, cutoff=DEFAULT_CUTOFF):
    dates = {_iso(date): i for i, date in enumerate(bars.date)}
    groups = defaultdict(list)
    for event in instance_events:
        event_date = str(event.get("event_date") or "")
        if not event_date or event_date > cutoff:
            continue
        pattern = event.get("pattern") or {}
        structure_key = event.get("structure_key") or pattern.get("structure_key")
        if not structure_key:
            continue
        geometry = event.get("frozen_geometry") or {}
        anchors = geometry.get("anchors") or []
        if not anchors:
            key_points = pattern.get("key_points") or {}
            for key, date in key_points.items():
                if not key.startswith("pivot_") or not key.endswith("_date"):
                    continue
                prefix = key[:-5]
                price = key_points.get(prefix)
                if price is not None:
                    anchors.append({"date": date, "price": price,
                                    "role": key_points.get(f"{prefix}_kind")})
        main = _evidence(pattern, "main_structure")
        selection = _evidence(pattern, "neckline_selection")
        boundary_available = _evidence(pattern, "boundary_available_at")
        declared_dates = [event_date, selection.get("structure_known_at"),
                          main.get("frozen_at"), boundary_available.get("date"),
                          event.get("boundary_available_at"), event.get("paired_at")]
        known_at = max(str(value) for value in declared_dates if value)
        if not known_at or known_at > cutoff or known_at not in dates:
            continue
        clean_anchors = []
        for anchor in anchors:
            date = str(anchor.get("date") or "")
            try:
                price = float(anchor.get("price"))
            except (TypeError, ValueError):
                continue
            if date in dates and date <= known_at:
                clean_anchors.append({"date": date, "index": dates[date], "price": price,
                                      "kind": anchor.get("role", anchor.get("kind", "unknown"))})
        if not clean_anchors:
            continue
        groups[structure_key].append({
            "event_date": event_date, "instance_id": event.get("instance_id"),
            "pattern": pattern, "geometry": geometry, "anchors": clean_anchors,
            "main": main, "selection": selection, "known_at": known_at,
            "known_index": dates[known_at], "pattern_type": pattern.get("pattern_type"),
            "boundary_available_at": boundary_available.get("date"),
        })

    structures = []
    for structure_key, events in groups.items():
        events.sort(key=lambda item: (item["event_date"], item["instance_id"] or ""))
        # Do not backfill later event geometry or anchors into the first archived observation.
        seed = events[0]
        anchors = seed["anchors"]
        main = seed["main"]
        support = (main.get("support_lower")
                   or (seed["geometry"].get("support_boundary") or {}).get("support_lower"))
        low_position = main.get("low_position") or {}
        low_anchors = sorted((point for point in anchors if point.get("kind") == "low"),
                             key=lambda point: int(point["index"]))
        first_trough_index = (int(low_anchors[0]["index"]) if low_anchors
                              else min(int(point["index"]) for point in anchors))
        lower_boundary = seed["geometry"].get("lower_boundary")
        if (seed["pattern_type"] == "triangle_bottom" and lower_boundary is None
                and len(low_anchors) >= 2):
            first, last = low_anchors[0], low_anchors[-1]
            span = int(last["index"]) - int(first["index"])
            if span > 0:
                slope = (float(last["price"]) - float(first["price"])) / span
                lower_boundary = {
                    "slope": slope,
                    "intercept": float(first["price"]) - slope * int(first["index"]),
                }
        structures.append({
            "structure_key": structure_key, "instance_id": seed["instance_id"],
            "pattern_type": seed["pattern_type"], "known_at": seed["known_at"],
            "known_index": seed["known_index"], "anchors": anchors,
            "first_trough_index": first_trough_index,
            "lower_boundary": lower_boundary,
            "formed_at": seed["pattern"].get("formed_at"),
            "maturity_date": main.get("maturity_date"),
            "support_lower": float(support) if support is not None else None,
            "low_position_pass": low_position.get("passed", True),
            "original_line": seed["geometry"].get("neckline") or seed["pattern"].get("neckline"),
            "source_instance_ids": [seed["instance_id"]] if seed["instance_id"] else [],
        })
    structures.sort(key=lambda item: (item["known_at"], item["structure_key"]))
    return structures


def load_structures(code, bars, cutoff=DEFAULT_CUTOFF, account_dir=None):
    account_dir = account_dir or (SOURCE / "accounts" / f"stock-{code}.json")
    account = json.load(open(account_dir, encoding="utf-8"))
    return structures_from_events(account.get("instance_events", []), bars, cutoff), account


def _line_value(line, index):
    return float(line["intercept"]) + float(line["slope"]) * int(index)


def _covers_structure(line, anchors, tick_size):
    # Structure-anchor coverage only; candle-by-candle formation coverage is deliberately not required.
    return all(_line_value(line, point["index"]) + 2 * tick_size >= point["price"]
               for point in anchors)


def build_boundary_revisions(bars, structure, pressure_points, tick_size=0.01):
    """Make upper-envelope revisions only when their 2ATR points are confirmed."""
    date_to_index = {_iso(date): i for i, date in enumerate(bars.date)}
    known_index = int(structure["known_index"] if "known_index" in structure
                      else date_to_index[structure["known_at"]])
    first_anchor = min(point["index"] for point in structure["anchors"])
    points = sorted((dict(point) for point in pressure_points
                     if point.get("kind") == "high"
                     and int(point["index"]) >= first_anchor),
                    key=lambda point: (int(point["confirmed_at"]), int(point["index"])))
    revisions = []
    active_points = []
    point_cursor = 0
    last_geometry = None
    support = structure.get("support_lower")
    invalidated_at = next((i for i in range(known_index, len(bars))
                           if support is not None and float(bars.close.iloc[i]) < float(support)),
                          len(bars))
    candidate_days = sorted({known_index} | {int(point["confirmed_at"]) for point in points
                                             if known_index < int(point["confirmed_at"]) < invalidated_at})
    for asof in candidate_days:
        if asof >= invalidated_at:
            continue
        while (point_cursor < len(points)
               and int(points[point_cursor]["confirmed_at"]) <= asof):
            active_points.append(points[point_cursor])
            point_cursor += 1
        # Keep only the two latest distinct confirmed 2ATR pressure highs.
        by_index = {int(point["index"]): point for point in active_points}
        pair = sorted(by_index.values(), key=lambda point: int(point["index"]))[-2:]
        if len(pair) != 2 or int(pair[0]["index"]) == int(pair[1]["index"]):
            continue
        left, right = pair
        span = int(right["index"]) - int(left["index"])
        if span <= 0:
            continue
        slope = (float(right["price"]) - float(left["price"])) / span
        line = {"slope": slope, "intercept": float(left["price"]) - slope * int(left["index"])}
        if not _covers_structure(line, structure["anchors"], tick_size):
            continue
        # Preserve the archived 20% cap for falling W/IHS lines.
        if structure["pattern_type"] in {"w_bottom", "inverse_head_shoulders"} and slope < 0:
            start, end = structure["anchors"][0]["index"], structure["anchors"][-1]["index"]
            start_level = _line_value(line, start)
            fall = (_line_value(line, end) - start_level) / max(abs(start_level), tick_size)
            if abs(fall) > 0.20:
                continue
        if len(pair) < 2:
            continue
        geometry = (round(slope, 12), round(float(line["intercept"]), 8))
        if geometry == last_geometry:
            continue
        known_at = max(structure["known_at"], _iso(bars.date.iloc[asof]))
        version_seed = f"{structure['structure_key']}|{known_at}|{geometry[0]}|{geometry[1]}"
        revisions.append({
            "boundary_version_id": "dyn-" + hashlib.sha256(version_seed.encode()).hexdigest()[:16],
            "known_at": known_at, "known_index": asof, "line": line,
            "supporting_point_indices": [int(left["index"]), int(right["index"])],
            "supporting_point_dates": [str(left.get("date") or _iso(bars.date.iloc[int(left["index"])])),
                                       str(right.get("date") or _iso(bars.date.iloc[int(right["index"])]))],
            "supporting_confirmed_dates": [str(left.get("confirmed_date") or
                                                _iso(bars.date.iloc[int(left["confirmed_at"])])),
                                           str(right.get("confirmed_date") or
                                                _iso(bars.date.iloc[int(right["confirmed_at"])]))],
            "coverage": "structure_anchors_only",
        })
        last_geometry = geometry
    return revisions


def build_breakout_signals(bars, structure, revisions, atr_multiple=1.0, minimum_buffer=0.02):
    """Emit a close breakout only on a later bar, never on revision availability day."""
    if not revisions or len(bars) < 2 or structure.get("low_position_pass") is False:
        return []
    atr_values = atr(bars, period=14).to_numpy(dtype=float)
    by_index = sorted(revisions, key=lambda item: item["known_index"])
    date_to_index = {_iso(date): i for i, date in enumerate(bars.date)}
    maturity_date = structure.get("maturity_date")
    maturity_index = (date_to_index[maturity_date] if maturity_date in date_to_index else
                      int(pd.DatetimeIndex(pd.to_datetime(bars.date)).searchsorted(
                          pd.Timestamp(maturity_date), side="left")) if maturity_date else 0)
    support = structure.get("support_lower")
    invalidated_at = next((i for i in range(int(revisions[0]["known_index"]), len(bars))
                           if support is not None and float(bars.close.iloc[i]) < float(support)),
                          len(bars))
    signals = []
    used_versions = set()
    active = None
    next_revision = 0
    for i in range(1, len(bars)):
        if i < maturity_index or i >= invalidated_at:
            continue
        while next_revision < len(by_index) and by_index[next_revision]["known_index"] <= i:
            active = by_index[next_revision]
            next_revision += 1
        if active is None or i <= int(active["known_index"]):
            continue
        version_id = active["boundary_version_id"]
        if version_id in used_versions:
            continue
        previous_level = _line_value(active["line"], i - 1)
        current_level = _line_value(active["line"], i)
        previous_atr = atr_values[i - 1]
        prior_atr = atr_values[i - 2] if i > 1 else np.nan
        if not np.isfinite(previous_atr) or previous_atr <= 0:
            continue
        buffer = max(float(minimum_buffer), float(atr_multiple) * float(previous_atr))
        prior_buffer = max(float(minimum_buffer), float(atr_multiple) * float(prior_atr)
                           if np.isfinite(prior_atr) and prior_atr > 0 else 0.0)
        previous_close = float(bars.close.iloc[i - 1])
        current_close = float(bars.close.iloc[i])
        previous_required = previous_level + prior_buffer
        if previous_close <= previous_required and current_close > current_level + buffer:
            signal_date = _iso(bars.date.iloc[i])
            entry_index = i + 1
            signals.append({
                "signal_date": signal_date,
                "entry_date": _iso(bars.date.iloc[entry_index]) if entry_index < len(bars) else None,
                "pattern_ids": [structure["instance_id"]],
                "pattern_types": [structure["pattern_type"]],
                "pattern_tags": [{"pattern_id": structure["instance_id"],
                                  "pattern_type": structure["pattern_type"],
                                  "structure_key": structure["structure_key"],
                                  "boundary_version_id": version_id}],
                "boundary_version_id": version_id,
                "boundary_value": current_level,
                "breakout_close": current_close,
                "previous_required_close": previous_required,
                "required_close": current_level + buffer,
            })
            used_versions.add(version_id)
    return signals


def pressure_points_2atr(bars, tick_size=0.01):
    """Reproduce the causal neckline-pressure high state: 2 ATR reversal, two ticks minimum."""
    if bars.empty:
        return []
    atr_values = atr(bars, period=14).to_numpy(dtype=float)
    highs = pd.to_numeric(bars.high, errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars.close, errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(bars.date).reset_index(drop=True)
    points, candidate = [], None
    for i in range(1, len(bars)):
        if not np.isfinite(highs[i]) or not np.isfinite(closes[i]):
            continue
        if candidate is None:
            if highs[i] >= highs[i - 1]:
                candidate = i
            continue
        if highs[i] > highs[candidate]:
            candidate = i
            continue
        if not np.isfinite(atr_values[candidate]) or atr_values[candidate] <= 0:
            continue
        reversal = max(2.0 * float(atr_values[candidate]), 2.0 * float(tick_size))
        if closes[i] <= highs[candidate] - reversal:
            points.append({"index": int(candidate), "date": _iso(dates.iloc[candidate]),
                           "price": float(highs[candidate]), "kind": "high",
                           "confirmed_at": int(i), "confirmed_date": _iso(dates.iloc[i]),
                           "scale": "pressure_2atr_atr_only"})
            candidate = None
    return points


def replay_symbol(code, cutoff=DEFAULT_CUTOFF, account_path=None, tick_size=0.01):
    bars, bars_path = load_bars(code, cutoff)
    structures, account = load_structures(code, bars, cutoff, account_dir=account_path)
    pivots = pressure_points_2atr(bars, tick_size=tick_size)
    pivot_confirm_dates = {point["confirmed_at"]: point["confirmed_date"] for point in pivots}
    all_revisions, all_signals = [], []
    for structure in structures:
        # Resolve missing archived point confirmation dates from the causal bar index.
        candidates = [point for point in pivots if point["kind"] == "high"
                      and point["index"] >= structure["anchors"][0]["index"]]
        revisions = build_boundary_revisions(bars, structure, candidates, tick_size=tick_size)
        signals = build_breakout_signals(bars, structure, revisions)
        for revision in revisions:
            revision.update({"structure_key": structure["structure_key"],
                             "instance_id": structure["instance_id"],
                             "pattern_type": structure["pattern_type"]})
        all_revisions.extend(revisions)
        all_signals.extend(signals)
    all_signals.sort(key=lambda item: (item["signal_date"], item["boundary_version_id"]))
    sim = simulate_fixed_trades(bars, all_signals, cash=STARTING_CASH, lot_size=LOT_SIZE)
    metrics = calculate_metrics(sim["trades"], STARTING_CASH, sim["ending_equity"], sim["daily_equity"])
    return {
        "code": code, "cutoff": cutoff, "status": "complete",
        "bars": len(bars), "structures": len(structures), "pressure_points_2atr": len(pivots),
        "boundary_revisions": len(all_revisions), "signals": len(all_signals),
        "metrics": metrics, "ending_equity": sim["ending_equity"],
        "trades": sim["trades"], "signals_detail": all_signals,
        "boundary_revisions_detail": all_revisions,
        "rejections": sim["rejections"], "accounting_checks": sim["accounting_checks"],
        "engine": sim["engine"],
        "provenance": {"bars_path": str(bars_path), "bars_sha256": _sha256(bars_path),
                       "account_path": str(account_path or (SOURCE / "accounts" / f"stock-{code}.json")),
                       "source_rule_version": account.get("source_rule_version"),
                       "engine_version": account.get("engine_version"),
                       "data_cutoff": cutoff},
    }


def run(symbols=None, cutoff=DEFAULT_CUTOFF, output_dir=None):
    selection = json.load(open(SAMPLE_FILE, encoding="utf-8"))
    sample = [item["code"] for item in selection["symbols"]]
    symbols = list(symbols or sample)
    output_dir = Path(output_dir or (HERE / "results" / f"cutoff-{cutoff}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results, failures = [], []
    for index, code in enumerate(symbols, start=1):
        account_path = SOURCE / "accounts" / f"stock-{code}.json"
        try:
            result = replay_symbol(code, cutoff=cutoff, account_path=account_path)
            results.append(result)
            (output_dir / f"stock-{code}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{index}/{len(symbols)} {code}: structures={result['structures']} "
                  f"revisions={result['boundary_revisions']} signals={result['signals']} "
                  f"trades={result['metrics']['trade_count']}", flush=True)
        except Exception as error:
            failures.append({"code": code, "error": f"{type(error).__name__}: {error}"})
            print(f"{index}/{len(symbols)} {code}: FAILED {failures[-1]['error']}", flush=True)
    elapsed = time.perf_counter() - started
    trades = [trade for result in results for trade in result["trades"]]
    pnl = sum(float(trade["pnl"]) for trade in trades)
    starting_capital = len(sample) * STARTING_CASH
    summary = {
        "prototype": "dynamic_neckline_v1", "scope": "saved_v12_structures_only",
        "cutoff": cutoff, "selected_sample_count": len(symbols),
        "source_sample_count": len(sample), "completed_accounts": len(results),
        "failed_accounts": failures, "runtime_seconds": elapsed,
        "total_structures": sum(item["structures"] for item in results),
        "total_2atr_pressure_points": sum(item["pressure_points_2atr"] for item in results),
        "total_boundary_revisions": sum(item["boundary_revisions"] for item in results),
        "total_signals": sum(item["signals"] for item in results),
        "total_trades": len(trades), "win_rate": (sum(t["pnl"] > 0 for t in trades) / len(trades)
                                                      if trades else None),
        "equal_start_capital_return": pnl / starting_capital if starting_capital else None,
        "net_pnl": pnl, "starting_capital": starting_capital,
        "limitations": ["uses only structures present in saved v12 instance_events; missed structures cannot be recovered",
                        "historical 2ATR pivots are recomputed from the current causal pivot helper",
                        "only structure-anchor coverage is required; full-formation candle coverage is not required",
                        "zero costs and slippage; no independent holdout in this run"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "data-manifest.json").write_text(json.dumps({
        "sample_selection_path": str(SAMPLE_FILE), "sample_selection_sha256": _sha256(SAMPLE_FILE),
        "sample_selection_rule": selection.get("selection_rule"),
        "cutoff": cutoff, "source_run_manifest_path": str(SOURCE / "run-manifest.json"),
        "source_run_manifest_sha256": _sha256(SOURCE / "run-manifest.json"),
        "source_rule_version": "caisen-main-bottoms-v12",
        "implementation_source": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "symbols": [{"code": item["code"],
                     "account_path": str(SOURCE / "accounts" / f"stock-{item['code']}.json"),
                     "account_sha256": _sha256(SOURCE / "accounts" / f"stock-{item['code']}.json"),
                     "bars_path": str(WORKSPACE_ROOT / "data/stocks" / f"{item['code']}.csv"),
                     "bars_sha256": _sha256(WORKSPACE_ROOT / "data/stocks" / f"{item['code']}.csv")}
                    for item in selection["symbols"]],
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="", help="comma-separated symbols; default is frozen sample")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    symbols = [part.strip() for part in args.symbols.split(",") if part.strip()] or None
    summary = run(symbols=symbols, cutoff=args.cutoff, output_dir=args.output_dir or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
