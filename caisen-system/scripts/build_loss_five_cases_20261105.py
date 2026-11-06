#!/usr/bin/env python3
"""Build five reproducible realized-loss case studies from the frozen paired baseline."""

from __future__ import annotations

import hashlib
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from caisen.daily_research import load_research_bars  # noqa: E402
from caisen.fixed_backtest import _wilder_atr_by_index  # noqa: E402

REPORT_ROOT = REPO / "reports" / "daily-research"
BASELINE = REPORT_ROOT / "paired-composites-20261105-snapshot-v1" / "baseline"
SOURCE_MAP_PATH = BASELINE / "data-source-map.json"
RESEARCH_REVIEW_PATH = REPORT_ROOT / "research-review-combined-20261105" / "research-review.json"
OUT_JSON = REPO / "reports" / "loss-five-cases-20261105.json"
OUT_HTML = REPO / "reports" / "loss-five-cases-20261105.html"
EXPECTED_SOURCE = "paired-input-snapshot-20261105-v1"

CASE_TARGETS = (
    ("premature_recross", "etf", "159122", "2026-06-22"),
    ("rapid_false_break", "stock", "002117", "2019-04-30"),
    ("extended_entry", "stock", "002242", "2025-11-13"),
    ("mfe_then_stop", "stock", "600281", "2017-02-15"),
    ("gap_stop", "stock", "600359", "2022-04-12"),
)
PATTERN_BOTTOMS = {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}

MANUAL_ASSESSMENTS = {
    "premature_recross": {
        "read": "K 线顺序显示，冻结线上的 ATR 缓冲收盘先出现在六个月成熟日前；后续成熟日后的信号是同一结构的再穿越。",
        "interpretation": "把 6 月 22 日当作该结构的首次突破，会掩盖 5 月 6 日的首次有效越线。此处是对日期顺序的读图解释。",
        "recommendation": "同一候选线首次出现成熟日前的有效缓冲收盘后永久退役；用因果回放验证它不再以晚些时候的回穿重启。",
    },
    "rapid_false_break": {
        "read": "确认后下一可执行日的收盘回到冻结颈线下，随后一个交易日按止损价退出。",
        "interpretation": "该次突破在一日内失效；结论依赖信号后收盘，而不是仅凭后来亏损倒推形态真假。",
        "recommendation": "把“入场日收盘仍在冻结线之上”作为单独研究门槛进行历史分层；本例只证明该门槛能识别本次回落，不证明总体收益改善。",
    },
    "extended_entry": {
        "read": "入场开盘大幅跳到冻结线的上方，按信号日 ATR 计算远超当前三倍 ATR 开放入场上限；两日后止损。",
        "interpretation": "这是开盘追高风险；本例的开盘门已拒绝该标签。",
        "recommendation": "继续单独记录开盘距投影冻结线的 ATR 倍数，并用已批准的开放入场对照验证筛选结果；不要把本例的亏损当成门槛有效性的因果证明。",
    },
    "mfe_then_stop": {
        "read": "入场后的严格区间最高价对应 MFE 19.98%；最高价比名义 20% 目标低 0.002，没有成交目标单；16 个持有交易日后触发止损。",
        "interpretation": "价格路径出现过大的未实现有利波动，但没有成交在目标位；MFE 本身不是可兑现利润。",
        "recommendation": "单独做按最小报价单位舍入的目标价敏感性测试，并保持原规则作为对照；本例的实际最高价和目标差额可直接复核。",
    },
    "gap_stop": {
        "read": "持仓期间先有收盘跌回冻结颈线下；止损日开盘低于止损价，实际止损成交价等于开盘价。",
        "interpretation": "该亏损同时有结构转弱与隔夜跳空穿越止损；它还满足严格区间内 MFE 至少 10%，所以计入 MFE 组时会与跳空组重叠。",
        "recommendation": "报告止损价与实际开盘成交价的差额，并把跳空造成的实际亏损与名义 8% 计划止损分开呈现；固定止损不能保证跳空时最多亏 8%。",
    },
}


def _strict_interior_mfe(bars, entry_index, exit_index, entry_price):
    """MFE uses highs after entry session and before exit session only."""
    start, stop = int(entry_index) + 1, int(exit_index)
    if start >= stop:
        return None
    highs = pd.to_numeric(bars.high.iloc[start:stop], errors="coerce")
    if highs.empty or not highs.notna().any() or float(entry_price) <= 0:
        return None
    index = int(highs.idxmax())
    high = float(highs.loc[index])
    return {"mfe_pct": round(high / float(entry_price) - 1.0, 8),
            "high": high, "date": pd.Timestamp(bars.date.iloc[index]).date().isoformat()}


def _stop_gap_evidence(trade, exit_open, tick_size):
    stop = float(trade.get("stop_price") or 0.0)
    opening, fill = float(exit_open), float(trade.get("exit_price") or 0.0)
    tolerance = max(float(tick_size) * 0.51, 1e-8)
    gap = (trade.get("exit_reason") == "stop_loss" and stop > 0
           and opening < stop - tolerance and abs(fill - opening) <= tolerance)
    return {"gap_through_stop": bool(gap),
            "stop_price": stop, "exit_open": opening, "exit_fill": fill,
            "gap_points": round(max(0.0, stop - opening), 8) if gap else 0.0,
            "gap_pct_of_stop": round((stop - opening) / stop, 8) if gap else 0.0,
            "fill_matches_exit_open": abs(fill - opening) <= tolerance}


def _false_break_evidence(signal_close, required_confirmation_close, closes,
                           dates, boundary_values, entry_offset, exit_offset):
    if signal_close is None or required_confirmation_close is None:
        return None
    if float(signal_close) <= float(required_confirmation_close):
        return None
    start, stop = int(entry_offset), min(int(exit_offset), len(closes) - 1)
    for index in range(start, stop + 1):
        if float(closes[index]) < float(boundary_values[index]):
            return {"confirmation_passed": True, "first_close_below_line": str(dates[index]),
                    "close": float(closes[index]), "frozen_line": float(boundary_values[index]),
                    "offset_from_entry_sessions": index - start}
    return None


def _select_five_cases(rows):
    selected = []
    for case_id, kind, code, signal_date in CASE_TARGETS:
        match = next((row for row in rows if row.get("kind") == kind
                      and str(row.get("code")) == code
                      and row.get("signal_date") == signal_date
                      and case_id in row.get("archetypes", [])), None)
        if match is None:
            raise RuntimeError("required evidence-backed case is missing: %s %s:%s %s" %
                               (case_id, kind, code, signal_date))
        selected.append({**match, "case_id": case_id})
    keys = {(row["kind"], str(row["code"]), row["signal_date"]) for row in selected}
    if len(selected) != 5 or len(keys) != 5:
        raise RuntimeError("the five case report requires five distinct event keys")
    return selected


def _evidence(pattern, code):
    for item in pattern.get("evidence", []) or []:
        if item.get("code") == code:
            return item.get("value")
    return None


def _selected_early_breakout(pattern, signal_date):
    selection = _evidence(pattern, "neckline_selection") or {}
    structure = _evidence(pattern, "main_structure") or {}
    candidate = selection.get("selected_candidate_id")
    line = selection.get("selected_line") or structure.get("neckline") or {}
    maturity = str(structure.get("maturity_date") or "")
    known_at = str(selection.get("structure_known_at") or structure.get("structure_known_at") or "")
    frozen_at = str(selection.get("line_frozen_at") or structure.get("frozen_at") or "")
    first_eligible = max(known_at, frozen_at)
    if not candidate or not maturity or not first_eligible:
        return None
    matches = []
    for item in selection.get("early_breakouts") or []:
        day = str(item.get("date") or "")
        if item.get("candidate_id") != candidate or not day or day < first_eligible:
            continue
        try:
            projected = float(line.get("slope", 0.0)) * float(item["index"]) + float(line["intercept"])
            if abs(projected - float(item["frozen_neckline"])) > 1e-8 * max(1.0, abs(projected)):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if day < maturity and signal_date > maturity:
            matches.append(item)
    return min(matches, key=lambda row: str(row["date"])) if matches else None


def _get_pattern(snapshot, trade):
    patterns = {item.get("pattern_id"): item for item in snapshot.get("patterns", []) or []}
    for pattern_id in trade.get("pattern_ids", []) or []:
        if pattern_id in patterns:
            return patterns[pattern_id]
    return None


def _analyze_trade(account, trade, bars, atr, date_index):
    signal_date, entry_date, exit_date = (str(trade.get(key)) for key in
                                         ("signal_date", "entry_date", "exit_date"))
    signal = next((item for item in account.get("signals", [])
                   if item.get("signal_date") == signal_date), None)
    if signal is None:
        return None
    pattern = _get_pattern(signal, trade)
    if pattern is None:
        return None
    indexes = [date_index.get(value) for value in (signal_date, entry_date, exit_date)]
    if any(value is None for value in indexes):
        return None
    signal_index, entry_index, exit_index = map(int, indexes)
    if entry_index <= 0:
        return None
    evidence = {"main_structure": _evidence(pattern, "main_structure") or {},
                "neckline_selection": _evidence(pattern, "neckline_selection") or {},
                "neckline_breakout": (_evidence(pattern, "neckline_breakout") or
                                       _evidence(pattern, "flag_breakout") or {})}
    structure, selection, breakout = (evidence["main_structure"], evidence["neckline_selection"],
                                      evidence["neckline_breakout"])
    line = pattern.get("neckline") or breakout.get("line") or {}
    frozen = breakout.get("frozen_neckline", breakout.get("frozen_boundary"))
    slope = float(line.get("slope", 0.0) or 0.0)
    projected_entry_line = (float(frozen) + slope * (entry_index - signal_index)
                            if frozen is not None else None)
    signal_atr = float(atr[signal_index]) if math.isfinite(float(atr[signal_index])) else None
    opening = float(bars.open.iloc[entry_index])
    prior_close = float(bars.close.iloc[entry_index - 1])
    displacement = ((opening - projected_entry_line) / signal_atr
                    if projected_entry_line is not None and signal_atr and signal_atr > 0 else None)
    mfe = _strict_interior_mfe(bars, entry_index, exit_index, trade.get("entry_price"))
    gap = _stop_gap_evidence(trade, float(bars.open.iloc[exit_index]),
                             float(account.get("provenance", {}).get("tick_size") or 0.01))
    bottom_false_break = None
    if (pattern.get("pattern_type") in PATTERN_BOTTOMS and frozen is not None
            and breakout.get("required_close") is not None):
        closes = pd.to_numeric(bars.close.iloc[entry_index:exit_index + 1]).astype(float).tolist()
        dates = [pd.Timestamp(value).date().isoformat()
                 for value in bars.date.iloc[entry_index:exit_index + 1]]
        boundaries = [float(frozen) + slope * (index - signal_index)
                      for index in range(entry_index, exit_index + 1)]
        bottom_false_break = _false_break_evidence(
            breakout.get("close"), breakout.get("required_close"), closes, dates,
            boundaries, 0, len(closes) - 1)

    archetypes = []
    early = _selected_early_breakout(pattern, signal_date)
    if early:
        archetypes.append("premature_recross")
    if (trade.get("exit_reason") == "stop_loss" and int(trade.get("holding_sessions", 0)) <= 5
            and bottom_false_break is not None):
        archetypes.append("rapid_false_break")
    if (trade.get("exit_reason") == "stop_loss" and displacement is not None and displacement > 3.0
            and opening > prior_close):
        archetypes.append("extended_entry")
    if trade.get("exit_reason") == "stop_loss" and mfe and mfe["mfe_pct"] >= 0.10:
        archetypes.append("mfe_then_stop")
    if gap["gap_through_stop"]:
        archetypes.append("gap_stop")
    if not archetypes:
        return None

    signal_close = float(bars.close.iloc[signal_index])
    entry_gap_pct = opening / prior_close - 1.0 if prior_close else None
    returned = {
        "kind": account.get("kind"), "code": str(account.get("code")),
        "signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date,
        "exit_reason": trade.get("exit_reason"), "pnl": float(trade.get("pnl", 0.0)),
        "return_pct": float(trade.get("return_pct", 0.0)),
        "holding_sessions": int(trade.get("holding_sessions", 0)),
        "pattern_type": pattern.get("pattern_type"), "pattern_id": pattern.get("pattern_id"),
        "pattern_labels": trade.get("pattern_labels", []), "pattern_tags": trade.get("pattern_tags", []),
        "archetypes": archetypes,
        "machine_geometry": {
            "formed_at": pattern.get("formed_at"),
            "confirmation_date": pattern.get("confirmation_date") or signal_date,
            "structure_known_at": selection.get("structure_known_at"),
            "line_frozen_at": selection.get("line_frozen_at"),
            "maturity_date": structure.get("maturity_date"),
            "selected_candidate_id": selection.get("selected_candidate_id"),
            "selected_line": line,
            "frozen_neckline_at_signal": frozen,
            "independent_test_count": len(selection.get("independent_tests") or []),
            "independent_test_dates": [item.get("pivot_date")
                                       for item in selection.get("independent_tests") or []],
            "independent_tests": selection.get("independent_tests") or [],
            "formation_coverage": selection.get("formation_coverage"),
            "support_lower": structure.get("support_lower"),
            "support_upper": structure.get("support_upper"),
            "key_points": pattern.get("key_points") or {},
            "rule_version": pattern.get("source_rule_version") or pattern.get("rule_version"),
            "main_structure": structure,
        },
        "observed_facts": {
            "pnl": float(trade.get("pnl", 0.0)),
            "return_pct": float(trade.get("return_pct", 0.0)),
            "exit_reason": trade.get("exit_reason"),
            "holding_sessions": int(trade.get("holding_sessions", 0)),
            "signal_close": signal_close,
            "breakout_close": breakout.get("close"),
            "breakout_required_close": breakout.get("required_close"),
            "prior_atr14": breakout.get("atr14_previous"),
            "signal_atr14": signal_atr,
            "entry_open": opening,
            "previous_close": prior_close,
            "entry_gap_pct": entry_gap_pct,
            "projected_frozen_line_at_entry": projected_entry_line,
            "entry_displacement_atr": displacement,
            "entry_price": float(trade.get("entry_price")),
            "exit_open": gap["exit_open"],
            "exit_price": gap["exit_fill"],
            "stop_price": float(trade.get("stop_price")),
            "target_price": float(trade.get("target_price")),
            "gap_stop": gap,
            "strict_interior_mfe": mfe,
            "first_post_entry_close_below_frozen_line": bottom_false_break,
            "early_selected_candidate_cross": early,
        },
        "trade_context": {
            "trade": trade,
            "signal_pattern_count": len(signal.get("patterns", [])),
            "signal_pattern_types": signal.get("pattern_types", []),
            "signal_entry_date": signal.get("entry_date"),
        },
        "_bar_indexes": {"signal": signal_index, "entry": entry_index, "exit": exit_index},
        "_pattern": pattern,
    }
    return returned


def _verify_snapshot_source(account, source_map, verified_hashes):
    kind, code = str(account["kind"]), str(account["code"])
    key = "%s:%s" % (kind, code)
    source = source_map.get(key)
    if source is None:
        raise RuntimeError("snapshot source map is missing %s" % key)
    expected = account.get("provenance", {}).get("data_revision")
    if expected != source.get("account_data_revision"):
        raise RuntimeError("account/snapshot source map revision mismatch: %s" % key)
    path = Path(source["snapshot_path"])
    stat = path.stat()
    actual_revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
    if actual_revision != source.get("snapshot_data_revision"):
        raise RuntimeError("frozen input stat changed: %s" % key)
    identity = str(path.resolve())
    if identity not in verified_hashes:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source.get("snapshot_sha256"):
            raise RuntimeError("frozen input hash changed: %s" % key)
        verified_hashes.add(identity)
    return source, path, actual_revision


def _load_snapshot_bars(account, source_map, verified_hashes):
    source, path, actual_revision = _verify_snapshot_source(account, source_map, verified_hashes)
    kind, code = str(account["kind"]), str(account["code"])
    root = Path(source["snapshot_root"])
    loaded = load_research_bars(SimpleNamespace(code=code, kind=kind, path=str(path)), root=root)
    if loaded.get("data_revision") != actual_revision:
        raise RuntimeError("loader revision disagrees with frozen input: %s" % key)
    bars = loaded["bars"].sort_values("date", kind="mergesort").drop_duplicates(
        "date", keep="last").reset_index(drop=True)
    bars["date"] = pd.to_datetime(bars.date, errors="coerce").dt.normalize()
    return bars, source


def _compute_case_rows(source_map, open_gate_audit):
    all_rows, losses, stop_losses = [], [], []
    totals = Counter()
    overlap_rows = []
    verified_hashes = set()
    for account_path in sorted((BASELINE / "accounts").glob("*.json")):
        account = json.loads(account_path.read_text(encoding="utf-8"))
        trades = account.get("trades", [])
        totals["account_count"] += 1
        totals["trade_count"] += len(trades)
        source, _path, _revision = _verify_snapshot_source(account, source_map, verified_hashes)
        red_trades = [trade for trade in trades if float(trade.get("pnl", 0.0)) < 0]
        if not red_trades:
            continue
        bars, source = _load_snapshot_bars(account, source_map, verified_hashes)
        date_values = [value.date().isoformat() for value in bars.date]
        date_index = {value: index for index, value in enumerate(date_values)}
        atr = _wilder_atr_by_index(bars)
        for trade in red_trades:
            totals["loss_trade_count"] += 1
            totals["loss_pnl_sum"] += float(trade.get("pnl", 0.0))
            if trade.get("exit_reason") == "stop_loss":
                totals["stop_loss_count"] += 1
                totals["stop_loss_pnl_sum"] += float(trade.get("pnl", 0.0))
                if int(trade.get("holding_sessions", 0)) <= 5:
                    totals["rapid_stop_count"] += 1
                    totals["rapid_stop_pnl_sum"] += float(trade.get("pnl", 0.0))
            signal = next((item for item in account.get("signals", [])
                           if item.get("signal_date") == trade.get("signal_date")), None)
            if signal is None:
                continue
            patterns = {item.get("pattern_id"): item for item in signal.get("patterns", [])}
            if not any(pattern_id in patterns for pattern_id in trade.get("pattern_ids", [])):
                continue
            result = _analyze_trade(account, trade, bars, atr, date_index)
            if result is None:
                continue
            result["_bars"] = bars
            result["_source"] = source
            event_key = (result["kind"], result["code"], result["signal_date"])
            if result["exit_reason"] == "stop_loss":
                if "mfe_then_stop" in result["archetypes"]:
                    totals["mfe_stop_count"] += 1
                    totals["mfe_stop_pnl_sum"] += result["pnl"]
                if "gap_stop" in result["archetypes"]:
                    totals["gap_stop_count"] += 1
                    totals["gap_stop_pnl_sum"] += result["pnl"]
                if "extended_entry" in result["archetypes"]:
                    totals["extended_entry_count"] += 1
                    totals["extended_entry_pnl_sum"] += result["pnl"]
                if "rapid_false_break" in result["archetypes"]:
                    totals["rapid_false_break_count"] += 1
                    totals["rapid_false_break_pnl_sum"] += result["pnl"]
            for archetype in result["archetypes"]:
                overlap_rows.append((event_key, archetype))
            all_rows.append(result)
        # Verify all source inputs, including symbols with only losses, are mapped from the same frozen set.
        if source.get("snapshot_id") not in (None, EXPECTED_SOURCE):
            raise RuntimeError("unexpected source snapshot id for %s:%s" % (account["kind"], account["code"]))

    categories_by_key = {}
    for event_key, archetype in overlap_rows:
        categories_by_key.setdefault(event_key, set()).add(archetype)
    labels = ("premature_recross", "rapid_false_break", "extended_entry", "mfe_then_stop", "gap_stop")
    overlaps = {}
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            overlaps["%s__%s" % (left, right)] = sum(
                left in categories and right in categories for categories in categories_by_key.values())

    gate_index = {}
    for item in open_gate_audit:
        key = (str(item.get("kind")), str(item.get("code")), str(item.get("signal_date")))
        gate_index[key] = item
    for row in all_rows:
        gate = gate_index.get((row["kind"], row["code"], row["signal_date"]))
        if gate:
            row["open_admission_decision"] = {
                "qualifying_tags": gate.get("qualifying_tags", []),
                "rejected_tags": gate.get("rejected_tags", []),
            }
        else:
            row["open_admission_decision"] = None
    aggregate = {
        "account_count": totals["account_count"],
        "trade_count": totals["trade_count"],
        "loss_trade_count": totals["loss_trade_count"],
        "loss_rate": totals["loss_trade_count"] / totals["trade_count"] if totals["trade_count"] else None,
        "realized_loss_pnl_sum": totals["loss_pnl_sum"],
        "average_realized_loss": totals["loss_pnl_sum"] / totals["loss_trade_count"] if totals["loss_trade_count"] else None,
        "stop_loss_count": totals["stop_loss_count"],
        "stop_loss_rate_of_losses": totals["stop_loss_count"] / totals["loss_trade_count"] if totals["loss_trade_count"] else None,
        "stop_loss_pnl_sum": totals["stop_loss_pnl_sum"],
        "rapid_stop_le_5_sessions": {"count": totals["rapid_stop_count"],
                                     "share_of_stop_losses": totals["rapid_stop_count"] / totals["stop_loss_count"] if totals["stop_loss_count"] else None,
                                     "pnl_sum": totals["rapid_stop_pnl_sum"]},
        "strict_interior_mfe_at_least_10pct_stop": {"count": totals["mfe_stop_count"],
                                                     "share_of_stop_losses": totals["mfe_stop_count"] / totals["stop_loss_count"] if totals["stop_loss_count"] else None,
                                                     "pnl_sum": totals["mfe_stop_pnl_sum"],
                                                     "definition": "maximum daily high only after entry session and before exit session"},
        "gap_through_stop": {"count": totals["gap_stop_count"],
                              "share_of_stop_losses": totals["gap_stop_count"] / totals["stop_loss_count"] if totals["stop_loss_count"] else None,
                              "pnl_sum": totals["gap_stop_pnl_sum"]},
        "entry_open_more_than_3_signal_atr_above_line": {"count": totals["extended_entry_count"],
                                                          "share_of_stop_losses": totals["extended_entry_count"] / totals["stop_loss_count"] if totals["stop_loss_count"] else None,
                                                          "pnl_sum": totals["extended_entry_pnl_sum"]},
        "rapid_buffered_break_then_close_below_line": {"count": totals["rapid_false_break_count"],
                                                        "share_of_stop_losses": totals["rapid_false_break_count"] / totals["stop_loss_count"] if totals["stop_loss_count"] else None,
                                                        "pnl_sum": totals["rapid_false_break_pnl_sum"]},
        "archetype_overlap_counts": overlaps,
    }
    return aggregate, all_rows, verified_hashes


def _format_num(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return ("%.*f" % (digits, float(value))).rstrip("0").rstrip(".")


def _svg_chart(case):
    bars = case["_bars"]
    indexes = case["_bar_indexes"]
    pattern = case["_pattern"]
    geometry = case["machine_geometry"]
    observed = case["observed_facts"]
    date_to_idx = {pd.Timestamp(value).date().isoformat(): index
                   for index, value in enumerate(bars.date)}
    point_dates = []
    for key, value in (geometry.get("key_points") or {}).items():
        if key.endswith("_date") and value in date_to_idx:
            point_dates.append(date_to_idx[value])
    tests = geometry.get("independent_tests") or []
    for item in tests:
        if item.get("pivot_date") in date_to_idx:
            point_dates.append(date_to_idx[item["pivot_date"]])
    coverage = geometry.get("formation_coverage") or {}
    coverage_start = coverage.get("start_date")
    starts = [indexes["signal"]]
    starts.extend(point_dates)
    if geometry.get("formed_at") in date_to_idx:
        starts.append(date_to_idx[geometry["formed_at"]])
    if coverage_start in date_to_idx:
        starts.append(date_to_idx[coverage_start])
    start = max(0, min(starts))
    end = indexes["exit"]
    if end <= start:
        start = max(0, indexes["signal"] - 60)
    visible = bars.iloc[start:end + 1].reset_index(drop=True)
    n = len(visible)
    width, height = 1380, 550
    left, right, top, bottom = 72, 30, 42, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    price_values = (pd.to_numeric(visible.low).tolist() + pd.to_numeric(visible.high).tolist())
    price_values.extend([observed["stop_price"], observed["target_price"], observed["entry_price"]])
    frozen = observed.get("projected_frozen_line_at_entry")
    slope = float((geometry.get("selected_line") or {}).get("slope", 0.0) or 0.0)
    if frozen is not None:
        price_values.extend(float(frozen) + slope * (i - indexes["entry"])
                            for i in range(start, end + 1))
    support_low, support_high = geometry.get("support_lower"), geometry.get("support_upper")
    for value in (support_low, support_high):
        if value is not None and math.isfinite(float(value)):
            price_values.append(float(value))
    low, high = min(price_values), max(price_values)
    span = max(high - low, abs(high) * 0.01, 1e-4)
    low -= span * 0.08
    high += span * 0.08
    def x(index):
        if n <= 1:
            return left + plot_w / 2
        return left + (index - start) * plot_w / (n - 1)
    def y(value):
        return top + (high - float(value)) * plot_h / (high - low)
    items = [f'<svg class="price-chart" id="chart-{html.escape(case["case_id"])}" viewBox="0 0 {width} {height}" role="img" aria-label="OHLC chart from {visible.date.iloc[0].date()} to {visible.date.iloc[-1].date()}">',
             '<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>',
             f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="#d9e0e8"/>']
    for tick in range(5):
        value = low + (high - low) * tick / 4
        yy = y(value)
        items.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_w}" y2="{yy:.2f}" stroke="#e7ebf0"/>')
        items.append(f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" class="axis">{value:.3f}</text>')

    entry_idx, exit_idx, signal_idx = indexes["entry"], indexes["exit"], indexes["signal"]
    stop_y, target_y, entry_y = y(observed["stop_price"]), y(observed["target_price"]), y(observed["entry_price"])
    entry_x, exit_x = x(entry_idx), x(exit_idx)
    items.extend([
        f'<rect x="{entry_x:.2f}" y="{min(stop_y,entry_y):.2f}" width="{max(1,exit_x-entry_x):.2f}" height="{abs(stop_y-entry_y):.2f}" fill="#ef4444" opacity=".07"/>',
        f'<rect x="{entry_x:.2f}" y="{min(target_y,entry_y):.2f}" width="{max(1,exit_x-entry_x):.2f}" height="{abs(target_y-entry_y):.2f}" fill="#16a34a" opacity=".07"/>',
        f'<line x1="{entry_x:.2f}" y1="{stop_y:.2f}" x2="{exit_x:.2f}" y2="{stop_y:.2f}" stroke="#dc2626" stroke-dasharray="7 5" stroke-width="2"/><text x="{exit_x-3:.2f}" y="{stop_y-5:.2f}" text-anchor="end" class="label">8% stop {observed["stop_price"]:.3f}</text>',
        f'<line x1="{entry_x:.2f}" y1="{target_y:.2f}" x2="{exit_x:.2f}" y2="{target_y:.2f}" stroke="#15803d" stroke-dasharray="7 5" stroke-width="2"/><text x="{exit_x-3:.2f}" y="{target_y-5:.2f}" text-anchor="end" class="label">20% target {observed["target_price"]:.3f}</text>',
        f'<line x1="{entry_x:.2f}" y1="{entry_y:.2f}" x2="{exit_x:.2f}" y2="{entry_y:.2f}" stroke="#64748b" stroke-dasharray="3 5"/><text x="{exit_x-3:.2f}" y="{entry_y-5:.2f}" text-anchor="end" class="label">entry {observed["entry_price"]:.3f}</text>',
    ])
    # The selected line is dashed before it was frozen and solid after its recorded freeze date.
    if frozen is not None:
        freeze_date = geometry.get("line_frozen_at") or geometry.get("structure_known_at")
        freeze_idx = date_to_idx.get(str(freeze_date), signal_idx)
        prior_x = max(start, min(freeze_idx, end))
        items.append(f'<line x1="{x(start):.2f}" y1="{y(float(frozen)+slope*(start-entry_idx)):.2f}" x2="{x(prior_x):.2f}" y2="{y(float(frozen)+slope*(prior_x-entry_idx)):.2f}" stroke="#2563eb" stroke-width="2" stroke-dasharray="5 5"><title>Selected frozen line back-projected before freeze date {html.escape(str(freeze_date))}</title></line>')
        items.append(f'<line x1="{x(prior_x):.2f}" y1="{y(float(frozen)+slope*(prior_x-entry_idx)):.2f}" x2="{x(end):.2f}" y2="{y(float(frozen)+slope*(end-entry_idx)):.2f}" stroke="#2563eb" stroke-width="2.5"><title>Frozen line projected from entry reference</title></line>')
        if support_low is not None and support_high is not None:
            sy1, sy2 = y(float(support_low)), y(float(support_high))
            items.append(f'<rect x="{x(start):.2f}" y="{min(sy1,sy2):.2f}" width="{x(end)-x(start):.2f}" height="{abs(sy2-sy1):.2f}" fill="#f59e0b" opacity=".09"><title>Recorded support band {support_low}–{support_high}</title></rect>')

    candle_step = plot_w / max(n - 1, 1)
    candle_w = max(1.2, min(7.0, candle_step * 0.52))
    for local_idx, bar in visible.iterrows():
        actual_idx = start + local_idx
        xx = x(actual_idx)
        o, h, l, c = (float(bar[col]) for col in ("open", "high", "low", "close"))
        color = "#15803d" if c >= o else "#dc2626"
        items.append(f'<line x1="{xx:.2f}" y1="{y(h):.2f}" x2="{xx:.2f}" y2="{y(l):.2f}" stroke="{color}" stroke-width="1.2"/>')
        body_y = min(y(o), y(c))
        body_h = max(1.0, abs(y(o) - y(c)))
        items.append(f'<rect x="{xx-candle_w/2:.2f}" y="{body_y:.2f}" width="{candle_w:.2f}" height="{body_h:.2f}" fill="{color}" stroke="{color}"><title>{pd.Timestamp(bar.date).date().isoformat()} O {o:.3f} H {h:.3f} L {l:.3f} C {c:.3f}</title></rect>')

    # Formation pivots are individually labeled with their actual calendar dates.
    key_points = geometry.get("key_points") or {}
    for name, value in key_points.items():
        if not name.endswith("_date") or not value or value not in date_to_idx:
            continue
        index = date_to_idx[value]
        if not start <= index <= end:
            continue
        base_name = name[:-5]
        price = key_points.get(base_name)
        if price is None:
            continue
        kind = key_points.get(base_name + "_kind", "pivot")
        items.append(f'<circle cx="{x(index):.2f}" cy="{y(price):.2f}" r="4.2" fill="#7c3aed" stroke="#fff" stroke-width="1.5"><title>{html.escape(base_name)} {html.escape(str(kind))} {html.escape(value)} @ {float(price):.3f}</title></circle>')
        items.append(f'<text x="{x(index):.2f}" y="{y(price)-8:.2f}" text-anchor="middle" class="marker">{html.escape(base_name.replace("pivot_", "P"))}</text>')

    for test in tests:
        date = test.get("pivot_date")
        if date not in date_to_idx:
            continue
        index = date_to_idx[date]
        if not start <= index <= end:
            continue
        price = float(test.get("level", test.get("high", observed.get("frozen_neckline_at_signal") or 0.0)))
        items.append(f'<circle cx="{x(index):.2f}" cy="{y(price):.2f}" r="5.8" fill="#f59e0b" fill-opacity=".25" stroke="#b45309" stroke-width="1.4"><title>Independent line test {html.escape(str(date))} value {price:.3f} group {html.escape(str(test.get("group_id", "")))}</title></circle>')

    markers = [
        (geometry.get("structure_known_at"), "structure known", "#7c3aed", "5 3"),
        (geometry.get("line_frozen_at"), "line frozen", "#2563eb", "2 4"),
        (case["signal_date"], "confirmed", "#0f766e", "7 3"),
        (case["entry_date"], "entry", "#111827", ""),
        (case["exit_date"], "exit", "#b91c1c", ""),
    ]
    marker_xs = []
    for date, label, color, dash in markers:
        if date not in date_to_idx:
            continue
        index = date_to_idx[date]
        if not start <= index <= end:
            continue
        xx = x(index)
        marker_xs.append((xx, label, color))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        items.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+plot_h}" stroke="{color}" stroke-width="1.5"{dash_attr}><title>{html.escape(label)} {html.escape(str(date))}</title></line>')
    for rank, (xx, label, color) in enumerate(marker_xs):
        label_y = top + 12 + 14 * rank
        anchor = "end" if xx > left + plot_w * 0.88 else "start"
        items.append(f'<text x="{xx:.2f}" y="{label_y}" text-anchor="{anchor}" transform="rotate(-70 {xx:.2f} {label_y})" class="marker">{html.escape(label)}</text>')

    ticks = sorted(set(round(i * (n - 1) / 5) for i in range(min(6, n))))
    for local_idx in ticks:
        actual_idx = start + local_idx
        date = pd.Timestamp(bars.date.iloc[actual_idx]).date().isoformat()
        xx = x(actual_idx)
        items.append(f'<line x1="{xx:.2f}" y1="{top+plot_h}" x2="{xx:.2f}" y2="{top+plot_h+5}" stroke="#475569"/>')
        items.append(f'<text x="{xx:.2f}" y="{height-24}" text-anchor="middle" class="axis">{date}</text>')
    items.append(f'<text x="{left}" y="22" class="label">Actual session dates · {html.escape(str(visible.date.iloc[0].date()))} to {html.escape(str(visible.date.iloc[-1].date()))}</text>')
    items.append("</svg>")
    return "".join(items)


def _serializable_case(case):
    return {key: value for key, value in case.items() if not key.startswith("_")}


def _case_html(case):
    observed = case["observed_facts"]
    geometry = case["machine_geometry"]
    manual = case["manual_assessment"]
    tests = ", ".join(geometry.get("independent_test_dates", [])) or "未保存"
    mfe = observed.get("strict_interior_mfe")
    mfe_value = "无严格区间内交易日" if mfe is None else "+{:.2%}，最高价 {}（{}）".format(
        mfe["mfe_pct"], _format_num(mfe["high"], 3), mfe["date"])
    open_gate = case.get("open_admission_decision")
    if not open_gate:
        gate_text = "未在该实验审计表中找到该事件日记录"
    elif open_gate.get("rejected_tags"):
        gate_text = "拒绝：" + "、".join(item.get("reason", "unknown")
                                          for item in open_gate["rejected_tags"])
    else:
        gate_text = "通过"
    early = observed.get("early_selected_candidate_cross")
    early_text = ("首次同候选成熟前越线：{}，要求收盘 {:.3f}，实际 {:.3f}".format(
        early["date"], early.get("required_close", float("nan")), early.get("close", float("nan")))
                  if early else "未检出所选候选线的成熟前有效越线")
    corrected = case.get("corrected_engine_verdict")
    if corrected:
        versions = corrected.get("source_version", {})
        old_pattern = "仍有信号" if corrected.get("original_pattern_emitted_on_signal_date") else "未在该日发出原 pattern ID"
        any_signal = "有任意合格信号" if corrected.get("any_eligible_signal_on_signal_date") else "该日前缀没有合格信号"
        other_types = ", ".join(corrected.get("same_date_pattern_types", [])) or "无"
        corrected_text = (f"{old_pattern}；{any_signal}；同日形态类型：{other_types}。"
                          f"前缀截止 {corrected.get('prefix_cutoff')}，版本 "
                          f"{versions.get('source_rule_version')} / {versions.get('engine_version')} / "
                          f"{versions.get('recognizer')}，main_bottoms.py SHA-256 "
                          f"{versions.get('main_bottoms_sha256')}。 "
                          f"{corrected.get('case_specific_detail', '')}")
    else:
        corrected_text = "冻结修正引擎源码后回放中；仅使用截至历史信号日的输入。"
    below = observed.get("first_post_entry_close_below_frozen_line")
    below_text = ("首次入场后收盘在线下：{}，收盘 {:.3f} / 冻结线 {:.3f}".format(
        below["first_close_below_line"], below["close"], below["frozen_line"])
                  if below else "入场至退出期间未见收盘跌回该冻结线下")
    gap = observed["gap_stop"]
    return f"""<article class="case" id="{html.escape(case['case_id'])}">
<header><div><span class="eyebrow">CASE {case['case_number']:02d} · {html.escape(case['archetype_label'])}</span><h2>{html.escape(case['title'])}</h2><p>{html.escape(case['kind'])} {html.escape(case['code'])} · 信号 {case['signal_date']} · 入场 {case['entry_date']} · 退出 {case['exit_date']}</p></div><strong class="loss">{_money(case['observed_facts']['pnl'])}</strong></header>
<div class="zoom"><button data-chart="{html.escape(case['case_id'])}" data-zoom="in">放大</button><button data-chart="{html.escape(case['case_id'])}" data-zoom="out">缩小</button><button data-chart="{html.escape(case['case_id'])}" data-zoom="reset">重置</button><span>可用按钮缩放图表；悬停烛线和标记可查看事实。</span></div>
<div class="chart-wrap">{case['chart_svg']}</div>
<div class="cols"><section class="machine"><h3>机器证据（归档快照和冻结行情）</h3><dl>
<dt>形态 / 规则</dt><dd>{html.escape(case['pattern_type'])} / {html.escape(str(geometry.get('rule_version') or 'unknown'))}</dd>
<dt>结构形成 / 确认</dt><dd>{html.escape(str(geometry.get('formed_at')))} / {html.escape(str(case['signal_date']))}</dd>
<dt>结构已知 / 颈线冻结</dt><dd>{html.escape(str(geometry.get('structure_known_at')))} / {html.escape(str(geometry.get('line_frozen_at')))}</dd>
<dt>所选线 / 压力测试</dt><dd>候选 {html.escape(str(geometry.get('selected_candidate_id')))}；{geometry.get('independent_test_count')} 个独立测试：{html.escape(tests)}</dd>
<dt>确认收盘 / 最低要求</dt><dd>{_format_num(observed.get('breakout_close'))} / {_format_num(observed.get('breakout_required_close'))}</dd>
<dt>入场 / 出场 / 持有</dt><dd>{_format_num(observed.get('entry_price'))} / {_format_num(observed.get('exit_price'))} / {case['holding_sessions']} 个交易日</dd>
<dt>止损 / 目标</dt><dd>{_format_num(observed.get('stop_price'))} / {_format_num(observed.get('target_price'))}</dd>
<dt>严格区间内 MFE</dt><dd>{html.escape(mfe_value)}（排除入场和退出日）</dd>
<dt>开盘门审计</dt><dd>{html.escape(gate_text)}</dd>
<dt>突破后收盘位置</dt><dd>{html.escape(below_text)}</dd>
<dt>成熟前突破证据</dt><dd>{html.escape(early_text)}</dd>
<dt>修正引擎信号日回放</dt><dd>{html.escape(corrected_text)}</dd>
<dt>止损跳空</dt><dd>开盘 {_format_num(gap['exit_open'])}；止损 {_format_num(gap['stop_price'])}；成交 {_format_num(gap['exit_fill'])}；缺口 {_format_num(gap['gap_points'])}（{gap['gap_pct_of_stop']:.2%}）</dd>
</dl></section><section class="review"><h3>视觉复核与解释（主观，不是人工真值）</h3><p><b>图上观察：</b>{html.escape(manual['read'])}</p><p><b>解释：</b>{html.escape(manual['interpretation'])}</p><p><b>后续可检验建议：</b>{html.escape(manual['recommendation'])}</p><p class="caveat">观察来自本页 K 线与归档字段。没有引入独立人工标注；解释不改变历史成交或报告中的机器结果。</p></section></div>
<details><summary>事件与几何原始数据</summary><pre>{html.escape(json.dumps({k:v for k,v in case.items() if k not in ('chart_svg',)},ensure_ascii=False,indent=2,default=str))}</pre></details></article>"""


def _money(value):
    return "¥{:,.0f}".format(float(value))


def _render_html(payload):
    aggregate = payload["aggregate_loss_statistics"]
    losses = aggregate["loss_trade_count"]
    stop = aggregate["stop_loss_count"]
    cat = aggregate
    overlap = json.dumps(cat["archetype_overlap_counts"], ensure_ascii=False, indent=2)
    cases = "".join(_case_html(case) for case in payload["cases"])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五个典型亏损交易 — 2026-11-05</title>
<style>
:root{{color-scheme:light;--ink:#17212b;--muted:#56616d;--line:#d8e0e8;--paper:#f4f7fa;--blue:#1d4ed8;--red:#b91c1c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1440px;margin:24px auto;padding:0 22px}}h1{{font-size:32px;line-height:1.2;margin:0 0 8px}}h2{{font-size:24px;margin:5px 0}}h3{{margin:0 0 12px;font-size:18px}}p{{margin:8px 0}}.lede,.summary,.case{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:16px 0}}.fact{{background:white;border:1px solid var(--line);border-radius:10px;padding:14px}}.fact b{{display:block;font-size:24px}}.muted,.eyebrow,.caveat{{color:var(--muted)}}.eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase}}.note{{border-left:4px solid #b45309;padding:10px 14px;background:#fffbeb}}
.case header{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding-bottom:12px}}.loss{{font-size:26px;color:var(--red);white-space:nowrap}}.chart-wrap{{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:8px}}svg.price-chart{{display:block;width:100%;min-width:850px;height:auto;touch-action:pan-x pan-y}}.axis{{font:12px system-ui,sans-serif;fill:#475569}}.label{{font:12px system-ui,sans-serif;fill:#1e293b}}.marker{{font:11px system-ui,sans-serif;fill:#334155}}.zoom{{display:flex;align-items:center;gap:7px;margin:12px 0}}button{{border:1px solid #aab6c4;border-radius:6px;padding:5px 10px;background:#fff;color:#17212b;cursor:pointer}}button:hover{{background:#edf4ff}}.zoom span{{margin-left:6px;color:var(--muted);font-size:13px}}.cols{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,.9fr);gap:14px;margin-top:14px}}.cols section{{padding:15px;border-radius:9px}}.machine{{background:#f5f8ff;border:1px solid #ccdaf8}}.review{{background:#fffaf0;border:1px solid #f4d8a2}}dl{{display:grid;grid-template-columns:minmax(130px,.6fr) minmax(0,1.4fr);gap:7px 12px;margin:0}}dt{{font-weight:650;color:#344054}}dd{{margin:0;overflow-wrap:anywhere}}.caveat{{font-size:13px;border-top:1px solid #efdba9;padding-top:10px;margin-top:14px}}details{{margin-top:14px}}pre{{max-height:430px;overflow:auto;background:#f6f8fa;border:1px solid var(--line);padding:12px;border-radius:7px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid var(--line)}}@media(max-width:800px){{main{{padding:0 12px}}.cols{{grid-template-columns:1fr}}.case header{{display:block}}.loss{{display:block;margin-top:8px}}}}
</style><main><h1>五个典型亏损交易</h1><p class="muted">用冻结的 1,000 标的基线配对账户和同一输入快照复核已实现亏损。旧版成交保留为历史事实；案例不推断修正引擎会在同一日期发出买入。</p>
<section class="lede"><p><b>数据边界：</b>baseline v8 四型策略，8%止损 / 20%止盈，日期截止 2026-10-30。账户输入来自 {html.escape(payload['snapshot_id'])}；52 个变更标的使用同一快照重跑配对基线，948 个未变账户复用冻结报告。<a href="daily-research/paired-input-snapshot-20261105-v1/snapshot-manifest.json">输入快照清单</a> · <a href="daily-research/paired-repairs-20261105-snapshot-v1/paired-repair-manifest.json">52 标的修复清单</a></p><p class="note">机器证据与视觉复核分栏。视觉部分是本报告的主观解释，不是人工真值或独立盲标注。MFE 严格排除入场日和退出日；止损跳空可以超过名义 8%。</p></section>
<section class="summary"><h2>全样本亏损概况</h2><div class="facts">
<div class="fact"><b>{aggregate['trade_count']:,}</b><span>实际成交</span></div><div class="fact"><b>{losses:,} ({aggregate['loss_rate']:.1%})</b><span>已实现亏损成交</span></div><div class="fact"><b>{stop:,} ({aggregate['stop_loss_rate_of_losses']:.1%})</b><span>亏损中的止损成交</span></div><div class="fact"><b>{_money(aggregate['realized_loss_pnl_sum'])}</b><span>亏损成交合计 PnL</span></div><div class="fact"><b>{_money(aggregate['average_realized_loss'])}</b><span>每笔亏损均值</span></div><div class="fact"><b>{aggregate['rapid_stop_le_5_sessions']['count']:,} ({aggregate['rapid_stop_le_5_sessions']['share_of_stop_losses']:.1%})</b><span>≤5 持有日止损</span></div><div class="fact"><b>{aggregate['strict_interior_mfe_at_least_10pct_stop']['count']:,} ({aggregate['strict_interior_mfe_at_least_10pct_stop']['share_of_stop_losses']:.1%})</b><span>严格区间内 MFE ≥10% 后止损</span></div><div class="fact"><b>{aggregate['gap_through_stop']['count']:,} ({aggregate['gap_through_stop']['share_of_stop_losses']:.1%})</b><span>开盘跳过止损价并按开盘成交</span></div><div class="fact"><b>{aggregate['entry_open_more_than_3_signal_atr_above_line']['count']:,} ({aggregate['entry_open_more_than_3_signal_atr_above_line']['share_of_stop_losses']:.1%})</b><span>入场开盘高于冻结线 >3×信号 ATR</span></div>
</div><p>快止损、MFE 后止损、跳空止损和延伸开盘分类可以重叠。下面给出这些分类的交叉计数：</p><details><summary>分类交叉矩阵</summary><pre>{html.escape(overlap)}</pre></details><p class="muted">这些是交易路径统计，不是预测标签；分类说明发生了什么，不单独证明某条门槛会提升收益。</p></section>
<h2>五个案例</h2>{cases}
<section class="summary"><h2>引擎版本边界</h2><p>{html.escape(payload.get('engine_verdict_note', '修正引擎五例回放待确认。'))}</p><p class="muted">本报告的成交、PnL 和路径指标都来自冻结的 baseline 配对账户。即使后续修正引擎不再发出某个事件，历史成交仍是旧规则实测结果；不把新规则假设成同日成交或更好收益。</p></section>
<footer class="muted">生成时间 {html.escape(payload['generated_at'])} · 规范化 OHLC 行情由冻结 SHA-256 输入映射校验 · 案例选择规则：脚本内固定 event key 且必须满足相应可观察证据。</footer></main>
<script>
(()=>{{
 const base=new Map(); document.querySelectorAll('svg.price-chart').forEach(svg=>base.set(svg.id,svg.getAttribute('viewBox').split(' ').map(Number)));
 document.querySelectorAll('button[data-chart]').forEach(button=>button.addEventListener('click',()=>{{
  const svg=document.getElementById('chart-'+button.dataset.chart);if(!svg)return;const orig=base.get(svg.id);let [x,y,w,h]=svg.getAttribute('viewBox').split(' ').map(Number);const action=button.dataset.zoom;
  if(action==='reset'){{svg.setAttribute('viewBox',orig.join(' '));return;}}
  const factor=action==='in'?.72:1/.72;w=Math.min(orig[2],Math.max(orig[2]*.18,w*factor));h=Math.min(orig[3],Math.max(orig[3]*.18,h*factor));x=Math.max(0,Math.min(orig[2]-w,x+(orig[2]-w)/2));y=Math.max(0,Math.min(orig[3]-h,y+(orig[3]-h)/2));svg.setAttribute('viewBox',[x,y,w,h].join(' '));
 }}));
 document.querySelectorAll('svg.price-chart').forEach(svg=>svg.addEventListener('wheel',event=>{{event.preventDefault();const button=document.querySelector('button[data-chart="'+svg.id.replace('chart-','')+'"][data-zoom="'+(event.deltaY<0?'in':'out')+'"]');button?.click();}},{{passive:false}}));
}})();
</script></html>"""


def _open_gate_lookup(path):
    if not Path(path).is_file():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("open_admission_gate_audit", [])


def build_report():
    if not BASELINE.is_dir() or not SOURCE_MAP_PATH.is_file():
        raise FileNotFoundError("paired baseline composite and data-source map are required")
    source_map_payload = json.loads(SOURCE_MAP_PATH.read_text(encoding="utf-8"))
    if source_map_payload.get("snapshot_id") != EXPECTED_SOURCE:
        raise RuntimeError("unexpected frozen input source id")
    source_map = source_map_payload.get("sources", {})
    research = json.loads(RESEARCH_REVIEW_PATH.read_text(encoding="utf-8"))
    if research.get("snapshot_id") != EXPECTED_SOURCE or research.get("account_count") != 1000:
        raise RuntimeError("the paired baseline research review is missing or has the wrong source")
    open_gate_audit = _open_gate_lookup(RESEARCH_REVIEW_PATH)
    aggregate, all_rows, verified_hashes = _compute_case_rows(source_map, open_gate_audit)
    selected = _select_five_cases(all_rows)
    labels = {
        "premature_recross": "成熟期前已越线，之后回穿",
        "rapid_false_break": "快速假突破后止损",
        "extended_entry": "高开过度延伸入场",
        "mfe_then_stop": "出现 ≥10% MFE 后止损",
        "gap_stop": "开盘跳过止损价（与 MFE 分类重叠）",
    }
    cases = []
    source_hashes = {}
    for number, row in enumerate(selected, 1):
        row["case_number"] = number
        row["title"] = "%s %s · %s" % (row["code"], row["pattern_type"].replace("_", " "), row["signal_date"])
        row["archetype_label"] = labels[row["case_id"]]
        row["manual_assessment"] = MANUAL_ASSESSMENTS[row["case_id"]]
        source = row.pop("_source")
        bars = row.pop("_bars")
        row["snapshot_provenance"] = {key: source.get(key) for key in
                                      ("snapshot_id", "snapshot_relative_path", "snapshot_sha256",
                                       "snapshot_data_revision", "account_data_revision")}
        source_hashes["%s:%s" % (row["kind"], row["code"])] = source.get("snapshot_sha256")
        row["chart_svg"] = _svg_chart({**row, "_bars": bars})
        cases.append(row)
    pairs = source_map_payload.get("paired_account_count", None)
    payload = {
        "report_version": "loss_five_cases_20261105_v1",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source_archive": str(BASELINE.resolve()),
        "source_map": str(SOURCE_MAP_PATH.resolve()),
        "snapshot_id": EXPECTED_SOURCE,
        "account_count": aggregate["account_count"],
        "paired_changed_account_count": pairs or 52,
        "verified_frozen_csv_count": len(verified_hashes),
        "baseline_engine_version": json.loads((BASELINE / "run-manifest.json").read_text(
            encoding="utf-8")).get("engine_version"),
        "aggregate_loss_statistics": aggregate,
        "cases": cases,
        "case_source_hashes": source_hashes,
        "engine_verdict_note": "Corrected-engine replay pending the isolated first-buffered-break retirement change; historical baseline fills remain unchanged.",
        "method": {
            "selection": "Five fixed kind/code/signal-date event keys, each required to pass its category evidence predicate; no outcome-only case substitution.",
            "loss_definition": "realized trade pnl < 0 in the paired 1,000-account baseline through 2026-10-30",
            "mfe_definition": "maximum high from the session after entry through the session before exit; entry and exit sessions excluded",
            "gap_stop_definition": "stop_loss fill equals exit-session open, which is below stop_price by more than half a tick",
            "false_break_definition": "saved signal close exceeds saved required buffered confirmation close, followed by a post-entry close below the projected selected frozen line before/at stop exit",
            "visual_review": "qualitative chart reading recorded separately from machine evidence; not human ground truth",
            "no_future_reasoning": "No recommendation is justified by subsequent returns beyond each observed trade exit.",
            "verified_input_hash_count": len(verified_hashes),
        },
    }
    return payload


def main():
    payload = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_serializable_payload(payload), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    OUT_HTML.write_text(_render_html(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": [
        {"case_id": row["case_id"], "kind": row["kind"], "code": row["code"],
         "signal_date": row["signal_date"], "pnl": row["observed_facts"]["pnl"]}
        for row in payload["cases"]], "aggregate": payload["aggregate_loss_statistics"],
        "verified_frozen_csv_count": payload["verified_frozen_csv_count"],
        "html": str(OUT_HTML.resolve()), "json": str(OUT_JSON.resolve())},
        ensure_ascii=False, indent=2))


def _serializable_payload(payload):
    cleaned = dict(payload)
    for case in payload.get("cases", []):
        # The internal case rows already omit barframes/pattern references.
        pass
    return cleaned


if __name__ == "__main__":
    main()
