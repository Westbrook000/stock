#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a standalone measured-target paired replay report with five real losses."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / "research/experiments/structure_boundary_losses_20261112/dynamic_reline/results/baseline-A-perf2-extrema-10w-attempt02"
SNAPSHOT = Path("/private/tmp/caisen-structure-boundary-snapshot-20261112-v2")
ECHARTS = ROOT / "caisen/static/vendor/echarts.min.js"
BOOK = ROOT / "2_多空转折一手抓(高清)_副本.md"
DEFAULT_RUN = ROOT / "reports/stop8-measured-target-100-20261116"

LABELS = {"w_bottom": "W 底", "inverse_head_shoulders": "头肩底", "triangle_bottom": "收敛三角形底部", "bull_flag": "下飘旗形上涨中继"}
DATE_INDEX = {}


def _date_indices(code, kind):
    key = (code, kind)
    if key not in DATE_INDEX:
        group = "etfs" if kind == "etf" else "stocks"
        frame = pd.read_csv(SNAPSHOT / "data" / group / f"{code}.csv", usecols=["date"])
        DATE_INDEX[key] = {str(d)[:10]: i for i, d in enumerate(frame.date)}
    return DATE_INDEX[key]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence_value(pattern, code):
    for item in pattern.get("evidence", []) or []:
        if item.get("code") == code:
            return item.get("value")
    return None


def tag_value(signal, code, pattern_id):
    for tag in (signal.get("pattern_tags") or signal.get("tags") or []):
        if tag.get("pattern_id") != pattern_id:
            continue
        val = tag.get(code)
        if val is not None:
            return val
    return None


def candidates(run_dir):
    result = []
    for account_file in sorted((run_dir / "accounts").glob("stock-*.json")):
        account = load(account_file)
        signal_by_id = {}
        for signal in account["signals"]:
            pattern = signal.get("pattern") or {}
            if pattern.get("pattern_id"):
                signal_by_id[pattern["pattern_id"]] = signal
        for trade in account["trades"]:
            if float(trade.get("pnl", 0)) < 0:
                exact_id = trade.get("measured_target_source_pattern_id") or (trade.get("pattern_ids") or [None])[0]
                signal = signal_by_id.get(exact_id)
                if signal:
                    result.append((account, trade, signal))
    return result


def geom(signal):
    p = signal.get("pattern") or {}
    kp = p.get("key_points") or {}
    pts = []
    for key, value in kp.items():
        m = re.fullmatch(r"pivot_(\d+)_date", key)
        if m:
            stem = f"pivot_{m.group(1)}"
            kind = kp.get(stem + "_kind")
            price = kp.get(stem)
            if value and price is not None:
                pts.append({"date": str(value), "price": float(price), "kind": kind, "index": kp.get(stem + "_index")})
    pts.sort(key=lambda x: (x["date"], x["kind"] or ""))
    lows = [x for x in pts if x["kind"] == "low"]
    highs = [x for x in pts if x["kind"] == "high"]
    neck = p.get("neckline") or {}
    selection = tag_value(signal, "neckline_selection", p.get("pattern_id")) or evidence_value(p, "neckline_selection") or {}
    tests = evidence_value(p, "neckline_tests") or []
    if not tests:
        tests = selection.get("independent_tests") or []
    candidates_data = selection.get("candidates") or []
    reported_candidate_count = max([int(c.get("candidate_count", 0)) for c in tests] + [0])
    diag = {"pattern_type": p.get("pattern_type"), "points": pts, "lows": lows, "highs": highs,
            "neckline": neck, "selection": selection, "tests": tests,
            "candidate_count": reported_candidate_count or None,
            "candidate_summary_count": len(candidates_data), "selected_rank": selection.get("selected_rank"),
            "selected_candidate_id": selection.get("selected_candidate_id")}
    if len(lows) >= 2:
        diag["low_gap_bars"] = abs(float(lows[-1].get("index") or 0) - float(lows[0].get("index") or 0))
        diag["low_level_gap_pct"] = abs(lows[-1]["price"] - lows[0]["price"]) / max(abs(lows[0]["price"]), 1e-9)
        signal_index = (signal.get("pattern") or {}).get("confirmed_at")
        if signal_index is not None:
            diag["elapsed_to_confirmation_bars"] = max(0, float(signal_index) - float(lows[0].get("index") or signal_index))
            diag["core_span_ratio"] = diag["low_gap_bars"] / max(diag["elapsed_to_confirmation_bars"], 1)
    if len(lows) >= 2 and highs:
        diag["measured_height"] = max(0.0, min(x["price"] for x in highs) - min(lows[0]["price"], lows[-1]["price"]))
        diag["formation_span_bars"] = max(float(x.get("index") or 0) for x in pts) - min(float(x.get("index") or 0) for x in pts)
        if diag["formation_span_bars"] > 0:
            diag["core_span_ratio"] = diag["low_gap_bars"] / diag["formation_span_bars"]
    if p.get("pattern_type") == "inverse_head_shoulders" and len(lows) >= 3:
        diag["shoulder_asymmetry_pct"] = abs(lows[0]["price"] - lows[-1]["price"]) / max(abs((lows[0]["price"] + lows[-1]["price"]) / 2), 1e-9)
        diag["head_depth_pct"] = (min(lows[0]["price"], lows[-1]["price"]) - lows[1]["price"]) / max(abs(lows[1]["price"]), 1e-9)
        dated_tests = [t for t in tests if t.get("pivot_date")]
        diag["shoulder_to_head_tests"] = sum(1 for t in dated_tests if lows[0]["date"] <= t["pivot_date"] <= lows[1]["date"])
        diag["head_to_shoulder_tests"] = sum(1 for t in dated_tests if lows[1]["date"] <= t["pivot_date"] <= lows[2]["date"])
        diag["outside_ihs_tests"] = max(0, len(dated_tests) - diag["shoulder_to_head_tests"] - diag["head_to_shoulder_tests"])
    elif p.get("pattern_type") == "w_bottom" and len(lows) >= 2:
        dated_tests = [t for t in tests if t.get("pivot_date")]
        diag["inside_w_core_tests"] = sum(1 for t in dated_tests if lows[0]["date"] <= t["pivot_date"] <= lows[-1]["date"])
        diag["outside_w_core_tests"] = max(0, len(dated_tests) - diag["inside_w_core_tests"])
    coverage = selection.get("formation_coverage")
    diag["formation_coverage"] = coverage
    selected = (signal.get("pattern") or {}).get("pattern_id")
    kind = signal.get("kind", "stock")
    indices = _date_indices(signal["code"], kind)
    effective_date = selection.get("effective_start_date")
    freeze_date = selection.get("freeze_date")
    observation_date = signal.get("signal_date")
    if neck.get("slope") is not None and effective_date in indices and freeze_date in indices:
        start_px = neck["slope"] * indices[effective_date] + neck["intercept"]
        freeze_px = neck["slope"] * indices[freeze_date] + neck["intercept"]
        diag["neckline_change_pct"] = (freeze_px / start_px - 1) if start_px else None
        diag["effective_start_date"] = effective_date
        diag["freeze_date"] = freeze_date
        diag["neckline_change_price"] = freeze_px - start_px
        eff = pd.Timestamp(effective_date)
        frz = pd.Timestamp(freeze_date)
        diag["touches_inside_frozen_span"] = sum(1 for t in tests
            if t.get("pivot_date") and eff <= pd.Timestamp(t["pivot_date"]) <= frz)
        diag["touches_outside_frozen_span"] = max(0, len(tests) - diag["touches_inside_frozen_span"])
        diag["neckline_start_price"] = start_px
        diag["neckline_freeze_price"] = freeze_px
    target_meta = evidence_value(p, "targets") or {}
    breakout_meta = evidence_value(p, "neckline_breakout") or {}
    apex_index = target_meta.get("triangle_apex_index")
    formation_start = kp.get("first_trough_index")
    breakout_index = breakout_meta.get("index")
    if apex_index is not None and formation_start is not None and breakout_index is not None and float(apex_index) > float(formation_start):
        diag["triangle_apex_index"] = float(apex_index)
        diag["triangle_breakout_position"] = (float(breakout_index) - float(formation_start)) / (float(apex_index) - float(formation_start))
        diag["triangle_base_width"] = target_meta.get("triangle_base_width")
    diag["signal_global_index"] = indices.get(observation_date)
    diag["higher_test_alternatives"] = sum(1 for c in candidates_data
                                           if c.get("independent_test_count", 0) > len(tests))
    return diag


def select_five(losses):
    geoms = [(x, geom(x[2])) for x in losses]
    picked = []
    used = set()

    def take(name, pool, score):
        pool = [x for x in pool if x[0][1].get("pattern_ids") and x[0][1]["pattern_ids"][0] not in used]
        recent_pool = [x for x in pool if recent(x)]
        if recent_pool:
            pool = recent_pool
        if not pool:
            return False
        item, g = max(pool, key=score)
        g["case_focus"] = name
        picked.append((item, g))
        used.add(item[1]["pattern_ids"][0])
        return True

    recent = lambda pair: pair[0][2].get("signal_date", "") >= "2015-01-01"
    def pref(pool):
        recent_pool = [z for z in pool if recent(z)]
        return recent_pool or pool
    take("W 底：双底核心长度与确认等待", [(x, g) for x, g in geoms if g["pattern_type"] == "w_bottom"],
         lambda z: -(z[1].get("core_span_ratio", 1e6)))
    take("W 底：颈线候选与压力触点", [(x, g) for x, g in geoms if g["pattern_type"] == "w_bottom"],
         lambda z: (z[1].get("higher_test_alternatives", 0), z[1].get("candidate_count", 0), -float(z[1].get("selected_rank") or 0)))
    take("冻结颈线斜率", [(x, g) for x, g in geoms if g.get("neckline", {}).get("slope", 0) < 0],
         lambda z: abs(z[1].get("neckline_change_pct", 0)))
    take("头肩底：肩部对称度", [(x, g) for x, g in geoms if g["pattern_type"] == "inverse_head_shoulders"],
         lambda z: z[1].get("shoulder_asymmetry_pct", 0))
    take("三角底或牛旗：边界与目标", [(x, g) for x, g in geoms if g["pattern_type"] in ("triangle_bottom", "bull_flag")],
         lambda z: (z[1]["pattern_type"] == "bull_flag", abs(z[1].get("neckline_change_pct", 0))))
    # Keep the requested count exact when a cohort has fewer than one of the
    # preferred structures; any fallback still uses an actual losing trade.
    for item, g in sorted(geoms, key=lambda z: z[0][1]["pnl"]):
        if len(picked) == 5:
            break
        if item[1].get("pattern_ids", [None])[0] not in used:
            g["case_focus"] = "未覆盖形态的最大实际亏损"
            picked.append((item, g))
            used.add(item[1].get("pattern_ids", [None])[0])
    return picked[:5]


def _safe(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def shape_summary(directory, is_measured=False):
    groups = {}
    for p in sorted((Path(directory) / "accounts").glob("stock-*.json")):
        account = load(p)
        for trade in account.get("trades", []):
            ptype = (trade.get("pattern_types") or ["unknown"])[0]
            if is_measured:
                source_id = trade.get("measured_target_source_pattern_id")
                if source_id is None or source_id != (trade.get("pattern_ids") or [None])[0]:
                    raise ValueError(f"{account['code']}: target source does not match primary saved pattern")
            row = groups.setdefault(ptype, {"trades": 0, "wins": 0, "pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0})
            pnl = float(trade.get("pnl", 0))
            row["trades"] += 1
            row["wins"] += int(pnl > 0)
            row["pnl"] += pnl
            if pnl > 0:
                row["gross_win"] += pnl
            elif pnl < 0:
                row["gross_loss"] -= pnl
    for row in groups.values():
        row["win_rate"] = row["wins"] / row["trades"] if row["trades"] else None
        row["profit_factor"] = row["gross_win"] / row["gross_loss"] if row["gross_loss"] else None
    return groups


def case_payload(account, trade, signal, diagnostic):
    code, kind = account["code"], account.get("kind", "stock")
    group = "etfs" if kind == "etf" else "stocks"
    frame = pd.read_csv(SNAPSHOT / "data" / group / f"{code}.csv")
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.loc[frame.date <= pd.Timestamp(trade["exit_date"])].reset_index(drop=False)
    # Start at the earliest real structure anchor when present; retain enough
    # pre-signal bars to see the run into the setup, and all bars through exit.
    dates = [pd.Timestamp(p["date"]) for p in diagnostic["points"] if p.get("date")]
    start = min(dates) if dates else pd.Timestamp(trade["signal_date"])
    begin = max(pd.Timestamp(frame.date.iloc[0]), min(start, pd.Timestamp(trade["signal_date"]) - pd.Timedelta(days=220)))
    data = frame.loc[frame.date >= begin].copy()
    max_bars = 1800
    if len(data) > max_bars:
        # Uniformly keep the time axis usable while retaining both endpoints.
        keep = set(int(i) for i in pd.Series(range(max_bars)).mul((len(data)-1)/(max_bars-1)).round())
        important_dates = {trade["signal_date"], trade["entry_date"], trade["exit_date"]}
        important_dates.update(p["date"] for p in diagnostic["points"])
        for date in important_dates:
            found = data.index[data.date.dt.strftime("%Y-%m-%d") == str(date)]
            if len(found):
                keep.add(int(found[0]))
        data = data.iloc[sorted(keep)].copy()
    bars = [{"date": row.date.strftime("%Y-%m-%d"), "index": int(row.index),
             "open": float(row.open), "high": float(row.high), "low": float(row.low),
             "close": float(row.close), "volume": float(row.volume)} for row in data.itertuples(index=False)]
    entry_global = next((b["index"] for b in bars if b["date"] == trade["entry_date"]), None)
    neck = diagnostic.get("neckline") or {}
    entry_neck = (float(neck["slope"]) * entry_global + float(neck["intercept"])) if entry_global is not None and neck.get("slope") is not None and neck.get("intercept") is not None else None
    close_breaches = []
    if entry_global is not None and neck.get("slope") is not None and neck.get("intercept") is not None:
        for b in bars:
            if b["date"] < trade["entry_date"]:
                continue
            if b["close"] < float(neck["slope"]) * b["index"] + float(neck["intercept"]):
                close_breaches.append(b)
                break
    exit_bar = next((b for b in bars if b["date"] == trade["exit_date"]), None)
    exit_open_vs_stop = (exit_bar["open"] / float(trade["stop_price"]) - 1) if exit_bar and trade.get("stop_price") else None
    target = trade.get("measured_target_price")
    reward_pct = (float(target) / float(trade["entry_price"]) - 1) if target is not None else None
    return {"code": code, "type": LABELS.get((signal.get("pattern") or {}).get("pattern_type"), "其他"),
            "signal_date": trade["signal_date"], "entry_date": trade["entry_date"],
            "entry_price": trade["entry_price"], "exit_date": trade["exit_date"],
            "exit_price": trade["exit_price"], "exit_reason": trade["exit_reason"],
            "pnl": trade["pnl"], "return_pct": trade["return_pct"],
            "stop_price": trade["stop_price"], "target_price": trade.get("measured_target_price"),
            "target_status": trade.get("measured_target_status"),
            "target_source_pattern_id": trade.get("measured_target_source_pattern_id"),
            "entry_neckline_price": entry_neck,
            "entry_neckline_gap_pct": (float(trade["entry_price"]) / entry_neck - 1) if entry_neck else None,
            "first_close_below_line_date": close_breaches[0]["date"] if close_breaches else None,
            "exit_open_vs_stop_pct": exit_open_vs_stop,
            "target_reward_pct": reward_pct,
            "target_reward_r": reward_pct / 0.08 if reward_pct is not None else None,
            "pattern_id": (signal.get("pattern") or {}).get("pattern_id"),
            "diagnostic": diagnostic, "bars": bars}


def narrative(case):
    d = case["diagnostic"]
    out = [f"实际结构：{case['type']}；信号 {case['signal_date']}，次日开盘成交 {case['entry_date']} @ {case['entry_price']:.4g}，{case['exit_date']} 以 {case['exit_reason']} 离场。"]
    if d.get("low_gap_bars") is not None:
        out.append(f"两端低点间距 {d['low_gap_bars']:.0f} 个交易条；首低到确认共 {d.get('elapsed_to_confirmation_bars', 0):.0f} 条；低点价差 {d.get('low_level_gap_pct', 0):.1%}。")
    if d.get("pattern_type") == "inverse_head_shoulders" and d.get("shoulder_asymmetry_pct") is not None:
        out.append(f"两侧肩低点价差约 {d['shoulder_asymmetry_pct']:.1%}；中间头部低于较低肩部约 {d.get('head_depth_pct', 0):.1%}（以头部价作分母）。")
    if d.get("neckline", {}).get("slope") is not None:
        out.append(f"冻结线斜率 {d['neckline']['slope']:.6g} / bar；绘图按存档全局日期索引回算。")
    if d.get("neckline_change_pct") is not None:
        out.append(f"有效起点 {d['effective_start_date']} 到冻结日 {d['freeze_date']}，线价变化 {d['neckline_change_price']:+.4g} ({d['neckline_change_pct']:+.1%})；颈线压力测试共 {len(d.get('tests', []))} 个，不能直接当作形态内部反弹数。")
    if d.get("pattern_type") == "w_bottom" and d.get("inside_w_core_tests") is not None:
        out.append(f"按双底两低点划分，颈线压力触点中 {d['inside_w_core_tests']} 个落在 W 核心区间，{d['outside_w_core_tests']} 个落在区间外；区间外触点可作候选匹配线索，不能算作 W 内部反弹确认。")
    if d.get("pattern_type") == "inverse_head_shoulders" and d.get("shoulder_to_head_tests") is not None:
        out.append(f"按左肩→头→右肩分段，颈线压力触点分别有 {d['shoulder_to_head_tests']} 个、{d['head_to_shoulder_tests']} 个，另有 {d['outside_ihs_tests']} 个在形态外。")
    if d.get("pattern_type") == "triangle_bottom" and d.get("triangle_breakout_position") is not None:
        out.append(f"存档三角顶点估计全局索引 {d['triangle_apex_index']:.1f}；确认突破位于结构起点至顶点区间的 {d['triangle_breakout_position']:.1%}，基底宽度 {d.get('triangle_base_width', 0):.4g}。这不表示三角形目标受 20% 上限约束。")
    if d.get("selected_rank") is not None:
        count_text = f"在 {d['candidate_count']} 个候选中" if d.get("candidate_count") else "（未提供候选总数）"
        out.append(f"颈线候选选择记录：rank {d['selected_rank']} {count_text}；存档候选摘要 {d.get('candidate_summary_count', 0)} 条，独立压力触点 {len(d.get('tests', []))} 个。")
    if case.get("target_status") == "pending_prior_high":
        out.append("等幅目标在入场时待解锁；该次持仓中保持待解锁并由 8% 止损退出。")
    elif case.get("target_price") is not None:
        out.append(f"入场后冻结的形态目标 {case['target_price']:.4g}（目标收益 {case['target_reward_pct']:.1%}，约 {case['target_reward_r']:.2f}R）；结算结果显示其未先触发。")
    if case.get("entry_neckline_price") is not None:
        out.append(f"按入场日期全局索引回算颈线约 {case['entry_neckline_price']:.4g}；入场价较线高 {case['entry_neckline_gap_pct']:+.1%}。")
    if case.get("first_close_below_line_date"):
        out.append(f"持仓期首个收盘低于该冻结线的日期为 {case['first_close_below_line_date']}；这是价格路径观察，不证明颈线错误。")
    if case.get("exit_open_vs_stop_pct") is not None:
        out.append(f"止损离场日开盘相对止损价 {case['exit_open_vs_stop_pct']:+.1%}，离场成交 {case['exit_price']:.4g}；反映止损日的缺口/撮合结果。")
    cov = d.get("formation_coverage")
    if isinstance(cov, dict):
        out.append(f"存档覆盖检查 {'通过' if cov.get('passed') else '未通过'}；最大越界数值 {cov.get('maximum_excess', 0):.4g}。")
    focus = d.get("case_focus", "")
    suggestions = {
        "W 底：双底核心长度与确认等待": "待验证的机制：确认等待远长于双底核心，结构可能已变陈旧。后续可比较不同等待时长的失效率；若加时限，会漏掉长时间筑底后有效突破的 W 底。",
        "W 底：颈线候选与压力触点": "待验证的机制：候选排序与压力点的分布可能影响颈线位置。应对照所有可行线及被选线的确认价差；加严触点要求会漏掉触点少但结构有效的形态。",
        "冻结颈线斜率": "待验证的机制：冻结期间线价显著漂移，可能使同一结构的确认位置变敏感。应比较入场价相对冻结线及备选线的距离；限制斜率会漏掉真实倾斜颈线。",
        "头肩底：肩部对称度": "待验证的机制：左右肩差异可能代表尺度/锚点配对不稳。应在同尺度下测肩部偏差和头部突出度；强制对称会漏掉不规则但有效的头肩底。",
        "三角底或牛旗：边界与目标": "待验证的机制：三角边界所处位置或牛旗目标解锁状态可能影响盈亏分布。先分别核对边界触点与入场目标风险比；提高目标门槛可能漏掉短目标但高胜率的反转。",
    }
    out.append(suggestions.get(focus, "待验证的几何匹配机制：对比存档候选边界、确认时的边界距离与未入选候选；任何筛选都需要报告被排除的有效案例比例。"))
    out.append("这些是可量化的几何线索，用于复核识别与匹配；单笔亏损本身不能证明形态识别错误或构成亏损原因。")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    summary = load(args.run / "daily-research.json")
    if summary.get("status") != "complete" or summary.get("completed_accounts") != 100:
        raise SystemExit("The paired 100-account measured-target replay is not complete")
    summary["strategy_version"] = "daily_four_patterns_measured_target_v2"
    summary["target_fraction"] = None
    summary["target_method"] = "signal_pattern_first_measured_target"
    account_files = sorted((args.run / "accounts").glob("stock-*.json"))
    summary["frozen_revision_match_count"] = sum(bool(load(p).get("provenance", {}).get("data_revision")) for p in account_files)
    rejection_counts = {}
    for p in sorted((args.run / "accounts").glob("stock-*.json")):
        a = load(p)
        for rejection in a.get("rejections", []):
            reason = rejection.get("reason", "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    summary["aggregate"]["rejection_reason_counts"] = dict(sorted(rejection_counts.items()))
    summary["aggregate"]["missing_measured_target_count"] = rejection_counts.get("missing_measured_target", 0)
    summary["aggregate"]["target_rejected_not_above_entry_count"] = rejection_counts.get("measured_target_not_above_entry", 0)
    (args.run / "report-metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    echart = ECHARTS.read_text(encoding="utf-8")
    losses = candidates(args.run)
    selected = select_five(losses)
    cases = [case_payload(*item, diagnostic) for item, diagnostic in selected]
    out = args.output or args.run / "stop8-measured-target-100-20261116.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The prose source pointers below are section headings in the bundled OCR
    # of the user's book; the report paraphrases the method and does not quote it.
    book_text = BOOK.read_text(encoding="utf-8")
    book_refs = []
    for heading in ("一、W底涨幅满足计算：多头买进信号", "十、头肩底涨幅满足计算：多头买进信号", "十二、收敛三角形底部涨幅满足：多头买进信号"):
        if heading in book_text:
            book_refs.append(heading)
    payload = {"summary": summary, "cases": cases, "book_refs": book_refs}
    measured_shapes = shape_summary(args.run, is_measured=True)
    baseline_shapes = shape_summary(BASELINE, is_measured=False)
    shape_rows = []
    for ptype in ("w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"):
        left = baseline_shapes.get(ptype, {})
        right = measured_shapes.get(ptype, {})
        shape_rows.append(f"<tr><td>{LABELS.get(ptype, ptype)}</td><td>固定20% · {left.get('trades', 0)}</td><td>{(left.get('win_rate') or 0):.1%}</td><td>{(left.get('profit_factor') or 0):.2f}</td><td>形态目标 · {right.get('trades', 0)}</td><td>{(right.get('win_rate') or 0):.1%}</td><td>{(right.get('profit_factor') or 0):.2f}</td></tr>")
    case_cards = "".join(f"<article class='case'><h3>{i+1}. {html.escape(c['type'])} · {html.escape(c['code'])} · {c['pnl']:,.0f}</h3><p>{html.escape(' '.join(narrative(c)))}</p><button class='toggle' id='toggle{i}' data-full='1' onclick='toggleRange({i})'>聚焦信号至退出</button><div class='chart' id='chart{i}'></div><div class='meta'>信号 {c['signal_date']} · 入场 {c['entry_date']} / {c['entry_price']:.4g} · 离场 {c['exit_date']} / {c['exit_price']:.4g} · 止损 {c['stop_price']:.4g} · 等幅目标 {('待解锁' if c['target_price'] is None else format(c['target_price'], '.4g'))}</div></article>" for i, c in enumerate(cases))
    ag = summary["aggregate"]
    bg = summary["fixed_8_20_baseline_aggregate"]
    win_delta = (ag["win_rate"] - bg["win_rate"]) if ag["win_rate"] is not None and bg["win_rate"] is not None else None
    target = ("未计算" if win_delta is None else f"{win_delta*100:+.2f} pp")
    return_delta = ag["cumulative_return"] - bg["cumulative_return"]
    pf_delta = ag["profit_factor"] - bg["profit_factor"]
    dd_delta = ag.get("max_drawdown_combined_equity", 0) - bg.get("max_drawdown_combined_equity", 0)
    rr_median = ag.get("target_reward_r_median")
    below20 = ag.get("target_return_below_fixed20_fraction")
    if len(cases) != 5:
        raise SystemExit(f"Expected 5 actual losing cases but found {len(cases)}")
    out.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定8%止损 + 形态等幅目标｜100账户</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#18212b;--muted:#627080;--line:#dce3e9;--red:#c74343;--green:#178569;--blue:#476c9b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1280px;margin:auto;padding:32px 22px 72px}}h1{{font-size:30px;line-height:1.2;margin:0 0 10px}}h2{{margin:38px 0 14px}}h3{{margin:0 0 10px;font-size:19px}}.sub,.muted,.meta{{color:var(--muted)}}.sub{{margin-bottom:24px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}}.metric,.card,.case{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700}}.metric small{{color:var(--muted)}}.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}.case{{margin:16px 0;padding:18px}}.chart{{height:540px;width:100%;margin:10px 0}}.meta{{font-size:13px}}.note{{border-left:4px solid var(--blue);padding:10px 14px;background:#eaf0f6;border-radius:4px;margin:14px 0}}.toggle{{border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--blue);padding:7px 12px;cursor:pointer}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid var(--line);text-align:right}}td:nth-child(1),td:nth-child(2),th:nth-child(1),th:nth-child(2){{text-align:left}}.foot{{font-size:13px;color:var(--muted)}}@media(max-width:760px){{main{{padding:20px 12px}}.grid{{grid-template-columns:repeat(2,1fr)}}.twocol{{grid-template-columns:1fr}}.chart{{height:420px}}}}
</style><script>{echart}</script><main><h1>固定 8% 止损 + 形态等幅目标</h1><div class="sub">100 个冻结账户 · 复用保存的 G2 信号 · 截止 {summary['cutoff']} · 新撮合耗时 {summary['elapsed_seconds']:.1f}s</div>
<div class="grid"><div class="metric"><small>完成账户</small><div class="value">{ag['account_count']}/100</div></div><div class="metric"><small>交易数 / 胜率</small><div class="value">{ag['trade_count']} / {ag['win_rate']:.1%}</div></div><div class="metric"><small>等权组合收益</small><div class="value">{ag['cumulative_return']:.1%}</div></div><div class="metric"><small>胜率变化（固定20%对照）</small><div class="value">{target}</div></div></div>
<div class="twocol"><section class="card"><h2>配对对照</h2><table><thead><tr><th>退出规则</th><th>交易数</th><th>胜率</th><th>等权收益</th><th>盈利因子</th></tr></thead><tbody><tr><td>8%止损 / 固定20%目标（保存基线）</td><td>{bg['trade_count']}</td><td>{bg['win_rate']:.1%}</td><td>{bg['cumulative_return']:.1%}</td><td>{bg['profit_factor']:.3f}</td></tr><tr><td>8%止损 / 形态等幅目标（本次）</td><td>{ag['trade_count']}</td><td>{ag['win_rate']:.1%}</td><td>{ag['cumulative_return']:.1%}</td><td>{ag['profit_factor']:.3f}</td></tr></tbody></table><p class="foot">最大组合回撤：固定目标 {bg['max_drawdown_combined_equity']:.1%}，等幅目标 {ag['max_drawdown_combined_equity']:.1%}。账户回报中位数 {bg['account_return_median']:.1%} → {ag['account_return_median']:.1%}；个体账户最大回撤中位数 {bg['account_drawdown_median']:.1%} → {ag['account_drawdown_median']:.1%}。100 个账户均各自从 100,000 起步，逐个使用相同信号。</p></section><section class="card"><h2>撮合与完整性</h2><p>账户核算：{'全部通过' if ag['accounting_all_pass'] else '存在未通过项'}。退出原因：{html.escape(json.dumps(ag['exit_reason_counts'],ensure_ascii=False))}。</p><p>拒绝 / 未成交记录：{ag['rejection_count']}：{html.escape(json.dumps(summary['aggregate'].get('rejection_reason_counts', {}),ensure_ascii=False))}；新策略实际交易 {ag['trade_count']} 笔，亏损 {ag['loss_count']} 笔。</p><p>目标可用 {ag['measured_target_status_counts'].get('active',0)} 笔，牛旗前高待解锁 {ag['measured_target_status_counts'].get('pending_prior_high',0)} 笔；缺失目标拒绝 {summary['aggregate'].get('missing_measured_target_count',0)}，目标不高于入场拒绝 {summary['aggregate'].get('target_rejected_not_above_entry_count',0)}。逐账户冻结数据修订匹配 {summary['frozen_revision_match_count']}/100。</p><p class="foot">识别与选信号没有重跑；每个账户完整保存信号、全部成交、拒绝原因、权益曲线及会计核对。</p></section></div>
<section class="card"><h2>按形态看配对胜率</h2><table><thead><tr><th>主形态</th><th>固定目标交易数</th><th>固定目标胜率</th><th>固定目标PF</th><th>等幅目标交易数</th><th>等幅目标胜率</th><th>等幅目标PF</th></tr></thead><tbody>{''.join(shape_rows)}</tbody></table><p class="foot">分组使用成交记录所带 primary pattern 类型；同一批候选信号可能因持仓占用而形成不同成交数。分组胜率用于定位差异，不作为识别有效性的证明。</p></section>
<section class="card"><h2>针对胜率的下一轮核对顺序</h2><p class="foot">以下是待验证的独立实验，不是已采用的入场过滤。每次只改一项，并报告胜率、收益、PF、最大回撤、成交数，以及被排除形态后来真实有效的比例。</p><table><thead><tr><th>顺序</th><th>检查项</th><th>要验证的问题</th><th>可能损失的有效机会</th></tr></thead><tbody><tr><td>1</td><td>W 底核心跨度 / 确认等待</td><td>双底完成后等待过久，是否增加失败突破率？</td><td>长时间筑底后才突破的有效案例</td></tr><tr><td>2</td><td>颈线候选排序 / 压力触点分布</td><td>所选线与具有更多独立压力测试的备选线，在确认价和后续胜率上有何差异？</td><td>触点数量较少但较早、较长的有效边界</td></tr><tr><td>3</td><td>冻结线漂移 / 突破距离</td><td>有效起点至冻结日线价变化、确认收盘相对颈线的距离，是否解释假突破？</td><td>斜率明显但结构真实的倾斜颈线</td></tr><tr><td>4</td><td>头肩底肩部与头部比例</td><td>左右肩差、头部突出度、所选颈线是否在同一尺度下稳定？</td><td>不对称但有效的头肩底</td></tr><tr><td>5</td><td>三角底的两边界 / 顶点位置</td><td>上沿下斜、下沿支撑和突破位置是否彼此对应？</td><td>较宽、未接近几何顶点前已有效突破的三角底</td></tr><tr><td>6</td><td>牛旗前高解锁时点</td><td>入场后待解锁目标的比例、等待时间与止损/目标先后次序如何？</td><td>目标较远、但前高突破前回撤后仍成功的交易</td></tr></tbody></table></section>
<div class="note"><b>结果读法：</b>本次胜率变化为 {target}；等权收益变化 {return_delta*100:+.2f} pp，PF 变化 {pf_delta:+.3f}，组合最大回撤变化 {dd_delta*100:+.2f} pp。胜率没有提升；这是同信号下退出规则的结果，不证明形态识别改善。等幅目标收益中位数约 {('未提供' if rr_median is None else f'{rr_median:.2f}R')}，目标收益低于固定 20% 的交易占 {('未提供' if below20 is None else f'{below20:.1%}')}；大多数形态目标比固定 20% 更远，胜率更难达到。几何案例是待验证线索；亏损不能单独证明识别或颈线匹配错误。<br>此前研究中的 +405% 数字不能作为有效基线：旧实现缺少下飘旗形目标重建分支，且通用目标计算存在入场价格索引回退；这会改变目标缺失时的持仓行为。</div>
<h2>五个真实亏损案例：回到结构和颈线核对</h2><p class="muted">每张图默认放大到信号至退出，可切换完整结构；显示原始结构点、全局日期索引回算的边界、入场、退出、止损和目标。线条仅在其有效区间显示：有效起点至冻结日为实线，冻结后延伸为虚线。</p>{case_cards}
<h2>规则来源与本次实现</h2><p>方法依据书中“W底涨幅满足计算”“头肩底涨幅满足计算”“收敛三角形底部涨幅满足”等幅测量段落：以颈线或突破位和形态高度构造价格目标。本次目标从入场后依形态冻结；固定 8% 止损仍以实际成交价计算。{html.escape('；'.join(book_refs))}</p><p>对牛旗，冻结旗形边界和冲击段构成投射；若需先突破前高才可测量，目标保持待解锁，不能借未来形态点回填。等幅目标缺失或无效的信号应在成交前按明确原因记录。</p><h2>限制</h2><p class="foot">保存的信号来自 G2 perf2-extrema10w baseline A；本次未重跑识别，因此只评估同信号条件下退出规则改变。历史 OHLC 同一根 K 线同时触及止损与目标时按 broker/策略中明确的撮合先后结算。样本跨不同年代和公司行动，结果仅适用于冻结快照及该回测实现。五例刻意展示不同几何检查，不是随机样本或代表性估计。</p></main>
<script>const R={_safe(payload)};const CH=[];const FOCUS=[];function chartFor(c,i){{const el=document.getElementById('chart'+i),chart=echarts.init(el);CH[i]=chart;const b=c.bars,dates=b.map(x=>x.date),candles=b.map(x=>[x.open,x.close,x.low,x.high]),vol=b.map(x=>x.volume);const idx=Object.fromEntries(dates.map((d,j)=>[d,j]));const focusStart=idx[c.signal_date]===undefined?Math.floor(dates.length*.7):idx[c.signal_date];FOCUS[i]=focusStart;const pct=n=>dates.length>1?100*n/(dates.length-1):0;const points=(c.diagnostic.points||[]).map(p=>idx[p.date]===undefined?null:[p.date,p.price,p.kind]).filter(Boolean);const pressure=(c.diagnostic.tests||[]).map(p=>idx[p.pivot_date]===undefined?null:[p.pivot_date,p.level,p.group_id]).filter(Boolean);const neck=c.diagnostic.neckline||{{}},eff=c.diagnostic.effective_start_date||dates[0],frz=c.diagnostic.freeze_date||c.signal_date;const lineVal=x=>neck.slope*x.index+neck.intercept;const solid=b.map(x=>x.date>=eff&&x.date<=frz?lineVal(x):null),dashed=b.map(x=>x.date>frz?lineVal(x):null);const lines=[];if(neck.slope!==undefined&&neck.intercept!==undefined){{lines.push({{name:'冻结期有效颈线',type:'line',data:solid,symbol:'none',connectNulls:false,lineStyle:{{color:'#7866a8',width:2}},z:5}},{{name:'冻结后延伸',type:'line',data:dashed,symbol:'none',connectNulls:false,lineStyle:{{color:'#7866a8',width:2,type:'dashed'}},z:5}})}}for(const [name,val,color] of [['入场',c.entry_price,'#2776c5'],['退出',c.exit_price,'#333'],['止损',c.stop_price,'#c74343'],['等幅目标',c.target_price,'#178569']])if(val!==null&&val!==undefined)lines.push({{name,type:'line',data:dates.map(()=>val),symbol:'none',lineStyle:{{color,width:1,type:'dotted'}}}});chart.setOption({{animation:false,tooltip:{{trigger:'axis',axisPointer:{{type:'cross'}}}},legend:{{top:4}},axisPointer:{{link:[{{xAxisIndex:'all'}}]}},grid:[{{left:62,right:34,top:45,height:'65%'}},{{left:62,right:34,top:'78%',height:'12%'}}],xAxis:[{{type:'category',data:dates,scale:true,boundaryGap:false,axisLine:{{onZero:false}},splitLine:{{show:false}},min:'dataMin',max:'dataMax'}},{{type:'category',gridIndex:1,data:dates,scale:true,boundaryGap:false,axisLabel:{{show:false}},axisLine:{{onZero:false}},splitLine:{{show:false}},min:'dataMin',max:'dataMax'}}],yAxis:[{{scale:true,splitArea:{{show:true}}}},{{scale:true,gridIndex:1,splitNumber:2,axisLabel:{{show:false}},splitLine:{{show:false}}}}],dataZoom:[{{type:'inside',xAxisIndex:[0,1],start:0,end:100}},{{show:true,xAxisIndex:[0,1],type:'slider',top:'93%',start:0,end:100}}],series:[{{name:'K线',type:'candlestick',data:candles,markLine:{{silent:true,symbol:['none','none'],data:[{{xAxis:c.diagnostic.freeze_date,name:'冻结日',lineStyle:{{color:'#7866a8',type:'solid'}},label:{{formatter:'冻结日'}}}},{{xAxis:c.signal_date,name:'信号日',lineStyle:{{color:'#2776c5',type:'dashed'}},label:{{formatter:'信号'}}}},{{xAxis:c.entry_date,name:'入场日',lineStyle:{{color:'#e29b38',type:'dashed'}},label:{{formatter:'入场'}}}},{{xAxis:c.exit_date,name:'退出日',lineStyle:{{color:'#333',type:'dashed'}},label:{{formatter:'退出'}}}}]}},itemStyle:{{color:'#c74343',color0:'#178569',borderColor:'#c74343',borderColor0:'#178569'}}}},...lines,{{name:'结构锚点',type:'scatter',data:points.map(p=>[p[0],p[1]]),symbolSize:9,itemStyle:{{color:'#f0a43b'}},tooltip:{{formatter:p=>'结构锚点<br>'+p.value[0]+'<br>'+Number(p.value[1]).toFixed(4)}}}},{{name:'颈线压力触点',type:'scatter',data:pressure.map(p=>[p[0],p[1]]),symbol:'diamond',symbolSize:10,itemStyle:{{color:'#8f5fa6'}},tooltip:{{formatter:p=>'压力触点<br>'+p.value[0]+'<br>'+Number(p.value[1]).toFixed(4)}}}},{{name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,data:vol,itemStyle:{{color:'#8ba0b2'}}}}]}})}}function toggleRange(i){{const c=R.cases[i],chart=CH[i],n=c.bars.length,focus=FOCUS[i],button=document.getElementById('toggle'+i),full=button.dataset.full==='1';const start=full?100*focus/Math.max(1,n-1):0;chart.dispatchAction({{type:'dataZoom',dataZoomIndex:0,start,end:100}});chart.dispatchAction({{type:'dataZoom',dataZoomIndex:1,start,end:100}});button.dataset.full=full?'0':'1';button.textContent=full?'查看完整结构':'聚焦信号至退出'}}R.cases.forEach(chartFor);window.addEventListener('resize',()=>CH.forEach(c=>c.resize()));</script></html>''', encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
