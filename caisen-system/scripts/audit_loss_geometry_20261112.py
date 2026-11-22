#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a read-only chart and geometry audit for five early-stop accounts."""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/daily-research/structure-boundary-study-20261112"
ACCOUNT_DIR = STUDY / "full-B-current/accounts"
DATA_ROOT = Path("/tmp/caisen-structure-boundary-snapshot-20261112-v2/data")
CODES = ["000573", "000955", "000779", "600684", "603009"]
CUTOFF = "2026-10-30"


def read_rows(code: str, kind: str) -> list[dict]:
    path = DATA_ROOT / ("etfs" if kind == "etf" else "stocks") / f"{code}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [r for r in csv.DictReader(stream) if r["date"] <= CUTOFF]


def evidence(pattern: dict, code: str):
    for item in pattern.get("evidence", []) or []:
        if item.get("code") == code:
            return item.get("value")
    return None


def trade_pattern(account: dict, trade: dict):
    signal = next((s for s in account.get("signals", [])
                   if s.get("signal_date") == trade.get("signal_date")), None)
    if not signal:
        return None, None
    pid = next(iter(trade.get("pattern_ids", []) or []), None)
    pattern = next((p for p in signal.get("patterns", []) or []
                    if p.get("pattern_id") == pid), signal.get("pattern"))
    tag = next((t for t in trade.get("pattern_tags", []) or []
                if t.get("pattern_id") == pid), {})
    return pattern, tag


def nearest_date_index(rows: list[dict], day: str | None):
    if not day:
        return None
    return next((i for i, r in enumerate(rows) if r["date"] == day), None)


def points_for(pattern: dict, rows: list[dict]) -> list[dict]:
    points, kp = [], pattern.get("key_points", {}) or {}
    for n in range(1, 10):
        idx, day = kp.get(f"pivot_{n}_index"), kp.get(f"pivot_{n}_date")
        val = kp.get(f"pivot_{n}")
        if not isinstance(idx, int) or idx < 0 or idx >= len(rows):
            continue
        row = rows[idx]
        if not day:
            day = row["date"]
        kind = kp.get(f"pivot_{n}_kind")
        if kind not in ("high", "low"):
            # Flag anchors alternate from the first low in this schema.
            kind = "low" if n % 2 else "high"
        if val is None:
            val = float(row["low"] if kind == "low" else row["high"])
        points.append({"index": idx, "date": day, "price": float(val),
                       "kind": kind, "label": f"结构点{n}·{'低' if kind == 'low' else '高'}"})
    return points


def geom_audit(trade: dict, pattern: dict, tag: dict, rows: list[dict]):
    selection = evidence(pattern, "neckline_selection") or tag.get("neckline_selection") or {}
    structure = evidence(pattern, "main_structure") or tag.get("main_structure") or {}
    boundary_events = evidence(pattern, "joint_lifecycle") or {}
    line = (selection.get("selected_line") or pattern.get("neckline") or
            structure.get("neckline") or {})
    points = points_for(pattern, rows)
    line = {"slope": float(line.get("slope", 0)),
            "intercept": float(line.get("intercept", 0))} if line else None
    freeze_date = (selection.get("freeze_date") or structure.get("frozen_at") or
                   boundary_events.get("paired_at"))
    freeze_idx = nearest_date_index(rows, freeze_date)
    sig_idx = nearest_date_index(rows, trade.get("signal_date"))
    ent_idx = nearest_date_index(rows, trade.get("entry_date"))
    exit_idx = nearest_date_index(rows, trade.get("exit_date"))
    effective_start = selection.get("effective_start_date") or structure.get("neckline_effective_start")
    effective_idx = nearest_date_index(rows, effective_start)
    if effective_idx is None and points:
        effective_idx = min(p["index"] for p in points)
    formation_idx = min((p["index"] for p in points), default=effective_idx or 0)

    def y(index):
        return line["slope"] * index + line["intercept"] if line else None

    bottom_family = pattern.get("pattern_type") in {"inverse_head_shoulders", "w_bottom", "double_bottom"}

    def coverage(start, end):
        if not bottom_family or not line or start is None or end is None or start > end:
            return None
        candidates = [(float(rows[i]["high"]) - y(i), i) for i in range(start, end + 1)]
        excess, idx = max(candidates, default=(None, None))
        return {"max_high_above_frozen_line": round(excess, 6),
                "date": rows[idx]["date"], "high": float(rows[idx]["high"]),
                "line_price": round(y(idx), 6), "start_date": rows[start]["date"],
                "end_date": rows[end]["date"]}

    cov_to_freeze = coverage(effective_idx, freeze_idx)
    cov_whole = coverage(formation_idx, freeze_idx)
    close_crosses = []
    if line and effective_idx is not None and freeze_idx is not None:
        close_crosses = [{"date": rows[i]["date"], "close": float(rows[i]["close"]),
                          "line": round(y(i), 6)}
                         for i in range(effective_idx, freeze_idx + 1)
                         if float(rows[i]["close"]) > y(i) + 0.01]

    # Inter-leg rebound evidence: maxima are measured from the same OHLC snapshot;
    # proximity to any selected independent touch is reported, not treated as proof.
    lows = [p for p in points if p["kind"] == "low"]
    tests = selection.get("independent_tests", []) or []
    rebounds = []
    for left, right in zip(lows, lows[1:]):
        if right["index"] <= left["index"] + 1:
            continue
        j = max(range(left["index"] + 1, right["index"]),
                key=lambda k: float(rows[k]["high"]))
        nearest = min(((abs(int(t.get("pivot_index", j)) - j), t)
                       for t in tests if isinstance(t.get("pivot_index"), int)),
                      default=(None, None))
        rebounds.append({"between": [left["date"], right["date"]],
                         "date": rows[j]["date"], "index": j,
                         "high": float(rows[j]["high"]),
                         "frozen_line": round(y(j), 6) if line else None,
                         "independent_touch_nearest_bars": nearest[0],
                         "nearest_touch_date": nearest[1].get("pivot_date") if nearest[1] else None})
    density = {"low_pivot_count": len(lows),
               "min_low_pivot_gap_sessions": min((b["index"] - a["index"]
                   for a, b in zip(lows, lows[1:])), default=None),
               "low_pivots": lows}
    low_values = [p["price"] for p in lows]
    anchor_low = (min(low_values) if pattern.get("pattern_type") == "inverse_head_shoulders"
                  else sum(low_values) / len(low_values) if low_values
                  else structure.get("support_lower"))
    target = None
    if (pattern.get("pattern_type") in {"inverse_head_shoulders", "w_bottom", "double_bottom"}
            and line and anchor_low is not None and sig_idx is not None):
        breakout = evidence(pattern, "neckline_breakout") or {}
        line_at_breakout = float(breakout.get("frozen_neckline", y(sig_idx)))
        breakout_close = float(breakout.get("close", rows[sig_idx]["close"]))
        height = max(0.0, line_at_breakout - float(anchor_low))
        engine_target = breakout_close + height
        double_line_target = line_at_breakout + height
        entry = float(trade.get("entry_price", 0) or 0)
        target = {"engine_method": "breakout close + (frozen neckline at breakout - lowest structure low)",
                  "anchor_low": round(float(anchor_low), 6),
                  "breakout_close": round(breakout_close, 6),
                  "line_at_breakout": round(line_at_breakout, 6),
                  "engine_projected_price": round(engine_target, 6),
                  "engine_upside_from_entry_pct": round((engine_target / entry - 1) * 100, 3)
                  if entry else None,
                  "engine_supports_strategy_20pct": engine_target >= entry * 1.2 if entry else None,
                  "alternative_double_line_projected_price": round(double_line_target, 6),
                  "is_stored_pattern_target": False,
                  "scope": "bottom-pattern geometry only; not applicable to bull_flag"}
    elif pattern.get("pattern_type") == "bull_flag":
        target = {"scope": "bull_flag uses impulse/correction geometry; bottom-pattern projection deliberately not applied.",
                  "flag_geometry": evidence(pattern, "flag_geometry")}
    max_tol = max((float(t.get("tolerance", 0) or 0) for t in tests), default=None)
    support_price = min((p["price"] for p in lows), default=structure.get("support_lower"))
    tolerance = {"max_touch_tolerance": max_tol,
                 "reference_low_price": support_price,
                 "max_tolerance_pct_of_low": round(max_tol / support_price * 100, 3)
                 if max_tol is not None and support_price else None,
                 "tests": [{"date": t.get("pivot_date"), "level": t.get("level"),
                            "tolerance": t.get("tolerance"),
                            "tolerance_pct_of_touch": round(float(t.get("tolerance", 0)) /
                                float(t.get("level", 1)) * 100, 3)
                                if float(t.get("level", 0) or 0) else None}
                           for t in tests]}
    return {
        "pattern_type": pattern.get("pattern_type"), "pattern_id": pattern.get("pattern_id"),
        "key_points": points, "line": line, "effective_start_date": effective_start,
        "freeze_date": freeze_date, "line_frozen_at": selection.get("line_frozen_at") or boundary_events.get("paired_at"),
        "confirmation_date": (evidence(pattern, "neckline_breakout") or
                              evidence(pattern, "flag_breakout") or {}).get("date") or
                             trade.get("signal_date"),
        "candidate_id": selection.get("selected_candidate_id"),
        "independent_tests": tests, "recorded_coverage": selection.get("formation_coverage"),
        "coverage_effective_to_freeze": cov_to_freeze,
        "coverage_whole_structure_to_freeze_alternative": cov_whole,
        "pre_freeze_closes_above_line_plus_1c": close_crosses,
        "inter_leg_rebounds": rebounds, "pivot_density": density,
        "target_geometry_experiment": target, "touch_tolerance": tolerance,
        "stored_pattern_targets": pattern.get("targets", []),
        "lifecycle_evidence": boundary_events,
        "indexes": {"formation": formation_idx, "effective": effective_idx,
                    "freeze": freeze_idx, "signal": sig_idx, "entry": ent_idx, "exit": exit_idx},
    }


def analyze_one(code: str, trade: dict, role: str, account: dict, rows: list[dict]):
    pattern, tag = trade_pattern(account, trade)
    if not pattern:
        return {"role": role, "trade": trade, "data_error": "pattern not embedded in selected signal"}
    geometry = geom_audit(trade, pattern, tag, rows)
    ix = geometry["indexes"]
    left = max(0, min(v for v in ix.values() if v is not None) - 12)
    right = min(len(rows) - 1, (ix["exit"] or ix["signal"] or left) + 15)
    chart_rows = [{"index": i, "date": rows[i]["date"],
                   **{k: float(rows[i][k]) for k in ("open", "high", "low", "close")}}
                  for i in range(left, right + 1)]
    events = [{"date": trade.get("signal_date"), "index": ix["signal"], "kind": "signal", "label": "信号"},
              {"date": trade.get("entry_date"), "index": ix["entry"], "kind": "entry", "label": "入场"},
              {"date": trade.get("exit_date"), "index": ix["exit"], "kind": "exit", "label": "退出"}]
    confirm_date = geometry.get("confirmation_date")
    events.append({"date": confirm_date, "index": nearest_date_index(rows, confirm_date),
                   "kind": "confirm", "label": "确认"})
    if geometry.get("freeze_date"):
        events.append({"date": geometry["freeze_date"], "index": ix["freeze"], "kind": "freeze", "label": "冻结"})
    for point in geometry["key_points"]:
        events.append({"date": point["date"], "index": point["index"], "kind": "pivot",
                       "label": point["label"], "price": point["price"]})
    for test in geometry["independent_tests"]:
        idx = test.get("pivot_index")
        events.append({"date": test.get("pivot_date"), "index": idx,
                       "kind": "touch", "label": "独立触点", "price": test.get("level"),
                       "tolerance": test.get("tolerance")})
    return {"role": role,
            "trade": {k: trade.get(k) for k in ("signal_date", "entry_date", "entry_price", "exit_date",
                "exit_price", "exit_reason", "return_pct", "pnl", "holding_sessions", "stop_price", "target_price")},
            "geometry": geometry, "chart": {"rows": chart_rows, "events": events}}


def select_cases():
    cases = []
    for code in CODES:
        account_path = ACCOUNT_DIR / f"stock-{code}.json"
        account = json.loads(account_path.read_text(encoding="utf-8"))
        rows = read_rows(code, account.get("kind", "stock"))
        losses = sorted((t for t in account.get("trades", [])
                         if t.get("exit_reason") == "stop_loss"),
                        key=lambda t: (int(t.get("holding_sessions", 999)), str(t.get("signal_date", ""))))
        early = [t for t in losses if int(t.get("holding_sessions", 999)) <= 5]
        loss = min(early, key=lambda t: str(t.get("signal_date", ""))) if early else losses[0] if losses else None
        wins = [t for t in account.get("trades", []) if float(t.get("pnl", 0) or 0) > 0]
        win = min(wins, key=lambda t: abs((date.fromisoformat(t["signal_date"]) -
                                           date.fromisoformat(loss["signal_date"])).days)) if wins and loss else None
        cases.append({"code": code, "kind": account.get("kind"), "account_status": account.get("status"),
                      "loss": analyze_one(code, loss, "loss", account, rows) if loss else None,
                      "win": analyze_one(code, win, "win", account, rows) if win else None,
                      "winner_available": win is not None,
                      "selection_rule": "earliest signal_date among stop_loss trades holding <=5 sessions"})
    return cases


def render_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>亏损交易结构与颈线几何审计</title>
<style>
:root{color-scheme:light;--ink:#17232b;--muted:#61717c;--line:#d8e0e4;--paper:#f4f6f6;--card:#fff;--teal:#087e78;--red:#b44436;--gold:#b57b12}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.52 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:28px max(22px,calc((100vw - 1180px)/2));background:#163c45;color:#fff}header h1{margin:4px 0;font-size:26px}header p{max-width:900px;color:#dce8e7;margin:8px 0}.wrap{max-width:1180px;margin:22px auto;padding:0 18px}.case{background:var(--card);border:1px solid var(--line);margin:18px 0;padding:20px}.case h2{margin:0 0 8px;font-size:20px}.meta{color:var(--muted);font-size:14px}.pill{display:inline-block;padding:2px 8px;border-radius:3px;background:#e8f2f0;color:#12685f;margin-right:5px}.loss{color:var(--red);font-weight:650}.win{color:var(--teal);font-weight:650}.plots{display:grid;grid-template-columns:1fr;gap:16px}.chartbox{border:1px solid var(--line);background:#fcfdfd;padding:12px;margin-top:14px}.charttitle{font-weight:650;margin:0 0 8px}.chart{width:100%;height:auto;min-height:280px;display:block}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.controls button,.events button{background:white;border:1px solid #aebbc0;border-radius:3px;padding:5px 9px;color:var(--ink);cursor:pointer}.controls button:hover,.events button:hover{border-color:var(--teal);color:var(--teal)}.events{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.events button{font-size:12px;padding:3px 7px}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:5px 0}.legend i{display:inline-block;width:13px;height:3px;vertical-align:middle;margin-right:4px;background:currentColor}.fold{border-top:1px solid var(--line);margin-top:14px;padding-top:8px}.fold summary{cursor:pointer;font-weight:650}.diagnostics{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:10px;margin-top:12px}.diag{background:#f5f8f8;padding:10px 12px;border-left:3px solid #78949a}.diag strong{display:block;margin-bottom:3px}.diag p{margin:3px 0;color:#3f4f57;font-size:13px}.caveat{color:var(--muted);font-size:13px}.error{background:#fff1ee;padding:10px;color:#902c21}footer{margin:28px 0;color:var(--muted);font-size:13px}@media(max-width:700px){header{padding:22px 16px}.wrap{padding:0 10px}.case{padding:14px}.chart{min-height:240px}}
</style>
<body><header><div class="wrap"><span>READ-ONLY CASE AUDIT · 截止 2026-10-30</span><h1>亏损交易的结构与颈线几何</h1><p>每账户取最早一笔持有不超过5个交易日的止损交易，并在有正收益交易时选日期最近的一笔作对照。图只画该笔交易自身信号所带的结构点、冻结线、独立触点、确认与交易日期。盈利或亏损本身不作为形态正确/错误的判断依据。</p></div></header><main class="wrap" id="cases"></main><footer class="wrap">数据：full-B-current/accounts 与冻结快照 OHLC CSV。覆盖复核均按原始CSV行号重算线值；图表横轴为交易日，点位日期使用嵌入证据或其对应行。若账户记录缺少结构证据，会明确标出。</footer>
<script>
const payload=DATA;
const esc=s=>String(s??"—").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function money(v){return v==null?'—':Number(v).toFixed(3)}
function summary(item){if(!item)return '<div class="error">没有符合条件的对照交易。</div>';if(item.data_error)return '<div class="error">缺失模式证据：'+esc(item.data_error)+'</div>';const t=item.trade,g=item.geometry;return `<div class="meta">${esc(t.signal_date)} 信号 → ${esc(t.entry_date)} 入场 ${money(t.entry_price)} → ${esc(t.exit_date)} ${esc(t.exit_reason)} ${money(t.exit_price)} · ${t.holding_sessions}持仓日 · <b class="${item.role}">${(Number(t.return_pct)*100).toFixed(2)}%</b> · PnL ${money(t.pnl)}</div>`}
function diag(item){if(!item||item.data_error)return '';let g=item.geometry,c=g.coverage_effective_to_freeze,w=g.coverage_whole_structure_to_freeze_alternative,t=g.target_geometry_experiment,r=g.inter_leg_rebounds||[],den=g.pivot_density||{},tol=g.touch_tolerance||{};let lifecycle=g.lifecycle_evidence||{};let targetText=t?.engine_projected_price!=null?`源公式估算 ${money(t.engine_projected_price)}；相对入场空间 ${t.engine_upside_from_entry_pct}%；${t.engine_supports_strategy_20pct?'达到':'未达到'}20%。替代2×颈线−低点=${money(t.alternative_double_line_projected_price)}。`:esc(t?.scope||'无足够冻结边界估算。');return `<div class="diagnostics"><div class="diag"><strong>记录内形成覆盖 · 有效线段至冻结</strong><p>${c?`${esc(c.start_date)}–${esc(c.end_date)}；最高价相对线最大超出 ${money(c.max_high_above_frozen_line)}，发生于 ${esc(c.date)}（高 ${money(c.high)}，线 ${money(c.line_price)}）。`:"无可用冻结边界。"}</p><p>原记录：${esc(JSON.stringify(g.recorded_coverage||"无"))}</p></div><div class="diag"><strong>替代口径 · 整个结构区间至冻结</strong><p>${w?`${esc(w.start_date)}–${esc(w.end_date)}；最大超出 ${money(w.max_high_above_frozen_line)} 于 ${esc(w.date)}（高 ${money(w.high)}，线 ${money(w.line_price)}）。`:"无可用数据。"}</p><p>若区间早于有效起点，此项只是整段结构一致性实验，不等于引擎当前拒绝条件。</p></div><div class="diag"><strong>内部反弹与触点关联</strong><p>${r.length?r.map(x=>`${esc(x.between.join('→'))}内部最高 ${esc(x.date)} / ${money(x.high)}；离最近独立触点${x.independent_touch_nearest_bars==null?'未知':x.independent_touch_nearest_bars+'根'}`).join('<br>'):"可用关键点中未形成相邻底点区间。"}</p><p>底点数 ${den.low_pivot_count??'—'}；相邻底点最短间隔 ${den.min_low_pivot_gap_sessions??'—'} 个交易日。关联距离是诊断线索，不直接判错。</p></div><div class="diag"><strong>冻结前是否已越线</strong><p>有效起点至冻结日，收盘高于冻结线1分钱的天数：${(g.pre_freeze_closes_above_line_plus_1c||[]).length}${g.pre_freeze_closes_above_line_plus_1c?.length?`；${g.pre_freeze_closes_above_line_plus_1c.slice(0,4).map(x=>esc(x.date)+' 收'+money(x.close)+'/线'+money(x.line)).join('；')}`:''}。</p><p>需区分独立触点容差、突破时段和冻结时点。</p></div><div class="diag"><strong>几何目标 vs 策略20%</strong><p>${targetText}</p><p>底部投影用源代码公式复算；结果仍是理论几何值，不是账户保存目标。模式原始目标：${esc(JSON.stringify(g.stored_pattern_targets))}；策略交易目标：${money(item.trade.target_price)}。</p></div><div class="diag"><strong>容差相对低价</strong><p>最大独立触点容差 ${money(tol.max_touch_tolerance)} / 结构最低点 ${money(tol.reference_low_price)} = ${tol.max_tolerance_pct_of_low??'—'}%。各触点：${(tol.tests||[]).map(x=>`${esc(x.date)} ${x.tolerance_pct_of_touch}%`).join('；')||'无独立触点'}。</p></div><div class="diag"><strong>冻结生命周期字段</strong><p>${lifecycle.instance_id?`状态 ${esc(lifecycle.state_at)}；entry_eligible=${esc(lifecycle.entry_eligible)}；paired_at=${esc(lifecycle.paired_at)}。`:"嵌入pattern未提供joint_lifecycle证据。"}</p><p>仅将字段并列展示，不单凭字段推断执行链实际消费情况。</p></div></div>`}
function chartHtml(item,id){if(!item||item.data_error)return '';let chart=item.chart;return `<div class="chartbox"><p class="charttitle">${item.role==='loss'?'亏损例':'盈利对照'} · ${esc(item.geometry.pattern_type)}</p><div class="legend"><span style="color:#344c56"><i></i>冻结线</span><span style="color:#17887e">●结构点</span><span style="color:#c28318">●独立触点</span><span style="color:#b44436">│确认/交易事件</span></div><div class="controls"><button data-chart="${id}" data-action="fit">全览</button><button data-chart="${id}" data-action="in">放大</button><button data-chart="${id}" data-action="out">缩小</button></div><svg class="chart" id="${id}" role="img" aria-label="${esc(item.role)}交易K线与形态结构"></svg><div class="events" id="${id}-events"></div></div>`}
const root=document.getElementById('cases');root.innerHTML=payload.cases.map((c,i)=>`<article class="case"><h2>案例 ${String(i+1).padStart(2,'0')} · ${esc(c.code)} <span class="pill">${esc(c.kind)}</span></h2><div class="meta">${esc(c.selection_rule)} · 对照：${c.winner_available?'有正收益交易':'无正收益交易'}</div><details class="fold"><summary>交易结果（点击展开）</summary>${summary(c.loss)}${c.win?summary(c.win):'<div class="error">本账户没有正收益对照交易。</div>'}</details><details class="fold" open><summary>图形证据与几何诊断</summary><p class="caveat">交易结果只定义样本筛选；几何量来自信号嵌入证据及冻结快照OHLC重算。覆盖复核截止冻结日；图上线在冻结后延长仅供观察。</p>${chartHtml(c.loss,`loss-${i}`)}${c.win?chartHtml(c.win,`win-${i}`):''}${diag(c.loss)}${c.win?`<h3>盈利对照诊断</h3>${diag(c.win)}`:''}</details></article>`).join('');
const chartData={};payload.cases.forEach((c,i)=>{if(c.loss&&!c.loss.data_error)chartData[`loss-${i}`]=c.loss;if(c.win&&!c.win.data_error)chartData[`win-${i}`]=c.win});
function draw(id,view){const item=chartData[id],svg=document.getElementById(id);if(!item||!svg)return;const rows=item.chart.rows,events=item.chart.events,g=item.geometry,W=Math.max(600,svg.clientWidth||900),H=320,L=54,R=18,T=14,B=30,iw=W-L-R,ih=H-T-B;const baseLo=rows[0].index,baseHi=rows[rows.length-1].index;let lo=baseLo,hi=baseHi;if(view&&view.span){let center=view.center??(baseLo+baseHi)/2;lo=Math.max(baseLo,Math.floor(center-view.span/2));hi=Math.min(baseHi,Math.ceil(center+view.span/2));}let vis=rows.filter(x=>x.index>=lo&&x.index<=hi);if(!vis.length)return;const ymin=Math.min(...vis.map(x=>x.low)),ymax=Math.max(...vis.map(x=>x.high)),pad=(ymax-ymin)*.08||.1,min=ymin-pad,max=ymax+pad;const x=i=>L+(i-lo)/(hi-lo||1)*iw,y=v=>T+(max-v)/(max-min)*ih;let out=[`<rect x="0" y="0" width="${W}" height="${H}" fill="#fcfdfd"/>`];for(let k=0;k<=4;k++){let v=min+(max-min)*k/4,yy=y(v);out.push(`<path d="M${L} ${yy}H${W-R}" stroke="#e6ecee"/><text x="${L-7}" y="${yy+4}" text-anchor="end" font-size="11" fill="#61717c">${v.toFixed(2)}</text>`)}vis.forEach((r,j)=>{let xx=x(r.index),green=r.close>=r.open,cl=green?'#148278':'#b44b40',bw=Math.max(2,Math.min(8,iw/vis.length*.58));out.push(`<path d="M${xx} ${y(r.high)}V${y(r.low)}" stroke="${cl}"/><rect x="${xx-bw/2}" y="${Math.min(y(r.open),y(r.close))}" width="${bw}" height="${Math.max(1,Math.abs(y(r.open)-y(r.close)))}" fill="${cl}"/>`);if(j%Math.max(1,Math.floor(vis.length/8))===0)out.push(`<text x="${xx}" y="${H-8}" text-anchor="middle" font-size="10" fill="#61717c">${r.date}</text>`)});if(g.line){let a=Math.max(vis[0].index,g.indexes.effective??vis[0].index),b=vis[vis.length-1].index;if(a<=b)out.push(`<path d="M${x(a)} ${y(g.line.slope*a+g.line.intercept)}L${x(b)} ${y(g.line.slope*b+g.line.intercept)}" stroke="#344c56" stroke-width="2" stroke-dasharray="6 3"/>`)}const colors={signal:'#9b3a31',confirm:'#a67613',entry:'#b44436',exit:'#771d18',freeze:'#6559a3',pivot:'#17887e',touch:'#c28318'};events.filter(e=>e.index!=null&&e.index>=lo&&e.index<=hi).forEach(e=>{let xx=x(e.index),color=colors[e.kind]||'#555',price=e.price??(e.kind==='exit'?item.trade.exit_price:e.kind==='entry'?item.trade.entry_price:null);out.push(`<path d="M${xx} ${T}V${H-B}" stroke="${color}" stroke-width="1" opacity=".52"/>`);if(price!=null)out.push(`<circle cx="${xx}" cy="${y(price)}" r="4" fill="${color}" stroke="white" stroke-width="1"><title>${esc(e.label)} ${esc(e.date)} ${money(price)}</title></circle>`)});svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out.join('');let ev=document.getElementById(id+'-events');if(ev)ev.innerHTML=events.filter(e=>e.index!=null).map((e,j)=>`<button data-event-chart="${id}" data-event-index="${e.index}">${esc(e.label)} ${esc(e.date)}</button>`).join('');}
const views={};Object.keys(chartData).forEach(id=>{let r=chartData[id].chart.rows;views[id]={span:r[r.length-1].index-r[0].index+1};draw(id,views[id])});document.addEventListener('click',e=>{let b=e.target.closest('button');if(!b)return;let id=b.dataset.chart||b.dataset.eventChart;if(!id)return;let rows=chartData[id].chart.rows,n=rows[rows.length-1].index-rows[0].index+1,v=views[id];if(b.dataset.action==='fit'){v.span=n;v.center=(rows[0].index+rows[rows.length-1].index)/2}if(b.dataset.action==='in')v.span=Math.max(20,(v.span||n)*.65);if(b.dataset.action==='out')v.span=Math.min(n,(v.span||n)/.65);if(b.dataset.eventIndex){v.center=Number(b.dataset.eventIndex);v.span=Math.min(n,Math.max(30,(v.span||n)*.55));}draw(id,v)});window.addEventListener('resize',()=>Object.keys(chartData).forEach(id=>draw(id,views[id])));
</script></body></html>'''.replace("DATA", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=STUDY)
    args = ap.parse_args()
    cases = select_cases()
    payload = {"title": "亏损交易的结构与颈线几何审计", "cutoff": CUTOFF,
               "source": {"accounts": str(ACCOUNT_DIR), "ohlcv": str(DATA_ROOT)}, "cases": cases}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "loss-geometry-audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "loss-geometry-audit.html").write_text(render_html(payload), encoding="utf-8")
    print(f"cases={len(cases)} losses={sum(c['loss'] is not None for c in cases)} "
          f"winners={sum(c['winner_available'] for c in cases)}")
    for c in cases:
        loss = c.get("loss")
        print(c["code"], loss and loss.get("trade", {}).get("signal_date"),
              loss and loss.get("geometry", {}).get("pattern_type"),
              "winner=" + str(c["winner_available"]))
    print(args.out_dir / "loss-geometry-audit.html")
    print(args.out_dir / "loss-geometry-audit.json")


if __name__ == "__main__":
    main()
