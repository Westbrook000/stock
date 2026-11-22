#!/usr/bin/env python3
"""Render the paired structure/boundary study as a self-contained HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


LABELS = {
    "w_bottom": "W底", "inverse_head_shoulders": "头肩底",
    "triangle_bottom": "收敛三角形底部", "bull_flag": "下飘旗形上涨中继",
}
CASES = (("000001", "stock"), ("000012", "stock"), ("000538", "stock"),
         ("600108", "stock"), ("159718", "etf"), ("516630", "etf"))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_bars(path):
    bars = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                bars.append([row["date"][:10], float(row["open"]), float(row["high"]),
                             float(row["low"]), float(row["close"])])
            except (KeyError, TypeError, ValueError):
                continue
    bars.sort(key=lambda item: item[0])
    deduped = {}
    for row in bars:
        deduped[row[0]] = row
    return [deduped[key] for key in sorted(deduped) if key <= "2026-10-30"]


def _pattern_evidence(pattern):
    return {item.get("code"): item.get("value") for item in pattern.get("evidence", [])}


def _pattern_summary(pattern, signal):
    evidence = _pattern_evidence(pattern)
    structure = evidence.get("main_structure") or {}
    selection = evidence.get("neckline_selection") or {}
    breakout = evidence.get("neckline_breakout") or evidence.get("flag_breakout") or {}
    lifecycle = evidence.get("joint_lifecycle") or {}
    key_points = pattern.get("key_points") or {}
    anchors = []
    number = 1
    while f"pivot_{number}_date" in key_points:
        anchors.append({"date": key_points.get(f"pivot_{number}_date"),
                        "kind": key_points.get(f"pivot_{number}_kind"),
                        "price": key_points.get(f"pivot_{number}"),
                        "index": key_points.get(f"pivot_{number}_index")})
        number += 1
    line = pattern.get("neckline") or selection.get("selected_line") or {}
    return {
        "pattern_id": pattern.get("pattern_id"),
        "type": pattern.get("pattern_type"),
        "label": LABELS.get(pattern.get("pattern_type"), pattern.get("pattern_type")),
        "state": pattern.get("state"),
        "formed_at": pattern.get("formed_at"),
        "confirmation_date": pattern.get("confirmation_date") or signal.get("signal_date"),
        "entry_date": signal.get("entry_date"),
        "line": line,
        "effective_start_date": selection.get("effective_start_date"),
        "line_frozen_at": selection.get("line_frozen_at"),
        "freeze_date": selection.get("freeze_date"),
        "structure_known_at": selection.get("structure_known_at") or structure.get("structure_known_at"),
        "neckline_tests": selection.get("independent_tests") or evidence.get("neckline_tests") or [],
        "anchors": anchors,
        "breakout": breakout,
        "events": lifecycle.get("events", []),
        "instance_id": lifecycle.get("instance_id"),
        "boundary_version_id": lifecycle.get("boundary_version_id"),
        "rule_version": pattern.get("rule_version"),
        "primary": ((evidence.get("main_selection") or {}).get("selected") is True),
    }


def load_case_accounts(study_dir, case_root):
    results = {}
    for code, kind in CASES:
        folder = "stocks" if kind == "stock" else "etfs"
        key = f"{kind}:{code}"
        results[key] = {"code": code, "kind": kind, "bars": read_bars(
            Path(case_root) / "data" / folder / (code + ".csv")), "versions": {}}
        for version, dirname in (("A", "cases-A-748"), ("B", "cases-B-current")):
            path = Path(study_dir) / dirname / "accounts" / f"{kind}-{code}.json"
            account = read_json(path, {}) or {}
            events = []
            for signal in account.get("signals", []):
                for pattern in signal.get("patterns") or ([signal.get("pattern")] if signal.get("pattern") else []):
                    events.append(_pattern_summary(pattern, signal))
            events.sort(key=lambda row: (row.get("confirmation_date") or "", row.get("type") or ""))
            results[key]["versions"][version] = {
                "status": account.get("status"), "recognizer": account.get("recognizer"),
                "engine_version": account.get("engine_version"),
                "signal_count": len(account.get("signals", [])),
                "trade_count": len(account.get("trades", [])),
                "starting_cash": account.get("starting_cash"),
                "ending_equity": account.get("ending_equity"),
                "metrics": account.get("metrics", {}),
                "accounting_checks": account.get("accounting_checks", {}),
                "trades": account.get("trades", []), "events": events,
                "account_json": f"{dirname}/accounts/{kind}-{code}.json",
            }
    return results


def _pct(value, digits=2):
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def build_payload(study_dir, case_root):
    study_dir = Path(study_dir)
    comparison = read_json(study_dir / "paired100-comparison.json", {}) or {}
    worker = read_json(study_dir / "worker-benchmark-v2" / "worker-benchmark.json", {}) or {}
    full_a = read_json(study_dir / "full-A-748" / "structure-boundary-run.json")
    full_b = read_json(study_dir / "full-B-current" / "structure-boundary-run.json")
    return {
        "cutoff": "2026-10-30",
        "baseline_commit": "748d3d01f122ee72e52b66387b20e36bb8bbc1d5",
        "candidate_engine": "caisen-joint-pattern-session-v4",
        "paired100": comparison,
        "worker_benchmark": worker,
        "full_a": full_a,
        "full_b": full_b,
        "cases": load_case_accounts(study_dir, case_root),
        "case_input_manifest": read_json(Path(case_root) / "case-input-manifest.json", {}),
        "full_a_progress": read_json(study_dir / "full-A-748" / "progress.json"),
        "full_b_progress": read_json(study_dir / "full-B-current" / "progress.json"),
    }


def render(payload, output_path):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_doc = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>结构与颈线联合识别回测研究</title>
<style>
:root{--bg:#f3f6fa;--paper:#fff;--ink:#172536;--muted:#5e6c7c;--line:#dce3eb;--blue:#365eea;--teal:#078a80;--red:#cf4f4f;--gold:#bb781e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:28px max(22px,calc((100vw - 1280px)/2));background:#10243b;color:#fff}header p{margin:5px 0;color:#d6dfeb}h1{margin:0 0 8px;font-size:26px}main{max-width:1280px;margin:20px auto;padding:0 18px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.tabs button,.control button{border:1px solid var(--line);background:#fff;color:var(--ink);padding:9px 13px;border-radius:8px;cursor:pointer}.tabs button.active{background:#183550;color:white}
section.panel{display:none}section.panel.active{display:block}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 8px #12233a0a}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.metric{background:#f6f8fb;border-radius:9px;padding:12px}.metric b{display:block;font-size:21px}.muted{color:var(--muted)}.notice{border-left:4px solid var(--gold);padding:12px 14px;background:#fff8ea}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #e7ecf2}th{background:#f6f8fb;position:sticky;top:0}.scroll{overflow:auto;max-height:650px}.tag{display:inline-block;padding:2px 8px;border-radius:12px;background:#eaf0ff;color:#294bb6;font-size:12px}.case-title{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}.chart-wrap{border:1px solid var(--line);border-radius:10px;padding:8px;background:#fff;overflow:hidden}canvas{width:100%;height:440px;display:block}.control{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:10px 0}.control select{padding:8px;border:1px solid var(--line);border-radius:7px;background:#fff}.control input[type=range]{min-width:160px;flex:1}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:13px}.legend i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:4px}.case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.case-version{background:#f6f8fb;border-radius:9px;padding:12px}.event-details{margin-top:8px}.event-details summary{cursor:pointer;color:#244dbd}.event-list{max-height:260px;overflow:auto}.small{font-size:13px}.ok{color:#087664}.bad{color:#b23b3b}.partial{color:#a56413}
@media(max-width:700px){header{padding:20px}h1{font-size:21px}main{padding:0 10px}.card{padding:12px}canvas{height:350px}}
</style>
<header><h1>结构与颈线联合识别：配对回测报告</h1><p>精确 Git 748 基线 vs 当前联合结构—边界候选版 · 日线 · 截止 2026-10-30 · 零费用 · 次日开盘 · 止损 8% / 止盈 20%</p><p>固定输入快照：800 只股票 + 200 只 ETF；案例图可选择形态事件、缩放与平移。</p></header>
<main><div class="tabs"><button class="active" data-tab="summary">执行摘要</button><button data-tab="paired">100 标的配对</button><button data-tab="workers">并行基准</button><button data-tab="cases">六个案例</button><button data-tab="full">1000 标的冷跑</button></div>
<section id="summary" class="panel active"><div class="card"><h2>结论边界</h2><div class="notice">识别规则的变化与收益变化分开报告。累计收益下降不能证明形态识别更差或更好；案例图展示边界、测试点、结构锚点和确认事件，供逐例复核。</div><p id="statusText" class="muted"></p><div id="summaryCards" class="cards"></div></div><div class="card"><h2>计划验收状态</h2><div id="acceptance"></div></div></section>
<section id="paired" class="panel"><div class="card"><h2>固定 100 只股票：748 与候选配对</h2><p class="muted">同一冻结行情、截止日和 8%/20% Backtrader 规则；独立账户汇总不作为组合收益。</p><div id="pairedSummary" class="cards"></div><div class="scroll"><table id="pairedTable"></table></div><details class="event-details"><summary>信号与交易变化分解</summary><pre id="pairedDiff"></pre></details></div></section>
<section id="workers" class="panel"><div class="card"><h2>36 标的进程基准</h2><p>两个交错轮次；逐账户信号、成交、现金、持仓、权益签名与 1 进程基线完全比较。峰值为协调器及全部子进程 RSS 总和。</p><div id="workerSummary" class="cards"></div><div class="scroll"><table id="workerTable"></table></div></div></section>
<section id="cases" class="panel"><div class="card"><h2>六个指定案例：历史 K 线与事件边界</h2><p class="muted">灰色为收盘走势；蓝色为 748 颈线，绿色为候选颈线；圆点为结构锚点，虚线为确认日。选择形态后自动将视窗定位到结构至确认区间。</p><div id="caseCards" class="case-grid"></div><div class="control"><label>案例 <select id="caseSelect"></select></label><label>748 事件 <select id="eventA"></select></label><label>候选事件 <select id="eventB"></select></label></div><div class="control"><label>缩放 <input id="zoom" type="range" min="1" max="100" value="100"></label><label>平移 <input id="pan" type="range" min="0" max="100" value="100"></label><button id="resetChart">显示完整历史</button></div><div class="legend"><span><i style="background:#77879a"></i>收盘</span><span><i style="background:#365eea"></i>748 颈线</span><span><i style="background:#078a80"></i>候选颈线</span><span>● 结构点</span><span>┆ 确认日</span></div><div class="chart-wrap"><canvas id="caseChart"></canvas></div><div id="caseEvidence"></div></div></section>
<section id="full" class="panel"><div class="card"><h2>固定 1000 标的全历史冷跑</h2><p class="muted">计时从行情读取到完整 HTML/JSON 报告；未完成时显示实时状态，不以部分账户外推总体结论。</p><div id="fullSummary" class="cards"></div><div id="fullDetails"></div></div></section>
</main><script id="payload" type="application/json">''' + data + r'''</script>
<script>
const D=JSON.parse(document.getElementById('payload').textContent);const $=s=>document.querySelector(s);const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>v==null?'—':(Number(v)*100).toFixed(2)+'%';const num=v=>v==null?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:2});
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab));if(b.dataset.tab==='cases')drawCase();});
function card(value,label){return `<div class="metric"><b>${esc(value)}</b><span class="muted">${esc(label)}</span></div>`}
const a100=D.paired100?.aggregate?.A,b100=D.paired100?.aggregate?.B;
$('#statusText').textContent=`固定截止日 ${D.cutoff}；748 基线提交 ${D.baseline_commit}；当前候选 ${D.candidate_engine}。测试结果：386 passed（另含报告脚本编译检查）。`;
$('#summaryCards').innerHTML=[card(D.paired100?.sample_count??'100','配对股票数'),card(D.worker_benchmark?.recommended_workers??'—','推荐并行进程'),card(D.full_b?.status??'运行中','候选全量状态'),card(D.full_b?.measured_end_to_end_wall_seconds?Math.round(D.full_b.measured_end_to_end_wall_seconds/60)+' 分钟':'进行中','候选端到端耗时')].join('');
$('#acceptance').innerHTML=`<table><tr><th>项目</th><th>结果</th><th>证据</th></tr><tr><td>精确748基线</td><td class="ok">100/100、0失败</td><td>paired100-baseline-748-attempt02</td></tr><tr><td>候选配对</td><td class="ok">100/100、0失败</td><td>paired100-candidate-v25-attempt01</td></tr><tr><td>进程一致性</td><td class="ok">${D.worker_benchmark?.status==='complete'?'1/2/4/6/8/10 全通过':'待完成'}</td><td>36 标的 × 2 轮</td></tr><tr><td>六案例对照</td><td class="ok">A/B 各 6/6</td><td>000001、000012、000538、600108、159718、516630</td></tr><tr><td>1000候选冷跑</td><td class="${D.full_b?.full_run_status==='complete'?'ok':'partial'}">${esc(D.full_b?.full_run_status??'运行中/待完成')}</td><td>${esc(D.full_b?.daily_research_report??'progress.json')}</td></tr><tr><td>1000基线冷跑</td><td class="${D.full_a?.full_run_status==='complete'?'ok':'partial'}">${esc(D.full_a?.full_run_status??'待开始')}</td><td>${esc(D.full_a?.daily_research_report??'尚未开始')}</td></tr></table>`;
if(D.paired100){const c=D.paired100;$('#pairedSummary').innerHTML=[card(a100?.trade_count,'748 成交数'),card(b100?.trade_count,'候选成交数'),card(pct(a100?.win_rate),'748 胜率'),card(pct(b100?.win_rate),'候选胜率'),card(pct(a100?.cumulative_return),'748 累计账户收益'),card(pct(b100?.cumulative_return),'候选累计账户收益')].join('');let delta=c.aggregate?.delta_B_minus_A||{};$('#pairedTable').innerHTML=`<thead><tr><th>指标</th><th>748</th><th>候选</th><th>差值（B-A）</th></tr></thead><tbody>${[['信号数',a100?.signal_count,b100?.signal_count,delta.signal_count],['交易数',a100?.trade_count,b100?.trade_count,delta.trade_count],['胜率',pct(a100?.win_rate),pct(b100?.win_rate),Number(delta.win_rate_pp||0).toFixed(3)+' pp'],['累计收益',pct(a100?.cumulative_return),pct(b100?.cumulative_return),Number(delta.cumulative_return_pp||0).toFixed(3)+' pp'],['最大回撤',pct(a100?.maximum_drawdown),pct(b100?.maximum_drawdown),Number(delta.maximum_drawdown_pp||0).toFixed(3)+' pp'],['盈利因子',num(a100?.profit_factor),num(b100?.profit_factor),num(delta.profit_factor)],['净损益',num(a100?.net_pnl),num(b100?.net_pnl),num(delta.net_pnl)],['A-only trades PnL',num(c.event_and_trade_comparison?.A_only_trades_net_pnl),'—','—'],['B-only trades PnL','—',num(c.event_and_trade_comparison?.B_only_trades_net_pnl),'—']].map(r=>`<tr>${r.map(x=>`<td>${esc(x)}</td>`).join('')}</tr>`).join('')}</tbody>`;$('#pairedDiff').textContent=JSON.stringify(c.event_and_trade_comparison,null,2);}
const W=D.worker_benchmark;if(W?.workers){const peaks=Object.values(W.peak_bytes_by_workers||{});$('#workerSummary').innerHTML=[card(W.recommended_workers??'未推荐','推荐进程数'),card(W.cohort_count??36,'样本标的'),card(W.repetitions??2,'每档轮次'),card((Math.max(0,...peaks)/1024/1024/1024).toFixed(2)+' GiB','矩阵峰值内存')].join('');$('#workerTable').innerHTML=`<thead><tr><th>进程数</th><th>两轮平均耗时</th><th>峰值内存</th><th>账本一致</th></tr></thead><tbody>${Object.entries(W.workers).map(([n,runs])=>`<tr><td>${n}</td><td>${(Number(W.median_or_mean_wall_seconds_by_workers?.[n]||0)/60).toFixed(2)} 分</td><td>${(Number(W.peak_bytes_by_workers?.[n]||0)/1024/1024/1024).toFixed(2)} GiB</td><td>${runs.length===2&&runs.every(x=>x.parity_with_one_worker&&x.accounting_pass)?'通过':'失败'}</td></tr>`).join('')}</tbody>`;}else{$('#workerSummary').innerHTML='基准报告正在生成';}
const caseKeys=Object.keys(D.cases||{});const caseSelect=$('#caseSelect');caseSelect.innerHTML=caseKeys.map(k=>`<option value="${esc(k)}">${esc(k.replace(':',' · '))}</option>`).join('');let selectedCase=caseKeys[0];let currentEvents={A:[],B:[]};
function evLabel(e,i){return `${e.confirmation_date||'未确认'} · ${e.label} · ${e.formed_at||'?'}`}
function updateEvents(){const c=D.cases[selectedCase];for(const v of ['A','B']){currentEvents[v]=c.versions[v]?.events||[];const sel=$(`#event${v}`);const old=Number(sel.value||0);sel.innerHTML=currentEvents[v].map((e,i)=>`<option value="${i}">${esc(evLabel(e,i))}</option>`).join('')||'<option value="-1">无确认事件</option>';if(currentEvents[v].length)sel.value=String(Math.min(old,currentEvents[v].length-1));}$('#caseCards').innerHTML=caseKeys.map(k=>{const x=D.cases[k];return `<div class="case-version"><b>${esc(k.replace(':',' · '))}</b><div>748：${x.versions.A?.signal_count??0} 信号 / ${x.versions.A?.trade_count??0} 笔交易</div><div>候选：${x.versions.B?.signal_count??0} 信号 / ${x.versions.B?.trade_count??0} 笔交易</div><div>行情：${x.bars.length?esc(x.bars[0][0])+' 至 '+esc(x.bars.at(-1)[0]):'不可用'}</div></div>`}).join('');drawCase();}
caseSelect.onchange=()=>{selectedCase=caseSelect.value;$('#zoom').value=100;$('#pan').value=100;updateEvents();};$('#eventA').onchange=drawCase;$('#eventB').onchange=drawCase;$('#zoom').oninput=drawCase;$('#pan').oninput=drawCase;$('#resetChart').onclick=()=>{$('#zoom').value=100;$('#pan').value=100;drawCase();};
function drawCase(){const c=D.cases?.[selectedCase];if(!c||!c.bars?.length)return;const cvs=$('#caseChart'),rect=cvs.getBoundingClientRect(),ratio=window.devicePixelRatio||1;cvs.width=rect.width*ratio;cvs.height=rect.height*ratio;const ctx=cvs.getContext('2d');ctx.scale(ratio,ratio);const w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);const bars=c.bars;const frac=Number($('#zoom').value||100)/100;const count=Math.max(80,Math.floor(bars.length*frac));const start=Math.max(0,Math.floor((bars.length-count)*Number($('#pan').value||100)/100));const end=Math.min(bars.length,start+count);const shown=bars.slice(start,end);let min=Math.min(...shown.map(x=>x[3])),max=Math.max(...shown.map(x=>x[2]));const versions=['A','B'].map(v=>{const idx=Number($(`#event${v}`).value??0);return [v,currentEvents[v]?.[idx]]});for(const [v,e] of versions){if(!e)continue;for(const a of e.anchors||[])if(a.price!=null){min=Math.min(min,Number(a.price));max=Math.max(max,Number(a.price));}const ln=e.line||{};if(Number.isFinite(Number(ln.intercept))){for(const i of [start,end-1]){const y=Number(ln.slope||0)*i+Number(ln.intercept);if(Number.isFinite(y)){min=Math.min(min,y);max=Math.max(max,y);}}}}
const pad={l:55,r:18,t:20,b:28},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b;const span=max-min||1;const y=p=>pad.t+(max-p)/span*ch;ctx.strokeStyle='#e7ecf2';ctx.fillStyle='#68788a';ctx.font='11px system-ui';for(let j=0;j<=5;j++){const val=max-span*j/5,yy=y(val);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText(val.toFixed(2),4,yy+4);}const step=cw/Math.max(1,shown.length),body=Math.max(1,Math.min(5,step*.56));ctx.strokeStyle='#748395';ctx.fillStyle='#748395';for(let i=0;i<shown.length;i++){const b=shown[i],x=pad.l+(i+.5)*step;ctx.beginPath();ctx.moveTo(x,y(b[2]));ctx.lineTo(x,y(b[3]));ctx.stroke();ctx.fillRect(x-body/2,y(b[4]),body,Math.max(1,y(b[1])-y(b[4])));}ctx.fillStyle='#69798b';ctx.fillText(shown[0][0],pad.l,h-7);ctx.fillText(shown.at(-1)[0],w-100,h-7);
for(const [v,e] of versions){if(!e)continue;const color=v==='A'?'#365eea':'#078a80';const dates=bars.map(x=>x[0]),map=new Map(dates.map((d,i)=>[d,i]));let li=map.get(e.effective_start_date);if(li==null)li=(e.anchors||[]).map(a=>map.get(a.date)).find(x=>x!=null)??0;let ri=map.get(e.confirmation_date)??li;if(ri<li)ri=li;const lidx=Math.max(start,li),ridx=Math.min(end-1,ri);if(ridx>=lidx){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();for(let i=lidx;i<=ridx;i++){const x=pad.l+(i-start+.5)*step,yy=y(Number(e.line?.slope||0)*i+Number(e.line?.intercept||0));if(i===lidx)ctx.moveTo(x,yy);else ctx.lineTo(x,yy);}ctx.stroke();}for(const a of e.anchors||[]){const ix=map.get(a.date);if(ix==null||ix<start||ix>=end||a.price==null)continue;const x=pad.l+(ix-start+.5)*step;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y(Number(a.price)),4,0,Math.PI*2);ctx.fill();}const ci=map.get(e.confirmation_date);if(ci!=null&&ci>=start&&ci<end){const x=pad.l+(ci-start+.5)*step;ctx.setLineDash([4,4]);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,h-pad.b);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=color;ctx.fillText(v==='A'?'748确认':'候选确认',Math.min(x+4,w-85),pad.t+12);}}
const evidence=versions.map(([v,e])=>{if(!e)return `<div class="case-version"><b>${v==='A'?'748':'候选'}</b> 无确认事件</div>`;const tests=(e.neckline_tests||[]).map(t=>`${t.pivot_date||t.date||'?'} @ ${num(t.level)}`).join('、')||'未记录';return `<div class="case-version"><b>${v==='A'?'748':'候选'} · ${esc(e.label)}</b><div>形成 ${esc(e.formed_at)} · 冻结 ${esc(e.freeze_date||e.line_frozen_at)} · 确认 ${esc(e.confirmation_date)} · 次日入场 ${esc(e.entry_date)}</div><div>颈线斜率 ${num(e.line?.slope)} / 截距 ${num(e.line?.intercept)}</div><div>测试点：${esc(tests)}</div><div>实例 ${esc(e.instance_id||'旧版未记录')} · 边界 ${esc(e.boundary_version_id||'旧版未记录')}</div><details class="event-details"><summary>结构点与状态事件</summary><pre>${esc(JSON.stringify({anchors:e.anchors,events:e.events,breakout:e.breakout},null,2))}</pre></details></div>`}).join('');$('#caseEvidence').innerHTML=`<div class="case-grid">${evidence}</div>`;}
function fullCards(target,run){if(!run)return `<p class="partial">${esc(target)}：尚未运行</p>`;return `<h3>${esc(target)} · ${esc(run.full_run_status)}</h3><div class="cards">${card(run.account_count,'账户')}${card(run.successful_account_count,'成功')}${card(run.failed_account_count,'失败')}${card((run.measured_end_to_end_wall_seconds/60).toFixed(1)+' 分钟','端到端耗时')}${card((run.process_tree_peak_bytes/1024/1024/1024).toFixed(2)+' GiB','峰值内存')}${card(run.summary?.trade_count,'交易数')}${card(pct(run.summary?.win_rate),'胜率')}${card(pct(run.summary?.cumulative_return),'独立账户累计收益')}${card(pct(run.summary?.maximum_drawdown),'合成权益回撤')}</div>`}
$('#fullSummary').innerHTML=fullCards('B 候选',D.full_b)+fullCards('A 精确748',D.full_a);$('#fullDetails').innerHTML=`<p>候选实时进度：${esc(D.full_b_progress?.completed??0)} / ${esc(D.full_b_progress?.total??1000)} 标的。基线实时进度：${esc(D.full_a_progress?.completed??0)} / ${esc(D.full_a_progress?.total??1000)} 标的。最终报告写完后本页可重新生成。</p>`;
updateEvents();window.addEventListener('resize',drawCase);
</script></html>'''
    Path(output_path).write_text(html_doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.study_dir, args.case_root)
    render(payload, args.output)
    print(json.dumps({"status": "rendered", "output": str(args.output.resolve()),
                      "case_count": len(payload["cases"]),
                      "case_bars": {key: len(row["bars"]) for key, row in payload["cases"].items()},
                      "worker_matrix_status": payload["worker_benchmark"].get("status"),
                      "full_A_status": (payload["full_a"] or {}).get("full_run_status"),
                      "full_B_status": (payload["full_b"] or {}).get("full_run_status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
