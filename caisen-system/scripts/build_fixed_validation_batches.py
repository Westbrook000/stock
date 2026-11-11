"""Freeze six answer-blind 20-case reviews for the main-parent strategy.

The export function consumes already frozen chart prefixes and a separate
machine sidecar. It never writes machine labels into the reviewer pages.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import argparse
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))


VALIDATION_VERSION = "caisen-main-parent-v2-caisen-v17-main-bottom-v1"
VALIDATION_SLUG = "main-parent-v17"
FIXED_STRATEGY_VERSION = "pattern_fixed_8_20_main_parent_v2"
MAIN_BOTTOM_RULE_VERSION = "caisen-v17-main-bottom-v1"
LEGACY_RULE_VERSION = "caisen-v16"
SOURCE_RULE_VERSION = "caisen-rules-v7"
CASE_COUNT = 120
BATCH_SIZE = 20
SHAPES = {
    "w_bottom": "W底", "spring": "破底翻", "bull_flag": "下飘旗形上涨中继",
    "inverse_head_shoulders": "头肩底", "triangle_bottom": "收敛三角形底部",
    "m_top": "M头", "false_breakout": "假突破", "bear_flag": "上飘旗形下跌中继",
    "head_shoulders": "头肩顶", "triangle_top": "收敛三角形头部",
    "other_shape": "其他形态", "no_book_pattern": "没有任何书本形态",
}
BOOK_TYPES = set(SHAPES) - {"other_shape", "no_book_pattern"}
REVIEW_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>主结构策略盲评 · v17</title><style>
:root{color-scheme:dark;--bg:#09111f;--panel:#111c2d;--ink:#edf3fc;--muted:#99abc3;--line:#2b3a50;--accent:#61d7c5;--orange:#ffb86b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}h1{font-size:18px;margin:0}header p,.hint{color:var(--muted);font-size:12px;margin:3px 0}.layout{padding:14px 18px;display:grid;grid-template-columns:minmax(0,1fr) 390px;gap:12px}.chartbox,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}.chartbox{padding:8px}.meta{display:flex;justify-content:space-between;color:var(--muted);padding:4px 4px 8px}canvas{width:100%;height:68vh;min-height:480px;display:block;background:#0b1524;border-radius:7px;touch-action:none}.panel{padding:12px;max-height:83vh;overflow:auto}.toolbar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px}button{background:#1d2d43;color:var(--ink);border:1px solid #40526b;border-radius:7px;padding:7px 9px;cursor:pointer}button:hover{border-color:var(--accent)}button.active{background:#17483f;border-color:var(--accent)}.global{padding:8px;border:1px solid #614c36;border-radius:8px;margin:8px 0;color:#ffcd8a}.shape-list{display:grid;grid-template-columns:1fr 1fr;gap:5px}.shape-list label{display:flex;gap:6px;align-items:center;font-size:12px}.shape-card{border-top:1px solid var(--line);padding:9px 2px;margin-top:7px}.shape-card h3{font-size:13px;margin:0 0 7px}.shape-fields{display:grid;grid-template-columns:1fr 1fr;gap:6px}.shape-fields label{font-size:11px;color:var(--muted)}select,input[type=date],textarea{width:100%;background:#0b1524;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px}textarea{min-height:50px;resize:vertical}.hint{margin:8px 0}.status{color:var(--accent);font-size:12px;min-height:16px}.casecount{color:var(--accent);font-variant-numeric:tabular-nums}@media(max-width:1000px){.layout{grid-template-columns:1fr}.panel{max-height:none}canvas{height:52vh;min-height:340px}}
</style></head><body><header><div><h1>主结构策略盲评 · caisen-v17 · 第__BATCH__批</h1><p>仅显示截至日行情。机器结果不在本页加载，完成并导出后由独立本地命令生成机器叠图。</p></div><div id="progress" class="casecount"></div></header>
<main class="layout"><section class="chartbox"><div class="meta"><span id="caption"></span><span>滚轮缩放 · 拖动平移 · 画线记录日期与价格</span></div><canvas id="chart"></canvas></section><aside class="panel"><div class="toolbar"><button id="prev">上一例</button><button id="next">下一例</button><button id="full">全范围</button><button id="draw">画线</button><button id="undo">撤销线</button><button id="clear">清空当前形态线</button></div>
<label class="global"><input type="checkbox" id="overall-uncertain"> 整体判断不确定（后续统计不纳入准确率计算）</label><p class="hint">可同时选择多个形态。每个形态单独记录确认状态、结构起止日期、备注和线段；“其他形态”和“没有任何书本形态”是不同选项。</p><div id="active-shape"></div><div class="shape-list" id="shape-list"></div><div id="shape-cards"></div><div class="toolbar" style="margin-top:10px"><button id="submit">提交并导出本批评审JSON</button></div><div class="status" id="status"></div></aside></main>
<script>window.REVIEW_CASES=__CASE_DATA__;</script><script>
(()=>{const cases=window.REVIEW_CASES,shapes=__SHAPES__,states=['已确认','未确认','已失效','不确定'],canvas=document.querySelector('#chart'),ctx=canvas.getContext('2d'),key='caisen-main-parent-v2-v17-batch-__BATCH__',status=document.querySelector('#status');let index=0,range=0,pan=0,drawing=false,drag=null,live=null,activeShape='';let saved=JSON.parse(localStorage.getItem(key)||'{}');
function current(){return cases[index]}function ensure(){const c=current();saved[c.id] ||= {overall_uncertain:false,shapes:{}};return saved[c.id]}function shapeRecord(id){const row=ensure();row.shapes[id] ||= {state:'不确定',start_date:'',end_date:'',notes:'',lines:[]};return row.shapes[id]}function save(){localStorage.setItem(key,JSON.stringify(saved))}function selected(){return [...document.querySelectorAll('.shape-check:checked')].map(x=>x.value)}
function renderControls(){const row=ensure();document.querySelector('#overall-uncertain').checked=!!row.overall_uncertain;const ids=selected();if(!ids.includes(activeShape))activeShape=ids[0]||'';document.querySelector('#active-shape').innerHTML=ids.length?'<label class="hint">线段归属形态<select id="line-owner">'+ids.map(id=>`<option value="${id}" ${id===activeShape?'selected':''}>${shapes[id]}</option>`).join('')+'</select></label>':'<p class="hint">先选择形态，再记录对应结构线。</p>';const owner=document.querySelector('#line-owner');if(owner)owner.onchange=()=>{activeShape=owner.value;render()};document.querySelector('#shape-cards').innerHTML=ids.map(id=>{const v=shapeRecord(id);return `<section class="shape-card"><h3>${shapes[id]}</h3><div class="shape-fields"><label>确认状态<select data-field="state" data-shape="${id}">${states.map(s=>`<option ${v.state===s?'selected':''}>${s}</option>`).join('')}</select></label><label>结构起始日期<input type="date" data-field="start_date" data-shape="${id}" value="${v.start_date||''}"></label><label>结构结束日期<input type="date" data-field="end_date" data-shape="${id}" value="${v.end_date||''}"></label><label>备注<textarea data-field="notes" data-shape="${id}">${escapeHtml(v.notes||'')}</textarea></label></div><p class="hint">已记录 ${v.lines.length} 条线段</p></section>`}).join('');document.querySelectorAll('[data-field]').forEach(el=>el.onchange=()=>{shapeRecord(el.dataset.shape)[el.dataset.field]=el.value;save();render()});}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function visible(){const n=current().bars.length,end=Math.max(1,n-pan);return [Math.max(0,end-range),end]}function geom(){const r=canvas.getBoundingClientRect();return {w:r.width,h:r.height,d:window.devicePixelRatio||1}}function plot(){const {w,h}=geom();return {l:58,r:12,t:14,b:26,w:w-70,h:h-40}}
function valueAt(x,y){const [s,e]=visible(),p=plot(),b=current().bars,prices=b.slice(s,e).flatMap(x=>[x[2],x[3]]),lo=Math.min(...prices),hi=Math.max(...prices),at=Math.max(s,Math.min(e-1,s+(x-p.l)/p.w*(e-s))),near=Math.max(s,Math.min(e-1,Math.round(at)));return {date:b[near][0],price:Number((hi-(y-p.t)/p.h*(hi-lo||1)).toFixed(4))}}
function xy(pt){const [s,e]=visible(),p=plot(),b=current().bars;let i=b.findIndex(x=>x[0]>=pt.date);if(i<0)i=b.length-1;const prices=b.slice(s,e).flatMap(x=>[x[2],x[3]]),lo=Math.min(...prices),hi=Math.max(...prices);return {x:p.l+(i-s)/(e-s)*p.w,y:p.t+(hi-pt.price)/(hi-lo||1)*p.h}}
function lineDraw(line,color){const a=xy(line.start),z=xy(line.end);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(z.x,z.y);ctx.stroke()}
function render(){const {w,h,d}=geom();canvas.width=Math.floor(w*d);canvas.height=Math.floor(h*d);ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b1524';ctx.fillRect(0,0,w,h);const [s,e]=visible(),bars=current().bars.slice(s,e),p=plot();if(!bars.length)return;const hi=Math.max(...bars.map(x=>x[2])),lo=Math.min(...bars.map(x=>x[3])),step=p.w/bars.length,cw=Math.max(1,step*.56),y=v=>p.t+(hi-v)/(hi-lo||1)*p.h,maxV=Math.max(1,...bars.map(x=>x[5]||0));ctx.strokeStyle='#26374c';ctx.fillStyle='#90a4bd';ctx.font='11px sans-serif';for(let g=0;g<=4;g++){const yy=p.t+p.h*g/4;ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke();ctx.fillText((hi-(hi-lo)*g/4).toFixed(2),5,yy+4)}bars.forEach((b,j)=>{const x=p.l+(j+.5)*step,up=b[4]>=b[1];ctx.strokeStyle=up?'#48c9a9':'#ef7777';ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,y(b[2]));ctx.lineTo(x,y(b[3]));ctx.stroke();ctx.fillRect(x-cw/2,Math.min(y(b[1]),y(b[4])),cw,Math.max(1,Math.abs(y(b[1])-y(b[4]))));ctx.globalAlpha=.45;ctx.fillRect(x-cw/2,h-23-Math.max(1,(b[5]||0)/maxV*45),cw,Math.max(1,(b[5]||0)/maxV*45));ctx.globalAlpha=1});for(const [id,v] of Object.entries(ensure().shapes||{}))for(const line of v.lines||[])lineDraw(line,id===activeShape?'#61d7c5':'#62748d');if(live)lineDraw(live,'#ffb86b');document.querySelector('#caption').textContent=`${current().kind==='etf'?'ETF':'股票'} ${current().code} · ${current().timeframe==='1w'?'周线':'日线'} · 截至 ${current().as_of}`;document.querySelector('#progress').textContent=`${index+1} / ${cases.length}`;document.querySelector('#draw').classList.toggle('active',drawing)}
function load(){const c=current(),v=saved[c.id]||{overall_uncertain:false,shapes:{}};document.querySelector('#overall-uncertain').checked=!!v.overall_uncertain;document.querySelectorAll('.shape-check').forEach(x=>x.checked=!!v.shapes?.[x.value]);renderControls();range=c.bars.length;pan=0;live=null;render()}
document.querySelector('#shape-list').innerHTML=Object.entries(shapes).map(([id,label])=>`<label><input type="checkbox" class="shape-check" value="${id}">${label}</label>`).join('');document.querySelectorAll('.shape-check').forEach(x=>x.onchange=()=>{const row=ensure();if(x.value==='no_book_pattern'&&x.checked)document.querySelectorAll('.shape-check:not([value="no_book_pattern"])').forEach(y=>y.checked=false);else if(x.checked)document.querySelector('.shape-check[value="no_book_pattern"]').checked=false;if(x.checked)shapeRecord(x.value);row.shapes=Object.fromEntries(Object.entries(row.shapes).filter(([id])=>document.querySelector(`.shape-check[value="${id}"]`)?.checked));renderControls();save();render()});document.querySelector('#overall-uncertain').onchange=e=>{ensure().overall_uncertain=e.target.checked;save()};
document.querySelector('#prev').onclick=()=>{save();index=(index+cases.length-1)%cases.length;load()};document.querySelector('#next').onclick=()=>{save();index=(index+1)%cases.length;load()};document.querySelector('#full').onclick=()=>{range=current().bars.length;pan=0;render()};document.querySelector('#draw').onclick=()=>{if(!selected().length){status.textContent='先选择一个形态';return}drawing=!drawing;render()};document.querySelector('#undo').onclick=()=>{if(activeShape)shapeRecord(activeShape).lines.pop();save();render()};document.querySelector('#clear').onclick=()=>{if(activeShape)shapeRecord(activeShape).lines=[];save();render()};canvas.onwheel=e=>{e.preventDefault();const old=range;range=Math.max(35,Math.min(current().bars.length,Math.round(range*(e.deltaY<0?.82:1.22))));if(range<old)pan=Math.min(current().bars.length-range,pan);render()};canvas.onpointerdown=e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(drawing&&activeShape){drag={start:valueAt(x,y),now:valueAt(x,y)};canvas.setPointerCapture(e.pointerId)}else{drag={pan:true,x:e.clientX,offset:pan};canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{if(!drag)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(drag.pan){pan=Math.max(0,Math.min(current().bars.length-range,drag.offset+Math.round((drag.x-e.clientX)/(geom().w/range))));render()}else{drag.now=valueAt(x,y);live={start:drag.start,end:drag.now};render()}};canvas.onpointerup=()=>{if(drag&&!drag.pan&&drag.now){shapeRecord(activeShape).lines.push({start:drag.start,end:drag.now});save()}drag=null;live=null;render()};
document.querySelector('#submit').onclick=()=>{save();const payload={validation_version:'__VERSION__',batch:__BATCH__,submitted_at:new Date().toISOString(),cases:cases.map(c=>{const answer=saved[c.id]||{overall_uncertain:false,shapes:{}};return {id:c.id,code:c.code,kind:c.kind,timeframe:c.timeframe,as_of:c.as_of,data_fingerprint:c.data_fingerprint||'',chart_range:{from:c.bars[0][0],to:c.bars[c.bars.length-1][0]},overall_uncertain:!!answer.overall_uncertain,shapes:answer.shapes||{}}})};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download=`main-parent-v17-blind-batch-__BATCH__-answers.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);status.textContent='已导出本批标注。请保存答案JSON，再用独立本地评估命令生成机器叠图。'};window.onresize=render;load()})();</script></body></html>'''


def _validate_cases(cases):
    if len(cases) != CASE_COUNT:
        raise ValueError("expected exactly %d frozen cases" % CASE_COUNT)
    ids = [str(case.get("id", "")) for case in cases]
    if any(not case_id for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("case IDs must be present and unique")
    for case in cases:
        if case.get("kind") not in {"stock", "etf"} or case.get("timeframe") not in {"1d", "1w"}:
            raise ValueError("every case needs stock/ETF kind and daily/weekly timeframe")
        as_of = str(case.get("as_of", ""))
        bars = case.get("bars") or []
        if not as_of or not bars:
            raise ValueError("every case needs as_of and chart bars")
        dates = [str(bar[0]) for bar in bars]
        if dates != sorted(dates):
            raise ValueError("bars must be ordered by date for %s" % case["id"])
        if dates[-1] > as_of:
            raise ValueError("bar after as_of in case %s" % case["id"])


def _blind_case(case):
    # Only fields required to render a chart are copied into the public HTML.
    result = {key: case[key] for key in ("id", "code", "kind", "timeframe", "as_of", "bars")}
    if case.get("source_data_sha256"):
        result["data_fingerprint"] = case["source_data_sha256"]
    return result


def existing_blind_codes(review_html):
    """Read prior blind cases solely to enforce code-disjoint sampling."""
    match = re.search(r"window\.REVIEW_CASES=(.*?);</script>",
                      Path(review_html).read_text(encoding="utf-8"), re.DOTALL)
    if not match:
        raise ValueError("could not find the frozen review cases")
    return {str(case["code"]) for case in json.loads(match.group(1))}


def _evidence(pattern, code):
    for item in getattr(pattern, "evidence", []) or []:
        if item.get("code") == code:
            return item.get("value") or {}
    return {}


def _has_current_detector_match(legacy_events, main_events, as_of):
    """Only current active/confirmed events qualify as detector-union cases."""
    active_states = {"forming", "confirmed"}
    for event in legacy_events:
        if (getattr(event, "pattern_type", None) in BOOK_TYPES
                and getattr(event, "direction", None) in {"bullish", "bearish"}
                and getattr(event, "state", None) in active_states):
            return True
    for event in main_events:
        structure = _evidence(event, "main_structure")
        selection = _evidence(event, "main_selection")
        if (structure.get("role") == "main" and selection.get("selected") is True
                and selection.get("decided_at") == as_of
                and getattr(event, "state", None) in active_states):
            return True
    return False


def _is_difficult_negative(legacy_events, main_events):
    """Flag live gate diagnostics, not ended patterns, for balanced review coverage."""
    for event in main_events:
        structure = _evidence(event, "main_structure")
        if structure.get("role") == "main" and structure.get("signal_eligible") is False:
            return True
    for event in legacy_events:
        if (getattr(event, "pattern_type", None) in BOOK_TYPES
                and getattr(event, "state", None) == "forming"):
            return True
    return False


def _preassigned_source(seed, code, kind, timeframe):
    """Choose the independent-window role before either detector is called."""
    key = "%s:sample-role:%s:%s:%s" % (seed, kind, code, timeframe)
    return "independent" if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0 else "union"


def select_frozen_sample(candidates, excluded_codes=(), seed=20261029):
    """Choose 120 unique-code cases from detector-union and independent windows.

    Candidates have ``sample_source`` equal to ``union`` (legacy/new detector
    emitted at the cutoff) or ``independent`` (neither emitted). Machine-side
    results stay in the returned sidecar and never enter the chart case.
    """
    excluded = {str(code) for code in excluded_codes} | {"600108"}
    eligible = [dict(candidate) for candidate in candidates
                if str(candidate.get("code")) not in excluded]
    randomized = random.Random(int(seed))
    selected = []
    used_codes = set()
    shortages = {}
    # Exactly 15 cases in each source x kind x timeframe cell. This keeps the
    # six batches balanced across stock/ETF and daily/weekly backgrounds.
    for source in ("union", "independent"):
        for kind in ("stock", "etf"):
            for timeframe in ("1d", "1w"):
                pool = [item for item in eligible
                        if item.get("sample_source") == source
                        and (source != "union" or item.get("union_signal") is True)
                        and item.get("kind") == kind and item.get("timeframe") == timeframe]
                randomized.shuffle(pool)
                count = 0
                for candidate in pool:
                    code = str(candidate["code"])
                    if code in used_codes:
                        continue
                    selected.append(candidate)
                    used_codes.add(code)
                    count += 1
                    if count == 15:
                        break
                if count != 15:
                    shortages["%s/%s/%s" % (source, kind, timeframe)] = count
    if shortages:
        raise ValueError("not enough code-disjoint stratified candidates: " + json.dumps(shortages, sort_keys=True))

    # Re-shuffle after quota selection so batch order does not disclose source.
    randomized.shuffle(selected)
    cases, sidecar = [], {}
    for number, candidate in enumerate(selected[:CASE_COUNT], 1):
        case_id = "M%03d" % number
        cases.append({"id": case_id, "code": str(candidate["code"]),
                      "kind": str(candidate["kind"]), "timeframe": str(candidate["timeframe"]),
                      "as_of": str(candidate["as_of"]),
                      "bars": candidate["bars"]})
        sidecar[case_id] = {key: value for key, value in candidate.items()
                            if key not in {"name", "display_name"}}
    _validate_cases(cases)
    return cases, sidecar


def _event_payload(pattern, geometry_bars=None):
    evidence = {item.get("code"): item.get("value") for item in getattr(pattern, "evidence", []) or []}
    geometry_bars = geometry_bars.reset_index(drop=True) if geometry_bars is not None else None
    lines = []
    neckline = getattr(pattern, "neckline", None) or evidence.get("main_structure", {}).get("neckline")
    if neckline and geometry_bars is not None and not geometry_bars.empty:
        pivot_indexes = sorted(value for key, value in (getattr(pattern, "key_points", {}) or {}).items()
                               if key.endswith("_index") and isinstance(value, int)
                               and 0 <= value < len(geometry_bars))
        first_index = pivot_indexes[0] if pivot_indexes else 0
        last_index = pivot_indexes[-1] if pivot_indexes else len(geometry_bars) - 1
        slope, intercept = float(neckline.get("slope", 0.0)), float(neckline.get("intercept", 0.0))
        lines.append({"kind": "neckline", "start": {
            "date": str(geometry_bars.date.iloc[first_index].date()),
            "price": slope * first_index + intercept}, "end": {
            "date": str(geometry_bars.date.iloc[last_index].date()),
            "price": slope * last_index + intercept}})
    structure = evidence.get("main_structure") or {}
    if structure and geometry_bars is not None and not geometry_bars.empty:
        start_date = structure.get("first_trough_date") or str(geometry_bars.date.iloc[0].date())
        end_date = structure.get("structure_end_date") or str(geometry_bars.date.iloc[-1].date())
        for field in ("support_lower", "support_upper"):
            if structure.get(field) is not None:
                lines.append({"kind": field, "start": {"date": start_date, "price": float(structure[field])},
                              "end": {"date": end_date, "price": float(structure[field])}})
    return {
        "pattern_id": getattr(pattern, "pattern_id", None),
        "pattern_type": getattr(pattern, "pattern_type", None),
        "direction": getattr(pattern, "direction", None),
        "state": getattr(pattern, "state", None),
        "formed_at": getattr(pattern, "formed_at", None),
        "confirmation_date": getattr(pattern, "confirmation_date", None),
        "rule_version": getattr(pattern, "rule_version", None),
        "source_rule_version": getattr(pattern, "source_rule_version", None),
        "parent_id": getattr(pattern, "parent_id", None),
        "neckline": getattr(pattern, "neckline", None),
        "key_points": getattr(pattern, "key_points", None),
        "main_structure": evidence.get("main_structure"),
        "main_selection": evidence.get("main_selection"),
        "neckline_breakout": evidence.get("neckline_breakout"),
        "weak_crossings": evidence.get("weak_crossings"),
        "execution_suppression": evidence.get("execution_suppression"),
        "machine_lines": lines,
    }


def scan_local_candidates(data_root, existing_review, legacy_detector, main_detector, linker,
                          *, max_instruments=None, seed=20261029):
    """Create seeded independent windows and current-event detector-union windows.

    ``legacy_detector`` is the frozen legacy/v16 compatibility callable; it is
    injected so tests can use deterministic synthetic detectors. No network
    market-data refresh or later bars are used.
    """
    import pandas as pd
    from caisen.market import aggregate_weekly, ObservedFridayCalendar
    from build_blind_pattern_review import load_bars

    root = Path(data_root)
    weekly_calendar = ObservedFridayCalendar()
    excluded_codes = existing_blind_codes(existing_review) | {"600108"}
    sources = []
    for kind, dirname in (("stock", "stocks"), ("etf", "etfs")):
        sources.extend((kind, path) for path in sorted((root / dirname).glob("*.csv")))
    sources.sort(key=lambda item: hashlib.sha256(
        (str(seed) + ":" + item[0] + ":" + item[1].stem).encode("utf-8")).hexdigest())
    if max_instruments is not None:
        sources = sources[:int(max_instruments)]

    candidates = []
    eligible_sources = [(kind, path) for kind, path in sources
                        if path.stem not in excluded_codes]

    def timeframe_for(kind, code):
        key = "%s:timeframe:%s:%s" % (seed, kind, code)
        return "1d" if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0 else "1w"

    def indexes_for(bars):
        return sorted({int(round((len(bars) - 1) * fraction)) for fraction in (.35, .55, .75, .9)})

    def independent_index(kind, code, timeframe, indexes):
        key = "%s:independent-asof:%s:%s:%s" % (seed, kind, code, timeframe)
        return indexes[int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(indexes)]

    def analyze(kind, code, bars, timeframe, index, role):
        daily_cutoff = bars.iloc[:index + 1].copy().reset_index(drop=True)
        if timeframe == "1w":
            signal_frame = aggregate_weekly(daily_cutoff, calendar=weekly_calendar)
            if signal_frame.empty:
                return None
            as_of = pd.Timestamp(signal_frame.date.iloc[-1]).date().isoformat()
            prefix = daily_cutoff[pd.to_datetime(daily_cutoff.date) <= pd.Timestamp(as_of)].reset_index(drop=True)
            structure_window = 520
        else:
            prefix = daily_cutoff
            signal_frame = prefix
            as_of = pd.Timestamp(prefix.date.iloc[-1]).date().isoformat()
            structure_window = 500
        old_window = signal_frame.tail(structure_window).copy().reset_index(drop=True)
        main_start = pd.Timestamp(as_of) - pd.DateOffset(years=10 if timeframe == "1w" else 5)
        main_window = signal_frame.loc[pd.to_datetime(signal_frame.date) >= main_start].copy().reset_index(drop=True)
        legacy = legacy_detector(old_window, code, kind, timeframe,
                                 background_bars=prefix, structure_window=structure_window)
        main_kwargs = {"code": code, "kind": kind, "timeframe": timeframe}
        if timeframe == "1w":
            main_kwargs["background_bars"] = prefix
        main = main_detector(signal_frame, **main_kwargs)
        bars_by_timeframe = {timeframe: old_window}
        if timeframe != "1d":
            bars_by_timeframe["1d"] = prefix
        linked = linker(main, legacy, bars_by_timeframe=bars_by_timeframe)
        union_signal = _has_current_detector_match(legacy, main, as_of)
        if role == "union" and not union_signal:
            return None
        old_events = [_event_payload(item, old_window) for item in legacy]
        main_events = [_event_payload(item, main_window) for item in main]
        legacy_ids = {old.pattern_id for old in legacy}
        linked_events = [_event_payload(item, old_window if getattr(item, "timeframe", timeframe) == timeframe
                                         and item.pattern_id in legacy_ids else main_window) for item in linked]
        chart_bars = signal_frame.tail(520 if timeframe == "1w" else 1260).copy().reset_index(drop=True)
        canonical_chart = [[pd.Timestamp(row.date).date().isoformat(), round(float(row.open), 4),
                            round(float(row.high), 4), round(float(row.low), 4),
                            round(float(row.close), 4), round(float(row.volume), 2)]
                           for row in chart_bars.itertuples(index=False)]
        fingerprint_source = json.dumps(canonical_chart, separators=(",", ":")).encode("utf-8")
        daily_fingerprint = prefix[["date", "open", "high", "low", "close", "volume"]].to_csv(
            index=False, date_format="%Y-%m-%d", float_format="%.12g").encode("utf-8")
        return {
            "code": str(code), "kind": kind, "as_of": as_of, "sample_source": role,
            "union_signal": union_signal,
            "sample_difficulty": ("gate_diagnostic" if role == "union"
                                  and _is_difficult_negative(legacy, main) else "background"),
            "timeframe": timeframe, "strategy_version": FIXED_STRATEGY_VERSION,
            "main_bottom_rule_version": MAIN_BOTTOM_RULE_VERSION,
            "legacy_detector_rule_version": LEGACY_RULE_VERSION,
            "source_rule_catalog_version": SOURCE_RULE_VERSION,
            "weekly_calendar_version": getattr(weekly_calendar, "version", None) if timeframe == "1w" else None,
            "fingerprint_scope": "canonical_visible_ohlcv_prefix_through_as_of",
            "bars": canonical_chart,
            "source_data_sha256": hashlib.sha256(fingerprint_source).hexdigest(),
            "daily_source_prefix_sha256": hashlib.sha256(daily_fingerprint).hexdigest(),
            "legacy_detector_events": old_events,
            "main_parent_events": main_events,
            "linked_execution_events": linked_events,
        }

    if max_instruments is not None:
        # Bounded diagnostic mode for synthetic and timing probes; not suitable
        # for freezing the balanced final set.
        for kind, path in eligible_sources[:int(max_instruments)]:
            code = path.stem
            bars = load_bars(path)
            if len(bars) < 150:
                continue
            timeframe = timeframe_for(kind, code)
            role = _preassigned_source(seed, code, kind, timeframe)
            indexes = indexes_for(bars)
            if role == "independent":
                indexes = [independent_index(kind, code, timeframe, indexes)]
            for index in indexes:
                candidate = analyze(kind, code, bars, timeframe, index, role)
                if candidate:
                    candidates.append(candidate)
                    if role == "union":
                        break
        return candidates

    # Freeze the independent code/timeframe/as-of cohort before any detector
    # runs. These 60 cases are never filtered, replaced, or reordered based on
    # machine output.
    independent_specs = []
    independent_codes = set()
    for kind in ("stock", "etf"):
        for timeframe in ("1d", "1w"):
            for source_kind, path in eligible_sources:
                if source_kind != kind:
                    continue
                code = path.stem
                if code in independent_codes or timeframe_for(kind, code) != timeframe:
                    continue
                if _preassigned_source(seed, code, kind, timeframe) != "independent":
                    continue
                bars = load_bars(path)
                if len(bars) < 150:
                    continue
                indexes = indexes_for(bars)
                index = independent_index(kind, code, timeframe, indexes)
                independent_specs.append((kind, code, path, timeframe, index))
                independent_codes.add(code)
                if sum(1 for spec in independent_specs if spec[0] == kind and spec[3] == timeframe) == 15:
                    break
            count = sum(1 for spec in independent_specs if spec[0] == kind and spec[3] == timeframe)
            if count < 15:
                raise ValueError("independent preselection shortage for %s/%s: %d of 15"
                                 % (kind, timeframe, count))
    # Only after the complete independent cohort is fixed do detector calls begin.
    for kind, code, path, timeframe, index in independent_specs:
        bars = load_bars(path)
        candidate = analyze(kind, code, bars, timeframe, index, "independent")
        if candidate is None:
            raise ValueError("preselected independent chart could not be built: %s" % code)
        candidates.append(candidate)

    # The union group is intentionally machine-selected. It contains current
    # forming/confirmed detections only; invalidated historical rows do not count.
    used_codes = set(independent_codes)
    union_shortages = {}
    for kind in ("stock", "etf"):
        for timeframe in ("1d", "1w"):
            count = 0
            for source_kind, path in eligible_sources:
                if source_kind != kind or path.stem in used_codes:
                    continue
                code = path.stem
                if timeframe_for(kind, code) != timeframe:
                    continue
                if _preassigned_source(seed, code, kind, timeframe) != "union":
                    continue
                bars = load_bars(path)
                if len(bars) < 150:
                    continue
                found = None
                for index in indexes_for(bars):
                    found = analyze(kind, code, bars, timeframe, index, "union")
                    if found:
                        break
                if found:
                    candidates.append(found)
                    used_codes.add(code)
                    count += 1
                    if count == 15:
                        break
            if count < 15:
                union_shortages["%s/%s" % (kind, timeframe)] = count
    if union_shortages:
        raise ValueError("detector-union shortage: " + json.dumps(union_shortages, sort_keys=True))
    return candidates


def build_local_validation_batches(data_root, existing_review, legacy_detector, main_detector,
                                   linker, output_dir, *, max_instruments=None, seed=20261029):
    candidates = scan_local_candidates(
        data_root, existing_review, legacy_detector, main_detector, linker,
        max_instruments=max_instruments, seed=seed,
    )
    cases, sidecar = select_frozen_sample(
        candidates, existing_blind_codes(existing_review), seed=seed,
    )
    return write_blind_batches(cases, sidecar, output_dir, seed=seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parent / "data")
    parser.add_argument("--previous-review", type=Path,
                        default=PROJECT / "reports" / "pattern-blind-review.html")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new empty directory; existing blind assets are never overwritten")
    parser.add_argument("--max-instruments", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)

    from caisen.main_bottoms import detect_main_bottoms, link_main_bottom_children
    from caisen.v16_baseline.patterns import detect_patterns as detect_patterns_v16

    candidates = scan_local_candidates(
        args.data_root, args.previous_review, detect_patterns_v16, detect_main_bottoms,
        link_main_bottom_children, max_instruments=args.max_instruments, seed=args.seed,
    )
    cases, sidecar = select_frozen_sample(
        candidates, existing_blind_codes(args.previous_review), seed=args.seed,
    )
    result = write_blind_batches(cases, sidecar, args.output_dir, seed=args.seed)
    print(json.dumps({"candidate_windows": len(candidates), "frozen_cases": len(cases),
                      "sample_composition": {
                          source: sum(item.get("sample_source") == source for item in sidecar.values())
                          for source in ("union", "independent")},
                      "sample_composition_by_kind_timeframe": {
                          "%s/%s/%s" % (source, kind, timeframe): sum(
                              item.get("sample_source") == source and item.get("kind") == kind
                              and item.get("timeframe") == timeframe for item in sidecar.values())
                          for source in ("union", "independent") for kind in ("stock", "etf")
                          for timeframe in ("1d", "1w")},
                      "difficult_negative_count": sum(item.get("sample_difficulty") == "gate_diagnostic"
                                                       for item in sidecar.values()), **result},
                     ensure_ascii=False, indent=2))


def _page(cases, batch_number):
    data = json.dumps([_blind_case(case) for case in cases], ensure_ascii=False,
                      separators=(",", ":")).replace("</", "<\\/")
    page = REVIEW_TEMPLATE.replace("__CASE_DATA__", data)
    page = page.replace("__SHAPES__", json.dumps(SHAPES, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("__VERSION__", VALIDATION_VERSION)
    page = page.replace("__BATCH_LABEL__", "%02d" % batch_number)
    page = page.replace("__BATCH__", str(int(batch_number)))
    return page


def write_blind_batches(cases, machine_sidecar, output_dir, seed=20261029):
    """Write 6 reviewer pages, a private sidecar, and a reproducible manifest."""
    cases = [dict(case) for case in cases]
    _validate_cases(cases)
    case_ids = {str(case["id"]) for case in cases}
    if set(map(str, machine_sidecar)) != case_ids:
        raise ValueError("machine sidecar must contain exactly one entry per case")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = [root / ("%s-blind-batch-%02d.html" % (VALIDATION_SLUG, number))
                for number in range(1, CASE_COUNT // BATCH_SIZE + 1)]
    expected += [root / (VALIDATION_SLUG + "-machine-sidecar.json"), root / (VALIDATION_SLUG + "-manifest.json")]
    existing = [path.name for path in expected if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing validation artifacts: " + ", ".join(existing))

    shuffled = list(cases)
    random.Random(int(seed)).shuffle(shuffled)
    batches = []
    html_files = []
    batch_for_case = {}
    for offset in range(0, CASE_COUNT, BATCH_SIZE):
        number = offset // BATCH_SIZE + 1
        path = expected[number - 1]
        batch_cases = shuffled[offset:offset + BATCH_SIZE]
        page = _page(batch_cases, number)
        path.write_text(page, encoding="utf-8")
        html_files.append(str(path.resolve()))
        batch_for_case.update({str(case["id"]): number for case in batch_cases})
        batches.append({"number": number, "html": str(path.resolve()),
                        "sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
                        "case_ids": [str(case["id"]) for case in batch_cases]})

    sidecar_path = expected[-2]
    sidecar_payload = {
        "validation_version": VALIDATION_VERSION,
        "fixed_strategy_version": FIXED_STRATEGY_VERSION,
        "main_bottom_rule_version": MAIN_BOTTOM_RULE_VERSION,
        "legacy_detector_rule_version": LEGACY_RULE_VERSION,
        "source_rule_catalog_version": SOURCE_RULE_VERSION,
        "cases": {str(key): dict(value, batch=batch_for_case[str(key)])
                  for key, value in sorted(machine_sidecar.items())},
        "note": "Machine-side evidence only. Human labels have not been supplied; this is not an accuracy result.",
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = expected[-1]
    manifest = {
        "validation_version": VALIDATION_VERSION,
        "fixed_strategy_version": FIXED_STRATEGY_VERSION,
        "main_bottom_rule_version": MAIN_BOTTOM_RULE_VERSION,
        "legacy_detector_rule_version": LEGACY_RULE_VERSION,
        "source_rule_catalog_version": SOURCE_RULE_VERSION,
        "sample_count": CASE_COUNT,
        "batch_size": BATCH_SIZE,
        "seed": int(seed),
        "method": "seeded independent code/time windows plus current active/confirmed legacy/new detector union",
        "fixed_strategy_version": FIXED_STRATEGY_VERSION,
        "main_bottom_rule_version": MAIN_BOTTOM_RULE_VERSION,
        "legacy_detector_rule_version": LEGACY_RULE_VERSION,
        "source_rule_catalog_version": SOURCE_RULE_VERSION,
        "prohibited_accuracy_claim": True,
        "batches": batches,
        "machine_sidecar": str(sidecar_path.resolve()),
        "sample_composition": {
            "%s/%s/%s" % (source, kind, timeframe): sum(
                item.get("sample_source") == source and item.get("kind") == kind
                and item.get("timeframe") == timeframe for item in machine_sidecar.values())
            for source in ("union", "independent") for kind in ("stock", "etf") for timeframe in ("1d", "1w")
        },
        "difficult_negative_count": sum(item.get("sample_difficulty") == "gate_diagnostic"
                                         for item in machine_sidecar.values()),
        "difficulty_definition": "current main-structure gate diagnostic or legacy forming candidate; ended history excluded from union role",
        "weekly_bar_source": "ObservedFridayCalendar; complete observed Monday-to-Friday weeks only, holiday weeks may be omitted",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"html_files": html_files, "machine_sidecar": str(sidecar_path.resolve()),
            "manifest": str(manifest_path.resolve())}


if __name__ == "__main__":
    main()
