"""Build a standalone, answer-blind 20-chart review set from local market data."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
STOCK_ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from caisen.patterns import (  # noqa: E402
    _at,
    _duration_balance_ratio,
    _line,
    _low_position_quality,
    _passes_duration_balance,
    _passes_shoulder_tolerance,
    _passes_trough_tolerance,
    detect_patterns,
)
from caisen.pivots import detect_pivots  # noqa: E402


RULE_VERSION = "caisen-v16"
SOURCE_RULE_VERSION = "caisen-rules-v7"
WINDOWS = {"1d": 500, "1w": 520}
BUY_TYPES = ("w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag")
ALL_BULL_TYPES = (*BUY_TYPES, "spring")
TYPE_NAMES = {
    "w_bottom": "W底",
    "spring": "破底翻",
    "inverse_head_shoulders": "头肩底",
    "triangle_bottom": "收敛三角形底部",
    "bull_flag": "下飘旗形上涨中继",
}
ALIASES = {"日期": "date", "代码": "code", "名称": "name", "开盘": "open",
           "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume",
           "成交股数": "volume"}


def load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig").rename(columns=ALIASES)
    frame.columns = [str(column).lstrip("\ufeff").lower() for column in frame.columns]
    if not {"date", "open", "high", "low", "close", "volume"}.issubset(frame.columns):
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (frame.dropna(subset=["date", "open", "high", "low", "close"])
            .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True))


def chart_payload(case_id: str, instrument: dict, prefix: pd.DataFrame) -> dict:
    chart = prefix.tail(520)
    return {
        "id": case_id,
        "code": instrument["code"],
        "kind": instrument["kind"],
        "as_of": str(pd.Timestamp(prefix.date.iloc[-1]).date()),
        "bars": [[str(pd.Timestamp(row.date).date()), round(float(row.open), 4),
                  round(float(row.high), 4), round(float(row.low), 4),
                  round(float(row.close), 4), round(float(row.volume), 2)]
                 for row in chart.itertuples(index=False)],
    }


def quality_event_evidence(pattern) -> dict:
    evidence = {item["code"]: item["value"] for item in pattern.evidence}
    return {key: evidence.get(key) for key in (
        "structure_duration", "first_valid_break", "low_position",
        "prepattern_drawdown", "pattern_quality_pass", "close_confirmation")}


def topology_candidates(frame: pd.DataFrame, code: str, kind: str):
    """Yield plausible raw bottom topologies with shared gate measurements."""
    structure = frame.tail(520).reset_index(drop=True)
    pivots = detect_pivots(structure, radius=2, atr_multiple=2.0)
    if len(pivots) < 3:
        return
    for width in (5, 3):
        for offset in range(0, len(pivots) - width + 1):
            seq = pivots[offset:offset + width]
            kinds = [pivot.kind for pivot in seq]
            expected = ["low", "high", "low"] if width == 3 else ["low", "high", "low", "high", "low"]
            if kinds != expected:
                continue
            anchor = max(pivot.index for pivot in seq)
            known = max(pivot.confirmed_at for pivot in seq)
            if known >= len(structure):
                continue
            lows = [float(p.price) for p in seq if p.kind == "low"]
            bottom = min(lows)
            if width == 3:
                neck = {"slope": 0.0, "intercept": float(seq[1].price)}
                types = ["w_bottom"]
                if not _passes_trough_tolerance("w_bottom", seq):
                    continue
            else:
                neck = _line((seq[1].index, seq[1].price), (seq[3].index, seq[3].price))
                types = []
                if (_passes_trough_tolerance("w_bottom", seq)
                        and _passes_duration_balance("w_bottom", seq)):
                    types.append("w_bottom")
                if (seq[2].price < seq[0].price and seq[2].price < seq[4].price
                        and _passes_shoulder_tolerance("inverse_head_shoulders", seq, neck)):
                    types.append("inverse_head_shoulders")
                lower = _line((seq[0].index, seq[0].price), (seq[4].index, seq[4].price))
                upper = neck
                if lower["slope"] > 0 and upper["slope"] < 0 and _passes_duration_balance("triangle_bottom", seq):
                    cross = (lower["intercept"] - upper["intercept"]) / (upper["slope"] - lower["slope"])
                    if .5 <= (seq[4].index - seq[0].index) / max(cross - seq[0].index, 1) <= .75:
                        types.append("triangle_bottom")
            if not types:
                continue
            start = pd.Timestamp(structure.date.iloc[seq[0].index]).normalize()
            boundary = start + pd.DateOffset(months=3)
            known_date = pd.Timestamp(structure.date.iloc[known]).normalize()
            if known_date <= boundary:
                yield {"reason": "short_structure", "seq": seq, "known": known,
                       "structure": structure, "types": types, "neck": neck,
                       "bottom": bottom, "duration_days": int((known_date - start).days)}
                continue
            eligible_indexes = structure.index[(pd.to_datetime(structure.date) > boundary)
                                               & (structure.index >= known)]
            if not len(eligible_indexes):
                continue
            asof = int(eligible_indexes[0])
            start_index = min(p.index for p in seq)
            close = structure.close.to_numpy(dtype=float)
            crossed = any(close[i] > _at(neck, i) for i in range(start_index + 1, asof + 1))
            if crossed:
                continue
            quality = _low_position_quality(bottom, start, frame, structure.date.iloc[asof])
            if not quality["passed"]:
                yield {"reason": "high_position", "seq": seq, "known": known,
                       "structure": structure, "types": types, "neck": neck,
                       "bottom": bottom, "duration_days": int((structure.date.iloc[asof] - start).days),
                       "asof_index": asof, "quality": quality}


def make_case(case_id, instrument, prefix, assessment, *, pattern=None, raw=None):
    case = chart_payload(case_id, instrument, prefix)
    answer = {
        "id": case_id,
        "code": instrument["code"],
        "kind": instrument["kind"],
        "assessment": assessment,
        "rule_version": RULE_VERSION,
        "source_rule_version": SOURCE_RULE_VERSION,
        "as_of": case["as_of"],
        "data_window": {"start": case["bars"][0][0], "end": case["bars"][-1][0]},
    }
    if pattern is not None:
        answer.update({"pattern_type": pattern.pattern_type,
                       "pattern_id": pattern.pattern_id,
                       "state_at_cutoff": pattern.state,
                       "formed_at": pattern.formed_at,
                       "confirmation_date": pattern.confirmation_date,
                       "evidence": quality_event_evidence(pattern)})
    if raw is not None:
        seq = raw["seq"]
        answer.update({"candidate_topology": raw["types"],
                       "formation_start": str(seq[0].date),
                       "last_structure_point": str(seq[-1].date),
                       "duration_days": raw["duration_days"],
                       "gate_evidence": raw.get("quality"),
                       "notice": "Diagnostic from a real local chart and shared rule gates; not an independent human label."})
    return case, answer


def choose_samples():
    files = []
    for kind, directory in (("stock", STOCK_ROOT / "data" / "stocks"),
                            ("etf", STOCK_ROOT / "data" / "etfs")):
        files.extend((path, kind) for path in sorted(directory.glob("*.csv"))[:100])
    pools = {"positive": [], "forming": [], "negative": [], "short_structure": [], "high_position": []}
    instruments = []
    for path, kind in files:
        bars = load_bars(path)
        if len(bars) < 120:
            continue
        instrument = {"code": path.stem, "kind": kind}
        instruments.append((instrument, bars))
        patterns = detect_patterns(bars.tail(500), code=path.stem, kind=kind,
                                   background_bars=bars, structure_window=500)
        for pattern in patterns:
            if pattern.pattern_type not in (*BUY_TYPES, "bull_flag"):
                continue
            if pattern.confirmation_date:
                cutoff = pd.Timestamp(pattern.confirmation_date)
                prefix = bars[bars.date <= cutoff].reset_index(drop=True)
                replay = detect_patterns(prefix.tail(500), code=path.stem, kind=kind,
                                         background_bars=prefix, structure_window=500)
                at_signal = next((item for item in replay if item.pattern_id == pattern.pattern_id
                                  and item.state == "confirmed"), None)
                if at_signal is not None:
                    pools["positive"].append((instrument, prefix, at_signal))
            elif pattern.state == "forming":
                pools["forming"].append((instrument, bars, pattern))
        for raw in topology_candidates(bars, path.stem, kind):
            if raw["reason"] == "short_structure":
                prefix = bars[bars.date <= raw["structure"].date.iloc[raw["known"]]].reset_index(drop=True)
                pools["short_structure"].append((instrument, prefix, raw))
            elif raw["reason"] == "high_position":
                cutoff = raw["structure"].date.iloc[raw["asof_index"]]
                prefix = bars[bars.date <= cutoff].reset_index(drop=True)
                pools["high_position"].append((instrument, prefix, raw))
        buy_patterns = [pattern for pattern in patterns if pattern.pattern_type in ALL_BULL_TYPES]
        if not buy_patterns:
            pools["negative"].append((instrument, bars, None))

    selection = []
    used_codes = set()
    # One machine-confirmed sample from each of four bullish families.
    for pattern_type in BUY_TYPES:
        selected = next((entry for entry in pools["positive"]
                         if entry[2].pattern_type == pattern_type and entry[0]["code"] not in used_codes), None)
        if selected:
            selection.append(("positive", selected))
            used_codes.add(selected[0]["code"])
    # Include an additional positive event and four genuinely forming candidates.
    for entry in pools["positive"]:
        if len([item for item in selection if item[0] == "positive"]) >= 6:
            break
        if entry[0]["code"] not in used_codes:
            selection.append(("positive", entry))
            used_codes.add(entry[0]["code"])
    for entry in pools["forming"]:
        if len([item for item in selection if item[0] == "forming"]) >= 4:
            break
        if entry[0]["code"] not in used_codes:
            selection.append(("forming", entry))
            used_codes.add(entry[0]["code"])
    for category in ("short_structure", "high_position", "negative"):
        limit = 4 if category != "negative" else 2
        for entry in pools[category]:
            if len([item for item in selection if item[0] == category]) >= limit:
                break
            if entry[0]["code"] not in used_codes:
                selection.append((category, entry))
                used_codes.add(entry[0]["code"])
    # Fill any remaining slots with unique, no-buy-pattern market windows.
    for entry in pools["negative"]:
        if len(selection) >= 20:
            break
        if entry[0]["code"] not in used_codes:
            selection.append(("negative", entry))
            used_codes.add(entry[0]["code"])
    random.Random(20261029).shuffle(selection)
    return selection[:20], {key: len(value) for key, value in pools.items()}, len(files)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>形态盲评 · 第一轮</title><style>
:root{color-scheme:dark;--bg:#09111f;--panel:#111c2d;--ink:#edf3fc;--muted:#99abc3;--line:#2b3a50;--accent:#61d7c5;--orange:#ffb86b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:16px;align-items:center}h1{font-size:18px;margin:0}header p{margin:3px 0 0;color:var(--muted);font-size:13px}.layout{padding:14px 18px;display:grid;gap:12px;grid-template-columns:minmax(0,1fr) 320px}.chartbox,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}.chartbox{min-height:66vh;position:relative;padding:8px}.chartbox canvas{width:100%;height:64vh;display:block;touch-action:none}.panel{padding:14px}.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;padding:4px 3px 10px}.toolbar{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:8px}button{background:#1d2d43;color:var(--ink);border:1px solid #40526b;border-radius:8px;padding:8px 10px;cursor:pointer}button:hover{border-color:var(--accent)}button.active{background:#17483f;border-color:var(--accent)}.choice{display:grid;gap:6px;margin:8px 0 12px}.choice label{display:flex;gap:8px;align-items:center;color:#d5e0ef}.fieldtitle{color:var(--muted);font-size:13px;display:block;margin:10px 0 5px}.subchoices{display:grid;grid-template-columns:1fr 1fr;gap:5px}.subchoices label{display:flex;gap:7px;align-items:center;color:#d5e0ef;font-size:13px}textarea{width:100%;min-height:110px;resize:vertical;background:#0b1524;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:9px}#status{color:var(--accent);font-size:12px;min-height:18px}.footer{padding:0 18px 16px;color:var(--muted);font-size:12px}.hint{color:var(--muted);font-size:12px;margin:8px 0}.casecount{font-variant-numeric:tabular-nums;color:var(--accent)}@media(max-width:950px){.layout{grid-template-columns:1fr}.chartbox{min-height:52vh}.chartbox canvas{height:50vh}}
</style></head><body>
<header><div><h1>形态盲评 · 第一轮</h1><p>逐图判断形态是否成立；机器结果与后续行情不显示。</p></div><div class="casecount" id="progress"></div></header>
<main class="layout"><section><div class="meta"><span id="caption"></span><span>滚轮缩放 · 拖动平移</span></div><div class="chartbox"><canvas id="chart"></canvas></div></section>
<aside class="panel"><div class="toolbar"><button id="prev">上一例</button><button id="next">下一例</button><button id="full-range">全范围</button><button id="draw">画线</button><button id="undo">撤销线</button><button id="clear">清空线</button></div>
<span class="fieldtitle">总体判断</span><div class="choice"><label><input type="radio" name="judgment" value="成立">形态成立</label><label><input type="radio" name="judgment" value="不成立">形态不成立</label><label><input type="radio" name="judgment" value="不确定">不确定</label></div>
<span class="fieldtitle">识别形态（可多选）</span><div class="subchoices"><label><input type="checkbox" name="pattern" value="w_bottom">W底</label><label><input type="checkbox" name="pattern" value="inverse_head_shoulders">头肩底</label><label><input type="checkbox" name="pattern" value="triangle_bottom">收敛三角底</label><label><input type="checkbox" name="pattern" value="bull_flag">下飘旗形</label><label><input type="checkbox" name="pattern" value="other_or_none">其他 / 无书本形态</label></div>
<span class="fieldtitle">突破确认状态</span><div class="choice"><label><input type="radio" name="confirmation" value="已确认">已确认</label><label><input type="radio" name="confirmation" value="未确认">未确认</label><label><input type="radio" name="confirmation" value="不确定">不确定</label></div>
<label class="fieldtitle" for="notes">备注</label><textarea id="notes" placeholder="可记录结构点、突破位置、低位依据等"></textarea>
<p class="hint">画线模式下在图表拖动即可标注。线段保存为日期与价格坐标，缩放后仍可复现。</p>
<div class="toolbar"><button id="export-json">导出评审JSON</button><button id="export-image">导出当前带标注图</button></div><div id="status"></div></aside></main>
<div class="footer">20个真实本地行情窗口，样本用于规则校准，不代表独立准确率估计。样本截至日已截断；此页不加载机器答案文件。</div>
<script>window.REVIEW_CASES=__CASE_DATA__;</script><script>
(()=>{const cases=window.REVIEW_CASES,canvas=document.querySelector('#chart'),ctx=canvas.getContext('2d'),status=document.querySelector('#status');let index=0,range=0,pan=0,drawing=false,drag=null,live=null;const storeKey='caisen-blind-review-v1';let saved=JSON.parse(localStorage.getItem(storeKey)||'{}');
function current(){return cases[index]}function lines(){return saved[current().id]?.lines||[]}function ensure(){const c=current();saved[c.id] ||= {judgment:'',patterns:[],confirmation:'',notes:'',lines:[]};return saved[c.id]}function save(){const v=ensure();v.judgment=document.querySelector('input[name=judgment]:checked')?.value||'';v.patterns=[...document.querySelectorAll('input[name=pattern]:checked')].map(x=>x.value);v.confirmation=document.querySelector('input[name=confirmation]:checked')?.value||'';v.notes=document.querySelector('#notes').value;localStorage.setItem(storeKey,JSON.stringify(saved));}
function visible(){let b=current().bars,n=b.length,end=Math.max(1,n-pan),start=Math.max(0,end-range);return [start,end]}
function coords(){const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;return {w:r.width,h:r.height,scale:d}}
function geometry(w,h){const pad={l:66,r:18,t:15,b:31},plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b,priceH=plotH*.74,volumeGap=14,volumeTop=pad.t+priceH+volumeGap,volumeH=plotH-priceH-volumeGap;return {pad,plotW,plotH,priceH,volumeTop,volumeH}}
function xyToValue(x,y){const [s,e]=visible(),{w,h}=coords(),{pad,plotW,priceH}=geometry(w,h),b=current().bars;const i=Math.max(s,Math.min(e-1,s+(x-pad.l)/plotW*(e-s))),prices=b.slice(s,e).flatMap(k=>[k[2],k[3]]),lo=Math.min(...prices),hi=Math.max(...prices),v=hi-(y-pad.t)/priceH*(hi-lo||1),near=Math.max(s,Math.min(e-1,Math.round(i)));return {date:b[near][0],price:Number(v.toFixed(4))}}
function valueToXY(pt){const [s,e]=visible(),{w,h}=coords(),{pad,plotW,priceH}=geometry(w,h),b=current().bars;let i=b.findIndex(k=>k[0]>=pt.date);if(i<0)i=b.length-1;const prices=b.slice(s,e).flatMap(k=>[k[2],k[3]]),lo=Math.min(...prices),hi=Math.max(...prices);return {x:pad.l+(i-s)/(e-s)*plotW,y:pad.t+(hi-pt.price)/(hi-lo||1)*priceH}}
function render(){const {w,h,scale}=coords(),dpr=scale;canvas.width=Math.floor(w*dpr);canvas.height=Math.floor(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b1524';ctx.fillRect(0,0,w,h);const [s,e]=visible(),b=current().bars.slice(s,e),{pad,plotW,plotH,priceH,volumeTop,volumeH}=geometry(w,h);if(!b.length)return;const lo=Math.min(...b.map(x=>x[3])),hi=Math.max(...b.map(x=>x[2])),step=plotW/b.length,cw=Math.max(2,step*.58),y=p=>pad.t+(hi-p)/(hi-lo||1)*priceH,maxVolume=Math.max(1,...b.map(x=>x[5]||0));ctx.font='11px sans-serif';ctx.strokeStyle='#24344a';ctx.fillStyle='#91a5c0';for(let g=0;g<=4;g++){let yy=pad.t+priceH*g/4;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText((hi-(hi-lo)*g/4).toFixed(2),5,yy+4)}ctx.strokeStyle='#2b3a50';ctx.beginPath();ctx.moveTo(pad.l,volumeTop-7);ctx.lineTo(w-pad.r,volumeTop-7);ctx.stroke();ctx.fillStyle='#91a5c0';ctx.fillText('成交量',5,volumeTop+11);
b.forEach((k,j)=>{let x=pad.l+(j+.5)*step,o=y(k[1]),hh=y(k[2]),ll=y(k[3]),c=y(k[4]),up=k[4]>=k[1];ctx.strokeStyle=up?'#48c9a9':'#ef7777';ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,hh);ctx.lineTo(x,ll);ctx.stroke();ctx.fillRect(x-cw/2,Math.min(o,c),cw,Math.max(1,Math.abs(c-o)));const vh=Math.max(1,(k[5]||0)/maxVolume*(volumeH-5));ctx.globalAlpha=.58;ctx.fillRect(x-cw/2,volumeTop+volumeH-vh,cw,vh);ctx.globalAlpha=1});
ctx.fillStyle='#91a5c0';[0,Math.floor(b.length/2),b.length-1].forEach(j=>{if(b[j])ctx.fillText(b[j][0],pad.l+j*step,h-7)});
function lineDraw(l,color='#61d7c5'){const a=valueToXY(l.start),z=valueToXY(l.end);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(z.x,z.y);ctx.stroke();ctx.fillStyle=color;ctx.beginPath();ctx.arc(a.x,a.y,3,0,7);ctx.arc(z.x,z.y,3,0,7);ctx.fill()};lines().forEach(l=>lineDraw(l));if(live)lineDraw(live,'#ffb86b');document.querySelector('#caption').textContent=`${current().kind==='etf'?'ETF':'股票'} ${current().code} · 截至 ${current().as_of}`;document.querySelector('#progress').textContent=`${index+1} / ${cases.length}`;document.querySelector('#draw').classList.toggle('active',drawing)}
function load(){const c=current(),v=saved[c.id]||{};document.querySelector('#notes').value=v.notes||'';document.querySelectorAll('input[name=judgment]').forEach(x=>x.checked=x.value===(v.judgment||''));document.querySelectorAll('input[name=confirmation]').forEach(x=>x.checked=x.value===(v.confirmation||''));document.querySelectorAll('input[name=pattern]').forEach(x=>x.checked=(v.patterns||[]).includes(x.value));range=c.bars.length;pan=0;live=null;render()}
document.querySelector('#prev').onclick=()=>{save();index=(index+cases.length-1)%cases.length;load()};document.querySelector('#next').onclick=()=>{save();index=(index+1)%cases.length;load()};document.querySelector('#full-range').onclick=()=>{range=current().bars.length;pan=0;render()};document.querySelector('#draw').onclick=()=>{drawing=!drawing;render()};document.querySelector('#undo').onclick=()=>{lines().pop();save();render()};document.querySelector('#clear').onclick=()=>{ensure().lines=[];save();render()};document.querySelector('#notes').oninput=save;document.querySelectorAll('input[name=judgment],input[name=pattern],input[name=confirmation]').forEach(x=>x.onchange=save);
canvas.onwheel=e=>{e.preventDefault();let old=range;range=Math.max(35,Math.min(current().bars.length,Math.round(range*(e.deltaY<0?.82:1.22))));if(range<old)pan=Math.min(current().bars.length-range,pan);render()};canvas.onpointerdown=e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(drawing){drag={start:xyToValue(x,y),now:xyToValue(x,y)};canvas.setPointerCapture(e.pointerId)}else{drag={pan:true,x:e.clientX,offset:pan};canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{if(!drag)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(drag.pan){const step=coords().w/range;pan=Math.max(0,Math.min(current().bars.length-range,drag.offset+Math.round((drag.x-e.clientX)/step)));render()}else{drag.now=xyToValue(x,y);live={start:drag.start,end:drag.now};render()}};canvas.onpointerup=e=>{if(drag&&!drag.pan&&drag.now){ensure().lines.push({start:drag.start,end:drag.now});save()}drag=null;live=null;render()};
function download(name,blob){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}document.querySelector('#export-json').onclick=()=>{save();const out={rule_version:'caisen-v16',window:{daily:500,weekly:520},exported_at:new Date().toISOString(),cases:cases.map(c=>({id:c.id,code:c.code,kind:c.kind,as_of:c.as_of,chart_range:{from:c.bars[0][0],to:c.bars[c.bars.length-1][0]},judgment:saved[c.id]?.judgment||'',pattern_types:saved[c.id]?.patterns||[],confirmation_status:saved[c.id]?.confirmation||'',notes:saved[c.id]?.notes||'',lines:saved[c.id]?.lines||[]}))};download('blind-pattern-review.json',new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));status.textContent='已导出评审JSON'};document.querySelector('#export-image').onclick=()=>{canvas.toBlob(blob=>download(`${current().id}-annotated.png`,blob));status.textContent='已导出当前图表PNG'};window.onresize=render;load()})();
</script></body></html>"""


def main():
    selection, pool_counts, input_count = choose_samples()
    cases, answers = [], []
    for number, (category, entry) in enumerate(selection, 1):
        case_id = f"R{number:02d}"
        if category == "positive":
            instrument, prefix, pattern = entry
            assessment = {"result": "detector accepted at signal cutoff",
                          "pattern_type": pattern.pattern_type,
                          "pattern_name": TYPE_NAMES.get(pattern.pattern_type, pattern.pattern_type)}
            case, answer = make_case(case_id, instrument, prefix, assessment, pattern=pattern)
        elif category == "forming":
            instrument, prefix, pattern = entry
            assessment = {"result": "forming candidate without confirmation",
                          "pattern_type": pattern.pattern_type,
                          "pattern_name": TYPE_NAMES.get(pattern.pattern_type, pattern.pattern_type)}
            case, answer = make_case(case_id, instrument, prefix, assessment, pattern=pattern)
        elif category in ("short_structure", "high_position"):
            instrument, prefix, raw = entry
            assessment = {"result": "rejected gate diagnostic", "reason": category,
                          "candidate_topology": raw["types"]}
            case, answer = make_case(case_id, instrument, prefix, assessment, raw=raw)
        else:
            instrument, prefix, _ = entry
            assessment = {"result": "no selected bullish pattern emitted at cutoff"}
            case, answer = make_case(case_id, instrument, prefix, assessment)
        cases.append(case)
        answers.append(answer)
    report = {
        "rule_version": RULE_VERSION,
        "source_rule_version": SOURCE_RULE_VERSION,
        "sample_count": len(cases),
        "input_files_considered": input_count,
        "candidate_pool_counts": pool_counts,
        "composition": {key: sum(answer["assessment"].get("reason") == key
                                    or (key == "positive" and answer["assessment"]["result"].startswith("detector"))
                                    or (key == "forming" and answer["assessment"]["result"].startswith("forming"))
                                    or (key == "negative" and answer["assessment"]["result"].startswith("no selected"))
                                    for answer in answers)
                       for key in ("positive", "forming", "short_structure", "high_position", "negative")},
        "cases": answers,
        "method_note": "Purposeful calibration examples from actual local OHLC data. Machine readings are kept in this separate development-side file and are not human labels or an independent accuracy estimate.",
    }
    report_dir = PROJECT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "pattern-blind-review.machine-answers.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__CASE_DATA__", json.dumps(cases, ensure_ascii=False, separators=(",", ":")))
    (report_dir / "pattern-blind-review.html").write_text(html, encoding="utf-8")
    print(json.dumps({"cases": len(cases), "input_files_considered": input_count,
                      "composition": report["composition"], "pool_counts": pool_counts,
                      "html": str(report_dir / "pattern-blind-review.html"),
                      "answers": str(report_dir / "pattern-blind-review.machine-answers.json")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
