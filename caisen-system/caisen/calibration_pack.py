"""Create blind, reproducible chart-review packs from the local market snapshot."""

from __future__ import annotations

import html
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .market import aggregate_weekly, default_china_calendar, discover_instruments, load_daily_bars


_LABELS = (
    ("W", "W底"), ("S", "破底翻"), ("WS", "W底破底翻"),
    ("BF+", "下飘旗形上涨中继"), ("IHS", "头肩底"), ("TB", "收敛三角形底部"),
    ("M", "M头"), ("FB", "假突破"), ("FBHS", "假突破（头肩顶）"),
    ("BF-", "上飘旗形下跌中继"), ("HS", "头肩顶"), ("TT", "收敛三角形头部"),
    ("N", "没有任何书本形态"),
)


def render_review_html(cases, seed=0):
    """Render an answer-ready HTML chart questionnaire without machine predictions."""
    payload = []
    for case in cases:
        bars = []
        for row in case.get("bars", []):
            bars.append({
                "date": str(pd.Timestamp(row["date"]).date()),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        payload.append({
            "case_id": str(case["case_id"]), "code": str(case["code"]),
            "name": str(case.get("name") or ""), "kind": str(case["kind"]),
            "timeframe": str(case["timeframe"]), "asof": str(case["asof"]),
            "bars": bars,
        })
    # A JSON script element is data, but escape HTML delimiters so untrusted
    # instrument names cannot terminate it and introduce executable markup.
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    legend = "".join(
        '<span><b>%s</b> %s</span>' % (html.escape(code), html.escape(name))
        for code, name in _LABELS
    )
    return _HTML.replace("__SEED__", str(int(seed))).replace("__LEGEND__", legend).replace("__CASES__", serialized)


def stratified_quotas(count, seed=0):
    """Allocate an exact sample count evenly across stock/ETF and daily/weekly."""
    if count < 1:
        raise ValueError("count must be positive")
    cells = [(kind, timeframe) for kind in ("stock", "etf") for timeframe in ("1d", "1w")]
    random.Random(seed).shuffle(cells)
    quotas = {cell: count // len(cells) for cell in cells}
    for cell in cells[:count % len(cells)]:
        quotas[cell] += 1
    return quotas


def build_blind_cases(root, count=20, seed=20261025):
    """Sample local daily and verified weekly charts, balanced by stocks and ETFs.

    Selection is random within each instrument/timeframe stratum. Machine
    predictions are deliberately not used, so this pack can surface misses as
    well as false positives; it is still a sample, not a prevalence estimate.
    """
    instruments = discover_instruments(root)
    rng = random.Random(seed)
    by_kind = defaultdict(list)
    for instrument in instruments:
        by_kind[instrument.kind].append(instrument)
    for items in by_kind.values():
        rng.shuffle(items)

    cells = [(kind, timeframe) for kind in ("stock", "etf") for timeframe in ("1d", "1w")]
    quotas = stratified_quotas(count, seed=seed)
    cells = list(quotas)

    calendar = default_china_calendar()
    cases, seen_codes = [], set()
    for kind, timeframe in cells:
        needed = quotas[(kind, timeframe)]
        if not needed:
            continue
        candidates = [item for item in by_kind[kind] if item.code not in seen_codes]
        found = 0
        while candidates and found < needed:
            instrument = candidates.pop()
            result = load_daily_bars(instrument, root=root)
            if result.quality_status in {"unavailable", "insufficient"}:
                continue
            if timeframe == "1d":
                frame = result.bars
                minimum, limit = 120, 180
            else:
                weekly = aggregate_weekly(result, calendar=calendar)
                frame = weekly.bars
                # Only periods backed by independently verified exchange dates
                # are eligible for formal human review.
                if calendar is not None and "week_start" in frame:
                    frame = frame[frame.week_start.map(calendar.is_week_verified)].reset_index(drop=True)
                minimum, limit = 40, 104
            if len(frame) < minimum:
                continue
            frame = frame.tail(limit)
            name = ""
            path = Path(instrument.path)
            try:
                head = pd.read_csv(path, encoding="utf-8-sig", nrows=1)
                name_col = next((column for column in head if str(column).lower() in {"name", "名称", "证券简称"}), None)
                if name_col:
                    name = str(head.iloc[0][name_col])
            except (OSError, UnicodeError, pd.errors.ParserError):
                pass
            records = frame[["date", "open", "high", "low", "close", "volume"]].to_dict("records")
            cases.append({
                "case_id": "blind-%04d" % (len(cases) + 1), "code": instrument.code,
                "name": name, "kind": kind, "timeframe": timeframe,
                "asof": str(pd.Timestamp(frame.date.iloc[-1]).date()), "bars": records,
            })
            seen_codes.add(instrument.code)
            found += 1
        if found < needed:
            raise ValueError("not enough usable local %s/%s charts: requested %d, found %d" % (
                kind, timeframe, needed, found
            ))
    return cases


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>蔡森量价形态盲标校准</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17212b;background:#f2f5f8}body{margin:0}.wrap{max-width:1120px;margin:24px auto;padding:0 16px}header,.case{background:#fff;border:1px solid #dbe3eb;border-radius:12px;padding:20px;margin-bottom:14px}h1{margin:0 0 8px}.sub,.prompt,.foot{color:#586879;line-height:1.6}.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:7px;margin:14px 0}.legend span{background:#f4f7fa;padding:7px;border-radius:6px;font-size:13px}.toolbar{display:flex;gap:8px;flex-wrap:wrap}button{border:0;border-radius:7px;background:#176b87;color:#fff;padding:10px 14px;font-weight:600;cursor:pointer}.secondary{background:#e8edf2;color:#26384a}.casehead{display:flex;justify-content:space-between;gap:10px;align-items:baseline}.casehead h2{font-size:18px;margin:0}.casehead span,.foot{font-size:13px;color:#586879}.chart{width:100%;height:auto;display:block}.prompt{margin:8px 0}.options{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px}.option{display:flex;gap:7px;border:1px solid #e1e7ed;border-radius:7px;padding:8px;cursor:pointer;font-size:13px}.option:has(input:checked){border-color:#176b87;background:#eef8fa}.state-row{display:flex;gap:8px;align-items:center;margin:12px 0;font-weight:600}.state-row select,.note{border:1px solid #d9e1e9;border-radius:7px;padding:9px;font:inherit}.note{width:calc(100% - 20px);min-height:50px}@media(max-width:700px){.casehead{display:block}.casehead span{display:block}}
</style></head><body><main class="wrap"><header><h1>蔡森量价形态 · 盲标校准</h1>
<div class="sub">本地当前行情快照 · 固定抽样种子 __SEED__ · 共 <b id="count"></b> 题。股票/ETF 与日/周线分层抽取，系统形态判断已隐藏。周线只采用有核验交易日历的完整周。</div>
<p class="prompt"><b>每题先标形态、再标状态。</b>结构已出现但尚未完成价格确认时，选择对应形态并标“形成中 / 未确认”。看不出书本形态才选 N。请按图上的周期和截止日期判断。</p>
<div class="legend">__LEGEND__</div><p class="foot">行情来自当前本地 CSV；抽样不依赖机器提名，因此可能发现漏识别，但本轮小样本不能估计总体准确率。当前快照回放不代表当时的未复权行情。</p>
<div class="toolbar"><button id="download">下载标注 JSON</button><button class="secondary" id="save">保存到此浏览器</button><button class="secondary" id="clear">清空作答</button><span id="status" class="foot"></span></div></header><div id="cases"></div></main>
<script type="application/json" id="case-data">__CASES__</script><script>
const cases=JSON.parse(document.getElementById('case-data').textContent),key='caisen-blind-review-v1-'+__SEED__,root=document.getElementById('cases');
const labels=__LABEL_CODES__;
document.getElementById('count').textContent=cases.length;
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function card(c,i){let id='case-'+i;return `<section class="case" id="${id}"><div class="casehead"><h2>第 ${i+1} 题 · ${esc(c.code)} ${esc(c.name)}</h2><span>${c.kind==='stock'?'股票':'ETF'} · ${c.timeframe==='1d'?'日线':'周线'} 截至 ${esc(c.asof)} · ${c.bars.length} 根</span></div><svg class="chart" viewBox="0 0 1000 390" role="img" aria-label="${esc(c.code)} K线和成交量"></svg><p class="prompt">形态类别（单选；复合结构请在备注补充次级形态）</p><div class="options">${labels.map(([v,n])=>`<label class="option"><input type="radio" name="p${i}" value="${esc(v)}"><span><b>${esc(v)}</b> ${esc(n)}</span></label>`).join('')}</div><label class="state-row">确认状态<select name="s${i}"><option value="">请选择</option><option value="forming">形成中 / 未确认</option><option value="confirmed">已确认</option><option value="none">无形态（N）</option></select></label><textarea class="note" placeholder="可选：关键高低点、颈线、确认/失效依据或复合形态"></textarea></section>`}
function draw(svg,bars){if(!bars.length){svg.innerHTML='<text x="30" y="40">无可用行情</text>';return}const W=1000,H=390,L=60,R=24,T=18,PH=250,VH=75,base=T+PH+18,lo=Math.min(...bars.map(x=>x.low)),hi=Math.max(...bars.map(x=>x.high)),pad=(hi-lo)*.05||1,min=lo-pad,max=hi+pad,mv=Math.max(...bars.map(x=>x.volume))||1,step=(W-L-R)/bars.length,cw=Math.max(1,step*.58);let out=[];out.push(`<text x="5" y="20" font-size="13">收盘价</text><text x="5" y="${base+VH-4}" font-size="13">成交量</text>`);for(let k=0;k<5;k++){let val=max-(max-min)*k/4,y=T+PH*k/4;out.push(`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#e4e9ef"/><text x="4" y="${y+4}" font-size="11" fill="#657487">${val.toFixed(2)}</text>`)}bars.forEach((b,j)=>{let x=L+j*step+step/2,py=p=>T+(max-p)/(max-min)*PH,up=b.close>=b.open,color=up?'#d34a4a':'#16866c',y1=py(b.high),y2=py(b.low),yo=py(b.open),yc=py(b.close),vh=b.volume/mv*VH;out.push(`<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" stroke="${color}"/><rect x="${x-cw/2}" y="${Math.min(yo,yc)}" width="${cw}" height="${Math.max(1,Math.abs(yc-yo))}" fill="${color}"/><rect x="${x-cw/2}" y="${base+VH-vh}" width="${cw}" height="${Math.max(1,vh)}" fill="${color}" opacity=".55"/>`)});for(let j=0;j<5;j++){let i=Math.round(j*(bars.length-1)/4),x=L+i*step;out.push(`<text x="${x}" y="${H-5}" text-anchor="middle" font-size="11" fill="#657487">${bars[i].date}</text>`)}svg.innerHTML=out.join('')}
function answer(){const out={};cases.forEach((c,i)=>{const choice=document.querySelector(`input[name="p${i}"]:checked`),state=document.querySelector(`[name="s${i}"]`).value;out[c.case_id]={code:c.code,name:c.name,asof:c.asof,pattern:choice?.value||'',state:choice?.value==='N'?'none':state,note:document.querySelector(`#case-${i} .note`).value}});return out}
function save(){localStorage.setItem(key,JSON.stringify(answer()));document.getElementById('status').textContent='已保存';return answer()}
document.getElementById('download').onclick=()=>{const answers=answer(),incomplete=cases.filter((c,i)=>!answers[c.case_id].pattern||(answers[c.case_id].pattern!=='N'&&!answers[c.case_id].state));if(incomplete.length){alert('还有 '+incomplete.length+' 题未完成，补齐形态和确认状态后再导出。');return}const blob=new Blob([JSON.stringify(save(),null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='caisen-blind-labels-'+__SEED__+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('save').onclick=save;document.getElementById('clear').onclick=()=>{localStorage.removeItem(key);location.reload()};
const saved=JSON.parse(localStorage.getItem(key)||'{}');cases.forEach((c,i)=>{root.insertAdjacentHTML('beforeend',card(c,i));draw(document.querySelector(`#case-${i} svg`),c.bars);const v=saved[c.case_id];if(v){const radio=document.querySelector(`input[name="p${i}"][value="${CSS.escape(v.pattern||v.choice||'')}"]`);if(radio)radio.checked=true;document.querySelector(`[name="s${i}"]`).value=v.state||'';document.querySelector(`#case-${i} .note`).value=v.note||''}});document.querySelectorAll('input,select,textarea').forEach(x=>x.addEventListener('change',()=>{const i=Number(x.name?.slice(1));if(x.name?.startsWith('p')&&x.value==='N')document.querySelector(`[name="s${i}"]`).value='none';else if(x.name?.startsWith('p')&&document.querySelector(`[name="s${i}"]`).value==='none')document.querySelector(`[name="s${i}"]`).value='';save()}));
</script></body></html>'''

# Inject the stable answer vocabulary once; keeps all labels out of machine output.
_HTML = _HTML.replace("__LABEL_CODES__", json.dumps(_LABELS, ensure_ascii=False))
