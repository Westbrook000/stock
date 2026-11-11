#!/usr/bin/env python3
"""Create an offline report of native PatternPy labels on local 600108 OHLC data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import pandas as pd
import numpy as np


PINNED_COMMIT = "2b1f29b17986acafffeb32c83e772417d540c100"
SOURCE_URL = "https://github.com/keithorange/PatternPy"
DEFAULT_WINDOWS = [3, 20, 60, 120, 250]
LABELS = {
    "double_pattern": "Double Bottom",
    "head_shoulder_pattern": "Inverse Head and Shoulder",
}


def load_upstream(root: Path):
    path = root / "tradingpatterns" / "tradingpatterns.py"
    spec = importlib.util.spec_from_file_location("patternpy_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(csv_path: Path, patternpy_root: Path, windows: list[int]):
    source_path = patternpy_root / "tradingpatterns" / "tradingpatterns.py"
    if not (patternpy_root / ".git").exists():
        raise ValueError(f"PatternPy root is not a git checkout: {patternpy_root}")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(patternpy_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"Expected PatternPy {PINNED_COMMIT}, found {commit}")
    subprocess.run(
        ["git", "-C", str(patternpy_root), "diff", "--quiet", commit, "--", "tradingpatterns/tradingpatterns.py", "LICENSE.md"],
        check=True,
    )
    upstream = load_upstream(patternpy_root)
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"date", "code", "name", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.strftime("%Y-%m-%d")
    raw = raw.sort_values("date", kind="stable").reset_index(drop=True)
    if raw["date"].duplicated().any():
        raise ValueError("Duplicate dates in local history; refusing to silently deduplicate")
    if raw.empty:
        raise ValueError("Local history is empty")
    bars = [
        {"date": row.date, "open": float(row.open), "high": float(row.high),
         "low": float(row.low), "close": float(row.close), "volume": float(row.volume)}
        for row in raw.itertuples(index=False)
    ]
    events = []
    for window in windows:
        if window < 2:
            raise ValueError("Native rolling window must be at least 2")
        native = raw.rename(columns={"high": "High", "low": "Low", "close": "Close"})
        native = upstream.detect_double_top_bottom(native.copy(), window=window, threshold=0.05)
        native = upstream.detect_head_shoulder(native, window=window)
        for col, label in LABELS.items():
            rows = native.index[native[col] == label].tolist()
            for index in rows:
                if index + 1 >= len(raw):
                    continue
                event = {
                    "date": bars[index]["date"], "index": index, "label": label,
                    "window": window, "known_date": bars[index + 1]["date"],
                    "price": bars[index]["low"],
                }
                if label == "Double Bottom":
                    def candle_range(bar):
                        avg = (bar["high"] + bar["low"]) / 2
                        return {"high": bar["high"], "low": bar["low"], "range_ratio": (bar["high"] - bar["low"]) / avg if avg else None}
                    event["evidence"] = {
                        "previous_bar_range": candle_range(bars[index - 1]),
                        "next_bar_range": candle_range(bars[index + 1]),
                        "threshold": 0.05,
                    }
                else:
                    start = max(0, index - window + 1)
                    event["evidence"] = {
                        "rolling_low_min": min(bar["low"] for bar in bars[start:index + 1]),
                        "previous_bar_low": bars[index - 1]["low"],
                        "label_bar_low": bars[index]["low"],
                        "next_bar_low": bars[index + 1]["low"],
                    }
                events.append(event)
    events.sort(key=lambda event: (event["date"], event["label"], event["window"]))
    license_text = (patternpy_root / "LICENSE.md").read_text(encoding="utf-8")
    metadata = {
        "code": "600108", "name": str(raw.iloc[0]["name"]), "rows": len(raw),
        "start": bars[0]["date"], "end": bars[-1]["date"],
        "csv": str(csv_path.resolve()),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "ohlc_adjustment": "Not verified; values shown exactly as stored in local CSV.",
        "patternpy_source": SOURCE_URL, "patternpy_commit": commit,
        "patternpy_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "license": "CC BY-NC-SA 4.0", "license_text": license_text,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "windows": windows, "double_bottom_threshold": 0.05,
        "functions": ["detect_double_top_bottom", "detect_head_shoulder"],
        "method_limit": "Native labels only. Both routines use shift(-1); event date is retrospective and earliest knowable date is the next available bar. No complete pattern geometry or neckline is returned.",
    }
    return bars, events, metadata


def build_report(html_path: Path, json_path: Path, csv_path: Path, patternpy_root: Path, windows: list[int]):
    bars, events, metadata = analyze(csv_path, patternpy_root, windows)
    payload = {"bars": bars, "events": events, "metadata": metadata}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    vendor = Path(__file__).parents[1] / "caisen" / "static" / "vendor" / "echarts.min.js"
    echarts = vendor.read_text(encoding="utf-8")
    safe_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_path.write_text(_html(metadata, safe_payload, echarts), encoding="utf-8")
    return metadata, len(events)


def _html(metadata, payload, echarts):
    meta_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>600108 PatternPy 原生识别</title><style>
:root{color-scheme:dark;--bg:#101519;--panel:#171f24;--line:#2b3940;--text:#e5ecee;--muted:#9baab0;--green:#36b88b;--red:#e36b70;--gold:#e3b665}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}header{padding:20px 24px 12px;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0 0 5px}p{margin:4px 0}.muted{color:var(--muted)}.wrap{padding:14px 20px;max-width:1600px;margin:auto}.metrics{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.metric,.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px}.metric{padding:8px 13px;min-width:125px}.metric b{display:block;font-size:19px}.metric span{color:var(--muted);font-size:12px}.panel{padding:12px;margin-bottom:12px}.toolbar{display:flex;align-items:end;gap:10px;flex-wrap:wrap}.toolbar label{display:grid;gap:3px;color:var(--muted);font-size:12px}select,input,button{font:inherit;color:var(--text);background:#101619;border:1px solid #3a4a51;border-radius:4px;padding:6px 9px}button{cursor:pointer}button:hover{border-color:var(--gold)}.note{border-left:3px solid var(--gold);padding:8px 11px;color:#d4dce0;background:#20282b}.note strong{color:#f0c777}#chart{height:510px;width:100%}.list-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.list-head h2{font-size:16px;margin:0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500;font-size:12px}tbody tr{cursor:pointer}tbody tr:hover{background:#202a2f}.tag{border:1px solid #486067;border-radius:3px;padding:1px 5px;color:#cae0dc}.pager{display:flex;gap:8px;align-items:center}.provenance{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:12px}a{color:#83c9ba}@media(max-width:760px){.wrap{padding:10px}#chart{height:390px}header{padding:14px}.hide-small{display:none}}
</style></head><body><header><h1>600108 · PatternPy 原生识别</h1><p class="muted">本地完整日线历史 · 仅呈现上游函数返回的原生标签</p></header><main class="wrap">
<section class="metrics" id="metrics"></section>
<section class="panel"><div class="toolbar"><label>形态<select id="type"><option value="all">全部</option><option value="Double Bottom">W底（Double Bottom）</option><option value="Inverse Head and Shoulder">头肩底（Inverse Head and Shoulder）</option></select></label><label>window<select id="window"><option value="all">全部 window</option></select></label><label>起始日期<input id="from" type="date"></label><label>结束日期<input id="to" type="date"></label><button id="reset">全历史</button><button id="recent">查看 2023 至今</button><span class="muted" id="filtered"></span></div></section>
<section class="panel note"><strong>识别边界：</strong>PatternPy 这两个原生函数使用 <code>shift(-1)</code>，所以标签依赖下一根日线；表中的“最早可知日”是标签日期后的下一交易日。标签不是完整 W 底或头肩底；上游没有返回结构锚点或颈线。本页不推导买点。Double Bottom 的 <code>threshold=0.05</code> 限制的是标签日前后两根 K 线各自的 high-low 振幅/该根 high-low 均价；它不约束两底价差。<code>window</code> 是 rolling 高低值回看长度，不是完整形态的最短时长；IHS 没有 threshold。</section>
<section class="panel"><div id="chart" aria-label="600108 日线蜡烛图及 PatternPy 原生标签"></div></section>
<section class="panel"><div class="list-head"><h2>原生标签明细</h2><div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><button id="export">导出筛选结果 CSV</button></div></div><div style="overflow:auto;margin-top:8px"><table><thead><tr><th>标签日期</th><th>上游标签</th><th>window</th><th>最早可知日</th><th>标签点 Low</th><th>原始条件旁证</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="panel"><h2>来源与参数</h2><p>原始项目 <a href="https://github.com/keithorange/PatternPy">keithorange/PatternPy</a>，commit <code>''' + metadata["patternpy_commit"] + r'''</code>。直接加载其 <code>tradingpatterns.py</code>；未改写算法。默认参数 window=3、Double Bottom threshold=0.05；其它 window 是独立原生重跑，不代表最小持续时间，且可能不改变结果。</p><p>本地 CSV：''' + metadata["csv"].replace("&", "&amp;").replace("<", "&lt;") + r''' · ''' + metadata["rows"] .__str__() + r''' 行 · ''' + metadata["start"] + " 至 " + metadata["end"] + r'''。价格复权状态未验证，展示文件内原值。</p><p>运行环境 Python ''' + metadata["runtime"]["python"] + " / pandas " + metadata["runtime"]["pandas"] + " / numpy " + metadata["runtime"]["numpy"] + r'''。上游声明 Python ^3.10、pandas ^2.0、numpy ^1.24；此页报告实际运行环境，不代表其官方支持该环境。</p><p>许可证原文随报告提供，项目声明 CC BY-NC-SA 4.0：</p><pre class="provenance">''' + metadata["license_text"].replace("&", "&amp;").replace("<", "&lt;") + r'''</pre><details><summary>机器可读运行元数据</summary><pre class="provenance">''' + meta_json.replace("&", "&amp;").replace("<", "&lt;") + r'''</pre></details></section>
</main><script>window.__caisenAmdDefine=window.define;if(window.define&&window.define.amd)window.define=undefined;</script><script>''' + echarts + r'''</script><script>if(window.__caisenAmdDefine)window.define=window.__caisenAmdDefine;delete window.__caisenAmdDefine;</script><script>
const DATA=''' + payload + r''';
const {bars,events,metadata}=DATA, dates=bars.map(x=>x.date), byType=document.querySelector('#type'), byWindow=document.querySelector('#window'), pageSize=50, DB_LABEL='W底（Double Bottom）', IHS_LABEL='头肩底（Inverse Head and Shoulder）', recentStart=dates.find(d=>d>='2023-01-01');let pageNo=0,chart=echarts.init(document.querySelector('#chart'));
for(const w of metadata.windows){const o=document.createElement('option');o.value=w;o.textContent=w===3?'3（上游默认）':w;o.selected=w===3;byWindow.append(o)}
function countsForWindow(){const scoped=events.filter(e=>byWindow.value==='all'||e.window===Number(byWindow.value));return {'Double Bottom':scoped.filter(e=>e.label==='Double Bottom').length,'Inverse Head and Shoulder':scoped.filter(e=>e.label==='Inverse Head and Shoulder').length}}
function renderMetrics(){const counts=countsForWindow(),scope=byWindow.value==='all'?'所有 window':`window ${byWindow.value}`;document.querySelector('#metrics').innerHTML=`<div class="metric"><b>${metadata.rows.toLocaleString()}</b><span>日线记录</span></div><div class="metric"><b>${counts['Double Bottom']}</b><span>${DB_LABEL} 原生标签 · ${scope}</span></div><div class="metric"><b>${counts['Inverse Head and Shoulder']}</b><span>${IHS_LABEL} 原生标签 · ${scope}</span></div><div class="metric"><b>${events.length}</b><span>总原始标签行（跨 window 计数，不去重）</span></div><div class="metric"><b>${metadata.start} — ${metadata.end}</b><span>完整本地历史</span></div>`}renderMetrics();
function filtered(){return events.filter(e=>(byType.value==='all'||e.label===byType.value)&&(byWindow.value==='all'||e.window===Number(byWindow.value))&&(!from.value||e.date>=from.value)&&(!to.value||e.date<=to.value))}
const from=document.querySelector('#from'),to=document.querySelector('#to');function option(){const ev=filtered(), marks=(label,color,symbol)=>ev.filter(e=>e.label===label).map(e=>({value:[e.date,e.price],event:e,symbol,symbolSize:11,itemStyle:{color},label:{show:false}}));document.querySelector('#filtered').textContent=`${ev.length.toLocaleString()} 条筛选标签`;chart.setOption({animation:false,backgroundColor:'transparent',legend:{top:4,textStyle:{color:'#cbd5d8'},data:[DB_LABEL,IHS_LABEL,'成交量']},tooltip:{trigger:'axis',axisPointer:{type:'cross'},formatter:p=>{const i=p[0]?.dataIndex;if(i==null)return '';const b=bars[i],hits=ev.filter(e=>e.index===i).map(e=>`${e.label} · window ${e.window}`).join('<br>');return `${b.date}<br>开 ${b.open}　高 ${b.high}<br>低 ${b.low}　收 ${b.close}<br>量 ${b.volume.toLocaleString()}${hits?'<br>'+hits:''}`}},grid:[{left:58,right:26,top:38,height:'61%'},{left:58,right:26,top:'75%',height:'16%'}],xAxis:[{type:'category',data:dates,scale:true,boundaryGap:false,axisLine:{lineStyle:{color:'#536168'}},axisLabel:{color:'#aab7bb'}},{type:'category',gridIndex:1,data:dates,axisLabel:{show:false}}],yAxis:[{scale:true,splitLine:{lineStyle:{color:'#29343a'}},axisLabel:{color:'#aab7bb'}},{gridIndex:1,scale:true,splitLine:{show:false},axisLabel:{show:false}}],dataZoom:[{type:'inside',xAxisIndex:[0,1],start:0,end:100},{type:'slider',xAxisIndex:[0,1],bottom:6,height:18,borderColor:'#35434a',textStyle:{color:'#aab7bb'}}],series:[{name:'600108',type:'candlestick',data:bars.map(b=>[b.open,b.close,b.low,b.high]),itemStyle:{color:'#e36b70',color0:'#36b88b',borderColor:'#e36b70',borderColor0:'#36b88b'},xAxisIndex:0,yAxisIndex:0},{name:DB_LABEL,type:'scatter',data:marks('Double Bottom','#e3b665','circle'),xAxisIndex:0,yAxisIndex:0,z:8,tooltip:{show:false}},{name:IHS_LABEL,type:'scatter',data:marks('Inverse Head and Shoulder','#69b8e5','diamond'),xAxisIndex:0,yAxisIndex:0,z:8,tooltip:{show:false}},{name:'成交量',type:'bar',data:bars.map(b=>b.volume),xAxisIndex:1,yAxisIndex:1,itemStyle:{color:'#56656a'}}]})}
function renderList(){const ev=filtered().slice().reverse(),pages=Math.max(1,Math.ceil(ev.length/pageSize));pageNo=Math.min(pageNo,pages-1);document.querySelector('#page').textContent=`${pageNo+1} / ${pages}`;document.querySelector('#rows').innerHTML=ev.slice(pageNo*pageSize,(pageNo+1)*pageSize).map(e=>`<tr data-index="${e.index}"><td>${e.date}</td><td><span class="tag">${e.label==='Double Bottom'?DB_LABEL:IHS_LABEL}</span></td><td>${e.window}</td><td>${e.known_date}</td><td>${e.price}</td><td>${e.label==='Double Bottom'?`前/后 bar range ratio: ${e.evidence.previous_bar_range.range_ratio?.toFixed(4)} / ${e.evidence.next_bar_range.range_ratio?.toFixed(4)}；threshold=${e.evidence.threshold}`:`rolling low min=${e.evidence.rolling_low_min}；前/标签/后 Low: ${e.evidence.previous_bar_low} / ${e.evidence.label_bar_low} / ${e.evidence.next_bar_low}`}</td></tr>`).join('');document.querySelectorAll('#rows tr').forEach(tr=>tr.onclick=()=>{const i=Number(tr.dataset.index),a=Math.max(0,i-40),b=Math.min(bars.length-1,i+40);chart.dispatchAction({type:'dataZoom',dataZoomIndex:0,startValue:dates[a],endValue:dates[b]});chart.dispatchAction({type:'showTip',seriesIndex:0,dataIndex:i})})}
function render(){renderList();option()}
document.querySelectorAll('#type,#window,#from,#to').forEach(el=>el.onchange=()=>{pageNo=0;if(el===byWindow)renderMetrics();render()});document.querySelector('#prev').onclick=()=>{pageNo=Math.max(0,pageNo-1);renderList()};document.querySelector('#next').onclick=()=>{pageNo++;renderList()};document.querySelector('#reset').onclick=()=>{from.value='';to.value='';chart.dispatchAction({type:'dataZoom',dataZoomIndex:0,start:0,end:100});render()};document.querySelector('#recent').onclick=()=>{if(!recentStart)return;from.value=recentStart;to.value='';render();chart.dispatchAction({type:'dataZoom',dataZoomIndex:0,startValue:recentStart,endValue:dates.at(-1)})};document.querySelector('#export').onclick=()=>{const ev=filtered(),rows=[['date','label','window','known_date','low'],...ev.map(e=>[e.date,e.label,e.window,e.known_date,e.price])],csv=rows.map(r=>r.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\r\n'),a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv'}));a.download='600108-patternpy-events.csv';a.click();URL.revokeObjectURL(a.href)};addEventListener('resize',()=>chart.resize());render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patternpy-root", type=Path, default=Path(__file__).parents[1] / "state/external/PatternPy-2b1f29b")
    parser.add_argument("--csv", type=Path, default=Path("data/stocks/600108.csv"))
    parser.add_argument("--output", type=Path, default=Path("caisen-system/reports/patternpy-600108-full-history.html"))
    parser.add_argument("--json", type=Path, default=Path("caisen-system/reports/patternpy-600108-full-history.json"))
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    args = parser.parse_args()
    metadata, count = build_report(args.output, args.json, args.csv, args.patternpy_root, args.windows)
    print(json.dumps({"html": str(args.output), "json": str(args.json), "rows": metadata["rows"], "start": metadata["start"], "end": metadata["end"], "events": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
