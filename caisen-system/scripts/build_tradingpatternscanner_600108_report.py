#!/usr/bin/env python3
"""Build a standalone, full-history TradingPatternScanner report for 600108."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).parents[1]
SOURCE_URL = "https://github.com/white07S/TradingPatternScanner"
DEFAULT_WINDOWS = [3, 20, 60, 120, 250]
LABELS = {
    "Double Bottom": "W底（Double Bottom）",
    "Inverse Head and Shoulder": "头肩底（Inverse Head and Shoulder）",
}


def load_upstream(root: Path):
    path = root / "tradingpatterns" / "tradingpatterns.py"
    spec = importlib.util.spec_from_file_location("tradingpatternscanner_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def analyze(csv_path: Path, scanner_root: Path, windows: list[int]):
    if not (scanner_root / ".git").exists():
        raise ValueError(f"TradingPatternScanner root is not a git checkout: {scanner_root}")
    commit = subprocess.run(
        ["git", "-C", str(scanner_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    upstream, _ = load_upstream(scanner_root)
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
    source_frame = raw.rename(columns={"high": "High", "low": "Low", "close": "Close"})
    for window in windows:
        if window < 2:
            raise ValueError("Native rolling window must be at least 2")
        db = upstream.detect_double_top_bottom(source_frame.copy(), window=window, threshold=0.05)
        hs = upstream.detect_head_shoulder(source_frame.copy(), window=window)
        for index, label in enumerate(db["double_pattern"].tolist()):
            if label != "Double Bottom" or index + 1 >= len(bars):
                continue
            previous, current, following = bars[index - 1], bars[index], bars[index + 1]
            previous_mid = (previous["high"] + previous["low"]) / 2
            next_mid = (following["high"] + following["low"]) / 2
            events.append({
                "date": current["date"], "known_date": following["date"],
                "index": index, "label": "Double Bottom", "window": window,
                "price": current["low"], "evidence": {
                    "previous_low": previous["low"], "label_low": current["low"],
                    "next_low": following["low"],
                    "previous_range_ratio": (previous["high"] - previous["low"]) / previous_mid if previous_mid else None,
                    "next_range_ratio": (following["high"] - following["low"]) / next_mid if next_mid else None,
                    "threshold": 0.05,
                },
            })
        for index, label in enumerate(hs["head_shoulder_pattern"].tolist()):
            if label != "Inverse Head and Shoulder" or index + 1 >= len(bars):
                continue
            start = max(0, index - window + 1)
            events.append({
                "date": bars[index]["date"], "known_date": bars[index + 1]["date"],
                "index": index, "label": "Inverse Head and Shoulder", "window": window,
                "price": bars[index]["low"], "evidence": {
                    "rolling_low_min": min(bar["low"] for bar in bars[start:index + 1]),
                    "previous_low": bars[index - 1]["low"], "label_low": bars[index]["low"],
                    "next_low": bars[index + 1]["low"],
                },
            })
    events.sort(key=lambda event: (event["date"], event["label"], event["window"]))
    metadata = {
        "code": "600108", "name": str(raw.iloc[0]["name"]), "rows": len(raw),
        "start": bars[0]["date"], "end": bars[-1]["date"], "csv": str(csv_path.resolve()),
        "ohlc_adjustment": "未验证；图表和事件价格按本地CSV原值展示。",
        "source_url": SOURCE_URL, "source_commit": commit,
        "license": "CC BY-NC-SA 4.0", "windows": windows,
        "default_window": 3, "double_bottom_threshold": 0.05,
        "functions": ["detect_double_top_bottom", "detect_head_shoulder"],
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "method_limit": "原生标签，不是完整形态几何识别。两个函数都使用下一根K线的shift(-1)，标签日之后的下一有效行情日才可知；上游不输出两底/左右肩锚点、颈线或突破确认。",
    }
    return bars, events, metadata


def esc_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_report(html_path: Path, csv_path: Path, scanner_root: Path, windows: list[int]):
    bars, events, metadata = analyze(csv_path, scanner_root, windows)
    vendor = REPO / "caisen" / "static" / "vendor" / "echarts.min.js"
    if not vendor.exists():
        raise FileNotFoundError(f"Bundled ECharts not found: {vendor}")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html(metadata, esc_json({"bars": bars, "events": events, "metadata": metadata}), vendor.read_text(encoding="utf-8")), encoding="utf-8")
    return metadata, events


def _html(metadata, payload, echarts):
    meta_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    template = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>600108 · TradingPatternScanner 全历史原生标签</title><style>
:root{color-scheme:dark;--bg:#101519;--panel:#171f24;--line:#2b3940;--text:#e5ecee;--muted:#9baab0;--green:#36b88b;--red:#e36b70;--gold:#e3b665;--blue:#69b8e5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}header{padding:20px 24px 12px;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0 0 5px}h2{font-size:16px;margin:0 0 8px}p{margin:4px 0}.muted{color:var(--muted)}.wrap{padding:14px 20px;max-width:1600px;margin:auto}.metrics{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.metric,.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px}.metric{padding:8px 13px;min-width:125px}.metric b{display:block;font-size:19px}.metric span{color:var(--muted);font-size:12px}.panel{padding:12px;margin-bottom:12px}.toolbar{display:flex;align-items:end;gap:10px;flex-wrap:wrap}.toolbar label{display:grid;gap:3px;color:var(--muted);font-size:12px}select,input,button{font:inherit;color:var(--text);background:#101619;border:1px solid #3a4a51;border-radius:4px;padding:6px 9px}button{cursor:pointer}button:hover{border-color:var(--gold)}.note{border-left:3px solid var(--gold);padding:8px 11px;color:#d4dce0;background:#20282b}.note strong{color:#f0c777}#chart{height:510px;width:100%}.list-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:500;font-size:12px}tbody tr{cursor:pointer}tbody tr:hover{background:#202a2f}.tag{border:1px solid #486067;border-radius:3px;padding:1px 5px;color:#cae0dc}.pager{display:flex;gap:8px;align-items:center}.provenance{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:12px}a{color:#83c9ba}.legend-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--gold)}.legend-diamond{display:inline-block;width:9px;height:9px;transform:rotate(45deg);margin:0 7px 0 5px;background:var(--blue)}@media(max-width:760px){.wrap{padding:10px}#chart{height:390px}header{padding:14px}}
</style></head><body><header><h1>600108 · TradingPatternScanner 原生识别</h1><p class="muted">本地完整日线历史 · W底和头肩底标签 · 可缩放、按标签/参数/日期筛选</p></header><main class="wrap">
<section class="metrics" id="metrics"></section>
<section class="panel"><div class="toolbar"><label>形态<select id="type"><option value="all">W底 + 头肩底</option><option value="Double Bottom">W底（Double Bottom）</option><option value="Inverse Head and Shoulder">头肩底（Inverse Head and Shoulder）</option></select></label><label>window<select id="window"></select></label><label>开始日期<input id="from" type="date"></label><label>结束日期<input id="to" type="date"></label><button id="reset">全历史</button><span id="filtered" class="muted"></span></div><p class="muted" style="margin-top:8px"><span class="legend-dot"></span>W底原生标签　<span class="legend-diamond"></span>头肩底原生标签</p></section>
<section class="panel note"><strong>先看标签含义：</strong>这是上游函数返回的原生标签，不等于完整的W底或头肩底形态。<code>window</code> 是滚动计算窗口，不是形态持续时间。代码没有输出两个底部、头部与双肩锚点，也没有颈线或突破条件；两种算法均引用下一根K线，标签在标签日后的下一有效交易日才可知。本页把标签点标在该日 Low，仅用于查看代码触发的位置。</section>
<section class="panel"><div id="chart" aria-label="600108全历史日线图及TradingPatternScanner标签"></div></section>
<section class="panel"><div class="list-head"><h2>原生标签明细</h2><div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div></div><div style="overflow:auto;margin-top:8px"><table><thead><tr><th>标签日期</th><th>上游标签</th><th>window</th><th>最早可知日</th><th>标签点 Low</th><th>邻近 Low / 代码条件旁证</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="panel"><h2>项目、参数与数据来源</h2><p>上游项目：<a href="__SOURCE_URL__">white07S/TradingPatternScanner</a> · commit <code>__SOURCE_COMMIT__</code> · 函数 <code>detect_double_top_bottom</code>、<code>detect_head_shoulder</code>。</p><p>W底标签使用上游默认 <code>window=3</code>、<code>threshold=0.05</code>；头肩底使用上游默认 <code>window=3</code>。其它窗口值是对原函数的参数敏感性展示，不代表不同周期的市场形态。</p><p><strong>与上一份 PatternPy 对照：</strong>这两个被调用函数的语法树逐项一致，因此同一行情和窗口的事件数一致；这次换了项目来源，没有得到一份独立算法的交叉验证。</p><p>数据：__CSV_PATH__ · __ROW_COUNT__ 根日K · __START_DATE__ 至 __END_DATE__。复权口径未核验，页面展示本地文件中的价格。运行环境 Python __PYTHON_VERSION__ / pandas __PANDAS_VERSION__ / NumPy __NUMPY_VERSION__。</p><p>原项目及其代码由 Preetam Sharma 发布，许可为 CC BY-NC-SA 4.0：<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">查看许可条款</a>。本页面链接并注明上游版本；没有把上游源代码复制进报告。</p><details><summary>运行元数据</summary><pre class="provenance">__METADATA__</pre></details></section>
</main><script>window.__caisenAmdDefine=window.define;if(window.define&&window.define.amd)window.define=undefined;</script><script>__ECHARTS__</script><script>if(window.__caisenAmdDefine)window.define=window.__caisenAmdDefine;delete window.__caisenAmdDefine;</script><script>
const DATA=__PAYLOAD__;
const {bars,events,metadata}=DATA, dates=bars.map(b=>b.date), typeEl=document.querySelector('#type'), windowEl=document.querySelector('#window'), fromEl=document.querySelector('#from'), toEl=document.querySelector('#to'), pageSize=50, dbName='W底（Double Bottom）', ihsName='头肩底（Inverse Head and Shoulder）';let pageNo=0;const chart=echarts.init(document.querySelector('#chart'));
for(const w of metadata.windows){const o=document.createElement('option');o.value=w;o.textContent=w===metadata.default_window?`${w}（上游默认）`:String(w);o.selected=w===metadata.default_window;windowEl.append(o)}
function filtered(){return events.filter(e=>(typeEl.value==='all'||e.label===typeEl.value)&&(windowEl.value==='all'||e.window===Number(windowEl.value))&&(!fromEl.value||e.date>=fromEl.value)&&(!toEl.value||e.date<=toEl.value))}
function renderMetrics(){const e=events.filter(x=>x.window===Number(windowEl.value)),db=e.filter(x=>x.label==='Double Bottom').length,ihs=e.filter(x=>x.label==='Inverse Head and Shoulder').length;document.querySelector('#metrics').innerHTML=`<div class="metric"><b>${metadata.rows.toLocaleString()}</b><span>完整日线记录</span></div><div class="metric"><b>${db.toLocaleString()}</b><span>W底原生标签 · window ${windowEl.value}</span></div><div class="metric"><b>${ihs.toLocaleString()}</b><span>头肩底原生标签 · window ${windowEl.value}</span></div><div class="metric"><b>${metadata.start} — ${metadata.end}</b><span>本地行情覆盖范围</span></div>`}
function renderChart(){const ev=filtered(),marks=(label,color,symbol)=>ev.filter(e=>e.label===label).map(e=>({value:[e.date,e.price],symbol,symbolSize:10,itemStyle:{color},label:{show:false}}));document.querySelector('#filtered').textContent=`当前 ${ev.length.toLocaleString()} 条原生标签`;chart.setOption({animation:false,backgroundColor:'transparent',legend:{top:4,textStyle:{color:'#cbd5d8'},data:[dbName,ihsName,'成交量']},tooltip:{trigger:'axis',axisPointer:{type:'cross'},formatter:p=>{const i=p[0]?.dataIndex;if(i==null)return '';const b=bars[i],hits=ev.filter(e=>e.index===i).map(e=>`${e.label==='Double Bottom'?dbName:ihsName} · window ${e.window}`).join('<br>');return `${b.date}<br>开 ${b.open}　高 ${b.high}<br>低 ${b.low}　收 ${b.close}<br>量 ${b.volume.toLocaleString()}${hits?'<br>'+hits:''}`}},grid:[{left:58,right:26,top:38,height:'61%'},{left:58,right:26,top:'75%',height:'16%'}],xAxis:[{type:'category',data:dates,scale:true,boundaryGap:false,axisLine:{lineStyle:{color:'#536168'}},axisLabel:{color:'#aab7bb'}},{type:'category',gridIndex:1,data:dates,axisLabel:{show:false}}],yAxis:[{scale:true,splitLine:{lineStyle:{color:'#29343a'}},axisLabel:{color:'#aab7bb'}},{gridIndex:1,scale:true,splitLine:{show:false},axisLabel:{show:false}}],dataZoom:[{type:'inside',xAxisIndex:[0,1],start:0,end:100},{type:'slider',xAxisIndex:[0,1],bottom:6,height:18,borderColor:'#35434a',textStyle:{color:'#aab7bb'}}],series:[{name:'600108',type:'candlestick',data:bars.map(b=>[b.open,b.close,b.low,b.high]),itemStyle:{color:'#e36b70',color0:'#36b88b',borderColor:'#e36b70',borderColor0:'#36b88b'},xAxisIndex:0,yAxisIndex:0},{name:dbName,type:'scatter',data:marks('Double Bottom','#e3b665','circle'),xAxisIndex:0,yAxisIndex:0,z:8,tooltip:{show:false}},{name:ihsName,type:'scatter',data:marks('Inverse Head and Shoulder','#69b8e5','diamond'),xAxisIndex:0,yAxisIndex:0,z:8,tooltip:{show:false}},{name:'成交量',type:'bar',data:bars.map(b=>b.volume),xAxisIndex:1,yAxisIndex:1,itemStyle:{color:'#56656a'}}]})}
function renderList(){const ev=filtered().slice().reverse(),pages=Math.max(1,Math.ceil(ev.length/pageSize));pageNo=Math.min(pageNo,pages-1);document.querySelector('#page').textContent=`${pageNo+1} / ${pages}`;document.querySelector('#rows').innerHTML=ev.slice(pageNo*pageSize,(pageNo+1)*pageSize).map(e=>{const v=e.evidence,detail=e.label==='Double Bottom'?`Low 前/标签/后 ${v.previous_low} / ${v.label_low} / ${v.next_low}；邻K振幅 ${(v.previous_range_ratio*100).toFixed(2)}% / ${(v.next_range_ratio*100).toFixed(2)}%`: `rolling min ${v.rolling_low_min}；Low 前/标签/后 ${v.previous_low} / ${v.label_low} / ${v.next_low}`;return `<tr data-index="${e.index}"><td>${e.date}</td><td><span class="tag">${e.label==='Double Bottom'?dbName:ihsName}</span></td><td>${e.window}</td><td>${e.known_date}</td><td>${e.price}</td><td>${detail}</td></tr>`}).join('');document.querySelectorAll('#rows tr').forEach(tr=>tr.onclick=()=>{const i=Number(tr.dataset.index),a=Math.max(0,i-40),b=Math.min(bars.length-1,i+40);chart.dispatchAction({type:'dataZoom',dataZoomIndex:0,startValue:dates[a],endValue:dates[b]});chart.dispatchAction({type:'showTip',seriesIndex:0,dataIndex:i})})}
function render(){renderMetrics();renderChart();renderList()}
document.querySelectorAll('#type,#window,#from,#to').forEach(el=>el.onchange=()=>{pageNo=0;render()});document.querySelector('#prev').onclick=()=>{pageNo=Math.max(0,pageNo-1);renderList()};document.querySelector('#next').onclick=()=>{pageNo++;renderList()};document.querySelector('#reset').onclick=()=>{typeEl.value='all';windowEl.value=String(metadata.default_window);fromEl.value='';toEl.value='';pageNo=0;chart.dispatchAction({type:'dataZoom',dataZoomIndex:0,start:0,end:100});render()};addEventListener('resize',()=>chart.resize());render();
</script></body></html>"""
    replacements = {
        "__SOURCE_URL__": html.escape(metadata["source_url"], quote=True),
        "__SOURCE_COMMIT__": metadata["source_commit"],
        "__CSV_PATH__": html.escape(metadata["csv"]),
        "__ROW_COUNT__": str(metadata["rows"]),
        "__START_DATE__": metadata["start"], "__END_DATE__": metadata["end"],
        "__PYTHON_VERSION__": metadata["runtime"]["python"],
        "__PANDAS_VERSION__": metadata["runtime"]["pandas"],
        "__NUMPY_VERSION__": metadata["runtime"]["numpy"],
        "__METADATA__": html.escape(meta_json), "__ECHARTS__": echarts,
        "__PAYLOAD__": payload,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scanner-root", type=Path, default=REPO / "state/external/TradingPatternScanner")
    parser.add_argument("--csv", type=Path, default=Path("data/stocks/600108.csv"))
    parser.add_argument("--output", type=Path, default=REPO / "reports/tradingpatternscanner-600108-full-history.html")
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    args = parser.parse_args()
    metadata, events = build_report(args.output, args.csv, args.scanner_root, args.windows)
    counts = {f"{label}@{window}": sum(e["label"] == label and e["window"] == window for e in events) for window in args.windows for label in LABELS}
    print(json.dumps({"html": str(args.output.resolve()), "rows": metadata["rows"], "start": metadata["start"], "end": metadata["end"], "event_count": len(events), "counts": counts, "commit": metadata["source_commit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
