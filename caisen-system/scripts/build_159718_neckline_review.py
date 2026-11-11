#!/usr/bin/env python3
"""Build a standalone, data-backed comparison of the 159718 neckline rules."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from caisen.fixed_backtest import simulate_fixed_trades
from caisen.main_bottoms import detect_main_bottoms


DATA = ROOT.parent / "data" / "etfs" / "159718.csv"
LOSS_REVIEW = ROOT / "reports" / "backtest-loss-pattern-review.html"
OUTPUT = ROOT / "reports" / "159718-neckline-review.html"
STRUCTURE_KNOWN_DATE = "2025-02-10"
ANALYSIS_DATE = "2025-05-29"
DATA_END = "2025-12-16"
OLD_ANCHOR_DATES = ["2023-04-04", "2024-04-19", "2025-06-23"]
CORRECTED_ANCHOR_DATES = ["2023-08-22", "2024-04-19", "2025-01-13"]


def _read_old_case():
    html = LOSS_REVIEW.read_text(encoding="utf-8")
    match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.S)
    if not match:
        raise RuntimeError("The saved loss review does not contain its case data")
    data = json.loads(match.group(1))
    return next(item for item in data["cases"] if item["code"] == "159718")


def _evidence(pattern):
    return {item["code"]: item["value"] for item in pattern.evidence}


def _run_trade(bars, signal_date, entry_date, pattern_id):
    result = simulate_fixed_trades(bars, [{
        "signal_date": signal_date,
        "entry_date": entry_date,
        "pattern_ids": [pattern_id],
        "pattern_types": ["inverse_head_shoulders"],
        "pattern_labels": ["头肩底"],
        "pattern_tags": [],
    }])
    if not result["trades"]:
        raise RuntimeError(f"Backtrader rejected the fixed case: {result['rejections']}")
    return result["trades"][0]


def build_report():
    all_bars = pd.read_csv(DATA, parse_dates=["date"])
    all_bars = all_bars[all_bars.date <= pd.Timestamp(DATA_END)].reset_index(drop=True)
    waiting_bars = all_bars[all_bars.date <= pd.Timestamp(STRUCTURE_KNOWN_DATE)].reset_index(drop=True)
    waiting_patterns = detect_main_bottoms(waiting_bars, code="159718", kind="etf", timeframe="1d",
                                           background_bars=waiting_bars)
    waiting = next((pattern for pattern in waiting_patterns
                    if pattern.pattern_type == "inverse_head_shoulders"
                    and [pattern.key_points.get(f"pivot_{number}_date") for number in (1, 2, 3)] == CORRECTED_ANCHOR_DATES
                    and pattern.confirmation_date is None), None)
    if waiting is None:
        raise RuntimeError("The corrected 159718 H&S anchors were not found before the buffered breakout")

    analysis_bars = all_bars[all_bars.date <= pd.Timestamp(ANALYSIS_DATE)].reset_index(drop=True)
    patterns = detect_main_bottoms(analysis_bars, code="159718", kind="etf", timeframe="1d",
                                   background_bars=analysis_bars)
    selected = next((pattern for pattern in patterns
                     if pattern.pattern_type == "inverse_head_shoulders"
                     and [pattern.key_points.get(f"pivot_{number}_date") for number in (1, 2, 3)] == CORRECTED_ANCHOR_DATES
                     and pattern.confirmation_date == ANALYSIS_DATE
                     and _evidence(pattern).get("main_selection", {}).get("selected") is True), None)
    if selected is None:
        raise RuntimeError("The corrected 159718 H&S did not confirm on its one-ATR neckline breakout")

    evidence = _evidence(selected)
    waiting_evidence = _evidence(waiting)
    selection = evidence["neckline_selection"]
    old_case = _read_old_case()
    old_anchor_dates = [item["date"] for item in old_case["pivots"]
                        if item.get("name") in {"P1", "P2", "P3"}]
    if old_anchor_dates != OLD_ANCHOR_DATES:
        raise RuntimeError("The saved old 159718 shoulder anchors changed; review before comparing")
    daily_dates = pd.DatetimeIndex(all_bars.date)
    new_entry_index = int(daily_dates.searchsorted(pd.Timestamp(ANALYSIS_DATE), side="right"))
    new_entry_date = str(daily_dates[new_entry_index].date())
    new_trade = _run_trade(all_bars, ANALYSIS_DATE, new_entry_date, selected.pattern_id)
    old_trade = {key: old_case[key] for key in (
        "signal_date", "entry_date", "entry_price", "exit_date", "exit_price",
        "exit_reason", "return_pct", "stop_price", "target_price", "holding_sessions")}

    # The recognizer fits line coefficients on its trailing five-year frame.
    local_start = pd.Timestamp(analysis_bars.date.iloc[-1]) - pd.DateOffset(years=5)
    indexed = all_bars[all_bars.date >= local_start].reset_index(drop=True)
    index_by_date = {str(row.date.date()): int(index) for index, row in indexed.iterrows()}
    tests = []
    for item in evidence["neckline_tests"]:
        if not item.get("valid"):
            continue
        tests.append({
            "date": item["pivot_date"], "known_date": item["known_date"],
            "level": item["level"], "departure_atr": item["departure_atr"],
            "index": index_by_date[item["pivot_date"]],
        })
    anchors = [{
        "label": {1: "左肩 P1", 2: "头部 P2", 3: "右肩 P3"}.get(number, f"结构点 P{number}"),
        "date": selected.key_points[f"pivot_{number}_date"],
        "price": selected.key_points[f"pivot_{number}"],
        "index": index_by_date[selected.key_points[f"pivot_{number}_date"]],
    } for number in (1, 2, 3)]
    old_anchors = [{
        "label": f"旧{item['name']}", "date": item["date"], "price": item["price"],
        "index": index_by_date[item["date"]],
    } for item in old_case["pivots"] if item.get("name") in {"P1", "P2", "P3"}]
    chart_start = "2023-01-01"
    chart_bars = []
    for row in all_bars[all_bars.date >= pd.Timestamp(chart_start)].itertuples(index=False):
        date = str(row.date.date())
        chart_bars.append({"date": date, "index": index_by_date[date], "open": float(row.open),
                           "high": float(row.high), "low": float(row.low), "close": float(row.close),
                           "volume": float(row.volume)})

    payload = {
        "code": "159718", "data_end": DATA_END, "analysis_date": ANALYSIS_DATE,
        "structure_known_date": STRUCTURE_KNOWN_DATE,
        "rule_version": selected.rule_version, "source_rule_version": selected.source_rule_version,
        "pattern_id": selected.pattern_id, "pattern_type": selected.pattern_type,
        "formed_at": selected.formed_at, "confirmed_at": selected.confirmation_date,
        "line": selected.neckline, "line_origin_date": str(indexed.date.iloc[0].date()),
        "line_origin_index": 0, "selection": selection, "anchors": anchors, "tests": tests,
        "old_anchors": old_anchors,
        "waiting": {"state": waiting.state, "breakout": waiting_evidence["neckline_breakout"],
                    "selection": waiting_evidence["neckline_selection"],
                    "neckline": waiting.neckline},
        "breakout": evidence["neckline_breakout"],
        "old": {"signal_date": old_case["signal_date"], "entry_date": old_case["entry_date"],
                "breakout_close": old_case["breakout_close"],
                "neckline_at_signal": old_case["neckline_at_signal"],
                "required_close": old_case["required_close"],
                "anchors": old_anchors, "trade": old_trade},
        "new": {"signal_date": ANALYSIS_DATE, "entry_date": new_entry_date,
                "breakout": evidence["neckline_breakout"], "trade": new_trade},
        "bars": chart_bars,
    }
    return _html(payload)


def _html(payload):
    json_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>159718 肩点校正与买入时机复核</title>
<style>
:root{{--ink:#17212c;--muted:#586776;--border:#d8e1e8;--paper:#f4f7f9;--purple:#6848bd;--amber:#dd8a13;--green:#07815f;--red:#c74949}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1420px;margin:0 auto;padding:26px 22px 48px}}h1{{margin:0 0 6px;font-size:27px}}h2{{margin:0 0 12px;font-size:19px}}p{{color:var(--muted);margin:5px 0 14px}}
.chips,.cards,.grid{{display:grid;gap:12px}}.chips{{grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin:18px 0}}
.chip,.card,.panel{{background:white;border:1px solid var(--border);border-radius:12px;padding:14px}}
.chip b,.chip span,.card b,.card small{{display:block}}.chip span,.card small{{color:var(--muted);font-size:13px}}.chip b{{font-size:17px;margin-top:3px}}
.cards{{grid-template-columns:repeat(auto-fit,minmax(260px,1fr));margin-bottom:14px}}.card{{border-top:4px solid var(--purple)}}.card.old{{border-top-color:var(--amber)}}.card.new{{border-top-color:var(--green)}}.card strong{{font-size:24px;display:block;margin:3px 0}}
.panel{{margin:14px 0}}.chartwrap{{overflow-x:auto}}svg{{width:100%;min-width:900px;height:auto;display:block}}.legend{{display:flex;flex-wrap:wrap;gap:15px;margin:8px 0 0;color:var(--muted);font-size:13px}}.key{{display:inline-block;width:15px;height:3px;vertical-align:middle;margin-right:5px;background:currentColor}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px 8px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}}th{{color:var(--muted);font-weight:600;background:#f8fafb}}.callout{{border-left:4px solid var(--purple);background:#f0edfa;padding:12px 14px;margin:12px 0}}.fine{{font-size:13px;color:var(--muted)}}.pass{{color:var(--green);font-weight:700}}.warn{{color:#9b5b00;font-weight:700}}
@media(max-width:700px){{main{{padding:18px 12px}}h1{{font-size:23px}}}}
</style></head><body><main>
<h1>159718 · 肩点校正后的头肩底买点</h1>
<p>本次保留“收盘价突破冻结颈线 + 前一日 ATR”的买入条件，只修正左右肩锚点。旧结构把左肩放得过早、右肩放在突破之后，导致确认偏晚。</p>
<div class="chips" id="chips"></div>
<section class="cards" id="trades"></section>
<section class="panel"><h2>完整结构与压力边界</h2><div class="chartwrap"><svg id="chart" viewBox="0 0 1360 630" role="img" aria-label="159718日K线、颈线和交易结果对比"></svg></div>
<div class="legend"><span style="color:#6848bd"><i class="key"></i>修正形态使用的冻结颈线</span><span style="color:#927fd0"><i class="key"></i>冻结线延伸</span><span style="color:#c74949">○ 旧肩点</span><span style="color:#17212c">● 修正肩点与头部</span><span style="color:#2f6db4">◆ 独立压力测试</span><span style="color:#07815f"><i class="key"></i>修正后 1ATR 突破日</span></div></section>
<section class="panel"><h2>肩部锚点、颈线与买入时点</h2><table id="compare"></table><div class="callout" id="diagnostics"></div><p class="fine">压力线仍由行情高点、当时已知 ATR 和独立测试组生成；买入触发仍是收盘突破冻结颈线加前一交易日 ATR。</p></section>
<section class="panel"><h2>判读口径与交易复核</h2><p>修正结构的左肩为 2023-08-22，头部为 2024-04-19，右肩为 2025-01-13。该结构于 2025-02-10 可知，之后等待冻结颈线加 1ATR 的收盘突破；2025-05-29 首次满足，下一交易日 2025-05-30 按开盘价入场。</p><p class="fine">新旧单笔结果按 Backtrader 的固定 8% 止损、20% 止盈、零费用规则复算。该案例用于检查锚点与信号日期，不代表整体策略收益结论。</p></section>
</main><script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('payload').textContent),fmt=x=>Number(x).toFixed(3),byDate=new Map(D.bars.map((b,i)=>[b.date,i]),);
const lineAt=index=>D.line.slope*index+D.line.intercept,selectedClose=Number(D.breakout.close),oldTrade=D.old.trade,newTrade=D.new.trade;
document.getElementById('chips').innerHTML=[['结构起点',D.formed_at],['结构可知日',D.structure_known_date],['修正左肩 / 头 / 右肩',D.anchors.map(a=>a.date).join(' · ')],['颈线测试',`${{D.tests.length}} 组独立测试`],['1ATR确认日',D.breakout.date],['规则版本',D.rule_version]].map(([k,v])=>`<div class="chip"><span>${{k}}</span><b>${{v}}</b></div>`).join('');
function tradeCard(title,cls,t){{return `<div class="card ${{cls}}"><small>${{title}} · Backtrader · 零费用</small><strong class="${{t.return_pct>=0?'pass':'warn'}}">${{(t.return_pct*100).toFixed(1)}}% · ${{t.exit_reason==='take_profit'?'止盈':'止损'}}</strong><div>信号 ${{t.signal_date}} → 入场 ${{t.entry_date}} @ ${{fmt(t.entry_price)}}</div><div>止损 ${{fmt(t.stop_price)}} · 止盈 ${{fmt(t.target_price)}}</div><div>退出 ${{t.exit_date}} @ ${{fmt(t.exit_price)}} · ${{t.holding_sessions}} 根交易K线</div></div>`}}
document.getElementById('trades').innerHTML=tradeCard('旧肩点识别：信号偏晚', 'old', oldTrade)+tradeCard('修正肩点：1ATR突破后入场','new',newTrade);
const selLevel=lineAt(D.breakout.index),tests=D.tests,fall=tests.length>1?(tests.at(-1).level/tests[0].level-1)*100:0;
document.getElementById('compare').innerHTML=`<thead><tr><th>证据</th><th>旧肩点识别</th><th>肩点校正后</th></tr></thead><tbody>
<tr><td>结构点</td><td>${{D.old.anchors.map(a=>`${{a.date}} @ ${{fmt(a.price)}}`).join('；')}}</td><td>${{D.anchors.map(a=>`${{a.date}} @ ${{fmt(a.price)}}`).join('；')}}</td></tr>
<tr><td>结构 / 颈线</td><td>右肩落在 2025-06-23，已接近突破后价格段</td><td>右肩于 2025-01-13 确认；结构于 ${{D.structure_known_date}} 可知，随后等待同一类价格触发</td></tr>
<tr><td>独立压力测试</td><td>旧线约 ${{fmt(D.old.neckline_at_signal)}} 附近</td><td>${{tests.map(t=>`${{t.date}} @ ${{fmt(t.level)}}`).join('、')}}；区间覆盖通过</td></tr>
<tr><td>1ATR价格确认</td><td>${{D.old.signal_date}} 收盘 ${{fmt(D.old.breakout_close)}}；旧规则要求 ≥ ${{fmt(D.old.required_close)}}</td><td>${{D.breakout.date}} 收盘 ${{fmt(selectedClose)}}；冻结线要求 ≥ ${{fmt(D.breakout.required_close)}}，通过</td></tr>
<tr><td>Backtrader 入场</td><td>${{oldTrade.entry_date}} @ ${{fmt(oldTrade.entry_price)}}</td><td>${{newTrade.entry_date}} @ ${{fmt(newTrade.entry_price)}}；突破后的下一交易日开盘</td></tr>
<tr><td>固定 8% / 20% 结果</td><td>${{oldTrade.exit_date}} @ ${{fmt(oldTrade.exit_price)}}，${{oldTrade.exit_reason==='take_profit'?'20%止盈':'8%止损'}}</td><td>${{newTrade.exit_date}} @ ${{fmt(newTrade.exit_price)}}，${{newTrade.exit_reason==='take_profit'?'20%止盈':'8%止损'}}</td></tr></tbody>`;
const reasonNames={{whole_interval_high_crossed:'区间高点穿越容差',slope_limit_exceeded:'颈线累计上升超限',insufficient_independent_tests:'独立测试不足',shoulder_geometry_not_supported_by_boundary:'未匹配头肩几何',missing_ihs_reaction_side_test:'缺少左/右反弹段测试',neckline_below_ihs_structure:'颈线低于头肩结构点',w_intervening_rim_not_covered:'未覆盖W中间反弹'}};
const rejected=(D.selection.rejected_candidates||[]).slice(0,4).map(r=>r.reason==='whole_interval_high_crossed'?`${{r.anchor_dates?.join(' / ')}} 压力组合被 ${{r.crossing_date}} 高点 ${{fmt(r.crossing_high)}} 穿过容差 ${{fmt(r.maximum_excess)}`:`${{reasonNames[r.reason]||r.reason}} · ${{r.anchor_dates?.join(' / ')||r.test_dates?.join(' / ')||'结构边界'}}`);
document.getElementById('diagnostics').innerHTML=`<b>候选淘汰统计（截至 ${{D.selection.diagnostics_as_of}}）</b>：${{Object.entries(D.selection.rejection_counts||{{}}).map(([k,v])=>`${{reasonNames[k]||k}} ${{v}} 组`).join('；')||'无淘汰记录'}}<br><b>代表性淘汰原因：</b>${{rejected.join('；')||'见候选排序表'}}<br><span class="fine">测试偏差越小越优；对照线还需满足完整肩部与 W 中间反弹几何，线本身不能单独命名形态。</span>`;

const svg=document.getElementById('chart'),NS='http://www.w3.org/2000/svg',W=1360,H=630,L=62,R=82,T=28,P=360,V=74,base=410,plotW=W-L-R;
const min=Math.min(...D.bars.map(b=>b.low),0.45),max=Math.max(...D.bars.map(b=>b.high),1.2),y=p=>T+(max-p)/(max-min)*P,x=i=>L+i/(D.bars.length-1)*plotW;
function add(tag,attrs,text){{const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{{}}))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.appendChild(n);return n}}
for(let i=0;i<=5;i++){{const price=min+(max-min)*i/5,yy=y(price);add('line',{{x1:L,y1:yy,x2:W-R,y2:yy,stroke:'#e0e7ec'}});add('text',{{x:W-R+8,y:yy+4,fill:'#566574','font-size':12}},fmt(price))}}
const candleW=Math.max(.75,plotW/D.bars.length*.68);
for(let i=0;i<D.bars.length;i++){{const b=D.bars[i],xx=x(i),up=b.close>=b.open,c=up?'#d64d4d':'#0d8a68';add('line',{{x1:xx,y1:y(b.high),x2:xx,y2:y(b.low),stroke:c,'stroke-width':.75,opacity:.8}});const top=y(Math.max(b.open,b.close)),height=Math.max(1,y(Math.min(b.open,b.close))-top);add('rect',{{x:xx-candleW/2,y:top,width:candleW,height,fill:c,opacity:.8}})}}
function mapIndex(date){{const i=byDate.get(date);return i===undefined?null:i}}
function boundary(from,to,color,width,dash,label){{const i0=mapIndex(from),i1=mapIndex(to);if(i0===null||i1===null)return;add('line',{{x1:x(i0),y1:y(lineAt(D.bars[i0].index)),x2:x(i1),y2:y(lineAt(D.bars[i1].index)),stroke:color,'stroke-width':width,'stroke-dasharray':dash||''}});if(label)add('text',{{x:x(i1)-8,y:y(lineAt(D.bars[i1].index))-8,fill:color,'font-size':12,'text-anchor':'end'}},label)}}
boundary(D.selection.coverage.start_date,D.selection.coverage.end_date,'#6848bd',3,'','v2 已验证区间');
boundary(D.selection.coverage.end_date,D.confirmed_at,'#927fd0',2,'7 5','冻结线延伸至突破日');
for(const test of tests){{const i=mapIndex(test.date);if(i===null)continue;const xx=x(i),yy=y(test.level);add('path',{{d:`M ${{xx}} ${{yy-6}} L ${{xx+6}} ${{yy}} L ${{xx}} ${{yy+6}} L ${{xx-6}} ${{yy}} Z`,fill:'#2f6db4',stroke:'white','stroke-width':1}});add('title',{{}},`${{test.date}} 独立测试 ${{fmt(test.level)}} · 知悉 ${{test.known_date}}`)}}
for(const a of D.old_anchors){{const i=mapIndex(a.date);if(i===null)continue;const xx=x(i),yy=y(a.price);add('circle',{{cx:xx,cy:yy,r:7,fill:'none',stroke:'#c74949','stroke-width':2}});add('text',{{x:xx,y:yy-10,fill:'#c74949','text-anchor':'middle','font-size':11}},`${{a.label}} ${{a.date.slice(5)}}`)}}
for(const a of D.anchors){{const i=mapIndex(a.date),xx=x(i),yy=y(a.price);add('circle',{{cx:xx,cy:yy,r:5,fill:'#17212c',stroke:'white','stroke-width':2}});add('text',{{x:xx,y:yy+18,fill:'#17212c','text-anchor':'middle','font-size':12}},`${{a.label}} ${{a.date.slice(5)}}`)}}
function tradeLines(t,color,label){{const i0=mapIndex(t.entry_date),i1=mapIndex(t.exit_date);if(i0===null||i1===null)return;for(const [price,stroke,name] of [[t.stop_price,color,label+'止损'],[t.target_price,'#07815f',label+'止盈']]){{const yy=y(price);add('line',{{x1:x(i0),y1:yy,x2:x(i1),y2:yy,stroke,'stroke-width':1.3,'stroke-dasharray':'4 4',opacity:.8}})}}const e=D.bars[i0];add('circle',{{cx:x(i0),cy:y(e.open),r:5,fill:color,stroke:'white','stroke-width':1.5}});const ex=D.bars[i1];add('circle',{{cx:x(i1),cy:y(t.exit_price),r:6,fill:t.exit_reason==='take_profit'?'#07815f':'#c74949',stroke:'white','stroke-width':1.5}});add('title',{{}},`${{label}} ${{t.entry_date}}入场 ${{fmt(t.entry_price)}}，${{t.exit_date}}退出 ${{fmt(t.exit_price)}}`)}}
tradeLines(oldTrade,'#c74949','旧肩点');tradeLines(newTrade,'#07815f','修正后');
const knownIndex=mapIndex(D.structure_known_date);if(knownIndex!==null){{const xx=x(knownIndex);add('line',{{x1:xx,y1:T,x2:xx,y2:base,stroke:'#dd8a13','stroke-dasharray':'6 5',opacity:.55}});add('text',{{x:xx-5,y:T+24,fill:'#a96400','font-size':11,'text-anchor':'end'}},`结构可知 ${{D.structure_known_date}}` )}}
const breakoutIndex=mapIndex(D.breakout.date);if(breakoutIndex!==null){{const xx=x(breakoutIndex);add('line',{{x1:xx,y1:T,x2:xx,y2:base,stroke:'#3c6f89','stroke-dasharray':'3 5',opacity:.7}});add('text',{{x:xx+5,y:T+12,fill:'#3c6f89','font-size':11}},`确认 ${{D.breakout.date}}`)}}
for(const date of ['2023-01-01','2023-07-01','2024-01-01','2024-07-01','2025-01-01','2025-07-01','2025-12-01']){{const entry=D.bars.find(b=>b.date>=date);if(!entry)continue;const i=mapIndex(entry.date);add('text',{{x:x(i),y:base+20,fill:'#566574','font-size':11,'text-anchor':'middle'}},entry.date.slice(0,7))}}
add('text',{{x:L,y:base+42,fill:'#788692','font-size':12}},'成交量');const vols=D.bars.map(b=>b.volume),vmax=Math.max(...vols);for(let i=0;i<D.bars.length;i+=2){{const b=D.bars[i],h=b.volume/vmax*V,xx=x(i);add('rect',{{x:xx,y:base+58+V-h,width:Math.max(1,candleW*1.5),height:h,fill:b.close>=b.open?'#d64d4d':'#0d8a68',opacity:.5}})}}
</script></body></html>""".replace("{{", "{").replace("}}", "}").replace("__PAYLOAD__", json_payload)


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_report(), encoding="utf-8")
    print(OUTPUT)
