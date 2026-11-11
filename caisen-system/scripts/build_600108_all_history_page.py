#!/usr/bin/env python3
"""Build a self-contained interactive 600108 full-history pattern page."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "reports/daily-research/cases/600108-all-history-v2-20261102.json"
OUTPUT = ROOT / "reports/daily-research/cases/600108-all-history-v2-20261102.html"

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>600108 全历史形态识别</title>
<style>
:root{--ink:#17263b;--muted:#64748b;--line:#dce4ee;--blue:#2861d8;--violet:#7455d8;--teal:#12877f;--red:#d24b45;--green:#17845f;--paper:#f3f6fa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:22px}.hero{background:linear-gradient(125deg,#102c4f,#1c4e83 68%,#316ca2);border-radius:20px;color:#fff;padding:25px 30px;box-shadow:0 14px 38px #193b671b}
.hero-top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.eyebrow{font-size:11px;letter-spacing:.13em;font-weight:800;opacity:.75}.hero h1{font-size:29px;line-height:1.2;margin:7px 0 8px}.hero p{color:#dbe8f7;margin:0;max-width:800px}.stamp{background:#ffffff1c;border:1px solid #ffffff38;border-radius:12px;padding:9px 13px;font-size:12px;white-space:nowrap}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:22px}.stat{border:1px solid #ffffff30;background:#ffffff12;border-radius:12px;padding:11px 13px}.stat strong{display:block;font-size:21px;line-height:1.25}.stat span{font-size:11px;color:#d5e5f5}
.notice{margin-top:14px;padding:10px 13px;border-radius:10px;background:#fff4d8;color:#654b17;border:1px solid #f3dfa9;font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:16px;margin-top:16px;align-items:start}.panel{background:#fff;border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 5px 18px #23415f07}.panel h2{font-size:16px;margin:0 0 9px}.panel h3{font-size:14px;margin:0 0 7px}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:#68788c;font-size:11px;margin:8px 0}.key{display:inline-flex;align-items:center;gap:5px}.dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--blue)}.dot.w{background:#df664c}.dot.hs{background:#2878c8}.dot.tri{background:#168b78}.dot.flag{background:#815ad1}
.canvasbox{width:100%;position:relative}.overviewbox{height:96px}.chartbox{height:410px}.canvasbox canvas{width:100%;height:100%;display:block;touch-action:pan-y}
.chart-tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0 2px}.range{display:flex;align-items:center;gap:8px;flex:1;min-width:230px}.range input{width:100%}.range label{font-size:11px;color:var(--muted);white-space:nowrap}.btn{background:white;border:1px solid var(--line);border-radius:8px;color:#39516e;padding:7px 10px;cursor:pointer;font-size:12px}.btn:hover{border-color:#95afd9;color:#1e55ad}.hint{font-size:11px;color:var(--muted)}
.list-head{display:flex;justify-content:space-between;align-items:center;margin:16px 0 9px}.list-head strong{font-size:15px}.filterbar{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:8px;margin-bottom:8px}.filterbar input{min-width:0;width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;padding:8px;color:#33465f;font:inherit;font-size:12px}.typechips{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 9px}.typechips label{border:1px solid var(--line);border-radius:999px;background:#fbfcfe;padding:5px 9px;font-size:11px;color:#536780;display:flex;align-items:center;gap:5px}.typechips input{accent-color:#5b55c7}.eventlist{border:1px solid var(--line);border-radius:10px;overflow:auto;max-height:440px}.events{border-collapse:collapse;width:100%;font-size:12px}.events thead{position:sticky;top:0;background:#f5f8fc;z-index:1;color:#607188}.events th,.events td{text-align:left;padding:8px 9px;border-bottom:1px solid #edf1f5;white-space:nowrap}.events th{font-size:11px;font-weight:700}.events tbody tr{cursor:pointer}.events tbody tr:hover{background:#f5f8ff}.events tbody tr.active{background:#eaf1ff}.shape{display:inline-block;border-radius:999px;padding:2px 7px;margin:1px 2px 1px 0;background:#edf3fb;color:#345171;font-size:10px}.shape.w{background:#fff0ed;color:#a64835}.shape.hs{background:#eaf3ff;color:#2762a4}.shape.tri{background:#e8f7f3;color:#147363}.shape.flag{background:#f3edff;color:#6245a8}
.detail{position:sticky;top:14px}.detail-top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.detail-date{font-size:19px;font-weight:800}.badge{border-radius:999px;background:#edf3ff;color:#315caa;padding:4px 8px;font-size:10px;white-space:nowrap}.field-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:13px 0}.field{background:#f6f8fb;border-radius:9px;padding:9px}.field span{display:block;color:#718198;font-size:10px}.field strong{display:block;margin-top:3px;font-size:13px;overflow-wrap:anywhere}.points{list-style:none;margin:8px 0;padding:0}.points li{padding:6px 0;border-bottom:1px solid #edf1f5;font-size:12px}.points b{color:#405875}.mini-note{font-size:11px;color:#728196}.neck{border-left:3px solid var(--violet);background:#f5f2ff;border-radius:0 8px 8px 0;padding:9px 10px;font-size:12px;margin:10px 0}.tagselect{display:flex;gap:5px;flex-wrap:wrap;margin:8px 0}.tagselect button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 7px;font-size:10px;color:#51647d;cursor:pointer}.tagselect button.selected{background:#ece8ff;border-color:#b3a4ec;color:#5741a8}
details{margin-top:10px;border-top:1px solid var(--line);padding-top:9px}details summary{cursor:pointer;font-size:12px;font-weight:700;color:#49617d}pre{font:10px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto;background:#f6f8fb;padding:10px;border-radius:8px;color:#40516a}
.footer{color:#7b899b;font-size:11px;padding:16px 3px 6px}.no-data{padding:22px;text-align:center;color:#718198}
@media(max-width:1050px){.layout{grid-template-columns:1fr}.detail{position:static}.stats{grid-template-columns:repeat(3,1fr)}}
@media(max-width:680px){.wrap{padding:10px}.hero{padding:19px}.hero h1{font-size:24px}.hero-top{display:block}.stamp{display:inline-block;margin-top:12px}.stats{grid-template-columns:repeat(2,1fr)}.filterbar{grid-template-columns:1fr 1fr}.chartbox{height:330px}.panel{padding:12px}.events th:nth-child(3),.events td:nth-child(3){display:none}}
</style>
</head>
<body>
<main class="wrap">
  <header class="hero">
    <div class="hero-top"><div><div class="eyebrow">FULL HISTORY · CAUSAL DAILY PATTERN EVENTS</div><h1>600108 · 全历史形态结果</h1><p>1997-08-18 至 2026-01-09。点击历史事件，可定位到对应K线并查看结构锚点、冻结颈线和确认/介入时间。</p></div><div class="stamp">600108 · A股日线<br>本地行情 · 当前引擎重跑</div></div>
    <div class="stats" id="stats"></div>
  </header>
  <div class="notice" id="provenance"></div>
  <div class="layout">
    <div>
      <section class="panel">
        <h2>全历史概览</h2><div class="canvasbox overviewbox"><canvas id="overview"></canvas></div>
        <div class="hint">全历史收盘价与所有确认事件。点击概览可移动下方窗口；下方滑块也可平移。</div>
      </section>
      <section class="panel" style="margin-top:13px">
        <div class="detail-top"><h2>形态K线图</h2><span class="badge" id="chartStatus"></span></div>
        <div class="legend"><span class="key"><i class="dot w"></i>W底</span><span class="key"><i class="dot hs"></i>头肩底</span><span class="key"><i class="dot tri"></i>收敛三角形底部</span><span class="key"><i class="dot flag"></i>上涨中继</span><span class="key">紫线：选中形态颈线</span></div>
        <div class="canvasbox chartbox"><canvas id="chart"></canvas></div>
        <div class="chart-tools">
          <div class="range"><label for="pan">平移</label><input id="pan" type="range" min="0" value="0"><label id="panLabel"></label></div>
          <div class="range" style="max-width:300px"><label for="zoom">显示K线</label><input id="zoom" type="range" min="120" max="2600" step="20" value="360"><label id="zoomLabel"></label></div>
          <button class="btn" id="focusEvent">定位所选形态</button><button class="btn" id="showAll">看全历史</button>
        </div>
        <div class="hint">鼠标滚轮缩放；拖动图表平移；也可点击任一历史事件跳转定位。</div>
      </section>
      <section class="panel" style="margin-top:13px">
        <div class="list-head"><strong>历史形态事件</strong><span id="resultCount" class="hint"></span></div>
        <div class="filterbar"><input id="search" placeholder="搜索日期、形态或状态"><input id="dateFrom" type="date" title="确认日期开始"><input id="dateTo" type="date" title="确认日期结束"></div><div class="typechips" id="typeFilter" aria-label="形态多选"></div>
        <div class="hint" style="margin-bottom:8px">形态可多选；按日期范围筛选确认事件。点击行查看结构详情。</div>
        <div class="eventlist"><table class="events"><thead><tr><th>确认日期</th><th>形态</th><th>结构形成</th><th>下一交易日</th></tr></thead><tbody id="tbody"></tbody></table></div>
      </section>
    </div>
    <aside class="panel detail" id="detail"></aside>
  </div>
  <footer class="footer" id="footer"></footer>
</main>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('dataset').textContent);
const B=D.bars||[], RAW=D.signals||[];
const DATE_INDEX=new Map(B.map((b,i)=>[b.date,i]));
const COLORS={w_bottom:'#d85a43',inverse_head_shoulders:'#2878c8',triangle_bottom:'#168b78',bull_flag:'#8059c7'};
const NAMES={w_bottom:'W底',inverse_head_shoulders:'头肩底',triangle_bottom:'收敛三角形底部',bull_flag:'下飘旗形上涨中继'};
function patterns(s){return Array.isArray(s.patterns)&&s.patterns.length?s.patterns:(s.pattern?[s.pattern]:[])}
const EVENTS=RAW.map((s,i)=>({i,s,ps:patterns(s),index:DATE_INDEX.get(s.signal_date)??-1}));
const TYPES=[...new Set(EVENTS.flatMap(e=>e.ps.map(p=>p.pattern_type||'unknown')))].sort();
const fmt=(x,n=3)=>x!==null&&x!==undefined&&Number.isFinite(Number(x))?Number(x).toFixed(n):'—';
const esc=x=>String(x??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\"/g,'&quot;').replace(/'/g,'&#39;');
const tagClass=t=>t==='w_bottom'?'w':t==='inverse_head_shoulders'?'hs':t==='triangle_bottom'?'tri':'flag';
const label=(p)=>p.pattern_label||p.label||NAMES[p.pattern_type]||p.pattern_type||'未知形态';
const stats=document.getElementById('stats');
const shapeCounts={};let shapeN=0;
EVENTS.forEach(e=>e.ps.forEach(p=>{const t=p.pattern_type||'unknown';shapeCounts[t]=(shapeCounts[t]||0)+1;shapeN++}));
const statData=[
  ['确认事件',EVENTS.length+' 条'],['形态标签',shapeN+' 个'],['行情K线',B.length.toLocaleString()+' 根'],
  ['区间',D.first_date+' 至 '+D.last_date],['识别用时',fmt(D.generated_seconds,1)+' 秒']
];
stats.innerHTML=statData.map(x=>'<div class=\"stat\"><strong>'+esc(x[1])+'</strong><span>'+esc(x[0])+'</span></div>').join('');
const prior=D.previous_case||{}, busy=D.concurrent_work_observed||{};
document.getElementById('provenance').innerHTML='规则版本：<b>'+esc(D.source_rule_version||'—')+'</b> · 引擎：<b>'+esc(D.engine_version||'—')+'</b> · 数据版本：<b>'+esc(D.data_revision||'—')+'</b>。本次重跑为<b>'+RAW.length+'条确认事件</b>；此前'+esc(prior.engine_version||'旧版')+'案例记录'+esc(prior.signal_events||'—')+'条，但未保存全部事件明细。'+(busy.run_id?'本次计算期间还观察到全市场扫描并行运行（'+busy.scanned+'/'+busy.total+'）。':'')+'页面列出当前版本可复算的完整历史结果。';
document.getElementById('footer').textContent='口径说明：本页列出因果识别器确认事件，不等同于已成交交易，也不代表策略盈利。截止日之后的行情未纳入；末日信号的下一交易日可能因本次截断而不可见。历史行情显示值来自本地CSV。';
const typeFilter=document.getElementById('typeFilter');
typeFilter.innerHTML=TYPES.map(t=>'<label><input type=\"checkbox\" value=\"'+esc(t)+'\" checked>'+esc(NAMES[t]||t)+' ('+shapeCounts[t]+')</label>').join('');
document.getElementById('dateFrom').value=D.first_date||'';
document.getElementById('dateTo').value=D.last_date||'';
let selected=EVENTS.length-1, subIndex=0, viewBars=Math.min(360,B.length), viewStart=Math.max(0,B.length-viewBars);
let dragging=false, dragX=0, dragStart=0;
function selectedEvent(){return EVENTS[selected]||null}
function selectedPattern(){const e=selectedEvent();return e&&e.ps[Math.min(subIndex,e.ps.length-1)]}
function selectedTag(e,p){if(!e||!p)return null;return (e.s.pattern_tags||e.s.tags||[]).find(t=>t.pattern_id===p.pattern_id)||null}
function neckInfo(e,p){const tag=selectedTag(e,p);const sel=tag&&tag.neckline_selection;const line=(p&&p.neckline)||(sel&&sel.selected_line);return {line,sel}}
function pointRows(p){const k=p&&p.key_points||{};const out=[];for(const key of Object.keys(k)){const m=key.match(/^pivot_(\\d+)_index$/);if(!m)continue;const n=m[1],idx=Number(k[key]);out.push({n,idx,date:k['pivot_'+n+'_date']||B[idx]?.date||'',price:k['pivot_'+n],kind:k['pivot_'+n+'_kind']||'',role:k['pivot_'+n+'_role']||''})}return out.sort((a,b)=>a.idx-b.idx)}
function mainEvidence(p){return (p&&p.evidence||[]).find(x=>x.code==='main_structure')?.value||{}}
function monthSpan(a,b){if(!a||!b)return null;const x=new Date(a),y=new Date(b);return Math.max(0,(y-x)/(86400000*30.4375))}
function renderDetail(){
 const e=selectedEvent(),p=selectedPattern(),box=document.getElementById('detail');
 if(!e||!p){box.innerHTML='<div class=\"no-data\">没有匹配事件</div>';return}
 const s=e.s,tag=selectedTag(e,p),neck=neckInfo(e,p),ev=mainEvidence(p),idx=e.index;
 const entry=s.entry_date,entryIdx=DATE_INDEX.get(entry),entryBar=entryIdx===undefined?null:B[entryIdx];
 const signalBar=idx>=0?B[idx]:null,neckVal=neck.line&&idx>=0?neck.line.slope*idx+neck.line.intercept:null;
 const pts=pointRows(p),duration=monthSpan(p.formed_at||p.key_points?.first_trough_date,s.signal_date);
 const targets=Array.isArray(p.targets)&&p.targets.length?p.targets.map(t=>typeof t==='object'?JSON.stringify(t):String(t)).join('；'):'未提供结构目标';
 const multi=e.ps.length>1?'<div class=\"tagselect\">'+e.ps.map((x,i)=>'<button class=\"'+(i===subIndex?'selected':'')+'\" data-sub=\"'+i+'\">'+esc(label(x))+'</button>').join('')+'</div>':'';
 const pointHtml=pts.map(x=>'<li><b>P'+x.n+'</b> · '+esc(x.date)+' · '+esc(x.kind||'结构点')+' · '+fmt(x.price,3)+'</li>').join('')||'<li>该形态未提供可绘制的枢轴字段</li>';
 box.innerHTML='<div class=\"detail-top\"><div><div class=\"detail-date\">'+esc(s.signal_date)+'</div><div class=\"mini-note\">确认事件 '+(e.i+1)+' / '+EVENTS.length+'</div></div><span class=\"badge\">'+esc(p.state||'confirmed')+'</span></div>'+
   multi+'<h2 style=\"margin-top:8px\">'+esc(label(p))+'</h2>'+
   '<div class=\"field-grid\"><div class=\"field\"><span>结构起点</span><strong>'+esc(p.formed_at||p.key_points?.first_trough_date||'—')+'</strong></div><div class=\"field\"><span>确认日期</span><strong>'+esc(s.signal_date)+'</strong></div>'+
   '<div class=\"field\"><span>下一交易日</span><strong>'+esc(entry||'截断后无下一根行情')+'</strong></div><div class=\"field\"><span>确认日收盘</span><strong>'+(signalBar?fmt(signalBar.close,3):'—')+'</strong></div>'+
   '<div class=\"field\"><span>次日开盘</span><strong>'+(entryBar?fmt(entryBar.open,3):'—')+'</strong></div><div class=\"field\"><span>结构时长</span><strong>'+(duration===null?'—':fmt(duration,1)+' 个月')+'</strong></div></div>'+
   '<div class=\"neck\"><b>选中颈线</b><br>确认日参考值：'+(neckVal===null?'—':fmt(neckVal,3))+' · 独立测试：'+(neck.sel?.independent_tests?.length??ev.neckline_independent_test_count??'—')+' 次<br>冻结：'+esc(neck.sel?.freeze_date||ev.frozen_at||'—')+' · 线规则：'+esc(neck.sel?.rule_version||ev.neckline_rule_version||'—')+'</div>'+
   '<h3>形态结构点</h3><ul class=\"points\">'+pointHtml+'</ul>'+
   '<div class=\"field-grid\"><div class=\"field\"><span>失效参考</span><strong>'+fmt(p.invalidation,3)+'</strong></div><div class=\"field\"><span>结构目标</span><strong>'+esc(targets)+'</strong></div></div>'+
   '<div class=\"mini-note\">已知时间：'+esc(ev.structure_known_at||p.confirmation_date||'—')+' · 信号入场与实际成交需区分；本页未重跑Backtrader订单账本。</div>'+
   '<details><summary>查看完整识别证据</summary><pre>'+esc(JSON.stringify({pattern:p,tag:tag},null,2))+'</pre></details>';
 box.querySelectorAll('[data-sub]').forEach(b=>b.addEventListener('click',()=>{subIndex=Number(b.dataset.sub);renderDetail();drawChart()}));
}
function setCanvas(canvas,height){
 const dpr=window.devicePixelRatio||1,w=Math.max(1,canvas.clientWidth);
 if(canvas.height!==Math.round(height*dpr)||canvas.width!==Math.round(w*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(height*dpr)}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,w,h:height};
}
function eventTypes(e){return e.ps.map(p=>p.pattern_type||'unknown')}
function eventColor(e){return COLORS[eventTypes(e)[0]]||'#6c7f95'}
function drawOverview(){
 const c=document.getElementById('overview'),{ctx,w,h}=setCanvas(c,96);ctx.clearRect(0,0,w,h);
 if(!B.length)return;const left=42,right=w-12,top=9,bottom=h-18,prices=B.map(x=>x.close),lo=Math.min(...prices),hi=Math.max(...prices),span=hi-lo||1;
 ctx.strokeStyle='#e6ebf1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
 ctx.strokeStyle='#7992b4';ctx.lineWidth=1.25;ctx.beginPath();B.forEach((b,i)=>{const x=left+i/(B.length-1)*(right-left),y=bottom-(b.close-lo)/span*(bottom-top);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();
 EVENTS.forEach(e=>{if(e.index<0)return;const x=left+e.index/(B.length-1)*(right-left);ctx.fillStyle=eventColor(e);ctx.beginPath();ctx.arc(x,top+3,2.5,0,Math.PI*2);ctx.fill()});
 const a=left+viewStart/(B.length-1)*(right-left),z=left+Math.min(B.length-1,viewStart+viewBars)/(B.length-1)*(right-left);
 ctx.fillStyle='#4273cc18';ctx.fillRect(a,top,z-a,bottom-top);ctx.strokeStyle='#577bd2';ctx.setLineDash([4,3]);ctx.strokeRect(a,top,z-a,bottom-top);ctx.setLineDash([]);
}
function drawChart(){
 const c=document.getElementById('chart'),height=c.clientHeight||410,{ctx,w,h}=setCanvas(c,height);ctx.clearRect(0,0,w,h);
 const e=selectedEvent(),p=selectedPattern();if(!B.length)return;
 viewBars=Math.min(Math.max(80,viewBars),B.length);viewStart=Math.max(0,Math.min(B.length-viewBars,viewStart));
 document.getElementById('pan').max=String(Math.max(0,B.length-viewBars));document.getElementById('pan').value=String(viewStart);
 document.getElementById('panLabel').textContent=(viewStart+1)+'–'+Math.min(B.length,viewStart+viewBars)+' / '+B.length;
 document.getElementById('zoom').value=String(viewBars);document.getElementById('zoomLabel').textContent=viewBars+' 根';
 const x0=58,x1=w-64,top=15,priceBottom=h-92,volTop=priceBottom+25,volBottom=h-27;
 const start=viewStart,end=Math.min(B.length,viewStart+viewBars),slice=B.slice(start,end);
 let lo=Math.min(...slice.map(b=>b.low)),hi=Math.max(...slice.map(b=>b.high));
 if(e&&e.index>=start&&e.index<end&&p){const line=neckInfo(e,p).line;if(line){for(const q of [start,end-1]){const y=line.slope*q+line.intercept;lo=Math.min(lo,y);hi=Math.max(hi,y)}}}
 const pad=(hi-lo)*.06||.01;lo=Math.max(0,lo-pad);hi+=pad;const span=hi-lo||1,maxVol=Math.max(...slice.map(b=>b.volume||0),1),step=(x1-x0)/Math.max(1,slice.length);
 const X=i=>x0+(i-start+.5)*step,Y=v=>priceBottom-(v-lo)/span*(priceBottom-top);
 ctx.font='10px -apple-system, sans-serif';ctx.textAlign='right';ctx.fillStyle='#738198';ctx.strokeStyle='#e7ecf2';ctx.lineWidth=1;
 for(let k=0;k<=4;k++){const y=top+k*(priceBottom-top)/4,val=hi-k*span/4;ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();ctx.fillText(val.toFixed(3),x0-7,y+3)}
 ctx.fillStyle='#99a5b5';ctx.textAlign='left';ctx.fillText('成交量',x0,volTop-8);
 slice.forEach((b,j)=>{const i=start+j,x=X(i),up=b.close>=b.open,col=up?'#d24b45':'#17845f';ctx.strokeStyle=col;ctx.lineWidth=Math.max(1,Math.min(2,step*.36));ctx.beginPath();ctx.moveTo(x,Y(b.high));ctx.lineTo(x,Y(b.low));ctx.stroke();const bw=Math.max(1,Math.min(8,step*.62)),y1=Y(Math.max(b.open,b.close)),y2=Y(Math.min(b.open,b.close));ctx.fillStyle=col;ctx.fillRect(x-bw/2,y1,bw,Math.max(1.1,y2-y1));ctx.fillStyle=col+'a0';const vh=(b.volume||0)/maxVol*(volBottom-volTop);ctx.fillRect(x-bw/2,volBottom-vh,bw,Math.max(1,vh))});
 if(e&&p){
   const line=neckInfo(e,p).line;
   if(line){const tag=selectedTag(e,p),sel=tag&&tag.neckline_selection;let from=sel?.effective_start_date?DATE_INDEX.get(sel.effective_start_date):null;if(from===undefined||from===null)from=e.index-viewBars;from=Math.max(start,from);const to=Math.min(end-1,e.index);if(to>=from){ctx.strokeStyle='#7455d8';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(X(from),Y(line.slope*from+line.intercept));ctx.lineTo(X(to),Y(line.slope*to+line.intercept));ctx.stroke();ctx.fillStyle='#6045ba';ctx.textAlign='left';ctx.fillText('颈线 '+(line.slope*to+line.intercept).toFixed(3),Math.min(x1-82,X(to)+5),Y(line.slope*to+line.intercept)-7)}}
   pointRows(p).forEach(pt=>{if(pt.idx<start||pt.idx>=end)return;const x=X(pt.idx),y=Y(Number(pt.price));ctx.fillStyle='#fff';ctx.strokeStyle='#342357';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#342357';ctx.textAlign='center';ctx.fillText('P'+pt.n,x,y-9)});
 }
 EVENTS.forEach(x=>{if(x.index<start||x.index>=end)return;const b=B[x.index],xx=X(x.index);ctx.fillStyle=eventColor(x);ctx.globalAlpha=x===e?1:.42;ctx.beginPath();ctx.moveTo(xx,Y(b.high)-7);ctx.lineTo(xx-4,Y(b.high)-14);ctx.lineTo(xx+4,Y(b.high)-14);ctx.closePath();ctx.fill();ctx.globalAlpha=1});
 const ticks=[start,Math.floor((start+end-1)/2),end-1];ctx.fillStyle='#66768b';ctx.textAlign='center';ticks.forEach(i=>ctx.fillText(B[i].date,x0+(i-start+.5)*step,h-7));
 document.getElementById('chartStatus').textContent=e?p.pattern_type+' · '+(e.s.signal_date||''):'选择一个事件';
}
function filtered(){
 const chosen=new Set([...typeFilter.querySelectorAll('input:checked')].map(o=>o.value)),q=document.getElementById('search').value.trim().toLowerCase(),from=document.getElementById('dateFrom').value,to=document.getElementById('dateTo').value;
 return EVENTS.filter(e=>{const types=eventTypes(e);if(chosen.size&&!types.some(t=>chosen.has(t)))return false;if(from&&e.s.signal_date<from)return false;if(to&&e.s.signal_date>to)return false;if(q&&!([e.s.signal_date,...(e.s.pattern_labels||[]),...e.ps.map(p=>p.state)].join(' ').toLowerCase().includes(q)))return false;return true}).slice().reverse()
}
function renderList(){
 const list=filtered(),tb=document.getElementById('tbody');document.getElementById('resultCount').textContent=list.length+' / '+EVENTS.length+' 条';
 tb.innerHTML=list.map(e=>'<tr data-id=\"'+e.i+'\" class=\"'+(e.i===selected?'active':'')+'\"><td>'+esc(e.s.signal_date)+'</td><td>'+e.ps.map(p=>'<span class=\"shape '+tagClass(p.pattern_type)+'\">'+esc(label(p))+'</span>').join('')+'</td><td>'+esc(e.ps[0]?.formed_at||'—')+'</td><td>'+esc(e.s.entry_date||'—')+'</td></tr>').join('')||'<tr><td colspan=\"4\" class=\"no-data\">没有符合条件的事件</td></tr>';
 tb.querySelectorAll('tr[data-id]').forEach(tr=>tr.addEventListener('click',()=>selectEvent(Number(tr.dataset.id))));
}
function selectEvent(id){
 selected=id;subIndex=0;const e=selectedEvent();if(e&&e.index>=0){viewStart=Math.max(0,Math.min(B.length-viewBars,e.index-Math.floor(viewBars/2)))}renderDetail();renderList();drawChart();drawOverview()
}
function setWindow(size,center){
 viewBars=Math.min(size,B.length);const e=selectedEvent();const idx=center??(e&&e.index>=0?e.index:B.length-1);viewStart=Math.max(0,Math.min(B.length-viewBars,idx-Math.floor(viewBars/2)));drawChart();drawOverview()
}
document.getElementById('pan').addEventListener('input',e=>{viewStart=Number(e.target.value);drawChart();drawOverview()});
document.getElementById('zoom').addEventListener('input',e=>setWindow(Number(e.target.value)));
document.getElementById('focusEvent').addEventListener('click',()=>selectEvent(selected));
document.getElementById('showAll').addEventListener('click',()=>setWindow(B.length,selectedEvent()?.index));
['search','dateFrom','dateTo'].forEach(id=>document.getElementById(id).addEventListener('input',renderList));typeFilter.addEventListener('change',renderList);
document.getElementById('overview').addEventListener('click',ev=>{const r=ev.currentTarget.getBoundingClientRect(),x=(ev.clientX-r.left)/r.width,idx=Math.round(x*(B.length-1));setWindow(viewBars,idx)});
document.getElementById('chart').addEventListener('wheel',ev=>{ev.preventDefault();const next=Math.max(120,Math.min(2600,viewBars+(ev.deltaY>0?80:-80)));setWindow(next)}, {passive:false});
document.getElementById('chart').addEventListener('pointerdown',ev=>{dragging=true;dragX=ev.clientX;dragStart=viewStart;ev.currentTarget.setPointerCapture(ev.pointerId)});
document.getElementById('chart').addEventListener('pointermove',ev=>{if(!dragging)return;const pixels=ev.currentTarget.clientWidth,delta=Math.round((dragX-ev.clientX)/Math.max(1,pixels)*viewBars);viewStart=Math.max(0,Math.min(B.length-viewBars,dragStart+delta));drawChart();drawOverview()});
document.getElementById('chart').addEventListener('pointerup',()=>dragging=false);
window.addEventListener('resize',()=>{drawOverview();drawChart()});
renderList();if(EVENTS.length)selectEvent(EVENTS.length-1);else{renderDetail();drawOverview();drawChart()}
</script>
</body>
</html>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    import math
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    compact = json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    compact = compact.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    OUTPUT.write_text(PAGE.replace("__DATA__", compact), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.resolve()),
        "bytes": OUTPUT.stat().st_size,
        "bars": payload.get("bars_count"),
        "events": len(payload.get("signals", [])),
        "engine": payload.get("engine_version"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
