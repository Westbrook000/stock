const names={w_bottom:'W底',spring:'破底翻',w_bottom_spring:'W底破底翻',bull_flag:'下飘旗形上涨中继',inverse_head_shoulders:'头肩底',triangle_bottom:'收敛三角形底部',m_top:'M头',false_breakout:'假突破',false_breakout_head_shoulders:'假突破（头肩顶）',bear_flag:'上飘旗形下跌中继',head_shoulders:'头肩顶',triangle_top:'收敛三角形头部'};
const actions={CONSIDER_BUY:'可考虑买入',WAIT:'等待',HOLD:'持有观察',SELL_IF_HELD:'如持有可考虑卖出',INSUFFICIENT_DATA:'数据不足'};
const reasonNames={timeframe_direction_conflict:'日线与周线确认方向冲突，请分别复核',pattern_requires_manual_review:'多个候选形态共用结构点，需人工复核',no_confirmed_actionable_pattern:'当前没有通过确认与交易空间检查的形态',no_current_pattern:'当前周期没有识别到有效形态',confirmed_bearish_structure_sell_warning_if_held:'已确认顶部或空头结构；如有持仓，请复核退出计划',bullish_structure_exists_without_new_entry:'存在多头结构，但当前没有新的入场信号',pattern_not_confirmed_bullish:'形态尚未确认为多头结构',volume_data_unverified:'成交量单位或企业行动未核验，形态仅作研究观察',invalid_structural_stop:'无法建立有效的结构止损',no_verified_target:'没有可用于计划的目标位',structural_risk_over_7pct:'结构止损对应风险超过 7%',first_target_already_reached_or_below_entry:'首个目标已触及或低于当前参考价',entry_above_maximum_allowed_price:'当前参考价高于最高允许买入价',insufficient_cash_or_liquidity:'现金或流动性不足以形成有效委托',net_first_target_below_2R:'首个目标空间不足 2R',buy_gates_not_met:'买入条件未全部满足'};
const $=id=>document.getElementById(id);let lastAnalysis=null;let scanRows=[];let scanShown=0;let scanTimer=null;let detailChart=null;let chartBarCount=0;let chartWindowBars=120;

document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>{document.querySelectorAll('nav button,.panel').forEach(item=>item.classList.remove('active'));button.classList.add('active');$(button.dataset.panel).classList.add('active')});
Object.entries(names).forEach(([key,value])=>{const option=document.createElement('option');option.value=key;option.textContent=value;$('scanPattern').append(option)});
fetch('/api/v1/health').then(response=>response.json()).then(result=>$('health').textContent=`${result.status} · ${result.rule_version} · 无下单`).catch(()=>$('health').textContent='服务未连接');

function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]))}
function patternName(pattern){return names[pattern.pattern_type]||pattern.pattern_type||'暂无形态'}
function stateLabel(pattern){let label;if(pattern.state==='forming')label=`${patternName(pattern)}，未确认`;else if(pattern.state==='confirmed')label=`${patternName(pattern)}，价格确认${pattern.volume_verified?'':'，量能未核验'}`;else if(pattern.state==='invalidated')label=`${patternName(pattern)}，已失效`;else if(pattern.state==='target_reached')label=`${patternName(pattern)}，目标已达`;else if(pattern.state==='followthrough_confirmed')label=`${patternName(pattern)}，箱体后续边界已确认（历史）`;else label=`${patternName(pattern)}，${pattern.state||'状态未知'}`;if(pattern.flags?.includes('manual_review_required'))label+='，待人工复核';return label}
function isTerminalPattern(pattern){return ['invalidated','target_reached','followthrough_confirmed'].includes(pattern.state)}
function actionLabel(action){return actions[action]||action||'状态未知'}
function reasonLabel(reason){return reasonNames[reason]||reason||'条件暂未满足'}
function isInternalEvidence(item){const code=item.code||'';return ['competing_topology_labels','confirmed_date','confirmation_index','break_index','manual_review'].includes(code)||code.endsWith('_pattern_id')||/^inner_spring_(break_index|confirmation_index)$/.test(code)}
function dataWarningLabel(warning){let value=String(warning??'');let timeframe='';const status=value.match(/^(1d|1w) data status:\s*(.*)$/);if(status){timeframe=status[1]==='1d'?'日线：':'周线：';value=`data status: ${status[2]}`}else{const prefix=value.match(/^(1d|1w):\s*/);if(prefix){timeframe=prefix[1]==='1d'?'日线：':'周线：';value=value.slice(prefix[0].length)}}if(value==='volume unit is not verified for this file hash'||value==='volume unit or corporate-action adjustment is unverified')value='成交量单位或复权口径尚未核验';else if(value==='corporate-action adjustment is not verified for this file hash')value='复权口径尚未核验';else if(value==='instrument registry is not bound to the current file hash')value='标的交易制度资料未与当前行情文件绑定';else if(value==='verified exchange calendar unavailable; weekly bars are preview-only')value='缺少已核验的交易日历，周线仅作预览';else if(value==='no completed weekly bars; latest week is preview-only')value='尚无已完成周线，当前周仅作预览';else if(value==='data status: insufficient')value='数据量不足，暂不用于正式判断';else {const split=value.match(/^unverified price discontinuity split (\d+) segment\(s\)$/);if(split)value=`发现未核验的价格断点，已分成 ${split[1]} 段分析`}return timeframe+value}
function dataWarningsMarkup(warnings){const items=[...new Set((warnings||[]).map(dataWarningLabel))];if(!items.length)return '';return `<details class="data-quality"><summary>数据待核验 · ${items.length} 项（分析仅作参考）</summary><ul>${items.map(item=>`<li>${escapeHtml(item)}</li>`).join('')}</ul></details>`}
function fmt(value,digits=2){return Number.isFinite(Number(value))?Number(value).toFixed(digits):'—'}
function planMarkup(plan){if(!plan)return '';const reasons=(plan.reason_codes||[]).map(reasonLabel).map(escapeHtml).join('；')||'风险与目标条件通过';return `<div class="plan-summary">计划参考：${escapeHtml(plan.decision)} · 参考价 ${fmt(plan.entry_price)} · 结构止损 ${fmt(plan.stop)} · 首目标 ${fmt(plan.targets?.[0])} · 最高允许价 ${fmt(plan.highest_allowed_entry)} · 风险 ${fmt((plan.planned_risk_pct||0)*100)}% · R ${fmt(plan.reward_risk)}<br>${reasons}${dataWarningsMarkup(plan.data_warnings)}</div>`}
function evidenceLabel(item){const labels={pivot_topology:'结构点顺序',neckline:'颈线',neckline_left:'左侧颈线',neckline_right:'右侧颈线',close_confirmation:'收盘突破确认',close_reclaim:'收盘收复',reclaim_level:'收复价位',break_level:'越界价位',volume_verified:'量能已核验',box_lower:'箱体下缘',box_upper:'箱体上缘',box_first_touch:'箱体起始触点',box_last_touch:'箱体末端触点',box_width_atr:'箱体宽度（ATR）',followthrough_confirmed:'后续边界确认',target_reached:'首目标触达',structure_invalidation:'结构极值失效',outer_w_invalidation:'外层W底失效',neckline_projection_invalidation:'颈线外推失效',w_geometry_metrics:'W底几何特征'};let code=item.code||'',prefix='';if(code.startsWith('inner_spring_')){prefix='内层';code=code.slice('inner_spring_'.length)}let value=item.value;if(code==='pivot_topology'&&Array.isArray(value))value=value.map(point=>point==='high'?'高点':point==='low'?'低点':String(point)).join(' → ');else if(code==='w_geometry_metrics'&&value&&typeof value==='object')value=`左段 ${value.first_leg_bars} 根 · 右段 ${value.second_leg_bars} 根 · 时长比 ${value.leg_duration_ratio} · 两底差 ${value.trough_gap_pct}%`;else if((code==='followthrough_confirmed'||code==='target_reached')&&value&&typeof value==='object')value=`${value.date||'日期未知'} · 价位 ${fmt(value.level??value.price)}`;else if((code==='structure_invalidation'||code==='outer_w_invalidation'||code==='neckline_projection_invalidation')&&value&&typeof value==='object')value=`${value.date||'日期未知'} · ${code==='neckline_projection_invalidation'?'颈线':'失效价位'} ${fmt(value.level)}`;else if(typeof value==='boolean')value=value?'是':'否';else if(value&&typeof value==='object')value=JSON.stringify(value);else if(typeof value==='number')value=fmt(value);return `${prefix}${labels[code]||code}：${String(value??'—')}`}
function patternCard(pattern){const pivotDate=((pattern.evidence||[]).find(item=>item.code==='confirmed_date')||{}).value;const confirmed=pattern.confirmation_date||pivotDate||'—';const confirmationLabel=pattern.confirmation_date?'价格确认':'结构点确认';const overlaps=(pattern.evidence||[]).find(item=>item.code==='competing_topology_labels')?.value||[];const warning=pattern.flags?.includes('manual_review_required')?`<div class="pattern-review-warning"><b>候选形态有重叠，暂不生成买卖计划</b>${overlaps.length?`<span>共享同一组结构点：${overlaps.map(patternType=>escapeHtml(names[patternType]||patternType)).join('、')}</span>`:''}</div>`:'';const evidence=(pattern.evidence||[]).filter(item=>!isInternalEvidence(item)).map(item=>escapeHtml(evidenceLabel(item))).join(' · ');return `<article class="card"><div><b>${escapeHtml(pattern.code)} · ${escapeHtml(patternName(pattern))}</b> <span class="tag">${escapeHtml(pattern.timeframe)} / ${escapeHtml(stateLabel(pattern))}</span><div class="muted">形成 ${escapeHtml(pattern.formed_at||'—')} · ${confirmationLabel} ${escapeHtml(confirmed)} · 引擎 ${escapeHtml(pattern.rule_version||'—')} · 规则集 ${escapeHtml(pattern.source_rule_version||'—')} · 数据 ${escapeHtml(pattern.data_version||'—')}</div>${warning}<div class="evidence">${evidence}</div>${planMarkup(pattern.trade_plan)}</div><button onclick="pick('${escapeHtml(pattern.pattern_id||'')}')">人工复核</button></article>`}
window.pick=id=>{$('reviewId').value=id;document.querySelector('[data-panel=review]').click()};

function miniCandles(bars){if(!bars?.length)return '<span class="mini-empty">暂无K线</span>';const width=184,height=48,pad=3,step=(width-pad*2)/bars.length;const lows=bars.map(bar=>Number(bar.low)),highs=bars.map(bar=>Number(bar.high));const min=Math.min(...lows),max=Math.max(...highs),range=max-min||1;const y=value=>pad+(max-value)/range*(height-pad*2);let shapes='';bars.forEach((bar,index)=>{const x=pad+index*step+step/2,open=y(Number(bar.open)),close=y(Number(bar.close)),up=Number(bar.close)>=Number(bar.open),color=up?'#d84a4a':'#18866b',body=Math.max(1,Math.abs(close-open));shapes+=`<line x1="${x}" y1="${y(Number(bar.high))}" x2="${x}" y2="${y(Number(bar.low))}" stroke="${color}" stroke-width="1"/><rect x="${x-step*.28}" y="${Math.min(open,close)}" width="${Math.max(1,step*.56)}" height="${body}" fill="${color}"/>`});return `<svg class="mini-candles" viewBox="0 0 ${width} ${height}" role="img" aria-label="最近${bars.length}根日K线">${shapes}</svg>`}
function scanCard(row){const plan=row.selected_plan;const activePatterns=(row.patterns||[]).filter(pattern=>!isTerminalPattern(pattern));const main=plan?row.patterns.find(pattern=>pattern.pattern_id===plan.pattern_id):(activePatterns[0]||row.patterns?.[0]);const label=main?.display_label||main?.pattern_type&&stateLabel(main)||'暂无当前形态';const warnings=[...(row.data_warnings||[]),...(plan?.data_warnings||[])];const quality=warnings.length?'数据待核验':'行情可用';const price=`${fmt(row.latest_close)} · ${row.latest_date||''}`;return `<article class="scan-row" data-code="${escapeHtml(row.code)}" data-kind="${escapeHtml(row.kind)}" tabindex="0" role="link" aria-label="查看${escapeHtml(row.name||row.code)}详情"><div class="scan-main"><div class="scan-title"><b>${escapeHtml(row.name||row.code)}</b><span class="muted">${escapeHtml(row.code)} · ${row.kind==='etf'?'ETF':'股票'}</span></div><span class="action-pill action-${escapeHtml(row.action||'WAIT')}">${escapeHtml(actionLabel(row.action))}</span><span class="pattern-label">${escapeHtml(label)}</span><span class="muted">${escapeHtml(reasonLabel(row.reason_codes?.[0]||row.explanation||''))}</span>${dataWarningsMarkup(warnings)}</div><div class="scan-chart">${miniCandles(row.mini_bars)}<span class="muted">${escapeHtml(price)}</span></div><div class="scan-metrics">${plan?`<b>风险 ${fmt((plan.planned_risk_pct||0)*100)}%</b><b>首目标 ${fmt(plan.targets?.[0])}</b><b>${fmt(plan.reward_risk)}R</b>`:`<b>${activePatterns.length} 个当前结构</b>${row.patterns?.length>activePatterns.length?`<span class="muted">另有 ${row.patterns.length-activePatterns.length} 个结束结构</span>`:''}<span class="muted">${escapeHtml(quality)}</span>`}<button class="detail-link">查看详情 →</button></div></article>`}
function renderMore(){const end=Math.min(scanRows.length,scanShown+100);$('results').insertAdjacentHTML('beforeend',scanRows.slice(scanShown,end).map(scanCard).join(''));scanShown=end;const more=$('loadMore');if(more)more.remove();if(scanShown<scanRows.length)$('results').insertAdjacentHTML('beforeend',`<button id="loadMore" class="load-more">再显示 100 个（剩余 ${scanRows.length-scanShown}）</button>`);$('loadMore')?.addEventListener('click',renderMore);document.querySelectorAll('.scan-row:not([data-bound])').forEach(row=>{row.dataset.bound='1';const open=()=>openTicker(row.dataset.code,row.dataset.kind);row.addEventListener('click',open);row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}})})}
function openTicker(code,kind){$('code').value=code;$('kind').value=kind;$('tf').value='1d';$('asof').value='';document.querySelector('[data-panel=detail]').click();analyze()}
function renderScan(rows,inventory){scanRows=rows||[];scanShown=0;$('results').innerHTML='';$('scanSummary').innerHTML=`<div class="scan-summary"><b>扫描完成</b> · 展示 ${scanRows.length} 个标的 · 股票 ${inventory?.stock_count??'—'} · ETF ${inventory?.etf_count??'—'} · 买入候选 ${scanRows.filter(row=>row.action==='CONSIDER_BUY').length} · 按可解释规则排序</div>`;renderMore()}

function decisionMarkup(decision){if(!decision)return '';const reasons=(decision.reason_codes||[]).map(reasonLabel).map(escapeHtml).join('；')||'风险与空间条件通过';const plan=decision.selected_plan;return `<section class="decision-box decision-${escapeHtml(decision.action)}"><div class="decision-heading"><span class="action-pill action-${escapeHtml(decision.action)}">${escapeHtml(actionLabel(decision.action))}</span><b>${escapeHtml(decision.explanation||'')}</b></div><div>日线/周线确认数：${decision.rank?.confirmed_timeframes??0} · 方向冲突：${decision.timeframe_conflict?'是':'否'} · 排序依据：${plan?`首目标 ${fmt(plan.targets?.[0])}，${fmt(plan.reward_risk)}R，风险 ${fmt((plan.planned_risk_pct||0)*100)}%`:'没有通过买入门槛的计划'}</div><div>原因：${reasons}</div>${dataWarningsMarkup(decision.data_warnings)}${planMarkup(plan)}</section>`}
function chartVisibleCount(value, timeframe, total){
  if(value==='all')return total;
  if(value==='custom')return Math.min(total,chartWindowBars);
  const preset=Number(value)||120;
  return Math.min(total,timeframe==='1w'?Math.max(13,Math.round(preset/5)):preset);
}
function applyChartRange(value){
  if(!lastAnalysis)return;
  chartWindowBars=chartVisibleCount(value,lastAnalysis.timeframe,chartBarCount);
  renderDetailChart(lastAnalysis,chartWindowBars);
}
function zoomChart(direction){
  if(!lastAnalysis||!chartBarCount)return;
  const next=Math.max(Math.min(20,chartBarCount),Math.min(chartBarCount,Math.round(chartWindowBars*(direction==='in'?.7:1/.7))));
  chartWindowBars=next;
  const presets=['60','120','250','500'].map(Number),nearest=presets.find(value=>value===next);
  $('chartRange').value=next>=chartBarCount?'all':nearest?String(nearest):'custom';
  renderDetailChart(lastAnalysis,next);
}
async function analyze(){
  const query=new URLSearchParams({code:$('code').value,kind:$('kind').value,timeframe:$('tf').value});
  if($('asof').value)query.set('asof',$('asof').value);
  $('detailRows').innerHTML='<p>正在分析…</p>';
  try{
    const response=await fetch('/api/v1/analysis?'+query),data=await response.json();
    if(!response.ok)throw new Error(data.detail||'分析请求失败');
    lastAnalysis=data;
    const details=data.decision?.patterns||data.patterns||[],plan=data.decision?.selected_plan;
    const patterns=details.map(pattern=>({...pattern,trade_plan:pattern.pattern_id===plan?.pattern_id?plan:(data.patterns||[]).find(item=>item.pattern_id===pattern.pattern_id)?.trade_plan}));
    const activePatterns=patterns.filter(pattern=>!isTerminalPattern(pattern));
    const completedPatterns=patterns.filter(isTerminalPattern);
    const activeCards=activePatterns.map(patternCard).join('')||'<p>截至该日没有当前有效形态。周线若缺少日历会显示预览K线，但不产生正式周线信号。</p>';
    const completedCards=completedPatterns.length?`<details class="completed-patterns"><summary>已结束的历史形态（${completedPatterns.length}）</summary>${completedPatterns.map(patternCard).join('')}</details>`:'';
    $('detailRows').innerHTML=decisionMarkup(data.decision)+activeCards+completedCards;
    if(data.decision?.timeframe_conflict)$('detailRows').insertAdjacentHTML('afterbegin','<p class="conflict-banner">日线与周线方向冲突：两周期独立保留，请人工复核。</p>');
    if(data.patterns?.[0])$('reviewId').value=data.patterns[0].pattern_id;
    const bars=data._bars||[];chartBarCount=bars.length;chartWindowBars=chartVisibleCount($('chartRange').value,data.timeframe,bars.length);
    renderDetailChart(data,chartWindowBars);
  }catch(error){$('detailRows').innerHTML=`<div class="error-box">分析失败：${escapeHtml(error.message)}</div>`}
}
function renderDetailChart(data,visibleCount){
  const allBars=data._bars||[];chartBarCount=allBars.length;
  const first=Math.max(0,allBars.length-visibleCount),bars=allBars.slice(first),patternOffset=data._chart_offset||0;
  $('chartNote').textContent=data.chart_is_preview?'周线预览：未验证交易日历，不用于正式信号':`显示 ${bars.length.toLocaleString()} / ${allBars.length.toLocaleString()} 根${data.timeframe==='1w'?'周':'日'}K · 可拖动下方滑块或滚轮缩放`;
  if(!window.echarts)return;
  detailChart=echarts.getInstanceByDom($('chart'))||echarts.init($('chart'));
  const levels=[],dates=bars.map(item=>item.date);
  const series=[{type:'candlestick',name:'K线',xAxisIndex:0,yAxisIndex:0,data:bars.map(item=>[item.open,item.close,item.low,item.high])}];
  for(const pattern of (data.patterns||[]).filter(pattern=>!isTerminalPattern(pattern)).slice(0,3)){
    if(pattern.neckline){
      const line=bars.map((_,index)=>[index,pattern.neckline.slope*(index+first-patternOffset)+pattern.neckline.intercept]);
      series.push({type:'line',name:patternName(pattern)+'颈线',xAxisIndex:0,yAxisIndex:0,data:line,showSymbol:false,lineStyle:{width:1,type:'dashed'}});
    }
    const marks=[];
    Object.entries(pattern.key_points||{}).forEach(([key,value])=>{
      if(key.endsWith('_index')&&Number.isFinite(value)){const index=value+patternOffset-first;if(index>=0&&index<bars.length)marks.push([index,pattern.key_points[key.replace('_index','')]])}
    });
    if(marks.length)series.push({type:'scatter',name:patternName(pattern)+'结构点',xAxisIndex:0,yAxisIndex:0,data:marks,symbolSize:9});
    if(pattern.invalidation)levels.push({name:'结构失效',yAxis:pattern.invalidation,lineStyle:{color:'#d45b58'}});
    (pattern.targets||[]).forEach((target,index)=>levels.push({name:index?'条件目标':'首目标',yAxis:target,lineStyle:{color:index?'#9b7acb':'#28a17a'}}));
  }
  if(levels.length)series[0].markLine={symbol:['none','none'],data:levels};
  series.push({type:'bar',name:'成交量',xAxisIndex:1,yAxisIndex:1,data:bars.map(item=>item.volume),itemStyle:{color:'#91a8b8'}});
  const quality=data.data_quality||{},verified=quality.volume_unit_verified&&quality.corporate_actions_verified;
  detailChart.setOption({
        animation:false,
        title:{text:`${data.code} ${data.timeframe} · ${data.chart_is_preview?'周线预览':verified?'量价数据已核验':'成交量/复权数据未核验'}`,left:8,textStyle:{fontSize:14}},
        tooltip:{trigger:'axis',axisPointer:{type:'cross',crossStyle:{color:'#697781'}}},
        axisPointer:{link:[{xAxisIndex:'all'}],label:{backgroundColor:'#43515a'}},
        legend:{type:'scroll',top:27,left:8,right:8,itemWidth:12,itemHeight:8,textStyle:{fontSize:11}},
        grid:[{left:120,right:70,top:66,height:'55%'},{left:120,right:70,top:'74%',height:'14%'}],
        xAxis:[{type:'category',data:dates,gridIndex:0,boundaryGap:true,axisLabel:{show:false}},{type:'category',data:dates,gridIndex:1,boundaryGap:true,gridIndex:1,axisLabel:{hideOverlap:true,fontSize:10}}],
        yAxis:[{scale:true,gridIndex:0,splitLine:{lineStyle:{color:'#e8edf0'}}},{scale:true,gridIndex:1,splitNumber:2,splitLine:{show:false}}],
        dataZoom:[
          {type:'inside',xAxisIndex:[0,1],start:0,end:100,zoomOnMouseWheel:true,moveOnMouseMove:true,moveOnMouseWheel:false,preventDefaultMouseMove:true},
          {type:'slider',xAxisIndex:[0,1],start:0,end:100,bottom:2,height:17,show:true,brushSelect:false,showDataShadow:false,borderColor:'#d7dfe5',fillerColor:'rgba(23,107,135,.16)',handleStyle:{color:'#176b87'}}
        ],
        series
      },true);
      requestAnimationFrame(()=>detailChart?.resize());
}
async function startScan(){if(scanTimer)clearInterval(scanTimer);$('scanBtn').disabled=true;$('results').innerHTML='';$('scanSummary').textContent='';$('scanStatus').textContent='正在提交全量扫描…';const params={kind:$('scanKind').value,category:$('scanCategory').value||undefined,pattern_type:$('scanPattern').value||undefined,limit:+$('scanLimit').value,timeframes:[...$('scanTimeframes').selectedOptions].map(option=>option.value)};if($('scanAfter').value)params.confirmed_after=$('scanAfter').value;if($('scanBefore').value)params.confirmed_before=$('scanBefore').value;try{const response=await fetch('/api/v1/scans',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});const job=await response.json();if(!response.ok)throw new Error(job.detail||'提交失败');$('scanStatus').textContent=`扫描任务已启动：${job.job_id}。正在逐个检查本地股票与 ETF…`;scanTimer=setInterval(async()=>{try{const current=await fetch('/api/v1/jobs/'+job.job_id).then(result=>result.json());$('scanStatus').textContent=`${current.status==='completed'?'完成':current.status==='failed'?'失败':'扫描中'} · ${Math.round((current.progress||0)*100)}% · ${current.scanned||0}/${current.total||0}`;if(current.status==='completed'){clearInterval(scanTimer);scanTimer=null;$('scanBtn').disabled=false;renderScan(current.results,current.inventory)}else if(current.status==='failed'){clearInterval(scanTimer);scanTimer=null;$('scanBtn').disabled=false;$('results').innerHTML=`<div class="error-box">扫描失败：${escapeHtml(current.error||'未知错误')}</div>`}}catch(error){$('scanStatus').textContent='读取扫描进度失败：'+error.message}},2000)}catch(error){$('scanBtn').disabled=false;$('scanStatus').textContent='扫描无法启动：'+error.message}}

$('analyzeBtn').onclick=analyze;$('scanBtn').onclick=startScan;$('chartRange').onchange=()=>applyChartRange($('chartRange').value);document.querySelectorAll('[data-chart-action]').forEach(button=>button.onclick=()=>{const action=button.dataset.chartAction;if(action==='in'||action==='out')zoomChart(action);else if(action==='all'){$('chartRange').value='all';applyChartRange('all')}else{$('chartRange').value=lastAnalysis?.timeframe==='1w'?'60':'120';applyChartRange($('chartRange').value)}});window.addEventListener('resize',()=>detailChart?.resize());
$('refreshOneBtn').onclick=()=>startRefresh({code:$('code').value.trim(),kind:$('kind').value},$('chartNote'),true);
$('refreshAllBtn').onclick=()=>{if(confirm('逐个更新本地全部股票和 ETF 行情？上游可能限流；失败标的会保留旧文件。'))startRefresh({scope:'all'},$('scanStatus'),false)};
function refreshNodes(meta){return {status:$(meta.one?'chartNote':'scanStatus'),report:$(meta.one?'refreshOneReport':'refreshAllReport')}}
function setRefreshBusy(busy){$('refreshOneBtn').disabled=busy;$('refreshAllBtn').disabled=busy}
function bindRefreshResume(meta,jobId){
  const button=refreshNodes(meta).report.querySelector('[data-resume-refresh]');
  if(!button)return;
  button.onclick=async()=>{
    button.disabled=true;
    try{
      const response=await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/resume`,{method:'POST'});
      const resumed=await response.json();if(!response.ok)throw new Error(resumed.detail||'恢复任务失败');
      setRefreshBusy(true);refreshNodes(meta).status.textContent='正在续跑未成功的标的…';watchRefreshJob({...meta,job_id:jobId});
    }catch(error){button.disabled=false;refreshNodes(meta).status.textContent='恢复任务失败：'+error.message}
  };
}
function watchRefreshJob(meta){
  const {status:statusNode,report:reportNode}=refreshNodes(meta),jobId=meta.job_id;
  setRefreshBusy(true);
  const timer=setInterval(async()=>{
    try{
      const response=await fetch('/api/v1/jobs/'+encodeURIComponent(jobId));
      const current=await response.json();if(!response.ok)throw new Error(current.detail||'读取任务失败');
      if(current.status==='completed'){
        clearInterval(timer);
        if(meta.reanalyze&&current.success_count){$('asof').value='';await analyze()}
        statusNode.textContent=window.CaisenRefreshReport.formatRefreshSummary(current,meta.one);
        reportNode.innerHTML=window.CaisenRefreshReport.renderRefreshReport({...current,job_id:jobId});
        sessionStorage.removeItem('caisen-refresh-job');setRefreshBusy(false);bindRefreshResume(meta,jobId);
        loadRecoverableRefreshJobs();
      }else if(current.status==='interrupted'||current.status==='failed'){
        clearInterval(timer);statusNode.textContent=`行情更新任务${current.status==='interrupted'?'已中断':'失败'}：${current.error||'可恢复未成功的标的'}`;
        reportNode.innerHTML=window.CaisenRefreshReport.renderRefreshReport({...current,job_id:jobId});
        setRefreshBusy(false);bindRefreshResume(meta,jobId);
        loadRecoverableRefreshJobs();
      }else statusNode.textContent=`行情更新中 ${Math.round((current.progress||0)*100)}% · ${current.completed||0}/${current.total||0} · 成功 ${current.success_count||0} · 失败 ${current.failure_count||0}`;
    }catch(error){statusNode.textContent=`行情更新服务暂不可用，任务 ${jobId} 已保留；服务恢复后将继续查询。`}
  },2000);
}
async function startRefresh(params,statusNode,reanalyze){
  const one=Boolean(params.code),reportNode=$(one?'refreshOneReport':'refreshAllReport');
  reportNode.innerHTML='';
  statusNode.textContent=one?'正在请求该标的完整历史行情…':'正在提交全市场行情更新…';
  setRefreshBusy(true);
  try{
    const response=await fetch('/api/v1/data/refresh',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
    const job=await response.json();if(!response.ok)throw new Error(job.detail||'更新任务提交失败');
    const meta={job_id:job.job_id,params,one,reanalyze};sessionStorage.setItem('caisen-refresh-job',JSON.stringify(meta));
    watchRefreshJob(meta);
  }catch(error){statusNode.textContent='行情更新无法启动：'+error.message;setRefreshBusy(false)}
}
function openRefreshJobPanel(meta){const panel=meta.one?'detail':'scan';document.querySelector(`nav button[data-panel="${panel}"]`)?.click()}
function bindRecoveryList(jobs){
  const node=$('refreshRecovery');
  node.innerHTML=window.CaisenRefreshReport.renderRefreshRecoveryList(jobs);
  node.querySelectorAll('[data-resume-refresh],[data-watch-refresh]').forEach(button=>{
    const job=jobs.find(item=>item.job_id===button.dataset.resumeRefresh||item.job_id===button.dataset.watchRefresh);
    if(!job)return;
    button.onclick=async()=>{
      const meta={job_id:job.job_id,params:{},one:job.scope==='single',reanalyze:false};
      if(button.hasAttribute('data-resume-refresh')){
        button.disabled=true;
        try{
          const response=await fetch(`/api/v1/jobs/${encodeURIComponent(job.job_id)}/resume`,{method:'POST'});
          const resumed=await response.json();if(!response.ok)throw new Error(resumed.detail||'恢复任务失败');
          sessionStorage.setItem('caisen-refresh-job',JSON.stringify(meta));
        }catch(error){button.disabled=false;node.insertAdjacentHTML('beforeend',`<p class="error-box">恢复失败：${escapeHtml(error.message)}</p>`);return}
      }else sessionStorage.setItem('caisen-refresh-job',JSON.stringify(meta));
      openRefreshJobPanel(meta);watchRefreshJob(meta);
    };
  });
}
async function loadRecoverableRefreshJobs(){
  try{const response=await fetch('/api/v1/data/refresh/jobs');const result=await response.json();
    if(!response.ok)throw new Error(result.detail||'读取恢复任务失败');bindRecoveryList(result.jobs||[])
  }catch(error){$('refreshRecovery').textContent=`暂时无法读取可恢复行情任务：${escapeHtml(error.message)}`}
}
function restoreRefreshJob(){
  try{const meta=JSON.parse(sessionStorage.getItem('caisen-refresh-job')||'null');if(meta?.job_id){
      const {status}=refreshNodes(meta);status.textContent=`正在恢复行情任务 ${meta.job_id}…`;watchRefreshJob(meta);return;
    }
  }catch(error){sessionStorage.removeItem('caisen-refresh-job')}
  loadRecoverableRefreshJobs();
}
restoreRefreshJob();
$('replayBtn').onclick=async()=>{await analyze();$('replayOutput').textContent=JSON.stringify(lastAnalysis,null,2)};
$('saveReview').onclick=async()=>{const id=$('reviewId').value;if(!id)return alert('先从形态详情或扫描结果选择形态');const response=await fetch(`/api/v1/patterns/${encodeURIComponent(id)}/review`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict:$('verdict').value,note:$('note').value})});$('reviewOutput').textContent=JSON.stringify(await response.json(),null,2)};
