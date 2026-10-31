const names={w_bottom:'W底',spring:'破底翻',bull_flag:'下飘旗形上涨中继',inverse_head_shoulders:'头肩底',triangle_bottom:'收敛三角形底部',m_top:'M头',false_breakout:'假突破',bear_flag:'上飘旗形下跌中继',head_shoulders:'头肩顶',triangle_top:'收敛三角形头部'};
const legacyNames={w_bottom_spring:'W底破底翻',false_breakout_head_shoulders:'假突破（头肩顶）'};
const actions={CONSIDER_BUY:'可考虑买入',WAIT:'等待',HOLD:'持有观察',SELL_IF_HELD:'如持有可考虑卖出',INSUFFICIENT_DATA:'数据不足'};
const reasonNames={pattern_not_buy_candidate:'该形态不属于四种买入候选',timeframe_direction_conflict:'日线与周线确认方向冲突，请分别复核',ambiguous_structure:'多个候选形态共用结构点，暂不生成买卖计划',pattern_waiting_for_breakout:'形态已成型，等待颈线或结构边界收盘突破',bull_flag_target_locked:'旗形边界已突破，但前高尚未收盘突破，测幅目标未解锁',no_confirmed_actionable_pattern:'当前没有通过确认与交易空间检查的形态',no_current_pattern:'当前周期没有识别到有效形态',confirmed_bearish_structure_sell_warning_if_held:'已确认顶部或空头结构；如有持仓，请复核退出计划',bullish_structure_exists_without_new_entry:'存在多头结构，但当前没有新的入场信号',pattern_not_confirmed_bullish:'形态尚未确认为多头结构',volume_data_unverified:'成交量单位或企业行动未核验，形态仅作研究观察',invalid_structural_stop:'无法建立有效的结构止损',no_verified_target:'没有可用于计划的目标位',structural_risk_over_7pct:'结构止损对应风险超过 7%',first_target_already_reached_or_below_entry:'首个目标已触及或低于当前参考价',entry_above_maximum_allowed_price:'当前参考价高于最高允许买入价',insufficient_cash_or_liquidity:'现金或流动性不足以形成有效委托',net_first_target_below_2R:'首个目标空间不足 2R',buy_gates_not_met:'买入条件未全部满足'};
const $=id=>document.getElementById(id);let lastAnalysis=null;let scanRows=[];let visibleScanRows=[];let scanShown=0;let scanTimer=null;let detailChart=null;let chartBarCount=0;let chartWindowBars=120;let selectedPattern=null;let chartCenter=null;

document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>{document.querySelectorAll('nav button,.panel').forEach(item=>item.classList.remove('active'));button.classList.add('active');$(button.dataset.panel).classList.add('active')});
Object.entries(names).forEach(([key,value])=>{const option=document.createElement('option');option.value=key;option.textContent=value;$('scanShapeFilter').append(option);const prefilter=document.createElement('option');prefilter.value=key;prefilter.textContent=value;$('scanPatternTypes').append(prefilter);const historical=document.createElement('option');historical.value=key;historical.textContent=value;$('historyPatterns').append(historical)});
fetchJson('/api/v1/health').then(result=>$('health').textContent=`${result.status} · ${result.rule_version} · 无下单`).catch(()=>$('health').textContent='服务未连接');

function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]))}
async function fetchJson(url,options){return window.CaisenHttpClient.fetchJson(url,options)}
function isMainBottom(pattern){return (pattern?.evidence||[]).some(item=>item.code==='main_structure'&&item.value?.role==='main')}
function mainStructure(pattern){return (pattern?.evidence||[]).find(item=>item.code==='main_structure')?.value||{}}
function isSelectedMain(pattern){return isMainBottom(pattern)&&(pattern.evidence||[]).some(item=>item.code==='main_selection'&&item.value?.selected===true)}
function patternName(pattern){const base=names[pattern.pattern_type]||legacyNames[pattern.pattern_type]||pattern.pattern_type||'暂无形态';return isMainBottom(pattern)?`主结构 · ${base}`:base}
function stateLabel(pattern){let label;if(pattern.state==='forming')label=`${patternName(pattern)}，未确认`;else if(pattern.state==='confirmed')label=`${patternName(pattern)}，价格确认${isMainBottom(pattern)&&!pattern.volume_verified?'，量能仅作辅助':pattern.volume_verified?'':'，量能未核验'}`;else if(pattern.state==='invalidated')label=`${patternName(pattern)}，已失效`;else if(pattern.state==='target_reached')label=`${patternName(pattern)}，目标已达`;else if(pattern.state==='followthrough_confirmed')label=`${patternName(pattern)}，箱体后续边界已确认（历史）`;else label=`${patternName(pattern)}，${pattern.state||'状态未知'}`;if(pattern.flags?.includes('ambiguous_structure'))label+='，结构存在歧义';return label}
function isTerminalPattern(pattern){return ['invalidated','target_reached','followthrough_confirmed'].includes(pattern.state)}
function actionLabel(action){return actions[action]||action||'状态未知'}
function reasonLabel(reason){return reasonNames[reason]||reason||'条件暂未满足'}
function isInternalEvidence(item){const code=item.code||'',value=item.value;return Boolean(['competing_topology_labels','confirmed_date','confirmation_index','break_index','structure_uncertainty'].includes(code)||code.endsWith('_pattern_id')||/^inner_spring_(break_index|confirmation_index)$/.test(code)||/cache|position_history|^_/.test(code)||(value&&typeof value==='object'&&(value._position_history||value._cache)))}
function dataWarningLabel(warning){let value=String(warning??'');let timeframe='';const status=value.match(/^(1d|1w) data status:\s*(.*)$/);if(status){timeframe=status[1]==='1d'?'日线：':'周线：';value=`data status: ${status[2]}`}else{const prefix=value.match(/^(1d|1w):\s*/);if(prefix){timeframe=prefix[1]==='1d'?'日线：':'周线：';value=value.slice(prefix[0].length)}}if(value==='volume unit is not verified for this file hash'||value==='volume unit or corporate-action adjustment is unverified')value='成交量单位或复权口径尚未核验';else if(value==='corporate-action adjustment is not verified for this file hash')value='复权口径尚未核验';else if(value==='instrument registry is not bound to the current file hash')value='标的交易制度资料未与当前行情文件绑定';else if(value==='verified exchange calendar unavailable; weekly bars are preview-only')value='缺少已核验的交易日历，周线仅作预览';else if(value==='no completed weekly bars; latest week is preview-only')value='尚无已完成周线，当前周仅作预览';else if(value==='data status: insufficient')value='数据量不足，暂不用于正式判断';else {const split=value.match(/^unverified price discontinuity split (\d+) segment\(s\)$/);if(split)value=`发现未核验的价格断点，已分成 ${split[1]} 段分析`}return timeframe+value}
function dataWarningsMarkup(warnings){const items=[...new Set((warnings||[]).map(dataWarningLabel))];if(!items.length)return '';return `<details class="data-quality"><summary>数据待核验 · ${items.length} 项（分析仅作参考）</summary><ul>${items.map(item=>`<li>${escapeHtml(item)}</li>`).join('')}</ul></details>`}
function fmt(value,digits=2){return Number.isFinite(Number(value))?Number(value).toFixed(digits):'—'}
function planMarkup(plan){if(!plan)return '';const reasons=(plan.reason_codes||[]).map(reasonLabel).map(escapeHtml).join('；')||'风险与目标条件通过';const reference=plan.main_bottom_reference?`<br>止损依据：冻结颈线 ${fmt(plan.structure_level)} · 实际底部参考 ${fmt(plan.main_bottom_reference.support_lower)}–${fmt(plan.main_bottom_reference.support_upper)}`:'';return `<div class="plan-summary">计划参考：${escapeHtml(plan.decision)} · 参考价 ${fmt(plan.entry_price)} · 结构止损 ${fmt(plan.stop)} · 首目标 ${fmt(plan.targets?.[0])} · 最高允许价 ${fmt(plan.highest_allowed_entry)} · 风险 ${fmt((plan.planned_risk_pct||0)*100)}% · R ${fmt(plan.reward_risk)}${reference}<br>${reasons}${dataWarningsMarkup(plan.data_warnings)}</div>`}
function evidenceLabel(item){let code=item.code||'',value=item.value;if(code==='main_structure'&&value&&typeof value==='object'){const duration=value.scale==='5y'?'日线近5自然年':value.scale==='10y'?'周线近10自然年':value.scale||'长周期';const support=value.support_lower!==undefined&&value.support_upper!==undefined?`支撑 ${fmt(value.support_lower)}–${fmt(value.support_upper)}`:'';const neckline=typeof value.neckline==='object'&&value.neckline?`颈线 ${fmt(value.neckline.slope)}·x+${fmt(value.neckline.intercept)}`:value.neckline!==undefined?`颈线 ${fmt(value.neckline)}`:'';return `主结构：${duration} · ${support} · ${neckline} · ${value.signal_eligible?'价格确认可参与':'等待有效价格确认'}${value.eligibility_reason?`（${value.eligibility_reason}）`:''}`}if(code==='neckline_tests'&&Array.isArray(value))return `颈线测试：${value.map(test=>`${test.group_id||'分组未知'} ${test.pivot_date||'日期未知'} / 知晓 ${test.known_date||'未记录'} @ ${fmt(test.level)}（${test.valid?'有效':'无效'}）`).join('；')}`;if(code==='main_children'&&Array.isArray(value))return `内含子形态：${value.map(child=>`${names[child.pattern_type]||child.pattern_type} ${child.anchor_start_date||''}–${child.anchor_end_date||''}`).join('；')||'无'}`;if(code==='support_tests'&&Array.isArray(value))return `支撑测试：${value.map(test=>`${test.date||test.pivot_date||'日期未知'} @ ${fmt(test.level??test.price)} ${test.group_id||''}`).join('；')||'无'}`;if(code==='neckline_breakout'&&value&&typeof value==='object')return `颈线收盘确认：${value.date||'日期未知'} 收盘 ${fmt(value.close)} · 冻结颈线 ${fmt(value.frozen_neckline)} · 要求收盘 ${fmt(value.required_close)}`;if(code==='weak_crossings'&&Array.isArray(value))return `颈线弱穿越：${value.map(item=>`${item.date||'日期未知'} 收盘 ${fmt(item.close)} / 要求 ${fmt(item.required_close)}`).join('；')||'无'}`;if(code==='volume_support'&&value&&typeof value==='object')return `量能辅助证据：${value.status==='supportive'?'支持':value.status==='unknown'?'未记录':'不支持'}${value.ratio===null||value.ratio===undefined?'':` · 比率 ${fmt(value.ratio)}`}${value.date?` · ${value.date}`:''}（不作买入门槛）`;if(code==='main_bottom_reference'&&value&&typeof value==='object')return `实际底部参考 ${fmt(value.support_lower)}–${fmt(value.support_upper)} · ${value.first_trough_date||'日期未知'} 至 ${value.latest_anchor_date||'日期未知'} · 冻结颈线 ${fmt(value.neckline_at_observed)}（观察日 ${value.observed_date||'未知'}）`;if(code==='main_selection'&&value&&typeof value==='object')return `${value.selected?'已选主结构':'未选主结构'} · 截至 ${value.decided_at||'未知日期'} · ${value.reason||'排序原因未记录'}`;if(code==='execution_suppression'&&value?.suppressed)return `已纳入上层主结构 ${value.parent_id||''}，不单独生成买入计划`;const labels={pivot_topology:'结构点顺序',neckline:'颈线',neckline_left:'左侧颈线',neckline_right:'右侧颈线',close_confirmation:'收盘突破确认',close_reclaim:'收盘收复',reclaim_level:'收复价位',break_level:'越界价位',volume_verified:'量能已核验',box_lower:'箱体下缘',box_upper:'箱体上缘',box_first_touch:'箱体起始触点',box_last_touch:'箱体末端触点',box_width_atr:'箱体宽度（ATR）',followthrough_confirmed:'后续边界确认',target_reached:'首目标触达',structure_invalidation:'结构极值失效',outer_w_invalidation:'外层W底失效',neckline_projection_invalidation:'颈线外推失效',neckline_invalidation:'颈线失效',w_geometry_metrics:'W底几何特征',structure_duration:'结构时间',first_valid_break:'首次有效突破',low_position:'历史价格位置',prepattern_drawdown:'形态前高回撤',pattern_quality_pass:'形态质量门槛'};let prefix='';if(code.startsWith('inner_spring_')){prefix='内层';code=code.slice('inner_spring_'.length)}const percent=number=>number===null||number===undefined||!Number.isFinite(Number(number))?'未记录':`${(Number(number)*100).toFixed(1)}%`;if(code==='pivot_topology'&&Array.isArray(value))value=value.map(point=>point==='high'?'高点':point==='low'?'低点':String(point)).join(' → ');else if(code==='w_geometry_metrics'&&value&&typeof value==='object')value=`左段 ${value.first_leg_bars} 根 · 右段 ${value.second_leg_bars} 根 · 时长比 ${value.leg_duration_ratio} · 两底差 ${value.trough_gap_pct}%`;else if(code==='structure_duration'&&value&&typeof value==='object'){const hasBreak=Boolean(value.first_valid_break_date),end=value.first_valid_break_date||value.observed_date||'观察日未知';value=`结构起点 ${value.start_date||'日期未知'} → ${hasBreak?'首次突破':'观察日'} ${end} · 持续 ${value.duration_days??'—'} 天 · ${value.eligible?'超过3自然月':'未超过3自然月'}${!hasBreak?'，尚未突破':''}`}else if(code==='first_valid_break')value=value||'未发生';else if(code==='low_position'&&value&&typeof value==='object'){const branches=[];if(value.position_pass)branches.push('历史价格低位');if(value.drawdown_pass)branches.push('前高回撤');value=`区间位置 ${percent(value.position_ratio)}（门槛≤30%） · 通过分支：${branches.join('、')||'无'}`}else if(code==='prepattern_drawdown'&&value&&typeof value==='object')value=`前高 ${value.high===null||value.high===undefined?'未记录':fmt(value.high)} · 回撤 ${percent(value.drawdown)}（门槛≥${percent(value.threshold)}，${value.passed?'通过':'未通过'}）`;else if(code==='pattern_quality_pass')value=value?'通过':'未通过';else if((code==='followthrough_confirmed'||code==='target_reached')&&value&&typeof value==='object')value=`${value.date||'日期未知'} · 价位 ${fmt(value.level??value.price)}`;else if((code==='structure_invalidation'||code==='outer_w_invalidation'||code==='neckline_projection_invalidation')&&value&&typeof value==='object')value=`${value.date||'日期未知'} · ${code==='neckline_projection_invalidation'?'颈线':'失效价位'} ${fmt(value.level)}`;else if(typeof value==='boolean')value=value?'是':'否';else if(value&&typeof value==='object'){if(value._position_history||value._cache)return `${prefix}内部数据`;value=JSON.stringify(value)}else if(typeof value==='number')value=fmt(value);return `${prefix}${labels[code]||code}：${String(value??'—')}`}
function patternCard(pattern){const pivotDate=((pattern.evidence||[]).find(item=>item.code==='confirmed_date')||{}).value;const confirmed=pattern.confirmation_date||pivotDate||'—';const confirmationLabel=pattern.confirmation_date?'价格确认':'结构点确认';const overlaps=(pattern.evidence||[]).find(item=>item.code==='competing_topology_labels')?.value||[];const selectedTag=isSelectedMain(pattern)?'<span class="tag main-selected">已选主结构</span>':'';const suppression=(pattern.evidence||[]).find(item=>item.code==='execution_suppression')?.value;const nestedNote=suppression?.suppressed?'<div class="muted">该结构已包含在上层主形态中，不单独生成买入计划。</div>':'';const warning=pattern.flags?.includes('ambiguous_structure')?`<div class="pattern-ambiguity-warning"><b>候选形态有重叠，暂不生成买卖计划</b>${overlaps.length?`<span>共享同一组结构点：${overlaps.map(patternType=>escapeHtml(names[patternType]||patternType)).join('、')}</span>`:''}</div>`:'';const evidence=(pattern.evidence||[]).filter(item=>!isInternalEvidence(item)).map(item=>escapeHtml(evidenceLabel(item))).join(' · ');return `<article class="card pattern-card" data-pattern-id="${escapeHtml(pattern.pattern_id||'')}"><div><b>${escapeHtml(pattern.code)} · ${escapeHtml(patternName(pattern))}</b> ${selectedTag}<span class="tag">${escapeHtml(pattern.timeframe)} / ${escapeHtml(stateLabel(pattern))}</span><div class="muted">形成 ${escapeHtml(pattern.formed_at||'—')} · ${confirmationLabel} ${escapeHtml(confirmed)} · 引擎 ${escapeHtml(pattern.rule_version||'—')} · 规则集 ${escapeHtml(pattern.source_rule_version||'—')} · 数据 ${escapeHtml(pattern.data_version||'—')}</div>${nestedNote}${warning}<div class="evidence">${evidence}</div>${planMarkup(pattern.trade_plan)}</div><button type="button" data-show-pattern="${escapeHtml(pattern.pattern_id||'')}">查看该形态</button></article>`}

function miniCandles(bars){if(!bars?.length)return '<span class="mini-empty">暂无K线</span>';const width=184,height=48,pad=3,step=(width-pad*2)/bars.length;const lows=bars.map(bar=>Number(bar.low)),highs=bars.map(bar=>Number(bar.high));const min=Math.min(...lows),max=Math.max(...highs),range=max-min||1;const y=value=>pad+(max-value)/range*(height-pad*2);let shapes='';bars.forEach((bar,index)=>{const x=pad+index*step+step/2,open=y(Number(bar.open)),close=y(Number(bar.close)),up=Number(bar.close)>=Number(bar.open),color=up?'#d84a4a':'#18866b',body=Math.max(1,Math.abs(close-open));shapes+=`<line x1="${x}" y1="${y(Number(bar.high))}" x2="${x}" y2="${y(Number(bar.low))}" stroke="${color}" stroke-width="1"/><rect x="${x-step*.28}" y="${Math.min(open,close)}" width="${Math.max(1,step*.56)}" height="${body}" fill="${color}"/>`});return `<svg class="mini-candles" viewBox="0 0 ${width} ${height}" role="img" aria-label="最近${bars.length}根日K线">${shapes}</svg>`}
function scanCard(row){const plan=row.selected_plan;const activePatterns=(row.patterns||[]).filter(pattern=>!isTerminalPattern(pattern));const main=plan?row.patterns.find(pattern=>pattern.pattern_id===plan.pattern_id):(activePatterns[0]||row.patterns?.[0]);const label=main?.display_label||main?.pattern_type&&stateLabel(main)||'暂无当前形态';const warnings=[...(row.data_warnings||[]),...(plan?.data_warnings||[])];const quality=warnings.length?'数据待核验':'行情可用';const price=`${fmt(row.latest_close)} · ${row.latest_date||''}`;const historicalNote=row.analysis_window?.historical?`<span class="historical-scan-note">历史区间截至 ${escapeHtml(row.analysis_window.to||row.latest_date||'未知')}；打开详情将加载最新行情</span>`:'';return `<article class="scan-row" data-code="${escapeHtml(row.code)}" data-kind="${escapeHtml(row.kind)}" tabindex="0" role="link" aria-label="查看${escapeHtml(row.name||row.code)}详情"><div class="scan-main"><div class="scan-title"><b>${escapeHtml(row.name||row.code)}</b><span class="muted">${escapeHtml(row.code)} · ${row.kind==='etf'?'ETF':'股票'}</span></div><span class="action-pill action-${escapeHtml(row.action||'WAIT')}">${escapeHtml(actionLabel(row.action))}</span><span class="pattern-label">${escapeHtml(label)}</span>${historicalNote}<span class="muted">${escapeHtml(reasonLabel(row.reason_codes?.[0]||row.explanation||''))}</span>${dataWarningsMarkup(warnings)}</div><div class="scan-chart">${miniCandles(row.mini_bars)}<span class="muted">${escapeHtml(price)}</span></div><div class="scan-metrics">${plan?`<b>风险 ${fmt((plan.planned_risk_pct||0)*100)}%</b><b>首目标 ${fmt(plan.targets?.[0])}</b><b>${fmt(plan.reward_risk)}R</b>`:`<b>${activePatterns.length} 个当前结构</b>${row.patterns?.length>activePatterns.length?`<span class="muted">另有 ${row.patterns.length-activePatterns.length} 个结束结构</span>`:''}<span class="muted">${escapeHtml(quality)}</span>`}<button class="detail-link">查看详情 →</button></div></article>`}
function renderMore(){if(!visibleScanRows.length){$('results').innerHTML='<div class="scan-empty">没有标的符合当前筛选。可重置筛选查看完整扫描结果。</div>';return}const end=Math.min(visibleScanRows.length,scanShown+100);$('results').insertAdjacentHTML('beforeend',visibleScanRows.slice(scanShown,end).map(scanCard).join(''));scanShown=end;const more=$('loadMore');if(more)more.remove();if(scanShown<visibleScanRows.length)$('results').insertAdjacentHTML('beforeend',`<button id="loadMore" class="load-more">再显示 100 个（剩余 ${visibleScanRows.length-scanShown}）</button>`);$('loadMore')?.addEventListener('click',renderMore);document.querySelectorAll('.scan-row:not([data-bound])').forEach(row=>{row.dataset.bound='1';const open=()=>openTicker(row.dataset.code,row.dataset.kind);row.addEventListener('click',open);row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}})})}
function openTicker(code,kind){$('code').value=code;$('kind').value=kind;$('tf').value='1d';document.querySelector('[data-panel=detail]').click();analyze()}
function applyScanFilters(){const action=$('scanActionFilter').value;const patternTypes=[...$('scanShapeFilter').selectedOptions].map(option=>option.value);visibleScanRows=CaisenScanFilters.filterScanRows(scanRows,{action,patternTypes});scanShown=0;$('results').innerHTML='';const actionCounts=scanRows.reduce((counts,row)=>{counts[row.action]=(counts[row.action]||0)+1;return counts},{ });$('scanFilterSummary').textContent=`筛选后 ${visibleScanRows.length} / ${scanRows.length} 个标的 · 可考虑买入 ${actionCounts.CONSIDER_BUY||0} · 等待 ${actionCounts.WAIT||0} · 持有观察 ${actionCounts.HOLD||0} · 卖出提示 ${actionCounts.SELL_IF_HELD||0} · 数据不足 ${actionCounts.INSUFFICIENT_DATA||0}`;$('buyOnlyBtn').setAttribute('aria-pressed',String(action==='CONSIDER_BUY'));$('buyOnlyBtn').classList.toggle('selected',action==='CONSIDER_BUY');renderMore()}
function renderScan(rows,inventory){scanRows=rows||[];$('scanResultFilters').hidden=false;$('scanActionFilter').value='';[...$('scanShapeFilter').options].forEach(option=>option.selected=false);(scanSettings.pattern_types||[]).forEach(type=>{const option=[...$('scanShapeFilter').options].find(item=>item.value===type);if(option)option.selected=true});const window=inventory?.requested_window||{};const range=`${window.from||'最早可用'} 至 ${window.to||'最新可用'}`;const effectiveEnds=scanRows.map(row=>row.analysis_window?.to).filter(Boolean).sort();const latestAnalyzed=effectiveEnds.at(-1);const historical=scanRows.some(row=>row.analysis_window?.historical);$('scanSummary').innerHTML=`<div class="scan-summary"><b>扫描完成</b> · 请求区间 ${escapeHtml(range)} · 实际分析行情截至 ${escapeHtml(latestAnalyzed||'无')}${historical?' · 历史区间结果，请按每行行情日期理解':''} · ${scanRows.length} 个标的 · 股票 ${inventory?.stock_count??'—'} · ETF ${inventory?.etf_count??'—'} · 排序保持买入优先。行动数量见筛选栏。</div>`;applyScanFilters()}

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
async function analyze(selectedId=null){
  if(typeof selectedId!=='string')selectedId=null;
  selectedPattern=null;chartCenter=null;
  const query=new URLSearchParams({code:$('code').value,kind:$('kind').value,timeframe:$('tf').value});
  $('detailRows').innerHTML='<p>正在分析…</p>';
  try{
    const data=await fetchJson('/api/v1/analysis?'+query);
    lastAnalysis=data;
    const details=data.decision?.patterns||data.patterns||[],plan=data.decision?.selected_plan;
    const patterns=details.map(pattern=>({...pattern,trade_plan:pattern.pattern_id===plan?.pattern_id?plan:(data.patterns||[]).find(item=>item.pattern_id===pattern.pattern_id)?.trade_plan}));
    const activePatterns=patterns.filter(pattern=>!isTerminalPattern(pattern));
    const completedPatterns=patterns.filter(isTerminalPattern);
    const mainParents=patterns.filter(isSelectedMain);
    const mainParentIds=new Set(mainParents.map(pattern=>String(pattern.pattern_id)));
    const mainChildren=patterns.filter(pattern=>mainParentIds.has(String(pattern.parent_id)));
    const groupedIds=new Set([...mainParentIds,...mainChildren.map(pattern=>String(pattern.pattern_id))]);
    const mainCards=mainParents.map(patternCard).join('');
    const childCards=mainChildren.length?`<details class="main-children"><summary>已纳入主结构的子形态（${mainChildren.length}）</summary>${mainChildren.map(patternCard).join('')}</details>`:'';
    const otherCards=activePatterns.filter(pattern=>!groupedIds.has(String(pattern.pattern_id))).map(patternCard).join('');
    const activeCards=(mainCards+childCards+otherCards)||'<p>当前没有有效形态。周线若缺少日历会显示预览K线，但不产生正式周线信号。</p>';
    const completedCards=completedPatterns.length?`<details class="completed-patterns"><summary>已结束的历史形态（${completedPatterns.length}）</summary>${completedPatterns.map(patternCard).join('')}</details>`:'';
    $('detailRows').innerHTML=decisionMarkup(data.decision)+activeCards+completedCards;
    if(data.decision?.timeframe_conflict)$('detailRows').insertAdjacentHTML('afterbegin','<p class="conflict-banner">日线与周线方向冲突：请分别查看两周期结构。</p>');
    const bars=data._bars||[];chartBarCount=bars.length;chartWindowBars=chartVisibleCount($('chartRange').value,data.timeframe,bars.length);
    renderDetailChart(data,chartWindowBars);
    if(selectedId)await showPattern(selectedId);
  }catch(error){$('detailRows').innerHTML=`<div class="error-box">分析失败：${escapeHtml(error.message)}</div>`}
}
function renderDetailChart(data,visibleCount){
  const allBars=data._bars||[];chartBarCount=allBars.length;
  const first=chartCenter===null?Math.max(0,allBars.length-visibleCount):Math.max(0,Math.min(allBars.length-visibleCount,Math.floor(chartCenter-visibleCount/2)));
  const bars=allBars.slice(first,first+visibleCount),patternOffset=data._chart_offset||0;
  $('chartNote').textContent=data.chart_is_preview?'周线预览：未验证交易日历，不用于正式信号':`显示 ${bars.length.toLocaleString()} / ${allBars.length.toLocaleString()} 根${data.timeframe==='1w'?'周':'日'}K · 可拖动下方滑块或滚轮缩放`;
  if(!window.echarts)return;
  detailChart=echarts.getInstanceByDom($('chart'))||echarts.init($('chart'));
  const levels=[],dates=bars.map(item=>item.date);
  const series=[{type:'candlestick',name:'K线',xAxisIndex:0,yAxisIndex:0,data:bars.map(item=>[item.open,item.close,item.low,item.high])}];
  for(const pattern of (selectedPattern?[selectedPattern]:(data.patterns||[]).filter(pattern=>!isTerminalPattern(pattern)).slice(0,3))){
    const shapeOffset=Number.isInteger(pattern.chart_offset)?pattern.chart_offset:patternOffset;
    if(pattern.neckline){
      const line=bars.map((_,index)=>[index,pattern.neckline.slope*(index+first-shapeOffset)+pattern.neckline.intercept]);
      series.push({type:'line',name:patternName(pattern)+'颈线',xAxisIndex:0,yAxisIndex:0,data:line,showSymbol:false,lineStyle:{width:1,type:'dashed'}});
    }
    const marks=[];
    Object.entries(pattern.key_points||{}).forEach(([key,value])=>{
      if(key.endsWith('_index')&&Number.isFinite(value)){const index=value+shapeOffset-first;if(index>=0&&index<bars.length)marks.push([index,pattern.key_points[key.replace('_index','')]])}
    });
    if(marks.length)series.push({type:'scatter',name:patternName(pattern)+'结构点',xAxisIndex:0,yAxisIndex:0,data:marks,symbolSize:9});
    if(pattern.invalidation)levels.push({name:'结构失效',yAxis:pattern.invalidation,lineStyle:{color:'#d45b58'}});
    (pattern.targets||[]).forEach((target,index)=>levels.push({name:index?'条件目标':'首目标',yAxis:target,lineStyle:{color:index?'#9b7acb':'#28a17a'}}));
  }
  if(selectedPattern){
    const geometry=CaisenPatternChart.patternGeometry(data,selectedPattern);
    const visible=p=>p.index>=first&&p.index<first+bars.length;
    series.splice(1,series.length-1);
    const anchorData=geometry.anchors.filter(visible).map((p,index)=>({value:[p.index-first,p.price],name:p.name,label:{position:index%2?'top':'bottom'}}));
    series.push({type:'line',name:patternName(selectedPattern)+'形状',data:anchorData,showSymbol:false,lineStyle:{width:2,color:'#d28a18'},z:5});
    series.push({type:'scatter',name:'结构点',data:anchorData,symbol:'circle',symbolSize:10,itemStyle:{color:'#c98212'},label:{show:true,formatter:p=>p.name,color:'#7d4f05',backgroundColor:'rgba(255,255,255,.92)',padding:3},z:7});
    for(const boundary of geometry.boundaries)series.push({type:'line',name:boundary.name,data:boundary.points.filter(visible).map(p=>[p.index-first,p.price]),showSymbol:false,lineStyle:{type:'dashed',color:'#176b87',width:2}});
    const markers=geometry.markers.filter(visible);
    series.push({type:'scatter',name:'确认与结束',data:markers.map((p,index)=>({value:[p.index-first,p.price],name:p.name,label:{position:index%2?'bottom':'top'}})),symbol:'pin',symbolSize:35,itemStyle:{color:'#a53d62'},label:{show:true,position:'top',formatter:p=>p.name},z:6});
    $('chartNote').textContent=`${patternName(selectedPattern)} · ${stateLabel(selectedPattern)} · ${allBars[geometry.start].date} 至 ${allBars[geometry.end].date} · 标记为原始结构点`;
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
let scanSettings={};
async function startScan(){if(scanTimer)clearInterval(scanTimer);const kinds=[$('scanStock').checked?'stock':null,$('scanEtf').checked?'etf':null].filter(Boolean);const dateFrom=$('scanDateFrom').value,dateTo=$('scanDateTo').value,limit=$('scanLimit').value.trim();if(!kinds.length){$('scanStatus').textContent='请至少选择股票或 ETF。';return}if(dateFrom&&dateTo&&dateFrom>dateTo){$('scanStatus').textContent='起始日期不能晚于截止日期。';return}scanSettings={kinds,pattern_types:[...$('scanPatternTypes').selectedOptions].map(option=>option.value)};$('scanBtn').disabled=true;scanRows=[];visibleScanRows=[];scanShown=0;$('results').innerHTML='';$('scanResultFilters').hidden=true;$('scanSummary').textContent='';$('scanFilterSummary').textContent='';$('scanStatus').textContent='正在提交扫描…';const params={timeframes:[...$('scanTimeframes').selectedOptions].map(option=>option.value),kinds};if(limit)params.limit=Number(limit);if(dateFrom)params.date_from=dateFrom;if(dateTo)params.date_to=dateTo;try{const job=await fetchJson('/api/v1/scans',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});$('scanStatus').textContent=`扫描任务已启动：${job.job_id}。正在逐个检查所选本地行情…`;scanTimer=setInterval(async()=>{try{const current=await fetchJson('/api/v1/jobs/'+encodeURIComponent(job.job_id));$('scanStatus').textContent=`${current.status==='completed'?'完成':current.status==='partial_failed'?'部分失败':current.status==='failed'?'失败':'扫描中'} · ${Math.round((current.progress||0)*100)}% · ${current.scanned||0}/${current.total||0}`;if(current.status==='completed'||current.status==='partial_failed'){clearInterval(scanTimer);scanTimer=null;$('scanBtn').disabled=false;renderScan(current.results,current.inventory);if(current.status==='partial_failed')$('scanStatus').textContent+=' · 部分标的失败，完整保存结果可在历史结果中查询'}else if(current.status==='failed'){clearInterval(scanTimer);scanTimer=null;$('scanBtn').disabled=false;$('results').innerHTML=`<div class="error-box">扫描失败：${escapeHtml(current.error||'未知错误')}</div>`}}catch(error){$('scanStatus').textContent='读取扫描进度失败：'+error.message}},2000)}catch(error){$('scanBtn').disabled=false;$('scanStatus').textContent='扫描无法启动：'+error.message}}

$('analyzeBtn').onclick=analyze;$('scanBtn').onclick=startScan;$('backtestBtn').onclick=startBacktest;$('chartRange').onchange=()=>applyChartRange($('chartRange').value);document.querySelectorAll('[data-chart-action]').forEach(button=>button.onclick=()=>{const action=button.dataset.chartAction;if(action==='in'||action==='out')zoomChart(action);else if(action==='all'){$('chartRange').value='all';applyChartRange('all')}else{selectedPattern=null;chartCenter=null;document.querySelectorAll('.pattern-card.selected').forEach(card=>{card.classList.remove('selected');card.querySelector('[data-show-pattern]')?.setAttribute('aria-pressed','false')});$('chartRange').value=lastAnalysis?.timeframe==='1w'?'60':'120';applyChartRange($('chartRange').value)}});window.addEventListener('resize',()=>detailChart?.resize());
$('scanActionFilter').onchange=applyScanFilters;$('scanShapeFilter').onchange=applyScanFilters;$('buyOnlyBtn').onclick=()=>{$('scanActionFilter').value=$('scanActionFilter').value==='CONSIDER_BUY'?'':'CONSIDER_BUY';applyScanFilters()};$('resetScanFilters').onclick=()=>{$('scanActionFilter').value='';[...$('scanShapeFilter').options].forEach(option=>option.selected=false);applyScanFilters()};
$('historyPage').textContent='';let historyPage=1,historyRunId='';
async function loadHistoryRuns(){
  try{
    const data=await fetchJson('/api/v1/scans?page=1'),select=$('historyRun'),previous=select.value;
    select.innerHTML='';
    if(!data.items?.length){select.innerHTML='<option value="">尚无历史扫描任务</option>';historyRunId='';$('historySummary').textContent='完成或部分失败的扫描会保存在这里。';$('historyResults').innerHTML='';return}
    data.items.forEach(run=>{
      const option=document.createElement('option'),params=run.params||{};
      option.value=run.run_id;
      const range=params._legacy_import?'旧任务参数未记录':`${params.date_from||'最早'} 至 ${params.date_to||'最新'}`;
      option.textContent=`${run.created_at} · ${run.status} · ${range} · ${run.scanned}/${run.total}`;
      select.append(option);
    });
    select.value=data.items.some(run=>run.run_id===previous)?previous:data.items[0].run_id;
    historyRunId=select.value;await loadHistoryResults(1);
  }catch(error){$('historySummary').textContent=`无法读取历史扫描：${error.message}`}
}
async function loadHistoryResults(page=1){
  historyRunId=$('historyRun').value;if(!historyRunId)return;historyPage=page;
  const query=new URLSearchParams({page:String(page),sort:$('historySort').value,order:$('historyOrder').value});
  const q=$('historySearch').value.trim(),kind=$('historyKind').value,category=$('historyCategory').value.trim(),action=$('historyAction').value;
  const patterns=[...$('historyPatterns').selectedOptions].map(item=>item.value),periods=[...$('historyTimeframes').selectedOptions].map(item=>item.value);
  if(q)query.set('q',q);if(kind)query.set('kind',kind);if(category)query.set('category',category);if(action)query.set('action',action);
  if(patterns.length)query.set('pattern_types',patterns.join(','));if(periods.length)query.set('timeframes',periods.join(','));
  $('historyResults').innerHTML='<p class="muted">正在读取保存的扫描结果…</p>';
  try{
    const [data,runs]=await Promise.all([fetchJson(`/api/v1/scans/${encodeURIComponent(historyRunId)}/results?${query}`),fetchJson('/api/v1/scans?page=1')]);
    const run=runs.items.find(item=>item.run_id===historyRunId),params=run?.params||{},legacy=Boolean(params._legacy_import);
    const window=legacy?'未记录':`${params.date_from||'最早可用'} 至 ${params.date_to||'最新可用'}`;
    const status={completed:'完成',partial_failed:'部分失败',failed:'全部失败',interrupted:'已中断',running:'扫描中',queued:'排队中'}[run?.status]||run?.status||'未知';
    const failures=typeof run?.inventory?.failed_count==='number'?`失败 ${run.inventory.failed_count}`:'失败数未记录';
    const legacyNote=legacy?'旧扫描未保存完整参数和版本信息。':'';
    const ruleVersion=run?.rule_version?`规则 ${escapeHtml(run.rule_version)}`:'规则版本未记录';
    $('historySummary').innerHTML=`<div class="scan-summary"><b>${escapeHtml(status)}</b> · 任务 ${escapeHtml(historyRunId)} · 请求区间 ${escapeHtml(window)} · 扫描进度 ${run?.scanned??0}/${run?.total??0} · ${escapeHtml(failures)} · 保存结果 ${data.total} · ${ruleVersion}<p>${escapeHtml(legacyNote)} ${escapeHtml(data.detail_notice||'')}</p></div>`;
    const rows=data.items||[];
    $('historyResults').innerHTML=rows.length?rows.map(row=>{
      const shapes=(row.patterns||[]).map(pattern=>`${names[pattern.pattern_type]||pattern.pattern_type||''} ${pattern.timeframe||''}`).join('、')||'无保存形态';
      return `<article class="history-result-row" data-code="${escapeHtml(row.code)}" data-kind="${escapeHtml(row.kind)}" tabindex="0" role="link"><b>${escapeHtml(row.name||row.code)}</b><span>${escapeHtml(row.code)} · ${row.kind==='etf'?'ETF':'股票'}</span><span class="action-pill action-${escapeHtml(row.action||'WAIT')}">${escapeHtml(actionLabel(row.action))}</span><span class="muted">${escapeHtml(row.category||'未分类')} · ${escapeHtml(shapes)} · 分析行情截至 ${escapeHtml(row.analysis_window?.to||row.latest_date||'未知')} · ${fmt(row.latest_close)} · 打开详情将加载最新行情</span></article>`
    }).join(''):'<div class="scan-empty">没有符合筛选条件的保存结果。</div>';
    $('historyPage').textContent=`第 ${page} 页 · 共 ${data.total} 条 · 每页 50 条`;$('historyPrev').disabled=page<=1;$('historyNext').disabled=page*50>=data.total;
    document.querySelectorAll('.history-result-row:not([data-bound])').forEach(row=>{row.dataset.bound='1';const open=()=>openTicker(row.dataset.code,row.dataset.kind);row.addEventListener('click',open);row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}})});
  }catch(error){$('historyResults').innerHTML=`<div class="error-box">无法读取保存结果：${escapeHtml(error.message)}</div>`}
}
document.querySelectorAll('[data-scan-view]').forEach(button=>button.onclick=()=>{document.querySelectorAll('[data-scan-view],.scan-view').forEach(item=>item.classList.remove('active'));button.classList.add('active');$(button.dataset.scanView).classList.add('active');if(button.dataset.scanView==='scanHistory')loadHistoryRuns()});$('loadHistoryBtn').onclick=loadHistoryRuns;$('historyRun').onchange=()=>loadHistoryResults(1);$('queryHistoryBtn').onclick=()=>loadHistoryResults(1);$('historyPrev').onclick=()=>loadHistoryResults(historyPage-1);$('historyNext').onclick=()=>loadHistoryResults(historyPage+1);
$('refreshOneBtn').onclick=()=>startRefresh({code:$('code').value.trim(),kind:$('kind').value},$('chartNote'),true);
$('refreshAllBtn').onclick=()=>{if(confirm('逐个更新本地全部股票和 ETF 行情？上游可能限流；失败标的会保留旧文件。'))startRefresh({scope:'all'},$('scanStatus'),false)};
function refreshNodes(meta){return {status:$(meta.one?'chartNote':'scanStatus'),report:$(meta.one?'refreshOneReport':'refreshAllReport')}}
function ledgerTable(title,rows,columns){if(!rows?.length)return `<section class="ledger-section"><h3>${title}</h3><p class="muted">暂无记录</p></section>`;return `<section class="ledger-section"><h3>${title}（${rows.length}）</h3><div class="table-scroll"><table><thead><tr>${columns.map(([label])=>`<th>${label}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${columns.map(([,key,format])=>`<td>${escapeHtml(format?format(row[key],row):row[key]??'—')}</td>`).join('')}</tr>`).join('')}</tbody></table></div></section>`}
function backtestMarkup(result){const money=value=>fmt(value);const window=result.analysis_window||{};const open=result.open_position;const summary=[['起始资金',money(result.starting_cash)],['可用现金',money(result.available_cash)],['持仓市值',money(result.marked_position_value)],['期末权益',money(result.ending_equity)],['信号数',result.signal_count??'—'],['已完成交易',result.trades?.length||0]];const openText=open?`${open.shares} 股 · 成本 ${money(open.entry_price)} · 估值 ${money(open.marked_value)}`:'无';return `<div class="backtest-report"><div class="backtest-notice"><b>撮合与账目验收</b> · Backtrader · 佣金 0 · 滑点 0 · 策略效果未验证</div><p>数据区间 ${escapeHtml(window.from||'无')} 至 ${escapeHtml(window.to||'无')} · 最新本地行情 ${escapeHtml(window.latest_available||'无')}${window.historical?' · 历史模拟':''}</p><div class="backtest-summary">${summary.map(([label,value])=>`<div><span>${label}</span><b>${escapeHtml(value)}</b></div>`).join('')}</div><p><b>当前持仓：</b>${escapeHtml(openText)}</p><p><b>账目核对：</b>权益=现金+持仓市值 ${result.accounting_invariants?.equity_equals_cash_plus_marked_position?'通过':'未通过'} · 零佣金 ${result.accounting_invariants?.zero_commission?'通过':'未通过'}</p>${ledgerTable('成交与平仓',result.trades,[['入场日','entry_date'],['出场日','exit_date'],['股数','shares'],['入场价','entry_price',money],['出场价','exit_price',money],['退出原因','exit_reason'],['盈亏','pnl',money]])}${ledgerTable('订单事件',result.order_events,[['日期','date'],['类型','role'],['状态','status'],['股数','filled_size'],['成交价','price',money],['佣金','commission',money]])}${ledgerTable('拒绝信号',result.rejections,[['信号日','signal_date'],['尝试日','entry_date'],['原因','reason'],['开盘价','open',money]])}<p class="muted">${(result.limitations||[]).map(escapeHtml).join(' · ')}</p></div>`}
async function startBacktest(){const code=$('backtestCode').value.trim(),cash=Number($('backtestCash').value),start=$('backtestStart').value,end=$('backtestEnd').value;if(!code){$('backtestStatus').textContent='请输入代码。';return}if(!Number.isFinite(cash)||cash<=0){$('backtestStatus').textContent='初始资金必须大于 0。';return}if(start&&end&&start>end){$('backtestStatus').textContent='起始日期不能晚于截止日期。';return}$('backtestBtn').disabled=true;$('backtestStatus').textContent='正在运行本地模拟，请稍候…';$('backtestResults').innerHTML='';const payload={code,kind:$('backtestKind').value,timeframe:$('backtestTimeframe').value,cash};if(start)payload.start=start;if(end)payload.end=end;try{const result=await fetchJson('/api/v1/backtests',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});$('backtestResults').innerHTML=backtestMarkup(result);$('backtestStatus').textContent='模拟完成；结果仅核对撮合与账目，不代表策略有效。'}catch(error){$('backtestStatus').textContent='模拟失败：'+error.message}finally{$('backtestBtn').disabled=false}}
let fixedBacktestTimer=null;
function renderFixedBacktestStatus(job){const summary=job.summary||{},reportUrl=job.report_url;const stats=[['独立账户',summary.security_timeframe_accounts??'—'],['交易数',summary.trade_count??'—'],['信号数',summary.signal_count??'—'],['成交形态数',summary.distinct_traded_patterns??'—'],['多形态重叠信号',summary.overlapping_multi_pattern_signals??'—']];const failure=(job.failures||[]).length?`<details><summary>行情失败 ${job.failures.length} 项</summary><ul>${job.failures.map(item=>`<li>${escapeHtml(item.code)} · ${escapeHtml(item.timeframe)} · ${escapeHtml(item.error)}</li>`).join('')}</ul></details>`:'';$('fixedResults').innerHTML=`${reportUrl?`<div class="fixed-report-summary">${stats.map(([label,value])=>`<div><span>${label}</span><b>${escapeHtml(value)}</b></div>`).join('')}</div><p>${escapeHtml(summary.note||'')}</p><p>策略 ${escapeHtml(job.strategy_version||'')} · 开发切点 ${escapeHtml(job.development_cutoff||'按数据推导')}</p><a class="button-link" href="${escapeHtml(reportUrl)}" target="_blank" rel="noopener">打开完整回测报告</a>`:''}${failure}`}
async function loadFixedBacktestHistory(page=1){
  try{
    const data=await fetchJson(`/api/v1/fixed-backtests?page=${page}`),items=data.items||[];
    $('fixedHistory').innerHTML=items.length?items.map(job=>{
      const params=job.params||{},status={queued:'排队中',running:'运行中',completed:'完成',partial_failed:'部分失败',failed:'失败',interrupted:'已中断'}[job.status]||job.status;
      const range=params.sample_scope_detail?`混合范围样本：${params.sample_scope_detail}`:`${params.date_from||'最早'} 至 ${params.date_to||'最新'}`;
      return `<article class="history-result-row"><b>${escapeHtml(job.created_at||job.job_id)}${params.sample?' · 样本':''}</b><span>${escapeHtml(status)} · ${escapeHtml(range)} · ${escapeHtml((params.kinds||[]).join(' / '))} · ${escapeHtml((params.timeframes||[]).join(' / '))}</span><span class="muted">策略 ${escapeHtml(job.strategy_version||'—')} · 账户 ${job.summary?.security_timeframe_accounts??job.total??'—'} · 交易 ${job.summary?.trade_count??'—'}</span>${job.report_url?`<a href="${escapeHtml(job.report_url)}" target="_blank" rel="noopener">打开已保存报告</a>`:''}${job.error?`<span class="muted">${escapeHtml(job.error)}</span>`:''}</article>`
    }).join(''):'<p class="muted">尚无历史固定规则回测。</p>';
  }catch(error){$('fixedHistory').innerHTML=`<div class="error-box">无法读取历史回测任务：${escapeHtml(error.message)}</div>`}
}
async function pollFixedBacktest(jobId){const job=await fetchJson(`/api/v1/fixed-backtests/${encodeURIComponent(jobId)}`);const label={queued:'排队中',running:'运行中',completed:'完成',partial_failed:'部分标的失败',failed:'失败',interrupted:'已中断'}[job.status]||job.status;$('fixedStatus').textContent=`${label} · ${Math.round((job.progress||0)*100)}% · ${job.completed||0}/${job.total||0}${job.current?.code?` · ${job.current.code} ${job.current.timeframe||''}`:''}${job.error?` · ${job.error}`:''}`;if(['completed','partial_failed','failed','interrupted'].includes(job.status)){if(fixedBacktestTimer)clearInterval(fixedBacktestTimer);fixedBacktestTimer=null;$('fixedRunBtn').disabled=false;renderFixedBacktestStatus(job);loadFixedBacktestHistory();return true}return false}
async function startFixedBacktest(){if(fixedBacktestTimer)clearInterval(fixedBacktestTimer);const kinds=[$('fixedStock').checked?'stock':null,$('fixedEtf').checked?'etf':null].filter(Boolean),timeframes=[...$('fixedTimeframes').selectedOptions].map(option=>option.value),scope=$('fixedScope').value,limit=$('fixedLimit').value.trim(),start=$('fixedStart').value,end=$('fixedEnd').value,cutoff=$('fixedCutoff').value;if(!kinds.length){$('fixedStatus').textContent='请至少选择股票或 ETF。';return}if(!timeframes.length){$('fixedStatus').textContent='请至少选择一个周期。';return}if(start&&end&&start>end){$('fixedStatus').textContent='起始日期不能晚于截止日期。';return}$('fixedRunBtn').disabled=true;$('fixedResults').innerHTML='';$('fixedStatus').textContent='正在提交固定规则回测…';const params={kinds,timeframes,cash:100000,scope,strategy_version:$('fixedStrategyVersion').value};if(limit)params.limit=Number(limit);if(start)params.date_from=start;if(end)params.date_to=end;if(cutoff)params.development_cutoff=cutoff;try{const job=await fetchJson('/api/v1/fixed-backtests',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});$('fixedStatus').textContent=`任务 ${job.job_id} 已启动；逐标的、逐周期独立运行。` ;const finished=await pollFixedBacktest(job.job_id);if(!finished)fixedBacktestTimer=setInterval(async()=>{try{if(await pollFixedBacktest(job.job_id))return}catch(error){$('fixedStatus').textContent=`读取回测进度失败：${error.message}`}},2000)}catch(error){$('fixedStatus').textContent=`回测无法启动：${error.message}`;$('fixedRunBtn').disabled=false}}
function setRefreshBusy(busy){$('refreshOneBtn').disabled=busy;$('refreshAllBtn').disabled=busy}
function watchRefreshJob(meta){
  const {status:statusNode,report:reportNode}=refreshNodes(meta),jobId=meta.job_id;
  setRefreshBusy(true);
  const timer=setInterval(async()=>{
    try{
      const current=await fetchJson('/api/v1/jobs/'+encodeURIComponent(jobId));
      if(current.status==='completed'){
        clearInterval(timer);
        if(meta.reanalyze&&current.success_count){await analyze()}
        statusNode.textContent=window.CaisenRefreshReport.formatRefreshSummary(current,meta.one);
        reportNode.innerHTML=window.CaisenRefreshReport.renderRefreshReport({...current,job_id:jobId});
        setRefreshBusy(false);
      }else if(current.status==='interrupted'||current.status==='failed'){
        clearInterval(timer);statusNode.textContent=`行情更新任务${current.status==='interrupted'?'已中断':'失败'}：${current.error||'请查看失败标的和原因'}`;
        reportNode.innerHTML=window.CaisenRefreshReport.renderRefreshReport({...current,job_id:jobId});
        setRefreshBusy(false);
      }else statusNode.textContent=`行情更新中 ${Math.round((current.progress||0)*100)}% · ${current.completed||0}/${current.total||0} · 成功 ${current.success_count||0} · 失败 ${current.failure_count||0}`;
    }catch(error){statusNode.textContent=`行情更新进度暂时无法读取：${error.message}`}
  },2000);
}
async function startRefresh(params,statusNode,reanalyze){
  const one=Boolean(params.code),reportNode=$(one?'refreshOneReport':'refreshAllReport');
  reportNode.innerHTML='';
  statusNode.textContent=one?'正在请求该标的完整历史行情…':'正在提交全市场行情更新…';
  setRefreshBusy(true);
  try{
    const job=await fetchJson('/api/v1/data/refresh',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
    const meta={job_id:job.job_id,params,one,reanalyze};
    watchRefreshJob(meta);
  }catch(error){statusNode.textContent='行情更新无法启动：'+error.message;setRefreshBusy(false)}
}

async function showPattern(id){
  const pattern=(lastAnalysis?.decision?.patterns||lastAnalysis?.patterns||[]).find(p=>p.pattern_id===id);
  if(!pattern){$('chartNote').textContent='该形态当前不可用，请重新分析';return}
  if(pattern.timeframe!==lastAnalysis.timeframe){$('tf').value=pattern.timeframe;await analyze(id);return}
  try{
    const geometry=CaisenPatternChart.patternGeometry(lastAnalysis,pattern);
    selectedPattern=pattern;chartCenter=(geometry.first+geometry.last+1)/2;chartWindowBars=geometry.last-geometry.first+1;
    $('chartRange').value='custom';renderDetailChart(lastAnalysis,chartWindowBars);
    document.querySelectorAll('.pattern-card').forEach(card=>{const active=card.dataset.patternId===id;card.classList.toggle('selected',active);card.querySelector('[data-show-pattern]')?.setAttribute('aria-pressed',String(active));if(active&&card.closest('details'))card.closest('details').open=true});
    $('chart').scrollIntoView({behavior:'instant',block:'center'});
  }catch(error){$('chartNote').textContent=error.message}
}
$('detailRows').addEventListener('click',event=>{const button=event.target.closest('[data-show-pattern]');if(button)showPattern(button.dataset.showPattern)});
$('fixedRunBtn').onclick=startFixedBacktest;$('fixedHistoryRefresh').onclick=()=>loadFixedBacktestHistory();document.querySelector('[data-panel=\"fixed-simulation\"]').addEventListener('click',()=>loadFixedBacktestHistory());
