const names={w_bottom:'W底',spring:'破底翻',bull_flag:'下飘旗形上涨中继',inverse_head_shoulders:'头肩底',triangle_bottom:'收敛三角形底部',m_top:'M头',false_breakout:'假突破',bear_flag:'上飘旗形下跌中继',head_shoulders:'头肩顶',triangle_top:'收敛三角形头部',FOCUS_NECKLINE:'颈线聚焦'};
const legacyNames={w_bottom_spring:'W底破底翻',false_breakout_head_shoulders:'假突破（头肩顶）'};
const actions={CONSIDER_BUY:'可考虑买入',WAIT:'等待',HOLD:'持有观察',SELL_IF_HELD:'如持有可考虑卖出',INSUFFICIENT_DATA:'数据不足',FOCUS_NECKLINE:'颈线聚焦'};
const reasonNames={pattern_not_buy_candidate:'该形态不属于四种买入候选',timeframe_direction_conflict:'日线与周线确认方向冲突，请分别复核',ambiguous_structure:'多个候选形态共用结构点，暂不生成买卖计划',pattern_waiting_for_breakout:'形态已成型，等待颈线或结构边界收盘突破',bull_flag_target_locked:'旗形边界已突破，但前高尚未收盘突破，测幅目标未解锁',no_confirmed_actionable_pattern:'当前没有通过确认与交易空间检查的形态',no_current_pattern:'当前周期没有识别到有效形态',confirmed_bearish_structure_sell_warning_if_held:'已确认顶部或空头结构；如有持仓，请复核退出计划',bullish_structure_exists_without_new_entry:'存在多头结构，但当前没有新的入场信号',pattern_not_confirmed_bullish:'形态尚未确认为多头结构',volume_data_unverified:'成交量单位或企业行动未核验，形态仅作研究观察',invalid_structural_stop:'无法建立有效止损',no_verified_target:'没有可用于计划的目标位',insufficient_cash_or_liquidity:'现金或流动性不足以形成有效委托',buy_gates_not_met:'买入条件未全部满足'};
const $=id=>document.getElementById(id);let lastAnalysis=null;let scanRows=[];let visibleScanRows=[];let scanShown=0;let scanTimer=null;let detailChart=null;let chartBarCount=0;let chartWindowBars=120;let selectedPattern=null;let chartCenter=null;let chartFirst=0;

document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>{document.querySelectorAll('nav button,.panel').forEach(item=>item.classList.remove('active'));button.classList.add('active');$(button.dataset.panel).classList.add('active')});
Object.entries(names).forEach(([key,value])=>{if(key==='FOCUS_NECKLINE')return;const option=document.createElement('option');option.value=key;option.textContent=value;$('scanShapeFilter').append(option);const prefilter=document.createElement('option');prefilter.value=key;prefilter.textContent=value;$('scanPatternTypes').append(prefilter);const historical=document.createElement('option');historical.value=key;historical.textContent=value;$('historyPatterns').append(historical)});
fetchJson('/api/v1/health').then(result=>{$('health').textContent=`${result.status} · ${result.rule_version} · 无下单`;window.CaisenPatternMigration=result.pattern_migration||null;window.applyScanPatternMigration?.(window.CaisenPatternMigration)}).catch(()=>$('health').textContent='服务未连接');

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
function planMarkup(plan){if(!plan)return '';const reasons=(plan.reason_codes||[]).map(reasonLabel).map(escapeHtml).join('；')||'交易风险与目标参考';const fillBased=plan.stop_basis==='actual_entry_fixed_fraction';const basis=fillBased?'止损和止盈按实际成交价固定为 −8% / +20%':`计划保护价 ${fmt(plan.stop)} · 首目标 ${fmt(plan.targets?.[0])}`;const reference=plan.main_bottom_reference?`<br>结构参考：冻结颈线 ${fmt(plan.structure_level)} · 底部参考 ${fmt(plan.main_bottom_reference.support_lower)}–${fmt(plan.main_bottom_reference.support_upper)}`:'';return `<div class="plan-summary">计划参考：${escapeHtml(plan.decision)} · 参考价 ${fmt(plan.entry_price)} · ${basis} · 参考风险 ${fmt((plan.planned_risk_pct||0)*100)}% · 收益风险比 ${fmt(plan.reward_risk)}${reference}<br>${reasons}${dataWarningsMarkup(plan.data_warnings)}</div>`}
function evidenceLabel(item){let code=item.code||'',value=item.value;if(code==='neckline_selection'&&value&&typeof value==='object'){const alternatives=(value.candidates||[]).filter(candidate=>!candidate.selected).length;const reasonNames={selected_by_ordered_evidence:'主线',fewer_independent_tests:'独立测试较少',shorter_test_span:'测试跨度较短',larger_normalized_residual:'触点偏差较大',older_latest_test:'最近测试较早',anchor_date_tie_break:'锚点日期决胜',ordered_after_selected:'排序靠后',slope_limit_exceeded:'斜率超限',slope_limit_exceeded_after_horizontal_conversion:'水平化复检未通过',whole_interval_high_crossed:'区间高点越界',insufficient_independent_tests:'独立测试不足',tests_do_not_match_triangle_upper_boundary:'未匹配三角形上沿',triangle_upper_envelope_crossed:'三角形上沿被穿越',w_intervening_rim_not_covered:'未覆盖W中间反弹',shoulder_geometry_not_supported_by_boundary:'未匹配头肩几何',missing_ihs_reaction_side_test:'缺少左右反弹段颈线测试',neckline_below_ihs_structure:'颈线低于头肩结构点'};const rejected=(value.rejected_candidates||[]).slice(0,3).map(item=>{let text=reasonNames[item.reason]||item.reason;if(item.crossing_date)text+=` ${item.crossing_date}高${fmt(item.crossing_high)} 超容差${fmt(item.maximum_excess)}`;else if(item.anchor_dates)text+=` ${item.anchor_dates.join(' / ')}`;else if(item.test_dates)text+=` ${item.test_dates.join('、')}`;return text});const rejectionCounts=Object.entries(value.rejection_counts||{}).map(([reason,count])=>`${reasonNames[reason]||reason} ${count}`).join('、');const ranked=(value.candidates||[]).slice(0,4).map(candidate=>`#${candidate.rank} ${candidate.selected?'主线':reasonNames[candidate.reason]||candidate.reason} ${candidate.independent_test_count}测/${candidate.test_span_bars}K/偏差${Number(candidate.normalized_residual||0).toFixed(2)}`).join('；');let summary=`颈线选线：${value.selected_candidate_id||'无ID'} · 已验证压力段 ${value.effective_start_date||'起点未知'} → ${value.coverage?.end_date||'终点未知'} · ${value.independent_tests?.length||0} 次测试 · 线冻结 ${value.line_frozen_at||value.freeze_date||'未知'} · 结构可知 ${value.structure_known_at||'未知'} · ${alternatives} 条备选；排序按测试数、跨度、偏差、最新测试及锚点日期；不参考价格高低、当前收盘或目标表现`;if(ranked)summary+=` · 排序证据 ${ranked}`;if(rejectionCounts)summary+=` · 淘汰统计 ${rejectionCounts}`;if(rejected.length)summary+=` · 淘汰示例 ${rejected.join('；')}`;return summary;}if(code==='main_structure'&&value&&typeof value==='object'&&value.neckline_rule_version){const duration=value.scale==='5y'?'日线近5自然年':value.scale==='10y'?'周线近10自然年':value.scale||'长周期';const support=value.support_lower!==undefined&&value.support_upper!==undefined?`支撑 ${fmt(value.support_lower)}–${fmt(value.support_upper)}`:'';return `主结构：${duration} · ${support} · 压力段 ${value.neckline_effective_start||'未记录'} 起 · ${value.neckline_independent_test_count||0} 次独立测试 · ${value.signal_eligible?'价格确认可参与':'等待有效价格确认'}${value.eligibility_reason?`（${({awaiting_fresh_neckline_breakout:'等待后续有效颈线突破'})[value.eligibility_reason]||value.eligibility_reason}）`:''}`}if(code==='main_structure'&&value&&typeof value==='object'){const duration=value.scale==='5y'?'日线近5自然年':value.scale==='10y'?'周线近10自然年':value.scale||'长周期';const support=value.support_lower!==undefined&&value.support_upper!==undefined?`支撑 ${fmt(value.support_lower)}–${fmt(value.support_upper)}`:'';const neckline=typeof value.neckline==='object'&&value.neckline?`颈线 ${fmt(value.neckline.slope)}·x+${fmt(value.neckline.intercept)}`:value.neckline!==undefined?`颈线 ${fmt(value.neckline)}`:'';return `主结构：${duration} · ${support} · ${neckline} · ${value.signal_eligible?'价格确认可参与':'等待有效价格确认'}${value.eligibility_reason?`（${({awaiting_fresh_neckline_breakout:'等待后续有效颈线突破'})[value.eligibility_reason]||value.eligibility_reason}）`:''}`}if(code==='neckline_tests'&&Array.isArray(value))return `颈线测试：${value.map(test=>`${test.group_id||'分组未知'} ${test.pivot_date||'日期未知'} / 知晓 ${test.known_date||'未记录'} @ ${fmt(test.level)}（${test.valid?'有效':'无效'}）`).join('；')}`;if(code==='main_children'&&Array.isArray(value))return `内含子形态：${value.map(child=>`${names[child.pattern_type]||child.pattern_type} ${child.anchor_start_date||''}–${child.anchor_end_date||''}`).join('；')||'无'}`;if(code==='support_tests'&&Array.isArray(value))return `支撑测试：${value.map(test=>`${test.date||test.pivot_date||'日期未知'} @ ${fmt(test.level??test.price)} ${test.group_id||''}`).join('；')||'无'}`;if(code==='neckline_breakout'&&value&&typeof value==='object')return `颈线收盘确认：${value.date||'日期未知'} 收盘 ${fmt(value.close)} · 冻结颈线 ${fmt(value.frozen_neckline)} · 要求收盘 ${fmt(value.required_close)}`;if(code==='weak_crossings'&&Array.isArray(value))return `颈线弱穿越：${value.map(item=>`${item.date||'日期未知'} 收盘 ${fmt(item.close)} / 要求 ${fmt(item.required_close)}`).join('；')||'无'}`;if(code==='volume_support'&&value&&typeof value==='object')return `量能辅助证据：${value.status==='supportive'?'支持':value.status==='unknown'?'未记录':'不支持'}${value.ratio===null||value.ratio===undefined?'':` · 比率 ${fmt(value.ratio)}`}${value.date?` · ${value.date}`:''}（不作买入门槛）`;if(code==='main_bottom_reference'&&value&&typeof value==='object')return `实际底部参考 ${fmt(value.support_lower)}–${fmt(value.support_upper)} · ${value.first_trough_date||'日期未知'} 至 ${value.latest_anchor_date||'日期未知'} · 冻结颈线 ${fmt(value.neckline_at_observed)}（观察日 ${value.observed_date||'未知'}）`;if(code==='main_selection'&&value&&typeof value==='object')return `${value.selected?'已选主结构':'未选主结构'} · 截至 ${value.decided_at||'未知日期'} · ${value.reason||'排序原因未记录'}`;if(code==='execution_suppression'&&value?.suppressed)return `已纳入上层主结构 ${value.parent_id||''}，不单独生成买入计划`;const labels={pivot_topology:'结构点顺序',neckline:'颈线',neckline_left:'左侧颈线',neckline_right:'右侧颈线',close_confirmation:'收盘突破确认',close_reclaim:'收盘收复',reclaim_level:'收复价位',break_level:'越界价位',volume_verified:'量能已核验',box_lower:'箱体下缘',box_upper:'箱体上缘',box_first_touch:'箱体起始触点',box_last_touch:'箱体末端触点',box_width_atr:'箱体宽度（ATR）',followthrough_confirmed:'后续边界确认',target_reached:'首目标触达',structure_invalidation:'结构极值失效',outer_w_invalidation:'外层W底失效',neckline_projection_invalidation:'颈线外推失效',neckline_invalidation:'颈线失效',w_geometry_metrics:'W底几何特征',structure_duration:'结构时间',first_valid_break:'首次有效突破',low_position:'历史价格位置',prepattern_drawdown:'形态前高回撤',pattern_quality_pass:'形态质量门槛'};let prefix='';if(code.startsWith('inner_spring_')){prefix='内层';code=code.slice('inner_spring_'.length)}const percent=number=>number===null||number===undefined||!Number.isFinite(Number(number))?'未记录':`${(Number(number)*100).toFixed(1)}%`;if(code==='pivot_topology'&&Array.isArray(value))value=value.map(point=>point==='high'?'高点':point==='low'?'低点':String(point)).join(' → ');else if(code==='w_geometry_metrics'&&value&&typeof value==='object')value=`左段 ${value.first_leg_bars} 根 · 右段 ${value.second_leg_bars} 根 · 时长比 ${value.leg_duration_ratio} · 两底差 ${value.trough_gap_pct}%`;else if(code==='structure_duration'&&value&&typeof value==='object'){const hasBreak=Boolean(value.first_valid_break_date),end=value.first_valid_break_date||value.observed_date||'观察日未知';value=`结构起点 ${value.start_date||'日期未知'} → ${hasBreak?'首次突破':'观察日'} ${end} · 持续 ${value.duration_days??'—'} 天 · ${value.eligible?'超过3自然月':'未超过3自然月'}${!hasBreak?'，尚未突破':''}`}else if(code==='first_valid_break')value=value||'未发生';else if(code==='low_position'&&value&&typeof value==='object'){const branches=[];if(value.position_pass)branches.push('历史价格低位');if(value.drawdown_pass)branches.push('前高回撤');value=`区间位置 ${percent(value.position_ratio)}（门槛≤30%） · 通过分支：${branches.join('、')||'无'}`}else if(code==='prepattern_drawdown'&&value&&typeof value==='object')value=`前高 ${value.high===null||value.high===undefined?'未记录':fmt(value.high)} · 回撤 ${percent(value.drawdown)}（门槛≥${percent(value.threshold)}，${value.passed?'通过':'未通过'}）`;else if(code==='pattern_quality_pass')value=value?'通过':'未通过';else if((code==='followthrough_confirmed'||code==='target_reached')&&value&&typeof value==='object')value=`${value.date||'日期未知'} · 价位 ${fmt(value.level??value.price)}`;else if((code==='structure_invalidation'||code==='outer_w_invalidation'||code==='neckline_projection_invalidation')&&value&&typeof value==='object')value=`${value.date||'日期未知'} · ${code==='neckline_projection_invalidation'?'颈线':'失效价位'} ${fmt(value.level)}`;else if(typeof value==='boolean')value=value?'是':'否';else if(value&&typeof value==='object'){if(value._position_history||value._cache)return `${prefix}内部数据`;value=JSON.stringify(value)}else if(typeof value==='number')value=fmt(value);return `${prefix}${labels[code]||code}：${String(value??'—')}`}
function patternCard(pattern){const pivotDate=((pattern.evidence||[]).find(item=>item.code==='confirmed_date')||{}).value;const confirmed=pattern.confirmation_date||pivotDate||'—';const confirmationLabel=pattern.confirmation_date?'价格确认':'结构点确认';const overlaps=(pattern.evidence||[]).find(item=>item.code==='competing_topology_labels')?.value||[];const selectedTag=isSelectedMain(pattern)?'<span class="tag main-selected">已选主结构</span>':'';const suppression=(pattern.evidence||[]).find(item=>item.code==='execution_suppression')?.value;const nestedNote=suppression?.suppressed?'<div class="muted">该结构已包含在上层主形态中，不单独生成买卖计划。</div>':'';const warning=pattern.flags?.includes('ambiguous_structure')?`<div class="pattern-ambiguity-warning"><b>候选形态有重叠，暂不生成买卖计划</b>${overlaps.length?`<span>共享同一组结构点：${overlaps.map(patternType=>escapeHtml(names[patternType]||patternType)).join('、')}</span>`:''}</div>`:'';const lifecycle=CaisenPatternChart.instanceLifecycle(pattern);const timeline=lifecycle.length?`<ol class="instance-timeline" data-instance-id="${escapeHtml(pattern.pattern_instance_id||pattern.pattern_id||'')}">${lifecycle.map(item=>`<li data-instance-id="${escapeHtml(item.instance_id||'')}"${item.event_id?` data-event-id="${escapeHtml(item.event_id)}"`:''}><time>${escapeHtml(item.date)}</time><span>${escapeHtml(item.label)}</span>${item.boundary_id?`<small>边界 ${escapeHtml(item.boundary_id)}</small>`:''}</li>`).join('')}</ol>`:'';const evidence=(pattern.evidence||[]).filter(item=>!isInternalEvidence(item)).map(item=>escapeHtml(evidenceLabel(item))).join(' · ');return `<article class="card pattern-card" data-pattern-id="${escapeHtml(pattern.pattern_id||'')}" data-instance-id="${escapeHtml(pattern.pattern_instance_id||'')}"><div><b>${escapeHtml(pattern.code)} · ${escapeHtml(patternName(pattern))}</b> ${selectedTag}<span class="tag">${escapeHtml(pattern.timeframe)} / ${escapeHtml(stateLabel(pattern))}</span><div class="muted">形成 ${escapeHtml(pattern.formed_at||'—')} · ${confirmationLabel} ${escapeHtml(confirmed)} · 引擎 ${escapeHtml(pattern.rule_version||'—')} · 规则集 ${escapeHtml(pattern.source_rule_version||'—')} · 数据 ${escapeHtml(pattern.data_version||'—')}</div>${nestedNote}${warning}${timeline}<div class="evidence">${evidence}</div>${planMarkup(pattern.trade_plan)}</div><button type="button" data-show-pattern="${escapeHtml(pattern.pattern_id||'')}">查看该形态</button></article>`}

function miniCandles(bars){if(!bars?.length)return '<span class="mini-empty">暂无K线</span>';const width=184,height=48,pad=3,step=(width-pad*2)/bars.length;const lows=bars.map(bar=>Number(bar.low)),highs=bars.map(bar=>Number(bar.high));const min=Math.min(...lows),max=Math.max(...highs),range=max-min||1;const y=value=>pad+(max-value)/range*(height-pad*2);let shapes='';bars.forEach((bar,index)=>{const x=pad+index*step+step/2,open=y(Number(bar.open)),close=y(Number(bar.close)),up=Number(bar.close)>=Number(bar.open),color=up?'#d84a4a':'#18866b',body=Math.max(1,Math.abs(close-open));shapes+=`<line x1="${x}" y1="${y(Number(bar.high))}" x2="${x}" y2="${y(Number(bar.low))}" stroke="${color}" stroke-width="1"/><rect x="${x-step*.28}" y="${Math.min(open,close)}" width="${Math.max(1,step*.56)}" height="${body}" fill="${color}"/>`});return `<svg class="mini-candles" viewBox="0 0 ${width} ${height}" role="img" aria-label="最近${bars.length}根日K线">${shapes}</svg>`}
function scanCard(row){const plan=row.selected_plan;const activePatterns=(row.patterns||[]).filter(pattern=>!isTerminalPattern(pattern));const main=plan?row.patterns.find(pattern=>pattern.pattern_id===plan.pattern_id):(activePatterns[0]||row.patterns?.[0]);const label=main?.display_label||main?.pattern_type&&stateLabel(main)||'暂无当前形态';const warnings=[...(row.data_warnings||[]),...(plan?.data_warnings||[])];const quality=warnings.length?'数据待核验':'行情可用';const price=`${fmt(row.latest_close)} · ${row.latest_date||''}`;const historicalNote=row.analysis_window?.historical?`<span class="historical-scan-note">历史区间截至 ${escapeHtml(row.analysis_window.to||row.latest_date||'未知')}；打开详情将加载最新行情</span>`:'';return `<article class="scan-row" data-code="${escapeHtml(row.code)}" data-kind="${escapeHtml(row.kind)}" tabindex="0" role="link" aria-label="查看${escapeHtml(row.name||row.code)}详情"><div class="scan-main"><div class="scan-title"><b>${escapeHtml(row.name||row.code)}</b><span class="muted">${escapeHtml(row.code)} · ${row.kind==='etf'?'ETF':'股票'}</span></div><span class="action-pill action-${escapeHtml(row.action||'WAIT')}">${escapeHtml(actionLabel(row.action))}</span><span class="pattern-label">${escapeHtml(label)}</span>${historicalNote}<span class="muted">${escapeHtml(reasonLabel(row.reason_codes?.[0]||row.explanation||''))}</span>${dataWarningsMarkup(warnings)}</div><div class="scan-chart">${miniCandles(row.mini_bars)}<span class="muted">${escapeHtml(price)}</span></div><div class="scan-metrics">${plan?`<b>风险 ${fmt((plan.planned_risk_pct||0)*100)}%</b><b>首目标 ${fmt(plan.targets?.[0])}</b><b>${fmt(plan.reward_risk)}R</b>`:`<b>${activePatterns.length} 个当前结构</b>${row.patterns?.length>activePatterns.length?`<span class="muted">另有 ${row.patterns.length-activePatterns.length} 个结束结构</span>`:''}<span class="muted">${escapeHtml(quality)}</span>`}<button class="detail-link">查看详情 →</button></div></article>`}
function renderMore(){if(!visibleScanRows.length){$('results').innerHTML='<div class="scan-empty">没有标的符合当前筛选。可重置筛选查看完整扫描结果。</div>';return}const end=Math.min(visibleScanRows.length,scanShown+100);$('results').insertAdjacentHTML('beforeend',visibleScanRows.slice(scanShown,end).map(scanCard).join(''));scanShown=end;const more=$('loadMore');if(more)more.remove();if(scanShown<visibleScanRows.length)$('results').insertAdjacentHTML('beforeend',`<button id="loadMore" class="load-more">再显示 100 个（剩余 ${visibleScanRows.length-scanShown}）</button>`);$('loadMore')?.addEventListener('click',renderMore);document.querySelectorAll('.scan-row:not([data-bound])').forEach(row=>{row.dataset.bound='1';const open=()=>openTicker(row.dataset.code,row.dataset.kind);row.addEventListener('click',open);row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}})})}
function openTicker(code,kind){$('code').value=code;$('kind').value=kind;$('tf').value='1d';document.querySelector('[data-panel=detail]').click();analyze()}
function applyScanFilters(){const action=$('scanActionFilter').value;const patternTypes=[...$('scanShapeFilter').selectedOptions].map(option=>option.value);visibleScanRows=CaisenScanFilters.filterScanRows(scanRows,{action,patternTypes});scanShown=0;$('results').innerHTML='';const actionCounts=scanRows.reduce((counts,row)=>{counts[row.action]=(counts[row.action]||0)+1;return counts},{ });$('scanFilterSummary').textContent=`筛选后 ${visibleScanRows.length} / ${scanRows.length} 个标的 · 可考虑买入 ${actionCounts.CONSIDER_BUY||0} · 等待 ${actionCounts.WAIT||0} · 持有观察 ${actionCounts.HOLD||0} · 卖出提示 ${actionCounts.SELL_IF_HELD||0} · 数据不足 ${actionCounts.INSUFFICIENT_DATA||0}`;$('buyOnlyBtn').setAttribute('aria-pressed',String(action==='CONSIDER_BUY'));$('buyOnlyBtn').classList.toggle('selected',action==='CONSIDER_BUY');renderMore()}
function renderScan(rows,inventory){scanRows=rows||[];$('scanResultFilters').hidden=false;$('scanActionFilter').value='';[...$('scanShapeFilter').options].forEach(option=>option.selected=false);(scanSettings.pattern_types||[]).forEach(type=>{const option=[...$('scanShapeFilter').options].find(item=>item.value===type);if(option)option.selected=true});const window=inventory?.requested_window||{};const range=`${window.from||'最早可用'} 至 ${window.to||'最新可用'}`;const effectiveEnds=scanRows.map(row=>row.analysis_window?.to).filter(Boolean).sort();const latestAnalyzed=effectiveEnds.at(-1);const historical=scanRows.some(row=>row.analysis_window?.historical);$('scanSummary').innerHTML=`<div class="scan-summary"><b>扫描完成</b> · 请求区间 ${escapeHtml(range)} · 实际分析行情截至 ${escapeHtml(latestAnalyzed||'无')}${historical?' · 历史区间结果，请按每行行情日期理解':''} · ${scanRows.length} 个标的 · 股票 ${inventory?.stock_count??'—'} · ETF ${inventory?.etf_count??'—'} · 排序保持买入优先。行动数量见筛选栏。</div>`;applyScanFilters()}

function decisionMarkup(decision){
  if(!decision)return '';
  const assessment=decision.assessment||lastAnalysis?.assessment,plan=decision.selected_plan;
  if(assessment){
    const timeframes=assessment.timeframes||{},selected=assessment.selected_timeframe?timeframes[assessment.selected_timeframe]||{}:{},status=assessment.status;
    const statusLabels={BUY:'颈线突破确认',FOCUS_NECKLINE:'颈线关注',NONE:'暂无突破或颈线关注状态',INSUFFICIENT_EVIDENCE:'证据不足，暂无法评估'};
    const reasons={first_valid_breakout_day:'本周期出现已记录的首次有效突破。',post_breakout_retest_above_92pct:'突破后出现回落，收盘仍在规则允许的回测范围内。',within_one_previous_bar_atr_below_boundary:'收盘位于冻结边界下方一个前周期 ATR 以内，进入颈线关注区。',breakout_retest_rule_not_met:'突破后的回测未满足当前确认条件。',outside_neckline_attention_band:'当前收盘不在颈线关注区间。',terminal_pattern_state:'形态已结束。',confirmed_event_not_saved:'形态虽已确认，但保存记录中缺少有效突破事件。',cutoff_ohlc_snapshot_missing:'缺少扫描时点 OHLC 快照。',previous_complete_bar_atr_missing:'缺少前一根完整 K 线 ATR。'};
    const reason=selected.reason||assessment.reason||'暂无评估理由',why=reasons[reason]||reasonLabel(reason),evidence=selected.evidence||assessment.evidence||{},tf=assessment.selected_timeframe?`${assessment.selected_timeframe==='1d'?'日线':'周线'}`:'未指定周期';
    const event=status==='BUY'&&evidence.first_break_date?`首次有效突破 ${evidence.first_break_date} · 收盘 ${fmt(evidence.first_break_close)}`:status==='FOCUS_NECKLINE'?`冻结边界 ${fmt(evidence.boundary)} · 前周期 ATR ${fmt(evidence.previous_complete_bar_atr)}`:'';
    const asOf=evidence.as_of_date||assessment.as_of_date||'日期未记录',latest=evidence.latest_close;
    const action=status==='INSUFFICIENT_EVIDENCE'?'':`<span class="action-pill action-${escapeHtml(decision.action)}">${escapeHtml(actionLabel(decision.action))}</span>`;
    const planReference=plan?`<details class="legacy-plan-reference"><summary>旧交易计划参考（不代表当前评估）</summary>${planMarkup(plan)}</details>`:'';
    return `<section id="assessment-status" class="decision-box decision-${escapeHtml(decision.action||'WAIT')}"><div class="decision-heading"><span class="assessment-status-pill assessment-${escapeHtml(status||'UNKNOWN')}">${escapeHtml(statusLabels[status]||'评估状态未知')}</span>${action}<b>${escapeHtml(statusLabels[status]||'评估状态未知')}</b></div><p class="assessment-reason">${escapeHtml(why)}</p><div class="assessment-evidence">评估周期 ${escapeHtml(tf)} · 截止 ${escapeHtml(asOf)}${latest===undefined?'':` · 最新收盘 ${fmt(latest)}`}${event?` · ${escapeHtml(event)}`:''}${decision.timeframe_conflict?' · 日线与周线结构方向冲突':''}</div>${status==='INSUFFICIENT_EVIDENCE'?`<div class="warning">${escapeHtml(decision.assessment_notice||'当前信息不足，不能将旧计划视为有效状态。')}</div>`:''}${dataWarningsMarkup(decision.data_warnings)}${planReference}</section>`;
  }
  const reasons=(decision.reason_codes||[]).map(reasonLabel).map(escapeHtml).join('；')||'风险与空间条件通过',reasonsHtml=escapeHtml(reasons);
  const planReference=plan?`<details class="legacy-plan-reference"><summary>旧交易计划参考（不代表当前评估）</summary>${planMarkup(plan)}</details>`:'';
  return `<section class="decision-box decision-${escapeHtml(decision.action)}"><div class="decision-heading"><span class="action-pill action-${escapeHtml(decision.action)}">${escapeHtml(actionLabel(decision.action))}</span><b>${escapeHtml(decision.explanation||'')}</b></div><div>日线/周线确认数：${decision.rank?.confirmed_timeframes??0} · 方向冲突：${decision.timeframe_conflict?'是':'否'}</div><div>原因：${reasonsHtml}</div>${dataWarningsMarkup(decision.data_warnings)}${planReference}</section>`;
}
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
function zoomChart(direction,anchorRatio=.5){
  if(!lastAnalysis||!chartBarCount)return;
  const next=Math.max(Math.min(20,chartBarCount),Math.min(chartBarCount,Math.round(chartWindowBars*(direction==='in'?.9:1/.9))));
  if(chartCenter===null)chartCenter=Math.max(0,chartBarCount-chartWindowBars/2);
  chartCenter+=((anchorRatio-.5)*chartWindowBars)-((anchorRatio-.5)*next);
  chartCenter=Math.max(next/2,Math.min(chartBarCount-next/2,chartCenter));
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
    const timeframeMains=CaisenScanFilters.mainPatterns({patterns}),activeMains=Object.fromEntries(Object.entries(timeframeMains).filter(([,pattern])=>!isTerminalPattern(pattern))),primary=activeMains[data.timeframe]||patterns.find(pattern=>!isTerminalPattern(pattern));
    const detailPatterns=[...new Set([primary,activeMains['1d'],activeMains['1w']].filter(Boolean))],completedPatterns=patterns.filter(isTerminalPattern);
    const activeCards=detailPatterns.length?detailPatterns.map(patternCard).join(''):'<p>当前没有有效形态。周线若缺少日历会显示预览K线，但不产生正式周线信号。</p>';
    const completedCards=completedPatterns.length?`<details class="completed-patterns"><summary>已结束的历史形态（${completedPatterns.length}）</summary>${completedPatterns.map(patternCard).join('')}</details>`:'';
    selectedPattern=primary||null;
    $('detailRows').innerHTML=decisionMarkup(data.decision)+activeCards+completedCards;
    if(data.decision?.timeframe_conflict)$('detailRows').insertAdjacentHTML('afterbegin','<p class="conflict-banner">日线与周线方向冲突：请分别查看两周期结构。</p>');
    const bars=data._bars||[];chartBarCount=bars.length;chartWindowBars=chartVisibleCount($('chartRange').value,data.timeframe,bars.length);
    renderDetailChart(data,chartWindowBars);
    if(selectedId)await showPattern(selectedId);
  }catch(error){$('detailRows').innerHTML=`<div class="error-box">分析失败：${escapeHtml(error.message)}</div>`}
}
function renderDetailChart(data,visibleCount){
  const allBars=data._bars||[];chartBarCount=allBars.length;
  const first=chartCenter===null?Math.max(0,allBars.length-visibleCount):Math.max(0,Math.min(allBars.length-visibleCount,Math.floor(chartCenter-visibleCount/2)));chartFirst=first;
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
    const mainShape=Boolean(selectedPattern.is_main_structure||selectedPattern.selected_main||(selectedPattern.evidence||[]).some(item=>item.code==='main_structure'&&item.value?.role==='main'));
    const overlays=mainShape?CaisenPatternChart.mainOverlays(geometry):geometry;
    const visible=p=>p.index>=first&&p.index<first+bars.length;
    series.splice(1,series.length-1);
    const anchorData=geometry.anchors.filter(visible).map((p,index)=>({value:[p.index-first,p.price],name:p.name,label:{position:index%2?'top':'bottom'}}));
    series.push({type:'line',name:patternName(selectedPattern)+'形状',data:anchorData,showSymbol:false,lineStyle:{width:2,color:'#d28a18'},z:5});
    series.push({type:'scatter',name:'结构点',data:anchorData,symbol:'circle',symbolSize:10,itemStyle:{color:'#c98212'},label:{show:true,formatter:p=>p.name,color:'#7d4f05',backgroundColor:'rgba(255,255,255,.92)',padding:3},z:7});
    for(const boundary of overlays.boundaries)series.push({type:'line',name:boundary.name,data:CaisenPatternChart.clipPoints(boundary.points,first,first+bars.length-1).map(p=>[p.index-first,p.price]),showSymbol:false,lineStyle:{type:boundary.type||'dashed',color:boundary.color||'#176b87',width:boundary.width||2}});
    const markers=overlays.markers.filter(visible);
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
          {type:'inside',xAxisIndex:[0,1],start:0,end:100,zoomOnMouseWheel:false,moveOnMouseMove:true,moveOnMouseWheel:false,preventDefaultMouseMove:true},
          {type:'slider',xAxisIndex:[0,1],start:0,end:100,bottom:2,height:17,show:true,brushSelect:false,showDataShadow:false,borderColor:'#d7dfe5',fillerColor:'rgba(23,107,135,.16)',handleStyle:{color:'#176b87'}}
        ],
        series
      },true);
      requestAnimationFrame(()=>detailChart?.resize());
}
let scanSettings={};
async function startScan(){if(scanTimer)clearInterval(scanTimer);const kinds=[$('scanStock').checked?'stock':null,$('scanEtf').checked?'etf':null].filter(Boolean);const dateFrom=$('scanDateFrom').value,dateTo=$('scanDateTo').value,limit=$('scanLimit').value.trim();if(!kinds.length){$('scanStatus').textContent='请至少选择股票或 ETF。';return}if(dateFrom&&dateTo&&dateFrom>dateTo){$('scanStatus').textContent='起始日期不能晚于截止日期。';return}scanSettings={kinds,pattern_types:[...$('scanPatternTypes').selectedOptions].map(option=>option.value)};$('scanBtn').disabled=true;scanRows=[];visibleScanRows=[];scanShown=0;$('results').innerHTML='';$('scanResultFilters').hidden=true;$('scanSummary').textContent='';$('scanFilterSummary').textContent='';$('scanStatus').textContent='正在提交扫描…';const params={timeframes:[...$('scanTimeframes').selectedOptions].map(option=>option.value),kinds};if(limit)params.limit=Number(limit);if(dateFrom)params.date_from=dateFrom;if(dateTo)params.date_to=dateTo;try{const job=await fetchJson('/api/v1/scans',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});$('scanStatus').textContent=`扫描任务已启动：${job.job_id}。正在逐个检查所选本地行情…`;scanTimer=setInterval(async()=>{try{const current=await fetchJson('/api/v1/jobs/'+encodeURIComponent(job.job_id));$('scanStatus').textContent=`${current.status==='completed'?'完成':current.status==='partial_failed'?'部分失败':current.status==='failed'?'失败':'扫描中'} · ${Math.round((current.progress||0)*100)}% · ${current.scanned||0}/${current.total||0}`;if(current.status==='completed'||current.status==='partial_failed'){clearInterval(scanTimer);scanTimer=null;$('scanBtn').disabled=false;renderScan(current.results,current.inventory);if(current.status==='partial_failed')$('scanStatus').textContent+=' · 部分标的失败，完整保存结果可在历史结果中查询'}else if(current.status==='failed'){clearInterval(scanTimer);scanTimer=null;$('scanBtn').disabled=false;$('results').innerHTML=`<div class="error-box">扫描失败：${escapeHtml(current.error||'未知错误')}</div>`}}catch(error){$('scanStatus').textContent='读取扫描进度失败：'+error.message}},2000)}catch(error){$('scanBtn').disabled=false;$('scanStatus').textContent='扫描无法启动：'+error.message}}

$('analyzeBtn').onclick=analyze;$('scanBtn').onclick=startScan;$('chartRange').onchange=()=>applyChartRange($('chartRange').value);$('chart').addEventListener('wheel',event=>{if(!lastAnalysis||!chartBarCount)return;event.preventDefault();const rect=$('chart').getBoundingClientRect(),ratio=Math.max(0,Math.min(1,(event.clientX-rect.left-120)/(rect.width-190)));zoomChart(event.deltaY<0?'in':'out',ratio)},{passive:false});document.querySelectorAll('[data-chart-action]').forEach(button=>button.onclick=()=>{const action=button.dataset.chartAction;if(action==='in'||action==='out')zoomChart(action);else if(action==='all'){$('chartRange').value='all';applyChartRange('all')}else{selectedPattern=null;chartCenter=null;document.querySelectorAll('.pattern-card.selected').forEach(card=>{card.classList.remove('selected');card.querySelector('[data-show-pattern]')?.setAttribute('aria-pressed','false')});$('chartRange').value=lastAnalysis?.timeframe==='1w'?'60':'120';applyChartRange($('chartRange').value)}});window.addEventListener('resize',()=>detailChart?.resize());
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
let dailyBacktestTimer=null;
function dailyBacktestConfigLabel(job){
  const p=job.params||job.effective_parameters||{},rules=job.recognition_rules||p.recognition_rules||{};
  const cash=job.trading_rules?.initial_cash_per_symbol??p.trading_rules?.initial_cash_per_symbol??p.initial_cash_per_symbol??'—';
  const strategy=job.strategy_version||p.strategy_version||p.strategy||'—';
  const selection=rules.neckline_selection_rule_version||'选线规则待记录',phase=rules.consumed_boundary_phase_rule||'结构阶段规则待记录';
  const workers=p.execution_mode==='spawn'?`${p.workers??'—'} 个进程（spawn）`:p.execution_mode==='in_process'&&Number(p.workers)>1?`${p.workers} 个线程（in_process）`:`${p.workers??'—'} 个单进程`;
  return `${workers} · ${p.recognizer||'—'} / ${job.engine_version||p.engine_version||'引擎待记录'} · 策略 ${strategy} · 颈线候选反转 ${rules.candidate_reversal_atr_multiple??'—'}×ATR / 突破收盘缓冲 ${rules.breakout_close_buffer_previous_atr_multiple??'—'}×前一完成ATR · ${selection} · ${phase} · 每标的 ${cash}`;
}
function renderDailyBacktest(job){
  const summary=job.summary||{},labels={trade_count:'成交笔数',signal_count:'信号数',win_rate:'胜率',total_return:'总收益率',profit_factor:'盈亏比',max_drawdown:'最大回撤',security_count:'标的数',completed:'完成账户'};
  const cards=Object.entries(summary).filter(([,value])=>value===null||['string','number','boolean'].includes(typeof value)).slice(0,12).map(([key,value])=>`<div><span>${escapeHtml(labels[key]||key.replaceAll('_',' '))}</span><b>${escapeHtml(value??'—')}</b></div>`).join('');
  const report=job.report_url?`<a class="button-link" href="${escapeHtml(job.report_url)}" target="_blank" rel="noopener">打开完整回测报告</a>`:'';
  const params=job.effective_parameters||job.params||{},recognition=job.recognition_rules||params.recognition_rules||{},trading=job.trading_rules||params.trading_rules||{};
  const versions=`策略 ${job.strategy_version||params.strategy_version||params.strategy||'—'} · 识别器 ${params.recognizer||'—'} · 执行引擎 ${job.engine_version||params.engine_version||'—'} · 规则 ${job.source_rule_version||params.source_rule_version||'—'}`;
  const execution=params.execution_mode==='spawn'?`${params.workers??'—'} 个进程（spawn，${params.worker_source||'—'}）`:params.execution_mode==='in_process'&&Number(params.workers)>1?`${params.workers} 个线程（in_process，${params.worker_source||'—'}）`:`${params.workers??'—'} 个单进程（${params.worker_source||'—'}）`;
  const rules=`颈线候选反转 ${recognition.candidate_reversal_atr_multiple??'—'}×ATR · 突破收盘缓冲 ${recognition.breakout_close_buffer_previous_atr_multiple??'—'}×前一完成 ATR · 选线 ${recognition.neckline_selection_rule_version||'—'} · 阶段 ${recognition.consumed_boundary_phase_rule||'—'} · 每标的 ${trading.initial_cash_per_symbol??params.initial_cash_per_symbol??'—'} · 每手 ${trading.lot_size??100}`;
  const experiment=recognition.recognition_experiment?`实验 ${recognition.recognition_experiment}`:'基线识别（未启用 min3 / 开盘价过滤实验）';
  const output=params.output_dir?`输出目录 ${params.output_dir}`:'';
  $('dailyBacktestResults').innerHTML=`${cards?`<div class="fixed-report-summary">${cards}</div>`:''}<p>${escapeHtml(summary.note||'')}</p><p class="muted">${escapeHtml(versions)}</p><p class="muted">实际配置：${escapeHtml(execution)} · ${escapeHtml(rules)} · ${escapeHtml(experiment)}</p>${output?`<p class="muted">${escapeHtml(output)}</p>`:''}${report}${job.error?`<div class="error-box">${escapeHtml(job.error)}</div>`:''}`;
}
async function loadDailyBacktestHistory(){
  try{
    const data=await fetchJson('/api/v1/backtests?page=1'),items=data.items||[];
    $('dailyBacktestHistory').innerHTML=items.length?items.map(job=>{
      const status={queued:'排队中',running:'运行中',completed:'完成',partial_failed:'部分失败',failed:'失败',interrupted:'已中断'}[job.status]||job.status;
      return `<article class="history-result-row"><b>${escapeHtml(job.created_at||job.job_id)}</b><span>${escapeHtml(status)} · ${Math.round((job.progress||0)*100)}% · ${job.completed??0}/${job.total??0}</span><span class="muted">${escapeHtml(job.universe?.id||job.params?.universe_id||'daily_800_stock_200_etf_v1')} · ${job.params?.workers??'—'} worker · ${escapeHtml(job.params?.recognizer||'识别器待记录')} · 交易 ${job.summary?.trade_count??'—'} · ${escapeHtml(job.strategy_version||job.params?.strategy||'策略版本待记录')} · ${escapeHtml(job.engine_version||'引擎版本待记录')}</span>${job.report_url?`<a href="${escapeHtml(job.report_url)}" target="_blank" rel="noopener">打开已保存报告</a>`:''}${job.error?`<div class="error-box">${escapeHtml(job.error)}</div>`:''}</article>`
    }).join(''):'<p class="muted">尚无日线回测任务。</p>';
  }catch(error){$('dailyBacktestHistory').innerHTML=`<div class="error-box">无法读取历史回测：${escapeHtml(error.message)}</div>`}
}
async function pollDailyBacktest(jobId){
  const job=await fetchJson(`/api/v1/backtests/${encodeURIComponent(jobId)}`),label={queued:'排队中',running:'运行中',completed:'完成',partial_failed:'部分失败',failed:'失败',interrupted:'已中断'}[job.status]||job.status;
  const current=job.current||{},phase={loading:'读取行情',recognition:'形态识别',recognition_complete:'识别完成',backtrader:'模拟撮合',saving:'保存结果'}[current.phase];
  const bars=current.bars_total?` · K线 ${current.bars_completed||0}/${current.bars_total}${current.current_date?`（${current.current_date}）`:''}`:'';
  const detail=current.symbol?` · ${current.symbol}${phase?` · ${phase}`:''}${bars}`:'';
  const config=` · ${dailyBacktestConfigLabel(job)}`;
  $('dailyBacktestStatus').textContent=`${label} · ${Math.round((job.progress||0)*100)}% · ${job.completed||0}/${job.total||0}${config}${detail}${job.error?` · ${job.error}`:''}`;
  if(['completed','partial_failed','failed','interrupted'].includes(job.status)){
    if(dailyBacktestTimer)clearInterval(dailyBacktestTimer);dailyBacktestTimer=null;$('dailyBacktestBtn').disabled=false;renderDailyBacktest(job);loadDailyBacktestHistory();return true;
  }
  return false;
}
async function startDailyBacktest(){
  if(dailyBacktestTimer)clearInterval(dailyBacktestTimer);$('dailyBacktestBtn').disabled=true;$('dailyBacktestResults').innerHTML='';$('dailyBacktestStatus').textContent='正在提交固定股票与 ETF 范围的日线回测…';
  try{
    const job=await fetchJson('/api/v1/backtests',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    $('dailyBacktestStatus').textContent=`任务 ${job.job_id} 已启动 · ${dailyBacktestConfigLabel(job)} · 输出 ${job.params?.output_dir||'待生成'}。`;
    if(!await pollDailyBacktest(job.job_id))dailyBacktestTimer=setInterval(async()=>{try{if(await pollDailyBacktest(job.job_id))return}catch(error){$('dailyBacktestStatus').textContent=`读取回测进度失败：${error.message}`}},2000);
  }catch(error){$('dailyBacktestStatus').textContent=`回测无法启动：${error.message}`;$('dailyBacktestBtn').disabled=false}
}

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
    selectedPattern=pattern;chartWindowBars=Math.min(chartBarCount,Math.max(20,chartBarCount-geometry.first));chartCenter=geometry.first+chartWindowBars/2;
    $('chartRange').value='custom';renderDetailChart(lastAnalysis,chartWindowBars);
    document.querySelectorAll('.pattern-card').forEach(card=>{const active=card.dataset.patternId===id;card.classList.toggle('selected',active);card.querySelector('[data-show-pattern]')?.setAttribute('aria-pressed',String(active));if(active&&card.closest('details'))card.closest('details').open=true});
    $('chart').scrollIntoView({behavior:'instant',block:'center'});
  }catch(error){$('chartNote').textContent=error.message}
}
$('detailRows').addEventListener('click',event=>{const button=event.target.closest('[data-show-pattern]');if(button)showPattern(button.dataset.showPattern)});
$('dailyBacktestBtn').onclick=startDailyBacktest;$('dailyBacktestHistoryRefresh').onclick=loadDailyBacktestHistory;document.querySelector('[data-panel=\"daily-backtest\"]').addEventListener('click',()=>loadDailyBacktestHistory());
