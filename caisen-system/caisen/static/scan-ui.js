(function(){
  let scanId='',page=1,action='',assessmentTimer=null,assessmentRunId='',assessmentWasRunning=false;
  const escape=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  function filters(){return {action,patterns:[...$('scanShapeFilter').selectedOptions].map(item=>item.value)}}
  function query(pageNumber,values={}){const q=new URLSearchParams({page:String(pageNumber),sort:values.sort||'rank',order:values.order||'asc'});for(const [key,value] of Object.entries({action:values.action,pattern_types:values.patterns?.join(','),kind:values.kind,q:values.q,category:values.category,timeframes:values.timeframes?.join(',')}))if(value)q.set(key,value);return q}
  async function results(id,pageNumber,values){return fetchJson(`/api/v1/scans/${encodeURIComponent(id)}/results?${query(pageNumber,values)}`)}
  function formed(pattern){return pattern.formed_at||pattern.formation_date||pattern.anchor_start_date||pattern.key_points?.pivot_1_date||'—'}
  function compact(pattern){if(!pattern)return '<span class="muted">暂无主形态</span>';const date=pattern.confirmation_date||pattern.confirmed_date||((pattern.evidence||[]).find(item=>item.code==='confirmed_date')||{}).value||'待确认';return `<span class="pattern-chip">${escape(pattern.pattern_type==='FOCUS_NECKLINE'?'颈线聚焦':names[pattern.pattern_type]||pattern.pattern_type||'暂无形态')} · ${escape(stateLabel(pattern))}</span><span class="muted">形成 ${escape(formed(pattern))} · 确认 ${escape(date)}</span>`}
  function rowMarkup(row){const patterns=CaisenScanFilters.mainPatterns(row),notice=row.assessment_notice?`<span class="assessment-notice">评估提示：${escape(row.assessment_notice)}</span>`:'';return `<article class="history-result-row" data-code="${escape(row.code)}" data-kind="${escape(row.kind)}" tabindex="0" role="link"><div class="result-row-heading"><b>${escape(row.name||row.code)}</b><span>${escape(row.code)} · ${row.kind==='etf'?'ETF':'股票'}</span><span class="action-pill action-${escape(row.action||'WAIT')}">${escape(actionLabel(row.action))}</span></div><div class="result-patterns"><span><b>日线</b> ${compact(patterns['1d'])}</span><span><b>周线</b> ${compact(patterns['1w'])}</span></div>${notice}</article>`}
  window.patternCard=pattern=>{const facts=CaisenPatternChart.patternDetails(lastAnalysis,pattern),formation=formed(pattern),confirmed=pattern.confirmation_date||pattern.confirmed_date||((pattern.evidence||[]).find(item=>item.code==='confirmed_date')||{}).value||'待确认',points=facts.points.map(point=>`${escape(point.name)} ${escape(point.date||'日期未记录')} @ ${fmt(point.price)}`).join(' · ')||'结构点日期/价格未记录',evidence=(pattern.evidence||[]).map(item=>escape(evidenceLabel(item))).join(' · ');return `<article class="card pattern-card" data-pattern-id="${escape(pattern.pattern_id||'')}"><div><b>${escape(pattern.code)} · ${escape(pattern.pattern_type==='FOCUS_NECKLINE'?'颈线聚焦':names[pattern.pattern_type]||pattern.pattern_type||'形态')}</b> <span class="tag">${escape(pattern.timeframe)} · ${escape(stateLabel(pattern))}</span><div class="muted">形成 ${escape(formation)} · ${pattern.confirmation_date||pattern.confirmed_date?'价格确认':'结构点确认'} ${escape(confirmed)}</div><div class="evidence">边界有效起点 ${escape(facts.boundaryStart||'日期未记录')} · 冻结日 ${escape(facts.freezeDate||'日期未记录')}<br>${points}</div><details class="pattern-evidence-fold"><summary>其他证据与结构信息</summary><div class="evidence">${evidence||'暂无附加证据'}</div></details></div><button type="button" data-show-pattern="${escape(pattern.pattern_id||'')}" aria-pressed="false">查看该形态</button></article>`};
  function bind(container){container.querySelectorAll('.history-result-row:not([data-bound])').forEach(row=>{row.dataset.bound='1';const open=()=>openTicker(row.dataset.code,row.dataset.kind);row.addEventListener('click',open);row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}})})}
  async function loadScanPage(nextPage=1){if(!scanId)return;page=nextPage;$('results').innerHTML='<p class="muted">正在读取本页保存结果…</p>';try{const data=await results(scanId,page,filters());$('results').innerHTML=data.items.length?data.items.map(rowMarkup).join(''):'<div class="scan-empty">没有符合筛选条件的保存结果。</div>';bind($('results'));$('scanPage').textContent=`第 ${page} 页 · 共 ${data.total} 条 · 每页 ${data.page_size||50} 条`;$('scanPrev').disabled=page<=1;$('scanNext').disabled=page*(data.page_size||50)>=data.total;$('scanFilterSummary').textContent=`筛选后共 ${data.total} 条保存结果`}catch(error){$('results').innerHTML=`<div class="error-box">无法读取扫描结果：${escape(error.message)}</div>`}}
  function showActionChips(){document.querySelectorAll('#scanStatusChips [data-action]').forEach(button=>{const active=button.dataset.action===action;button.classList.toggle('selected',active);button.setAttribute('aria-pressed',String(active))})}
  function syncPatternGroup(selectId,groupId,onChange){const select=$(selectId),group=$(groupId);group.innerHTML=[...select.options].map(option=>CaisenScanFilters.patternChipMarkup(option.value,option.textContent,option.selected,option.disabled)).join('');group.querySelectorAll('button').forEach(button=>button.onclick=()=>{const option=[...select.options].find(item=>item.value===button.dataset.value);if(!option||option.disabled)return;option.selected=!option.selected;button.classList.toggle('selected',option.selected);button.setAttribute('aria-pressed',String(option.selected));onChange()})}
  function applyPatternMigration(migration){
    const families=migration?.families||{},known=Object.keys(families).length>0;
    for(const [selectId,historical] of [['scanPatternTypes',false],['scanShapeFilter',false],['historyPatterns',true]]){
      const select=$(selectId);
      for(const option of select.options){
        const status=families[option.value]?.status;
        option.textContent=CaisenScanFilters.patternSupportLabel(names[option.value]||option.value,status,historical);
        option.disabled=!historical&&(!known||status!=='supported');
        if(option.disabled)option.selected=false;
      }
    }
    const supported=known?Object.entries(families).filter(([type,item])=>item.status==='supported'&&names[type]).map(([type])=>names[type]):[];
    const unsupported=known?Object.entries(families).filter(([type,item])=>item.status==='unsupported'&&names[type]).map(([type])=>names[type]):[];
    $('patternSupportNote').textContent=known?`当前扫描支持：${supported.join('、')||'暂无'}。${unsupported.length?`${unsupported.join('、')}暂未支持，扫描筛选已禁用；历史结果仍可查询。`:''}`:'正在读取当前形态支持状态；确认前暂不开放形态筛选。';
    syncPatternGroup('scanShapeFilter','scanShapeChips',scanPageFilters);syncPatternGroup('historyPatterns','historyPatternChips',()=>historyPage(1));
  }
  async function scanPageFilters(){await loadScanPage(1)}
  async function scan(params){if(window.scanTimer)clearInterval(window.scanTimer);const kinds=[$('scanStock').checked?'stock':null,$('scanEtf').checked?'etf':null].filter(Boolean),from=$('scanDateFrom').value,to=$('scanDateTo').value,limit=$('scanLimit').value.trim();if(!kinds.length){$('scanStatus').textContent='请至少选择股票或 ETF。';return}if(from&&to&&from>to){$('scanStatus').textContent='起始日期不能晚于截止日期。';return}window.scanSettings={kinds,pattern_types:[...$('scanPatternTypes').selectedOptions].map(item=>item.value)};$('scanBtn').disabled=true;$('scanResultFilters').hidden=true;$('results').innerHTML='';$('scanStatus').textContent='正在提交扫描…';params={timeframes:[...$('scanTimeframes').selectedOptions].map(item=>item.value),kinds};if(limit)params.limit=Number(limit);if(from)params.date_from=from;if(to)params.date_to=to;try{const job=await fetchJson('/api/v1/scans',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});scanId=job.job_id;$('scanStatus').textContent=`扫描任务已启动：${job.job_id}。正在检查所选本地行情…`;window.scanTimer=setInterval(async()=>{try{const current=await fetchJson('/api/v1/jobs/'+encodeURIComponent(job.job_id));$('scanStatus').textContent=`${current.status==='completed'?'完成':current.status==='partial_failed'?'部分失败':current.status==='failed'?'失败':'扫描中'} · ${Math.round((current.progress||0)*100)}% · ${current.scanned||0}/${current.total||0}`;if(['completed','partial_failed'].includes(current.status)){clearInterval(window.scanTimer);window.scanTimer=null;$('scanBtn').disabled=false;$('scanResultFilters').hidden=false;action='';showActionChips();[...$('scanShapeFilter').options].forEach(option=>option.selected=false);(window.scanSettings.pattern_types||[]).forEach(type=>{const option=[...$('scanShapeFilter').options].find(item=>item.value===type);if(option)option.selected=true});syncPatternGroup('scanShapeFilter','scanShapeChips',scanPageFilters);const range=current.inventory?.requested_window||{};$('scanSummary').innerHTML=`<div class="scan-summary"><b>扫描完成</b> · 请求区间 ${escape(range.from||'最早可用')} 至 ${escape(range.to||'最新可用')} · 股票 ${current.inventory?.stock_count??'—'} · ETF ${current.inventory?.etf_count??'—'} · 每页 50 条。</div>`;await loadScanPage(1);if(current.status==='partial_failed')$('scanStatus').textContent+=' · 部分标的失败'}else if(current.status==='failed'){clearInterval(window.scanTimer);window.scanTimer=null;$('scanBtn').disabled=false;$('results').innerHTML=`<div class="error-box">扫描失败：${escape(current.error||'未知错误')}</div>`}}catch(error){$('scanStatus').textContent='读取扫描进度失败：'+escape(error.message)}},2000)}catch(error){$('scanBtn').disabled=false;$('scanStatus').textContent='扫描无法启动：'+escape(error.message)}}
  function stopAssessmentPolling(){if(assessmentTimer){clearTimeout(assessmentTimer);assessmentTimer=null}}
  function showAssessmentProgress(progress,items,pageNumber,id){
    if(assessmentRunId!==id){stopAssessmentPolling();assessmentRunId=id;assessmentWasRunning=false}
    const status=progress?.status||'unknown',notices=(items||[]).filter(item=>item.assessment_notice).length;
    if(status==='running'||status==='scan_running'){
      assessmentWasRunning=true;
      $('historyAssessmentStatus').innerHTML=status==='scan_running'
        ?'<div class="assessment-pending">历史评估尚未启动：扫描任务仍在运行。此时筛选结果尚未完整。</div>'
        :`<div class="assessment-pending">旧结果后台重评进度：${progress.processed??0}/${progress.total??0}。完成前新旧结果可能混合，当前行动、形态筛选与排序可能不完整。${notices?` 本页 ${notices} 条仍保留原始行动。`:''}</div>`;
      stopAssessmentPolling();assessmentTimer=setTimeout(()=>{if($('historyRun').value===id)historyPage(pageNumber,{silent:true})},1800);
    }else if(status==='completed'){
      assessmentWasRunning=false;
      $('historyAssessmentStatus').innerHTML=`<div class="assessment-complete">历史评估完成：${progress.processed??0}/${progress.total??0}。${notices?`本页 ${notices} 条因证据不足无法重评，保留原始行动。`:'当前页没有证据不足提示。'}</div>`;
      stopAssessmentPolling();
    }else if(status==='failed'){
      $('historyAssessmentStatus').innerHTML=`<div class="assessment-failed">历史评估失败：${escape(progress.error||'未知错误')}。结果包含未完成评估项，当前筛选可能不完整。</div>`;stopAssessmentPolling();
    }else{
      $('historyAssessmentStatus').innerHTML=`<div class="assessment-pending">历史评估状态：${escape(status)}。在状态明确前，筛选结果可能不完整。</div>`;
      stopAssessmentPolling();assessmentTimer=setTimeout(()=>{if($('historyRun').value===id)historyPage(pageNumber,{silent:true})},1800);
    }
  }
  async function historyPage(pageNumber=1,options={}){
    const id=$('historyRun').value;if(!id)return;if(assessmentRunId!==id){stopAssessmentPolling();assessmentRunId=id;assessmentWasRunning=false;$('historyAssessmentStatus').textContent='正在读取历史评估状态…'}window.historyRunId=id;window.historyPage=pageNumber;
    const values={q:$('historySearch').value.trim(),kind:$('historyKind').value,category:$('historyCategory').value.trim(),action:$('historyAction').value,patterns:[...$('historyPatterns').selectedOptions].map(item=>item.value),timeframes:[...$('historyTimeframes').selectedOptions].map(item=>item.value),sort:$('historySort').value,order:$('historyOrder').value};
    if(!options.silent)$('historyResults').innerHTML='<p class="muted">正在读取保存结果…</p>';
    try{
      const [data,runs]=await Promise.all([results(id,pageNumber,values),fetchJson('/api/v1/scans?page=1')]);
      const run=runs.items.find(item=>item.run_id===id),params=run?.params||{},range=params._legacy_import?'未记录':`${params.date_from||'最早可用'} 至 ${params.date_to||'最新可用'}`;
      $('historySummary').innerHTML=`<div class="scan-summary"><b>${escape(run?.status||'未知')}</b> · 任务 ${escape(id)} · 请求区间 ${escape(range)} · 扫描进度 ${run?.scanned??0}/${run?.total??0} · 保存结果 ${data.total}<p>${escape(data.detail_notice||'')}</p></div>`;
      $('historyResults').innerHTML=data.items.length?data.items.map(rowMarkup).join(''):'<div class="scan-empty">没有符合筛选条件的保存结果。</div>';bind($('historyResults'));
      $('historyPage').textContent=`第 ${pageNumber} 页 · 共 ${data.total} 条 · 每页 ${data.page_size||50} 条`;$('historyPrev').disabled=pageNumber<=1;$('historyNext').disabled=pageNumber*(data.page_size||50)>=data.total;
      const refreshAfterCompletion=data.assessment_progress?.status==='completed'&&assessmentWasRunning;
      showAssessmentProgress(data.assessment_progress,data.items,pageNumber,id);
      if(refreshAfterCompletion&&!options.finalRefresh){stopAssessmentPolling();await historyPage(pageNumber,{silent:true,finalRefresh:true})}
    }catch(error){$('historyResults').innerHTML=`<div class="error-box">无法读取保存结果：${escape(error.message)}</div>`;$('historyAssessmentStatus').innerHTML=`<div class="assessment-failed">历史评估进度读取失败：${escape(error.message)}。</div>`;stopAssessmentPolling()}
  }
  window.startScan=scan;window.loadHistoryResults=historyPage;window.applyScanFilters=scanPageFilters;
  window.applyScanPatternMigration=applyPatternMigration;applyPatternMigration(window.CaisenPatternMigration);
  $('scanBtn').onclick=scan;$('scanPrev').onclick=()=>loadScanPage(page-1);$('scanNext').onclick=()=>loadScanPage(page+1);$('historyPrev').onclick=()=>historyPage((window.historyPage||1)-1);$('historyNext').onclick=()=>historyPage((window.historyPage||1)+1);
  document.querySelectorAll('#scanStatusChips [data-action]').forEach(button=>button.onclick=()=>{action=button.dataset.action;showActionChips();scanPageFilters()});$('resetScanFilters').onclick=()=>{action='';[...$('scanShapeFilter').options].forEach(item=>item.selected=false);applyPatternMigration(window.CaisenPatternMigration);showActionChips();scanPageFilters()};$('buyOnlyBtn').onclick=()=>{action=action==='CONSIDER_BUY'?'':'CONSIDER_BUY';showActionChips();scanPageFilters()};
})();
