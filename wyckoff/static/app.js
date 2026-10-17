const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ---------- tabs ----------
$$('.tab').forEach(b => b.onclick = () => {
  $$('.tab').forEach(x => x.classList.remove('active'));
  $$('.view').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  $('#view-' + b.dataset.view).classList.add('active');
  if (b.dataset.view === 'data') refreshJobs();
});

async function api(path, opts) {
  const r = await fetch(path, opts ? { headers: {'Content-Type':'application/json'}, ...opts } : undefined);
  if (!r.ok) throw new Error((await r.text()).slice(0, 200));
  return r.json();
}

// ---------- 榜单 ----------
let scanRows = [];
async function loadScan() {
  $('#scan-status').textContent = '载入…';
  try {
    const d = await api('/api/scan');
    scanRows = d.rows || [];
    $('#scan-status').textContent = d.available ? `共${d.count}只 · 更新于最近一次扫描` : (d.hint || '');
    renderScan();
  } catch (e) { $('#scan-status').textContent = '错误: ' + e.message; }
}
function renderScan() {
  const f = $('#scan-filter').value;
  let rows = scanRows;
  if (f === 'buy') rows = rows.filter(r => r.action === 'BUY');
  else if (f === 'watch') rows = rows.filter(r => r.buy_score >= 1 || r.sell_score >= 1);
  else if (f !== 'all') rows = rows.filter(r => r.kind === f);
  const act = (r) => r.action === 'BUY' ? '<b class="up">买入</b>' : r.action === 'SELL' ? '<b class="down">卖出</b>' : '观望';
  $('#scan-table tbody').innerHTML = rows.slice(0, 300).map(r => `
    <tr class="${r.action==='BUY'?'row-buy':''}">
      <td><a href="#" data-code="${r.code}" data-kind="${r.kind}" class="link">${r.code}</a></td>
      <td>${r.name||''}</td><td>${r.kind==='etf'?'ETF':'股票'}</td>
      <td>${r.close}</td><td>${act(r)}</td>
      <td>${(r.buy_score||0).toFixed(1)}</td><td>${(r.sell_score||0).toFixed(1)}</td>
      <td>${phaseZh(r.phase)}</td><td>${r.trend===1?'↑多':r.trend===-1?'↓空':'—'}</td>
      <td>${r.entry_type||''}</td><td class="small">${r.reason||''}</td><td class="muted">${r.date||''}</td>
    </tr>`).join('');
  $$('#scan-table a.link').forEach(a => a.onclick = (e) => {
    e.preventDefault();
    $$('.tab')[1].click();
    $('#sym-input').value = a.dataset.code;
    analyzeSymbol();
  });
}
$('#scan-filter').onchange = renderScan;
$('#btn-full-scan').onclick = async () => {
  $('#scan-status').textContent = '扫描任务已提交（全市场约5000+标的，需数分钟）…';
  const j = await api('/api/scan/run', { method: 'POST', body: JSON.stringify({ kind:'both', sample:0, workers: Math.min(12, navigator.hardwareConcurrency||4) }) });
  pollJob(j.job_id, (job) => {
    if (job.status === 'done') { $('#scan-status').textContent = '扫描完成'; loadScan(); }
    else if (job.status === 'error') $('#scan-status').textContent = '扫描失败: ' + job.error;
    else $('#scan-status').textContent = '扫描中…';
  });
};

function phaseZh(p) {
  return {accumulation:'吸筹',markup:'上涨',distribution:'派发',markdown:'下跌'}[p]||p||'';
}

// ---------- 轮询任务 ----------
function pollJob(id, cb) {
  const t = setInterval(async () => {
    try {
      const job = await api('/api/jobs/' + id);
      cb(job);
      if (job.status !== 'running') clearInterval(t);
    } catch (e) { clearInterval(t); cb({status:'error', error:e.message}); }
  }, 2500);
}

// ---------- 个股 ----------
let chart = null;
async function analyzeSymbol() {
  const code = $('#sym-input').value.trim();
  if (!code) return;
  $('#sym-status').textContent = '分析中…';
  try {
    const bars = $('#sym-bars').value;
    const d = await api(`/api/symbol/${code}?kind=auto&bars=${bars}`);
    $('#sym-status').textContent = '';
    renderSymbol(d);
  } catch (e) { $('#sym-status').textContent = '错误: ' + e.message; }
}
$('#btn-go').onclick = analyzeSymbol;
$('#sym-input').onkeydown = e => { if (e.key === 'Enter') analyzeSymbol(); };

const BUY_EV = new Set(['SPRING','TEST','LPS','SOS','JOC','ST','SC','PS']);
function renderSymbol(d) {
  $('#sym-head').classList.remove('hidden');
  const l = d.latest || {};
  $('#sym-head').innerHTML = `
    <b>${d.code} ${d.name}</b> <span class="muted">${d.kind}</span> ·
    ${l.date} 收盘 ${l.close} → <b class="${l.action==='BUY'?'up':l.action==='SELL'?'down':''}">${{BUY:'买入',SELL:'卖出',HOLD:'观望'}[l.action]||l.action}</b>
    <span class="badge">阶段 ${phaseZh(l.phase)}</span>
    <span class="badge">诊断分 买${l.buy_score} / 卖${l.sell_score}（不控制交易）</span>
    ${l.stop?`<span class="badge">止损 ${l.stop}</span>`:''}
    ${l.targets&&l.targets.length?`<span class="badge">目标 ${l.targets.join(' / ')}</span>`:''}
    <div class="small muted">${l.reason||''}</div>`;

  const dom = $('#chart-kline');
  if (!chart || chart.getDom() !== dom) chart = echarts.init(dom);
  const evByType = {};
  d.events.forEach(e => {
    const isBuy = e.entry_allowed && !e.failed;
    (evByType[e.type] ||= []).push({ value: [e.bar, e.price], failed: e.failed, buy: isBuy });
  });
  const series = [{
    type: 'candlestick', data: d.candles,
    itemStyle: { color:'#e15b5b', color0:'#3d9b7a', borderColor:'#e15b5b', borderColor0:'#3d9b7a' },
    markLine: { symbol:'none', silent:true, label:{show:false}, data:(d.level_regions||[]).flatMap((l,i) => {
      const color=`hsl(${(i*137.508)%360},65%,42%)`;
      return [l.bottom,l.price,l.top].flatMap(p=>[
        [{coord:[l.start,p],lineStyle:{color,type:'dotted',opacity:.5}},{coord:[l.known,p]}],
        [{coord:[l.known,p],lineStyle:{color,type:'solid',opacity:.85}},{coord:[d.dates.length-1,p]}]
      ]);
    }) },
    markArea: {silent:true,label:{show:false},data:(d.level_regions||[]).map((l,i)=>[
      {coord:[l.start,l.bottom],itemStyle:{color:`hsla(${(i*137.508)%360},65%,42%,.08)`}},
      {coord:[d.dates.length-1,l.top]}
    ])}

  }];

  Object.entries(evByType).forEach(([type, pts]) => {
    series.push({
      type:'scatter', name:type, data: pts, symbol:'triangle',
      symbolRotate: pts[0].buy ? 0 : 180, symbolSize: 9,
      itemStyle:{ color: pts[0].buy ? '#3d9b7a' : '#e15b5b', opacity: 0.9 },
      label:{ show:true, formatter: type, position:'top', fontSize:9, color:'#556' }
    });
  });
  series.push({
    type:'bar', name:'成交量', xAxisIndex:1, yAxisIndex:1, data: d.volumes.map(x=>({
      value:x.v, itemStyle:{ color: x.up ? 'rgba(225,91,91,.55)' : 'rgba(61,155,122,.55)' }})),
    barWidth:'60%'
  });
  chart.setOption({
    animation:false,
    tooltip:{ trigger:'axis' },
    legend:{ top:0, textStyle:{fontSize:10}, data:['SPRING','TEST','SOS','JOC','SC','SOW','UT','BC'] },
    grid:[{left:50,right:12,top:28,height:'58%'},{left:50,right:12,top:'72%',height:'16%'}],
    xAxis:[
      {type:'category',data:d.dates,scale:true,axisLabel:{fontSize:10}},
      {type:'category',data:d.dates,gridIndex:1,axisLabel:{show:false}}
    ],
    yAxis:[{scale:true},{gridIndex:1,axisLabel:{show:false}}],
    dataZoom:[{type:'inside',xAxisIndex:[0,1]},{type:'slider',xAxisIndex:[0,1],bottom:0,height:14}],
    series
  }, true);

  $('#sym-events tbody').innerHTML = d.events.slice(-40).reverse().map(e =>
    `<tr><td>${e.date}</td><td><b>${e.type}</b>${e.kind?'('+zhKind(e.kind)+')':''}</td>
     <td>${e.price.toFixed(2)}</td><td>${e.quality.toFixed(2)}</td>
     <td>${e.failed?'<span class="down">已失败</span>':'有效'}</td></tr>`).join('');
  $('#sym-zones tbody').innerHTML = [
    ...(d.support_lines||[]).map(p=>({p,side:'支撑'})),
    ...(d.resistance_lines||[]).map(p=>({p,side:'阻力'}))
  ].sort((a,b)=>b.p-a.p).map(l=>`<tr><td>${l.side}</td><td>${l.p.toFixed(4)}</td></tr>`).join('');
  $('#sym-signals tbody').innerHTML = (d.signals||[]).slice(-30).reverse().map(s =>
    `<tr><td>${s.date}</td><td class="${s.action==='BUY'?'up':s.action==='SELL'?'down':''}">${s.action}</td>
     <td class="small">${s.reason}</td><td>${s.buy.toFixed(1)}/${s.sell.toFixed(1)}</td></tr>`).join('');
}
function zhKind(k){return {ideal:'理想',absorption:'吸收',confirmed:'已确认',panic:'候选',sc_test:'SC测试',spring_test:'Spring测试',breakout_test:'突破测试',needs_test:'需测试',shock:'震仓',tested:'已测试',top:'顶部',bottom:'底部'}[k]||k;}
function indexOfDate(d, dateStr){ const i = d.dates.indexOf(dateStr); return i >= 0 ? i : 0; }

// ---------- 回测 ----------
$('#btn-bt').onclick = async () => {
  const code = $('#bt-code').value.trim(); if (!code) return;
  $('#bt-status').textContent = '回测中…';
  try {
    const r = await api('/api/backtest/single', {method:'POST', body: JSON.stringify({
      code, kind: $('#bt-kind').value, since: $('#bt-since').value || null })});
    $('#bt-status').textContent = '';
    $('#bt-result').innerHTML = `
      <p>交易 <b>${r.n_trades}</b> 笔 · 胜率 <b>${pct(r.win_rate)}</b> · 盈亏因子 ${r.profit_factor}
      · 总收益 <b class="${r.total_ret>0?'up':'down'}">${pct(r.total_ret)}</b>
      vs 买入持有 ${pct(r.buy_hold)} · 最大回撤 ${pct(r.max_dd)} · ${r.years}年</p>
      <table class="mini"><thead><tr><th>入场</th><th>出场</th><th>收益率</th><th>原因</th></tr></thead><tbody>
      ${r.trade_list.slice(0,50).map(t=>`<tr><td>${t.entry}</td><td>${t.exit}</td>
        <td class="${t.ret>0?'up':'down'}">${(t.ret*100).toFixed(1)}%</td><td>${t.exit_reason}</td></tr>`).join('')}
      </tbody></table>`;
  } catch (e) { $('#bt-status').textContent = '错误: ' + e.message; }
};

$('#btn-bb').onclick = async () => {
  let params = null;
  const pv = $('#bb-params').value.trim();
  if (pv) { try { params = JSON.parse(pv); } catch { alert('参数JSON不合法'); return; } }
  $('#bb-status').textContent = '批量回测提交…';
  const j = await api('/api/backtest/batch', {method:'POST', body: JSON.stringify({
    kind: $('#bb-kind').value, sample: +$('#bb-sample').value, since: $('#bb-since').value, params })});
  pollJob(j.job_id, job => {
    if (job.status === 'done') {
      $('#bb-status').textContent = '完成';
      const s = job.result.summary;
      $('#bb-result').innerHTML = `<table class="mini"><tbody>${Object.entries(s).map(([k,v])=>
        `<tr><td>${zhMetric(k)}</td><td><b>${typeof v==='number'&&(k.includes('ret')||k.includes('dd')||k.includes('win_rate'))?pct(v):v}</b></td></tr>`).join('')}</tbody></table>
        <p class="small muted">报告: ${job.result.report}</p>`;
    } else if (job.status === 'error') $('#bb-status').textContent = '失败: '+job.error;
    else $('#bb-status').textContent = '运行中…';
  });
};

$('#btn-grid').onclick = async () => {
  const grid = {};
  $('#g-defs').value.split('\n').forEach(line => {
    const m = line.trim(); if (!m) return;
    const [k, vs] = m.split('=');
    grid[k.trim()] = vs.split(',').map(v => isNaN(+v) ? v.trim() : +v);
  });
  if (!Object.keys(grid).length) { alert('请填写网格定义'); return; }
  $('#g-status').textContent = '网格搜索提交…（组合数较多时耗时较长）';
  const j = await api('/api/backtest/grid', {method:'POST', body: JSON.stringify({
    kind: $('#g-kind').value, sample: +$('#g-sample').value, grid })});
  pollJob(j.job_id, job => {
    if (job.status === 'done') {
      $('#g-status').textContent = `完成 ${job.result.combos} 组组合`;
      const rows = job.result.rows || [];
      const cols = rows.length ? Object.keys(rows[0]) : [];
      $('#g-result').innerHTML = `<table class="mini small"><thead><tr>${cols.map(c=>`<th>${zhMetric(c)}</th>`).join('')}</tr></thead>
        <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${typeof r[c]==='number'&&String(r[c]).length>4?r[c].toFixed(3):r[c]??''}</td>`).join('')}</tr>`).join('')}</tbody></table>
        <p class="muted">${job.result.path||''}</p>`;
    } else if (job.status === 'error') $('#g-status').textContent = '失败: '+job.error;
    else $('#g-status').textContent = '网格运行中…';
  });
};

// ---------- 数据任务 ----------
$('#btn-upd-stocks').onclick = async () => {
  const j = await api('/api/data/update', {method:'POST', body: JSON.stringify({what:'stocks'})});
  $('#scan-status').textContent=''; refreshJobsLater(); alert('股票更新任务已启动: '+j.job_id);
};
$('#btn-upd-etfs').onclick = async () => {
  const j = await api('/api/data/update', {method:'POST', body: JSON.stringify({what:'etfs'})});
  refreshJobsLater(); alert('ETF补全任务已启动: '+j.job_id);
};
function refreshJobsLater(){ setTimeout(refreshJobs, 800); }
async function refreshJobs() {
  const rows = await api('/api/jobs');
  $('#jobs-table tbody').innerHTML = rows.map(j =>
    `<tr><td class="muted">${j.id}</td><td>${j.kind}</td>
     <td class="${j.status==='done'?'up':j.status==='error'?'down':''}">${{running:'运行中',done:'完成',error:'失败'}[j.status]||j.status}</td>
     <td class="muted">${j.created}</td>
     <td class="small">${j.error || (j.result ? JSON.stringify(j.result).slice(0,140) : '')}</td></tr>`).join('');
}

function pct(x){ return x==null?'—':(x*100).toFixed(1)+'%'; }
function zhMetric(k){return {symbols:'标的数',symbols_with_trades:'有交易标的',total_trades:'总交易数',avg_win_rate:'平均胜率',
 avg_profit_factor:'平均盈亏因子',avg_total_ret:'平均总收益',median_total_ret:'收益中位数',avg_max_dd:'平均最大回撤',
 avg_trades_per_year:'年均交易',n_trades:'交易数',win_rate:'胜率',profit_factor:'盈亏因子',total_ret:'总收益',
 buy_hold:'买入持有',max_dd:'最大回撤',years:'年限',combo:'组合'}[k]||k;}

loadScan();
