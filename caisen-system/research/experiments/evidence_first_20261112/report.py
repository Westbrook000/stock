# -*- coding: utf-8 -*-
"""Compare exact 748 runtime, prior repair, and evidence-first replay; explain 000538."""
from pathlib import Path
import sys,json,gzip,html,shutil,re
PROJECT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(PROJECT))
from caisen.daily_research import load_research_bars
from caisen.main_bottoms import _first_frozen_boundary_crossing
from caisen.indicators import atr
from caisen.market import discover_instruments
BASE=PROJECT/'reports/daily-research';STATE=PROJECT/'state/evidence-first-v3-20261112';OUT=BASE/'748-and-evidence-first-20261112'
def read(p):return json.loads(p.read_text())
def main():
 OUT.mkdir(exist_ok=True)
 reference=read(BASE/'paired-100-v8-v12-20261109/comparison.json')
 baseline=read(BASE/'git-748d3d01-paired100-spawn8-20261112/daily-research.json')
 oldfix=read(PROJECT/'state/v11-return-attribution-20261111/comparison.json')['repair']
 intermediate_path=BASE/'evidence-first-v2-paired100-spawn8-20261112/daily-research.json'
 intermediate=read(intermediate_path) if intermediate_path.exists() else None
 newpath=BASE/'evidence-first-v3-paired100-spawn8-20261112/daily-research.json'
 newest=read(newpath) if newpath.exists() else None
 source=PROJECT/'state/v11-return-attribution-20261111/accounts/scoped_phase_repair/000538.json.gz'
 with gzip.open(source,'rt') as f:old=json.load(f)
 signal=next(s for s in old['signals'] if s['signal_date']=='2021-08-19')
 p=(signal.get('patterns') or [signal['pattern']])[0];e={x['code']:x['value'] for x in p['evidence']};selection=e['neckline_selection'];line=p['neckline']
 root=BASE/'paired-input-snapshot-20261105-v1';instrument=next(i for i in discover_instruments(root) if i.code=='000538' and i.kind=='stock');bars=load_research_bars(instrument,root=root)['bars'];bars=bars[bars.date<='2026-10-30'].reset_index(drop=True);dates=bars.date.dt.strftime('%Y-%m-%d').tolist();av=atr(bars).values
 available=max(dates.index(selection['line_frozen_at']),dates.index(selection['structure_known_at']))
 first=_first_frozen_boundary_crossing(bars.close.values,av,line,available+1,dates.index('2021-08-19'),.01,{})
 first['date']=dates[first['index']]
 earlytrade=next(t for t in old['trades'] if t['entry_date']=='2020-06-24')
 earlysignal=next(s for s in old['signals'] if s['signal_date']=='2020-06-23');earlypattern=(earlysignal.get('patterns') or [earlysignal['pattern']])[0];earlyline=earlypattern['neckline']
 latetrade=next(t for t in old['trades'] if t['entry_date']=='2021-08-20')
 trace=read(STATE/'538-audit.json')
 rows=[]
 for day in ('2020-07-03','2020-08-28','2021-08-19'):
  i=dates.index(day);level=line['slope']*i+line['intercept'];rows.append({'date':day,'close':float(bars.close.iloc[i]),'neckline':level,'atr_previous':float(av[i-1]),'required_close':level+max(.02,float(av[i-1]))})
 left=dates.index('2019-03-12');right=dates.index('2021-11-25')+1
 earlier_line=[earlyline['slope']*i+earlyline['intercept'] if dates[i]<=earlytrade['exit_date'] else None for i in range(left,right)]
 chart={'dates':dates[left:right],'ohlc':bars.iloc[left:right][['open','close','low','high']].values.tolist(),'line':[line['slope']*i+line['intercept'] for i in range(left,right)],'earlier_line':earlier_line,'freeze':selection['line_frozen_at'],'first':first,'maturity':'2020-08-28','old_confirm':'2021-08-19','earlier_trade':earlytrade,'late_trade':latetrade,'anchors':[{'name':'短W左底','date':'2020-02-28','price':42.78},{'name':'短W右底','date':'2020-03-17','price':39.14}]}
 metrics=[('历史v8',reference['versions']['old']['summary']),('748原样基线／规则v9',baseline['summary']),('上一版修复／0daac68a',oldfix['summary']),('中间对照／67f637d8',intermediate['summary'] if intermediate else None),('最终优化／2823819f',newest['summary'] if newest else None)]
 checks={'748_accounts':len(list((BASE/'git-748d3d01-paired100-spawn8-20261112/accounts').glob('*.json'))),'748_accounting_pass':all(all(read(f)['accounting_checks'].values()) for f in (BASE/'git-748d3d01-paired100-spawn8-20261112/accounts').glob('*.json')),'no_late_order_for_original_troughs':not trace['stale_2021_08_19_for_2020_troughs']}
 if newest:
  newaccount=read(BASE/'evidence-first-v3-paired100-spawn8-20261112/accounts/stock-000538.json');checks['new_account_000538_no_stale_signal']=not any(s['signal_date']=='2021-08-19' and any(pat['pattern_type']=='w_bottom' and pat['key_points'].get('pivot_1_date')=='2020-02-28' and pat['key_points'].get('pivot_2_date')=='2020-03-17' for pat in (s.get('patterns') or [s['pattern']])) for s in newaccount['signals']);checks['new_100_accounts_successful']=newest['summary']['successful_account_count']==100;checks['new_100_accounting_pass']=all(all(read(f)['accounting_checks'].values()) for f in (BASE/'evidence-first-v3-paired100-spawn8-20261112/accounts').glob('*.json'))
  chart['new_signals_in_2020']=[{'date':s['signal_date'],'types':s['pattern_types']} for s in newaccount['signals'] if '2020-05-01'<=s['signal_date']<='2020-10-31'];chart['new_trades_in_2020']=[t for t in newaccount['trades'] if '2020-05-01'<=t['entry_date']<='2020-10-31']
 complete=bool(newest and newest['summary']['successful_account_count']==100 and newest['summary']['failed_account_count']==0 and checks.get('new_100_accounting_pass') and len(list((BASE/'evidence-first-v3-paired100-spawn8-20261112/accounts').glob('*.json')))==100)
 data={'status':'complete' if complete else 'optimized_run_pending','metrics':[{'label':name,'summary':summary} for name,summary in metrics],'first_crossing':first,'date_audit':rows,'chart':chart,'checks':checks,'baseline_748_seconds':baseline['timings']['total_seconds'],'optimized_seconds':newest['timings']['total_seconds'] if newest else None,'source_748':str(PROJECT/'state/748-baseline-20261112/source-manifest.json'),'source_optimized':str(STATE/'source-manifest.json')}
 (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
 def mr(name,s):
  if s is None:return '<tr><td>'+html.escape(name)+'</td><td colspan="6">正在运行，尚无完整结果</td></tr>'
  values=[s['trade_count'],f"{s['win_rate']:.2%}",f"{s['cumulative_return']:.2%}",f"{s['maximum_drawdown']:.2%}",f"{s['profit_factor']:.3f}",s['successful_account_count']]
  return '<tr><td>'+html.escape(name)+'</td>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'
 table=''.join(mr(n,s) for n,s in metrics)
 auditrows=''.join('<tr><td>'+r['date']+'</td><td>'+f"{r['close']:.2f}"+'</td><td>'+f"{r['neckline']:.4f}"+'</td><td>'+f"{r['atr_previous']:.4f}"+'</td><td>'+f"{r['required_close']:.4f}"+'</td></tr>' for r in rows)
 chartjson=json.dumps(chart,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>748基线验证与000538首次突破审计</title><style>body{font:15px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:18px;color:#1f3047;background:#f3f6fa}section{background:white;padding:24px;margin:18px 0;border-radius:14px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.chart{height:520px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{color:#9b5400}a{color:#1d59ad}</style><h1>748基线验证与000538首次突破审计</h1><p>同100股票 · 冻结行情截止2026-10-30 · 8进程 · 次日开盘、止损8%、止盈20%、零费用</p>
<section><h2>原样748与当前版本比较</h2><table><tr><th>版本</th><th>交易数</th><th>胜率</th><th>等初始资金收益</th><th>汇总最大回撤</th><th>盈利因子</th><th>成功账户</th></tr>TABLE</table><p>748运行代码直接从Git独立导出并逐文件核对，不是只替换选线循环，也没有将规则v9改名为v8。不同任务均使用相同冻结行情及100成员。</p><p class="note">当前优化行只有在100账户全部写完后才填入结果。收益是等初始资金独立研究账户的汇总，不是收益率相加；回撤是汇总权益口径。</p></section>
<section><h2>本轮验收结论</h2><p>748与最终版均完成100/100账户，逐账户账目校验通过。748冷跑约33分01秒；最终版约28分05秒（1684.94秒）。这一耗时仅针对100只股票，不能当作1000标的30分钟验收。</p><p class="note">最终版收益26.08%，低于748的31.55%及上一修复版的32.60%；盈利因子1.198，亦未改善。修复解决迟到/重复确认及边界身份问题，但不能称为盈利优化成功。当前对照同时调整多个相互依赖路径，不能从总收益差直接断言每个单项改动的贡献。</p></section><section><h2>红圈附近其实已经有另一笔入场</h2><p>上一版完整账目存在：<b>2020-06-23确认 → 2020-06-24以53.24入场 → 2020-07-07止盈20%</b>。它来自2019-08-06至2020-03-17的较长W底。截图只展示2021年的后续亏损交易，不能据此判断2020年没有买入。</p><p>截图中的短W底则从2020-02-28到2020-03-17，到2020-08-28才满六个月。这组短W不能在红圈附近买，也不能在一年后重新越线时当作首次确认。</p></section>
<section><h2>按截图自己的冻结线逐日核算</h2><table><tr><th>日期</th><th>收盘</th><th>自身颈线</th><th>前一根ATR14</th><th>有效突破门槛</th></tr>AUDIT</table><p>自身首次有效越线为<b>2020-07-03</b>，但早于六个月成熟日2020-08-28；因此该线已被提前突破消耗。2021-08-19属于再次越线，不能重新发买点。2020-04-30是压力触点最早就绪日，不等于交易资格已成立。</p><div id="chart" class="chart"></div></section>
<section><h2>当前优化的规则</h2><ol><li>先按证据选择第一条边界，再判断突破，不挑当天恰好越过的低排名线。</li><li>同线身份使用精确几何，或共享至少两个压力触点且证据时点偏差不足半报价单位的数值重拟合；不使用ATR容差合并不同线；失败候选淘汰也按相同边界身份判断，不按旧失败价位的ATR区域排除新线。</li><li>独立审计每条冻结线自己的首次有效突破，晚发现、未选中及周期未成熟时发生过的越线，都不能在以后变成首次买点。</li><li>按边界缓存首次事件和扫描进度，只追加检查新K线；返回的事件必须在观察日内。不同几何的边界进入形态ID，避免相同触点集合的身份冲突。</li></ol><p>红圈不符合这组短W的六个月门槛，修复不会为了让图看起来更好而提前买入；同一时间满足当前工程规则的较长结构仍可正常确认。</p></section>
<section><h2>核对资料与限制</h2><p>新增与相关保护测试27项通过；最终版扩展相关检查57项通过，9项未纳入本轮执行。先前已确认的3项旧断言失败仍保留，不宣称完整套件全绿。截图问题另用000538真实行情从历史起点逐日回放，确认未再发出该短W的2021-08-19迟到信号。</p><p><a href="data.json">逐日数字及版本指标</a> · <a href="../git-748d3d01-paired100-spawn8-20261112/daily-research.html">748完整回测</a> · <a href="../evidence-first-v3-paired100-spawn8-20261112/daily-research.html">最终优化完整回测</a> · <a href="../../../state/evidence-first-v3-20261112/538-audit.json">000538逐日证据</a></p><p>所有历史均已用于研究，不是独立留出验证。更高收益不等于形态更准确。</p></section>
<script src="../../../caisen/static/vendor/echarts.min.js"></script><script id="data" type="application/json">CHARTJSON</script><script>const c=JSON.parse(document.getElementById('data').textContent),e=echarts.init(document.getElementById('chart'));const pts=c.anchors.map(p=>({name:p.name,coord:[p.date,p.price]}));pts.push({name:'2020实际入场',coord:[c.earlier_trade.entry_date,c.earlier_trade.entry_price],itemStyle:{color:'#0d9a80'}},{name:'2020止盈',coord:[c.earlier_trade.exit_date,c.earlier_trade.exit_price],itemStyle:{color:'#0d9a80'}},{name:'截图迟到入场',coord:[c.late_trade.entry_date,c.late_trade.entry_price],itemStyle:{color:'#ce424f'}});e.setOption({animation:false,legend:{data:['K线','截图自身冻结线','2020长W自身颈线']},tooltip:{trigger:'axis'},grid:{left:65,right:25,bottom:75},xAxis:{type:'category',data:c.dates},yAxis:{scale:true},dataZoom:[{type:'inside'},{type:'slider'}],series:[{name:'K线',type:'candlestick',data:c.ohlc,itemStyle:{color:'#d54b55',color0:'#219c80',borderColor:'#d54b55',borderColor0:'#219c80'},markPoint:{data:pts,label:{fontSize:10,formatter:p=>p.name}}},{name:'2020长W自身颈线',type:'line',data:c.earlier_line,symbol:'none',lineStyle:{color:'#0d9a80',width:1.5,type:'dashed'}},{name:'截图自身冻结线',type:'line',data:c.line,symbol:'none',lineStyle:{color:'#6153d6',width:2},markLine:{symbol:'none',label:{formatter:p=>p.name,rotate:90,position:'insideEndTop',fontSize:10},data:[{name:'触点就绪',xAxis:c.freeze},{name:'自身首次越线／未满六个月',xAxis:c.first.date},{name:'满六个月',xAxis:c.maturity},{name:'旧版再次确认',xAxis:c.old_confirm}]}}]});window.addEventListener('resize',()=>e.resize());</script></html>'''.replace('TABLE',table).replace('AUDIT',auditrows).replace('CHARTJSON',chartjson)
 (OUT/'index.html').write_text(page);print(OUT/'index.html')
if __name__=='__main__':main()
