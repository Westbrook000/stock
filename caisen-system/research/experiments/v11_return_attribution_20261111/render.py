# -*- coding: utf-8 -*-
"""Render the finished attribution campaign; never invent missing results."""
from pathlib import Path
import gzip
import html
import json
import re
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from caisen.daily_research import load_research_bars
from caisen.market import discover_instruments

STATE = PROJECT / 'state/v11-return-attribution-20261111'
REPORT = PROJECT / 'reports/daily-research/v11-phase-scope-repair-20261111'
BASE = PROJECT / 'reports/daily-research/rollback-33ca8475-paired100-process8-20261111'
FROZEN = PROJECT / 'reports/daily-research/paired-input-snapshot-20261105-v1'


def main():
    d = json.loads((STATE / 'comparison.json').read_text())
    assert d['status'] == 'complete' and not d['repair']['missing']
    assert d['repair']['summary']['successful_account_count'] == 100
    assert d['repair']['accounting_passed']
    REPORT.mkdir(parents=True, exist_ok=True)
    variant = d['selected_repair']
    instruments = {x.code: x for x in discover_instruments(FROZEN) if x.kind == 'stock'}
    cases = []
    rankings = []
    for f in sorted((STATE / 'accounts' / variant).glob('*.gz')):
        with gzip.open(f, 'rt') as handle:
            account = json.load(handle)
        old = json.loads((BASE / 'accounts' / ('stock-' + account['code'] + '.json')).read_text())
        rankings.append({'code': account['code'], 'equity_delta': account['ending_equity'] - old['ending_equity'],
                         'old_trades': len(old['trades']), 'new_trades': len(account['trades'])})
        if account['code'] not in ('000538', '600633', '600691'):
            continue
        old_days = {s['signal_date'] for s in old['signals']}
        fresh_trades = [t for t in account['trades'] if t['signal_date'] not in old_days]
        if not fresh_trades:
            continue
        # Show one added trade per preselected code; charts explain geometry,
        # while the full ledger contains every winning and losing trade.
        trade = fresh_trades[0]
        signal = next(s for s in account['signals'] if s['signal_date'] == trade['signal_date'])
        pattern = (signal.get('patterns') or [signal['pattern']])[0]
        evidence = {x['code']: x['value'] for x in pattern['evidence']}
        selection = evidence.get('neckline_selection', {})
        frame = load_research_bars(instruments[account['code']], root=FROZEN)['bars']
        frame = frame[frame.date <= '2026-10-30'].reset_index(drop=True)
        dates = frame.date.dt.strftime('%Y-%m-%d').tolist()
        start_date = min(pattern['formed_at'], selection.get('effective_start_date') or pattern['formed_at'])
        left = max(0, next(i for i, date in enumerate(dates) if date >= start_date) - 20)
        right = min(len(frame), dates.index(trade['exit_date']) + 30)
        line = pattern['neckline']
        frozen = selection.get('line_frozen_at')
        points = pattern['key_points']
        anchors = [{'date': value, 'price': points[key.replace('_date', '')], 'name': key}
                   for key, value in points.items()
                   if re.fullmatch(r'pivot_\d+_date', key)]
        cases.append({'code': account['code'], 'family': pattern['pattern_type'], 'trade': trade,
                      'formed_at': pattern['formed_at'], 'frozen_at': frozen,
                      'confirmation_date': pattern['confirmation_date'], 'anchors': anchors,
                      'dates': dates[left:right],
                      'ohlc': frame.iloc[left:right][['open', 'close', 'low', 'high']].values.tolist(),
                      'neckline': [line['slope'] * i + line['intercept'] if dates[i] >= start_date else None
                                   for i in range(left, right)],
                      'tests': evidence.get('neckline_tests', []),
                      'note': '该信号在修复前版本没有同日买点；此处展示修复后完整几何，不补画未经保存的旧线。'})
    d['account_deltas'] = sorted(rankings, key=lambda x: x['equity_delta'])
    d['cases'] = cases
    bridge = json.loads((STATE / 'trade-bridge.json').read_text())
    d['accounting_bridge'] = {k: bridge[k] for k in ('net_pnl_delta', 'bridge', 'counts')}
    parity = STATE / 'production-variant-parity.json'
    if parity.exists():
        d['production_variant_parity'] = json.loads(parity.read_text())
    tests = STATE / 'test-results.json'
    if tests.exists():
        d['test_results'] = json.loads(tests.read_text())
    labels = {'current': '修复前基线', 'no_phase_consumption': '停用阶段消耗（仅诊断）',
              'no_global_premature_failure': '旧线失败不终止整组结构',
              'v9_selection': '恢复v9选线循环', 'moving_support': '改回斜线支撑',
              'no_right_shoulder_guard': '停用右肩检查（仅诊断）',
              'no_prefix_cache': '停用前缀缓存（控制组）',
              'precise_boundary_identity': '精确区分冻结直线', 'scoped_phase_repair': '组合修复'}
    def row(label, summary):
        return '<tr><td>' + html.escape(label) + '</td>' + ''.join('<td>' + str(value) + '</td>' for value in (
            summary['trade_count'], f"{summary['win_rate']:.2%}", f"{summary['cumulative_return']:.2%}",
            f"{summary['maximum_drawdown']:.2%}", f"{summary['profit_factor']:.3f}")) + '</tr>'
    header = '<tr><th>版本／实验</th><th>成交数</th><th>胜率</th><th>等初始资金收益</th><th>汇总权益最大回撤</th><th>盈利因子</th></tr>'
    full_rows = row('历史v8（原始存档信号配对重放）', d['reference_v8']) + row('当前回退版本 v11', d['current']['summary']) + row('组合修复／完整100账户', d['repair']['summary'])
    panel_rows = ''.join(row(labels[k], value['summary']) for k, value in d['panel_results'].items())
    charts = ''.join('<section><h2>' + case['code'] + ' · ' + html.escape(case['family']) + '</h2><p>' +
        html.escape(f"形成 {case['formed_at']}；冻结 {case['frozen_at']}；确认 {case['confirmation_date']}；实际入场 {case['trade']['entry_date']}；退出 {case['trade']['exit_date']}；单笔收益 {case['trade']['return_pct']:.2%}") +
        '</p><div class="chart" id="case' + str(i) + '"></div><p>' + case['note'] + '</p></section>' for i, case in enumerate(cases))
    data = json.dumps({'cases': cases}, ensure_ascii=False,
                      separators=(',', ':')).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>收益下降归因与修复回测</title>
<style>body{font:15px/1.65 system-ui;max-width:1200px;margin:30px auto;padding:18px;color:#203047;background:#f2f5f8}section{background:white;border-radius:12px;margin:18px 0;padding:22px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.chart{height:440px}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#175cc0}</style>
<h1>收益下降归因与修复回测</h1><p>同100只股票 · 冻结行情截止2026-10-30 · Backtrader 8%止损／20%止盈 · 8个独立工作进程</p>
<section><h2>完整100账户复验</h2><table>FULL</table><p>每账户独立10万元，零费用；100账户均成功，账目核验通过。这里的回撤是汇总权益回撤，并非最差单账户回撤。收益高低不能替代形态识别真值。</p></section>
<section><h2>单因素归因：12标的开发筛查</h2><table>PANEL</table><p>每组均用相同12股票，完整历史；预先固定随机种子及两个问题案例。各单项影响不相加；组合组用于检查交互。未对所有单因素运行完整100股票，不能把本表的效应量外推为100账户归因。</p></section>
<section><h2>修复的内容</h2><p>旧实现会把一条过早突破颈线的失败，提升为整组底点的失败，因而挡住后来独立支持的新线。修复后，旧线仍被消耗，但不再结束其他边界的搜索。</p><p>另一个修复是将“颈线触线容差”和“同一条线的身份”分开：ATR容差用于判断K线是否触线，不再用于将不同直线合并成同一个已触发阶段。完全相同的线增加触点仍不能重新发出买点。</p><p>右肩完整性、结构时长、低位条件、1ATR突破、缓存与8%／20%成交规则保留。两项修复通过先失败后通过的回归测试。</p></section>
<section><h2>账目上的下降来自哪里</h2><p>修复前相较v8少赚328.76万元：被删除旧交易贡献-406.22万元，新增交易补回61.54万元，共同入场的仓位或退出差异补回15.92万元。这是账目分解，规则级归因以上面的单因素实验为准。</p></section>
CHARTS
<section><h2>验证与版本</h2><p>新增身份与失败作用域回归测试、既有生命周期保护共18项通过；扩展检查48项通过。另3项行为断言在未修复基线中也失败，已原样保留并记录，未宣称完整测试套件全通过。</p><p>交付规则为 caisen-main-bottoms-v11-phase-scope-v1。回测来自隔离候选 scoped_phase_repair，保留原模块版本标签；交付源码与候选的行为AST核对一致，差异为说明文字和版本标签。</p><p>本次100账户由12个开发筛查账户及88个续跑账户组成，不是一次独立100账户冷跑，不宣称已达到30分钟冷跑目标。</p></section>
<section><h2>证据及限制</h2><p>所有100标的此前已用于研究，本轮是研究复验，不是独立样本外验证。三个图例按预先选定代码展示首笔新增成交，不只挑盈利图。旧版未保存的失败颈线不在图上补造。</p><p>单因素仅在12标的上逐项筛查；完整100账户只验证了组合修复。因此本轮证实两项错误及其组合净影响，尚不能将相较v8的全部收益差距唯一归因给某项改动。修复后收益仍低于v8，汇总回撤也略有增加。</p><p><a href="data.json">完整指标、账户差异及图表数据</a> · <a href="../../../state/v11-return-attribution-20261111/comparison.json">原始实验对照</a> · <a href="../../../state/v11-return-attribution-20261111/protocol.json">实验口径</a></p></section>
<script src="../../../caisen/static/vendor/echarts.min.js"></script><script id="data" type="application/json">DATA</script><script>
const report=JSON.parse(document.getElementById('data').textContent);report.cases.forEach((c,i)=>{const e=echarts.init(document.getElementById('case'+i));const pts=c.anchors.map(p=>({name:p.name,coord:[p.date,p.price]}));pts.push({name:'实际入场',coord:[c.trade.entry_date,c.trade.entry_price]},{name:'退出',coord:[c.trade.exit_date,c.trade.exit_price]});e.setOption({animation:false,legend:{data:['K线','自身冻结颈线']},tooltip:{trigger:'axis'},grid:{left:65,right:30,bottom:70},xAxis:{type:'category',data:c.dates},yAxis:{scale:true},dataZoom:[{type:'inside',zoomOnMouseWheel:true},{type:'slider'}],series:[{name:'K线',type:'candlestick',data:c.ohlc,itemStyle:{color:'#d74b51',color0:'#219b80',borderColor:'#d74b51',borderColor0:'#219b80'},markPoint:{data:pts,label:{formatter:p=>p.name,fontSize:10}}},{name:'自身冻结颈线',type:'line',data:c.neckline,symbol:'none',lineStyle:{color:'#6550cd',width:2},markLine:{symbol:'none',label:{formatter:p=>p.name},data:[{name:'冻结',xAxis:c.frozen_at},{name:'首次确认',xAxis:c.confirmation_date}]}}]});window.addEventListener('resize',()=>e.resize())});
</script></html>'''.replace('FULL', header + full_rows).replace('PANEL', header + panel_rows).replace('CHARTS', charts).replace('DATA', data)
    (REPORT / 'index.html').write_text(page)
    (REPORT / 'data.json').write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print(REPORT / 'index.html')


if __name__ == '__main__':
    main()
