const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(require.resolve('../../caisen/static/app.js'), 'utf8');
const chart = source.slice(source.indexOf('function renderDetailChart('), source.indexOf('let scanSettings'));
const bars = Array.from({length: 8}, (_, i) => ({date: '2026-01-0' + (i + 1), close: 18}));
const selected = {pattern_id:'chosen', formed_at:'2026-01-02', confirmation_date:'2026-01-06', invalidation:15};
let option;
const nodes = {};
const context = vm.createContext({
  $: id => nodes[id] ||= {}, window:{echarts:true},
  echarts:{getInstanceByDom: () => ({setOption: value => { option = value; }})},
  requestAnimationFrame: () => {},
  selectedPattern:selected, chartCenter:6,
  fmt: String, isTerminalPattern: () => false, patternName: () => 'W底', stateLabel: () => '已确认',
  CaisenPatternChart:{patternGeometry: () => ({
    anchors:[{index:1,price:16,name:'左底'},{index:5,price:17,name:'右底'}],
    boundaries:[{name:'颈线',points:[{index:2,price:20},{index:6,price:18}]}],
    markers:[], start:1, end:6, lifecycle:{line_known_at:'2026-01-03'}
  }), visibleBoundaryPoints:require('../../caisen/static/pattern-chart.js').visibleBoundaryPoints}
});
vm.runInContext(chart, context);
const input = {_bars:bars, timeframe:'1d', code:'603233', patterns:[selected,{invalidation:14}]};

test('only selected structure contributes levels; clipped boundary survives zoom', () => {
  context.renderDetailChart(input, 4);
  assert.deepEqual(Array.from(option.series[0].markLine.data, item => item.yAxis), [15]);
  const boundary = option.series.find(series => series.name === '颈线');
  assert.equal(JSON.stringify(boundary.data), '[[0,19],[2,18]]');
  assert.match(nodes.chartPattern.textContent, /2026-01-02.*2026-01-03.*2026-01-06/);
});

test('no selected shape means no unlabeled overlapping overlays', () => {
  context.selectedPattern = null;
  context.renderDetailChart(input, 4);
  assert.deepEqual(Array.from(option.series, item => item.name), ['K线','成交量']);
  assert.doesNotMatch(option.title.text, /未核验/);
});
