const test = require('node:test');
const assert = require('node:assert/strict');
const { filterScanRows } = require('../../caisen/static/scan-filters.js');

const rows = [
  { code: '000001', action: 'CONSIDER_BUY', patterns: [{ pattern_type: 'w_bottom', state: 'confirmed' }] },
  { code: '000002', action: 'WAIT', patterns: [{ pattern_type: 'triangle_bottom', state: 'forming' }] },
  { code: '000003', action: 'CONSIDER_BUY', patterns: [{ pattern_type: 'triangle_bottom', state: 'confirmed' }] },
  { code: '000004', action: 'CONSIDER_BUY', patterns: [{ pattern_type: 'triangle_bottom', state: 'target_reached' }] },
  { code: '000005', action: 'SELL_IF_HELD', patterns: [{ pattern_type: 'false_breakout', state: 'confirmed' }] },
];

test('combines action and active-pattern filters without changing scan order', () => {
  const filtered = filterScanRows(rows, { action: 'CONSIDER_BUY', patternType: 'triangle_bottom' });
  assert.deepEqual(filtered.map(row => row.code), ['000003']);
});

test('filters by an active pattern and ignores terminal historical shapes', () => {
  const filtered = filterScanRows(rows, { patternType: 'triangle_bottom' });
  assert.deepEqual(filtered.map(row => row.code), ['000002', '000003']);
});

test('empty filters preserve all rows and original order', () => {
  assert.deepEqual(filterScanRows(rows), rows);
});

test('a filter with no matches returns an empty list', () => {
  assert.deepEqual(filterScanRows(rows, { action: 'CONSIDER_BUY', patternType: 'bear_flag' }), []);
});

test('selects the latest selected main pattern once per timeframe from lightweight API flags', () => {
  const { mainPatterns } = require('../../caisen/static/scan-filters.js');
  const row = { patterns: [
    { pattern_type: 'w_bottom', timeframe: '1d', formed_at: '2025-01-01', is_main_structure: true, selected_main: true },
    { pattern_type: 'spring', timeframe: '1d', formed_at: '2025-02-01', is_main_structure: true, selected_main: true },
    { pattern_type: 'triangle_bottom', timeframe: '1w', formed_at: '2025-01-15', is_main_structure: true, selected_main: true },
    { pattern_type: 'm_top', timeframe: '1d', formed_at: '2025-03-01', is_main_structure: true, selected_main: false },
  ] };
  const mains = mainPatterns(row);
  assert.equal(mains['1d'].pattern_type, 'spring');
  assert.equal(mains['1w'].pattern_type, 'triangle_bottom');
  assert.deepEqual(Object.keys(mains).sort(), ['1d', '1w']);
});

test('renders keyboard-operable native shape buttons with checked state and escaped text', () => {
  const { patternChipMarkup } = require('../../caisen/static/scan-filters.js');
  assert.equal(patternChipMarkup('w_bottom', 'W底 <形态>', true), '<button type="button" class="pattern-filter-chip selected" data-value="w_bottom" aria-pressed="true">W底 &lt;形态&gt;</button>');
});

test('marks unsupported scan families without disabling historical lookup', () => {
  const { patternSupportLabel, patternChipMarkup } = require('../../caisen/static/scan-filters.js');
  assert.equal(patternSupportLabel('下飘旗形上涨中继', 'unsupported'), '下飘旗形上涨中继（当前扫描暂未支持）');
  assert.equal(patternSupportLabel('下飘旗形上涨中继', 'unsupported', true), '下飘旗形上涨中继（当前扫描暂未支持，历史结果可查）');
  assert.equal(patternSupportLabel('W底', 'supported'), 'W底');
  assert.match(patternChipMarkup('bull_flag', '下飘旗形上涨中继（当前扫描暂未支持）', false, true), /disabled aria-disabled="true"/);
});
