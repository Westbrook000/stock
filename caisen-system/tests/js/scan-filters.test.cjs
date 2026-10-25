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
