const test = require('node:test');
const assert = require('node:assert/strict');
const scanResults = require('../../caisen/static/scan-results.js');

test('scan result query sends display filters to the paged result endpoint', () => {
  const query = new URLSearchParams(scanResults.buildQuery({
    page: 3,
    action: 'CONSIDER_BUY',
    patternTypes: ['w_bottom', 'triangle_bottom'],
  }));
  assert.equal(query.get('page'), '3');
  assert.equal(query.get('sort'), 'rank');
  assert.equal(query.get('order'), 'asc');
  assert.equal(query.get('action'), 'CONSIDER_BUY');
  assert.equal(query.get('pattern_types'), 'w_bottom,triangle_bottom');
});

test('scan result query leaves optional filters empty when showing all results', () => {
  const query = new URLSearchParams(scanResults.buildQuery({page: 1}));
  assert.equal(query.get('page'), '1');
  assert.equal(query.has('action'), false);
  assert.equal(query.has('pattern_types'), false);
});
