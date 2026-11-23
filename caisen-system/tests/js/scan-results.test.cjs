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

test("history shows one representative per timeframe and keeps the actual signal", () => {
  const {historyPatterns} = require('../../caisen/static/scan-results.js');
  const buy = {pattern_id:'buy', timeframe:'1d', state:'confirmed', confirmation_date:'2026-11-20'};
  const duplicate = {pattern_id:'duplicate', timeframe:'1d', state:'forming', formed_at:'2026-11-21'};
  const weekly = {pattern_id:'weekly', timeframe:'1w', state:'forming', formed_at:'2026-11-01', is_main_structure:true};
  const history = {pattern_id:'ended', timeframe:'1w', state:'invalidated', formed_at:'2026-11-19'};
  const row = {action:'CONSIDER_BUY', selected_plan:{pattern_id:'buy'},
    patterns:[...Array(100).fill(duplicate), history, weekly, buy]};
  assert.deepEqual(historyPatterns(row), [buy, weekly]);
  assert.deepEqual(historyPatterns({...row,patterns:[history]}), []);
  assert.deepEqual(historyPatterns({...row, action:'WATCH', watch:[{pattern_id:'duplicate'}]}), [duplicate, weekly]);
});

test('compact candle coordinates decode in OHLC drawing order', () => {
  const encoded = Buffer.from([10, 25, 0, 255, 25, 20, 14, 30]).toString('base64');
  assert.deepEqual(scanResults.decodeMiniCandles(encoded), [
    {open: 10, close: 25, low: 0, high: 255},
    {open: 25, close: 20, low: 14, high: 30},
  ]);
  assert.deepEqual(scanResults.decodeMiniCandles(Buffer.alloc(100).toString('base64')), []);
});
