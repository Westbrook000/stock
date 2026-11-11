const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const CaisenPatternChart = require('../../caisen/static/pattern-chart.js');
const CaisenScanFilters = require('../../caisen/static/scan-filters.js');

class Element {
  constructor(id) {
    this.id = id;
    this.listeners = {};
    this.options = [];
    this.selectedOptions = [];
    this.value = ({ code: '000001', kind: 'stock', tf: '1d', chartRange: '120' })[id] || '';
    this.dataset = {};
    this.attributes = {};
    this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
    this.scrollIntoView = () => {};
  }
  append(option) { this.options.push(option); }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  setAttribute(name, value) { this.attributes[name] = value; }
  getBoundingClientRect() { return { left: 0, width: 900 }; }
  dispatch(type, event) { for (const callback of this.listeners[type] || []) callback(event); }
}

test('workbench initializes without removed controls and clicking a shape focuses it through latest bars', async () => {
  const elements = new Map();
  const getElement = id => {
    if (id === 'scanActionFilter') return null;
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  };
  const bars = Array.from({ length: 200 }, (_, i) => ({
    date: new Date(Date.UTC(2024, 0, i + 1)).toISOString().slice(0, 10),
    open: 10, high: 11, low: 9, close: 10, volume: 100,
  }));
  const pattern = {
    pattern_id: 'shape-1', code: '000001', timeframe: '1d', pattern_type: 'w_bottom',
    state: 'confirmed', formed_at: bars[50].date, chart_offset: 0,
    pattern_instance_id: 'instance-shape-1',
    instance_state_events: [
      { instance_id: 'instance-shape-1', event_id: 'event-forming', event_seq: 1,
        state: 'forming', event_date: bars[52].date, entry_eligible: false },
      { instance_id: 'instance-shape-1', event_id: 'event-confirmed', event_seq: 2,
        state: 'confirmed', event_date: bars[80].date, entry_eligible: true },
    ],
    key_points: { pivot_1_index: 50, pivot_1_date: bars[50].date, pivot_1: 9,
      pivot_2_index: 65, pivot_2_date: bars[65].date, pivot_2: 9.2 },
    evidence: [{ code: 'main_structure', value: { role: 'main' } },
      { code: 'main_selection', value: { selected: true } },
      { code: 'neckline_selection', value: { selected_candidate_id: 'boundary-1',
        line_frozen_at: bars[70].date, selected_line: { slope: 0, intercept: 10 } } }],
  };
  const analysis = { code: '000001', timeframe: '1d', _bars: bars, _chart_offset: 0,
    patterns: [pattern], decision: { action: 'WAIT', patterns: [pattern], selected_plan: null } };
  const chart = { options: null, setOption(value) { this.options = value; }, resize() {} };
  const echarts = { getInstanceByDom() { return chart; }, init() { return chart; } };
  const window = {
    CaisenPatternChart, CaisenScanFilters, echarts,
    CaisenHttpClient: { fetchJson: async url => url.includes('/health') ? { status: 'ok' } : analysis },
    addEventListener() {},
  };
  const document = {
    getElementById: getElement,
    createElement: () => new Element('option'),
    querySelectorAll: () => [],
    querySelector: selector => getElement(selector),
  };
  const context = { window, document, CaisenPatternChart, CaisenScanFilters, echarts,
    URLSearchParams, console, setTimeout, clearTimeout, setInterval, clearInterval,
    requestAnimationFrame: callback => callback(), confirm: () => true };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../caisen/static/app.js'), 'utf8'), context);

  await getElement('analyzeBtn').onclick();
  assert.match(getElement('detailRows').innerHTML, /instance-timeline/);
  assert.match(getElement('detailRows').innerHTML, /结构与边界配对冻结/);
  assert.match(getElement('detailRows').innerHTML, /event-confirmed/);
  assert.equal(chart.options.series.some(series => series.name === '确认与结束' &&
    series.data.some(point => point.name === '首次突破确认 · BUY')), true);
  const initialStart = chart.options.xAxis[0].data[0];
  getElement('detailRows').dispatch('click', {
    target: { closest: selector => selector === '[data-show-pattern]' ? { dataset: { showPattern: 'shape-1' } } : null },
  });
  await new Promise(resolve => setImmediate(resolve));

  const focusedDates = chart.options.xAxis[0].data;
  assert(focusedDates[0] < initialStart, 'clicked shape should move the left edge back to its formation');
  assert.equal(focusedDates.at(-1), bars.at(-1).date, 'focused chart should still end at the latest bar');
});
