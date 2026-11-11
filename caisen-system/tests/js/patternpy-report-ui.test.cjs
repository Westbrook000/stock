const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const htmlPath = process.argv[2] || path.resolve(__dirname, '../../reports/patternpy-600108-full-history.html');
const jsonPath = process.argv[3] || path.resolve(__dirname, '../../reports/patternpy-600108-full-history.json');
const html = fs.readFileSync(htmlPath, 'utf8');
const payload = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
const script = html.match(/<script>\s*(const DATA=.*?)<\/script>\s*<\/body>/s)?.[1];
assert.ok(script, 'generated report must contain its standalone UI script');

class Element {
  constructor(id) { this.id = id; this.value = id === 'type' ? 'all' : id === 'window' ? 'all' : ''; this._html = ''; this.options = []; this.rows = []; }
  set innerHTML(value) {
    this._html = value;
    if (this.id === 'rows') this.rows = [...value.matchAll(/<tr data-index="(\d+)">/g)].map(match => ({dataset: {index: match[1]}, onclick: null}));
  }
  get innerHTML() { return this._html; }
  append(option) { this.options.push(option); if (option.selected) this.value = String(option.value); }
}

const ids = ['metrics','type','window','from','to','filtered','chart','page','rows','prev','next','reset','recent','export'];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
const listeners = [];
const actions = [];
let zoom = null;
let setOptionCount = 0;
const chart = {
  setOption(option) {
    setOptionCount++;
    zoom = option.dataZoom?.[0] && {start: option.dataZoom[0].start, end: option.dataZoom[0].end};
  },
  dispatchAction(action) {
    actions.push(action);
    if (action.type === 'dataZoom') zoom = {startValue: action.startValue, endValue: action.endValue, start: action.start, end: action.end};
  },
  resize() {},
};
const context = {
  document: {
    querySelector: selector => elements[selector.slice(1)],
    querySelectorAll: selector => selector === '#rows tr' ? elements.rows.rows : ['#type','#window','#from','#to'].map(id => elements[id.slice(1)]),
    createElement: tag => tag === 'option' ? {value:'',textContent:'',selected:false} : new Element(tag),
  },
  echarts: {init: () => chart},
  addEventListener: (...args) => listeners.push(args),
  URL: {createObjectURL: () => '', revokeObjectURL() {}},
  Blob: class {}, console, Math, Number, String, Array, Object, JSON,
};
vm.runInNewContext(script, context, {timeout: 10000});

function metricValue(label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = elements.metrics.innerHTML.match(new RegExp(`<b>(\\d+)<\\/b><span>${escaped}`));
  assert.ok(match, `metric exists for ${label}`);
  return Number(match[1]);
}
function listedCount() { return Number(elements.filtered.textContent.replace(/[^\d]/g, '')); }
function countEvents(window, label) { return payload.events.filter(event => event.window === window && event.label === label).length; }
function countWindow(window) { return payload.events.filter(event => event.window === window).length; }

assert.equal(elements.window.value, '3');
assert.equal(metricValue('W底（Double Bottom） 原生标签 · window 3'), countEvents(3, 'Double Bottom'));
assert.equal(metricValue('头肩底（Inverse Head and Shoulder） 原生标签 · window 3'), countEvents(3, 'Inverse Head and Shoulder'));
assert.equal(listedCount(), countWindow(3));

elements.window.value = '20';
elements.window.onchange();
assert.equal(metricValue('W底（Double Bottom） 原生标签 · window 20'), countEvents(20, 'Double Bottom'));
assert.equal(metricValue('头肩底（Inverse Head and Shoulder） 原生标签 · window 20'), countEvents(20, 'Inverse Head and Shoulder'));
assert.equal(listedCount(), countWindow(20));
elements.type.value = 'Double Bottom';
elements.type.onchange();
assert.equal(listedCount(), countEvents(20, 'Double Bottom'));
assert.match(elements.rows.innerHTML, /W底（Double Bottom）/);

const firstRecentDate = payload.bars.find(bar => bar.date >= '2023-01-01').date;
elements.recent.onclick();
assert.equal(elements.from.value, firstRecentDate);
assert.ok(firstRecentDate >= '2023-01-01');
assert.ok(payload.bars.some(bar => bar.date === firstRecentDate));
assert.equal(actions.at(-1).type, 'dataZoom');
assert.equal(actions.at(-1).startValue, firstRecentDate);
assert.equal(actions.at(-1).endValue, payload.bars.at(-1).date);

elements.reset.onclick();
elements.window.value = '20'; elements.window.onchange();
elements.type.value = 'Double Bottom'; elements.type.onchange();
assert.ok(elements.rows.rows.length > 0);
const selectedIndex = Number(elements.rows.rows[0].dataset.index);
const selectedEvent = payload.events.find(event => event.index === selectedIndex && event.window === 20 && event.label === 'Double Bottom');
assert.ok(selectedEvent);
elements.rows.rows[0].onclick();
const rowZoom = actions.filter(action => action.type === 'dataZoom').at(-1);
assert.equal(rowZoom.type, 'dataZoom');
assert.equal(payload.bars[selectedIndex].date, selectedEvent.date);
assert.ok(payload.bars.findIndex(bar => bar.date === rowZoom.startValue) <= selectedIndex);
assert.ok(payload.bars.findIndex(bar => bar.date === rowZoom.endValue) >= selectedIndex);
const zoomBeforePagination = {...zoom};
const optionCallsBeforePagination = setOptionCount;
elements.next.onclick();
assert.equal(setOptionCount, optionCallsBeforePagination, 'pagination should update only the list');
assert.deepEqual(zoom, zoomBeforePagination, 'pagination must preserve the event zoom window');

console.log('PatternPy report UI state: selected-window metrics, event list, recent trading-date range, event navigation, and zoom persistence passed');
