const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync(require.resolve("../../caisen/static/app.js"), "utf8");
const internalFn = source.match(/^function isInternalEvidence\(item\)\{[^\n]*\}/m)?.[0];
const labelFn = source.match(/^function evidenceLabel\(item\)\{[^\n]*\}/m)?.[0];
assert.ok(internalFn && labelFn, "production evidence formatters are available");
const context = vm.createContext({ fmt: value => String(value) });
vm.runInContext(`${internalFn}\n${labelFn}\nglobalThis.api={isInternalEvidence,evidenceLabel};`, context);
const { isInternalEvidence, evidenceLabel } = context.api;

test("pattern quality evidence is concise, Chinese, and uses observed dates before confirmation", () => {
  const mature = evidenceLabel({ code: "structure_duration", value: {
    start_date: "2025-01-01", observed_date: "2025-05-01", duration_days: 120,
    first_valid_break_date: "2025-05-01", eligible: true
  }});
  assert.match(mature, /结构起点 2025-01-01 → 首次突破 2025-05-01/);
  assert.match(mature, /持续 120 天 · 超过3自然月/);

  const pending = evidenceLabel({ code: "structure_duration", value: {
    start_date: "2025-01-01", observed_date: "2025-04-15", duration_days: 104,
    first_valid_break_date: null, eligible: true
  }});
  assert.match(pending, /观察日 2025-04-15/);
  assert.match(pending, /尚未突破/);
  assert.doesNotMatch(pending, /首次突破 2025-04-15/);
});

test("low-position and drawdown evidence explain percentiles, thresholds, and passing branch", () => {
  const position = evidenceLabel({ code: "low_position", value: {
    position_ratio: 0.25, position_pass: true, drawdown_pass: false, passed: true
  }});
  assert.match(position, /区间位置 25\.0%（门槛≤30%）/);
  assert.match(position, /通过分支：历史价格低位/);

  const drawdown = evidenceLabel({ code: "prepattern_drawdown", value: {
    high: 100, drawdown: 0.35, threshold: 0.30, passed: true
  }});
  assert.match(drawdown, /前高 100 · 回撤 35\.0%（门槛≥30\.0%，通过）/);
  assert.equal(evidenceLabel({ code: "pattern_quality_pass", value: true }), "形态质量门槛：通过");
  assert.equal(evidenceLabel({ code: "first_valid_break", value: "2025-05-01" }), "首次有效突破：2025-05-01");
});

test("internal position-history caches are filtered instead of rendered", () => {
  assert.equal(isInternalEvidence({ code: "position_history_cache", value: { _position_history: true, dates: [1, 2] } }), true);
  assert.equal(isInternalEvidence({ code: "quality_evidence", value: { _cache: [1, 2] } }), true);
});
