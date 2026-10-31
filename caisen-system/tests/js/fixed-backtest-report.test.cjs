const test = require("node:test");
const assert = require("node:assert/strict");
const report = require("../../caisen/static/fixed-backtest-report.js");

test("fixed report renders trades and escapes untrusted pattern labels", () => {
  const html = report.render({
    summary: { security_timeframe_accounts: 1, trade_count: 1, signal_count: 1,
      distinct_traded_patterns: 1, overlapping_multi_pattern_signals: 0 },
    groups: [], patterns: [],
    trades: [{ code: "AAA", kind: "stock", timeframe: "1d", pattern_labels: ["<img src=x>"] ,
      signal_date: "2024-01-01", entry_date: "2024-01-02", entry_price: 10,
      exit_date: "2024-01-03", exit_price: 9.2, exit_reason: "stop_loss", pnl: -800,
      pattern_ids: ["p1"] }]
  });
  assert.match(html, /止损/);
  assert.match(html, /&lt;img src=x&gt;/);
  assert.doesNotMatch(html, /<img src=x>/);
});

test("fixed report translates display labels without changing source enums", () => {
  const html = report.render({
    summary: { security_timeframe_accounts: 1, trade_count: 1, signal_count: 1,
      distinct_traded_patterns: 1, overlapping_multi_pattern_signals: 0,
      equal_initial_cash_portfolio: { starting_capital: 100000, metrics: { profit_factor: 1.2 } } },
    groups: [{ dimension: "kind", value: "stock", trade_count: 1, win_rate: 1, net_pnl: 10, profit_factor: 1.2 },
      { dimension: "shape", value: "w_bottom", trade_count: 1, win_rate: 1, net_pnl: 10, profit_factor: 1.2 }],
    patterns: [], results: [],
    trades: [{ code: "AAA", kind: "etf", timeframe: "1d", pattern_types: ["w_bottom"],
      signal_date: "2024-01-01", entry_date: "2024-01-02", entry_price: 10,
      exit_date: "2024-01-03", exit_price: 9.2, exit_reason: "stop_loss", pnl: -800,
      pattern_ids: ["p1"] }]
  });
  assert.match(html, /盈利因子/);
  assert.match(html, /股票/);
  assert.match(html, /ETF/);
  assert.match(html, /日线/);
  assert.match(html, /W底/);
  assert.match(html, /止损/);
  assert.doesNotMatch(html, /stock|w_bottom|stop_loss|\b1d\b/);
});
