(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    const dataNode = root.document.getElementById("fixed-backtest-data");
    const mount = root.document.getElementById("fixed-backtest-root");
    if (dataNode && mount) {
      try { mount.innerHTML = api.render(JSON.parse(dataNode.textContent)); }
      catch (error) { mount.textContent = "报告数据无法读取。"; }
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function escape(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char];
    });
  }

  const labels = {
    dimension: { shape: "形态", kind: "标的类型", timeframe: "周期", year: "年份" },
    kind: { stock: "股票", etf: "ETF" },
    timeframe: { "1d": "日线", "1w": "周线" },
    shape: { w_bottom: "W底", inverse_head_shoulders: "头肩底", triangle_bottom: "底部三角形", bull_flag: "上升旗形", m_top: "M顶", head_shoulders: "头肩顶", triangle_top: "顶部三角形", bear_flag: "下跌旗形", spring: "破底翻", false_breakout: "假突破", multi_shape: "多形态", unknown: "未知" },
    reason: { stop_loss: "止损", take_profit: "止盈", end_of_period: "期末强平", insufficient_cash: "资金不足", position_already_open: "已有持仓", no_next_t1_session: "无T+1交易日" },
    holdout: { historical_development_analysis: "历史留出数据已暴露，非独立样本" }
  };

  function label(value, category) {
    const map = labels[category] || {};
    if (category === "holdout" && value) return "历史留出数据已查看，不是独立未触碰样本";
    return map[value] || value;
  }

  function table(headers, rows, className) {
    const tableClass = className ? " class=\"" + escape(className) + "\"" : "";
    return "<div class=\"scroll\"><table" + tableClass + "><thead><tr>" + headers.map((h) => "<th>" + escape(h) + "</th>").join("") +
      "</tr></thead><tbody>" + rows.map((row) => "<tr>" + row.map((cell) => "<td>" + escape(cell) + "</td>").join("") +
      "</tr>").join("") + "</tbody></table></div>";
  }

  function render(report) {
    const summary = report.summary || {};
    const portfolio = summary.equal_initial_cash_portfolio || {};
    const portfolioMetrics = portfolio.metrics || {};
    const percent = (value) => value == null ? "—" : (Number(value) * 100).toFixed(2) + "%";
    const metrics = [
      ["独立账户", summary.security_timeframe_accounts || 0],
      ["交易数", summary.trade_count || 0],
      ["信号数", summary.signal_count || 0],
      ["不同成交形态", summary.distinct_traded_patterns || 0],
      ["多形态重叠信号", summary.overlapping_multi_pattern_signals || 0],
      ["组合初始资金", Number(portfolio.starting_capital || 0).toFixed(0)],
      ["组合期末权益", Number(portfolio.ending_equity || 0).toFixed(2)],
      ["组合总盈亏", Number(portfolio.net_pnl || 0).toFixed(2)],
      ["等初始资金组合收益", percent(portfolioMetrics.cumulative_return)],
      ["组合年化 / 区间", percent(portfolioMetrics.annualized_return) + " / " +
        (portfolioMetrics.annualization_period_years == null ? "—" : Number(portfolioMetrics.annualization_period_years).toFixed(2) + "年")],
      ["组合胜率", percent(portfolioMetrics.win_rate)],
      ["组合最大回撤", percent(portfolioMetrics.maximum_drawdown)],
      ["组合盈利因子", portfolioMetrics.profit_factor == null ? "—" : Number(portfolioMetrics.profit_factor).toFixed(2)]
    ];
    const cards = "<div class=\"cards\">" + metrics.map((item) => "<div class=\"card\"><small>" + escape(item[0]) +
      "</small><b>" + escape(item[1]) + "</b></div>").join("") + "</div>";
    const groups = (report.groups || []).map((row) => [label(row.dimension, "dimension"), label(row.value, row.dimension), row.trade_count,
      row.win_rate == null ? "—" : (row.win_rate * 100).toFixed(1) + "%", Number(row.net_pnl).toFixed(2),
      row.profit_factor == null ? "—" : Number(row.profit_factor).toFixed(2)]);
    const accounts = (report.results || []).map((item) => {
      const m = item.metrics || {};
      return [item.code, label(item.kind, "kind"), label(item.timeframe, "timeframe"), m.trade_count,
        m.win_rate == null ? "—" : (m.win_rate * 100).toFixed(1) + "%",
        (Number(m.cumulative_return || 0) * 100).toFixed(2) + "%",
        (Number(m.annualized_return || 0) * 100).toFixed(2) + "% / " +
          (m.annualization_period_years == null ? "—" : Number(m.annualization_period_years).toFixed(2) + "年"),
        (Number(m.maximum_drawdown || 0) * 100).toFixed(2) + "%",
        m.profit_factor == null ? "—" : Number(m.profit_factor).toFixed(2),
        m.average_pnl == null ? "—" : Number(m.average_pnl).toFixed(2),
        m.average_holding_days == null ? "—" : Number(m.average_holding_days).toFixed(1) + "天",
        m.average_holding_sessions == null ? "—" : Number(m.average_holding_sessions).toFixed(1) + "个交易日",
        Object.entries(m.exit_reason_counts || {}).map(([reason, count]) => label(reason, "reason") + ":" + count).join(" / ") || "—",
        (Number(m.capital_utilization || 0) * 100).toFixed(1) + "%",
        item.data_start, item.trade_start, item.evaluation_start, item.data_end,
        item.split && item.split.development_cutoff, label(item.split && item.split.holdout_status, "holdout")];
    });
    const trades = (report.trades || []).map((trade) => [trade.code, label(trade.kind, "kind"), label(trade.timeframe, "timeframe"),
      (trade.pattern_labels || trade.pattern_types || []).map((value) => label(value, "shape")).join(" / "), trade.signal_date, trade.entry_date,
      Number(trade.entry_price).toFixed(3), trade.exit_date, Number(trade.exit_price).toFixed(3),
      label(trade.exit_reason, "reason"), Number(trade.pnl).toFixed(2), (trade.pattern_ids || []).join(", ")]);
    const patterns = (report.patterns || []).map((item) => [item.code, label(item.kind, "kind"), label(item.timeframe, "timeframe"),
      label(item.pattern_type, "shape"), item.pattern_id]);
    return cards + "<section><h2>账户指标</h2><small>年化按整段回测期计算并显示周期；没有交易时不可定义的交易比率以—显示。</small>" +
      table(["代码", "类型", "周期", "交易数", "胜率", "累计收益", "年化 / 区间", "最大回撤", "盈利因子", "平均盈亏", "平均持有天数", "平均持有交易日", "退出原因", "资金使用率", "行情起始", "交易起始", "评价起始", "数据截止", "冻结切点", "留出说明"], accounts, "account-table") + "</section>" +
      "<section><h2>分组表现</h2><small>形态分组遇到多标签交易时会重复计入；其他账户收益不合并。</small>" +
      table(["分组", "类别", "交易数", "胜率", "净盈亏", "盈利因子"], groups) + "</section>" +
      "<section><h2>逐笔交易</h2>" + table(["代码", "类型", "周期", "形态标签", "信号日", "入场日", "入价", "退出日", "出价", "原因", "盈亏", "形态ID"], trades) + "</section>" +
      "<section><h2>去重形态明细</h2>" + table(["代码", "类型", "周期", "形态", "形态ID"], patterns) + "</section>";
  }

  return { escape: escape, render: render };
});
