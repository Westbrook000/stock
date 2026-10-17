// 交易模拟：限价成交、止损（同K线歧义保守按止损）、分批止盈、推保护
// R记账法：每笔平仓贡献 fraction * (平仓价-入场价)/risk * 方向
export function openTrade({ side, entry, stop, targets = [], size = 1, plan = null }) {
  const risk = Math.abs(entry - stop);
  return {
    side, entry, stop, risk, size,
    filled: false,
    closed: false,
    entryPrice: null,
    remaining: size,
    realizedR: 0,
    partialClosed: [],
    targets: targets.map(t => ({ ...t, hit: false })),
    plan,
    breakevenStop: false,
    events: [],
  };
}

function targetPrice(t, target) {
  return t.side === "buy"
    ? t.entryPrice + target.r * t.risk
    : t.entryPrice - target.r * t.risk;
}

export function stepTrade(t, bar) {
  const result = { filled: false, closed: false, reason: null };
  if (t.closed) return result;
  const buy = t.side === "buy";
  const wasFilled = t.filled;

  if (!wasFilled) {
    const touched = buy ? bar.low <= t.entry : bar.high >= t.entry;
    if (!touched) return result;
    t.filled = true;
    t.entryPrice = t.entry; // 限价按挂单价成交
    result.filled = true;
    t.events.push({ type: "fill", price: t.entryPrice });
  } else {
    // 持仓K线：同K线先检查止损（保守）
    const stopHit = buy ? bar.low <= t.stop : bar.high >= t.stop;
    if (stopHit) return closeAt(t, t.stop, "stop", result);
  }

  for (const target of t.targets) {
    if (target.hit) continue;
    const price = targetPrice(t, target);
    const hit = buy ? bar.high >= price : bar.low <= price;
    if (!hit) continue;
    target.hit = true;
    const qty = Math.min(target.fraction, t.remaining);
    t.remaining -= qty;
    t.realizedR += (target.r * qty) / t.size;
    t.partialClosed.push({ r: target.r, price, qty });
    t.events.push({ type: "target", r: target.r, price });
    if (t.plan && !t.breakevenStop && target.r >= t.plan.protectAtR) {
      t.stop = t.entryPrice;
      t.breakevenStop = true;
      t.events.push({ type: "protect", price: t.entryPrice });
    }
    if (t.remaining <= 1e-9) return closeAt(t, price, "target", result, true);
  }
  return result;
}

function closeAt(t, price, reason, result, alreadyPartial = false) {
  if (!alreadyPartial && t.remaining > 1e-9) {
    const dir = t.side === "buy" ? 1 : -1;
    const r = ((price - t.entryPrice) * dir) / t.risk;
    t.realizedR += (r * t.remaining) / t.size;
    t.partialClosed.push({ r, price, qty: t.remaining });
  }
  t.remaining = 0;
  t.closed = true;
  t.closedReason = reason;
  t.events.push({ type: reason === "stop" ? "stop" : "exit", price });
  result.closed = true;
  result.reason = reason;
  return result;
}
