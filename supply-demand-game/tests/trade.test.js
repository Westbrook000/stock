import test from "node:test";
import assert from "node:assert/strict";
import { openTrade, stepTrade } from "../js/engine/trade.js";

const B = (o, h, l, c) => ({ open: o, high: h, low: l, close: c });

test("限价单：价格触及克线才成交，未触及不成交", () => {
  const t = openTrade({ side: "buy", entry: 101, stop: 99.5, targets: [{ r: 2, fraction: 1 }], size: 1 });
  const r1 = stepTrade(t, B(105, 105.5, 102, 104)); // 没碰到101
  assert.equal(r1.filled, false);
  const r2 = stepTrade(t, B(103, 103, 100.5, 102)); // 下影插到100.5 → 成交
  assert.equal(r2.filled, true);
  assert.equal(t.entryPrice, 101); // 限价按挂单价成交，不按更优影线
});

test("止损在竭线内：影线扫损后行情反转 —— 被洗出", () => {
  const t = openTrade({ side: "buy", entry: 101, stop: 100.8, targets: [{ r: 8, fraction: 1 }] , size: 1});
  stepTrade(t, B(101.5, 102, 100.5, 101.8)); // 成交
  const out = stepTrade(t, B(101.8, 103, 100.2, 102.5)); // 下影100.2扫止损后收回
  assert.equal(out.closed, true);
  assert.equal(out.reason, "stop");
  assert.ok(t.realizedR < 0);
});

test("止损在竭线外+ATR缓冲：同样影线存活", () => {
  const t = openTrade({ side: "buy", entry: 101, stop: 99.5, targets: [{ r: 2, fraction: 1 }], size: 1 });
  stepTrade(t, B(101.5, 102, 100.5, 101.8));
  const out = stepTrade(t, B(101.8, 103, 100.2, 102.5));
  assert.equal(out.closed, false);
});

test("分批止盈：1号目标平一半并推保护至保本", () => {
  const t = openTrade({ side: "buy", entry: 100, stop: 96, targets: [{ r: 2, fraction: 0.5 }, { r: 4, fraction: 0.5 }], size: 1, plan: { protectAtR: 2 } });
  stepTrade(t, B(100, 108, 99.5, 107)); // 冲过2R=108? high 108 触发
  assert.equal(t.partialClosed.length, 1);
  assert.equal(t.breakevenStop, true);   // 推保护
  assert.equal(t.stop, 100);             // 止损上移到入场价
  const out = stepTrade(t, B(107, 107.5, 99, 99.5)); // 回落击保本止损
  assert.equal(out.closed, true);
  assert.equal(out.reason, "stop");
  assert.ok(Math.abs(t.realizedR) < 1e-9 || t.realizedR >= -0.01); // 保本≈0R（一半仓位已+2R）
});

test("同根K线先止损后止盈：保守按止损计（不确定顺序）", () => {
  const t = openTrade({ side: "buy", entry: 100, stop: 97, targets: [{ r: 2, fraction: 1 }], size: 1 });
  stepTrade(t, B(100, 100.5, 99.9, 100.2));
  const out = stepTrade(t, B(100.2, 106, 96, 105)); // 一根K线同时穿越止损与2R
  assert.equal(out.closed, true);
  assert.equal(out.reason, "stop");
});

test("空头镜像：供给区做空，止盈在需求区上方，破供给区竭线上方止损", () => {
  const t = openTrade({ side: "sell", entry: 130, stop: 132.5, targets: [{ r: 2, fraction: 1 }], size: 1 });
  stepTrade(t, B(129, 130.5, 128, 129)); // 触及130成交
  const out = stepTrade(t, B(129, 129.5, 124.5, 125)); // 到2R=125
  assert.equal(out.closed, true);
  assert.equal(out.reason, "target");
  assert.ok(t.realizedR > 1.9);
});
