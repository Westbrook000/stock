import test from "node:test";
import assert from "node:assert/strict";
import { detectZones, zoneKeLine, zoneTouches, breakZone, zonePattern } from "../js/engine/zones.js";

// 构造：下跌 → 3根小实体基带 → 暴涨动态K线 = 跌平涨 需求区
function makeBars() {
  const bars = [];
  const push = (o, h, l, c) => bars.push({ open: o, high: h, low: l, close: c });
  let p = 120;
  for (let i = 0; i < 6; i++) { p -= 3; push(p + 3, p + 4, p - 1, p); }        // 下跌段
  const baseStart = bars.length;
  push(102, 103.5, 100.5, 102.5);   // 基带1 小实体(0.5/3)
  push(102.5, 103.2, 100.6, 101.5); // 基带2 小实体(1.0/2.6)
  const baseEnd = bars.length - 1;
  push(102, 112, 101.5, 111.5);     // 出线：暴涨动态K线
  for (let i = 0; i < 4; i++) { const q = 112 + i; push(q, q + 1, q - 0.5, q + 0.5); }
  return { bars, baseStart, baseEnd };
}

test("跌平涨 → 需求区 D", () => {
  const { bars, baseStart, baseEnd } = makeBars();
  const zones = detectZones(bars, { atrPeriod: 5, minImpulseAtr: 1.5, minImpulseChange: 0.05 });
  assert.equal(zones.length, 1);
  const z = zones[0];
  assert.equal(z.type, "D");
  assert.equal(z.pattern, "drop-base-rally");
  assert.equal(z.baseStart, baseStart);
  assert.equal(z.baseEnd, baseEnd);
});

test("涨平跌 → 供给区 S", () => {
  const { bars } = makeBars();
  const mirror = bars.map(b => {
    const k = 220;
    return { open: k - b.open, high: k - b.low, low: k - b.high, close: k - b.close };
  });
  const zones = detectZones(mirror, { atrPeriod: 5, minImpulseAtr: 1.5, minImpulseChange: 0.05 });
  assert.equal(zones.length, 1);
  assert.equal(zones[0].type, "S");
  assert.equal(zones[0].pattern, "rally-base-drop");
});

test("需求区三线：竭线=基带影线最低，克线三种画法", () => {
  const { bars, baseStart, baseEnd } = makeBars();
  const z = detectZones(bars, { atrPeriod: 5, minImpulseAtr: 1.5, minImpulseChange: 0.05 })[0];
  assert.equal(z.jieLine, 100.5); // 基带最低影线
  assert.equal(zoneKeLine(z, bars, "wick"), 103.5);   // 影线顶（风险最高）
  assert.equal(zoneKeLine(z, bars, "bodyTop"), 102.5); // 实体顶（成交几率较低）
  assert.equal(zoneKeLine(z, bars, "bodyBottom"), 101.5); // 实体底（推荐）
  assert.equal(zoneKeLine(z, bars), 101.5); // 默认=推荐画法
});

test("触碰计数：价格回落进区内 touch+1，收盘破竭线区域死亡", () => {
  const { bars } = makeBars();
  const z = detectZones(bars, { atrPeriod: 5, minImpulseAtr: 1.5, minImpulseChange: 0.05 })[0];
  const ke = zoneKeLine(z, bars, "bodyTop");
  const touchBar = { open: 105, high: 105.5, low: ke - 0.5, close: 106 }; // 插一下收回
  assert.equal(zoneTouches(z, bars, touchBar, 30, "bodyTop"), 1);
  breakZone(z, { open: 101, high: 101.5, low: 99, close: 99.5 }, 31);
  assert.equal(z.alive, false);
  assert.equal(z.brokenAt, 31);
});

test("zonePattern 命名四形态", () => {
  assert.equal(zonePattern("down", "up"), "drop-base-rally");
  assert.equal(zonePattern("up", "up"), "rally-base-rally");
  assert.equal(zonePattern("up", "down"), "rally-base-drop");
  assert.equal(zonePattern("down", "down"), "drop-base-drop");
});

test("无暴力出线的盘整不产生区域（动态K线是区域的必要条件）", () => {
  const bars = [];
  for (let i = 0; i < 20; i++) bars.push({ open: 100, high: 101, low: 99, close: 100.2 });
  assert.equal(detectZones(bars, { atrPeriod: 5, minImpulseAtr: 1.5, minImpulseChange: 0.05 }).length, 0);
});
