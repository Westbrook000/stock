import test from "node:test";
import assert from "node:assert/strict";
import { locatePosition, detectTrend, ltsdAction } from "../js/engine/ltsd.js";

test("方位：大供给克线与大需求克线之间三等分", () => {
  const bigSupplyKe = 130, bigDemandKe = 100; // 区间30 → 每档10
  assert.equal(locatePosition(128, bigSupplyKe, bigDemandKe), "high");
  assert.equal(locatePosition(115, bigSupplyKe, bigDemandKe), "mid");
  assert.equal(locatePosition(102, bigSupplyKe, bigDemandKe), "low");
  assert.equal(locatePosition(135, bigSupplyKe, bigDemandKe), "above"); // 已入大供给区
  assert.equal(locatePosition(95, bigSupplyKe, bigDemandKe), "below");
});

test("趋势：EMA5 vs EMA13 + 斜率", () => {
  const up = []; let p = 100;
  for (let i = 0; i < 30; i++) { p *= 1.005; up.push(p); }
  assert.equal(detectTrend(up), "up");
  const down = up.slice().reverse();
  assert.equal(detectTrend(down), "down");
  const flat = new Array(30).fill(100);
  assert.equal(detectTrend(flat), "side");
});

test("L.T.S.D 九宫格（书中一览表）", () => {
  // 高位：供给区一律卖出；需求区一律等待
  assert.equal(ltsdAction("S", "high", "up"), "sell");
  assert.equal(ltsdAction("D", "high", "down"), "wait");
  // 中位：供给区 下跌/横盘卖出 上涨等待；需求区 上涨/横盘买入 下跌等待
  assert.equal(ltsdAction("S", "mid", "down"), "sell");
  assert.equal(ltsdAction("S", "mid", "up"), "wait");
  assert.equal(ltsdAction("D", "mid", "up"), "buy");
  assert.equal(ltsdAction("D", "mid", "side"), "buy");
  assert.equal(ltsdAction("D", "mid", "down"), "wait");
  // 低位：需求区一律买入；供给区一律等待
  assert.equal(ltsdAction("D", "low", "down"), "buy");
  assert.equal(ltsdAction("S", "low", "up"), "wait");
});
