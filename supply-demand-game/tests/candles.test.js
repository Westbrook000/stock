import test from "node:test";
import assert from "node:assert/strict";
import { classifyBar, bodyTop, bodyBottom, barAtr, isStatic, isDynamic } from "../js/engine/candles.js";

const B = (o, h, l, c) => ({ open: o, high: h, low: l, close: c });

test("静态K线：实体 < 振幅50%", () => {
  // 振幅10 实体3 → 静态
  const bar = B(100, 105, 95, 103);
  assert.equal(classifyBar(bar), "static");
  assert.equal(isStatic(bar), true);
  assert.equal(isDynamic(bar), false);
});

test("动态K线：实体 >= 振幅50%", () => {
  const up = B(100, 106, 99, 105);   // 振幅7 实体5
  const down = B(105, 106, 100, 100); // 振幅6 实体5
  assert.equal(classifyBar(up), "dynamic");
  assert.equal(classifyBar(down), "dynamic");
});

test("十字星(振幅0)归为静态", () => {
  assert.equal(classifyBar(B(100, 100, 100, 100)), "static");
});

test("bodyTop/bodyBottom 不依赖阴阳", () => {
  const bull = B(100, 108, 96, 106);
  const bear = B(106, 108, 96, 100);
  assert.equal(bodyTop(bull), 106);
  assert.equal(bodyTop(bear), 106);
  assert.equal(bodyBottom(bull), 100);
  assert.equal(bodyBottom(bear), 100);
});

test("barAtr: 真实波幅", () => {
  const prevClose = 100;
  // 跳空高开：high 110 low 102 → tr = max(8, |110-100|, |102-100|) = 10
  assert.equal(barAtr(B(105, 110, 102, 108), prevClose), 10);
});
