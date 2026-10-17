import test from "node:test";
import assert from "node:assert/strict";
import { scoreFreshness, scoreStrength, scoreDwell, scoreRR, scoreTotal, gradeTrade } from "../js/engine/score.js";

test("新鲜度：0触碰=3分 1触碰=1.5分 ≥2触碰=0分", () => {
  assert.equal(scoreFreshness(0), 3);
  assert.equal(scoreFreshness(1), 1.5);
  assert.equal(scoreFreshness(2), 0);
  assert.equal(scoreFreshness(5), 0);
});

test("力度：出线动态K线体/ATR ≥2=2分 ≥1.2=1分 否则0", () => {
  assert.equal(scoreStrength(2.4, 1.0), 2);
  assert.equal(scoreStrength(1.5, 1.0), 1);
  assert.equal(scoreStrength(0.9, 1.0), 0);
});

test("滞留时间：基带≥3根=2分 1-2根=1分", () => {
  assert.equal(scoreDwell(3), 2);
  assert.equal(scoreDwell(2), 1);
  assert.equal(scoreDwell(1), 1);
});

test("盈亏比：≥3=3分 ≥2=1.5分 <2=0分（书中要求至少2:1）", () => {
  assert.equal(scoreRR(4), 3);
  assert.equal(scoreRR(2.5), 1.5);
  assert.equal(scoreRR(1.9), 0);
});

test("总分=四维之和，满分10", () => {
  const z = { touches: 0, impulseBodyAtr: 2.5, atr: 1, baseBars: 3, rr: 3 };
  assert.equal(scoreTotal(z), 10);
});

test("交易分级：<5不可交易 5-6低质 7-8可交易 9-10高分重仓", () => {
  assert.equal(gradeTrade(4), "skip");
  assert.equal(gradeTrade(6), "low");
  assert.equal(gradeTrade(8), "good");
  assert.equal(gradeTrade(10), "prime");
});
