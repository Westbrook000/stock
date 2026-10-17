import test from "node:test";
import assert from "node:assert/strict";
import { validateZoneGuess, validateScoreGuess, validateDecision, gradeEpisode, experienceFor } from "../js/engine/game.js";

const truth = [{
  id: "z1", type: "D", baseStart: 6, baseEnd: 7, pattern: "drop-base-rally",
  keLine: 101, jieLine: 100.5, touches: 0, impulseBodyAtr: 3, baseBars: 2, rr: 3,
  position: "low", trend: "up",
}];

test("区域猜测：基带范围±1根容差命中", () => {
  const hit = validateZoneGuess({ type: "D", baseStart: 6, baseEnd: 7 }, truth);
  assert.equal(hit.ok, true);
  assert.equal(hit.zone.id, "z1");
  const nearMiss = validateZoneGuess({ type: "D", baseStart: 8, baseEnd: 9 }, truth);
  assert.equal(nearMiss.ok, false);
  assert.ok(nearMiss.hint.length > 0); // 必须给出教学提示
  const wrongType = validateZoneGuess({ type: "S", baseStart: 6, baseEnd: 7 }, truth);
  assert.equal(wrongType.ok, false);
  assert.match(wrongType.hint, /需求|供给/);
});

test("评分猜测：逐维反馈对错与书中依据", () => {
  const guess = { freshness: 3, strength: 2, dwell: 1, rr: 3 };
  const res = validateScoreGuess(guess, truth[0]);
  assert.equal(res.total, 9);
  assert.equal(res.dims.freshness.ok, true);
  assert.equal(res.dims.strength.ok, true);
  assert.equal(res.dims.dwell.ok, true);   // baseBars=2 → 1分
  assert.equal(res.dims.rr.ok, true);
  assert.ok(res.dims.freshness.why.includes("0次") || res.dims.freshness.why.includes("新鲜"));
});

test("决策校验：L.T.S.D九宫格 + 止损必须在竭线外", () => {
  const good = validateDecision(
    { action: "buy", stop: 100.0, entry: 101, targetsR: [2, 4], plan: "p1" },
    truth[0], { position: "low", trend: "up" });
  assert.equal(good.ok, true);
  const badStop = validateDecision(
    { action: "buy", stop: 100.8, entry: 101, targetsR: [2], plan: "p1" },
    truth[0], { position: "low", trend: "up" });
  assert.equal(badStop.ok, false);
  assert.match(badStop.errors.join(), /竭线/);
  const wrongSide = validateDecision(
    { action: "sell", stop: 105, entry: 101, targetsR: [2], plan: "p1" },
    truth[0], { position: "low", trend: "up" });
  assert.equal(wrongSide.ok, false);
  assert.match(wrongSide.errors.join(), /L\.T\.S\.D|低位/);
});

test("盈亏比不足2:1的目标被拒绝（书中离场规则）", () => {
  const res = validateDecision(
    { action: "buy", stop: 100.0, entry: 101, targetsR: [1.5], plan: "p2" },
    truth[0], { position: "low", trend: "up" });
  assert.equal(res.ok, false);
  assert.match(res.errors.join(), /2[:：]1|盈亏比/);
});

test("关卡评分：识别/评分/决策/结果四段给星", () => {
  const s = gradeEpisode({ zoneOk: true, scoreErr: 0, decisionOk: true, realizedR: 3 });
  assert.equal(s.stars, 3);
  const s2 = gradeEpisode({ zoneOk: true, scoreErr: 2, decisionOk: false, realizedR: -1 });
  assert.ok(s2.stars <= 1);
});

test("经验值：R倍数与星数换算，高分重仓有加成", () => {
  assert.ok(experienceFor({ stars: 3, realizedR: 4, tradeGrade: "prime" }) >
            experienceFor({ stars: 3, realizedR: 4, tradeGrade: "good" }));
  assert.ok(experienceFor({ stars: 2, realizedR: -1 }) >= 0); // 亏钱也给学习经验但少
});
