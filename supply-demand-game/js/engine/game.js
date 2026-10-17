// 游戏流程逻辑：区域验证 / 评分对账 / 决策校验 / 关卡评级（全部可测试的纯函数）
import { ltsdAction } from "./ltsd.js";
import { scoreFreshness, scoreStrength, scoreDwell, scoreRR, gradeTrade, GRADE_LABELS } from "./score.js";
import { classifyBar } from "./candles.js";

// 阶段1：玩家画区域（框选基带 + 声明类型）。基带范围±1根容差。
export function validateZoneGuess(guess, truthZones, bars) {
  const hit = truthZones.find(z => z.type === guess.type
    && Math.abs(z.baseStart - guess.baseStart) <= 1
    && Math.abs(z.baseEnd - guess.baseEnd) <= 1);
  if (hit) return { ok: true, zone: hit };

  const wrongType = truthZones.find(z => Math.abs(z.baseStart - guess.baseStart) <= 1
    && Math.abs(z.baseEnd - guess.baseEnd) <= 1);
  if (wrongType) {
    const name = wrongType.type === "D" ? "需求区" : "供给区";
    return { ok: false, hint: `位置找对了，但出线方向说明这是${name}（${patternZh(wrongType.pattern)}）——暴涨离开的是需求区，暴跌离开的是供给区。` };
  }
  if (bars && guess.baseStart >= 0) {
    const slice = bars.slice(guess.baseStart, guess.baseEnd + 1);
    if (slice.some(b => classifyBar(b) === "dynamic")) {
      return { ok: false, hint: "你框的范围里有实体≥振幅50%的动态K线——那里的订单已被消耗，不会囤积未成交单。基带必须是一串小实体静态K线。" };
    }
    return { ok: false, hint: "这里没有区域：静态区之后必须紧跟一根暴力动态K线「出线」（离开幅度≥5%且实体≥1.5ATR）才构成供需区。没有出线的盘整只是噪音。" };
  }
  return { ok: false, hint: "该位置未形成供需区。" };
}

export function patternZh(p) {
  return { "drop-base-rally": "跌平涨", "rally-base-rally": "涨平涨",
    "rally-base-drop": "涨平跌", "drop-base-drop": "跌平跌" }[p] || p;
}

// 阶段2：玩家给四维打分，逐维对账并给出书中依据
export function validateScoreGuess(guess, zone) {
  const correct = {
    freshness: scoreFreshness(zone.touches),
    strength: scoreStrength(zone.impulseBodyAtr, zone.atr ?? 1),
    dwell: scoreDwell(zone.baseBars),
    rr: scoreRR(zone.rr),
  };
  const why = {
    freshness: `价格回踩该区域${zone.touches}次——${zone.touches === 0 ? "从未被触碰，未成交买单满仓，最新鲜" : zone.touches === 1 ? "已被消耗一次，只剩半血" : "反复消耗，订单所剩无几"}`,
    strength: `出线动态K线实体为${(zone.impulseBodyAtr ?? 1).toFixed(1)}倍ATR——离开越暴力，供求越失衡`,
    dwell: `基带由${zone.baseBars}根静态K线组成——${zone.baseBars >= 3 ? "滞留久，囤积订单多" : "时间短，囤积有限"}`,
    rr: `目标盈亏比${zone.rr}:1——书中要求离场至少2:1`,
  };
  const dims = {};
  for (const k of Object.keys(correct)) {
    dims[k] = { given: guess[k], correct: correct[k], ok: guess[k] === correct[k], why: why[k] };
  }
  const total = Object.values(correct).reduce((a, b) => a + b, 0);
  return { dims, total, grade: gradeTrade(total), gradeLabel: GRADE_LABELS[gradeTrade(total)] };
}

// 阶段3：交易决策校验（方向 vs L.T.S.D九宫格；止损 vs 竭线；目标 vs 2:1）
export function validateDecision(decision, zone, ctx) {
  const errors = [];
  const want = ltsdAction(zone.type, ctx.position, ctx.trend);
  if (decision.action !== want) {
    errors.push(`L.T.S.D一览表判定：${posZh(ctx.position)}位 + ${trendZh(ctx.trend)}势 + ${zone.type === "D" ? "需求" : "供给"}区 → ${actZh(want)}。你在批发价想买/在零售价想卖了吗？`);
  }
  if (decision.action === "buy" && decision.stop >= zone.jieLine) {
    errors.push("止损必须放在竭线之外（下方）——线内的止损会被影线扫掉，那是给市场倾向交的学费。");
  }
  if (decision.action === "sell" && decision.stop <= zone.jieLine) {
    errors.push("空单止损必须放在供给区竭线上方——同一条规则，镜像方向。");
  }
  if (!decision.targetsR || decision.targetsR.length === 0 || Math.max(...decision.targetsR) < 2) {
    errors.push("离场点位盈亏比至少2:1（书中离场规则01）——1.5:1的目标不值得开仓。");
  }
  return { ok: errors.length === 0, errors, ltsdWant: want };
}

const posZh = p => ({ high: "高", mid: "中", low: "低", above: "极高", below: "极低" }[p] || p);
const trendZh = t => ({ up: "上升", down: "下降", side: "横盘" }[t] || t);
const actZh = a => ({ buy: "买入", sell: "卖出", wait: "等待" }[a] || a);

// 阶段4：关卡结算评级
export function gradeEpisode({ zoneOk, scoreErr, decisionOk, realizedR }) {
  let stars = 3;
  if (!zoneOk) stars -= 1;
  if (scoreErr >= 2) stars -= 1;
  if (!decisionOk) stars -= 1;
  if (realizedR < -0.5 && decisionOk && zoneOk && scoreErr < 2) stars = Math.max(stars, 2); // 决策对但行情亏：不苛责
  return { stars: Math.max(0, Math.min(3, stars)) };
}

export function experienceFor({ stars, realizedR, tradeGrade }) {
  let xp = stars * 100 + Math.max(0, realizedR) * 50;
  if (tradeGrade === "prime") xp *= 1.25;
  else if (tradeGrade === "good") xp *= 1.1;
  if (realizedR < 0) xp = stars * 40; // 亏损保留学习经验
  return Math.round(xp);
}
