// 概率增强系统：新鲜度3 / 力度2 / 滞留时间2 / 盈亏比3，满分10
export function scoreFreshness(touches) {
  if (touches === 0) return 3;
  if (touches === 1) return 1.5;
  return 0;
}

export function scoreStrength(impulseBody, atr) {
  const ratio = atr > 0 ? impulseBody / atr : 0;
  if (ratio >= 2) return 2;
  if (ratio >= 1.2) return 1;
  return 0;
}

export function scoreDwell(baseBars) {
  return baseBars >= 3 ? 2 : 1;
}

export function scoreRR(rr) {
  if (rr >= 3) return 3;
  if (rr >= 2) return 1.5;
  return 0;
}

export function scoreTotal(zone) {
  return scoreFreshness(zone.touches)
    + scoreStrength(zone.impulseBodyAtr, zone.atr ?? 1)
    + scoreDwell(zone.baseBars)
    + scoreRR(zone.rr);
}

export function gradeTrade(total) {
  if (total < 5) return "skip";
  if (total < 7) return "low";
  if (total < 9) return "good";
  return "prime";
}

export const GRADE_LABELS = {
  skip: "放弃（不值得下注）",
  low: "低质量（轻仓试错）",
  good: "可交易（标准仓位）",
  prime: "高分A+（可加大仓位）",
};
