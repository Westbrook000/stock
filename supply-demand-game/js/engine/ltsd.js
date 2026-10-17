// L.T.S.D 体系：Location方位 / Trend趋势 / Supply&Demand点位
// 方位：大级别供给区克线与大级别需求区克线之间三等分（高/中/低）
export function locatePosition(price, bigSupplyKe, bigDemandKe) {
  if (price > bigSupplyKe) return "above";
  if (price < bigDemandKe) return "below";
  const span = bigSupplyKe - bigDemandKe;
  if (span <= 0) return "mid";
  const p = (price - bigDemandKe) / span;
  if (p >= 2 / 3) return "high";
  if (p <= 1 / 3) return "low";
  return "mid";
}

function ema(values, period) {
  const k = 2 / (period + 1);
  let e = values[0];
  const out = [e];
  for (let i = 1; i < values.length; i++) {
    e = values[i] * k + e * (1 - k);
    out.push(e);
  }
  return out;
}

export function detectTrend(closes) {
  if (closes.length < 16) return "side";
  const fast = ema(closes, 5);
  const slow = ema(closes, 13);
  const n = closes.length - 1;
  const slope = fast[n] - fast[Math.max(0, n - 3)];
  if (fast[n] > slow[n] * 1.001 && slope > 0) return "up";
  if (fast[n] < slow[n] * 0.999 && slope < 0) return "down";
  return "side";
}

// 书中 L.T.S.D 一览表
const TABLE = {
  S: {
    high: { up: "sell", side: "sell", down: "sell" },
    mid: { up: "wait", side: "sell", down: "sell" },
    low: { up: "wait", side: "wait", down: "wait" },
  },
  D: {
    high: { up: "wait", side: "wait", down: "wait" },
    mid: { up: "buy", side: "buy", down: "wait" },
    low: { up: "buy", side: "buy", down: "buy" },
  },
};

export function ltsdAction(zoneType, position, trend) {
  const pos = position === "above" ? "high" : position === "below" ? "low" : position;
  return (TABLE[zoneType]?.[pos]?.[trend]) ?? "wait";
}
