// 供需区：基带(静态K线) + 暴力出线(动态K线)；克线/竭线；触碰与失效生命周期
import { atrSeries, body, bodyTop, bodyBottom, classifyBar, isDynamic } from "./candles.js";

// 四形态命名（入线方向, 出线方向）
export function zonePattern(preDir, outDir) {
  if (preDir === "down" && outDir === "up") return "drop-base-rally";   // 跌平涨 需求
  if (preDir === "up" && outDir === "up") return "rally-base-rally";    // 涨平涨 需求
  if (preDir === "up" && outDir === "down") return "rally-base-drop";   // 涨平跌 供给
  return "drop-base-drop";                                              // 跌平跌 供给
}

export function detectZones(bars, opts = {}) {
  const {
    atrPeriod = 14, minImpulseAtr = 1.5, minImpulseChange = 0.05,
    maxBaseBars = 3, trendLookback = 4,
  } = opts;
  const atr = atrSeries(bars, atrPeriod);
  const zones = [];
  let idSeq = 1;

  for (let i = atrPeriod; i < bars.length; i++) {
    const bar = bars[i];
    if (!isDynamic(bar)) continue;
    const dir = bar.close > bar.open ? "up" : "down";
    const prevClose = bars[i - 1].close;
    const change = Math.abs(bar.close / prevClose - 1);
    if (change < minImpulseChange) continue;
    if (atr[i] > 0 && body(bar) < minImpulseAtr * atr[i]) continue;

    // 出线前紧邻的静态基带（1~maxBaseBars根）
    let j = i - 1;
    while (j >= 0 && j > i - 1 - maxBaseBars && classifyBar(bars[j]) === "static") j--;
    const baseStart = j + 1;
    const baseEnd = i - 1;
    if (baseStart > baseEnd) continue;
    if (baseStart - 1 - trendLookback < 0) continue;

    const pre = bars[baseStart - 1].close >= bars[baseStart - 1 - trendLookback].close ? "up" : "down";
    const type = dir === "up" ? "D" : "S";
    const pattern = zonePattern(pre, dir);

    const overlap = zones.some(z => z.type === type
      && z.baseStart <= baseEnd && z.baseEnd >= baseStart);
    if (overlap) continue;

    const lows = bars.slice(baseStart, baseEnd + 1).map(b => b.low);
    const highs = bars.slice(baseStart, baseEnd + 1).map(b => b.high);
    const bTops = bars.slice(baseStart, baseEnd + 1).map(bodyTop);
    const bBots = bars.slice(baseStart, baseEnd + 1).map(bodyBottom);

    zones.push({
      id: `z${idSeq++}`,
      type, pattern,
      baseStart, baseEnd, impulseIdx: i,
      jieLine: type === "D" ? Math.min(...lows) : Math.max(...highs),
      keWick: type === "D" ? Math.max(...highs) : Math.min(...lows),
      keBodyTop: Math.max(...bTops),
      keBodyBottom: Math.min(...bBots),
      impulseBodyAtr: atr[i] > 0 ? body(bar) / atr[i] : 1,
      atr: atr[i],
      baseBars: baseEnd - baseStart + 1,
      createdIdx: i,
      touches: 0,
      alive: true,
      brokenAt: null,
    });
  }
  return zones;
}

// 克线三种画法：wick影线极端(风险最高/成交率最高) | bodyTop实体顶 | bodyBottom实体底(推荐)
export function zoneKeLine(zone, bars, style) {
  const s = style || "recommended";
  if (zone.type === "D") {
    if (s === "wick") return zone.keWick;
    if (s === "bodyTop") return zone.keBodyTop;
    if (s === "bodyBottom") return zone.keBodyBottom;
    return zone.keBodyBottom; // 书中推荐画法
  }
  if (s === "wick") return zone.keWick;
  if (s === "bodyBottom") return zone.keBodyBottom;
  if (s === "bodyTop") return zone.keBodyTop;
  return zone.keBodyBottom; // 供给区推荐画法同样是实体最低处（最优价，接受更低成交率）
}

// 一根K线是否触碰区域（从价格侧进入克线以内）
export function touchesZone(zone, bar) {
  if (!zone.alive) return false;
  const ke = zone._ke ?? zone.keBodyBottom;
  if (zone.type === "D") return bar.low <= ke && bar.close > zone.jieLine;
  return bar.high >= ke && bar.close < zone.jieLine;
}

export function zoneTouches(zone, bars, bar, uptoIdx, keStyle) {
  const ke = zoneKeLine(zone, bars, keStyle);
  let n = 0;
  const end = Math.min(uptoIdx, bars.length);
  for (let i = zone.createdIdx + 1; i < end; i++) {
    const b = bars[i];
    if (zone.type === "D" ? b.low <= ke : b.high >= ke) n++;
  }
  if (zone.type === "D" ? bar.low <= ke : bar.high >= ke) n++;
  return n;
}

// 收盘穿越竭线 → 区域死亡（单量衰竭）
export function breakZone(zone, bar, idx) {
  const dead = zone.type === "D" ? bar.close < zone.jieLine : bar.close > zone.jieLine;
  if (dead) {
    zone.alive = false;
    zone.brokenAt = idx;
  }
  return dead;
}
