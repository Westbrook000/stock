// K线分类：静态（订单积累）与动态（订单消耗）
export function body(bar) {
  return Math.abs(bar.close - bar.open);
}

export function range(bar) {
  return bar.high - bar.low;
}

export function bodyTop(bar) {
  return Math.max(bar.open, bar.close);
}

export function bodyBottom(bar) {
  return Math.min(bar.open, bar.close);
}

export function classifyBar(bar) {
  const r = range(bar);
  if (r <= 1e-9) return "static";
  return body(bar) / r < 0.5 ? "static" : "dynamic";
}

export const isStatic = (bar) => classifyBar(bar) === "static";
export const isDynamic = (bar) => classifyBar(bar) === "dynamic";

export function barAtr(bar, prevClose) {
  return Math.max(
    bar.high - bar.low,
    Math.abs(bar.high - prevClose),
    Math.abs(bar.low - prevClose),
  );
}

export function atrSeries(bars, period = 14) {
  const trs = bars.map((bar, i) =>
    barAtr(bar, i > 0 ? bars[i - 1].close : bar.open));
  const out = new Array(bars.length).fill(NaN);
  let sum = 0;
  for (let i = 0; i < trs.length; i++) {
    sum += trs[i];
    if (i >= period) sum -= trs[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}
