// 游戏控制器：菜单→观察→识别→评分→决策→实盘→结算
import { Chart } from "./chart.js";
import { validateZoneGuess, validateScoreGuess, validateDecision, gradeEpisode, experienceFor, patternZh } from "../engine/game.js";
import { zoneKeLine, touchesZone } from "../engine/zones.js";
import { openTrade, stepTrade } from "../engine/trade.js";
import { locatePosition, detectTrend, ltsdAction } from "../engine/ltsd.js";
import { scoreTotal, gradeTrade, GRADE_LABELS } from "../engine/score.js";

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const chart = new Chart($("#kline"));

let profile = JSON.parse(localStorage.getItem("sd_profile") || '{"xp":0,"done":{}}');
let ep = null;          // 当前关卡数据
let phase = "menu";
let pickedZone = null;  // 识别成功的区域
let attempts = 0;
let trade = null, tradeIdx = 0, decision = null;

const PLANS = {
  none: { label: "不推保护" },
  p1: { label: "计划1：2R推保本，3R全离场", protectAtR: 2, exitAllAtR: 3 },
  p2: { label: "计划2：1R推保本，2R全离场", protectAtR: 1, exitAllAtR: 2 },
};

// ---------- 菜单 ----------
async function showMenu() {
  setPhase("menu");
  const idx = await (await fetch("data/episodes/index.json")).json();
  $("#levels").innerHTML = idx.map(e => {
    const stars = profile.done[e.id] ?? 0;
    return `<button class="level" data-id="${e.id}">
      <b>${e.id.split("_")[0]}</b> ${e.name}<span class="tag ${e.type === "D" ? "d" : "s"}">${e.type === "D" ? "需求区" : "供给区"}</span>
      <span class="tag">${patternZh(e.pattern)}</span>
      ${e.touches >= 2 ? `<span class="tag warn">陈旧×${e.touches}</span>` : `<span class="tag ok">新鲜×${e.touches}</span>`}
      <span class="stars">${"★".repeat(stars)}${"☆".repeat(3 - stars)}</span></button>`;
  }).join("");
  $$("#levels .level").forEach(b => b.onclick = () => startEpisode(b.dataset.id));
  $("#hud-xp").textContent = `XP ${profile.xp} · Lv${Math.floor(profile.xp / 500) + 1}`;
}

function setPhase(p) {
  phase = p;
  document.body.dataset.phase = p;
  $$(".panel").forEach(el => el.classList.toggle("hidden", el.dataset.panel !== p));
  $("#phase-label").textContent = { menu: "选择关卡", observe: "① 观察行情", identify: "② 识别区域", score: "③ 概率评分", decide: "④ L.T.S.D 决策", simulate: "⑤ 实盘演绎", settle: "⑥ 结算" }[p];
}

// ---------- 关卡 ----------
async function startEpisode(id) {
  ep = await (await fetch(`data/episodes/${id}.json`)).json();
  pickedZone = null; attempts = 0; trade = null; decision = null;
  chart.zones = []; chart.big = null; chart.trade = null; chart.selection = null;
  chart.setData(ep.bars, ep.decisionIdx);
  $("#ep-title").textContent = `${ep.name}(${ep.code}) · ${ep.bars[ep.decisionIdx].date}`;
  setPhase("observe");
  say("从决策点向左上/左下扫视：找到一根暴力的动态K线（出线），再找它起点前的小实体基带。悬停K线可查看静态/动态分类。");
}

// 观察 → 识别
$("#btn-to-identify").onclick = () => {
  setPhase("identify");
  chart.onSelect = (s, e) => {
    chart.selection = { start: s, end: e };
    $("#sel-info").textContent = `已框选 ${s}~${e}（${e - s + 1}根）`;
    chart.draw();
  };
  say("拖拽框选「静态基带」（1~3根小实体K线，不含出线），然后声明区域类型。");
  $("#zone-type").value = "";
};

$("#btn-confirm-zone").onclick = () => {
  if (!chart.selection) return say("先在图上拖拽框选基带范围。");
  const guess = { type: $("#zone-type").value, baseStart: chart.selection.start, baseEnd: chart.selection.end };
  if (!guess.type) return say("请选择这是需求区还是供给区。");
  const res = validateZoneGuess(guess, ep.zones, ep.bars);
  attempts++;
  if (res.ok) {
    pickedZone = { ...res.zone };
    chart.zones = [pickedZone, ...ep.zones.filter(z => z.id !== res.zone.id && z.createdIdx < ep.decisionIdx && z.baseEnd < ep.decisionIdx - 5)];
    chart.big = { supplyKe: ep.big.supplyKe, demandKe: ep.big.demandKe };
    chart.selection = null;
    chart.draw();
    say(`✅ 识别正确：${patternZh(res.zone.pattern)}，${res.zone.type === "D" ? "需求区（批发价，等着买）" : "供给区（零售价，等着卖）"}。进入评分。`, "good");
    setTimeout(() => enterScore(), 600);
  } else {
    say(`❌ ${res.hint}${attempts >= 3 ? '（第3次失败——点"看答案"继续，记住这次教训）' : ""}`, "bad");
    if (attempts >= 3) $("#btn-reveal").classList.remove("hidden");
  }
};

$("#btn-reveal").onclick = () => {
  pickedZone = { ...ep.zones.find(z => z.id === ep.targetZone) };
  chart.zones = [pickedZone];
  chart.big = { supplyKe: ep.big.supplyKe, demandKe: ep.big.demandKe };
  chart.draw();
  enterScore();
};

function enterScore() {
  setPhase("score");
  const z = pickedZone;
  $("#score-hint").textContent = `区域：${z.baseStart}~${z.baseEnd} 基带 · 出线力度${z.impulseBodyAtr}ATR · 已被回踩${z.touches}次 · 到对面大区域盈亏比${z.rr}:1`;
}

$("#btn-submit-score").onclick = () => {
  const guess = {
    freshness: +$("#s-fresh").value, strength: +$("#s-strong").value,
    dwell: +$("#s-dwell").value, rr: +$("#s-rr").value,
  };
  const res = validateScoreGuess(guess, pickedZone);
  pickedZone.rr = pickedZone.rr ?? 2;
  $("#score-report").innerHTML = Object.entries(res.dims).map(([k, d]) => {
    const name = { freshness: "新鲜度", strength: "力度", dwell: "滞留时间", rr: "盈亏比" }[k];
    return `<div class="dim ${d.ok ? "ok" : "bad"}"><b>${name}</b> 你:${d.given} 正确:${d.correct}<br><small>${d.why}</small></div>`;
  }).join("") + `<div class="dim total">总分 <b>${res.total}/10</b> → ${res.gradeLabel}</div>`;
  pickedZone._scoreTotal = res.total;
  pickedZone._scoreErr = Object.values(res.dims).filter(d => !d.ok).length;
  setTimeout(() => enterDecide(), 400);
};

function enterDecide() {
  setPhase("decide");
  const z = pickedZone;
  const closes = ep.bars.slice(0, ep.decisionIdx + 1).map(b => b.close);
  const price = closes[closes.length - 1];
  const position = locatePosition(price, ep.big.supplyKe, ep.big.demandKe);
  const trend = detectTrend(closes);
  const want = ltsdAction(z.type, position, trend);
  decision = { position, trend, want };
  $("#ltsd-info").innerHTML = `方位: <b>${{ high: "高位", mid: "中位", low: "低位", above: "大供给区内", below: "大需求区内" }[position]}</b>
    · 趋势: <b>${{ up: "上升", down: "下降", side: "横盘" }[trend]}</b>
    · 九宫格判定: <b class="${want === "buy" ? "up" : want === "sell" ? "down" : ""}">${{ buy: "买入", sell: "卖出", wait: "等待" }[want]}</b>
    <br><small>现价${price.toFixed(2)}，大供给${ep.big.supplyKe.toFixed(2)} / 大需求${ep.big.demandKe.toFixed(2)} 三等分定位</small>`;
  const keDefault = zoneKeLine(z, ep.bars, "bodyBottom");
  $("#ke-styles").onchange = () => {
    const ke = zoneKeLine(z, ep.bars, $("#ke-styles").value);
    z._ke = ke;
    chart.draw();
    refreshLevels();
  };
  z._ke = keDefault;
  refreshLevels();
}

function refreshLevels() {
  const z = pickedZone;
  const risk = Math.abs(z._ke - z.jieLine);
  $("#entry-label").textContent = `克线(挂单价) ${z._ke.toFixed(2)} · 风险(到竭线) ${risk.toFixed(2)}`;
  const stop = z.type === "D" ? z.jieLine - 0.5 * (z.atr || risk * 0.2) : z.jieLine + 0.5 * (z.atr || risk * 0.2);
  $("#stop-label").textContent = `止损(竭线外+0.5ATR): ${stop.toFixed(2)}`;
  $("#stop-manual").checked = false;
  decision.stop = stop;
  const t2 = z.type === "D" ? z._ke + 2 * risk : z._ke - 2 * risk;
  $("#t2-label").textContent = `2R目标: ${t2.toFixed(2)}`;
}

$("#stop-manual").onchange = () => {
  $("#stop-slider").disabled = !$("#stop-manual").checked;
  if ($("#stop-manual").checked) {
    const z = pickedZone;
    const risk = Math.abs(z._ke - z.jieLine);
    $("#stop-slider").min = (z.type === "D" ? z._ke - 3 * risk : z.jieLine).toFixed(2);
    $("#stop-slider").max = (z.type === "D" ? z.jieLine : z._ke + 3 * risk).toFixed(2);
    $("#stop-slider").step = 0.01;
    $("#stop-slider").value = $("#stop-slider").max;
    onStopSlide();
  }
};
$("#stop-slider").oninput = onStopSlide;
function onStopSlide() {
  if (!$("#stop-manual").checked) return;
  decision.stop = +$("#stop-slider").value;
  $("#stop-label").textContent = `止损(手动): ${decision.stop.toFixed(2)} ${decision.stop > pickedZone.jieLine ? "⚠ 在竭线内" : ""}`;
}

$("#btn-submit-decision").onclick = () => {
  const z = pickedZone;
  const act = $("#d-action").value;
  if (act === "wait") { finishEpisode({ action: "wait" }); return; }
  const targetsR = [...$$(".t-check:checked")].map(c => +c.value);
  const plan = PLANS[$("#d-plan").value];
  const stop = $("#stop-manual").checked ? decision.stop
    : (z.type === "D" ? z.jieLine - 0.5 * (z.atr || 1) : z.jieLine + 0.5 * (z.atr || 1));
  const res = validateDecision({ action: act, stop, entry: z._ke, targetsR, plan: plan.key }, z, decision);
  $("#decision-errors").innerHTML = res.ok ? "<p class='ok'>✅ 决策符合书中规则，执行。</p>"
    : res.errors.map(e => `<p class='bad'>⚠ ${e}</p>`).join("")
    + `<button id="btn-force" class="primary">坚持我的下注（后果自负）</button>`;
  if (!res.ok) $("#btn-force").onclick = () => place(act, targetsR, plan, stop);
  decision.plan = plan;
  decision.res = res;
  if (res.ok) place(act, targetsR, plan, stop);
};

function place(act, targetsR, plan, stop) {
  const z = pickedZone;
  const entry = z._ke;
  trade = openTrade({
    side: act === "buy" ? "buy" : "sell", entry, stop,
    targets: targetsR.map((r, i) => ({ r, fraction: i === 0 ? 0.5 : 1 })),
    size: 1, plan: plan.protectAtR ? { protectAtR: plan.protectAtR } : null,
  });
  const risk = Math.abs(entry - stop);
  chart.trade = {
    entry, stop,
    targets: targetsR.map(r => act === "buy" ? entry + r * risk : entry - r * risk),
    events: [],
  };
  tradeIdx = ep.decisionIdx + 1;
  setPhase("simulate");
  $("#btn-next-bar").focus();
  say(`限价单已挂在克线 ${entry.toFixed(2)}，止损 ${stop.toFixed(2)}。按空格推进行情，看市场怎么回答你。`);
}

function nextBar() {
  if (!trade || trade.closed) { settle(); return; }
  const bar = ep.bars[tradeIdx];
  if (!bar) { settle(); return; }
  const before = trade.events.length;
  const out = stepTrade(trade, bar);
  for (const ev of trade.events.slice(before)) {
    ev.idx = tradeIdx;
    chart.trade.events.push(ev);
    if (ev.type === "fill") say(`🎯 价格回踩克线，限价单成交于 ${ev.price.toFixed(2)} —— 批发价买入，这就是区域挂单的意义。`, "good");
    if (ev.type === "target") say(`💰 触及${ev.r}R目标，分批止盈一半 @ ${ev.price.toFixed(2)}。`, "good");
    if (ev.type === "protect") say(`🛡 推保护：止损上移至保本 ${ev.price.toFixed(2)}，这笔交易已不可能亏钱。`, "good");
  }
  if (out.closed) {
    chart.trade.events.push({ type: out.reason === "stop" ? "stop" : "exit", price: out.reason === "stop" ? trade.stop : trade.entryPrice, idx: tradeIdx });
    say(out.reason === "stop"
      ? `🛑 止损离场 @ ${trade.stop.toFixed(2)}。${ep.outcome === "busted" ? "区域被击穿=单量衰竭，止损保护了你。" : "影线扫损——止损离竭线太近会被洗出去。"}`
      : `🏁 全部止盈离场。`, out.reason === "stop" && ep.outcome === "busted" ? "good" : "bad");
    setTimeout(settle, 900);
  }
  chart.reveal(1);
  tradeIdx++;
  if (tradeIdx >= ep.bars.length && !trade.closed) {
    trade.realizedR += (trade.side === "buy" ? (ep.bars.at(-1).close - trade.entryPrice) / trade.risk : (trade.entryPrice - ep.bars.at(-1).close) / trade.risk);
    settle();
  }
}

$("#btn-next-bar").onclick = nextBar;
$("#btn-autoplay").onclick = () => {
  if ($("#btn-autoplay").dataset.on) { clearInterval(window._ap); $("#btn-autoplay").dataset.on = ""; return; }
  $("#btn-autoplay").dataset.on = "1";
  window._ap = setInterval(() => { if (phase !== "simulate" || trade?.closed) { clearInterval(window._ap); $("#btn-autoplay").dataset.on = ""; } else nextBar(); }, 350);
};

function finishEpisode({ action }) {
  pickedZone._scoreErr = pickedZone._scoreErr ?? 0;
  const smartWait = action === "wait" && decision.want === "wait";
  const missed = action === "wait" && decision.want !== "wait";
  settle({ action, smartWait, missed });
}

function settle(extra = {}) {
  if (phase === "settle") return;
  setPhase("settle");
  const realizedR = trade ? trade.realizedR : 0;
  const g = gradeEpisode({
    zoneOk: !!pickedZone, scoreErr: pickedZone?._scoreErr ?? 3,
    decisionOk: extra.smartWait || (decision?.res?.ok ?? false), realizedR,
  });
  let stars = g.stars;
  if (extra.missed) stars = Math.max(0, stars - 1);
  const grade = gradeTrade(pickedZone?._scoreTotal ?? 0);
  const xp = experienceFor({ stars, realizedR, tradeGrade: extra.action === "wait" ? "skip" : grade });
  profile.xp += xp;
  profile.done[ep.id] = Math.max(profile.done[ep.id] || 0, stars);
  localStorage.setItem("sd_profile", JSON.stringify(profile));
  const fb = [];
  fb.push(pickedZone ? "✅ 区域识别正确（基带+出线+类型）" : "❌ 区域未识别出来——出线动态K线是区域成立的必要条件");
  fb.push((pickedZone?._scoreErr ?? 3) === 0 ? "✅ 四维评分全对" : `⚠ 评分错${pickedZone?._scoreErr}项——分数决定仓位，不是装饰`);
  if (extra.action === "wait") fb.push(extra.smartWait ? "✅ 九宫格让你等待，你选择了等待——纪律本身就是盈利" : "⚠ 放弃了一次九宫格允许的交易，空仓也是一种下注");
  else fb.push(decision?.res?.ok ? "✅ 止损在竭线外、盈亏比≥2:1、方向符合L.T.S.D" : "⚠ 你带着违规完成了交易——市场会替你交学费");
  fb.push(trade ? `📊 本单结果 ${realizedR >= 0 ? "+" : ""}${realizedR.toFixed(2)}R（真实走势结局：${ep.outcome === "bounce" ? "区域守住反弹" : "区域被击穿"}）` : "未成交/未开仓");
  $("#settle-panel").innerHTML = `
    <div class="stars-big">${"★".repeat(stars)}${"☆".repeat(3 - stars)}</div>
    <div>+${xp} XP（累计 ${profile.xp}）</div>
    <ul class="fb">${fb.map(f => `<li>${f}</li>`).join("")}</ul>
    <button id="btn-retry">重玩本关</button> <button id="btn-menu" class="primary">返回选关</button>`;
  $("#btn-retry").onclick = () => startEpisode(ep.id);
  $("#btn-menu").onclick = showMenu;
}

// ---------- 全局 ----------
function say(msg, kind = "") {
  const el = $("#say");
  el.textContent = msg;
  el.className = kind;
}
document.addEventListener("keydown", e => {
  if (e.code === "Space" && (phase === "observe" || phase === "simulate")) {
    e.preventDefault();
    if (phase === "simulate") nextBar(); else chart.reveal(1);
  }
});
$("#btn-back-menu").onclick = showMenu;
chart.onSelect = null;
showMenu();
