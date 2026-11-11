# -*- coding: utf-8 -*-

from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'reports/daily-research/v17-directional-suppression-neckline-revisions-100-20261110-attempt02'
partial = json.loads((RUN / 'partial-profit-v17.json').read_text())
ratchet = json.loads((RUN / 'ratchet-experiment-v17.json').read_text())
targets = json.loads((RUN / 'exit-target-grid-v17.json').read_text())
old_report = (ROOT / 'reports/daily-research/evolution-100-v8-v12-v15-v16-20261110.html').read_text()
match = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', old_report, re.S)
if not match:
    raise SystemExit('Could not locate v8 reference metrics')
v8 = json.loads(match.group(1))['metrics']['v8']

rows = []
def add(label, m, caveat=''):
    rows.append((label, m, caveat))
add('历史参考 v8', v8, '既有同100只报告；与本次所用CSV未能字节级证明一致')
add('v17 基线：止损8% / 止盈20%', partial['variants']['all_20_baseline'])
for k, label in [('8','全仓 +8%止盈'),('10','全仓 +10%止盈'),('12','全仓 +12%止盈'),('15','全仓 +15%止盈')]:
    add(label, targets['variants'][k]['summary'], '单一目标敏感性试验')
add('分批：60%在+8%，40%在+20%', partial['variants']['60pct_8_40pct_20'])
add('分批：70%在+8%，30%在+20%', partial['variants']['70pct_8_30pct_20'])
add('分批：50%在+8%，50%在+20%', partial['variants']['half_8_half_20'])
add('全仓：触及+8%后止损抬至+1%，仍看+20%', ratchet['variants']['trigger_8_lock_1'])
add('全仓：触及+8%后止损抬至成本，仍看+20%', ratchet['variants']['trigger_8_lock_0'])

def pct(x): return '—' if x is None else f'{x*100:.2f}%'
def num(x): return '—' if x is None else f'{x:,.0f}'
trs=''.join(f"<tr><td>{html.escape(label)}</td><td>{m.get('trade_count', '—'):,}</td><td>{pct(m.get('win_rate'))}</td><td>{pct(m.get('cumulative_return'))}</td><td>{m.get('profit_factor',0):.3f}</td><td>{pct(m.get('maximum_drawdown'))}</td><td>{html.escape(caveat)}</td></tr>" for label,m,caveat in rows)
chart_rows=''.join(f"<div class='barrow'><span>{html.escape(label)}</span><div class='track'><i style='width:{max(0,min(100,(m.get('cumulative_return',0) or 0)*100/65)):.1f}%'></i></div><b>{pct(m.get('cumulative_return'))}</b></div>" for label,m,_ in rows)
bands=[]
for label,m,_ in [('v17基线',partial['variants']['all_20_baseline'],''),('60/40分批',partial['variants']['60pct_8_40pct_20'],''),('70/30分批',partial['variants']['70pct_8_30pct_20'],''),('8%后锁1%',ratchet['variants']['trigger_8_lock_1'],'')]:
    for key,title in [('through_2020','截至2020'),('2021_2023','2021–2023'),('2024_plus','2024以后')]:
        b=m['signal_date_bands'][key]
        bands.append((label,title,b['trades'],b['win_rate'],b['net_pnl']))
band_rows=''.join(f'<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{n:,}</td><td>{pct(w)}</td><td>¥{pnl:,.0f}</td></tr>' for a,b,n,w,pnl in bands)

page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v17退出策略试验</title><style>
:root{{--ink:#142c35;--muted:#5e7078;--line:#dce5e8;--bg:#f2f6f6;--blue:#2866b1;--green:#14806e;--red:#b3544e;--amber:#aa6a10}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1100px;margin:28px auto;padding:0 18px}}header,.card{{background:white;border:1px solid var(--line);border-radius:18px;padding:26px;box-shadow:0 8px 30px #18333b0a;margin-bottom:18px}}header{{background:linear-gradient(135deg,#122f3a,#235664);color:white}}header p{{color:#d5e3e5;margin-bottom:0}}h1{{font-size:30px;line-height:1.2;margin:8px 0}}h2{{font-size:20px;margin:0 0 14px}}.eyebrow{{text-transform:uppercase;letter-spacing:.11em;font-size:12px;opacity:.8}}.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}}.chip{{border:1px solid #ffffff3d;background:#ffffff12;border-radius:99px;padding:5px 10px;font-size:13px}}.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{padding:16px;background:#f5f8f8;border-radius:12px}}.metric strong{{display:block;font-size:24px;margin-top:4px}}.muted{{color:var(--muted)}}.callout{{background:#fff5e6;border-left:4px solid var(--amber);padding:14px 16px;border-radius:8px;margin-top:16px}}.good{{background:#eaf6f2;border-left-color:var(--green)}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:10px 8px;vertical-align:top}}th{{font-size:12px;color:var(--muted);white-space:nowrap}}td:nth-child(n+2){{white-space:nowrap}}.scroll{{overflow:auto}}.barrow{{display:grid;grid-template-columns:minmax(220px,330px) 1fr 85px;align-items:center;gap:12px;margin:10px 0}}.track{{height:13px;background:#e9eff1;border-radius:20px;overflow:hidden}}.track i{{display:block;height:100%;background:linear-gradient(90deg,#36a28c,#2866b1);border-radius:20px}}.barrow b{{text-align:right}}a{{color:#2466a5}}code{{background:#eef2f3;padding:2px 5px;border-radius:4px}}.foot{{font-size:13px;color:var(--muted)}}@media(max-width:700px){{.summary{{grid-template-columns:1fr}}.barrow{{grid-template-columns:1fr 70px}}.barrow .track{{grid-column:1/3;grid-row:2}}}}
</style></head><body><main><header><div class="eyebrow">Caisen · Backtrader · 100-stock full-history replay</div><h1>提高胜率后，收益还剩多少？</h1><p>用同一批100只股票、相同v17信号、每账户10万元，对比分批止盈和触发后抬高止损。识别器没有改动；这是退出管理的研究试验，不是生产策略变更。</p><div class="chips"><span class="chip">截至 2026-10-30</span><span class="chip">8%初始止损 / 20%目标</span><span class="chip">零佣金、零滑点</span><span class="chip">100账户账目全部通过</span></div></header>
<section class="card"><h2>结论先看</h2><div class="summary"><div class="metric">v8参考收益<strong>{pct(v8['cumulative_return'])}</strong><span class="muted">胜率 {pct(v8['win_rate'])}</span></div><div class="metric">v17基线收益<strong>{pct(partial['variants']['all_20_baseline']['cumulative_return'])}</strong><span class="muted">胜率 {pct(partial['variants']['all_20_baseline']['win_rate'])}</span></div><div class="metric">最佳胜率方案<strong>{pct(partial['variants']['70pct_8_30pct_20']['win_rate'])}</strong><span class="muted">收益仅 {pct(partial['variants']['70pct_8_30pct_20']['cumulative_return'])}</span></div></div><div class="callout"><b>本轮没有找到同时胜过 v8 胜率与收益的方案。</b> 60/40分批让胜率超过50%，但累计收益仅17.86%；70/30的胜率略高，收益降至16.17%。8%后锁盈的全仓跟踪止损方案胜率不足50%，并更早退出了部分最终止盈交易。</div><div class="callout good"><b>结果用途：</b>退出试验说明“单纯调止盈/抬止损”不是达到双重目标的充分条件。不能据此把高胜率分批方案直接设为新策略。</div></section>
<section class="card"><h2>收益率对比</h2><p class="muted">以100个独立账户合计初始本金¥10,000,000为分母；按各方案结束权益合计计算，未将账户收益率直接相加。</p>{chart_rows}</section>
<section class="card"><h2>所有方案指标</h2><div class="scroll"><table><thead><tr><th>方案</th><th>交易数</th><th>胜率</th><th>累计收益</th><th>盈利因子</th><th>最大回撤</th><th>说明</th></tr></thead><tbody>{trs}</tbody></table></div></section>
<section class="card"><h2>分时期：胜率提升不是稳定的</h2><div class="scroll"><table><thead><tr><th>方案</th><th>信号日期区间</th><th>交易数</th><th>胜率</th><th>该组交易净盈亏</th></tr></thead><tbody>{band_rows}</tbody></table></div><p class="foot">60/40分批在截至2020和2024以后达到约52%–53%，但2021–2023只有46.24%，该区间净盈亏为负。历史样本均参与过开发，因此这些时期分组只能用于诊断，不能视作独立样本外验证。</p></section>
<section class="card"><h2>分批试验如何计算</h2><ul><li>首次信号仍按原规则在下一交易日开盘买入；每个标的独立账户，100股整手，现金允许范围内买入。</li><li>部分仓位在实际成交价上涨8%时限价止盈；剩余仓位保持原8%止损并继续争取20%止盈。60/40与70/30按100股整手向下取整。</li><li>全仓抬止损试验在收盘确认当日最高价达到触发位后，撤销旧 bracket；新止损从下一根K线生效，20%目标继续有效。该日内最高价不能提前用于同日交易。</li><li>回放基于Backtrader。5种分批组合方案中基线与已保存v17逐笔交易/盈亏一致，100/100账户账目检查通过；跟踪止损方案同样100/100通过现金、权益与仓位核对。</li></ul><p class="foot">注意：历史档案显示v17识别截至日为2026-10-30。本次用当前本地CSV重放并逐账户精确复现v17基线成交，但v8旧报告没有冻结逐行行情快照，故v8比较是同标的、同截止日、同执行口径的报告级参考，不是字节级行情快照配对。费用和滑点均设为0。</p></section>
<section class="card"><h2>下一步决策</h2><ol><li>保留v17识别基线与这份退出实验，不把任一退出变体写入生产规则。</li><li>把研发重心放回因果的动态颈线与形态结构识别：颈线在结构锚点可知后配对冻结，之后新证据只创建新边界版本；每版只允许自身冻结后的首次突破买点。</li><li>优先复验已有动态颈线v3原型（同100只、3147笔、胜率33.71%、收益54.81%）并补齐完整因果扫描；再测试能否在维持足够趋势收益的同时过滤失败突破。</li><li>最终必须在锁定样本上同时超过v8的33.76%胜率与61.34%累计收益，并保持账户账目可复核；目前没有方案达到。</li></ol></section>
<section class="card foot">数据与可复核明细：<a href="partial-profit-v17.json">分批止盈逐账户 JSON</a> · <a href="ratchet-experiment-v17.json">抬高止损逐账户 JSON</a> · <a href="exit-target-grid-v17.json">固定止盈敏感度</a> · <a href="daily-research.html">v17完整回测报告</a> · <a href="../evolution-100-v8-v12-v15-v16-20261110.html">v8/v12/v15/v16对照报告</a><p>实验生成耗时：分批{partial['elapsed_seconds']:.1f}s；跟踪止损{ratchet['elapsed_seconds']:.1f}s。研究结论不构成自动交易指令。</p></section></main></body></html>'''
out=RUN/'exit-management-experiment-v17.html'
out.write_text(page,encoding='utf-8')
print(out)
