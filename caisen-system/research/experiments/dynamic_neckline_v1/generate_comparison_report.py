# -*- coding: utf-8 -*-
"""Build an offline HTML report for the paired 100-stock recognizer comparisons."""

from __future__ import annotations

import gzip
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAISEN = HERE.parents[2]
WORKSPACE = CAISEN.parent
RESULTS = HERE / "results"
OUT = RESULTS / "paired-100-v2-20261110-attempt01"
PAIR = CAISEN / "reports/daily-research/paired-100-v8-v12-20261109/comparison.json"
V15 = CAISEN / "reports/daily-research/v15-suppression-neckline-revision-100-20261110-attempt03/daily-research.json"
V8_ACCOUNTS = CAISEN / "reports/daily-research/full-20261104-v8-thread2-boundary/accounts"
V12_ACCOUNTS = CAISEN / "reports/daily-research/full-unified-v12-20261106-w4-rerun/accounts"
V1_DIR = RESULTS / "paired-100-v1-20261110-attempt01"
V2_DIR = RESULTS / "paired-100-v2-20261110-attempt01"


def pct(value, digits=2):
    return "—" if value is None else f"{100 * float(value):,.{digits}f}%"


def money(value, digits=0):
    return f"¥{float(value):,.{digits}f}"


def load_trade(directory, code, signal_date):
    data = json.load(open(directory / f"stock-{code}.json", encoding="utf-8"))
    return next((row for row in data.get("trades", [])
                 if row.get("signal_date") == signal_date), None)


def _svg_chart(code, start, end, annotations, neckline_specs=(), title=""):
    path = WORKSPACE / "data/stocks" / f"{code}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (frame.dropna(subset=["date", "open", "high", "low", "close"])
             .sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
             .reset_index(drop=True))
    frame["source_index"] = np.arange(len(frame))
    view = frame[(frame.date >= pd.Timestamp(start)) & (frame.date <= pd.Timestamp(end))]
    if view.empty:
        return "<div class='chart-empty'>本地行情窗口无数据</div>"
    view = view.reset_index(drop=True)
    width, height = 1000, 370
    left, right, top, bottom = 62, 975, 28, 300
    lo = float(view.low.min())
    hi = float(view.high.max())
    pad = max((hi - lo) * 0.08, 0.01)
    lo, hi = lo - pad, hi + pad
    n = len(view)
    xstep = (right - left) / max(n - 1, 1)
    candle_w = min(7.0, max(1.2, xstep * 0.56))
    date_index = {row.date.strftime("%Y-%m-%d"): int(row.source_index)
                  for row in frame.itertuples()}
    original_indices = {int(row.source_index): index for index, row in enumerate(view.itertuples())}

    def sx(local):
        return left + local * xstep

    def sy(price):
        return bottom - (float(price) - lo) / (hi - lo) * (bottom - top)

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title or code)}'>",
             f"<defs><clipPath id='clip-{code}-{start.replace('-', '')}'><rect x='{left}' y='{top}' width='{right-left}' height='{bottom-top}'/></clipPath></defs>"]
    for j in range(5):
        price = lo + (hi - lo) * j / 4
        y = sy(price)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}' class='grid'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='axis'>{price:.2f}</text>")
    parts.append(f"<g clip-path='url(#clip-{code}-{start.replace('-', '')})'>")
    for i, row in enumerate(view.itertuples()):
        x = sx(i)
        up = row.close >= row.open
        color = "#d94b52" if up else "#138a70"
        y1, y2 = sy(row.high), sy(row.low)
        yo, yc = sy(row.open), sy(row.close)
        body_y, body_h = min(yo, yc), max(abs(yc - yo), 1.0)
        parts.append(f"<line x1='{x:.1f}' y1='{y1:.1f}' x2='{x:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='1.05'/>")
        parts.append(f"<rect x='{x-candle_w/2:.1f}' y='{body_y:.1f}' width='{candle_w:.1f}' height='{body_h:.1f}' fill='{color}' opacity='.86'/>")
    for spec in neckline_specs:
        trade = spec.get("trade") or {}
        tag = next(iter(trade.get("pattern_tags") or []), {})
        selection = tag.get("neckline_selection") or {}
        line = selection.get("selected_line") or {}
        try:
            slope, intercept = float(line["slope"]), float(line["intercept"])
            first_src = date_index.get(str(selection.get("effective_start_date")))
            last_src = date_index.get(str(trade.get("signal_date")))
            if first_src is None or last_src is None:
                continue
            visible_start = max(first_src, int(view.source_index.iloc[0]))
            visible_end = min(last_src, int(view.source_index.iloc[-1]))
            if visible_end < visible_start:
                continue
            x1 = sx(original_indices[visible_start])
            x2 = sx(original_indices[visible_end])
            y1 = sy(intercept + slope * visible_start)
            y2 = sy(intercept + slope * visible_end)
            parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{spec['color']}' stroke-width='2.6' stroke-dasharray='{spec.get('dash', '')}'/>")
        except (KeyError, TypeError, ValueError):
            continue
    for item in annotations:
        date = item.get("date")
        source_idx = date_index.get(str(date))
        if source_idx is None or source_idx not in original_indices:
            continue
        x = sx(original_indices[source_idx])
        label = html.escape(str(item.get("label", "")))
        color = item.get("color", "#295a9f")
        if item.get("price") is not None:
            y = sy(item["price"])
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.2' fill='{color}' stroke='white' stroke-width='1.5'/>")
            label_y = max(top + 12, y - 9)
        else:
            parts.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{bottom}' stroke='{color}' stroke-width='1.4' stroke-dasharray='4 4' opacity='.8'/>")
            label_y = top + 14 + int(item.get("label_offset", 0))
        parts.append(f"<text x='{min(max(x, left+36), right-36):.1f}' y='{label_y:.1f}' text-anchor='middle' fill='{color}' class='marker-label'>{label}</text>")
    parts.append("</g>")
    tick_count = min(5, n)
    for k in range(tick_count):
        local = round(k * (n - 1) / max(tick_count - 1, 1))
        label = view.date.iloc[local].strftime("%Y-%m")
        parts.append(f"<text x='{sx(local):.1f}' y='{bottom+24}' text-anchor='middle' class='axis'>{label}</text>")
    parts.append(f"<text x='{left}' y='{height-12}' class='axis'>{html.escape(title)}</text></svg>")
    return "".join(parts)


def _account_v1_stats():
    returns, drawdowns, trades = [], [], []
    for path in sorted(V1_DIR.glob("stock-*.json")):
        data = json.load(open(path, encoding="utf-8"))
        returns.append(data["metrics"]["cumulative_return"])
        drawdowns.append(data["metrics"]["maximum_drawdown"])
        trades.extend(data.get("trades", []))
    pnls = [float(trade["pnl"]) for trade in trades]
    positive = [value for value in pnls if value > 0]
    negative = -sum(value for value in pnls if value < 0)
    return {"trades": len(trades), "win_rate": sum(x > 0 for x in pnls) / len(pnls),
            "profit_factor": sum(positive) / negative, "net_pnl": sum(pnls),
            "positive_accounts": sum(x > 0 for x in returns),
            "median_dd": float(np.percentile(drawdowns, 50)),
            "p90_dd": float(np.percentile(drawdowns, 90)),
            "max_dd": max(drawdowns)}


def _relative_account_distribution(pair, version, metric):
    values = [account[version][metric] for account in pair["account_ranking"]]
    return {"p10": float(np.percentile(values, 10)), "median": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "max": max(values),
            "positive": sum(value > 0 for value in values)}


def build_report():
    pair = json.load(open(PAIR, encoding="utf-8"))
    v15 = json.load(open(V15, encoding="utf-8"))
    v1 = json.load(open(V1_DIR / "summary.json", encoding="utf-8"))
    v2 = json.load(open(V2_DIR / "summary.json", encoding="utf-8"))
    old = pair["versions"]["old"]["summary"]
    new = pair["versions"]["new"]["summary"]
    newer = v15["summary"]
    v1stats = _account_v1_stats()
    old_ret_dist = _relative_account_distribution(pair, "old", "cumulative_return")
    new_ret_dist = _relative_account_distribution(pair, "new", "cumulative_return")
    old_dd_dist = _relative_account_distribution(pair, "old", "maximum_drawdown")
    new_dd_dist = _relative_account_distribution(pair, "new", "maximum_drawdown")

    # The paired comparison decomposition is exact by its trade matching keys.
    bridge = pair["pnl_bridge"]
    removed = pair["removed_added_breakdown"]["old_only"]
    added = pair["removed_added_breakdown"]["new_only"]
    shape_old = pair["versions"]["old"]["groups"]["shape"]
    shape_new = pair["versions"]["new"]["groups"]["shape"]
    shape_rows = []
    for key in ("w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"):
        a, b = shape_old[key], shape_new[key]
        shape_rows.append((key, a, b, b["trade_count"] - a["trade_count"],
                           b["net_pnl"] - a["net_pnl"]))

    examples = []
    t_old = load_trade(V8_ACCOUNTS, "600633", "2025-02-06")
    t_new = load_trade(V12_ACCOUNTS, "600633", "2025-02-17")
    if t_old and t_new:
        examples.append({"code": "600633", "title": "同一轮急涨：旧版更早识别，v12 换成后来的 W 底信号",
                         "start": "2024-09-01", "end": "2025-03-15",
                         "old_trade": t_old, "new_trade": t_new,
                         "old_line": [{"trade": t_old, "color": "#7456d8"}],
                         "new_line": [{"trade": t_new, "color": "#e58b24", "dash": "6 4"}],
                         "notes": [
                             f"v8：2025-02-06 首次信号，次日开盘 {t_old['entry_price']:.2f} 买入，{t_old['exit_date']} 止盈，交易回报 {pct(t_old['return_pct'])}。",
                             f"v12：未在 02-06 产生该头肩底买入信号；02-17 以 W 底信号买入，次日开盘 {t_new['entry_price']:.2f}，{t_new['exit_date']} 止损，交易回报 {pct(t_new['return_pct'])}。",
                             "这是两个算法在同一股票上的不同信号，不是对同一笔订单的完全配对实验。价格之后的走势不能单独证明旧形态识别正确。",
                         ]})
    t_795 = load_trade(V8_ACCOUNTS, "600795", "2023-11-10")
    if t_795:
        examples.append({"code": "600795", "title": "盈利机会被删减的账户：老版 2023 年 W 底",
                         "start": "2023-04-01", "end": "2024-04-15",
                         "old_trade": t_795, "new_trade": None,
                         "old_line": [{"trade": t_795, "color": "#7456d8"}], "new_line": [],
                         "notes": [
                             f"v8：2023-11-10 信号，{t_795['entry_date']} 买入，{t_795['exit_date']} 达到止盈，单笔收益 {pct(t_795['return_pct'])}、盈利 {money(t_795['pnl'])}。",
                             "v12 在这一日期没有对应买入信号；该股票账户由 24 笔降至 12 笔，累计收益从 +447.5% 降至 +41.5%。",
                             "同账户另有 2020、2022 年旧版止盈交易未在 v12 重现，说明交易频率降低时，丢失的并不全是亏损单。",
                         ]})
    t_538 = load_trade(V8_ACCOUNTS, "000538", "2017-03-21")
    if t_538:
        examples.append({"code": "000538", "title": "大幅分歧：000538 的旧版 W 底交易",
                         "start": "2016-10-01", "end": "2017-12-15",
                         "old_trade": t_538, "new_trade": None,
                         "old_line": [{"trade": t_538, "color": "#7456d8"}], "new_line": [],
                         "notes": [
                             f"v8：{t_538['signal_date']} 信号，{t_538['entry_date']} 入场，{t_538['exit_date']} 止盈，单笔回报 {pct(t_538['return_pct'])}。",
                             "v12 的同股票账户由 37 笔、+202.0% 变为 19 笔、-0.9%；未找到该日期对应信号。",
                             "这说明收益变化集中在被移除的历史机会；它不证明旧版更准确，需用盲评判断该 W 底本身是否成立。",
                         ]})

    case_html = []
    for item in examples:
        annotations = []
        old_trade = item["old_trade"]
        annotations.extend([
            {"date": old_trade["entry_date"], "price": old_trade["entry_price"], "label": "v8 买入", "color": "#245db3"},
            {"date": old_trade["exit_date"], "price": old_trade["exit_price"], "label": "v8 退出", "color": "#138a70"},
        ])
        new_trade = item["new_trade"]
        if new_trade:
            annotations.extend([
                {"date": new_trade["entry_date"], "price": new_trade["entry_price"], "label": "v12 买入", "color": "#d87921", "label_offset": 16},
                {"date": new_trade["exit_date"], "price": new_trade["exit_price"], "label": "v12 退出", "color": "#c94949", "label_offset": 30},
            ])
        chart = _svg_chart(item["code"], item["start"], item["end"], annotations,
                           item["old_line"] + item["new_line"], item["title"])
        notes = "".join(f"<li>{html.escape(note)}</li>" for note in item["notes"])
        case_html.append(f"""<article class='case card'><div class='case-head'><div><span class='eyebrow'>{item['code']} · 实际日K</span><h3>{html.escape(item['title'])}</h3></div><span class='tag'>研究样例，不作标签真值</span></div><div class='chart'>{chart}</div><div class='legend'><span><i style='background:#7456d8'></i>v8 颈线 / 交易</span><span><i style='background:#e58b24'></i>v12 颈线</span><span><i style='background:#245db3'></i>v8 入场</span><span><i style='background:#c94949'></i>v12/止损退出</span></div><ul class='case-notes'>{notes}</ul></article>""")

    shape_html = []
    labels = {"w_bottom": "W 底", "inverse_head_shoulders": "头肩底",
              "triangle_bottom": "收敛三角形底部", "bull_flag": "下飘旗形上涨中继"}
    for name, a, b, trades_delta, pnl_delta in shape_rows:
        shape_html.append(f"<tr><td>{labels[name]}</td><td>{a['trade_count']:,}</td><td>{b['trade_count']:,}</td><td class='{'down' if trades_delta < 0 else 'up'}'>{trades_delta:+,}</td><td>{money(a['net_pnl'])}</td><td>{money(b['net_pnl'])}</td><td class='{'down' if pnl_delta < 0 else 'up'}'>{money(pnl_delta)}</td><td>{pct(a['win_rate'])} → {pct(b['win_rate'])}</td></tr>")

    years = pair["year_groups"]
    year_rows = []
    for decade in years["old"]:
        a, b = years["old"][decade], years["new"][decade]
        year_rows.append(f"<tr><td>{decade}</td><td>{a['count']:,}</td><td>{money(a['pnl'])}</td><td>{b['count']:,}</td><td>{money(b['pnl'])}</td><td class='down'>{money(b['pnl']-a['pnl'])}</td></tr>")

    # Matched account-level outliers anchor the cumulative return change.
    accounts = {row["code"]: row for row in pair["account_ranking"]}
    outlier_codes = ["600795", "600691", "600633", "000538", "600717"]
    outlier_rows = []
    for code in outlier_codes:
        row = accounts[code]
        outlier_rows.append(f"<tr><td>{code}</td><td>{row['old']['trade_count']}</td><td>{pct(row['old']['cumulative_return'])}</td><td>{pct(row['old']['maximum_drawdown'])}</td><td>{row['new']['trade_count']}</td><td>{pct(row['new']['cumulative_return'])}</td><td>{pct(row['new']['maximum_drawdown'])}</td><td class='down'>{money(row['delta'])}</td></tr>")

    old_only = pair["old_only_metrics"]
    new_only = pair["new_only_metrics"]
    total_delta = new["net_pnl"] - old["net_pnl"]
    shape_rows_html = "".join(shape_html)
    decade_rows_html = "".join(year_rows)
    outlier_rows_html = "".join(outlier_rows)
    examples_html = "".join(case_html)
    oldonly_wins = removed["wins"]
    oldonly_losses = removed["losses"]
    oldonly_wr = removed["wins"] / max(old_only["count"], 1)
    v2_changes = v2["total_trades"] - v1["total_trades"]
    runtime_change = v1["runtime_seconds"] / v2["runtime_seconds"] if v2["runtime_seconds"] else None

    html_doc = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>蔡森识别策略前后100标的收益变化分析</title><style>
:root{{--ink:#172033;--muted:#65728a;--line:#e1e7ef;--paper:#f5f7fb;--white:#fff;--blue:#2c5aa0;--violet:#7658d5;--teal:#14846e;--amber:#cf861b;--red:#c54b4b;--shadow:0 14px 38px rgba(24,39,75,.07)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif}}.wrap{{max-width:1180px;margin:auto;padding:32px 24px 70px}}header{{padding:38px 42px;background:linear-gradient(125deg,#172e52,#244b7b 62%,#4f6f9c);color:white;border-radius:24px;box-shadow:var(--shadow);position:relative;overflow:hidden}}header:after{{content:'';position:absolute;width:330px;height:330px;border:1px solid rgba(255,255,255,.15);border-radius:50%;right:-80px;top:-160px;box-shadow:0 0 0 34px rgba(255,255,255,.04),0 0 0 70px rgba(255,255,255,.03)}}.eyebrow{{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#7891b6;font-weight:700}}header .eyebrow{{color:#c7d7ed}}h1{{font-size:clamp(27px,4vw,42px);line-height:1.2;margin:12px 0 14px;max-width:800px}}header p{{max-width:840px;color:#e1e9f4;margin:0}}.meta{{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}}.meta span,.tag{{border:1px solid rgba(255,255,255,.25);padding:5px 10px;border-radius:999px;font-size:12px;color:inherit}}section{{margin-top:28px}}h2{{font-size:24px;margin:0 0 14px}}h3{{font-size:18px;margin:5px 0 0}}p{{margin:8px 0 12px}}.hero{{padding:23px 27px;background:#fff9ea;border:1px solid #f2deb4;border-left:5px solid var(--amber);border-radius:14px;margin-top:22px}}.hero strong{{font-size:18px}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:20px}}.metric{{background:white;border:1px solid var(--line);border-radius:15px;padding:17px 18px;box-shadow:var(--shadow)}}.metric small{{display:block;color:var(--muted)}}.metric b{{display:block;font-size:24px;margin-top:4px;line-height:1.25}}.metric em{{display:block;font-style:normal;color:var(--muted);font-size:12px;margin-top:4px}}.card{{background:white;border:1px solid var(--line);border-radius:18px;padding:22px 24px;box-shadow:var(--shadow);margin-top:14px}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}}table{{border-collapse:collapse;width:100%;font-size:13.5px;min-width:760px}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#f0f4f9;color:#53627b;font-weight:700;position:sticky;top:0}}tr:last-child td{{border-bottom:0}}.down{{color:#b84242;font-weight:650}}.up{{color:#087b68;font-weight:650}}.bridge{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:18px 0}}.bridge div{{border-radius:12px;padding:14px;background:#f3f6fa;min-height:94px}}.bridge div:not(:last-child){{position:relative}}.bridge b{{display:block;font-size:19px;margin-top:4px}}.bridge small{{color:var(--muted)}}.bar-chart{{display:grid;gap:9px;margin:16px 0}}.bar-row{{display:grid;grid-template-columns:145px 1fr 90px;align-items:center;gap:12px;font-size:13px}}.track{{height:13px;background:#eef2f7;border-radius:10px;overflow:hidden}}.fill{{height:100%;border-radius:10px;background:linear-gradient(90deg,#7295d2,#395f9d)}}.fill.red{{background:linear-gradient(90deg,#e39696,#c44e4e)}}.pill{{display:inline-block;background:#edf3fb;color:#365783;padding:4px 9px;border-radius:999px;font-size:12px;margin:2px}}.case-head{{display:flex;justify-content:space-between;align-items:start;gap:12px;margin-bottom:12px}}.case .tag{{border-color:#e5eaf0;color:#617087;white-space:nowrap}}.chart{{border:1px solid #e6ebf2;border-radius:13px;background:#fcfdff;overflow:hidden}}svg{{display:block;width:100%;height:auto}}.grid{{stroke:#e8edf3;stroke-width:1}}.axis{{fill:#738096;font-size:11px}}.marker-label{{font-size:11px;font-weight:700;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}}.legend{{display:flex;flex-wrap:wrap;gap:16px;color:#69768a;font-size:12px;margin:9px 2px 12px}}.legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}.case-notes{{padding-left:20px;margin:8px 0 0}}.case-notes li{{padding:3px 0}}.callout{{border-left:3px solid var(--blue);background:#f2f6fc;padding:12px 15px;margin:14px 0;border-radius:0 10px 10px 0}}.warning{{border-left-color:var(--red);background:#fff4f3}}.good{{border-left-color:var(--teal);background:#eff9f5}}.small{{font-size:12.5px;color:var(--muted)}}.steps{{counter-reset:step;display:grid;gap:10px}}.step{{display:grid;grid-template-columns:38px 1fr;gap:13px;align-items:start}}.step:before{{counter-increment:step;content:counter(step);width:32px;height:32px;display:grid;place-items:center;background:#eaf0f8;color:#345a8e;border-radius:10px;font-weight:800}}.footer{{padding:22px;color:var(--muted);font-size:12px;text-align:center}}@media(max-width:850px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}.bridge{{grid-template-columns:1fr 1fr}}header{{padding:28px 24px}}.wrap{{padding:16px 12px 45px}}.card{{padding:17px}}.bar-row{{grid-template-columns:105px 1fr 75px}}}}@media(max-width:460px){{.metrics{{grid-template-columns:1fr 1fr}}.metric b{{font-size:20px}}.bridge{{grid-template-columns:1fr}}}}
</style></head><body><div class='wrap'><header><span class='eyebrow'>CAISEN · 100 STOCKS · PAIRED FULL-HISTORY STUDY</span><h1>策略改动后，收益为什么下降？</h1><p>用相同100只股票、相同截止日和固定8%止损／20%止盈，对照 v8 与 v12；再把 v15 及“动态换颈线”研究实验放在一起，分清收益、胜率、风险和交易频率各自发生了什么。</p><div class='meta'><span>固定样本：100只股票</span><span>统一截止：2026-10-30</span><span>每标的独立10万元</span><span>零费用 / 零滑点</span><span>报告生成：2026-11-10</span></div></header>
<div class='hero'><strong>结论：v12收益下降的主因是交易次数与资金使用率大幅下降，不是单笔交易胜率明显变差。</strong><p>v8到v12的胜率几乎不变（33.76%→33.61%），单笔平均盈亏约¥1,623→¥1,625，盈利因子反而从1.236升到1.269；但实际交易少了55.8%，资金使用率从10.46%降到4.09%。代价是累计收益由+61.34%降至+27.13%；收益回落同时，组合回撤也由11.82%降至5.33%。</p></div>
<div class='metrics'><div class='metric'><small>v8累计收益</small><b>{pct(old['cumulative_return'])}</b><em>净收益 {money(old['net_pnl'])}</em></div><div class='metric'><small>v12累计收益</small><b>{pct(new['cumulative_return'])}</b><em>净收益 {money(new['net_pnl'])}</em></div><div class='metric'><small>累计收益变化</small><b class='down'>−{money(abs(total_delta))}</b><em>−{pct(abs(new['cumulative_return']-old['cumulative_return']))}（百分点）</em></div><div class='metric'><small>胜率变化</small><b>{pct(old['win_rate'])} → {pct(new['win_rate'])}</b><em>只下降 {100*(new['win_rate']-old['win_rate']):.2f} 个百分点</em></div><div class='metric'><small>组合最大回撤</small><b>{pct(old['maximum_drawdown'])} → {pct(new['maximum_drawdown'])}</b><em>下降约 {100*(old['maximum_drawdown']-new['maximum_drawdown'])/old['maximum_drawdown']:.1f}%</em></div></div>
<section><h2>一、对照是否公平</h2><div class='card'><p>这部分是主要结论依据。比对数据确认：样本均为同一组100只股票；共同截止日为<strong>2026-10-30</strong>；100/100个行情前缀一致，旧版重放100/100匹配原账户，已配对的1,221笔交易共用相同撮合结果。</p><div class='grid2'><div><h3>v8 → v12 关键指标</h3><div class='table-wrap'><table><thead><tr><th>指标</th><th>v8</th><th>v12</th><th>变化</th></tr></thead><tbody><tr><td>信号日期数</td><td>{pair['signal_overlap']['old_dates']:,}</td><td>{pair['signal_overlap']['new_dates']:,}</td><td class='down'>−{pair['signal_overlap']['old_dates']-pair['signal_overlap']['new_dates']:,}</td></tr><tr><td>实际交易</td><td>{old['trade_count']:,}</td><td>{new['trade_count']:,}</td><td class='down'>−{100*(1-new['trade_count']/old['trade_count']):.1f}%</td></tr><tr><td>单笔平均盈亏</td><td>{money(old['expectancy'])}</td><td>{money(new['expectancy'])}</td><td class='up'>基本持平</td></tr><tr><td>盈利因子</td><td>{old['profit_factor']:.3f}</td><td>{new['profit_factor']:.3f}</td><td class='up'>+{new['profit_factor']-old['profit_factor']:.3f}</td></tr><tr><td>资金使用率</td><td>{pct(old['capital_utilization'])}</td><td>{pct(new['capital_utilization'])}</td><td class='down'>−{100*(old['capital_utilization']-new['capital_utilization']):.2f}个百分点</td></tr><tr><td>盈利账户数</td><td>{old_ret_dist['positive']}/100</td><td>{new_ret_dist['positive']}/100</td><td class='down'>少{old_ret_dist['positive']-new_ret_dist['positive']}个</td></tr></tbody></table></div></div><div><h3>交易频率与资金使用</h3><div class='bar-chart'><div class='bar-row'><span>信号日期 v8</span><div class='track'><div class='fill' style='width:100%'></div></div><b>{pair['signal_overlap']['old_dates']:,}</b></div><div class='bar-row'><span>信号日期 v12</span><div class='track'><div class='fill red' style='width:{100*pair['signal_overlap']['new_dates']/pair['signal_overlap']['old_dates']:.1f}%'></div></div><b>{pair['signal_overlap']['new_dates']:,}</b></div><div class='bar-row'><span>实际交易 v8</span><div class='track'><div class='fill' style='width:100%'></div></div><b>{old['trade_count']:,}</b></div><div class='bar-row'><span>实际交易 v12</span><div class='track'><div class='fill red' style='width:{100*new['trade_count']/old['trade_count']:.1f}%'></div></div><b>{new['trade_count']:,}</b></div><div class='bar-row'><span>资金使用 v8</span><div class='track'><div class='fill' style='width:100%'></div></div><b>{pct(old['capital_utilization'])}</b></div><div class='bar-row'><span>资金使用 v12</span><div class='track'><div class='fill red' style='width:{100*new['capital_utilization']/old['capital_utilization']:.1f}%'></div></div><b>{pct(new['capital_utilization'])}</b></div></div><p class='small'>信号日期不是订单数。存在持仓时会拒绝新的入场信号；实际交易还受当时现金和整手限制。</p></div></div></div>
<div class='card'><h3>净收益减少的逐笔拆分</h3><p>在交易匹配定义下，v8→v12净收益减少 <strong>{money(abs(total_delta))}</strong>。被v12移除的2,559笔旧版交易本身净贡献为正；v12新增448笔交易也为正贡献，但不足以抵消旧版被移除的收益。</p><div class='bridge'><div><small>v8净收益起点</small><b>{money(old['net_pnl'])}</b></div><div><small>移除 v8-only 交易</small><b class='down'>−{money(old_only['pnl'])}</b><small>{old_only['count']:,}笔，净贡献为正</small></div><div><small>加入 v12-only 交易</small><b class='up'>+{money(new_only['pnl'])}</b><small>{new_only['count']:,}笔</small></div><div><small>共同交易盈亏变化</small><b class='down'>{money(bridge['shared_trade_pnl_delta'])}</b><small>1,221笔匹配交易</small></div><div><small>v12净收益终点</small><b>{money(new['net_pnl'])}</b></div></div><p>v8-only交易里有<strong>{oldonly_wins:,}笔盈利、{oldonly_losses:,}笔亏损</strong>，胜率{pct(oldonly_wr)}，但盈利合计{money(removed['winning_pnl'])}，亏损合计{money(removed['losing_pnl'])}，最终净贡献<strong>{money(old_only['pnl'])}</strong>。严格筛选删掉了更多亏损单，同时也删掉了有正期望的盈利单。</p><div class='callout warning'><strong>归因边界：</strong>v8-only表示“在新版实际成交记录中找不到对应成交”。个别缺失也可能受新版已有持仓、入场先后及复利现金影响，不能把2,559笔全部简化成“颈线识别漏掉”。整体信号数量、日期重叠和逐笔PNL桥接能证明收益差落在哪，但每笔仍需按信号/拒单事件再分因。</div></div></section>
<section><h2>二、主要变化集中在哪里</h2><div class='card'><h3>形态分组：W底是收益差额的最大来源</h3><div class='table-wrap'><table><thead><tr><th>形态标签</th><th>v8笔数</th><th>v12笔数</th><th>笔数变化</th><th>v8净收益</th><th>v12净收益</th><th>净收益变化</th><th>胜率 v8 → v12</th></tr></thead><tbody>{shape_rows_html}</tbody></table></div><p class='small'>单笔可挂多个形态标签，分组可能重复计入同一交易，不能把各形态净收益变化直接相加。W底笔数从2,789降到756，是最大的活动量变化；其净收益减少约{money(shape_old['w_bottom']['net_pnl']-shape_new['w_bottom']['net_pnl'])}。W底胜率也由{pct(shape_old['w_bottom']['win_rate'])}略降至{pct(shape_new['w_bottom']['win_rate'])}，说明不能只用W底交易数量解释全部识别质量。</p></div><div class='grid2'><div class='card'><h3>旧版重复结构信号</h3><p>配对报告统计，v8有<strong>{pair['repeated_structure_signal_occurrences']['old']['repeated_occurrences']}</strong>次同一日期结构重复出现；v12为<strong>{pair['repeated_structure_signal_occurrences']['new']['repeated_occurrences']}</strong>。这类收敛有助于减少同结构重复开仓，但也降低了参与次数。</p><p>结构时间映射中，{pair['structure_timing']['old_structure_no_new_signal']:,}个旧版结构未在新版产生对应买入，{pair['structure_timing']['new_structure_only']:,}个新版独有，{pair['structure_timing']['unchanged']:,}个未变，另有{pair['structure_timing']['earlier']:,}个新版较早、{pair['structure_timing']['later']:,}个较晚。这些是算法事件映射统计，不是人工准确率。</p></div><div class='card'><h3>各年代收益都变少</h3><div class='table-wrap'><table><thead><tr><th>年代</th><th>v8笔数</th><th>v8净收益</th><th>v12笔数</th><th>v12净收益</th><th>变化</th></tr></thead><tbody>{decade_rows_html}</tbody></table></div><p class='small'>最大差额落在2000年代和2010年代。v12不是只避开某一个差行情，而是全时期减少了相当一部分盈利机会。</p></div></div></section>
<section><h2>三、典型个股案例</h2><p>图上K线和交易点来自本地行情文件，紫线为v8该笔交易保存的颈线；橙虚线为v12保存的颈线。图形只解释算法差异与实际成交路径，不替代盲评形态真值。</p>{examples_html}<div class='card'><h3>账户层差异最大的样本</h3><div class='table-wrap'><table><thead><tr><th>代码</th><th>v8笔数</th><th>v8收益</th><th>v8回撤</th><th>v12笔数</th><th>v12收益</th><th>v12回撤</th><th>净收益差额</th></tr></thead><tbody>{outlier_rows_html}</tbody></table></div><p class='small'>收益按各标的独立10万元账户截至统一截止日计算；收益变化受交易顺序、持仓占用和账户复利共同影响。</p></div></section>
<section><h2>四、后续版本和“同底点换颈线”实验</h2><div class='card'><h3>生产回测后续版本：v15并未改变核心结论</h3><div class='table-wrap'><table><thead><tr><th>版本</th><th>成交数</th><th>胜率</th><th>净收益率</th><th>盈利因子</th><th>组合回撤</th><th>资金使用率</th></tr></thead><tbody><tr><td>v8（配对基线）</td><td>{old['trade_count']:,}</td><td>{pct(old['win_rate'])}</td><td>{pct(old['cumulative_return'])}</td><td>{old['profit_factor']:.3f}</td><td>{pct(old['maximum_drawdown'])}</td><td>{pct(old['capital_utilization'])}</td></tr><tr><td>v12</td><td>{new['trade_count']:,}</td><td>{pct(new['win_rate'])}</td><td>{pct(new['cumulative_return'])}</td><td>{new['profit_factor']:.3f}</td><td>{pct(new['maximum_drawdown'])}</td><td>{pct(new['capital_utilization'])}</td></tr><tr><td>v15（独立全量结果）</td><td>{newer['trade_count']:,}</td><td>{pct(newer['win_rate'])}</td><td>{pct(newer['cumulative_return'])}</td><td>{newer['profit_factor']:.3f}</td><td>{pct(newer['maximum_drawdown'])}</td><td>{pct(newer['capital_utilization'])}</td></tr></tbody></table></div><p>v15相对v12累计收益增加仅约{pct(newer['cumulative_return']-new['cumulative_return'])}，胜率下降约{100*(newer['win_rate']-new['win_rate']):.2f}个百分点，回撤提高约{100*(newer['maximum_drawdown']-new['maximum_drawdown']):.2f}个百分点。现有数据不足以称它解决了收益问题。</p></div><div class='card'><h3>研究实验：结构已知后，允许边界版本随压力变化</h3><p>该实验以保存的v12结构为输入，压力点按2ATR确认；初始线可由形态底点之前形成的压力点支持，形态结构点已可知后才可用；后续线至少移动1ATR并经过冷却后才产生新版本。仍使用零费用、次日开盘、8%止损、20%止盈。</p><div class='table-wrap'><table><thead><tr><th>方案</th><th>成交数</th><th>胜率</th><th>净收益率*</th><th>盈利因子</th><th>单账户回撤中位数 / P90 / 最大</th><th>线版本数</th><th>信号数</th></tr></thead><tbody><tr><td>v1：频繁按最新压力点换线</td><td>{v1['total_trades']:,}</td><td>{pct(v1['win_rate'])}</td><td>{pct(v1['equal_start_capital_return'])}</td><td>{v1stats['profit_factor']:.3f}</td><td>{pct(v1stats['median_dd'])} / {pct(v1stats['p90_dd'])} / {pct(v1stats['max_dd'])}</td><td>{v1['total_boundary_revisions']:,}</td><td>{v1['total_signals']:,}</td></tr><tr><td>v2：增加1ATR重大变动门槛</td><td>{v2['total_trades']:,}</td><td>{pct(v2['win_rate'])}</td><td>{pct(v2['equal_start_capital_return'])}</td><td>{v2['profit_factor']:.3f}</td><td>{pct(v2['account_max_drawdown_distribution']['median'])} / {pct(v2['account_max_drawdown_distribution']['p90'])} / {pct(v2['account_max_drawdown_distribution']['max'])}</td><td>{v2['total_boundary_revisions']:,}</td><td>{v2['total_signals']:,}</td></tr><tr><td>v8生产基线</td><td>{old['trade_count']:,}</td><td>{pct(old['win_rate'])}</td><td>{pct(old['cumulative_return'])}</td><td>{old['profit_factor']:.3f}</td><td>{pct(old_dd_dist['median'])} / {pct(old_dd_dist['p90'])} / {pct(old_dd_dist['max'])}</td><td>—</td><td>{pair['signal_overlap']['old_dates']:,}</td></tr></tbody></table></div><p>*净收益率为100个各自10万元账户的总净盈亏除以1,000万元起始资金，不把账户收益相乘。</p><div class='callout warning'><strong>不能把v1/v2的更高总收益视为策略胜利：</strong>v2胜率仅34.47%，没达到50%；账户最大回撤中位数67.66%、P90 83.83%、最差90.20%。它有662,623次边界版本、409,340个信号、10,364笔成交，远多于v8的3,780笔成交。高收益主要伴随重复换线和账户复利，风险/周转不可接受；同时它只从v12既有结构重放，结构覆盖不完整，也没有要求完整形成区间的每根K线都受线覆盖。</div><p>相比v1，v2把边界版本数减少37.2%、实际成交减少14.4%，胜率仅提高0.31个百分点，累计收益率从+401.5%降至+302.2%。在100个账户中，53个账户收益下降、47个上升。允许换线重发买点有时能捕捉后续有效突破，但现有“1ATR变化”定义仍然会生成过多版本，尚不能直接进入生产策略。</p><p class='small'>运行状态：100/100账户完成，0失败，所有账户账目检查通过；8进程约153秒。对照v1约739秒的结果不是严格单因素性能实验：v1串行，v2用了8进程且计算规则也有变化。</p></div></section>
<section><h2>五、为什么胜率没有随筛选提高</h2><div class='grid2'><div class='card'><h3>固定盈亏比决定胜率含义</h3><p>8%止损、20%止盈对应不含跳空与仓位复利的理论盈亏平衡胜率约28.6%。33%上下的胜率仍可能有正期望；胜率单独不足以评估质量。止损跳空会使实际亏损超过8%，报告中也有止盈跨越目标的情况。</p><p>v12胜率只比v8低0.14个百分点，PF更高，说明筛选没有产生强的胜率提升，但平均风险收益尚有改善。与此同时低资金使用率显著限制总回报。</p></div><div class='card'><h3>识别真值还没有被检验</h3><p>更少信号不等于更准确。要回答“漏掉的交易是不是错误形态”，需要将被删信号和新增信号去掉收益结果后盲评结构、颈线和确认日，再解盲对照交易。当前回测胜负不能反过来定义形态真伪。</p><p>前向交易效果还会受复权口径、退市样本、费用、滑点、涨跌停与历史逐日数据可得性影响；本组是当前仍有行情文件的固定股票样本，不代表无幸存偏差的市场总体。</p></div></div></section>
<section><h2>六、建议的下一步</h2><div class='card'><div class='steps'><div class='step'><div><strong>把“识别准确率”和“组合收益”拆开验收。</strong><br>以v8、v12的全部差异信号为盲评池，按形态/周期/年代分层抽样，先看结构点、颈线覆盖和首次确认日，再看盈亏。单独报告误报、漏报和不确定例。</div></div><div class='step'><div><strong>保留换线能力，但只让新的边界版本创建新的突破事件。</strong><br>线必须因果确认、结构相关、覆盖所需K线并满足显著变化；边界更新日不发信号，更新之后必须出现新的“从线下到线上的有效收盘穿越”。历史边界与已发订单不回写。</div></div><div class='step'><div><strong>先做交易级归因，再做收益优化。</strong><br>为每个v8-only候选分别标记：未识别、被父结构抑制、边界不合格、确认延迟、已持仓拒单或现金不足。否则无法判断回撤下降与收益下降分别由哪条规则造成。</div></div><div class='step'><div><strong>评估时同时设活动量和风险边界。</strong><br>至少并列胜率、净期望、PF、回撤、成交保留率和资金使用率。不要单靠累计收益选择v2这类重复换线版本，也不要仅因交易减少就把v12称为更准确。</div></div><div class='step'><div><strong>在固定100只以外做锁定样本。</strong><br>本次100只已用于多轮开发和规则比较。下一轮应冻结不同标的、不同年份的留出组，并保留一个前向观察期；报告中分别列出开发结果与锁定验证结果。</div></div></div></div></section>
<section><h2>数据来源与复现范围</h2><div class='card'><p>主要v8/v12对照：<code>paired-100-v8-v12-20261109/comparison.json</code>（相同100股票、2026-10-30共同截止、共同交易撮合）；后续对照：<code>v15-suppression-neckline-revision-100-20261110-attempt03/daily-research.json</code>；动态换线实验：<code>paired-100-v1-20261110-attempt01</code>与<code>paired-100-v2-20261110-attempt01</code>。</p><p>策略：每标的独立10万元，100股/份整手，次日开盘入场，成交价下方8%止损、上方20%止盈，零佣金/滑点，无持有期限；其余撮合规则沿用Backtrader执行器。</p><p class='small'>核心指标来自已保存的逐笔成交和账户权益序列。胜率、收益和回撤反映该回测设定，不是未来收益承诺。完整逐标的压缩明细与行情/规则指纹位于本网页同目录的配套实验文件夹。</p></div></section><div class='footer'>蔡森量价结构研究 · 100标的配对归因 · 本报告将历史回测证据与研究假设分开展示</div></div></body></html>"""
    output = OUT / "analysis-v8-v12-neckline-revision-100.html"
    output.write_text(html_doc, encoding="utf-8")
    print(output)
    print(f"bytes={output.stat().st_size:,}")


if __name__ == "__main__":
    build_report()
