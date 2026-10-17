#!/usr/bin/env python3
"""画单标的K线：day0水平支撑/阻力线 + Zigzag拐点 +（可选）Wyckoff事件标记。

用法:
    python3 scripts/plot_zones.py --code 515450 --kind etf                 # 全历史水平线+Zigzag
    python3 scripts/plot_zones.py --code 515450 --kind etf --since 2025-06-01 --events
"""
import argparse
import yaml
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stock-day0-matplotlib")
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib import font_manager      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff.analysis.engine import run_analysis      # noqa: E402
from wyckoff.config import DATA_DIR, StrategyConfig   # noqa: E402
from wyckoff.data_store import load_symbol            # noqa: E402

MARKS = {"SPRING": ("^", "#e67e22"), "TEST": ("P", "#8e44ad"), "ST": ("o", "#16a085"),
         "SPRING_FAIL": ("x", "#7f0000"), "SC": ("s", "#2c3e50"), "SC_CANDIDATE": (".", "#999999"), "SC_FAIL": ("x", "#7f0000"), "SOW": ("v", "#c0392b"), "UT": ("v", "#d35400"),
         "JOC": ("D", "#e74c3c"), "SOS": ("*", "#f39c12"), "LPS": ("p", "#1abc9c")}
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#8c564b", "#e377c2",
           "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5"]


def pick_cjk_font():
    for name in ("PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"):
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            return name
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True)
    p.add_argument("--kind", default="auto", choices=["auto", "stock", "etf"])
    p.add_argument("--since", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--events", action="store_true", help="叠加 Spring/ST/SC 等事件标记")
    p.add_argument("--set", nargs="*", default=[], help="扁平参数覆盖 key=value")
    args = p.parse_args()

    df = load_symbol(args.code, None if args.kind == "auto" else args.kind)
    if df is None:
        print("无数据")
        return
    cfg = StrategyConfig.load()
    if args.set:
        overrides = {}
        for kv in args.set:
            k, v = kv.split("=", 1)
            overrides[k] = yaml.safe_load(v)
        cfg = cfg.with_overrides(**overrides)
    res = run_analysis(df, cfg, code=args.code)

    feats = res.features
    if args.since:
        feats = feats[feats["date"] >= args.since]
        keep = set(feats.index)
        events = [e for e in res.events if e.confirm_idx in keep]
    else:
        events = res.events
    idx = feats.index.to_numpy()
    n = len(idx)
    x = np.arange(n)
    o, h, l, c = (feats[k].to_numpy(float) for k in ("open", "high", "low", "close"))

    font = pick_cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(18, 8))
    up = c >= o
    ax.vlines(x, l, h, color=np.where(up, "#c0392b", "#27ae60"), linewidth=0.7)
    ax.bar(x[up], np.maximum((c - o)[up], 1e-6), 0.7, bottom=o[up], color="#c0392b")
    ax.bar(x[~up], np.maximum((o - c)[~up], 1e-6), 0.7, bottom=c[~up], color="#27ae60")

    # Current/as-of snapshot, projected back to the earliest member pivot as requested.
    # Dotted before this exact cluster became known; solid from known_from onward.
    # Cluster envelopes are explanatory graphics; trading still uses the median line.
    lines = res.snapshot_levels
    last_label_y = -np.inf
    label_gap = .014*(h.max()-l.min())
    for i, lv in enumerate(sorted(lines, key=lambda z:z.price)):
        start = max(0, int(np.searchsorted(idx, lv.first_pivot_idx)))
        known = max(start, min(n-1, int(np.searchsorted(idx, lv.known_from))))
        col = tuple(.75*v for v in plt.cm.turbo(.06+.88*i/max(len(lines)-1,1))[:3])
        if start >= n:
            continue
        ax.fill_between([start,n-1], lv.bottom, lv.top, color=col, alpha=.09, zorder=0)
        for price, width in [(lv.bottom,.7),(lv.top,.7),(lv.price,1.3)]:
            if known>start:
                ax.hlines(price,start,known,color=col,linewidth=width,linestyle=':',alpha=.6)
            ax.hlines(price,known,n-1,color=col,linewidth=width,alpha=.9)
        role = '支撑' if lv.price<=c[-1] else '阻力'
        tag = f'R{i+1} {role} {lv.bottom:.3f}–{lv.top:.3f} (n={lv.peak_count})' + (' 待确认' if lv.provisional else '')
        label_y = max(lv.price,last_label_y+label_gap)
        last_label_y = label_y
        ax.plot([n-1,n+n*.006],[lv.price,label_y],color=col,linewidth=.5,alpha=.65)
        ax.text(n+n*.008,label_y,tag,fontsize=7,color=col,va='center')
        first_price = next(p.price for p in res.pivots if p.price_idx==lv.first_pivot_idx)
        if idx[0]<=lv.first_pivot_idx<=idx[-1]:
            ax.plot(start,first_price,'o',color=col,markersize=4,zorder=5)

    # Zigzag 折线 + 拐点三角
    piv = [pv for pv in res.pivots if idx[0] <= pv.price_idx <= idx[-1]]
    if piv:
        pos = {p_.price_idx: int(np.searchsorted(idx, p_.price_idx)) for p_ in piv}
        px = [pos[p_.price_idx] for p_ in piv]
        py = [p_.price for p_ in piv]
        ax.plot(px, py, color="#666666", linewidth=0.8, alpha=0.6, zorder=3)
        for p_ in piv:
            j = pos[p_.price_idx]
            ax.plot(j, p_.price, marker="^" if p_.kind == "low" else "v", markersize=5,
                    color="#666666", markerfacecolor="#666666" if p_.confirmed else "white", zorder=4)

    if args.events:
        failed_springs = {e.meta.get("spring_idx") for e in res.events
                          if e.etype == "SPRING_FAIL" and e.confirm_idx <= idx[-1]}
        for e in events:
            j = int(np.searchsorted(idx, e.confirm_idx))
            if j >= n or idx[j] != e.confirm_idx:
                continue
            sym, col = MARKS.get(e.etype, ("x", "#888"))
            ax.plot(j, e.price, marker=sym, color=col, markersize=9 if e.etype == "SPRING" else 6,
                    markerfacecolor="none" if (e.failed_idx is not None
                                    or (e.etype == "SPRING" and e.idx in failed_springs)) else col,
                    markeredgewidth=1.4, zorder=5)
        ax.legend(handles=[plt.Line2D([], [], marker=v[0], color=v[1], linestyle="", label=k)
                           for k, v in MARKS.items()], loc="upper left", fontsize=8, ncol=5)

    step = max(1, n // 20)
    dts = pd.to_datetime(pd.Series(feats["date"].to_numpy()[::step]))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dts], rotation=45, fontsize=8)
    ax.set_xlim(-1, n * 1.13)
    ax.set_title(f"{res.code} {res.name} · day0 / ZigZag 0.2.2 · "
                 f"{f'merge_percent={cfg.support.merge_percent:g}%' if cfg.support.merge_distance is None else f'merge_distance={cfg.support.merge_distance:g}'} · "
                 f"{len(lines)}个支撑阻力区域 · {'ATR自适应' if cfg.zigzag.adaptive else '固定'}ZigZag阈值")
    fig.text(.02,.015,'同色上下界=簇内拐点最小/最大价，中线=中位数；起点=最早成员拐点。点线为回溯或待确认展示；空心拐点/待确认区域含原版首尾端点，仅作绘图。',
             fontsize=8,color='#666666')
    ax.grid(alpha=0.2)
    out = args.out or str(DATA_DIR / f"{args.code}_day0_regions_full.png")
    fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(out, dpi=130)
    import json
    rows = [{'line_id':lv.line_id,'price':lv.price,'bottom':lv.bottom,'top':lv.top,
             'count':lv.peak_count,'first_pivot_date':str(res.features.date.iloc[lv.first_pivot_idx].date()),
             'known_from_date':str(res.features.date.iloc[lv.known_from].date()) if lv.known_from<len(res.features) else None,
             'provisional':lv.provisional,
             'pivot_indices':list(lv.pivot_indices)} for lv in sorted(lines,key=lambda z:z.price)]
    last_threshold = float(np.clip(cfg.zigzag.atr_mult*res.features.atr.iloc[-1]/res.features.close.iloc[-1],
                                  cfg.zigzag.min_threshold,cfg.zigzag.max_threshold)) if cfg.zigzag.adaptive else cfg.zigzag.threshold
    details = {'code':res.code,'merge_percent':cfg.support.merge_percent,'merge_distance':cfg.support.merge_distance,
               'min_bars_between_peaks':cfg.zigzag.min_bars_between_peaks,'peaks':cfg.zigzag.peaks,'lookback_bars':cfg.support.lookback_bars,
               'zigzag_adaptive':cfg.zigzag.adaptive,'last_zigzag_threshold':last_threshold,
               'confirmed_pivots':sum(p.confirmed for p in res.pivots),'chart_pivots':len(res.pivots),'regions':rows,
               'plot_semantics':'current snapshot extended to earliest member pivot; dotted before knowledge'}
    Path(out).with_suffix('.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    print('保存:',out)
    print(f'确认拐点 {sum(p.confirmed for p in res.pivots)} / 绘图拐点 {len(res.pivots)} / 区域 {len(lines)} / 最新K线阈值={last_threshold:.2%}')


if __name__ == "__main__":
    main()
