#!/usr/bin/env python3
"""画单标的K线：合并支撑带 + Zigzag拐点 +（可选）Wyckoff事件标记。

用法:
    python3 scripts/plot_zones.py --code 515450 --kind etf                 # 全历史支撑带+Zigzag
    python3 scripts/plot_zones.py --code 515450 --kind etf --since 2025-06-01 --events
"""
import argparse
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
         "SC": ("s", "#2c3e50"), "SOW": ("v", "#c0392b"), "UT": ("v", "#d35400"),
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
            overrides[k] = float(v) if any(ch in v for ch in ".e") else int(v)
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

    # 支撑带：从拐点形成画到破位（未破位画到最右），每条一个颜色 + 右侧标注
    bands = [lv for lv in res.levels if lv.kind == "support" and lv.created_idx <= idx[-1]]
    for i, lv in enumerate(bands):
        col = PALETTE[i % len(PALETTE)]
        i0 = max(0, int(np.searchsorted(idx, lv.created_idx)))
        i1 = n - 1 if lv.broken_idx is None else min(n - 1, int(np.searchsorted(idx, lv.broken_idx)))
        if i1 <= i0:
            continue
        width = (lv.top - lv.bottom) / ((lv.top + lv.bottom) / 2)
        ax.fill_between(x[i0:i1 + 1], lv.bottom, lv.top, color=col, alpha=0.17, zorder=0)
        ax.hlines([lv.bottom, lv.top], i0, i1, color=col, linewidth=0.8, alpha=0.8, zorder=1)
        tag = f"Z{i+1} {lv.bottom:.3f}~{lv.top:.3f}[{width*100:.1f}%×{lv.touches}]"
        if lv.broken_idx is not None:
            tag += " 破"
        ax.text(i1 + n * 0.004, (lv.bottom + lv.top) / 2, tag, fontsize=7.5, color=col,
                va="center", ha="left")

    # Zigzag 折线 + 拐点三角
    piv = [pv for pv in res.pivots if idx[0] <= pv.price_idx <= idx[-1]]
    if piv:
        pos = {p_.price_idx: int(np.searchsorted(idx, p_.price_idx)) for p_ in piv}
        px = [pos[p_.price_idx] for p_ in piv]
        py = [p_.price for p_ in piv]
        ax.plot(px, py, color="#3b5bdb", linewidth=1.0, alpha=0.75, zorder=3)
        for p_ in piv:
            j = pos[p_.price_idx]
            ax.plot(j, p_.price, marker="^" if p_.kind == "low" else "v", markersize=5,
                    color="#3b5bdb", zorder=4)

    if args.events:
        for e in events:
            j = int(np.searchsorted(idx, e.confirm_idx))
            if j >= n or idx[j] != e.confirm_idx:
                continue
            sym, col = MARKS.get(e.etype, ("x", "#888"))
            ax.plot(j, e.price, marker=sym, color=col, markersize=9 if e.etype == "SPRING" else 6,
                    markerfacecolor="none" if e.failed_idx is not None else col,
                    markeredgewidth=1.4, zorder=5)
        ax.legend(handles=[plt.Line2D([], [], marker=v[0], color=v[1], linestyle="", label=k)
                           for k, v in MARKS.items()], loc="upper left", fontsize=8, ncol=5)

    step = max(1, n // 20)
    dts = pd.to_datetime(pd.Series(feats["date"].to_numpy()[::step]))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dts], rotation=45, fontsize=8)
    ax.set_xlim(-1, n * 1.13)
    thr = (f"自适应3×ATR%[{cfg.zigzag.min_threshold:.0%}~{cfg.zigzag.max_threshold:.0%}]"
           if cfg.zigzag.adaptive else f"{cfg.zigzag.threshold:.0%}")
    ax.set_title(f"{res.code} {res.name} {'全历史' if not args.since else f'{args.since}起'} · "
                 f"支撑带(只向下合并·全局宽度≤{cfg.zigzag.zone_max_width:.0%}) "
                 f"+ Zigzag({thr}/depth{cfg.zigzag.depth})   共{len(bands)}条带 / {len(piv)}个拐点")
    ax.grid(alpha=0.2)
    out = args.out or str(DATA_DIR / f"{args.code}_zones_full.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("保存:", out)


if __name__ == "__main__":
    main()
