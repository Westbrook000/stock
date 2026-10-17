#!/usr/bin/env python3
"""B版支撑/阻力（照 day0market/support_resistance 的算法）与A版（我们的因果支撑带）对照。

B版流水线（与 pricelevels 包一致）：
  收盘价ZigZag拐点(固定%delta) → min_bars_between_peaks 贪心过滤 →
  Ward层次聚类(distance_threshold = merge_percent×均价/100) → 簇中位数=价位 →
  TouchScorer 打分(cut_body -2 / cut_wick -1 / touch +2 / touch@HL +1)
注意：B版**不是因果的**（拐点要等反转走完才知道，且原库还有 shift(-1)），只能事后看图。

用法:
    python3 scripts/compare_sr.py --code 515450 --kind etf --since 2023-01-01
"""
import argparse
import math
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
from wyckoff.analysis.indicators import add_candle_features  # noqa: E402

PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#8c564b", "#e377c2",
           "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5"]


def pick_cjk_font():
    for name in ("PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"):
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            return name
    return None


# ---------- B版：day0market 风格 ----------
def peak_valley_pivots(close, up_delta, down_delta):
    """等价于 zigzag.peak_valley_pivots：收盘价新高后回撤 down_delta% → 记峰值(+1)；反之谷值(-1)。"""
    n = len(close)
    labels = np.zeros(n, dtype=int)
    if n < 2:
        return labels
    state, ext_i, ext_p = 0, 0, close[0]
    for i in range(1, n):
        p = close[i]
        if state >= 0 and p > ext_p:                       # 抬高候选峰值
            ext_p, ext_i = p, i
            state = 1
        if state <= 0 and p < ext_p:                       # 压低候选谷值
            ext_p, ext_i = p, i
            state = -1
        if state == 1 and ext_p > 0 and (ext_p - p) / ext_p >= down_delta:
            labels[ext_i] = 1                              # 峰值确认
            state, ext_p, ext_i = -1, p, i
        elif state == -1 and p > 0 and (p - ext_p) / ext_p >= up_delta:
            labels[ext_i] = -1                             # 谷值确认
            state, ext_p, ext_i = 1, p, i
    return labels


def filter_by_bars_between(indexes, min_bars):
    """day0 的 _filter_by_bars_between：相邻拐点间隔不足则丢弃后者。"""
    keep, last = [], -10 ** 9
    for i in indexes:
        if i - last >= min_bars:
            keep.append(i)
            last = i
    return keep


def ward_cluster(values, threshold):
    """1维 Ward 层次聚类，distance_threshold 停止（等价 sklearn ward+distance_threshold）。"""
    clusters = [[v] for v in values]
    while len(clusters) > 1:
        best, bi, bj = None, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ni, nj = len(clusters[i]), len(clusters[j])
                d = math.sqrt(ni * nj / (ni + nj)) * abs(np.mean(clusters[i]) - np.mean(clusters[j]))
                if best is None or d < best:
                    best, bi, bj = d, i, j
        if best > threshold:
            break
        merged = clusters[bi] + clusters[bj]
        clusters = [c for k, c in enumerate(clusters) if k not in (bi, bj)] + [merged]
    return clusters


def touch_score(ohlc, price, cut_body=-2, cut_wick=-1, touch_hl=1, touch_normal=2,
                min_gap_between_cuts=5, close_tol_pct=0.05, trend_pct=0.5):
    """day0 TouchScorer 的打分口径。"""
    o, h, l, c = (ohlc[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    score, last_cut = 0.0, -10 ** 9
    for i in range(len(c)):
        body_hi, body_lo = max(o[i], c[i]), min(o[i], c[i])
        cuts_body = body_hi > price > body_lo
        cuts_wick = not cuts_body and h[i] > price > l[i]
        if cuts_body and i - last_cut > min_gap_between_cuts:
            score += cut_body
            last_cut = i
        elif cuts_wick and i - last_cut > min_gap_between_cuts:
            score += cut_wick
            last_cut = i
        elif h[i] <= price and abs(h[i] - price) < c[i] * close_tol_pct / 100:
            score += touch_normal if _in_up_trend(h, l, price, i, trend_pct) else 0
        elif l[i] >= price and abs(l[i] - price) < c[i] * close_tol_pct / 100:
            score += touch_normal if _in_down_trend(h, l, price, i, trend_pct) else 0
    return score


def _in_up_trend(h, l, price, i, trend_pct):
    for pos in range(i, -1, -1):
        if h[pos] > price:
            return False
        if price - l[pos] > price * trend_pct / 100:
            return True
    return False


def _in_down_trend(h, l, price, i, trend_pct):
    for pos in range(i, -1, -1):
        if l[pos] < price:
            return False
        if l[pos] - price > price * trend_pct / 100:
            return True
    return False


def b_levels(df, delta, merge_percent, min_bars, selector="median"):
    close = df["close"].to_numpy(float)
    labels = peak_valley_pivots(close, delta, delta)
    peaks = filter_by_bars_between(list(np.where(labels == 1)[0]), min_bars)
    valleys = filter_by_bars_between(list(np.where(labels == -1)[0]), min_bars)
    dist = merge_percent * float(np.mean(close)) / 100
    out = []
    for idxs, kind in ((peaks, "resistance"), (valleys, "support")):
        if not idxs:
            continue
        for cl in ward_cluster([float(close[i]) for i in idxs], dist):
            price = float(np.median(cl)) if selector == "median" else float(np.mean(cl))
            out.append(dict(kind=kind, price=price, peaks=len(cl),
                            first=min(idxs), last=max(idxs),
                            score=touch_score(df, price)))
    return sorted(out, key=lambda x: x["price"]), dist, len(peaks) + len(valleys)


# ---------- 画图 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--kind", default="auto", choices=["auto", "stock", "etf"])
    ap.add_argument("--since", default=None)
    ap.add_argument("--delta", type=float, default=0.05, help="B版ZigZag固定%阈值")
    ap.add_argument("--merge-percent", type=float, default=1.0, help="B版聚类距离（占均价%）")
    ap.add_argument("--min-bars", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_symbol(args.code, None if args.kind == "auto" else args.kind)
    if df is None:
        print("无数据")
        return
    cfg = StrategyConfig.load()
    res = run_analysis(df, cfg, code=args.code)
    feats = res.features
    if args.since:
        feats = feats[feats["date"] >= args.since]
    idx = feats.index.to_numpy()
    n = len(idx)
    x = np.arange(n)
    o, h, l, c = (feats[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sub = df.iloc[idx]

    lv_b, dist, npiv = b_levels(sub, args.delta, args.merge_percent, args.min_bars)
    sup_a = [z for z in res.levels if z.kind == "support" and z.created_idx <= idx[-1]]
    res_a = [z for z in res.levels if z.kind == "resistance" and z.created_idx <= idx[-1]]

    font = pick_cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    for ax in axes:
        up = c >= o
        ax.vlines(x, l, h, color=np.where(up, "#c0392b", "#27ae60"), linewidth=0.7)
        ax.bar(x[up], np.maximum((c - o)[up], 1e-6), 0.7, bottom=o[up], color="#c0392b")
        ax.bar(x[~up], np.maximum((o - c)[~up], 1e-6), 0.7, bottom=c[~up], color="#27ae60")
        ax.grid(alpha=0.2)

    # 上：A版（因果支撑带/阻力带）
    ax = axes[0]
    for i, z in enumerate(sup_a + res_a):
        col = PALETTE[i % len(PALETTE)]
        i0 = max(0, int(np.searchsorted(idx, z.created_idx)))
        i1 = n - 1 if z.broken_idx is None else min(n - 1, int(np.searchsorted(idx, z.broken_idx)))
        if i1 <= i0:
            continue
        ax.fill_between(x[i0:i1 + 1], z.bottom, z.top, color=col, alpha=0.18, zorder=0)
        ax.hlines([z.bottom, z.top], i0, i1, color=col, linewidth=0.8, alpha=0.8)
        kind = "支" if z.kind == "support" else "阻"
        ax.text(i1 + n * 0.004, (z.bottom + z.top) / 2,
                f"{kind}{i+1} {z.bottom:.3f}~{z.top:.3f}(×{z.touches})"
                + ("破" if z.broken_idx and z.broken_idx <= idx[-1] else ""),
                fontsize=7, color=col, va="center")
    thr = f"自适应3×ATR%[{cfg.zigzag.min_threshold:.0%}~{cfg.zigzag.max_threshold:.0%}]"
    ax.set_title(f"A版（本项目）：{thr}/depth{cfg.zigzag.depth} 高低点拐点 → 蜡烛区间带 → "
                 f"相邻单向合并(≤{cfg.zigzag.zone_max_width:.0%}) → 收盘破位失效   【因果】  "
                 f"支撑{len(sup_a)}条/阻力{len(res_a)}条", fontsize=11)

    # 下：B版（day0market 风格水平线）
    ax = axes[1]
    for i, z in enumerate(lv_b):
        col = PALETTE[i % len(PALETTE)]
        ax.hlines(z["price"], 0, n - 1, color=col, linewidth=1.4, alpha=0.85)
        ax.text(n * 1.004, z["price"],
                f"{'阻' if z['kind']=='resistance' else '支'}{i+1} {z['price']:.3f} "
                f"[峰{z['peaks']} 分{z['score']:+.0f}]", fontsize=7.5, color=col, va="center")
    ax.set_title(f"B版（照 day0market/support_resistance）：收盘价ZigZag(delta={args.delta:.0%}) → "
                 f"Ward聚类(距离阈值={args.merge_percent}%均价={dist:.4f}) → 中位数=线 → TouchScorer   "
                 f"【非因果，拐点须等反转走完】  共{npiv}个峰谷 → {len(lv_b)}条线", fontsize=11)

    step = max(1, n // 20)
    dts = pd.to_datetime(pd.Series(feats["date"].to_numpy()[::step]))
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels([dd.strftime("%Y-%m-%d") for dd in dts], rotation=45, fontsize=8)
    for a in axes:
        a.set_xlim(-1, n * 1.14)
    out = args.out or str(DATA_DIR / f"{args.code}_A_vs_B.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("保存:", out)

    print(f"\nA版：支撑{len(sup_a)} + 阻力{len(res_a)} 条带，宽度 "
          f"{min((z.top-z.bottom)/((z.top+z.bottom)/2) for z in sup_a)*100:.1f}%"
          f"~{max((z.top-z.bottom)/((z.top+z.bottom)/2) for z in sup_a)*100:.1f}%（支撑）")
    print(f"B版：{len(lv_b)} 条线（峰谷{npiv}个），线宽=0，打分范围 "
          f"{min(z['score'] for z in lv_b):+.0f}~{max(z['score'] for z in lv_b):+.0f}")
    print("B版明细：")
    for z in lv_b:
        print(f"  {z['kind']:10s} {z['price']:.3f} 峰数={z['peaks']:2d} 打分={z['score']:+5.0f} "
              f"首末峰={sub['date'].iloc[z['first']].date()}~{sub['date'].iloc[z['last']].date()}")


if __name__ == "__main__":
    main()
