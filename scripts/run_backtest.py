#!/usr/bin/env python3
"""批量回测。

用法:
    python3 scripts/run_backtest.py --sample 100 --since 2015-01-01
    python3 scripts/run_backtest.py --code 600108 --kind stock
    python3 scripts/run_backtest.py --all-etf --set buy_threshold=1.6 stop_loss=0.07
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff.backtest.runner import run_batch, run_symbol, save_report, summarize  # noqa: E402
from wyckoff.scanner import build_universe  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", help="单标的回测")
    p.add_argument("--kind", default="stock", choices=["stock", "etf"])
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--all", action="store_true")
    p.add_argument("--all-etf", action="store_true")
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--set", nargs="*", default=[], help="扁平参数覆盖 key=value")
    p.add_argument("--tag", default="manual")
    args = p.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = float(v) if any(c in v for c in ".e") else int(v)
        except ValueError:
            overrides[k] = v

    if args.code:
        r = run_symbol(args.code, args.kind, overrides or None, args.since, args.until)
        if not r:
            print("无数据或样本过短")
            return
        trades = r.pop("trade_list")
        print(f"{args.code} 交易{r['n_trades']}笔 胜率{r['win_rate']} PF{r['profit_factor']} "
              f"总收益{r['total_ret']:.1%} vs 买入持有{r['buy_hold']:.1%} 最大回撤{r['max_dd']:.1%}")
        for t in trades[:15]:
            print(f"  {t['entry']}→{t['exit']} {t['ret']:+.2%} ({t['exit_reason']})")
        return

    if args.all_etf:
        kinds = ("etf",)
    elif args.all:
        kinds = ("stock", "etf")
    else:
        kinds = ("stock", "etf")
    codes = build_universe(kinds, sample=0 if args.all or args.all_etf else (args.sample or 100))
    t0 = time.time()
    rows = run_batch(codes, overrides or None, workers=args.workers,
                     start=args.since, end=args.until,
                     progress_cb=lambda d, tt: print(f"  {d}/{tt} {time.time()-t0:.0f}s", flush=True))
    s = summarize(rows)
    path = save_report(rows, s, args.tag)
    print("汇总:", s)
    print("报告:", path)


if __name__ == "__main__":
    main()
