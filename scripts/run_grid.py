#!/usr/bin/env python3
"""参数网格：验证策略可调性。

用法:
    python3 scripts/run_grid.py --sample 212 --workers 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff.backtest.optimizer import grid_search      # noqa: E402
from wyckoff.scanner import build_universe              # noqa: E402

GRID = {
    "buy_threshold": [1.4, 1.8, 2.2],
    "recover_window": [2, 3, 5],
    "stop_loss": [0.05, 0.08],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=212)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--since", default=None)
    p.add_argument("--tag", default="tunability")
    args = p.parse_args()

    codes = build_universe(("stock", "etf"), sample=args.sample)
    print(f"网格 {len(GRID['buy_threshold']) * len(GRID['recover_window']) * len(GRID['stop_loss'])} 组"
          f" × {len(codes)} 标的")

    def cb(ci, total, cfg, summary):
        print(f"[{ci}/{total}] {cfg} -> win={summary.get('avg_win_rate')} "
              f"PF={summary.get('avg_profit_factor')} ret={summary.get('avg_total_ret')} "
              f"trades={summary.get('total_trades')}")

    df = grid_search(GRID, codes, workers=args.workers, start=args.since,
                     tag=args.tag, progress_cb=cb)
    cols = ["buy_threshold", "recover_window", "stop_loss", "total_trades",
            "avg_win_rate", "avg_profit_factor", "avg_total_ret", "median_total_ret",
            "avg_max_dd", "avg_trades_per_year"]
    print()
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
