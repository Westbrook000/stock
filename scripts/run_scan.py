#!/usr/bin/env python3
"""全市场信号扫描。

用法:
    python3 scripts/run_scan.py --sample 60       # 抽样验证
    python3 scripts/run_scan.py --all             # 全量（股票+ETF）
    python3 scripts/run_scan.py --all --set merge_percent=0.5   # 带参数覆盖
"""
import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff.scanner import save_scan, scan  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--kind", default="both", choices=["stock", "etf", "both"])
    p.add_argument("--set", nargs="*", default=[], help="key=value 参数覆盖(扁平)")
    args = p.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)

    kinds = ("stock", "etf") if args.kind == "both" else (args.kind,)
    sample = 0 if args.all else (args.sample or 50)
    rows = scan(kinds=kinds, workers=args.workers, sample=sample, cfg_overrides=overrides or None)
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    path = save_scan(ok)
    from collections import Counter
    print(f"标的数 {len(ok)} 错误 {len(errs)}")
    print("action分布:", Counter(r["action"] for r in ok))
    buys = sorted((r for r in ok if r["action"] == "BUY"), key=lambda r: -r["buy_score"])
    print(f"BUY榜单(前10): 共{len(buys)}")
    for r in buys[:10]:
        print(f"  {r['code']} {r['name'][:8]:10s} {r['kind']:5s} score={r['buy_score']:4.1f} "
              f"{r['entry_type']:12s} {r['reason'][:40]}")
    sells = [r for r in ok if r["action"] == "SELL"]
    print(f"SELL数: {len(sells)}")
    print("保存:", path)


if __name__ == "__main__":
    main()
