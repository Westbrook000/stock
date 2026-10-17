#!/usr/bin/env python3
"""行情数据增量更新：A股 + ETF。

用法: python3 scripts/update_data.py [--what stocks|etfs|both] [--workers 10]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff.downloader import update_all_stocks  # noqa: E402


def update_etfs_task(workers: int):
    """ETF 增量=重新拉全量再合并（新浪接口特性），只更新本地已有的。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from wyckoff.data_store import list_symbols
    from wyckoff.downloader import fetch_etf_list, update_etf
    rows = fetch_etf_list()
    have = set(list_symbols("etf"))
    todo = [r for r in rows.itertuples() if r.code in have]
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(update_etf, r.code, r.name, r.market): r.code for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            ok += res["status"] == "ok"
            if i % 100 == 0:
                print(f"  etf {i}/{len(todo)} ok={ok}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", default="both", choices=["stocks", "etfs", "both"])
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    if args.what in ("stocks", "both"):
        t0 = time.time()

        def cb(i, total, res):
            if i % 300 == 0:
                print(f"  stocks {i}/{total} ({time.time()-t0:.0f}s)", flush=True)
        res = update_all_stocks(workers=args.workers, progress_cb=cb)
        ok = sum(1 for r in res if r["status"] in ("ok", "fresh"))
        err = [r for r in res if "error" in str(r["status"])]
        print(f"股票更新完成: 有效{ok}/{len(res)} 错误{len(err)} 用时{time.time()-t0:.0f}s")

    if args.what in ("etfs", "both"):
        t0 = time.time()
        ok = update_etfs_task(args.workers)
        print(f"ETF更新完成: ok={ok} 用时{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
