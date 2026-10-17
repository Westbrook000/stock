#!/usr/bin/env python3
"""ETF 全量下载（可断点续传）。

用法:
    python3 scripts/download_etfs.py            # 下载全部，跳过已存在
    python3 scripts/download_etfs.py --force    # 全部重新下载
    python3 scripts/download_etfs.py --limit 50 --workers 8
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff.downloader import download_all_etfs, fetch_etf_list  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.limit > 0:
        from concurrent.futures import ThreadPoolExecutor
        from wyckoff.downloader import download_etf
        rows = list(fetch_etf_list().itertuples())[: args.limit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda r: download_etf(r.code, r.name, r.market), rows))
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"limit模式完成: 成功{ok}/{len(rows)}")
        for r in results:
            if r["status"] != "ok":
                print("  ", r)
        return

    t0 = time.time()
    counter = {"n": 0}

    def cb(i, total, res):
        if res["status"] != "ok":
            print(f"[{i}/{total}] {res['code']} {res['status']}", flush=True)
        if i % 100 == 0:
            print(f"[{i}/{total}] 用时{time.time()-t0:.0f}s", flush=True)
        counter["n"] = i

    results = download_all_etfs(workers=args.workers, force=args.force, progress_cb=cb)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"完成: 成功{ok} 失败{len(results)-ok} 总用时{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
