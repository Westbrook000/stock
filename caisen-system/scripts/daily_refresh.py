"""Refresh every locally stored stock and ETF history file."""

from __future__ import annotations

import sys
import time
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parent
sys.path.insert(0, str(SYSTEM_ROOT))

from caisen.data_refresh import refresh_one  # noqa: E402
from caisen.market import discover_instruments  # noqa: E402


def main() -> int:
    instruments = discover_instruments(REPO_ROOT)
    total = len(instruments)
    successes = 0
    failures = []
    print(f"开始更新本地全部行情：{total} 个标的", flush=True)
    for index, instrument in enumerate(instruments, start=1):
        try:
            result = refresh_one(REPO_ROOT, instrument.code, instrument.kind)
            successes += 1
            print(
                f"[{index}/{total}] 成功 {instrument.kind} {instrument.code} "
                f"{result['first_date']} 至 {result['last_date']}",
                flush=True,
            )
        except Exception as error:
            failures.append((instrument, str(error)))
            print(
                f"[{index}/{total}] 失败 {instrument.kind} {instrument.code}: {error}",
                flush=True,
            )
        if index < total:
            time.sleep(0.25)

    print(f"更新结束：成功 {successes}，失败 {len(failures)}，总计 {total}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
