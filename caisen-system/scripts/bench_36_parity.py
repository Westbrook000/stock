#!/usr/bin/env python3
"""Benchmark 36 pinned symbols (10-worker spawn) and compute exact canonical parity + 1,000-symbol ETA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from caisen.daily_research import UNIVERSE_ID, load_research_bars, run_daily_research  # noqa: E402
from caisen.market import discover_instruments  # noqa: E402

UNIVERSE_PATH = REPO / "research" / "universes" / (UNIVERSE_ID + ".json")
BENCH_PATH = REPO / "reports" / "daily-research" / "benchmark" / "benchmark-execution-v6-current-rules-36.json"


def _load_36_symbols():
    data = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    return [(item["kind"], item["code"]) for item in data["inputs"]]


def _canonical_digest(output_dir: Path):
    account_dir = output_dir / "accounts"
    records = []
    for path in sorted(account_dir.glob("*.json")):
        acc = json.loads(path.read_text(encoding="utf-8"))
        # Strip non-deterministic timings / provenance paths; keep exact business ledger + signals.
        signals_canonical = [
            {
                "signal_date": s.get("signal_date"),
                "entry_date": s.get("entry_date"),
                "pattern_ids": s.get("pattern_ids"),
                "pattern_types": s.get("pattern_types"),
                "patterns": s.get("patterns"),
            }
            for s in acc.get("signals", [])
        ]
        records.append({
            "kind": acc.get("kind"),
            "code": acc.get("code"),
            "status": acc.get("status"),
            "ending_equity": acc.get("ending_equity"),
            "signals": signals_canonical,
            "trades": acc.get("trades", []),
        })
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), records


def _universe_bar_stats(symbols_36):
    found = {(item.kind, item.code): item for item in discover_instruments(ROOT)}
    univ = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    all_1000 = [("stock", c) for c in univ["stocks"]] + [("etf", c) for c in univ["etfs"]]

    def count_rows(pairs):
        total_rows = 0
        total_sq = 0
        for kind, code in pairs:
            inst = found.get((kind, code))
            if inst is None:
                continue
            n = len(load_research_bars(inst, root=ROOT)["bars"])
            total_rows += n
            total_sq += n * n
        return total_rows, total_sq

    r36, sq36 = count_rows(symbols_36)
    r1000, sq1000 = count_rows(all_1000)
    return {"bars_36": r36, "bars_sq_36": sq36, "bars_1000": r1000, "bars_sq_1000": sq1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--compare-to", type=Path, default=None)
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)

    symbols = _load_36_symbols()
    stats = _universe_bar_stats(symbols)
    t0 = time.perf_counter()
    payload = run_daily_research(
        UNIVERSE_PATH,
        out,
        workers=args.workers,
        recognizer="boundary_first",
        execution_mode="spawn",
        root=ROOT,
        symbols=symbols,
    )
    wall = time.perf_counter() - t0
    digest, records = _canonical_digest(out)
    eta_linear_min = (wall * (stats["bars_1000"] / max(stats["bars_36"], 1))) / 60.0
    eta_sq_min = (wall * (stats["bars_sq_1000"] / max(stats["bars_sq_36"], 1))) / 60.0

    summary = {
        "output": str(out),
        "workers": args.workers,
        "wall_seconds": round(wall, 2),
        "timings": payload.get("timings"),
        "canonical_sha256": digest,
        "trade_count": payload["summary"]["trade_count"],
        "signal_count": payload["summary"]["signal_count"],
        "cumulative_return": payload["summary"]["cumulative_return"],
        "win_rate": payload["summary"]["win_rate"],
        "bar_stats": stats,
        "eta_1000_linear_minutes": round(eta_linear_min, 2),
        "eta_1000_bars_sq_minutes": round(eta_sq_min, 2),
    }

    if args.compare_to is not None:
        ref_digest, ref_records = _canonical_digest(args.compare_to)
        summary["parity_passed"] = (digest == ref_digest)
        summary["reference_sha256"] = ref_digest
        if not summary["parity_passed"]:
            diffs = []
            for r_ref, r_cur in zip(ref_records, records):
                if r_ref != r_cur:
                    diffs.append({"kind": r_cur["kind"], "code": r_cur["code"],
                                  "ref_signals": len(r_ref["signals"]), "cur_signals": len(r_cur["signals"]),
                                  "ref_trades": len(r_ref["trades"]), "cur_trades": len(r_cur["trades"]),
                                  "ref_eq": r_ref["ending_equity"], "cur_eq": r_cur["ending_equity"]})
            summary["first_diffs"] = diffs[:10]

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.compare_to is not None and not summary["parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
