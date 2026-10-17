"""全市场扫描器：并行运行策略引擎，输出信号榜单并缓存。"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import SIGNALS_DIR, StrategyConfig
from .data_store import list_symbols, load_frame


def _analyze_one(args: Tuple[str, str, Optional[Dict]]) -> Optional[Dict]:
    code, kind, cfg_dict = args
    try:
        df = load_frame(code, kind)
        if df is None or len(df) < 120:
            return None
        cfg = StrategyConfig.load()
        if cfg_dict:
            cfg = cfg.with_overrides(**cfg_dict)
        from .analysis.engine import run_analysis
        res = run_analysis(df, cfg)
        row = dict(res.latest)
        row.update({"code": code, "kind": kind, "name": res.name,
                    "bars": len(df)})
        return row
    except Exception as exc:  # noqa: BLE001
        return {"code": code, "kind": kind, "error": str(exc)[:80]}


def build_universe(kinds: Tuple[str, ...] = ("stock", "etf"),
                   sample: int = 0) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k in kinds:
        codes = list_symbols(k)
        if sample > 0:
            stride = max(1, len(codes) // sample)
            codes = codes[::stride][:sample]
        out.extend((c, k) for c in codes)
    return out


def scan(kinds: Tuple[str, ...] = ("stock", "etf"), workers: int = 8,
         sample: int = 0, cfg_overrides: Optional[Dict] = None) -> List[Dict]:
    universe = build_universe(kinds, sample)
    total = len(universe)
    results: List[Dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_one, (c, k, cfg_overrides)): (c, k) for c, k in universe}
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            done += 1
            if row:
                results.append(row)
            if done % 200 == 0:
                print(f"  scan {done}/{total} ({time.time()-t0:.0f}s)", flush=True)
    return results


def save_scan(rows: List[Dict], tag: str = "") -> Path:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = SIGNALS_DIR / f"scan_{tag}{stamp}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (SIGNALS_DIR / "latest.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def load_latest_scan() -> Optional[List[Dict]]:
    p = SIGNALS_DIR / "latest.json"
    if not p.exists():
        return None
    from .analysis.engine import STRATEGY_VERSION
    rows = json.loads(p.read_text(encoding="utf-8"))
    current = [r for r in rows if r.get('strategy_version') == STRATEGY_VERSION]
    return current or None
