"""参数网格搜索：验证策略可调性（每组参数→批量回测→汇总指标）。"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import BACKTEST_DIR
from .runner import run_batch, summarize
from datetime import datetime


def grid_search(param_grid: Dict[str, List], codes: List[Tuple[str, str]],
                workers: int = 8, start: Optional[str] = None, end: Optional[str] = None,
                tag: str = "grid", progress_cb=None) -> pd.DataFrame:
    keys = list(param_grid)
    rows = []
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    for ci, values in enumerate(combos, 1):
        cfg_dict = dict(zip(keys, values))
        results = run_batch(codes, cfg_dict, workers=workers, start=start, end=end)
        summary = summarize(results)
        rows.append({**cfg_dict, **summary, "combo": ci})
        if progress_cb:
            progress_cb(ci, len(combos), cfg_dict, summary)
    df = pd.DataFrame(rows).sort_values("avg_total_ret", ascending=False, na_position="last")
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = BACKTEST_DIR / f"{tag}_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    df.attrs["path"] = str(path)
    return df
