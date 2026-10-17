"""回测执行：单标的/批量并行/指标汇总。"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import BACKTEST_DIR, StrategyConfig
from ..data_store import load_frame
from .daily import simulate_daily


def run_symbol(code: str, kind: str = "stock", cfg_dict: Optional[Dict] = None,
               start: Optional[str] = None, end: Optional[str] = None) -> Optional[Dict]:
    cfg = StrategyConfig.load()
    if cfg_dict:
        cfg = cfg.with_overrides(**cfg_dict)
    df = load_frame(code, kind)
    if df is None or len(df) < 300:
        return None
    df = df[(df["close"] > 0) & (df["volume"] >= 0)].reset_index(drop=True)
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if start:
        first = int(df["date"].searchsorted(pd.Timestamp(start)))
        df = df.iloc[max(0, first - 600):].reset_index(drop=True)
    eligible = df[df["date"] >= pd.Timestamp(start)] if start else df
    if len(eligible) < 200:
        return None

    from ..analysis.engine import run_analysis
    res = run_analysis(df, cfg)
    metrics = simulate_daily(res.features, res.signals, cfg.trade, start=start, end=end)
    years = max(1 / 365.25, (eligible["date"].iloc[-1] - eligible["date"].iloc[0]).days / 365.25)
    return {"code": code, "kind": kind, "name": res.name, "years": round(years, 2),
            **metrics, "trades_per_year": metrics["n_trades"] / years}


def summarize(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if r and r["n_trades"] > 0]
    all_rows = [r for r in rows if r]
    if not all_rows:
        return {}
    rets = np.array([r["total_ret"] for r in all_rows])
    wins = np.array([r["win_rate"] for r in ok if r["win_rate"] is not None])
    return {
        "symbols": len(all_rows),
        "symbols_with_trades": len(ok),
        "total_trades": int(sum(r["n_trades"] for r in all_rows)),
        "avg_win_rate": round(float(np.mean(wins)), 3) if len(wins) else None,
        "avg_profit_factor": round(float(np.mean([r["profit_factor"] for r in ok
                    if r["profit_factor"] is not None and np.isfinite(r["profit_factor"])])), 3)
                    if any(r["profit_factor"] is not None and np.isfinite(r["profit_factor"]) for r in ok) else None,
        "net_pnl": sum(r.get("net_pnl", 0.0) for r in all_rows),
        "profit_factor": (sum(r.get("gross_profit", 0.0) for r in all_rows) /
                          sum(r.get("gross_loss", 0.0) for r in all_rows))
                         if sum(r.get("gross_loss", 0.0) for r in all_rows) > 0 else None,
        "avg_total_ret": round(float(np.mean(rets)), 4),
        "median_total_ret": round(float(np.median(rets)), 4),
        "avg_max_dd": round(float(np.mean([r["max_dd"] for r in ok])), 4) if ok else None,
        "avg_trades_per_year": round(float(np.mean([r["trades_per_year"] for r in ok])), 2) if ok else None,
    }


def run_batch(codes: List[Tuple[str, str]], cfg_dict: Optional[Dict] = None,
              workers: int = 8, start: Optional[str] = None, end: Optional[str] = None,
              progress_cb=None) -> List[Dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results: List[Dict] = []
    total = len(codes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_symbol, c, k, cfg_dict, start, end): (c, k) for c, k in codes}
        done = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                code, kind = futures[fut]
                raise RuntimeError(f"回测失败 {kind}:{code}") from exc
            results.append(r)
            done += 1
            if progress_cb and done % 100 == 0:
                progress_cb(done, total)
    return results


def save_report(rows: List[Dict], summary: Dict, tag: str) -> str:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = BACKTEST_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "trade_list"} for r in rows if r])
    path = d / f"report_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    trades = []
    for r in rows or []:
        if not r:
            continue
        for t in r["trade_list"]:
            trades.append({"code": r["code"], **t})
    if trades:
        pd.DataFrame(trades).to_csv(d / f"trades_{stamp}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(d / f"summary_{stamp}.csv", index=False, encoding="utf-8-sig")
    return str(path)
