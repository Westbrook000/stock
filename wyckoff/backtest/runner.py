"""回测执行：单标的/批量并行/指标汇总。"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import numpy as np
import pandas as pd

from ..config import BACKTEST_DIR, StrategyConfig
from ..data_store import load_frame
from .engine_bt import WyckoffStrategy, signals_to_feed


def run_symbol(code: str, kind: str = "stock", cfg_dict: Optional[Dict] = None,
               start: Optional[str] = None, end: Optional[str] = None) -> Optional[Dict]:
    cfg = StrategyConfig.load()
    if cfg_dict:
        cfg = cfg.with_overrides(**cfg_dict)
    df = load_frame(code, kind)
    if df is None or len(df) < 300:
        return None
    df = df[(df["close"] > 0) & (df["volume"] >= 0)].reset_index(drop=True)
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if len(df) < 200:
        return None

    from ..analysis.engine import run_analysis
    res = run_analysis(df, cfg)
    feed = signals_to_feed(res.features, res.signals)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(feed)
    cerebro.addstrategy(
        WyckoffStrategy,
        take_profit=cfg.trade.take_profit,
        stop_pct=cfg.trade.stop_loss,
        max_hold_days=cfg.trade.max_hold_days,
        use_signal_stop=True,
    )
    cerebro.broker.setcash(cfg.trade.initial_cash)
    cerebro.broker.setcommission(commission=cfg.trade.commission)
    strat = cerebro.run()[0]

    trades = strat.trades
    pairs = _pair_trades(trades)
    final_val = cerebro.broker.getvalue()
    ret = final_val / cfg.trade.initial_cash - 1.0
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    eq_dd = _max_drawdown_from_trades(pairs, cfg.trade.initial_cash)
    years = max(1.0, (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25)

    rets = [p["ret"] for p in pairs]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win = sum(wins) if wins else 0.0
    gross_loss = -sum(losses) if losses else 0.0
    return {
        "code": code, "kind": kind, "name": res.name,
        "bars": len(df), "years": round(years, 1),
        "n_trades": len(pairs),
        "win_rate": round(len(wins) / len(rets), 3) if rets else None,
        "avg_ret": round(float(np.mean(rets)), 4) if rets else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else None),
        "total_ret": round(float(ret), 4),
        "buy_hold": round(float(bh), 4),
        "max_dd": round(float(eq_dd), 4),
        "trades_per_year": round(len(pairs) / years, 2),
        "trade_list": pairs,
    }


def _pair_trades(trades: List[Dict]) -> List[Dict]:
    pairs, open_trade = [], None
    for t in trades:
        if t["direction"] == "buy":
            open_trade = t
        elif t["direction"] == "sell" and open_trade is not None:
            if open_trade["price"] <= 0:
                open_trade = None
                continue
            ret = t["price"] / open_trade["price"] - 1.0
            pairs.append({"entry": str(open_trade["dt"]), "exit": str(t["dt"]),
                          "entry_price": round(open_trade["price"], 3),
                          "exit_price": round(t["price"], 3),
                          "entry_type": open_trade.get("entry_type", ""),
                          "ret": round(ret, 4), "exit_reason": t.get("reason", "")})
            open_trade = None
    return pairs


def _max_drawdown_from_trades(pairs: List[Dict], cash: float) -> float:
    if not pairs:
        return 0.0
    equity = cash
    peak = cash
    mdd = 0.0
    for p in pairs:
        equity *= (1 + p["ret"])
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
    return mdd


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
        "avg_profit_factor": round(float(np.nanmean([r["profit_factor"] for r in ok
                                            if r["profit_factor"] not in (None, 99.0)] or [np.nan])), 2)
            if ok else None,
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
            except Exception:  # noqa: BLE001
                r = None
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
