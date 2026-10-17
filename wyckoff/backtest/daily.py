"""日线撮合：收盘信号次日开盘执行，价格跳空按开盘成交，持仓按日盯市。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_daily(frame, signals):
    """提取数组供多组参数复用，不缓存可变持仓状态。"""
    return {**{k: frame[k].to_numpy() for k in ("date", "open", "high", "low", "close", "volume")},
            **{k: signals[k].to_numpy() for k in ("action", "stop", "entry_type")}}


def simulate_daily(frame, signals, cfg, start=None, end=None, include_equity=False):
    data = frame if isinstance(frame, dict) else prepare_daily(frame, signals)
    dates = np.asarray(data["date"], dtype="datetime64[ns]")
    first = int(np.searchsorted(dates, np.datetime64(start))) if start else 0
    last = int(np.searchsorted(dates, np.datetime64(end), side="right")) if end else len(dates)
    cash = float(cfg.initial_cash)
    quantity, entry_idx = 0, -1
    entry_price = entry_cost = stop = target = 0.0
    entry_type = pending = ""
    trades, equity = [], []
    commission = float(cfg.commission)
    slip = float(getattr(cfg, "slippage", 0.0))
    peak, max_dd = cash, 0.0

    def sell(i, price, reason):
        nonlocal cash, quantity, pending
        fill = float(price) * (1 - slip)
        proceeds = quantity * fill * (1 - commission)
        pnl = proceeds - entry_cost
        cash += proceeds
        trades.append({"entry": str(dates[entry_idx])[:10], "exit": str(dates[i])[:10],
                       "entry_price": entry_price, "exit_price": fill, "quantity": quantity,
                       "entry_type": entry_type, "ret": pnl / entry_cost, "pnl": pnl,
                       "fees": quantity * entry_price * commission + quantity * fill * commission,
                       "exit_reason": reason})
        quantity, pending = 0, ""

    for i in range(first, last):
        o, h, l, c = (float(data[k][i]) for k in ("open", "high", "low", "close"))
        tradable = data["volume"][i] > 0 and o > 0 and np.isfinite(o + h + l + c)
        sold_today = False
        if quantity and i > entry_idx and tradable:
            if o <= stop:
                sell(i, o, "stop_gap")
            elif target and o >= target:
                sell(i, o, "take_profit_gap")
            elif pending:
                sell(i, o, pending)
            elif l <= stop:
                # 无法得知日内先后；同日触及止盈止损时采取止损优先的保守假设。
                sell(i, stop, "stop")
            elif target and h >= target:
                sell(i, target, "take_profit")
            sold_today = not quantity

        if (not quantity and not sold_today and i > first and i < last - 1
                and data["action"][i - 1] == "BUY" and tradable):
            fill = o * (1 + slip)
            ref = float(data["stop"][i - 1])
            if not np.isfinite(ref) or ref <= 0:
                ref = fill * (1 - cfg.stop_loss)
            # 已跳空跌过结构止损时不买；stop_loss是最大风险距离，不放宽较近的结构止损。
            if ref < fill:
                quantity = int((cash * 0.99) / (fill * (1 + commission)) / 100) * 100
                if quantity:
                    entry_idx, entry_price = i, fill
                    entry_cost = quantity * fill * (1 + commission)
                    cash -= entry_cost
                    stop = max(ref, fill * (1 - cfg.stop_loss))
                    target = fill * (1 + cfg.take_profit) if cfg.take_profit else 0.0
                    entry_type = str(data["entry_type"][i - 1])
                    pending = ""

        if quantity:
            if data["action"][i] == "SELL":
                pending = "signal"
            elif cfg.max_hold_days and i - entry_idx >= cfg.max_hold_days:
                pending = "timeout"
        # 每段回测末尾统一平仓，成本计入该段；终日不新开仓。
        if i == last - 1 and quantity and tradable:
            sell(i, c, "period_end")
        value = cash + quantity * c
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1)
        equity.append(value)

    wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    losses = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    result = {"n_trades": len(trades), "win_rate": sum(t["pnl"] > 0 for t in trades) / len(trades) if trades else None,
              "avg_ret": float(np.mean([t["ret"] for t in trades])) if trades else None,
              "profit_factor": wins / losses if losses else (float("inf") if wins else None),
              "gross_profit": wins, "gross_loss": losses, "net_pnl": (equity[-1] if equity else cash) - cfg.initial_cash,
              "total_ret": (equity[-1] if equity else cash) / cfg.initial_cash - 1,
              "max_dd": max_dd, "trade_list": trades, "bars": last - first,
              "buy_hold": float(data["close"][last - 1] / data["close"][first] - 1) if last > first else 0.0}
    if include_equity:
        result["equity"] = equity
        result["dates"] = [str(x)[:10] for x in dates[first:last]]
    return result
