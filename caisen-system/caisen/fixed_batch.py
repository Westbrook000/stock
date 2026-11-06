"""Read-only aggregation helpers for saved historical backtest reports."""
from collections import defaultdict
import pandas as pd
from .fixed_backtest import STARTING_CASH, calculate_metrics


def summarize_batch(results):
    groups = defaultdict(list)
    trades = []
    signal_patterns = {}
    overlap_count = 0
    for result in results:
        for trade in result.get("trades", []):
            tagged = dict(trade, code=result["code"], kind=result["kind"], timeframe=result["timeframe"])
            trades.append(tagged)
            tags = trade.get("pattern_tags") or [
                {"pattern_id": pattern_id, "pattern_type": pattern_type}
                for pattern_id, pattern_type in zip(trade.get("pattern_ids", []), trade.get("pattern_types", []))]
            for tag in tags:
                pattern_id, pattern_type = tag["pattern_id"], tag["pattern_type"]
                key = (result["code"], result["kind"], result["timeframe"], pattern_id)
                signal_patterns[key] = pattern_type
        overlap_count += sum(1 for signal in result.get("signals", []) if len(signal.get("pattern_ids", [])) > 1)
        for trade in result.get("trades", []):
            shape_types = trade.get("pattern_types", []) or ["unknown"]
            for shape_type in shape_types:
                groups[("shape", str(shape_type))].append(trade)
            if len(shape_types) > 1:
                groups[("shape", "multi_shape")].append(trade)
            for dimension, value in (("timeframe", result["timeframe"]), ("kind", result["kind"]),
                                     ("year", pd.Timestamp(trade["entry_date"]).year)):
                groups[(dimension, str(value))].append(trade)
    grouped = []
    for (dimension, value), rows in sorted(groups.items()):
        pnls = [float(row["pnl"]) for row in rows]
        wins = sum(pnl > 0 for pnl in pnls)
        gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
        grouped.append({"dimension": dimension, "value": value, "trade_count": len(rows),
                        "win_rate": wins / len(rows) if rows else None,
                        "net_pnl": sum(pnls),
                        "profit_factor": sum(pnl for pnl in pnls if pnl > 0) / gross_loss if gross_loss else None})
    pattern_rows = [{"code": key[0], "kind": key[1], "timeframe": key[2], "pattern_id": key[3],
                     "pattern_type": value} for key, value in sorted(signal_patterns.items())]
    portfolio = _equal_initial_cash_portfolio(results)
    return {
        "summary": {"security_timeframe_accounts": len(results), "trade_count": len(trades),
                    "signal_count": sum(int(result.get("signal_count", 0)) for result in results),
                    "distinct_traded_patterns": len(pattern_rows), "overlapping_multi_pattern_signals": overlap_count,
                    "note": "账户收益率不直接求和或简单平均；组合收益按每账户相同初始资金计算，零交易账户仍纳入资金分母。" ,
                    "equal_initial_cash_portfolio": portfolio},
        "groups": grouped,
        "patterns": pattern_rows,
        "trades": trades,
        "results": results,
    }


def _equal_initial_cash_portfolio(results):
    account_count = len(results)
    starting_capital = sum(float(result.get("starting_cash", STARTING_CASH)) for result in results)
    ending_equity = sum(float(result.get("ending_equity", result.get("starting_cash", STARTING_CASH)))
                        for result in results)
    timelines = []
    dates = set()
    for result in results:
        points = sorted(result.get("daily_equity", []), key=lambda row: row["date"])
        timelines.append(points)
        dates.update(point["date"] for point in points)
    portfolio_curve = []
    ordered_dates = sorted(dates)
    for day in ordered_dates:
        total_equity = 0.0
        invested_value = 0.0
        for result, points in zip(results, timelines):
            initial = float(result.get("starting_cash", STARTING_CASH))
            eligible = [point for point in points if point["date"] <= day]
            if eligible:
                last = eligible[-1]
                equity = float(last["equity"])
                invested_value += float(last.get("invested_value", 0.0))
            else:
                equity = initial
            total_equity += equity
        portfolio_curve.append({"date": day, "equity": total_equity,
                                "invested_value": invested_value})
    all_trades = [trade for result in results for trade in result.get("trades", [])]
    metrics = calculate_metrics(all_trades, starting_capital, ending_equity, portfolio_curve)
    return {"account_count": account_count, "starting_capital": starting_capital,
            "ending_equity": ending_equity, "net_pnl": ending_equity - starting_capital,
            "metrics": metrics, "daily_equity": portfolio_curve,
            "method": "equal_initial_cash_accounts_aligned_on_union_dates",
            "zero_trade_accounts_included_in_capital_denominator": True,
            "evaluation_start_dates": sorted({str(result.get("evaluation_start")) for result in results
                                              if result.get("evaluation_start")})}
