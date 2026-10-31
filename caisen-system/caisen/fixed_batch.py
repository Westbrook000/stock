"""Bounded-memory batch runner and self-contained HTML report for fixed replay."""

from __future__ import annotations

import html
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .fixed_backtest import (
    STARTING_CASH,
    STRATEGY_VERSION,
    SUPPORTED_STRATEGY_VERSIONS,
    calculate_metrics,
    run_fixed_backtest,
)


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("not JSON serializable: %r" % (value,))


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


def _html_report(payload):
    doc = json.dumps(payload, ensure_ascii=False, default=_json_default).replace("</", "<\\/")
    title = "固定止盈止损形态回测报告"
    context = payload.get("report_context") or {}
    note = html.escape(context.get("note", "回测仅反映所选本地行情区间的执行结果。形态效果仍需独立样本验证。"))
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:15px/1.55 system-ui,-apple-system,sans-serif;margin:28px auto;padding:0 18px;max-width:1200px;color:#182230;background:#f7f8fa}}h1{{font-size:25px}}.note{{background:#fff4d4;padding:14px;border-radius:8px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.card,section{{background:white;border:1px solid #e2e6eb;border-radius:8px;padding:14px;margin:12px 0}}.card b{{display:block;font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #e7eaee}}.scroll{{overflow-x:auto;max-width:100%}}.account-table{{min-width:1800px;white-space:nowrap}}small{{color:#586575}}
</style><main><h1>{title}</h1><p class="note">{note} 每标的、每周期独立账户；账户收益率不可相加。留出段历史已暴露于查看，不代表未触碰样本。</p><div id="fixed-backtest-root"></div></main>
<script type="application/json" id="fixed-backtest-data">{payload}</script><script src="/static/fixed-backtest-report.js?v=20261029v3"></script></html>""".format(title=html.escape(title), note=note, payload=doc)


def run_fixed_batch(items, output_dir, development_cutoff=None, max_workers=1,
                    progress_callback=None, report_context=None,
                    strategy_version=STRATEGY_VERSION):
    """Run isolated accounts serially; progress callback receives small snapshots.

    ``max_workers`` is accepted by the API contract for future bounded work,
    but this initial runner is intentionally serial to avoid exhausting local
    CPU or memory while every daily prefix is recognized causally.
    """
    _ = max_workers
    if strategy_version not in SUPPORTED_STRATEGY_VERSIONS:
        raise ValueError("unsupported fixed strategy version: %s" % strategy_version)
    prepared = list(items)
    if development_cutoff is None:
        all_dates = sorted({pd.Timestamp(value).normalize() for item in prepared
                            for value in pd.to_datetime(item["bars"].get("date", []), errors="coerce").dropna()})
        if len(all_dates) > 1:
            import math
            development_cutoff = all_dates[min(len(all_dates) - 1, math.floor(len(all_dates) * 0.7))].date().isoformat()
    root_dir = Path(output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root_dir / ("fixed-backtest-%s-%s" % (stamp, uuid4().hex[:6]))
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    errors = []
    total = len(prepared)
    for index, item in enumerate(prepared, 1):
        result = None
        try:
            result = run_fixed_backtest(item["bars"], item["code"], item["kind"],
                                        item.get("timeframe", "1d"),
                                        cash=float(item.get("cash", 100000)),
                                        calendar=item.get("calendar"),
                                        development_cutoff=development_cutoff,
                                        trade_start=item.get("trade_start"),
                                        strategy_version=strategy_version)
            results.append(result)
        except Exception as exc:  # preserve completed accounts and their replay ledger
            errors.append({"code": str(item.get("code")), "kind": str(item.get("kind")),
                           "timeframe": str(item.get("timeframe", "1d")),
                           "error": "%s: %s" % (type(exc).__name__, exc)})
        payload = summarize_batch(results)
        payload.update({"strategy_version": strategy_version, "development_cutoff": development_cutoff,
                        "batch_execution": {"account_count": total, "completed_count": len(results),
                                            "worker_count": 1, "requested_worker_count": int(max_workers),
                                            "status": "partial_failed" if errors else "running",
                                            "errors": errors}})
        if report_context:
            payload["report_context"] = dict(report_context)
        _write_artifacts(directory, payload)
        if progress_callback:
            progress_callback({"completed": index, "total": total,
                               "current": str(item.get("code")),
                               "timeframe": str(item.get("timeframe", "1d")),
                               "trade_count": result["metrics"]["trade_count"] if result else 0,
                               "status": "partial_failed" if errors else "running"})
        result = None
    payload = summarize_batch(results)
    payload.update({"strategy_version": strategy_version, "development_cutoff": development_cutoff,
                    "batch_execution": {"account_count": total, "completed_count": len(results),
                                        "worker_count": 1, "requested_worker_count": int(max_workers),
                                        "status": "partial_failed" if errors else "complete", "errors": errors}})
    if report_context:
        payload["report_context"] = dict(report_context)
    _write_artifacts(directory, payload)
    return payload


def _write_artifacts(directory, payload):
    json_path = directory / "fixed-backtest-results.json"
    html_path = directory / "fixed-backtest-report.html"
    script_path = directory / "fixed-backtest-report.js"
    payload["output"] = {"json": str(json_path.resolve()), "html": str(html_path.resolve()),
                         "javascript": str(script_path.resolve())}
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    _atomic_write(html_path, _html_report(payload))
    if not script_path.exists():
        shutil.copyfile(Path(__file__).parent / "static" / "fixed-backtest-report.js", script_path)


def _atomic_write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
