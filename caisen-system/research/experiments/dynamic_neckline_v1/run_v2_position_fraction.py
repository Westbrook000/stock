"""Replay V2 neckline signals with partial exits and bounded cash exposure."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import backtrader as bt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORKSPACE = ROOT.parent
SOURCE = HERE / "results/paired-100-v2-20261110-attempt01"
DATA = WORKSPACE / "data/stocks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from caisen.fixed_backtest import calculate_metrics  # noqa: E402
from replay_partial_profit_v17 import PartialProfitStrategy  # noqa: E402


class SizedPartialStrategy(PartialProfitStrategy):
    params = tuple(PartialProfitStrategy.params._getpairs().items()) + (
        ("allocation_fraction", 1.0),
    )

    def next_open(self):
        today = self.data.datetime.date(0).isoformat()
        if self.position.size:
            for signal in self.by_entry.get(today, []):
                self.rejections.append({
                    "signal_date": signal["signal_date"], "entry_date": today,
                    "pattern_ids": signal.get("pattern_ids", []),
                    "reason": "position_already_open",
                })
            return
        if self.pending:
            return
        for signal in self.by_entry.get(today, []):
            opening = float(self.data.open[0])
            lot = int(self.p.lot_size)
            fraction = max(0.0, min(1.0, float(self.p.allocation_fraction)))
            shares = int((self.broker.getcash() * fraction) // (opening * lot)) * lot
            if shares <= 0:
                self.rejections.append({
                    "signal_date": signal["signal_date"], "entry_date": today,
                    "pattern_ids": signal.get("pattern_ids", []),
                    "reason": "insufficient_cash",
                })
                continue
            order = self.buy(size=shares, exectype=bt.Order.Market)
            self.pending = order.ref
            self.meta[order.ref] = {"role": "entry", "signal": signal, "shares": shares}
            break


def replay(frame, signals, allocation_fraction, cash=100_000.0, lot_size=100):
    frame = frame.copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = (frame.sort_values("date", kind="mergesort")
             .drop_duplicates("date", keep="last").reset_index(drop=True))
    final_date = frame.date.iloc[-1].date().isoformat()
    sentinel = frame.iloc[-1].copy()
    sentinel[["open", "high", "low", "close"]] = float(frame.close.iloc[-1])
    sentinel["volume"] = 0.0
    frame = pd.concat([frame, pd.DataFrame([sentinel])], ignore_index=True)
    frame = frame.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    valid_dates = {pd.Timestamp(value).date().isoformat() for value in frame.index[:-1]}
    signals = [dict(row) for row in signals
               if row.get("entry_date") in valid_dates and row.get("entry_date") != final_date]

    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.addstrategy(
        SizedPartialStrategy, signals=signals, fraction=0.6,
        runner_breakeven=False, ratchet_trigger=None, ratchet_floor=None,
        allocation_fraction=float(allocation_fraction), lot_size=int(lot_size),
        last_date=final_date,
    )
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_coo(True)
    strategy = cerebro.run(runonce=False)[0]
    ending_equity = float(cerebro.broker.getcash())
    curve = (strategy.curve[:-1] if strategy.curve and strategy.curve[-1]["shares"] == 0
             else strategy.curve)
    checks = {
        "cash_plus_position_equals_equity": all(math.isclose(
            point["cash"] + point["invested_value"], point["equity"],
            rel_tol=1e-9, abs_tol=1e-5) for point in curve),
        "realized_pnl_matches_ending_equity": math.isclose(
            float(cash) + sum(trade["pnl"] for trade in strategy.trades), ending_equity,
            rel_tol=1e-9, abs_tol=1e-4),
        "closed_trade_positions": not strategy.position.size,
    }
    return {"trades": strategy.trades, "daily_equity": curve,
            "ending_equity": ending_equity, "accounting_checks": checks}


def summarize(rows):
    trades = [trade for account in rows for trade in account["trades"]]
    wins = [float(trade["pnl"]) for trade in trades if float(trade["pnl"]) > 0]
    losses = [-float(trade["pnl"]) for trade in trades if float(trade["pnl"]) < 0]
    all_dates = sorted({point["date"] for account in rows
                        for point in account["daily_equity"]})
    curves = []
    invested_curves = []
    for account in rows:
        by_date = {point["date"]: (float(point["equity"]),
                                    float(point["invested_value"]))
                   for point in account["daily_equity"]}
        equity = 100_000.0
        invested = 0.0
        curve, invested_curve = [], []
        for date in all_dates:
            if date in by_date:
                equity, invested = by_date[date]
            curve.append(equity)
            invested_curve.append(invested)
        curves.append(curve)
        invested_curves.append(invested_curve)
    portfolio_curve = [{
        "date": date,
        "equity": sum(curve[index] for curve in curves),
        "invested_value": sum(curve[index] for curve in invested_curves),
    } for index, date in enumerate(all_dates)]
    metrics = calculate_metrics(
        trades, len(rows) * 100_000.0,
        sum(account["ending_equity"] for account in rows), portfolio_curve,
    )
    metrics.update({
        "portfolio_maximum_drawdown": metrics["maximum_drawdown"],
        "accounts_positive": sum(account["metrics"]["cumulative_return"] > 0
                                  for account in rows),
        "median_account_return": sorted(account["metrics"]["cumulative_return"]
                                         for account in rows)[len(rows) // 2],
        "median_account_maximum_drawdown": sorted(
            account["metrics"]["maximum_drawdown"] for account in rows)[len(rows) // 2],
        "accounting_all_pass": all(all(account["accounting_checks"].values())
                                    for account in rows),
    })
    return metrics


def run(allocations, output):
    source_files = sorted(SOURCE.glob("stock-*.json.gz"))
    if len(source_files) != 100:
        raise RuntimeError(f"Expected 100 frozen accounts, found {len(source_files)}")
    records = {str(value): [] for value in allocations}
    started = time.perf_counter()
    for index, path in enumerate(source_files, 1):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        code = saved["code"]
        bars = pd.read_csv(DATA / f"{code}.csv")
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.loc[bars.date <= "2026-10-30"].reset_index(drop=True)
        for allocation in allocations:
            outcome = replay(bars, saved["signals_detail"], allocation)
            if not all(outcome["accounting_checks"].values()):
                raise AssertionError((code, allocation, outcome["accounting_checks"]))
            metrics = calculate_metrics(outcome["trades"], 100_000.0,
                                        outcome["ending_equity"], outcome["daily_equity"])
            records[str(allocation)].append({
                "code": code, "trades": outcome["trades"],
                "daily_equity": outcome["daily_equity"],
                "ending_equity": outcome["ending_equity"],
                "metrics": metrics, "accounting_checks": outcome["accounting_checks"],
            })
        if index % 10 == 0:
            print(f"{index}/100 accounts, elapsed={time.perf_counter()-started:.1f}s", flush=True)

    result = {
        "scope": "V2 saved V12 structures, paired 100 stocks, 2026-10-30 cutoff",
        "exit_rule": "60% at +8%, remaining shares at -8% stop or +20% target",
        "variants": {key: summarize(value) for key, value in records.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "accounts": records,
        "limitations": [
            "historical development sample, not an independent holdout",
            "signals are V2 research boundaries on saved V12 structure candidates",
            "position fraction changes exposure only; it does not change recognition",
            "zero fees and slippage",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_seconds": result["elapsed_seconds"],
                      "variants": result["variants"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocations", nargs="+", type=float, default=[0.45])
    parser.add_argument("--output", type=Path, default=SOURCE / "v2-position-fraction-sweep.json")
    args = parser.parse_args()
    run(args.allocations, args.output)


if __name__ == "__main__":
    main()
