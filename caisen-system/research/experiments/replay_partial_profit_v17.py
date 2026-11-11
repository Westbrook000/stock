"""Research-only Backtrader replay of partial 8% profit-taking variants.

Reads the frozen v17 signals and the matching local OHLCV history. Does not
modify the production backtest strategy or recognition rules.
"""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
import json
import math
import sys
import time

import backtrader as bt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports/daily-research/v17-directional-suppression-neckline-revisions-100-20261110-attempt02"
ACCOUNTS = RUN / "accounts"
DATA = ROOT.parent / "data/stocks"
OUT = RUN / "partial-profit-v17.json"
sys.path.insert(0, str(ROOT))
from caisen.fixed_backtest import simulate_fixed_trades, calculate_metrics  # noqa: E402


class PartialProfitStrategy(bt.Strategy):
    params = (("signals", None), ("fraction", 0.6), ("runner_breakeven", False),
              ("ratchet_trigger", None), ("ratchet_floor", None),
              ("lot_size", 100), ("last_date", None))

    def __init__(self):
        self.by_entry = defaultdict(list)
        for signal in self.p.signals:
            self.by_entry[pd.Timestamp(signal["entry_date"]).date().isoformat()].append(signal)
        self.pending = None
        self.meta = {}
        self.active_exits = {}
        self.trade = None
        self.trades = []
        self.rejections = []
        self.curve = []
        self.fills = []
        self.same_bar_conflicts = 0
        self.partial_entries = 0
        self.ratchet_entries = 0
        self.pending_ratchet_replacement = None

    def next_open(self):
        today = self.data.datetime.date(0).isoformat()
        if self.position.size:
            for signal in self.by_entry.get(today, []):
                self.rejections.append({"signal_date": signal["signal_date"], "entry_date": today,
                                        "pattern_ids": signal.get("pattern_ids", []),
                                        "reason": "position_already_open"})
            return
        if self.pending:
            return
        for signal in self.by_entry.get(today, []):
            opening = float(self.data.open[0])
            shares = int(self.broker.getcash() // (opening * int(self.p.lot_size))) * int(self.p.lot_size)
            if shares <= 0:
                self.rejections.append({"signal_date": signal["signal_date"], "entry_date": today,
                                        "pattern_ids": signal.get("pattern_ids", []),
                                        "reason": "insufficient_cash"})
                continue
            order = self.buy(size=shares, exectype=bt.Order.Market)
            self.pending = order.ref
            self.meta[order.ref] = {"role": "entry", "signal": signal, "shares": shares}
            break

    def next(self):
        today = self.data.datetime.date(0).isoformat()
        if today == self.p.last_date and self.position.size:
            for order in list(self.active_exits.values()):
                if order.alive():
                    self.cancel(order)
            self.broker.set_coc(True)
            order = self.close(exectype=bt.Order.Market)
            self.meta[order.ref] = {"role": "end_of_period"}
        elif (self.position.size and self.trade is not None
              and self.p.ratchet_trigger is not None
              and not self.trade.get("ratchet_active")
              and float(self.data.high[0]) >= self.trade["entry_price"] * (1 + self.p.ratchet_trigger)):
            self.trade["ratchet_active"] = True
            self.ratchet_entries += 1
            self.pending_ratchet_replacement = {
                "stop": self.trade["entry_price"] * (1 + float(self.p.ratchet_floor)),
                "target": self.trade["entry_price"] * 1.20,
            }
            for exit_order in list(self.active_exits.values()):
                if exit_order.alive():
                    self.cancel(exit_order)
        close = float(self.data.close[0])
        invested = float(self.position.size) * close
        self.curve.append({"date": today, "cash": float(self.broker.getcash()),
                           "invested_value": invested,
                           "equity": float(self.broker.getcash()) + invested,
                           "shares": int(self.position.size)})

    def _exit_pair(self, tranche, shares, stop_price, target_price, target_role):
        stop = self.sell(size=shares, exectype=bt.Order.Stop, price=stop_price)
        target = self.sell(size=shares, exectype=bt.Order.Limit, price=target_price, oco=stop)
        self.meta[stop.ref] = {"role": "stop_loss", "tranche": tranche}
        self.meta[target.ref] = {"role": target_role, "tranche": tranche}
        self.active_exits[stop.ref] = stop
        self.active_exits[target.ref] = target
        if tranche == "runner":
            self.trade["runner_stop_ref"] = stop.ref
            self.trade["runner_target_ref"] = target.ref

    def _install_brackets(self):
        trade = self.trade
        shares = trade["shares"]
        lot = int(self.p.lot_size)
        if self.p.ratchet_trigger is not None:
            trade["partial_shares"] = 0
            trade["runner_shares"] = shares
            trade["ratchet_active"] = False
            self._exit_pair("runner", shares, trade["entry_price"] * 0.92,
                            trade["entry_price"] * 1.20, "take_profit_20")
            return
        first = int(shares * float(self.p.fraction) // lot) * lot
        runner = shares - first
        if first <= 0 or runner <= 0:
            first, runner = 0, shares
        else:
            self.partial_entries += 1
        trade["partial_shares"] = first
        trade["runner_shares"] = runner
        stop = trade["entry_price"] * 0.92
        if first:
            self._exit_pair("partial", first, stop, trade["entry_price"] * 1.08, "take_profit_8")
        self._exit_pair("runner", runner, stop, trade["entry_price"] * 1.20, "take_profit_20")

    def _finalize_trade(self, exit_date, forced=False):
        if self.trade is None:
            return
        trade = self.trade
        pnl = float(trade["realized_pnl"])
        exit_shares = int(trade["shares"])
        exit_price = trade["entry_price"] + pnl / exit_shares if exit_shares else trade["entry_price"]
        reasons = set(trade["exit_reasons"])
        if forced:
            reason = "end_of_period"
        elif reasons == {"take_profit_20"}:
            reason = "take_profit_20"
        elif reasons == {"take_profit_8"}:
            reason = "take_profit_8"
        elif "stop_loss" in reasons and "take_profit_8" in reasons:
            reason = "partial_8_then_stop"
        elif "stop_loss" in reasons:
            reason = "stop_loss"
        else:
            reason = "mixed_targets"
        shares = int(trade["shares"])
        self.trades.append({**{key: value for key, value in trade.items()
                               if key not in {"realized_pnl", "exit_reasons", "runner_stop_ref",
                                              "runner_target_ref", "remaining_shares",
                                              "ratchet_active"}},
                            "exit_date": exit_date, "exit_price": exit_price,
                            "exit_reason": reason, "exit_reasons": sorted(reasons),
                            "pnl": pnl, "return_pct": pnl / (trade["entry_price"] * shares),
                            "holding_sessions": max(len(self) - 1 - trade["entry_index"], 0),
                            "holding_days": (pd.Timestamp(exit_date) - pd.Timestamp(trade["entry_date"])).days})
        self.trade = None

    def notify_order(self, order):
        meta = self.meta.get(order.ref)
        if not meta or order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            role = meta["role"]
            self.fills.append({"role": role, "date": bt.num2date(order.executed.dt).date().isoformat(),
                               "size": int(order.executed.size), "price": float(order.executed.price),
                               "cash": float(self.broker.getcash()),
                               "position": int(self.position.size)})
            if role == "entry":
                signal = meta["signal"]
                date = bt.num2date(order.executed.dt).date().isoformat()
                price = float(order.executed.price)
                self.trade = {"entry_index": len(self) - 1, "entry_date": date,
                              "entry_price": price, "shares": abs(int(order.executed.size)),
                              "signal_date": signal["signal_date"],
                              "pattern_ids": signal.get("pattern_ids", []),
                              "pattern_types": signal.get("pattern_types", []),
                              "pattern_tags": signal.get("pattern_tags", []),
                              "split": signal.get("split", "development"),
                              "realized_pnl": 0.0, "exit_reasons": [],
                              "remaining_shares": abs(int(order.executed.size))}
                self._install_brackets()
            else:
                self.active_exits.pop(order.ref, None)
                date = bt.num2date(order.executed.dt).date().isoformat()
                if role == "end_of_period":
                    if self.trade is not None:
                        pnl = (float(order.executed.price) - self.trade["entry_price"]) * abs(int(order.executed.size))
                        self.trade["realized_pnl"] += pnl
                        self.trade["exit_reasons"].append(role)
                        self.trade["remaining_shares"] = 0
                    self._finalize_trade(date, forced=True)
                elif self.trade is not None:
                    fill_size = abs(int(order.executed.size))
                    pnl = (float(order.executed.price) - self.trade["entry_price"]) * fill_size
                    self.trade["realized_pnl"] += pnl
                    self.trade["exit_reasons"].append(role)
                    self.trade["remaining_shares"] -= fill_size
                    if role in {"take_profit_8", "take_profit_20"}:
                        stop = self.trade["entry_price"] * (1.0 if self.p.runner_breakeven else 0.92)
                        target = self.trade["entry_price"] * 1.20
                        if float(self.data.low[0]) <= stop and float(self.data.high[0]) >= target:
                            self.same_bar_conflicts += 1
                    if (role == "take_profit_8" and self.p.runner_breakeven
                            and self.position.size > 0 and self.trade.get("runner_stop_ref") in self.meta):
                        old_stop_ref = self.trade["runner_stop_ref"]
                        old_stop = self.active_exits.get(old_stop_ref)
                        if old_stop is not None and old_stop.alive():
                            self.cancel(old_stop)
                        runner_shares = self.trade["runner_shares"]
                        self._exit_pair("runner", runner_shares,
                                        self.trade["entry_price"], self.trade["entry_price"] * 1.20,
                                        "take_profit_20")
                    if self.trade["remaining_shares"] <= 0:
                        self._finalize_trade(date)
        elif order.status in (order.Canceled, order.Margin, order.Rejected, order.Expired):
            self.active_exits.pop(order.ref, None)
            if meta["role"] == "entry":
                self.rejections.append({"reason": order.getstatusname().lower()})
            if (self.pending_ratchet_replacement is not None and self.trade is not None
                    and self.position.size > 0
                    and not any(candidate.alive() for candidate in self.active_exits.values())):
                replacement = self.pending_ratchet_replacement
                self.pending_ratchet_replacement = None
                self._exit_pair("runner", int(self.position.size), replacement["stop"],
                                replacement["target"], "take_profit_20")
        if self.pending == order.ref:
            self.pending = None


def replay(frame, signals, fraction, breakeven=False, cash=100000.0, lot_size=100,
           ratchet_trigger=None, ratchet_floor=None):
    frame = frame.copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    final_date = frame.date.iloc[-1].date().isoformat()
    sentinel = frame.iloc[-1].copy()
    sentinel[["open", "high", "low", "close"]] = float(frame.close.iloc[-1])
    sentinel["volume"] = 0.0
    frame = pd.concat([frame, pd.DataFrame([sentinel])], ignore_index=True)
    frame = frame.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    valid_dates = {pd.Timestamp(value).date().isoformat() for value in frame.index[:-1]}
    signals = [signal for signal in signals if signal.get("entry_date") in valid_dates
               and signal.get("entry_date") != final_date]
    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.addstrategy(PartialProfitStrategy, signals=signals, fraction=fraction,
                        runner_breakeven=breakeven, ratchet_trigger=ratchet_trigger,
                        ratchet_floor=ratchet_floor, lot_size=lot_size, last_date=final_date)
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_coo(True)
    strategy = cerebro.run(runonce=False)[0]
    curve = strategy.curve[:-1] if strategy.curve and strategy.curve[-1]["shares"] == 0 else strategy.curve
    ending = float(cerebro.broker.getcash())
    return {"trades": strategy.trades, "rejections": strategy.rejections,
            "daily_equity": curve, "ending_equity": ending,
            "fills": strategy.fills,
            "partial_entries": strategy.partial_entries,
            "ratchet_entries": strategy.ratchet_entries,
            "same_bar_conflicts": strategy.same_bar_conflicts,
            "accounting_checks": {
                "cash_plus_position_equals_equity": all(math.isclose(
                    p["cash"] + p["invested_value"], p["equity"], rel_tol=1e-9, abs_tol=1e-5)
                    for p in curve),
                "realized_pnl_matches_ending_equity": math.isclose(
                    100000.0 + sum(t["pnl"] for t in strategy.trades), ending,
                    rel_tol=1e-9, abs_tol=1e-4),
                "closed_trade_positions": not strategy.position.size,
            }}


def main():
    files = sorted(ACCOUNTS.glob("stock-*.json"))
    if len(files) != 100:
        raise SystemExit(f"Expected 100 frozen stock accounts, got {len(files)}")
    variants = [(0.0, False, "all_20_baseline"), (0.5, False, "half_8_half_20"),
                (0.6, False, "60pct_8_40pct_20"), (0.7, False, "70pct_8_30pct_20"),
                (0.6, True, "60pct_8_runner_breakeven_then_20")]
    records = {name: [] for _, _, name in variants}
    account_curves = {name: [] for _, _, name in variants}
    started = time.time()
    for index, source in enumerate(files, 1):
        account = json.loads(source.read_text())
        code = account["code"]
        bars_path = DATA / f"{code}.csv"
        frame = pd.read_csv(bars_path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame.date <= "2026-10-30"].reset_index(drop=True)
        official = simulate_fixed_trades(frame, account["signals"], cash=100000.0)
        stored_trades = account["trades"]
        if len(official["trades"]) != len(stored_trades):
            raise AssertionError(f"{code}: baseline trade count mismatch")
        for left, right in zip(official["trades"], stored_trades):
            for key in ("entry_date", "exit_date", "exit_reason", "shares"):
                if left.get(key) != right.get(key):
                    raise AssertionError(f"{code}: baseline {key} mismatch")
            if not math.isclose(left["pnl"], right["pnl"], abs_tol=1e-7):
                raise AssertionError(f"{code}: baseline pnl mismatch")
        for fraction, breakeven, name in variants:
            outcome = replay(frame, account["signals"], fraction, breakeven)
            if not all(outcome["accounting_checks"].values()):
                raise AssertionError(f"{code}/{name}: accounting failed: {outcome['accounting_checks']}")
            if name == "all_20_baseline":
                if len(outcome["trades"]) != len(stored_trades):
                    raise AssertionError(f"{code}: custom baseline count mismatch")
                for left, right in zip(outcome["trades"], stored_trades):
                    if left["entry_date"] != right["entry_date"] or left["exit_date"] != right["exit_date"]:
                        raise AssertionError(f"{code}: custom baseline dates mismatch")
                    if not math.isclose(left["pnl"], right["pnl"], abs_tol=1e-6):
                        raise AssertionError(f"{code}: custom baseline PnL mismatch")
            metrics = calculate_metrics(outcome["trades"], 100000.0,
                                        outcome["ending_equity"], outcome["daily_equity"])
            records[name].append({"code": code, "metrics": metrics,
                                  "ending_equity": outcome["ending_equity"],
                                  "trade_count": len(outcome["trades"]),
                                  "partial_entries": outcome["partial_entries"],
                                  "same_bar_conflicts": outcome["same_bar_conflicts"],
                                  "accounting_checks": outcome["accounting_checks"],
                                  "trades": outcome["trades"]})
            account_curves[name].append(outcome["daily_equity"])
        if index % 10 == 0:
            print(f"{index}/100 accounts; elapsed={time.time()-started:.1f}s", flush=True)
    summaries = {}
    for name, accounts in records.items():
        trades = [trade for account in accounts for trade in account["trades"]]
        ending = sum(account["ending_equity"] for account in accounts)
        metrics = calculate_metrics(trades, 10_000_000.0, ending, [])
        daily_map = []
        all_dates = set()
        for account_curve in account_curves[name]:
            values = {point["date"]: (float(point["equity"]), float(point["invested_value"]))
                      for point in account_curve}
            daily_map.append(values)
            all_dates.update(values)
        dates = sorted(all_dates)
        account_values = [100000.0] * len(accounts)
        invested_values = [0.0] * len(accounts)
        equity_curve = []
        pointers = [0] * len(accounts)
        rows_by_date = [sorted(curve.items()) for curve in daily_map]
        for date in dates:
            for account_index, rows in enumerate(rows_by_date):
                pointer = pointers[account_index]
                if pointer < len(rows) and rows[pointer][0] == date:
                    account_values[account_index], invested_values[account_index] = rows[pointer][1]
                    pointers[account_index] += 1
            equity_curve.append({"date": date, "equity": sum(account_values),
                                 "invested_value": sum(invested_values)})
        total_invested = 0.0
        for point in equity_curve:
            total_invested += point["invested_value"]
        # Use the actual synchronized independent-account equity curve for drawdown.
        peak = 10_000_000.0
        max_dd = 0.0
        for point in equity_curve:
            peak = max(peak, point["equity"])
            max_dd = max(max_dd, (peak - point["equity"]) / peak if peak > 0 else 0.0)
        metrics["maximum_drawdown"] = max_dd
        metrics["capital_utilization"] = (total_invested / (sum(p["equity"] for p in equity_curve))
                                          if equity_curve and sum(p["equity"] for p in equity_curve) > 0 else 0.0)
        bands = {"through_2020": [], "2021_2023": [], "2024_plus": []}
        for trade in trades:
            year = int(str(trade["signal_date"])[:4])
            band = ("through_2020" if year <= 2020 else
                    "2021_2023" if year <= 2023 else "2024_plus")
            bands[band].append(trade)
        metrics["signal_date_bands"] = {
            band: {"trades": len(rows), "wins": sum(t["pnl"] > 0 for t in rows),
                   "win_rate": sum(t["pnl"] > 0 for t in rows) / len(rows) if rows else None,
                   "net_pnl": sum(float(t["pnl"]) for t in rows)}
            for band, rows in bands.items()}
        metrics.update({"account_count": len(accounts),
                        "accounts_with_all_checks": sum(all(a["accounting_checks"].values()) for a in accounts),
                        "median_account_return": sorted(a["metrics"]["cumulative_return"] for a in accounts)[len(accounts)//2],
                        "positive_accounts": sum(a["metrics"]["cumulative_return"] > 0 for a in accounts),
                        "partial_entries": sum(a["partial_entries"] for a in accounts),
                        "same_bar_conflicts": sum(a["same_bar_conflicts"] for a in accounts),
                        "average_account_win_rate": sum(a["metrics"]["win_rate"] or 0 for a in accounts) / len(accounts)})
        summaries[name] = metrics
    output = {"baseline_run_id": "7f79f305fe214de48844a9f065632774",
              "recognizer": "caisen-daily-boundary-first-v11",
              "recognizer_schema": "caisen-daily-continuous-v12",
              "source_rule_version": "caisen-main-bottoms-v16",
              "cutoff": "2026-10-30", "account_count": 100,
              "starting_capital": 10_000_000,
              "variants": summaries, "account_details": records,
              "elapsed_seconds": time.time()-started,
              "market_data_identity": "Each local CSV was replayed through 2026-10-30; v17 saved-trade replay verified exactly against current loaded bars and stored baseline ledger.",
              "interpretation": "Research only; historical development sample, not out-of-sample validation."}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "elapsed_seconds": output["elapsed_seconds"],
                      "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
