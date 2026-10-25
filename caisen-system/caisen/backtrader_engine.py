"""Backtrader execution harness for zero-cost order/ledger validation."""

import math

import backtrader as bt
import pandas as pd


def _iso(value):
    return pd.Timestamp(value).date().isoformat()


def _valid(open_price, signal):
    stop = float(signal["stop"])
    target = float(signal["target"])
    maximum = float(signal["maximum_entry"])
    return stop < open_price < target and open_price <= maximum


class _PlannedOrderStrategy(bt.Strategy):
    params = (("signals", None), ("default_lot_size", 100))

    def __init__(self):
        self.signal_by_entry = {_iso(item["entry_date"]): dict(item) for item in self.p.signals}
        self.signal_dates = set(self.signal_by_entry)
        self.seen = set()
        self.order_metadata = {}
        self.entries = {}
        self.order_events = []
        self.trades = []
        self.rejections = []

    def next_open(self):
        today = self.data.datetime.date(0).isoformat()
        signal = self.signal_by_entry.get(today)
        if signal is None or signal["entry_date"] in self.seen:
            return
        self.seen.add(signal["entry_date"])
        if self.position:
            self.rejections.append({"pattern_id": signal.get("pattern_id"), "signal_date": signal.get("signal_date"),
                                    "entry_date": today, "reason": "position_already_open"})
            return
        opening = float(self.data.open[0])
        if not _valid(opening, signal):
            self.rejections.append({"pattern_id": signal.get("pattern_id"), "signal_date": signal.get("signal_date"),
                                    "entry_date": today, "reason": "next_open_outside_plan", "open": opening})
            return
        lot = int(signal.get("lot_size", self.p.default_lot_size))
        size = int(self.broker.getcash() // (opening * lot)) * lot
        if size <= 0:
            self.rejections.append({"pattern_id": signal.get("pattern_id"), "signal_date": signal.get("signal_date"),
                                    "entry_date": today, "reason": "insufficient_cash"})
            return
        parent, stop_order, target_order = self.buy_bracket(
            size=size, exectype=bt.Order.Market,
            stopprice=float(signal["stop"]), limitprice=float(signal["target"]),
        )
        common = {"pattern_id": signal.get("pattern_id"), "signal_date": signal.get("signal_date"),
                  "entry_date": today, "planned_stop": float(signal["stop"]), "planned_target": float(signal["target"]),
                  "shares": size}
        self.order_metadata[parent.ref] = dict(common, role="entry", side="buy")
        self.order_metadata[stop_order.ref] = dict(common, role="stop", side="sell")
        self.order_metadata[target_order.ref] = dict(common, role="target", side="sell")

    def notify_order(self, order):
        metadata = self.order_metadata.get(order.ref)
        if not metadata:
            return
        status = order.getstatusname().lower()
        event = dict(metadata, order_ref=order.ref, status=status)
        if order.status == order.Completed:
            filled_at = bt.num2date(order.executed.dt).date().isoformat()
            event.update({"date": filled_at, "filled_size": abs(int(order.executed.size)),
                          "price": float(order.executed.price), "commission": float(order.executed.comm),
                          "cash_after": float(self.broker.getcash())})
            if metadata["role"] == "entry":
                self.entries[metadata["pattern_id"]] = {
                    "entry_date": filled_at, "entry_price": float(order.executed.price),
                    "shares": abs(int(order.executed.size)), "signal_date": metadata["signal_date"],
                    "planned_stop": metadata["planned_stop"], "planned_target": metadata["planned_target"],
                }
            else:
                entry = self.entries.get(metadata["pattern_id"])
                if entry:
                    exit_price = float(order.executed.price)
                    quantity = abs(int(order.executed.size))
                    self.trades.append({**entry, "exit_date": filled_at, "exit_price": exit_price,
                                        "exit_reason": metadata["role"], "exit_shares": quantity,
                                        "pnl": (exit_price - entry["entry_price"]) * quantity})
        self.order_events.append(event)


def run_backtrader_simulation(bars, signals, cash=100000, default_lot_size=100):
    """Run planned one-shot entry orders and stop/target brackets through Backtrader."""
    frame = bars.copy()
    if frame.empty:
        return _empty_result(cash, "no daily bars")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
    frame = frame.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    dated_signals = [dict(signal, entry_date=_iso(signal["entry_date"])) for signal in signals]
    available_dates = {_iso(value) for value in frame.index}
    missing = [{"pattern_id": item.get("pattern_id"), "signal_date": item.get("signal_date"),
                "entry_date": item["entry_date"], "reason": "no_next_daily_session"}
               for item in dated_signals if item["entry_date"] not in available_dates]
    dated_signals = [item for item in dated_signals if item["entry_date"] in available_dates]
    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    feed = bt.feeds.PandasData(dataname=frame)
    cerebro.adddata(feed)
    cerebro.addstrategy(_PlannedOrderStrategy, signals=dated_signals, default_lot_size=int(default_lot_size))
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_coo(True)
    strategy = cerebro.run(runonce=False)[0]
    available_cash = float(cerebro.broker.getcash())
    position = strategy.getposition(feed)
    shares = int(position.size)
    close = float(frame.close.iloc[-1])
    marked_value = float(shares * close)
    ending_equity = float(cerebro.broker.getvalue())
    open_position = None
    if shares:
        active_entries = [entry for entry in strategy.entries.values()]
        entry = active_entries[-1] if active_entries else {}
        open_position = {**entry, "shares": shares, "mark_price": close, "marked_value": marked_value}
    reconciles = math.isclose(ending_equity, available_cash + marked_value, rel_tol=1e-9, abs_tol=1e-6)
    return {
        "engine": "backtrader", "commission": 0.0, "slippage": 0.0,
        "starting_cash": float(cash), "available_cash": available_cash,
        "marked_position_value": marked_value, "ending_equity": ending_equity,
        "open_position": open_position, "trades": strategy.trades,
        "order_events": strategy.order_events, "rejections": strategy.rejections + missing,
        "accounting_invariants": {"equity_equals_cash_plus_marked_position": reconciles,
                                  "zero_commission": all(event.get("commission", 0.0) == 0.0 for event in strategy.order_events)},
        "limitations": ["模拟执行验收，不评价策略收益有效性", "同一标的单次只持一笔，止损订单按先于首目标的顺序提交"],
    }


def _empty_result(cash, reason):
    return {"engine": "backtrader", "commission": 0.0, "slippage": 0.0,
            "starting_cash": float(cash), "available_cash": float(cash),
            "marked_position_value": 0.0, "ending_equity": float(cash), "open_position": None,
            "trades": [], "order_events": [], "rejections": [],
            "accounting_invariants": {"equity_equals_cash_plus_marked_position": True, "zero_commission": True},
            "limitations": [reason]}
