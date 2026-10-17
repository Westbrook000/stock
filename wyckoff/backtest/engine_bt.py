"""保留旧版 Backtrader 接口兼容性。

当前回测入口 runner 使用 daily.simulate_daily 撮合和统计；此处旧 Strategy 不参与研究回测。
"""
from __future__ import annotations

import backtrader as bt
import pandas as pd
from .daily import simulate_daily, prepare_daily


ENT_CODES = {"": 0, "spring": 1, "test": 2, "lps": 3, "st": 4,
             "demand_zone": 5, "sos": 6, "joc": 7, "st_spring": 8, "st_sc": 9, "st_breakout": 10, "sc": 11}
ENT_NAMES = {v: k for k, v in ENT_CODES.items()}


class SignalData(bt.feeds.PandasData):
    """附加线: sig(1=BUY确认, -1=SELL确认, 0=无), stop(买入参考止损价), ent(入场类型码)."""
    lines = ("sig", "stop", "ent")
    params = (("sig", -1), ("stop", -1), ("ent", 0))


def signals_to_feed(df: pd.DataFrame, sig: pd.DataFrame) -> SignalData:
    data = df[["date", "open", "high", "low", "close", "volume"]].copy()
    s = pd.Series(0.0, index=data.index)
    s[sig["action"].eq("BUY").to_numpy()] = 1.0
    s[sig["action"].eq("SELL").to_numpy()] = -1.0
    data["sig"] = s.to_numpy()
    stop = sig["stop"].ffill().to_numpy()
    data["stop"] = stop
    data["ent"] = sig["entry_type"].map(ENT_CODES).fillna(0).to_numpy()
    data = data.set_index("date")
    return SignalData(dataname=data)


class WyckoffStrategy(bt.Strategy):
    """规则（对齐原著风控 + 旧脚本经验）：
    - BUY确认收盘可见 → 次日开盘市价买入（一笔一标的）。
    - 退出：SELL信号 / 止损(stop) / 止盈(take_profit) / 持仓超时(max_hold)。
    - 止损优先用Spring破位低点，缺失时用入场价-stop_pct。
    """
    params = dict(
        take_profit=0.15,
        stop_pct=0.05,
        max_hold_days=20,
        use_signal_stop=True,
        trail=False,
        trail_atr_mult=2.5,
    )

    def __init__(self) -> None:
        self.entry_price = None
        self.entry_idx = None
        self.entry_ent = ""
        self.stop_level = None
        self.trades: list = []
        self._pending = None
        self.order = None
        self.atr = bt.indicators.ATR(self.data, period=14)

    # 信号在收盘产生，开盘执行
    def next(self):
        if self.order:
            return
        sig = self.data.lines.sig[0]
        if self.position:
            exit_reason = None
            price = None
            bar_low, bar_high = self.data.low[0], self.data.high[0]
            if self.stop_level and bar_low <= self.stop_level:
                exit_reason, price = "stop", min(self.stop_level, bar_high)
            elif self.p.take_profit and bar_high >= self.entry_price * (1 + self.p.take_profit):
                exit_reason, price = "take_profit", self.entry_price * (1 + self.p.take_profit)
            elif self.p.trail and self.atr[0]:
                new_stop = self.data.close[0] - self.p.trail_atr_mult * self.atr[0]
                self.stop_level = max(self.stop_level or 0.0, new_stop)
            if exit_reason is None and sig < 0:
                exit_reason, price = "signal", self.data.close[0]
            if exit_reason is None and self.p.max_hold_days:
                held = len(self) - self.entry_idx
                if held >= self.p.max_hold_days:
                    exit_reason, price = "timeout", self.data.close[0]
            if exit_reason:
                self._pending = ("sell", exit_reason, price)
                self.order = self.close()
            return

        if sig > 0:
            ref_stop = self.data.lines.stop[0]
            if self.p.use_signal_stop and ref_stop and ref_stop > 0 and ref_stop < self.data.close[0]:
                self.stop_level = float(ref_stop)
            else:
                self.stop_level = self.data.close[0] * (1 - self.p.stop_pct)
            self.entry_ent = ENT_NAMES.get(int(self.data.lines.ent[0]), "")
            self._pending = ("buy", "signal", None)
            self.order = self.order_target_percent(target=0.99)

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.entry_idx = len(self)
                if self.stop_level and self.entry_price:
                    # 止损相对成交价保护最大幅度
                    self.stop_level = max(self.stop_level, self.entry_price * (1 - self.p.stop_pct))
                self.trades.append(dict(dt=self.data.datetime.date(0), direction="buy",
                                        price=order.executed.price,
                                        reason=self._pending[1] if self._pending else "",
                                        entry_type=self.entry_ent))
            else:
                self.trades.append(dict(dt=self.data.datetime.date(0), direction="sell",
                                        price=order.executed.price,
                                        reason=self._pending[1] if self._pending else "",
                                        entry_type=self.entry_ent))
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            self.order = None
