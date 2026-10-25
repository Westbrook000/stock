"""Build causal pattern entry plans and delegate all fills to Backtrader."""

import pandas as pd

from .backtrader_engine import run_backtrader_simulation
from .market import aggregate_weekly, default_china_calendar
from .patterns import current_pattern_snapshot, detect_patterns
from .plans import build_trade_plan


def valid_next_open(open_price, stop, first_target, highest_allowed, structure_valid=True):
    return bool(structure_valid and stop < open_price < first_target and open_price <= highest_allowed)


def run_backtest(bars: pd.DataFrame, code: str, kind: str, timeframe="1d", cash=100000,
                 start=None, calendar=None):
    """Replay rules causally, use daily bars for fills, and return a Backtrader ledger."""
    daily = bars.copy().reset_index(drop=True)
    if daily.empty:
        result = run_backtrader_simulation(daily, [], cash=cash)
        result.update({"code": code, "kind": kind, "timeframe": timeframe, "signal_count": 0,
                       "strategy_effectiveness": "deferred"})
        return result
    daily["date"] = pd.to_datetime(daily["date"])
    # Pattern search is bounded by project windows: 500 daily bars or a
    # conservative 5 daily sessions per each of 520 weekly bars plus warm-up.
    lookback = 520 * 5 + 50 if timeframe == "1w" else 500
    if start:
        first_signal = int(daily.date.searchsorted(pd.Timestamp(start)))
        daily = daily.iloc[max(0, first_signal - lookback):].reset_index(drop=True)
    else:
        daily = daily.tail(lookback).reset_index(drop=True)
    signal_bars = aggregate_weekly(
        daily, calendar=calendar if calendar is not None else default_china_calendar()
    ) if timeframe == "1w" else daily
    signals = []
    seen = set()
    daily_dates = pd.to_datetime(daily.date)
    if len(signal_bars):
        for end in range(min(25, len(signal_bars)), len(signal_bars)):
            prefix = signal_bars.iloc[:end + 1].copy()
            signal_date = pd.Timestamp(prefix.date.iloc[-1])
            if start and signal_date < pd.Timestamp(start):
                continue
            entry_index = int(daily_dates.searchsorted(signal_date, side="right"))
            if entry_index < len(daily):
                entry_date = str(pd.Timestamp(daily.date.iloc[entry_index]).date())
            else:
                # A final signal with no next bar is deliberately rejected by the simulator.
                entry_date = str((signal_date + pd.Timedelta(days=1)).date())
            for pattern in current_pattern_snapshot(detect_patterns(prefix, code, kind, timeframe)):
                if pattern.direction != "bullish" or pattern.state != "confirmed" or pattern.pattern_id in seen:
                    continue
                seen.add(pattern.pattern_id)
                signal_close = float(prefix.close.iloc[-1])
                plan = build_trade_plan(pattern, signal_close, cash=cash,
                                        lot_size=int(daily.attrs.get("quality", {}).get("lot_size", 100)),
                                        bars=prefix,
                                        tick_size=float(daily.attrs.get("quality", {}).get("tick_size", 0.01)))
                if plan["decision"] != "CONSIDER_BUY" or not plan.get("targets"):
                    continue
                signals.append({
                    "signal_date": str(signal_date.date()), "entry_date": entry_date,
                    "pattern_id": pattern.pattern_id, "stop": float(plan["stop"]),
                    "target": float(plan["targets"][0]),
                    "maximum_entry": float(plan["highest_allowed_entry"]),
                    "lot_size": int(daily.attrs.get("quality", {}).get("lot_size", 100)),
                })
    result = run_backtrader_simulation(daily, signals, cash=cash,
                                       default_lot_size=int(daily.attrs.get("quality", {}).get("lot_size", 100)))
    result.update({"code": code, "kind": kind, "timeframe": timeframe,
                   "signal_count": len(signals), "strategy_effectiveness": "deferred",
                   "signal_source": "causal_pattern_confirmations",
                   "limitations": result["limitations"] + ["策略收益与识别表现未做效果验证"]})
    return result
