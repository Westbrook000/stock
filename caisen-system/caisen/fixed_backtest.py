"""Causal fixed 8% stop / 20% target pattern replay.

Each code/timeframe is an isolated cash account. Signal discovery is causal:
the recognizer only sees bars available at that observation, while retaining
the full observed prefix as background for its quality gates.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd
import backtrader as bt

from .market import aggregate_weekly, default_china_calendar
from .patterns import PATTERN_LABELS
from .migration import migration_status


STRATEGY_VERSION_V1 = "pattern_fixed_8_20_v1"
STRATEGY_VERSION_V2 = "pattern_fixed_8_20_main_parent_v2"
SUPPORTED_STRATEGY_VERSIONS = frozenset({STRATEGY_VERSION_V1, STRATEGY_VERSION_V2})
STRATEGY_VERSION = STRATEGY_VERSION_V2
MAIN_BOTTOM_TYPES = {"w_bottom", "inverse_head_shoulders", "triangle_bottom"}
PATTERN_TYPES = set(MAIN_BOTTOM_TYPES)
LOT_SIZE = 100
STARTING_CASH = 100000.0
STOP_FRACTION = 0.08
TARGET_FRACTION = 0.20


def _iso(value):
    return pd.Timestamp(value).date().isoformat()


def calculate_metrics(trades, starting_cash, ending_equity, daily_equity):
    """Return metrics with undefined trade ratios represented as JSON nulls."""
    trades = list(trades)
    pnls = [float(trade["pnl"]) for trade in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    exit_reasons = defaultdict(int)
    for trade in trades:
        exit_reasons[str(trade.get("exit_reason", "unknown"))] += 1
    gross_loss = -sum(losses)
    peak = float(starting_cash)
    max_drawdown = 0.0
    invested = []
    equity_values = []
    for point in daily_equity:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
        invested.append(float(point.get("invested_value", 0.0)))
        equity_values.append(equity)
    if trades:
        first = pd.Timestamp(daily_equity[0]["date"]) if daily_equity else min(pd.Timestamp(trade["entry_date"]) for trade in trades)
        last = pd.Timestamp(daily_equity[-1]["date"]) if daily_equity else max(pd.Timestamp(trade["exit_date"]) for trade in trades)
        years = max((last - first).days / 365.25, 1 / 365.25)
        annualized = ((float(ending_equity) / float(starting_cash)) ** (1 / years) - 1
                      if float(starting_cash) > 0 else None)
    else:
        annualized = 0.0
    observed_period_years = (max((pd.Timestamp(daily_equity[-1]["date"]) - pd.Timestamp(daily_equity[0]["date"])).days, 0) / 365.25
                             if daily_equity else None)
    return {
        "trade_count": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else None,
        "cumulative_return": (float(ending_equity) / float(starting_cash) - 1.0)
        if float(starting_cash) > 0 else None,
        "annualized_return": annualized,
        "annualization_period_years": observed_period_years,
        "maximum_drawdown": max_drawdown,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else (None if not wins else None),
        "average_pnl": (sum(pnls) / len(pnls)) if trades else None,
        "expectancy": (sum(pnls) / len(pnls)) if trades else None,
        "average_holding_days": (sum(int(t.get("holding_days", 0)) for t in trades) / len(trades)) if trades else None,
        "average_holding_sessions": (sum(int(t.get("holding_sessions", t.get("holding_days", 0))) for t in trades) / len(trades)) if trades else None,
        "exit_reason_counts": dict(sorted(exit_reasons.items())),
        "capital_utilization": (sum(invested) / sum(equity_values)) if sum(equity_values) > 0 else 0.0,
    }


class _FixedRiskRewardStrategy(bt.Strategy):
    params = (("signals", None), ("lot_size", LOT_SIZE), ("last_date", None))

    def __init__(self):
        self.by_entry = defaultdict(list)
        for signal in self.p.signals:
            self.by_entry[_iso(signal["entry_date"])].append(dict(signal))
        self.order_meta = {}
        self.pending = None
        self.exit_orders = {}
        self.position_meta = None
        self.trades = []
        self.rejections = []
        self.order_events = []
        self.curve = []

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
            lot = int(signal.get("lot_size", self.p.lot_size))
            shares = int(self.broker.getcash() // (opening * lot)) * lot
            if shares <= 0:
                self.rejections.append({"signal_date": signal["signal_date"], "entry_date": today,
                                        "pattern_ids": signal.get("pattern_ids", []), "reason": "insufficient_cash"})
                continue
            order = self.buy(size=shares, exectype=bt.Order.Market)
            self.pending = order.ref
            self.order_meta[order.ref] = {"role": "entry", "signal": signal, "shares": shares,
                                          "signal_date": signal["signal_date"]}
            break

    def next(self):
        today = self.data.datetime.date(0).isoformat()
        if today == self.p.last_date and self.position.size:
            for order in self.exit_orders.values():
                if order.alive():
                    self.cancel(order)
            # Enable close-on-close only for this final liquidation order.
            self.broker.set_coc(True)
            order = self.close(exectype=bt.Order.Market)
            self.pending = order.ref
            self.order_meta[order.ref] = {"role": "end_of_period"}
        close = float(self.data.close[0])
        invested_value = float(self.position.size) * close
        self.curve.append({"date": today, "cash": float(self.broker.getcash()),
                           "invested_value": invested_value,
                           "equity": float(self.broker.getcash()) + invested_value,
                           "shares": int(self.position.size)})

    def notify_order(self, order):
        meta = self.order_meta.get(order.ref)
        if not meta:
            return
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            date = bt.num2date(order.executed.dt).date().isoformat()
            role = meta["role"]
            self.order_events.append({"order_ref": order.ref, "role": role, "date": date,
                                      "status": "completed", "size": abs(int(order.executed.size)),
                                      "price": float(order.executed.price),
                                      "commission": float(order.executed.comm),
                                      "cash_after": float(self.broker.getcash())})
            if role == "entry":
                signal = meta["signal"]
                self.position_meta = {"entry_index": len(self) - 1, "entry_date": date,
                                      "entry_price": float(order.executed.price), "shares": abs(int(order.executed.size)),
                                      "signal_date": signal["signal_date"], "pattern_ids": signal.get("pattern_ids", []),
                                      "pattern_types": signal.get("pattern_types", []),
                                      "pattern_tags": signal.get("pattern_tags", []),
                                      "pattern_labels": signal.get("pattern_labels", []),
                                      "stop_price": float(order.executed.price) * (1 - STOP_FRACTION),
                                      "target_price": float(order.executed.price) * (1 + TARGET_FRACTION),
                                      "split": signal.get("split", "development")}
                # These children are submitted only after the entry fill. The
                # broker cannot evaluate them against the already processed
                # entry bar; it activates them on the next trading bar (T+1).
                stop = self.sell(size=self.position_meta["shares"], exectype=bt.Order.Stop,
                                 price=self.position_meta["stop_price"])
                target = self.sell(size=self.position_meta["shares"], exectype=bt.Order.Limit,
                                   price=self.position_meta["target_price"], oco=stop)
                self.exit_orders = {stop.ref: stop, target.ref: target}
                self.order_meta[stop.ref] = {"role": "stop_loss"}
                self.order_meta[target.ref] = {"role": "take_profit"}
            else:
                entry = self.position_meta
                if entry:
                    pnl = (float(order.executed.price) - entry["entry_price"]) * entry["shares"]
                    entry_index = max(0, int(entry["entry_index"]))
                    exit_index = max(entry_index, len(self) - 1)
                    entry_date = pd.Timestamp(entry["entry_date"])
                    exit_date = pd.Timestamp(date)
                    self.trades.append({**entry, "exit_date": date, "exit_price": float(order.executed.price),
                                        "exit_reason": role, "pnl": pnl,
                                        "return_pct": float(order.executed.price) / entry["entry_price"] - 1,
                                        "holding_sessions": max(exit_index - entry_index, 0),
                                        "holding_days": (exit_date - entry_date).days,
                                        "exit_shares": abs(int(order.executed.size))})
                    self.position_meta = None
        elif order.status in (order.Canceled, order.Margin, order.Rejected, order.Expired):
            self.order_events.append({"order_ref": order.ref, "role": meta["role"],
                                      "status": order.getstatusname().lower()})
            if meta["role"] == "entry":
                self.rejections.append({"signal_date": meta.get("signal_date"), "reason": order.getstatusname().lower()})
        if self.pending == order.ref:
            self.pending = None


def simulate_fixed_trades(bars, signals, cash=STARTING_CASH, lot_size=LOT_SIZE):
    """Submit actual orders to Backtrader's broker and return its fill ledger."""
    frame = bars.copy().reset_index(drop=True)
    if frame.empty:
        return {"trades": [], "rejections": [], "daily_equity": [], "order_events": [], "ending_equity": float(cash)}
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    final_date = _iso(frame.date.iloc[-1])
    sentinel = frame.iloc[-1].copy()
    sentinel["open"] = sentinel["high"] = sentinel["low"] = sentinel["close"] = float(frame.close.iloc[-1])
    sentinel["volume"] = 0.0
    # A same-date broker bar lets a real Backtrader market order fill at the
    # actual final close without changing the calendar or looking past it.
    frame = pd.concat([frame, pd.DataFrame([sentinel])], ignore_index=True)
    frame = frame.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    valid_dates = {_iso(value) for value in frame.index[:-1]}
    dated_signals = [dict(signal, entry_date=_iso(signal["entry_date"])) for signal in signals]
    missing = [{"signal_date": signal["signal_date"], "entry_date": signal["entry_date"],
                "pattern_ids": signal.get("pattern_ids", []), "reason": "no_next_t1_session"}
               for signal in dated_signals if signal["entry_date"] not in valid_dates or signal["entry_date"] == final_date]
    dated_signals = [signal for signal in dated_signals if signal["entry_date"] in valid_dates and signal["entry_date"] != final_date]
    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    data = bt.feeds.PandasData(dataname=frame)
    cerebro.adddata(data)
    cerebro.addstrategy(_FixedRiskRewardStrategy, signals=dated_signals, lot_size=int(lot_size),
                        last_date=final_date)
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_coo(True)
    strategy = cerebro.run(runonce=False)[0]
    ending = float(cerebro.broker.getcash())
    curve = strategy.curve[:-1] if strategy.curve and strategy.curve[-1]["shares"] == 0 else strategy.curve
    if curve and not strategy.position.size:
        curve[-1].update({"cash": ending, "invested_value": 0.0, "equity": ending, "shares": 0})
    return {"trades": strategy.trades, "rejections": strategy.rejections + missing,
            "order_events": strategy.order_events, "daily_equity": curve,
            "ending_equity": ending, "engine": "backtrader_broker"}


def _quality_pass(pattern):
    if pattern.pattern_type == "bull_flag":
        return True
    for item in pattern.evidence:
        if item.get("code") == "pattern_quality_pass":
            return bool(item.get("value"))
        if item.get("code") == "structure_uncertainty":
            value = item.get("value") or {}
            if "pattern_quality_pass" in value:
                return bool(value["pattern_quality_pass"])
    return False


def _main_bottom_detector():
    from . import main_bottoms

    return main_bottoms


def _link_main_bottom_children(main_events, legacy_patterns, bars_by_timeframe=None):
    return _main_bottom_detector().link_main_bottom_children(
        main_events, legacy_patterns, bars_by_timeframe=bars_by_timeframe)


def _evidence_value(pattern, code):
    for item in getattr(pattern, "evidence", []) or []:
        if item.get("code") == code:
            return item.get("value")
    return None


def _execution_suppressed(pattern):
    value = _evidence_value(pattern, "execution_suppression") or {}
    return bool(value.get("suppressed"))


def _pattern_audit_tag(pattern):
    return {
        "pattern_id": pattern.pattern_id,
        "pattern_type": pattern.pattern_type,
        "label": PATTERN_LABELS.get(pattern.pattern_type, pattern.pattern_type),
        "parent_id": getattr(pattern, "parent_id", None),
        "state": getattr(pattern, "state", None),
        "confirmation_date": getattr(pattern, "confirmation_date", None),
        "main_structure": _evidence_value(pattern, "main_structure"),
        "main_selection": _evidence_value(pattern, "main_selection"),
        "neckline_breakout": _evidence_value(pattern, "neckline_breakout"),
        "weak_crossings": _evidence_value(pattern, "weak_crossings"),
        "execution_suppression": _evidence_value(pattern, "execution_suppression"),
    }


def _eligible_main_patterns(patterns, observed_date):
    """Only selected, signal-eligible main parents with a current price break buy."""
    eligible = []
    for pattern in patterns:
        if pattern.pattern_type not in MAIN_BOTTOM_TYPES or pattern.direction != "bullish":
            continue
        if getattr(pattern, "parent_id", None) or _execution_suppressed(pattern):
            continue
        structure = _evidence_value(pattern, "main_structure") or {}
        selection = _evidence_value(pattern, "main_selection") or {}
        breakout = _evidence_value(pattern, "neckline_breakout") or {}
        if (not structure or structure.get("role") != "main"
                or structure.get("signal_eligible") is not True
                or selection.get("selected") is not True
                or selection.get("decided_at") != observed_date
                or selection.get("primary_pattern_id") != pattern.pattern_id
                or pattern.state != "confirmed" or pattern.confirmation_date != observed_date
                or breakout.get("date") != observed_date):
            continue
        eligible.append(pattern)
    return eligible


def _prefix_signals_v1(daily, timeframe, code, kind, calendar, trade_start=None):
    """Replay the original v1 causal signal loop against frozen v16 code."""
    from .v16_baseline.patterns import PATTERN_LABELS as v16_pattern_labels
    from .v16_baseline.patterns import detect_patterns as v16_detect_patterns

    v1_pattern_types = {"w_bottom", "inverse_head_shoulders", "triangle_bottom", "bull_flag"}
    daily = daily.copy().reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily.date)
    result = []
    seen = set()
    first_signal_index = (int(pd.to_datetime(daily.date).searchsorted(pd.Timestamp(trade_start), side="left"))
                          if trade_start else 30)
    for index in range(max(30, first_signal_index), len(daily)):
        background = daily.iloc[:index + 1].copy()
        if timeframe == "1w":
            signal_frame = aggregate_weekly(background, calendar=calendar)
            window = 520
        else:
            signal_frame = background
            window = 500
        if len(signal_frame) < 8:
            continue
        observed_date = pd.Timestamp(signal_frame.date.iloc[-1])
        structure_window = signal_frame.tail(window).copy().reset_index(drop=True)
        patterns = v16_detect_patterns(structure_window, code, kind, timeframe,
                                       background_bars=background,
                                       structure_window=window)
        found = []
        for pattern in patterns:
            if pattern.pattern_type not in v1_pattern_types or pattern.direction != "bullish":
                continue
            if pattern.confirmation_date != _iso(observed_date) or pattern.pattern_id in seen:
                continue
            if not _quality_pass(pattern):
                continue
            seen.add(pattern.pattern_id)
            found.append(pattern)
        if not found:
            continue
        entry_index = int(pd.to_datetime(daily.date).searchsorted(observed_date, side="right"))
        if entry_index >= len(daily):
            continue
        result.append({
            "signal_date": _iso(observed_date),
            "entry_date": _iso(daily.date.iloc[entry_index]),
            "pattern_ids": sorted(pattern.pattern_id for pattern in found),
            "pattern_types": sorted({pattern.pattern_type for pattern in found}),
            "pattern_labels": sorted({v16_pattern_labels.get(pattern.pattern_type, pattern.pattern_type)
                                       for pattern in found}),
            "pattern_tags": [{"pattern_id": pattern.pattern_id, "pattern_type": pattern.pattern_type,
                              "label": v16_pattern_labels.get(pattern.pattern_type, pattern.pattern_type)}
                             for pattern in sorted(found, key=lambda item: (item.pattern_type, item.pattern_id))],
        })
    return result


def _prefix_signals(daily, timeframe, code, kind, calendar, trade_start=None,
                    strategy_version=STRATEGY_VERSION_V2):
    if strategy_version == STRATEGY_VERSION_V1:
        return _prefix_signals_v1(daily, timeframe, code, kind, calendar, trade_start)
    if strategy_version != STRATEGY_VERSION_V2:
        raise ValueError("unsupported fixed strategy version: %s" % strategy_version)
    daily = daily.copy().reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily.date)
    result = []
    seen = set()
    first_signal_index = int(pd.to_datetime(daily.date).searchsorted(pd.Timestamp(trade_start), side="left")) if trade_start else 30
    for index in range(max(30, first_signal_index), len(daily)):
        background = daily.iloc[:index + 1].copy()
        if timeframe == "1w":
            signal_frame = aggregate_weekly(background, calendar=calendar)
            window = 520
        else:
            signal_frame = background
            window = 500
        if len(signal_frame) < 8:
            continue
        observed_date = pd.Timestamp(signal_frame.date.iloc[-1])
        structure_window = signal_frame.tail(window).copy().reset_index(drop=True)
        main_events = _main_bottom_detector().detect_main_bottoms(
            signal_frame, code=code, kind=kind, timeframe=timeframe,
            **({"background_bars": background} if timeframe == "1w" else {}))
        observed_iso = _iso(observed_date)
        found = _eligible_main_patterns(main_events, observed_iso)
        found = [pattern for pattern in found if pattern.pattern_id not in seen]
        for pattern in found:
            seen.add(pattern.pattern_id)
        if not found:
            continue
        entry_index = int(pd.to_datetime(daily.date).searchsorted(observed_date, side="right"))
        if entry_index >= len(daily):
            continue
        result.append({
            "signal_date": _iso(observed_date),
            "entry_date": _iso(daily.date.iloc[entry_index]),
            "pattern_ids": sorted(pattern.pattern_id for pattern in found),
            "pattern_types": sorted({pattern.pattern_type for pattern in found}),
            "pattern_labels": sorted({PATTERN_LABELS.get(pattern.pattern_type, pattern.pattern_type)
                                       for pattern in found}),
            "pattern_tags": [_pattern_audit_tag(pattern) for pattern in
                             sorted(found, key=lambda item: (item.pattern_type, item.pattern_id))],
        })
    return result


def _default_split_date(bars, development_cutoff=None):
    if development_cutoff:
        return _iso(development_cutoff)
    dates = pd.to_datetime(bars.date).drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return None
    return _iso(dates.iloc[min(len(dates) - 1, math.floor(len(dates) * 0.7))])


def run_fixed_backtest(bars, code, kind, timeframe="1d", cash=STARTING_CASH,
                       calendar=None, development_cutoff=None, trade_start=None,
                       strategy_version=STRATEGY_VERSION_V2):
    """Run one isolated security/timeframe fixed-risk account."""
    if strategy_version not in SUPPORTED_STRATEGY_VERSIONS:
        raise ValueError("unsupported fixed strategy version: %s" % strategy_version)
    if not math.isclose(float(cash), STARTING_CASH, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("%s requires exactly %.0f initial cash per account" % (strategy_version, STARTING_CASH))
    daily = bars.copy().reset_index(drop=True)
    if daily.empty:
        sim = simulate_fixed_trades(daily, [], cash=cash)
        result = _result_payload(sim, code, kind, timeframe, cash, [], None, strategy_version)
        result.update({"data_start": None, "trade_start": _iso(trade_start) if trade_start else None,
                       "evaluation_start": None, "data_end": None})
        return result
    daily["date"] = pd.to_datetime(daily.date)
    daily = daily.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    data_start = _iso(daily.date.iloc[0])
    if trade_start:
        evaluation_index = int(pd.to_datetime(daily.date).searchsorted(pd.Timestamp(trade_start), side="left"))
        evaluation_daily = daily.iloc[evaluation_index:].reset_index(drop=True)
    else:
        evaluation_daily = daily.copy()
    evaluation_start = _iso(evaluation_daily.date.iloc[0]) if not evaluation_daily.empty else None
    split_date = _default_split_date(daily, development_cutoff)
    signals = _prefix_signals(daily, timeframe, code, kind,
                              calendar if calendar is not None else default_china_calendar(),
                              trade_start=trade_start, strategy_version=strategy_version)
    for signal in signals:
        signal["split"] = "development" if split_date is None or signal["signal_date"] < split_date else "historical_holdout_exposed"
    sim = simulate_fixed_trades(evaluation_daily, signals, cash=cash)
    result = _result_payload(sim, code, kind, timeframe, cash, signals, split_date, strategy_version)
    result["data_start"] = data_start
    result["trade_start"] = _iso(trade_start) if trade_start else data_start
    result["evaluation_start"] = evaluation_start
    result["data_end"] = _iso(daily.date.iloc[-1])
    return result


def _result_payload(sim, code, kind, timeframe, starting_cash, signals, split_date,
                    strategy_version=STRATEGY_VERSION_V2):
    metrics = calculate_metrics(sim["trades"], starting_cash, sim["ending_equity"], sim["daily_equity"])
    by_split = {}
    for label in ("development", "historical_holdout_exposed"):
        selected = [trade for trade in sim["trades"] if trade.get("split") == label]
        # Isolated account equity is reported only at whole-series level. Split
        # return is normalized to the account's initial cash without pretending
        # the slices were separately compounded accounts.
        net = sum(float(trade["pnl"]) for trade in selected)
        by_split[label] = {"trade_count": len(selected), "net_pnl": net,
                           "return_on_initial_cash": net / float(starting_cash)}
    assumptions = {"initial_cash_per_security_timeframe": float(starting_cash),
                   "commission": 0.0, "slippage": 0.0, "lot_size": LOT_SIZE,
                   "stop_fraction": STOP_FRACTION, "target_fraction": TARGET_FRACTION,
                   "stop_target_from_actual_entry": True, "t_plus_one": True,
                   "weekly_requires_completed_bars": True,
                   "annualization": "CAGR over observed trade period; less than one year remains labeled in metrics",
                   "capital_returns_are_independent_and_must_not_be_summed": True}
    if strategy_version == STRATEGY_VERSION_V2:
        assumptions.update({"bottom_signal_source": "causal_selected_main_parent",
                            "bottom_signal_gates": ["selected_main_parent", "six_calendar_month_structure_gate",
                                                    "close_above_frozen_neckline_by_max_two_ticks_or_previous_atr14"],
                            "legacy_detector_fallback": False,
                            "supported_pattern_types": sorted(MAIN_BOTTOM_TYPES)})
    return {
        "strategy_version": strategy_version,
        "engine": sim.get("engine", "backtrader_broker"),
        "code": str(code), "kind": str(kind), "timeframe": str(timeframe),
        "starting_cash": float(starting_cash), "ending_equity": float(sim["ending_equity"]),
        "signal_count": len(signals), "signals": signals, "trades": sim["trades"],
        "rejections": sim["rejections"], "order_events": sim.get("order_events", []),
        "daily_equity": sim["daily_equity"],
        "metrics": metrics,
        "pattern_migration": migration_status(),
        "split": {"development_cutoff": split_date, "method": "chronological_70_30_frozen_cutoff",
                  "holdout_status": "historical_development_analysis; not an independent untouched holdout",
                  "groups": by_split},
        "assumptions": assumptions,
    }
