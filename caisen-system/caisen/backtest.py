"""Compatibility import for the sole daily fixed-risk Backtrader strategy."""
from .fixed_backtest import run_fixed_backtest


def run_backtest(bars, code, kind, timeframe='1d', cash=100000, start=None, calendar=None):
    return run_fixed_backtest(bars, code, kind, timeframe=timeframe, cash=cash,
                              trade_start=start, calendar=calendar)
