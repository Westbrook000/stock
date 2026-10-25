from typing import List

import numpy as np
import pandas as pd

from .indicators import atr
from .schemas import Pivot


def detect_pivots(bars: pd.DataFrame, radius: int = 2, atr_multiple: float = 1.0) -> List[Pivot]:
    """Local swing points are published only after `radius` later bars."""
    if len(bars) < radius * 2 + 1:
        return []
    av = atr(bars, period=14)
    raw = []
    highs, lows = bars["high"].to_numpy(), bars["low"].to_numpy()
    for i in range(radius, len(bars) - radius):
        high_slice = highs[i-radius:i+radius+1]
        low_slice = lows[i-radius:i+radius+1]
        is_high = highs[i] == high_slice.max() and np.count_nonzero(high_slice == highs[i]) == 1
        is_low = lows[i] == low_slice.min() and np.count_nonzero(low_slice == lows[i]) == 1
        if is_high == is_low:
            continue
        kind, price = ("high", highs[i]) if is_high else ("low", lows[i])
        scale = float(av.iloc[i])
        if not np.isfinite(scale) or scale <= 0:
            continue
        raw.append((i, kind, float(price), scale))
    # Collapse adjacent same-direction extrema, then require meaningful swing distance.
    collapsed = []
    for candidate in raw:
        if collapsed and candidate[1] == collapsed[-1][1]:
            better = candidate[2] > collapsed[-1][2] if candidate[1] == "high" else candidate[2] < collapsed[-1][2]
            if better:
                collapsed[-1] = candidate
            continue
        if collapsed and abs(candidate[2] - collapsed[-1][2]) < max(candidate[3], collapsed[-1][3]) * atr_multiple:
            continue
        collapsed.append(candidate)
    result = []
    dates = pd.to_datetime(bars["date"])
    for i, kind, price, _ in collapsed:
        confirm = i + radius
        result.append(Pivot(i, str(dates.iloc[i].date()), price, kind, confirm,
                            str(dates.iloc[confirm].date()), "%gATR" % atr_multiple))
    return result
