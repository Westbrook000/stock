"""Local Python port of ZigZag 0.2.2, the version required by day0.

Reference: https://github.com/jbn/ZigZag/blob/0.2.2/zigzag/core.pyx
Copyright (c) 2014, John Bjorn Nelson; BSD-3-Clause, see
third_party/ZIGZAG_LICENSE.txt. Scalar thresholds follow the upstream state
machine including initial/terminal annotation. Per-extreme thresholds and
confirmation timestamps are application extensions.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

PEAK, VALLEY = 1, -1


@dataclass(frozen=True)
class Pivot:
    price_idx: int
    price: float
    kind: str
    confirm_idx: Optional[int]
    confirmed: bool = True


def _inputs(close, up_thresh, down_thresh):
    close = np.asarray(close, dtype=float)
    if close.ndim != 1 or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError('close must be finite positive 1-D prices')
    def aligned(value):
        result = np.full(len(close), float(value)) if np.isscalar(value) else np.asarray(value, dtype=float)
        if result.shape != close.shape or not np.isfinite(result).all():
            raise ValueError('threshold must be finite and align with close')
        return result
    up, down = aligned(up_thresh), aligned(down_thresh)
    if (up <= 0).any() or (down >= 0).any() or (down <= -1).any():
        raise ValueError('up_thresh must be positive; down_thresh must be in (-1,0)')
    return close, up, down


def _initial(close, up, down):
    first = high = low = close[0]
    high_idx = low_idx = 0
    for t in range(1,len(close)):
        price = close[t]
        if price/low >= 1+up[low_idx]:
            return (VALLEY if low_idx == 0 else PEAK), t
        if price/high <= 1+down[high_idx]:
            return (PEAK if high_idx == 0 else VALLEY), t
        if price > high:
            high,high_idx = price,t
        if price < low:
            low,low_idx = price,t
    return (VALLEY if first < close[-1] else PEAK), None


def identify_initial_pivot(close, up_thresh, down_thresh):
    values,up,down = _inputs(close,up_thresh,down_thresh)
    if not len(values):
        raise ValueError('cannot classify an empty series')
    return _initial(values,up,down)[0]


def _scan(close, up_thresh, down_thresh):
    values,up,down = _inputs(close,up_thresh,down_thresh)
    marks = np.zeros(len(values),dtype=int)
    confirmations = {}
    if not len(values):
        return values,marks,confirmations
    initial,initial_known = _initial(values,up,down)
    marks[0] = initial
    # The forced initial label becomes stable only once the initial direction is known.
    if initial_known is not None:
        confirmations[0] = (initial,initial_known)
    trend,last_idx,last_price = -initial,0,values[0]
    for t in range(1,len(values)):
        price = values[t]
        ratio = price/last_price
        if trend == VALLEY:
            if ratio >= 1+up[last_idx]:
                marks[last_idx] = trend
                confirmations[last_idx] = (trend,t)
                trend,last_price,last_idx = PEAK,price,t
            elif price < last_price:
                last_price,last_idx = price,t
        else:
            if ratio <= 1+down[last_idx]:
                marks[last_idx] = trend
                confirmations[last_idx] = (trend,t)
                trend,last_price,last_idx = VALLEY,price,t
            elif price > last_price:
                last_price,last_idx = price,t
    # Exact 0.2.2 endpoint rules, including the one-observation case.
    if last_idx == len(values)-1:
        marks[last_idx] = trend
    elif marks[-1] == 0:
        marks[-1] = -trend
    return values,marks,confirmations


def peak_valley_pivots(close, up_thresh, down_thresh):
    """Upstream-compatible labels: +1 peak, -1 valley, zero otherwise."""
    return _scan(close,up_thresh,down_thresh)[1]


def day0_pivots(close, threshold) -> List[Pivot]:
    """Full upstream chart pivots, with confirmation metadata added."""
    down = -float(threshold) if np.isscalar(threshold) else -np.asarray(threshold,dtype=float)
    values,marks,confirmations = _scan(close,threshold,down)
    result = []
    for idx in np.flatnonzero(marks):
        known = confirmations.get(int(idx))
        confirmed = bool(known is not None and known[0] == marks[idx])
        result.append(Pivot(int(idx),float(values[idx]),'high' if marks[idx] == PEAK else 'low',
                            known[1] if confirmed else None,confirmed))
    return result


def filter_pivots(pivots, min_bars_between_peaks=0, peaks='All'):
    """day0 BaseZigZagLevels: side filter first, then chronological greedy spacing."""
    if peaks not in ('All','High','Low'):
        raise ValueError('peaks must be All, High, or Low')
    if (isinstance(min_bars_between_peaks,bool)
            or not isinstance(min_bars_between_peaks,(int,float,np.integer,np.floating))
            or not np.isfinite(min_bars_between_peaks) or min_bars_between_peaks < 0
            or min_bars_between_peaks != int(min_bars_between_peaks)):
        raise ValueError('min_bars_between_peaks must be a nonnegative integer')
    min_bars_between_peaks = int(min_bars_between_peaks)
    selected = sorted((p for p in pivots if peaks == 'All' or p.kind == ('high' if peaks == 'High' else 'low')),
                      key=lambda p:p.price_idx)
    if not min_bars_between_peaks:
        return selected
    result = []
    for pivot in selected:
        if not result or pivot.price_idx-result[-1].price_idx >= min_bars_between_peaks:
            result.append(pivot)
    return result


def confirmed_close_pivots(close, threshold, min_bars_between_peaks=0, peaks='All') -> List[Pivot]:
    return [p for p in filter_pivots(day0_pivots(close,threshold),min_bars_between_peaks,peaks) if p.confirmed]


def pivots_known_by(confirm_bar,pivots):
    return [p for p in pivots if p.confirmed and p.confirm_idx is not None and p.confirm_idx <= confirm_bar]
