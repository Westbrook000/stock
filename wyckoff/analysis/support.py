"""Causal horizontal day0 clusters with local percentage-based Ward fitting."""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from .day0_backend import cluster_pivots, resolve_merge_distance
from .zigzag import Pivot


@dataclass
class Level:
    line_id: int
    price: float
    peak_count: int
    known_from: int
    kind: str
    valid_to: Optional[int] = None
    first_pivot_idx: int = 0
    first_confirm_idx: Optional[int] = 0
    bottom: float = 0.0
    top: float = 0.0
    pivot_indices: Tuple[int, ...] = ()
    provisional: bool = False

    @property
    def active(self):
        return self.valid_to is None


@dataclass
class SupportSeries:
    support: np.ndarray
    resistance: np.ndarray
    support_id: np.ndarray
    resistance_id: np.ndarray
    support_count: np.ndarray
    resistance_count: np.ndarray
    levels: List[Level]

    @staticmethod
    def compute(close, atr, pivots: List[Pivot], cfg):
        # atr retained for callers; day0 supports mutually exclusive absolute/percent modes.
        close = np.asarray(close, dtype=float)
        if cfg.lookback_bars < 0:
            raise ValueError('lookback_bars must be nonnegative; 0 means full history')
        n = len(close)
        support, resistance = np.full(n, np.nan), np.full(n, np.nan)
        sup_id, res_id = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
        sup_count, res_count = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        by_confirm = {}
        for p in pivots:
            if p.confirmed and p.confirm_idx is not None:
                by_confirm.setdefault(p.confirm_idx, []).append(p)
        known, active, versions = [], [], []
        previous_key, next_id = None, 0
        for t in range(n):
            known.extend(by_confirm.get(t, []))
            start = max(0, t-cfg.lookback_bars+1) if cfg.lookback_bars else 0
            eligible = [p for p in known if p.price_idx >= start]
            window = close[start:t+1]
            distance = resolve_merge_distance(window,cfg.merge_percent,cfg.merge_distance)
            key = (tuple((p.price_idx,p.kind) for p in eligible), distance)
            if key != previous_key:
                records = cluster_pivots(eligible, window, cfg.merge_percent, cfg.level_selector, cfg.merge_distance)
                old, active, used = list(active), [], set()
                for record in sorted(records, key=lambda x:x['price']):
                    members = tuple(record['pivot_indices'])
                    candidates = [x for x in old if x.line_id not in used
                                  and set(x.pivot_indices).intersection(members)]
                    match = min(candidates, key=lambda x:(-len(set(x.pivot_indices).intersection(members)),
                                                          abs(x.price-record['price']))) if candidates else None
                    if match and match.pivot_indices == members and match.price == record['price']:
                        line = match
                    else:
                        if match:
                            line_id = match.line_id
                        else:
                            line_id, next_id = next_id, next_id+1
                        line = Level(line_id=line_id, price=float(record['price']),
                                     peak_count=int(record['peak_count']), known_from=t,
                                     kind='support' if record['price']<=close[t] else 'resistance',
                                     first_pivot_idx=record['first_pivot_idx'],
                                     first_confirm_idx=record['first_confirm_idx'],
                                     bottom=record['bottom'], top=record['top'], pivot_indices=members)
                        versions.append(line)
                    used.add(line.line_id)
                    active.append(line)
                kept = {id(line) for line in active}
                for line in old:
                    if id(line) not in kept:
                        line.valid_to = t-1
                previous_key = key
            below = [x for x in active if x.price <= close[t]]
            above = [x for x in active if x.price > close[t]]
            if below:
                best = max(below,key=lambda x:x.price)
                support[t],sup_id[t],sup_count[t] = best.price,best.line_id,best.peak_count
            if above:
                best = min(above,key=lambda x:x.price)
                resistance[t],res_id[t],res_count[t] = best.price,best.line_id,best.peak_count
        return SupportSeries(support,resistance,sup_id,res_id,sup_count,res_count,versions)


    @staticmethod
    def snapshot(close, pivots, cfg):
        """Exact day0 offline fit inputs, including its annotated terminal endpoint.

        Snapshot levels are for rendering only. compute() independently keeps
        confirmation timestamps, so a provisional endpoint never enters events.
        """
        n = len(close)
        start = max(0,n-cfg.lookback_bars) if cfg.lookback_bars else 0
        eligible = [p for p in pivots if p.price_idx >= start]
        records = cluster_pivots(eligible,close[start:],cfg.merge_percent,cfg.level_selector,cfg.merge_distance)
        return [Level(line_id=i,price=float(r['price']),peak_count=int(r['peak_count']),
                      known_from=r['established_idx'] if r['established_idx'] is not None else n,
                      kind='support' if r['price']<=close[-1] else 'resistance',
                      first_pivot_idx=r['first_pivot_idx'],first_confirm_idx=r['first_confirm_idx'],
                      bottom=r['bottom'],top=r['top'],pivot_indices=tuple(r['pivot_indices']),
                      provisional=r['provisional'])
                for i,r in enumerate(sorted(records,key=lambda r:r['price']))]
