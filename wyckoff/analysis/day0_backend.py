"""Local port of day0market/support_resistance's Ward/median level algorithm.

Source: pricelevels/cluster.py and BaseLevelFinder._get_distance, revision
fdd4a7f0fdccf6c4007da79fd6f3879164462c3c. Alex Hurko, WTFPL.
No clone import, network request, or runtime Git dependency. Membership metadata
is added for drawing only; it does not alter Ward clustering or aggregation.
"""
from functools import lru_cache
from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'fdd4a7f0fdccf6c4007da79fd6f3879164462c3c'


@lru_cache(maxsize=1)
def _clustering_class():
    try:
        module = importlib.import_module('sklearn.cluster')
    except ModuleNotFoundError:
        site = ROOT / '.venv' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
        if site.exists():
            sys.path.append(str(site))
        try:
            module = importlib.import_module('sklearn.cluster')
        except ModuleNotFoundError as exc:
            raise RuntimeError('Install requirements-day0.txt in the project .venv') from exc
    return module.AgglomerativeClustering


def _cluster(prices, distance, selector):
    """Port of upstream _cluster_prices_to_levels; also retain member positions."""
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('prices must be a finite positive 1-D array')
    if not np.isfinite(distance) or distance <= 0 or selector not in ('mean', 'median'):
        raise ValueError('Positive Ward distance and mean/median selector required')
    if not len(values):
        return []
    if len(values) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = _clustering_class()(distance_threshold=float(distance), n_clusters=None,
                                     linkage='ward', metric='euclidean').fit(values.reshape(-1, 1)).labels_
    frame = pd.DataFrame({'price': values, 'cluster': labels, 'peak_count': 1})
    grouped = frame.groupby('cluster').agg({'price': selector, 'peak_count': 'sum'}).reset_index()
    result = grouped.to_dict('records')
    for record in result:
        members = np.flatnonzero(labels == record['cluster'])
        record['members'] = members.tolist()
        record['bottom'] = float(values[members].min())
        record['top'] = float(values[members].max())
    return result


def cluster_prices(prices, distance, selector='median'):
    """Compatibility primitive; application entry below always uses merge_percent."""
    return _cluster(prices, distance, selector)


def resolve_merge_distance(close, merge_percent=None, merge_distance=None):
    values = np.asarray(close,dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('close must contain positive finite prices')
    if (merge_distance is None) == (merge_percent is None):
        raise ValueError('Set exactly one of merge_distance and merge_percent; set the other to null')
    parameter = merge_distance if merge_distance is not None else merge_percent
    if not np.isfinite(parameter) or parameter <= 0:
        raise ValueError('merge distance/percent must be positive')
    return float(merge_distance if merge_distance is not None else values.mean()*merge_percent/100)


def cluster_pivots(pivots, close, merge_percent=0.5, selector='median', merge_distance=None):
    """day0 merge_distance/merge_percent modes; membership is drawing metadata."""
    distance = resolve_merge_distance(close,merge_percent,merge_distance)
    records = _cluster([p.price for p in pivots], distance, selector)
    for record in records:
        members = [pivots[i] for i in record.pop('members')]
        record['first_pivot_idx'] = min(p.price_idx for p in members)
        known = [p.confirm_idx for p in members if p.confirm_idx is not None]
        record['first_confirm_idx'] = min(known) if known else None
        record['established_idx'] = max(known) if all(p.confirmed for p in members) else None
        record['provisional'] = not all(p.confirmed for p in members)
        record['pivot_indices'] = [p.price_idx for p in members]
        record['distance'] = distance
    return records
