# day0 support_resistance local port

- Source: https://github.com/day0market/support_resistance
- Reference revision: fdd4a7f0fdccf6c4007da79fd6f3879164462c3c
- Author/license declared in upstream setup.py: Alex Hurko / WTFPL.
- Project implementation: wyckoff/analysis/day0_backend.py. Ported Ward AgglomerativeClustering, pandas groupby median/mean and peak counts, plus percentage-to-distance conversion from BaseLevelFinder._get_distance.
- Runtime uses local sklearn/scipy/pandas/numpy. It does not import the cloned source, run Git, or contact a remote API. The clone and earlier optional-import patch are retained as historical source references only.
- Install local numerical dependencies: python3 -m venv --system-site-packages .venv; .venv/bin/python -m pip install --no-deps -r requirements-day0.txt. setup_day0.py is optional source retrieval, not a runtime prerequisite.
- merge_percent is in percent points: 0.5 means 0.5%; Ward threshold = sample mean close × merge_percent / 100. Ward linkage threshold is not a maximum cluster diameter.
- Added metadata retains member indices, minimum/maximum pivot price and earliest pivot for plotting. These additions do not modify cluster assignment or central price aggregation.
- Upstream defaults retained: all pivot types, no minimum-bars filter, median selector; no additional minimum cluster size. Full confirmed history is the default sample. A singleton is handled explicitly because sklearn requires at least two samples.
- The causal trading adapter excludes the upstream zigzag library's forced first/unconfirmed terminal endpoints and preserves the user-requested ATR-adaptive threshold. This is an explicit difference from an offline whole-chart ZigZag fit. Current snapshot drawings extend to the earliest constituent pivot, with a dotted pre-knowledge interval; trading never uses that retrospective extension.

## v3: ZigZag 0.2.2 compatibility and original options

This section supersedes the earlier endpoint/parameter differences above.

- All ZigZagClusterLevels options are configurable: peak_percent_delta (percent points), min_bars_between_peaks (default 0), peaks (All/High/Low), merge_distance or merge_percent, level_selector (median/mean). The two merge modes are mutually exclusive; unused option must be null. Percentage remains the default.
- day0 pins ZigZag==0.2.2. Its core.pyx state machine, initial classifier and first/last annotation are ported into local zigzag.py. Source: https://github.com/jbn/ZigZag/blob/0.2.2/zigzag/core.pyx . Copyright (c) 2014, John Bjorn Nelson; BSD-3-Clause, notice retained in ZIGZAG_LICENSE.txt.
- Scalar, non-adaptive mode follows 0.2.2. Adaptive candidate-index thresholds and confirmation timestamps are application extensions. Initial direction becomes usable after the first threshold establishes it; unstable initial fallback and unconfirmed terminal points are excluded from trading.
- Offline chart snapshots include the original endpoint labels. Side filtering precedes chronological greedy spacing, as in day0. Snapshot clusters absent from the current causal fit are marked provisional and never substituted into event detection.
