# Restore day0 ZigZagClusterLevels parameters and ZigZag 0.2.2 semantics

用户明确要求保留原版参数、在strategy.yaml中配置、ZigZag与day0一致，可保留自适应阈值。

- Canonical ZigZag options: peak_percent_delta=5.0 (percent points), min_bars_between_peaks=0, peaks=All. Retain adaptive/atr_mult/min_threshold/max_threshold as existing extensions. Legacy threshold fraction remains a Python/YAML compatibility alias.
- Support options: merge_distance=null, merge_percent=0.5, level_selector=median. Require exactly one merge mode. lookback_bars=0 is the existing application window extension.
- Port exactly day0's pinned ZigZag==0.2.2 core state transitions and first/last annotations into the project. Numeric inputs remain positive finite close prices. Adaptive mode substitutes candidate-index thresholds for the upstream scalar, preserving the state transitions.
- Apply High/Low/All before the original greedy time-spacing filter, outside core ZigZag calculation.
- Keep full endpoint-inclusive snapshots for plotting. Add confirmation metadata; event support inputs exclude provisional endpoints. Initial endpoint classification becomes stable when initial direction is established. Match snapshot cluster knowledge times to actual causal versions; otherwise mark provisional.
- Bump strategy version so old cached scans/backtests are not reused. Update provenance, BSD license, documentation, chart metadata and YAML comments.
- No added or executed tests and no 300-symbol backtest in this task; render the existing 515450 chart to exercise the configured path.
