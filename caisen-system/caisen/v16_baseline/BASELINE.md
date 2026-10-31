# Frozen v16 pattern baseline

This package is a byte-for-byte copy of the dirty user working tree captured on
2026-10-29 before this task changed the long-cycle analyzer. It is not copied
from Git HEAD. Use `caisen.v16_baseline.patterns.detect_patterns` for same-cutoff
comparisons. Its local dependencies and active rule catalog are frozen beside
it so later engine changes do not silently alter the baseline.
