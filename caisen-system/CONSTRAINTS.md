# Caisen system quality constraints

Last reviewed: 2026-10-27. User decision: additional quality dimensions are recorded but do not block work.

## Always enforced

- Every behavior change has a focused regression test written before implementation.
- The focused tests and complete project test suite must pass before reporting the work as complete.
- Never delete, skip, or weaken an existing assertion to make a check pass without recording the reason and scope here.
- Preserve causal time boundaries: a pattern detected from an input prefix may use only bars in that prefix; appending future bars must not rewrite already-known pivots or events.
- Human-label calibration tools and records are retired; do not add blind-review pages, export human labels, or claim accuracy from deleted calibration records.
- Do not trigger full-universe network refreshes as a build or verification step.

## Enforced checks

| Dimension | Rule | Check | When |
|---|---|---|---|
| Regression behavior | All tests pass; new behavior has focused tests | `.venv/bin/python -m pytest -q caisen-system/tests` | Task end |
| Browser script syntax | Changed JavaScript parses | `node --check <changed-file.js>` | After frontend changes |
| Calibration retirement | No human-label calibration runtime, UI, or stored label artifacts remain after the approved migration | `SPEC-manual-review-retirement.md` | Any future change to pattern-evaluation workflow |

## Measured, not enforced

| Metric | Current baseline | Direction |
|---|---|---|
| Full test count | 132 passing after approved feature retirement | Must not decrease by deleting or skipping tests |
| Coverage | Not currently collected | Measure before setting a gate |
| External security scan | Not configured | Record findings if run; do not block work |
| Performance / accessibility | Not currently measured | Record findings; do not block work |

## Exceptions

2026-10-27 user-approved feature retirement: remove tests exclusively covering deleted manual review, calibration, arbitrary cutoff replay, refresh recovery, old refresh hash/coverage validation, and pre-scan filters. Replace retained behavior assertions with current API/calendar/ambiguity tests; preserve prefix-causality and trading arithmetic assertions. Test-count decrease from retiring these features is authorized and is not evidence of lower detection quality.

2026-10-29 user-approved scope exception: create a new independent 120-case blind pattern review and preserve its newly generated labels and artifacts for that review only. This does not authorize restoring deleted historical calibration records, the retired calibration tools, or the old review UI.

2026-10-29 user-approved performance change: replace the scan's 10-thread analysis pool with up to 10 spawn-based processes after a same-instrument benchmark showed threads slower than serial and processes materially faster. Pattern rules and result scope stay unchanged; the parent process retains all SQLite writes. Spawn workers receive the market root explicitly and may use materially more memory, so record process-tree RSS in the performance report and do not claim a full-market runtime from the 10-instrument sample.

2026-11-01 user-approved strategy retirement: keep only daily four-shape, next-open, actual-fill 8% stop / 20% target, zero-cost, 100,000-per-symbol Backtrader execution. Retire executable baseline-v1, old main-parent-v2 strategy selection, weekly backtest, 7% structural-risk / 2R entry gates, structural stops and measured-target exits. Tests exclusively asserting those retired behaviors must be replaced by explicit rejection or fixed-policy assertions; do not remove causal-prefix, geometry or ledger invariants. Historical reports remain readable. Daily recognition now uses continuous ATR/pivot state with a five-year geometry window; record differences from old window-reset calculations. Multi-threading is explicitly requested for this backtest, without changing market-scan multiprocessing.

2026-11-04 user-approved candidate pressure-pivot change: confirm candidate resistance pivots only after a causal close reversal of 2 ATR, in both batch and incremental recognition. This changes candidate dates and potentially pattern/signal dates; retain exact 1 ATR reference expectations as explicit reference tests where outputs differ, and add exact 2 ATR outcomes with the cause recorded. Independent neckline-test departure (1 ATR), breakout buffer (1 prior ATR), and formation coverage rules remain unchanged. Tag new candidate pressure points with the configured reversal scale; untagged structural pivots remain usable for geometry only. Bump rule/engine metadata so old performance qualifications cannot silently certify this rule version; preserve archived reports.

2026-11-05 user-approved recross correction: a first buffered close above a frozen neckline consumes that same complete structure/boundary phase. If it occurs before the structure's maturity date, it is permanently rejected and a later recross of the equivalent frozen line cannot produce a buy; if it is the first eligible mature breakout, retain that confirmation and suppress repeat signals from the same phase. Distinct supported pressure lines and new complete structure anchors remain eligible. The corrected 600108 first-1200-bar regression intentionally changes from five v8 outputs to two: 1998-03-13 repeats the 1998-02-25 triangle boundary; 1999-06-28 is the first buffered cross before its six-month maturity; 1999-12-07 recrosses a boundary first crossed on 1999-06-21 before maturity. Retain 1998-02-25 and 2002-06-24. The 159718 direct-session regression uses production ETF tick 0.01 and consumes the 2023-09-22 / 2024-02-05 / 2024-11-22 IHS phase on 2025-04-01 (line slope -0.00024299065, intercept 0.91256075, 0.02 tolerance); the 2025-05-21 boundary projects within that frozen tolerance at the consumed phase's comparison indices and is suppressed in continuous streaming. Retain the distinct 2025-04-01 IHS and 2025-05-29 W. The bounded `detect_main_bottoms` batch replay still emits its 2025-05-21 IHS as the first buffered cross in its own prefix/ATR context; this pre-existing batch/stream history difference is documented and not forced into payload parity. Historical evidence from consumed phases is required; do not weaken the existing geometry, prefix-causality, weak-crossing, or batch/stream assertions.
