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
