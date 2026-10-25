# Caisen system quality constraints

Last reviewed: 2026-10-25. User decision: additional quality dimensions are recorded but do not block work.

## Always enforced

- Every behavior change has a focused regression test written before implementation.
- The focused tests and complete project test suite must pass before reporting the work as complete.
- Never delete, skip, or weaken an existing assertion to make a check pass without recording the reason and scope here.
- Preserve causal time boundaries: a historical label may use only bars available by its `asof` date.
- Human calibration labels are the reference; machine predictions and later returns must not leak into blind-review pages.
- Calibration outputs must state the local-data basis and may not be described as accuracy or strategy-performance evidence unless the sample design supports that claim.
- Do not trigger full-universe network refreshes as a build or verification step.

## Enforced checks

| Dimension | Rule | Check | When |
|---|---|---|---|
| Regression behavior | All tests pass; new behavior has focused tests | `.venv/bin/python -m pytest -q caisen-system/tests` | Task end |
| Browser script syntax | Changed JavaScript parses | `node --check <changed-file.js>` | After frontend changes |
| Calibration privacy | Blind pages omit machine pattern outputs and safely encode case data | `tests/test_calibration_pack.py` | Every edit to calibration packs |

## Measured, not enforced

| Metric | Current baseline | Direction |
|---|---|---|
| Full test count | 144 passing before this change | Must not decrease by deleting or skipping tests |
| Coverage | Not currently collected | Measure before setting a gate |
| External security scan | Not configured | Record findings if run; do not block work |
| Performance / accessibility | Not currently measured | Record findings; do not block work |

## Exceptions

None.
