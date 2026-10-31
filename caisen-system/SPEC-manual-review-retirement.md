# Module Specification: manual-review-retirement

Status: Capability-map scope accepted by the user on 2026-10-27. Approved and implemented on 2026-10-27. Verified with the current automated suite and local browser.

## Objective

Remove the application's manual pattern-review workflow and all human-label calibration tools and artifacts. The running product will not ask users to accept/reject/correct machine patterns or provide a blind-label calibration process.

## Scope

- Remove the “人工复核” navigation item and page, pattern-card review buttons, review inputs and save behavior.
- Remove `PUT /api/v1/patterns/{pattern_id}/review`, its request schema, review persistence functions and the `reviews` table/data.
- Remove standalone blind-calibration pack generation and evaluation commands/modules, calibration pages, labels, clarifications, machine-review reports and tests that exist only for this workflow.
- Remove calibration-specific instructions and claims from README, capability/spec documents and constraints. Do not claim recognition accuracy from the deleted labels.
- Preserve current active pattern rules and parameter values. Describe engineering thresholds as fixed rule-version parameters, not as statistically calibrated values. Do not change pattern geometry as part of this retirement.
- Preserve automated prefix-causality tests that do not require human labels or a user-selected historical cutoff.
- Keep source-book rule cards, source citations and machine pattern evidence; these are not human review records.
- Project-local user annotations/calibration answer files and corresponding SQLite rows are in the deletion scope. Do not delete them until this specification is approved and the actual cleanup step is authorized at action time. The user also authorized deleting /Users/sotawest/Downloads/caisen-round-01-v2-answers.json.

## Ambiguity handling after review removal

- A pattern with competing/shared topology must not display a dead-end “待人工复核” prompt or expose a review action.
- Preserve the existing buy-safety behavior: ambiguous structures remain visible with a clear structural-ambiguity warning and produce `WAIT`; they do not generate a buy plan until the ambiguity is resolved by existing deterministic rules.
- Remove review-specific wording from labels and explanations while retaining understandable reasons such as “结构点存在多种形态解释，暂不生成买入计划”.

## Project structure and interfaces

- Frontend: `caisen/static/index.html`, `app.js`, `app.css`.
- API and persistence: `caisen/api.py`, `schemas.py`, `store.py`.
- Calibration-only code/data: `caisen/calibration.py`, `caisen/calibration_pack.py`, `scripts/create_blind_calibration_round.py`, `scripts/evaluate_calibration_labels.py`, `calibration/` and tests solely for those modules.
- Preserve unrelated Backtrader and market/pattern tests. Existing fixtures that assert the presence of a user-review API/table must be removed or replaced with retirement-contract assertions.

## Testing strategy

- Assert the UI has no manual-review navigation, review button, form, or save handler.
- Assert OpenAPI has no pattern-review route and storage initialization does not create a `reviews` table.
- Assert competing topology continues to return a wait/no-buy result with an ambiguity explanation, without human review.
- Assert no calibration generator/evaluator command or runtime module remains and no user label/report artifact remains after the approved deletion step.
- Re-run the pattern and decision suites to ensure rule recognition and buy gates are unchanged.
- Commands: `.venv/bin/python -m pytest caisen-system/tests`; inspect `caisen-system/` for calibration-only artifacts and stale user-review routes/labels.

## Boundaries

- Always preserve the active ten-rule catalog, source audit and current rule parameters unless a later approved spec changes them.
- Never present historical human-label agreement as verified accuracy after the data is removed.
- Retain pattern lifecycle/causality state required by current analysis and chart navigation; this is not a manual review feature.

## Success criteria

- No user-facing manual review or calibration workflow remains in the app, API, backend code, project instructions, or runtime tests.
- Requested human annotation and calibration records are removed in the approved implementation, while current pattern recognition and trade decision behavior remain covered by automated tests.
- The project no longer depends on human labels to explain or execute a live pattern decision.

## Open questions

- None. The user explicitly included the original Downloads answer file in the approved deletion scope.
- The implementation must perform project-local data deletion only after the specification and implementation plan are approved.
