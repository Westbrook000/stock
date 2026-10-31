# Module Specification: history-replay-retirement

Status: Capability-map scope accepted by the user on 2026-10-27. Approved and implemented on 2026-10-27. Verified with the current automated suite and local browser.

## Objective

Remove the user-facing historical replay page and all arbitrary historical cutoff (`asof` / `as_of`) analysis functionality and dependent tools. The system analyzes the latest available local data; it does not return a user-requested historical analysis JSON payload.

## Scope

- Remove the “历史回放” navigation and panel, replay button/output, detail-page as-of date input, replay click handlers, copy and examples.
- Remove the `asof` query parameter from `GET /api/v1/analysis` and remove any replay-only endpoint/helper.
- Remove arbitrary cutoff parameters and date truncation from market loading, weekly aggregation, pattern detection and result generation (`as_of`, `asof`). Data dates and pattern occurrence/known/confirmation/terminal dates remain part of ordinary current analysis.
- Remove historical replay, replay-only fixture data, calibration/evaluation consumers and tests that exist only to calculate a user-specified historical cutoff. Calibration consumers and artifacts are retired by `manual-review-retirement` before this module removes their provider logic.
- Remove code and documentation whose only purpose is the replay UI or the as-of analysis flow. Do not leave dormant API routes or unused replay helpers.
- Preserve Backtrader simulation and its explicitly configured date range if used for simulated execution; that range controls a simulation study, not the removed point-in-time analysis feature.
- Preserve prefix-based causality safeguards: run the detector on progressively longer prefixes and assert that adding future bars does not alter already-known earlier anchors/confirmation. These checks do not expose a historical cutoff to users.
- Weekly completeness is evaluated against the live exchange calendar and available bars, without a caller-provided historical as-of date.
- Preserve current data-version provenance and the chart-navigation feature for ended patterns.

## Project structure and interfaces

- UI: `caisen/static/index.html`, `app.js`, `app.css`.
- API: `caisen/api.py` and any schemas or helpers used solely by arbitrary historical analysis.
- Data/pattern APIs: `caisen/market.py`, `caisen/patterns.py`, weekly calendar functions, and callers.
- Remove replay-only README/spec references and fixtures. Keep current analysis, scan, detail, pattern lifecycle, calendar and Backtrader simulation features.
- Commands: `.venv/bin/python -m pytest caisen-system/tests`; `.venv/bin/python caisen-system/scripts/serve.py`.

## Testing strategy

- Assert `/api/v1/analysis` no longer accepts `asof` and returns latest-data analysis only.
- Assert the app has no replay page, replay action, or historical cutoff input.
- Search backend and frontend runtime code for replay-only cutoff arguments and prove the old flow has no remaining route/helper.
- Preserve tests for weekly-calendar completeness and prefix-based causality; remove tests whose only purpose is user-selected historical cutoff replay.
- Run the full Caisen suite and a local browser smoke of scan → detail → ended-pattern chart selection.

## Boundaries

- Do not remove event dates, known dates, historical ended pattern records or their chart evidence; they are required for current pattern history.
- Do not remove Backtrader simulation date ranges or automated prefix-causality safeguards.
- Do not add any replacement JSON replay output.

## Success criteria

- No user-facing history-replay function or `asof`-cutoff analysis API/code remains.
- Users can still inspect ended shapes on the current chart and use Backtrader simulation.
- Automated causality checks still prove that future bars cannot rewrite earlier known pattern events.

## Open questions

- None. This draft interprets “删除全部 as-of 逻辑” as deleting arbitrary point-in-time analysis cutoffs, while retaining event dates, weekly live-calendar checks, simulation ranges and prefix-causality verification.
