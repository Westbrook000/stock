# Module Specification: analysis-workbench

Status: Base scope approved for implementation. Amendments approved by the user on 2026-10-27; implementation authorized.

## Primary flow

1. User clicks “全量扫描”. The app scans every current local stock and ETF CSV once, across selected daily/weekly timeframes, and checks all ten active shapes. Shape selections do not narrow or resubmit the scan. The app shows asynchronous progress and completion totals.
2. Completed results are merged to one row per instrument, sorted by decision-engine keys. `CONSIDER_BUY` rows passing the 7% risk / 2R gates appear first. The sort keys and reason are visible; the ordering is a screening priority, never a probability.
3. Each row shows code/name, action, current pattern/state, latest close/date, data warning, and a compact recent candlestick thumbnail.
4. Clicking a row or thumbnail opens the detail view for that code and loads its latest analysis without a second search. Detail shows the buy/sell/wait decision, daily and weekly evidence, current shape labels such as `W底，未确认`, chart anchors/boundaries, confirmation/invalidity, stop/targets, quality/provenance, rule/data versions and reasons. Clicking an ended shape in the detail history immediately overlays that event's geometry on its matching timeframe chart and moves the visible window to the event.
5. After the scan completes, the user can filter the returned rows by action and any of the ten active shapes. A prominent “可考虑买入” quick filter and combined action/shape filters operate client-side; default remains the full buy-first result order and changing filters does not resubmit the scan. Shape filtering is result presentation only and never changes the scan universe or analysis performed.

## Scan contract

- Every full scan inventory is all local stocks plus all local ETFs from the file set observed at scan start; no scope or shape selector narrows the one-click full scan. Analysis and scans remain local/offline. An explicit refresh action starts a separate data-update run with in-session progress and per-symbol success/failure reporting. Refresh recovery is not supported: do not persist refresh inventories/checkpoints for resumption, expose resume or recoverable-job APIs, or offer startup recovery. If the process stops, that run is lost; the user or scheduled task must start a new full refresh run.
- `POST /api/v1/scans` scans every active shape for each instrument/timeframe in the selected market scope; it does not accept or apply a shape filter. The action and shape controls apply only after the complete scan response is returned.
- One row per `(kind, code)`, with compact `mini_bars` for display and full pattern detail retrievable through pattern ID.
- A failed/unavailable file stays visible as `INSUFFICIENT_DATA` with a reason; it is not silently omitted.
- Progress is monotonic and reports discovered/scanned/total, warning/failure counts and the current data snapshot timestamp when available.
- Stable deterministic ordering is preserved for ties. Filters may narrow the result but never change source universe metadata.
- Post-scan filters preserve the server result order and inventory/failure totals. Counts reflect filtered rows, reset returns to the full scan, and a zero-result filter explains that no row matched without implying that the scan failed.
- No official calendar means weekly status is visibly insufficient/preview-only; daily decisions remain independent.

## User decisions recorded in the 2026-10-27 amendment

- A scan always analyzes every local stock and ETF and all active shapes. Users filter by action and shape only after the scan completes.
- Do not retain resumable refresh functionality in the UI, API, or persisted backend state. In-session progress and final per-run outcomes remain available while that run is alive; a later run starts from the beginning.
- Sell output remains `SELL_IF_HELD`: an exit consideration for a user who holds the instrument. The system does not collect or infer account positions and must not issue a short-sale instruction.
- Ended patterns remain visible in detail. Selecting one draws its anchor points, connected structure geometry, neckline/boundary, confirmation point, and terminal target/invalidation/follow-through marker, then positions the chart to show the full event interval with surrounding context. The selected pattern's timeframe is selected automatically.
- The workbench no longer exposes manual-review controls, a history-replay page, or an arbitrary `asof` date control. Their removals and backend/dependency scope are specified by `manual-review-retirement` and `history-replay-retirement`.

## Detail and navigation

- Scan row, chart thumbnail, or code button navigates to detail and immediately requests the latest analysis.
- Detail defaults to the latest available data. Users cannot request an arbitrary historical cutoff; event history is inspected by selecting a stored ended pattern on the current detail chart.
- Show state with plain Chinese copy; distinguish pattern direction from action. `SELL_IF_HELD` is an exit warning, not a short instruction.
- Compact layout works at the current local desktop viewport and has a readable empty/loading/error state.

## Verification

- API tests verify scan universe includes stocks/ETFs, all ten active shapes are checked regardless of post-scan filters, one ticker is merged to one row, rank ordering matches disclosed keys, a failure row remains visible, and thumbnail dates/prices are returned.
- Refresh API contract has no resume endpoint or recoverable-job listing, and restart does not restore a refresh run. Current-run progress and per-symbol outcomes are still reported.
- Browser/UI smoke verifies one-click scan, progress/completion display, buy-first order, sparkline, click-through, and automatic detail analysis.
- Detail displays unconfirmed labels, risk/target/reasons, daily/weekly conflict/unknown calendar, and provenance warnings.
- Ended-pattern selection switches to the matching timeframe and displays all available event anchors/geometry and terminal markers; the viewport includes the full event interval even when it is older than the default latest-bars window.
- The navigation contains no history-replay or manual-review page, and detail contains no review button or as-of date input.
- Python tests and JavaScript syntax checks pass; local startup remains `127.0.0.1:8301`.

## 2026-10-28 approved scan-scope amendment

- Before scanning, users may select stocks, ETFs, or both; an optional symbol-count cap is applied to the stable discovered inventory before files are read. These controls reduce the number of instruments analyzed.
- Optional start/end dates define the bars supplied to pattern recognition. The end is inclusive. Daily bars are sliced first, and weekly bars are aggregated from that same slice, so data after the requested end cannot influence either timeframe. Weeks cut by the selected range remain preview-only under the existing calendar rules.
- Optional active-pattern selections prefill the existing post-scan result filter. All active patterns are still detected and all evidence participates in the decision, including opposing patterns and risk gates. Shape selection does not reduce detection time or change the decision.
- Each result records the requested interval, actual first/last available bar, and whether its endpoint predates the file's latest available bar. Empty windows remain visible as `INSUFFICIENT_DATA`. The UI reports the latest analyzed date and warns that historical decisions apply to that date.
- No historical replay or human-review workflow is introduced. This is a bounded scan over current local files and the selected bar interval.
- Opening a row computed on an older interval goes to latest detail analysis. The row explicitly says so before navigation; its historical decision must not be mistaken for the latest detail decision.


## 2026-10-29 approved long-cycle bottom amendment

- The three bottom recognizers reuse the existing `w_bottom`, `inverse_head_shoulders`, and `triangle_bottom` family IDs. `main_structure.role=main` identifies a long-cycle recognition; no eleventh pattern family is introduced.
- Main structures receive the full causal prefix after the selected scan interval is applied. The recognizer uses the trailing five natural years for daily structure and ten natural years of completed weekly bars for weekly structure. The full supplied prefix remains available for background position evidence, and no bars before the selected scan start or after its end are included.
- The existing seven pattern recognizers keep their current 500 daily / 520 weekly structure windows and their existing decision rules. Each new main event and existing events are linked using the exact per-timeframe legacy structure windows so child pivot indexes resolve to their actual dates.
- Main event geometry remains in `main_structure`; `main_selection` records the selected primary with its as-of date and reason. Main events are retained across the family instead of collapsing to the latest event. Scan results keep all main event details, the selected primary, linked children and ended events.
- Only a selected, price-eligible main event can create a plan. Missing/unverified volume remains visible as supporting evidence and a warning, but does not block these three main shapes. The existing 7% structural-risk ceiling and 2R first-target gate still apply.
- A bullish child linked to a selected main structure carries `execution_suppression` and cannot generate a second nested buy plan. Detail groups the selected primary and its children; ended main structures remain selectable in history.
- Chart indexes are mapped to the full daily/weekly bar sequence with a per-event offset. Main structure anchors, independent neckline-test groups, weak crossings and frozen-neckline close confirmation are plotted by their actual dates.
- Existing saved scan-result payloads are immutable. The asset cache version changes for the new response and UI shape.
