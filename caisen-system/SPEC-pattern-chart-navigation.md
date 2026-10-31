# Module Specification: pattern-chart-navigation

Status: Capability-map scope accepted by the user on 2026-10-27. Approved and implemented on 2026-10-27. Verified with the current automated suite and local browser.

## Objective

Let a user inspect the actual geometry of an ended pattern, even when it occurred earlier than the chart's default latest-bars window. In the detail view, selecting an ended-pattern card must immediately show that event's structure on the correct daily or weekly candlestick chart and position the viewport over the full event.

## User flow

1. Open a stock or ETF detail at current data and selected timeframe.
2. Expand “已结束的历史形态” and click one pattern card.
3. The chart switches to the selected pattern's timeframe if needed, draws only that selected pattern's historical geometry, and scrolls/zooms to its anchors through its terminal event.
4. The selected card remains visibly selected and its pattern name, time period, terminal status and target/invalidation level can be matched to the chart.
5. “回到最新” restores the current chart window and clears the historical selection overlay.

## Drawing contract

- Selectable terminal states are `invalidated`, `target_reached`, and `followthrough_confirmed`.
- Draw all available frozen `key_points` at their event dates/prices, with labels or a readable legend mapping marker names to the structure (for example left trough, neckline pivot, head, right shoulder).
- Draw the event's frozen neckline or structure boundary across its valid date interval. Do not recompute the line using later bars.
- Mark the price-confirmation point/date when present.
- Mark the terminal event at its date and price, and label its reason: invalidation, first target reached, or follow-through boundary confirmed.
- Show an event's target/invalidation reference levels when available. Do not display plan/target levels as observed prices unless the event payload says they were reached.
- Use the stored event anchors and their known/occurrence dates; do not infer points from visual similarity or introduce future bars to change historical geometry.
- The viewport includes the earliest anchor through the terminal event, plus visible context on both sides. If the event lies outside the loaded chart data, the detail API must provide enough bars for the event or return a clear unavailable-data message.
- Only the selected ended pattern is overlaid, avoiding clutter from other historical patterns. Current-pattern rendering remains unchanged.

## Boundaries

- This capability changes chart presentation and selection only. It does not change recognition rules, event state, trade plans, or ranking.
- It does not restore history replay or arbitrary date cutoff controls.
- It does not add manual review or annotation controls.
- Chart data remains the current local snapshot; dates describe event location, not a historically captured market-data snapshot.

## Project structure and interfaces

- Frontend: `caisen/static/app.js`, `index.html`, `app.css`.
- API: existing detail analysis payload in `caisen/api.py`; extend it only if the selected terminal event's bars/indices are not currently available.
- Tests: `caisen-system/tests/test_workbench.py` and focused JavaScript chart behavior checks under `caisen-system/tests/js/`.
- Use the current FastAPI + vanilla JavaScript + ECharts stack; no new dependency is required by this specification.

## Testing strategy

- Verify each terminal state renders the correct event points, line/boundary and terminal marker.
- Verify a weekly event switches from daily to weekly chart and selection uses the matching period.
- Verify selecting an event older than the default 120-bar window moves the chart to its complete anchor-to-terminal span.
- Verify selection never changes pattern payload, decision, order plan, or current analysis timestamp.
- Verify missing chart bars produce a visible explanation rather than silently displaying the wrong region.
- Commands: `.venv/bin/python -m pytest caisen-system/tests`; `node --check caisen-system/caisen/static/app.js`; local UI at `http://127.0.0.1:8301/`.

## Success criteria

- Clicking an ended pattern card immediately selects the correct chart timeframe, overlays the event geometry, and shows its full interval with context.
- No future bars or recomputed anchors alter the stored historical pattern.
- Current pattern chart behavior and decision output are unchanged.

## Open questions

- None. If a marker name cannot be derived from the existing event payload, the UI should use the rule card's anchor names and keep the event's prices/dates authoritative.
