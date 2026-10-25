# Module Specification: analysis-workbench

Status: approved for implementation under the user's explicit full-build instruction.

## Primary flow

1. User clicks “全量扫描”. The app scans every current local stock and ETF CSV once, across selected daily/weekly timeframes, and shows asynchronous progress and completion totals.
2. Completed results are merged to one row per instrument, sorted by decision-engine keys. `CONSIDER_BUY` rows passing the 7% risk / 2R gates appear first. The sort keys and reason are visible; the ordering is a screening priority, never a probability.
3. Each row shows code/name, action, current pattern/state, latest close/date, data warning, and a compact recent candlestick thumbnail.
4. Clicking a row or thumbnail opens the detail view for that code and loads its latest analysis without a second search. Detail shows the buy/sell/wait decision, daily and weekly evidence, current shape labels such as `W底，未确认`, chart anchors/boundaries, confirmation/invalidity, stop/targets, quality/provenance, rule/data versions and reasons.

## Scan contract

- Default inventory is all local stocks plus all local ETFs, from the file set observed at job start. Analysis and scans remain local/offline; an explicit refresh action starts a separate symbol-level data-update job with progress and failure reporting. Refresh jobs persist the frozen inventory once and use normalized per-symbol checkpoints while keeping the frequently updated job summary small; interrupted jobs can resume without re-fetching successful symbols, and completed jobs can retry failed symbols. On page startup the workbench queries the compact recoverable-job list so restart recovery remains visible even when session storage is unavailable.
- One row per `(kind, code)`, with compact `mini_bars` for display and full pattern detail retrievable through pattern ID.
- A failed/unavailable file stays visible as `INSUFFICIENT_DATA` with a reason; it is not silently omitted.
- Progress is monotonic and reports discovered/scanned/total, warning/failure counts and the current data snapshot timestamp when available.
- Stable deterministic ordering is preserved for ties. Filters may narrow the result but never change source universe metadata.
- No official calendar means weekly status is visibly insufficient/preview-only; daily decisions remain independent.

## Detail and navigation

- Scan row, chart thumbnail, or code button navigates to detail and immediately requests the latest analysis.
- Detail defaults to current data; historical `asof` replay remains explicit and states that the data is a current snapshot truncated to the chosen date.
- Show state with plain Chinese copy; distinguish pattern direction from action. `SELL_IF_HELD` is an exit warning, not a short instruction.
- Compact layout works at the current local desktop viewport and has a readable empty/loading/error state.

## Verification

- API tests verify scan universe includes stocks/ETFs, one ticker is merged to one row, rank ordering matches disclosed keys, a failure row remains visible, and thumbnail dates/prices are returned.
- Browser/UI smoke verifies one-click scan, progress/completion display, buy-first order, sparkline, click-through, and automatic detail analysis.
- Detail displays unconfirmed labels, risk/target/reasons, daily/weekly conflict/unknown calendar, and provenance warnings.
- Python tests and JavaScript syntax checks pass; local startup remains `127.0.0.1:8301`.
