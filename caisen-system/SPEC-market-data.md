# Module Specification: market-data

Status: Base scope approved for implementation by the user on 2026-10-24; explicit refresh added 2026-10-25. Amendments approved by the user on 2026-10-27; implementation authorized. This module reads existing local stock and ETF files, supports user-triggered and scheduled updates, and provides normalized bars and analysis data-version identifiers to pattern analysis, scans, and simulation. It does not repair prices, detect patterns, or decide whether to buy.

## User outcome

When the user starts the one-click full scan, every current local CSV in the selected market scope can be analyzed without manually selecting symbols. The system reports the files and dates used and retains analysis-version provenance. Refresh assumes that provider-returned OHLCV values are correct and does not perform content or historical-coverage validation before replacement. Known adjustment, volume, calendar, or corporate-action provenance remains marked unknown where it is not established; successful retrieval alone does not verify those attributes.

## Observed local inputs

The current workspace contains 3,655 files under `data/stocks/*.csv` and 1,639 under `data/etfs/*.csv`. These counts describe the present file inventory, not a guaranteed complete current or historical market universe. Both sampled file types use UTF-8-with-BOM CSV and columns `日期,代码,名称,开盘,最高,最低,收盘,成交量`.

The basis note at `data/market_data_current/.market-data-basis.json` says stock prices are QFQ and ETFs are raw, with data from Sina Finance via AkShare. Historical source notes may describe older refresh manifests, but the 2026-10-27 refresh requirement no longer creates or relies on a refresh hash manifest. The instrument registry at `caisen-system/config/instruments.json` is currently empty. A filename or successful refresh alone does not verify volume units, corporate actions, ETF category, or trading rules.

The reader in `caisen/market.py` maps Chinese and English OHLCV aliases, removes malformed rows, and flags >35% overnight gaps. Weekly completion uses a versioned XSHG calendar. Official SSE closure notices for 2020–2026 are stored in `caisen/data/xshg-official-closures.json` and directly define formal exchange sessions. `exchange-calendars==4.5.6` supplies a fallback historical calendar for 2000–2019, which remains preview-only; audit found that this package version differs from official SSE sessions in 2020 and 2025, so it is not used for formal dates in those years. A week becomes formal only when its exchange-verified final session has elapsed and the source includes it. Calendar dates outside 2020–2026 remain preview-only until an official annual notice is recorded.

Calendar references: [SSE 2026 annual holiday schedule, notice 2025-12-22](https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml); the Shanghai and Shenzhen 2026 holiday notices specify the same exchange closures. The historical package API and supported session lookup are documented in the [exchange-calendars project README](https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md).

## Scope and boundaries

### In scope

- Discover every current CSV in the two configured local directories; identify its code and kind from its path, while checking row-level codes/names for contradictions.
- Normalize provider and local data field names and date formatting into the application's OHLCV shape. Analysis results retain a data-version identifier so a saved result can be associated with the market data used to produce it; this is not an update-time old-file hash comparison, backup, or integrity gate.
- On refresh, treat provider-returned OHLCV content as correct. Do not reject a response based on price/volume values, row count, historical coverage, or comparison with the existing file. Do not create old-file content-hash snapshots or a refresh hash manifest. Missing required fields or unparseable dates that prevent normalization may still fail the import.
- For analysis input, keep existing parseability and usability reporting needed to construct bars; do not present upstream successful retrieval as proof that prices, volume units, adjustment basis, corporate actions, or market coverage are correct. Expose unknown provenance and applicable warnings rather than silently asserting verification.
- Split bars into analysis segments at unresolved corporate-action or data discontinuities so pivots, patterns, and weekly aggregation cannot bridge the boundary.
- Aggregate daily bars into weekly OHLCV using a configurable exchange-calendar provider and the last actual session in that calendar week. A week is complete only when the calendar confirms its final session has elapsed and the source includes that session; otherwise it is preview-only and excluded from formal weekly signals.
- Give each normalized symbol/timeframe result a stable data-version key derived from input hash, normalization version, calendar version, adjustment declaration, and aggregation version.
- Do not expose or accept an arbitrary historical cutoff for analysis. Current analysis always uses the latest available local history; event occurrence/confirmation dates and current-snapshot provenance remain visible.
- Provide a user-triggered update for one existing symbol or all current local CSV symbols and a scheduled daily all-symbol refresh. Update runs expose in-session progress and per-symbol successes/failures. The existing local automation is scheduled for 05:00 Asia/Shanghai and invokes `scripts/daily_refresh.py`; the computer and Codex desktop app must be available for this local automation to run.
- Fetch stock daily history using the configured stock adjustment mode and ETF history using the configured ETF endpoint. After field/date normalization, replace the corresponding local CSV using a same-directory temporary file and atomic rename. Do not compare provider history with local history or reject on OHLCV values, symbol-row values, row count, or missing prior dates. A provider, transport, normalization, or write error leaves that symbol's existing file untouched.
- Do not keep old/new refresh content hashes, CSV snapshots, or a refresh manifest. Preserve analytical `data_version`/source provenance on analysis results independently from refresh rollback or validation.
- Retry transient provider transport/throttling errors up to three total attempts, with 0.5 s then 1.0 s delays. Report attempts and exhausted per-symbol failures for the current run.
- Refresh job progress and per-symbol outcomes are transient to the running process. Do not persist refresh inventories/checkpoints to resume work, mark jobs interrupted for recovery, expose resume/recoverable-job APIs, or restore active refreshes after a process restart. A subsequent invocation starts a new run over the current inventory; it is not a continuation.
- Retry only transient provider transport errors and HTTP 408/425/429/5xx responses up to three total attempts, with 0.5 s then 1.0 s exponential delays. Record attempts in the current-run result and include the attempt count in exhausted-failure diagnostics. Exhausted transient failures remain per-symbol failures and leave that source file unchanged.

### Out of scope

- Adjusting historical OHLCV beyond the provider's declared QFQ stock response, reconstructing cash-dividend or split factors, guaranteeing delisted/closed-fund coverage, and deciding that current files represent the entire market.
- Pattern recognition, buy/sell recommendation, candidate ranking, order sizing, or Backtrader execution. These modules consume this module’s quality and provenance fields.
- Treating a 35% gap heuristic as authoritative corporate-action detection.

## Interface contract

The module exposes a small Python interface; HTTP routes remain owned by the workbench module.

- `discover_instruments(root) -> list[InstrumentFile]`: stable-sorted stock and ETF file inventory with read status.
- `load_daily_bars(instrument, root) -> MarketDataResult`: normalized daily OHLCV plus provenance and quality report from the latest available local history.
- `aggregate_weekly(result, calendar) -> MarketDataResult`: completed weekly bars and a separate optional provisional week, using the live exchange date/calendar.
- `MarketDataResult` contains `instrument`, `bars`, `source`, analysis `data_version`, `adjustment_basis`, `first_date`, `last_date`, `quality_status`, `warnings`, `rejected_rows`, `segments`, and `calendar_version` when applicable. Analysis version provenance is retained; refresh rollback hashes and old-file snapshots are not.
- `quality_status` is one of `usable`, `usable_with_warnings`, `insufficient`, or `unavailable`. This describes input usability; it is not a buy/sell verdict.
- A single-symbol refresh request must resolve to an existing local stock/ETF file; otherwise return HTTP 404 before queueing a job. A bulk refresh requires explicit `scope=all`.
- `GET /api/v1/jobs/{id}` returns current-run progress while the process retains that job. No refresh resume endpoint or recoverable-refresh-job listing exists. A process restart loses the in-progress job state; the next user or scheduled refresh starts from the beginning.

All timestamps are exchange-local dates rather than inferred UTC instants. Missing or contradictory provenance is represented as `unknown` with an explanation, not guessed from price appearance.

## Adjustment and volume semantics

- Price adjustment basis is reported per symbol and timeframe as `qfq`, `hfq`, `raw`, `unknown`, or `mixed`; a cross-symbol scan may contain multiple bases.
- Price adjustment factors and share/volume factors are separate provenance fields. Cash distributions do not directly rescale historical share volume. This module will not manufacture a volume adjustment factor.
- The observed current stock QFQ and ETF raw data may be used for exploratory pattern analysis with explicit labels. If the adjustment factor history or volume unit cannot be verified, pass that limitation to consumers. The decision module owns whether a specific buy plan is actionable under these limitations.
- Split/merger/distribution discontinuities that lack verified metadata create a segment boundary or quality warning; they are not auto-corrected.

## Current snapshot and reproducibility

- Current analyses state `history_basis=current_snapshot`; no UI/API promises a historical point-in-time replay from today's adjusted files.
- Analysis results retain data-version provenance so a result remains attributable to its input version. Refresh does not use old/new content hashes, snapshots, or hash manifests to decide whether incoming data is accepted.
- The full-scan inventory comprises files discovered in the local directories at scan start. The scan job records the inventory timestamp/counts and any files that fail. It does not silently skip unreadable files.

## Acceptance tests

1. A directory fixture containing Chinese and English headers discovers stock and ETF files, normalizes the same OHLCV values, and reports kind/code from the file identity.
2. Analysis loading reports missing/unparseable fields or dates when they prevent construction of bars; an upstream refresh does not reject values for abnormal OHLCV, incomplete prior-date coverage, row count, or apparent symbol mismatch.
3. Analysis data-version provenance remains deterministic for a given input and normalization/calendar version; it is used to identify the source analysis version and is not an update-time acceptance check or rollback snapshot.
4. An unresolved discontinuity splits segments; a pattern fixture on either side cannot be aggregated into one continuous weekly segment.
5. Analysis interfaces expose no arbitrary cutoff parameter; current loading uses the full locally available history.
6. Calendar fixtures for a normal week, a holiday-shortened week, and a Friday holiday prove that only calendar-complete weeks become formal weekly bars; an unfinished week appears only as preview.
7. A full local inventory run reports discovered, readable, warning, insufficient, and failed file counts; the reported discovered count equals the actual number of CSV files at scan start.
8. Existing analysis remains offline; only an explicit or scheduled refresh calls the provider. Refresh tests prove field/date normalization and atomic file replacement; provider/transport/normalization/write failures leave the source file untouched.
9. Refresh tests prove abnormal OHLCV values, short history, and missing dates relative to the old file do not trigger content/coverage rejection when the response can be normalized. No old-file hash snapshots or refresh manifest are written.
10. Injected transient 429/timeout failures retry on the configured backoff; exhausted retries report attempts and preserve that symbol's existing file.
11. Current-run refresh progress and per-symbol outcomes are available while the process is alive. No refresh checkpoints are persisted; after restart the prior run cannot be resumed, and no resume or recoverable-job endpoint is exposed.
12. Analysis data-version provenance remains available independently of refresh validation/rollback behavior.

## Explicit limitations to surface

- Current inventory: 3,655 stock CSVs and 1,639 ETF CSVs (recomputed at runtime; counts may change).
- Observed adjustment basis is stock QFQ and ETF raw, with no corporate-action adjustment applied in the referenced manifest.
- Per-file capture time is incomplete in the referenced manifest, and historical constituents/delisted funds are not guaranteed.
- The empty instrument registry means that category, tick size, lot size, ETF settlement rules, and volume units are not established for every symbol.
- Formal weekly bars currently cover 2020–2026 using archived official SSE annual notices. Library-only sessions before 2020 and 2027+ dates remain previews until official schedules are independently checked and recorded.
- Bulk updates can be rate-limited or blocked upstream. Partial runs expose symbol-level failures and retain old files where an update fails. The local daily refresh schedule is 05:00 Asia/Shanghai; interrupted runs are not resumed.
