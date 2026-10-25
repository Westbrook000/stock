# Module Specification: market-data

Status: approved for implementation by the user on 2026-10-24; explicit refresh added 2026-10-25 at the user's request. This module reads existing local stock and ETF files, supports user-triggered updates, and provides normalized bars, quality facts, and reproducible data-version identifiers to pattern analysis, scans, replay, and simulation. It does not repair prices, detect patterns, or decide whether to buy.

## User outcome

When the user starts the later one-click full scan, every current local CSV can be analyzed without manually selecting symbols. The system can tell the user which files and dates it used, whether each input passed basic checks, and where analysis is limited by adjustment, volume, calendar, or corporate-action uncertainty. Data-quality uncertainty remains visible beside pattern and decision output; it is never silently treated as verified.

## Observed local inputs

The current workspace contains 3,655 files under `data/stocks/*.csv` and 1,639 under `data/etfs/*.csv`. These counts describe the present file inventory, not a guaranteed complete current or historical market universe. Both sampled file types use UTF-8-with-BOM CSV and columns `日期,代码,名称,开盘,最高,最低,收盘,成交量`.

The basis note at `data/market_data_current/.market-data-basis.json` says stock prices are QFQ and ETFs are raw, with data from Sina Finance via AkShare. The refresh manifest `data/market_data_versions/20261018-active-full-refresh-v1.manifest.json` records a 2026-10-18 refresh, mixed adjustment basis, no corporate-action adjustments applied, and incomplete exception/history metadata. The instrument registry at `caisen-system/config/instruments.json` is currently empty. A manifest or filename alone does not verify volume units, corporate actions, ETF category, or trading rules.

The reader in `caisen/market.py` maps Chinese and English OHLCV aliases, removes malformed rows, and flags >35% overnight gaps. Weekly completion uses a versioned XSHG calendar. Official SSE closure notices for 2020–2026 are stored in `caisen/data/xshg-official-closures.json` and directly define formal exchange sessions. `exchange-calendars==4.5.6` supplies a fallback historical calendar for 2000–2019, which remains preview-only; audit found that this package version differs from official SSE sessions in 2020 and 2025, so it is not used for formal dates in those years. A week becomes formal only when its exchange-verified final session is present in the symbol file and the as-of date has reached it. Calendar dates outside 2020–2026 remain preview-only until an official annual notice is recorded.

Calendar references: [SSE 2026 annual holiday schedule, notice 2025-12-22](https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml); the Shanghai and Shenzhen 2026 holiday notices specify the same exchange closures. The historical package API and supported session lookup are documented in the [exchange-calendars project README](https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md).

## Scope and boundaries

### In scope

- Discover every current CSV in the two configured local directories; identify its code and kind from its path, while checking row-level codes/names for contradictions.
- Normalize date, open, high, low, close, volume, code, and name while preserving the exact file path, SHA-256, row count, first/last valid date, declared adjustment basis, and source manifest metadata where available.
- Validate required columns, parseable dates/numbers, unique daily dates, positive prices, OHLC consistency, non-negative volume, ordering, missing-session warnings, duplicate rows, and suspicious price/volume discontinuities.
- Return quality facts and warnings without silently changing source values. Retain normalized valid rows when recoverable; expose rejected row counts and reasons. A file with no usable bars is unavailable with a clear reason.
- Split bars into analysis segments at unresolved corporate-action or data discontinuities so pivots, patterns, and weekly aggregation cannot bridge the boundary.
- Aggregate daily bars into weekly OHLCV using a configurable exchange-calendar provider and the last actual session in that calendar week. A week is complete only when the calendar confirms its final session has elapsed and the source includes that session; otherwise it is preview-only and excluded from formal weekly signals.
- Give each normalized symbol/timeframe result a stable data-version key derived from input hash, normalization version, calendar version, adjustment declaration, and aggregation version.
- Support an `as_of` cutoff by removing later bars before aggregation and analysis. Report the distinction between a current adjusted snapshot truncated to a date and a true historical snapshot captured at that date.
- Provide a user-triggered update for one existing symbol or all current local CSV symbols. Update jobs expose progress, per-symbol successes/failures, and the source/adjustment/date/hash used.
- Fetch stock daily history as QFQ and ETF history as raw prices, matching the declared existing basis. Validate dates and OHLCV before replacement; if the provider includes a code column, every row must resolve to the requested six-digit symbol. Also parse the existing file and require the provider response to cover its full valid date range and every existing valid trading date. If existing history cannot be validated, provider identity differs, or any existing date is missing, reject the refresh before creating a snapshot or metadata change; never replace a longer local history with a shorter/partial/wrong-symbol provider response. Write each file via same-directory temporary file plus atomic rename. Reject responses with fewer than 20 valid rows, and leave that symbol's existing file untouched on provider, validation, coverage, or write failure.
- Preserve old and new content hashes in a refresh manifest. The manifest-level `updated_at` must match the most recently refreshed symbol entry, and each symbol entry retains its own refresh timestamp. Keep provider-native volume units explicit; do not mark corporate actions or volume units as verified merely because a download succeeded. Reject unsupported stock code families with a clear per-symbol error rather than silently routing them to the wrong exchange.
- Preserve the exact prior CSV bytes once under `data/market_data_versions/snapshots/<kind>/<code>-<sha256>.csv` before replacement. Record the relative snapshot path in the manifest so an operator can restore the prior file.
- Pace bulk refreshes at least 250 ms between symbols and display job progress. Provider-side throttling can still stop the run.
- Persist the bulk job's starting inventory once in normalized `refresh_job_items` rows and checkpoint each attempted symbol by updating only that row. Keep the `jobs.payload` summary bounded: do not rewrite the full inventory or cumulative result arrays on each symbol. `GET /api/v1/jobs/{id}` reconstructs the public inventory/results/failures from summary plus item rows. On process startup, mark queued/running refresh jobs as interrupted instead of leaving stale `running` status. A resume action reuses the frozen inventory, skips symbols already refreshed successfully, and retries only failed or not-yet-attempted symbols; each retried symbol still passes the normal full-history/atomic-replacement checks.
- Retry only transient provider transport errors and HTTP 408/425/429/5xx responses up to three total attempts, with 0.5 s then 1.0 s exponential delays. Record attempts in the result and per-symbol refresh manifest entry; include the attempt count in exhausted-failure diagnostics. Do not retry schema, symbol-identity, OHLCV, or history-coverage validation errors. Exhausted transient failures remain per-symbol failures and leave the source file and metadata unchanged.

### Out of scope

- Adjusting historical OHLCV beyond the provider's declared QFQ stock response, reconstructing cash-dividend or split factors, guaranteeing delisted/closed-fund coverage, and deciding that current files represent the entire market.
- Pattern recognition, buy/sell recommendation, candidate ranking, order sizing, or Backtrader execution. These modules consume this module’s quality and provenance fields.
- Treating a 35% gap heuristic as authoritative corporate-action detection.

## Interface contract

The module exposes a small Python interface; HTTP routes remain owned by the workbench module.

- `discover_instruments(root) -> list[InstrumentFile]`: stable-sorted stock and ETF file inventory with read status.
- `load_daily_bars(instrument, root, as_of=None) -> MarketDataResult`: normalized daily OHLCV plus provenance and quality report.
- `aggregate_weekly(result, calendar, as_of=None) -> MarketDataResult`: completed weekly bars and a separate optional provisional week.
- `MarketDataResult` contains `instrument`, `bars`, `source`, `file_sha256`, `data_version`, `adjustment_basis`, `first_date`, `last_date`, `quality_status`, `warnings`, `rejected_rows`, `segments`, and `calendar_version` when applicable.
- `quality_status` is one of `usable`, `usable_with_warnings`, `insufficient`, or `unavailable`. This describes input usability; it is not a buy/sell verdict.
- A single-symbol refresh request must resolve to an existing local stock/ETF file; otherwise return HTTP 404 before queueing a job. A bulk refresh requires explicit `scope=all`.
- `GET /api/v1/jobs/{id}` returns reconstructed inventory and symbol checkpoints. `POST /api/v1/jobs/{id}/resume` accepts only interrupted/failed refresh jobs or completed refresh jobs with per-symbol failures; it rejects other job types and jobs with no remaining work.
- `GET /api/v1/data/refresh/jobs` lists compact summaries for interrupted, failed, active, and partially completed refresh jobs. The workbench displays them after a browser session is lost and provides resume or progress actions without requiring the original session storage entry.

All timestamps are exchange-local dates rather than inferred UTC instants. Missing or contradictory provenance is represented as `unknown` with an explanation, not guessed from price appearance.

## Adjustment and volume semantics

- Price adjustment basis is reported per symbol and timeframe as `qfq`, `hfq`, `raw`, `unknown`, or `mixed`; a cross-symbol scan may contain multiple bases.
- Price adjustment factors and share/volume factors are separate provenance fields. Cash distributions do not directly rescale historical share volume. This module will not manufacture a volume adjustment factor.
- The observed current stock QFQ and ETF raw data may be used for exploratory pattern analysis with explicit labels. If the adjustment factor history or volume unit cannot be verified, pass that limitation to consumers. The decision module owns whether a specific buy plan is actionable under these limitations.
- Split/merger/distribution discontinuities that lack verified metadata create a segment boundary or quality warning; they are not auto-corrected.

## History and reproducibility

- `as_of` means no rows after the requested date enter calculations. It does not claim that the source file, QFQ factors, or instrument universe is the historical snapshot known on that date.
- When the input is the current file set, every output states `history_basis=current_snapshot`; UI copy must say “当前存续标的回放 / 当前复权快照截断” as appropriate.
- Per-file SHA-256 and normalization/calendar versions are persisted in results. If a file changes, its data version changes and prior analysis remains attributable to the prior hash.
- The full-scan inventory comprises files discovered in the local directories at scan start. The scan job records the inventory timestamp/counts and any files that fail. It does not silently skip unreadable files.

## Acceptance tests

1. A directory fixture containing Chinese and English headers discovers stock and ETF files, normalizes the same OHLCV values, and reports kind/code from the file identity.
2. Malformed date, duplicate date, invalid OHLC, negative volume, missing columns, empty file, and row-code mismatch each produce deterministic warnings or unavailable results with rejected counts.
3. Stable-file hashing yields stable `data_version`; changing any source byte or normalization version changes it. Provenance records the actual file SHA-256 and last valid date.
4. An unresolved discontinuity splits segments; a pattern fixture on either side cannot be aggregated into one continuous weekly segment.
5. An `as_of` fixture proves that later daily rows do not affect normalized output, weekly aggregation, or data version for the same truncated input state.
6. Calendar fixtures for a normal week, a holiday-shortened week, a Friday holiday, and a midweek `as_of` prove that only calendar-complete weeks become formal weekly bars; an unfinished week appears only as preview.
7. A full local inventory run reports discovered, readable, warning, insufficient, and failed file counts; the reported discovered count equals the actual number of CSV files at scan start.
8. Existing data tests and full Caisen test suite pass. Local analysis never makes network requests; only an explicit refresh job calls the provider. Refresh tests use an injected provider and prove failures preserve the original symbol file. A provider response that omits the local earliest date or any existing valid trading date is rejected before snapshot/manifest writes.
9. When a provider response includes a symbol column, a mismatched or blank symbol is rejected before snapshot, CSV, basis, or manifest writes. Repeated refreshes advance the manifest-level timestamp and keep it equal to the latest symbol-entry timestamp.
10. Injected transient 429/timeout failures retry on the configured backoff and can succeed without corrupting local history; results and manifest entries show attempts. Exhausted retries report the count, preserve the CSV and create no snapshot/manifest update. Deterministic validation failures are not retried.
11. Bulk refresh checkpoint tests prove every attempted symbol is persisted, including mixed success/failure runs, while the summary payload stays bounded and per-symbol state is normalized. Startup recovery changes stale queued/running refresh jobs to interrupted without losing the inventory or completed results.
12. Resume tests prove successful symbols are not fetched again, failed and unfinished symbols are retried from the saved inventory, and resume rejects unrelated/completed-success jobs. The UI keeps the job ID across page reloads and offers a resume/retry action.

## Explicit limitations to surface

- Current inventory: 3,655 stock CSVs and 1,639 ETF CSVs (recomputed at runtime; counts may change).
- Observed adjustment basis is stock QFQ and ETF raw, with no corporate-action adjustment applied in the referenced manifest.
- Per-file capture time is incomplete in the referenced manifest, and historical constituents/delisted funds are not guaranteed.
- The empty instrument registry means that category, tick size, lot size, ETF settlement rules, and volume units are not established for every symbol.
- Formal weekly bars currently cover 2020–2026 using archived official SSE annual notices. Library-only sessions before 2020 and 2027+ dates remain previews until official schedules are independently checked and recorded.
- Bulk updates can be rate-limited or blocked upstream. Partial runs expose symbol-level failures and retain old files where update fails; no automatic refresh schedule is implied.
