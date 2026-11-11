# Anchors-Only Neckline Integration Plan

> **For agentic workers:** implementation continues under the existing user authorization for this Caisen research goal. Use the TDD sequence below for each behavior.

**Goal:** Evaluate the user's V2 neckline rule in the shared daily recognition path: form causal 2 ATR pressure boundaries first, pair them with complete bottom structures, accept boundaries that cover structural anchors without requiring every intermediate candle to stay below the line, and use those same immutable events in scans and Backtrader.

**Architecture:** Extend the existing `boundary_first` path and its cached pressure-pair preparation; preserve the V7 path as a comparison reference. A changed frozen line remains a separate boundary version and can emit one fresh post-freeze buy. The backtest execution experiment remains versioned and separate from the recognition change.

**Tech Stack:** Python, pandas/NumPy, existing daily incremental engine, Backtrader, pytest.

**Spec:** `SPEC-pattern-engine.md`, 2026-11-08 unified daily four-pattern amendment, updated by this plan's anchor-coverage clarification.

## Global Constraints

- Use the fixed paired 100 stocks, full local history, and 2026-10-30 cutoff for comparison with V8.
- Keep pressure-point reversal at 2 ATR; candidate freeze requires all structure anchors and two independent pressure tests to be causally known.
- Preserve 1 ATR close breakout buffer, six-month daily / twelve-month weekly bottom maturity, 20% decline cap for W/IHS, and distinct frozen boundary versions.
- Scan and daily Backtrader must consume the same daily recognition events.
- Keep V8 output as the reference; do not overwrite saved historical reports.
- Report the 45% cash allocation plus 60/40 exit experiment separately from shape-recognition quality.

## Review Focus

- A candle high crossing a proposed line between structural anchors must not veto anchor-only coverage; an anchor above the resistance line must still reject it.
- A changed line may trigger one new buy only after its freeze; a crossing before freeze must never be backfilled.
- Suppression compares structural anchor spans, not the parent's observation lifetime; a distinct same-anchor neckline keeps an independent one-time entry lifecycle.
- Prefix extension must not alter earlier boundary IDs, events, or entry dates.
- Same-day signals from several instances must merge labels without changing the chosen execution or ledger.
- A scan and Backtrader replay on the same prefix must agree on instance IDs and first breakout dates.

---

### Task 1: Anchor-Only Boundary Validation

**Files:**
- Modify: `caisen/main_bottoms.py`
- Test: `tests/test_main_bottoms.py`

**Interface:** Add an explicit `coverage_mode` plus structural-anchor input to `_independent_neckline_tests`; retain current interval coverage as the reference mode. The `boundary_first` caller uses `structure_anchors_only` and checks each frozen structure anchor against the candidate line with the existing contemporaneous tolerance.

- [x] Add failing tests for an intermediate non-anchor high crossing the line and for an actual structural anchor crossing it.
- [x] Run those tests and confirm failure is due to the missing anchor-coverage interface while the reference interval mode remains intact.
- [x] Implement anchor-only candidate coverage for `boundary_first`; retain all independent-touch, departure, slope, low-position, maturity, and causal checks.
- [x] Run focused tests for anchor/full-interval coverage, pair-cache admission, bidirectional suppression, and distinct frozen boundary re-entry.
- [x] Reproduce and fix a frozen-pair-on-final-prefix-bar drop; the same bar can freeze the pair, but cannot confirm the breakout.
- [x] Add a regression preventing an old confirmed pattern's extended observation end from suppressing a later bottom formation.

### Task 2: Shared Daily Path and Boundary Revisions

**Files:**
- Modify: `caisen/daily_incremental.py`, scan/research call sites as needed.
- Tests: `tests/test_daily_incremental_guards.py`, `tests/test_delayed_maturity_breakout.py`.

- [x] Existing failing-first regressions cover the same anchors receiving a distinct frozen line, one new post-freeze entry, unchanged-line suppression, and directional suppression.
- [x] Connect the new coverage mode to the causal boundary-first pressure-pair cache and shared recognition session; scan and fixed Backtrader both consume this recognizer.
- [x] Verify the long-history 600108 fixture keeps signal identities and that extending the input prefix does not rewrite earlier events.
- [ ] Run daily scan/Backtrader prefix parity on 600108, 516630, 159718, and 000538.

### Task 3: Backtrader Candidate and Paired-100 Report

**Files:**
- Modify/add: versioned daily research execution strategy and paired-100 runner/report.
- Tests: existing Backtrader accounting tests plus execution-specific regression cases.

- [ ] Keep baseline V8 and V12 8/20 executions intact.
- [ ] Add an opt-in candidate execution variant for 45% cash use with 60% of shares at +8%, the remaining shares at -8% / +20%, zero costs, and the existing T+1 entry.
- [ ] Verify identical signals with the new execution variant have reconciled cash, open value, realized PnL, and terminal close for all 100 accounts.
- [ ] Run the 100-account full-history recognition and Backtrader comparison, then report win rate, return, drawdown, account distribution, and signal/ledger parity against V8.

## Completion Gate

Do not claim the objective is complete until the new shared recognizer and execution variant are exercised on all fixed 100 accounts, daily scan and Backtrader agree on events, the run's full accounting passes, and the resulting comparison still exceeds V8 on win rate and cumulative return without hiding drawdown or implementation limits.
