# Module Specification: pattern-engine

Status: approved for implementation under the user's 2026-10-24 instruction to continue the build automatically.

## Outcome

Turn validated daily or calendar-complete weekly bars into explainable, causal pattern events. A pattern can be a candidate before it is confirmed, so results must preserve labels such as `W底，未确认`; absence of a breakout is not absence of a pattern. This module reports shape evidence and lifecycle state only. It does not decide whether the user should buy or sell.

## Inputs and boundaries

- Consume normalized `MarketDataResult` or a DataFrame carrying the same provenance/quality attributes. Respect `segment_id`; never construct a pattern across a data break.
- Daily and weekly bars are analyzed independently. Formal weekly analysis receives only calendar-complete bars; provisional weekly data may be shown as preview and cannot generate a formal pattern.
- Candidate output must distinguish pivot extreme date from pivot confirmation date. A pivot is knowable only after the configured right-side bars have elapsed.
- Only rules present in `caisen/rules/patterns.v2.json` can become active recognitions. The catalog is authoritative; unsupported source-audit candidates are never emitted as recognized rules. A spring/fake-breakout event requires a prior consolidation box with four alternating boundary touches, then a close beyond the frozen boundary and a close back inside. The current box geometry numbers are versioned engineering starts and remain manually reviewed; they are not attributed to the book. A W-bottom-spring candidate requires that inner box event to overlap an outer W, and its lifecycle follows the outer W. Flags use five confirmed swing points and search the full 500-bar analysis window; older results retain their v1 version.
- No probability, buy/sell action, or claim of empirical accuracy is produced.

## Lifecycle contract

Each event has a stable identity and append-only observations with one current state:

`forming` → `price_confirmed` → `volume_confirmed` or `waiting_for_retest` → `invalidated` / `target_reached` / `followthrough_confirmed`.

The display layer maps rule ID plus state to labels such as `W底，未确认` and `W底，价格确认，量能未验证`. A candidate with only one pivot or incomplete topology is not a pattern result; the minimum evidenced topology is included with its unmet confirmation conditions. State history records event date and evidence. Recomputing a historical prefix must not use any later bar to select its pivots, neckline, state, or target. Later invalidation is recorded as a later event and must not erase the earlier as-of view.

Before price confirmation, a bullish candidate becomes terminal `invalidated` on the first close below its frozen structural invalidation level; a bearish candidate becomes terminal on the first close above it. The latest event per rule remains in the current analysis payload so the UI can show it in completed history, but terminal states do not count as current buy/hold candidates. The earlier event remains `forming` when replayed as of a date before that invalidation.

A projected neckline that later leaves the price range spanned by its frozen pivots (including crossing zero) also ends the candidate at the first such bar. This is a lifecycle event, not a reason to erase the earlier as-of view. A neckline already on the wrong side of its final anchor remains rejected at formation.

## Causality and geometry

- Pivot detection records both `extreme_index/date` and `known_index/date`; right-side confirmation delay is explicit. ATR thresholds use values known at the candidate/confirmation time, without backfilling future ATR values. If ATR is missing or non-positive at a pivot, that pivot is skipped with no future-value substitution.
- Frozen neckline/boundary parameters use only anchors known at the time. Horizontal and sloped lines are permitted where the active rule card permits them.
- A candidate is rejected when its neckline at the final pivot is on the wrong side of that pivot (the last top/bottom never reached the required boundary), when it later crosses zero or leaves the frozen pivot price range, or when its first measured target is not a positive price. An invalid target means the book rule is not satisfied and no pattern candidate is emitted, per the user's clarification about 688216.
- Head-and-shoulders candidates use a versioned engineering shoulder tolerance: the absolute difference between the two shoulder prices divided by the head-to-neckline height at the head must be at most 7%. This was narrowed from the prior permissive geometry after the same development labels showed two `N` false positives. It is a development-set calibration parameter, not a book rule, and requires frozen holdout validation.
- A confirmed breakout is a close beyond the frozen boundary at or after `max(pivot.known_index)` for all required anchors. A close that occurred before the final pivot became knowable cannot confirm that pattern. The record captures the first eligible confirmation index/date and does not move it when new bars arrive.
- After confirmation, the first target touch is recorded as a terminal `target_reached` event with the target price and date. For OHLC bars where the close invalidates the structure and the high/low also touches the target, invalidation wins because intrabar order is unknown. A reached target remains visible as completed historical structure but cannot create a current entry or exit decision.
- A spring that later closes above the frozen box upper boundary and a false breakout that later closes below the frozen box lower boundary receive terminal `followthrough_confirmed` events. They remain in history, leave the current-pattern list, and do not create a new entry/exit action by themselves.
- Segment boundaries terminate a lifecycle; they cannot be bridged by pivots, target measurements, weekly aggregation, or retest logic.
- Parent/derived relationships are emitted only when both IDs and derivation evidence are stable. Duplicate detections of one event are deduplicated; distinct rule IDs may coexist on the same evidence.
- W/M and converging-triangle candidates must have a maximum-to-minimum adjacent leg-duration ratio of 2.5 or less. W-bottom candidates also require the absolute difference between the lowest and highest trough to be no more than 15% of the higher trough. The 15% W-bottom tolerance was added in `caisen-rules-v5` after the user's explicit clarification that 688216 is `N` despite a 17.9% to 43.9% trough gap; the known forming W example 688330 has a 7.4% gap and remains recognized. Both are development-set engineering parameters, not numbers stated by the book; the cutoff needs a new frozen blind set before describing general recognition quality. The original round_01_v2 answer file is kept unchanged; the correction is stored separately in `round_01_v2.clarifications.json`.
- A shared pivot topology can satisfy more than one shape family. Such competing labels remain visible with their evidence, but receive `manual_review_required` and cannot generate an actionable buy plan until reviewed. The same evidence does not count as independent confirmations.
- Flag candidates use five confirmed pivots, an impulse lasting at least four bars, a countertrend correction between 12% and 50% of the impulse, and a flag range no greater than 60% of the impulse. These are versioned engineering starting points, not numbers claimed by the book. A flag boundary break confirms the structure; the prior impulse extreme separately unlocks the measured projection.

## Result contract

Every pattern includes ID, rule ID/version, data version, code/kind/timeframe, direction, lifecycle state, formed date, last observation date, extreme anchors and their known dates, neckline/boundaries, breakout/confirmation date if any, invalidation, target measurements and whether each is unlocked, volume evidence/verification, source rule reference, warnings, and optional parent/derived IDs. Pattern IDs derive from instrument, timeframe, rule ID and frozen anchor dates, not future bars.

Incomplete adjustment, volume, registry, or calendar provenance is attached to evidence and remains visible; the engine does not silently treat unknown as verified. Thresholds such as ATR multiples and volume comparisons are labeled engineering parameters and are versioned separately from source rules.

## Recognition scope

Use the 12 source-audited rule IDs and their catalog-defined topology. Standard supported shapes are automatic; compound/ambiguous variants are returned with `manual_review_required` only if the catalog includes that recognition, otherwise omitted with a support reason in diagnostics. The engine must not infer a compound head-and-shoulders false breakout from a generic neckline reclaim unless its structural prerequisites are present.

## Verification

1. For each supported rule: positive, near-miss negative, false breakout, later invalidation and nested/overlapping fixtures.
2. Prefix audit: compute on every prefix containing a candidate; append future rows; previous event IDs, anchor prices/dates, known dates and historical states remain identical.
3. ATR/pivot causality: append extreme future volatility and prove earlier pivot thresholds and known dates do not change; a missing ATR at a candidate pivot must not be backfilled from later rows.
4. Segmentation: no event anchors, confirmation, retest or target can cross a data break.
5. Daily and weekly independence; no incomplete weekly preview creates a formal event.
6. User's ten submitted examples are loaded as calibration labels, not as proof of strategy performance. `N` means no active candidate from any of the 12 book patterns at the labeled cutoff; terminal historical shapes do not count. Apply explicit user corrections from the separate clarification file without overwriting the original answer file. Keep user labels separate from machine self-evaluation; report disagreements for review.
7. A head-and-shoulders candidate with a non-positive projected target is rejected just like other pattern families; no pattern candidate or plan is emitted.
8. Run pattern tests and the full Caisen test suite.

## Limits

Recognition metrics remain uncalibrated until the labeled cases have been reviewed and a frozen held-out set exists. A visual similarity or backtest outcome is not ground truth for whether a chart pattern existed. This module cannot claim a signal is actionable or profitable.
