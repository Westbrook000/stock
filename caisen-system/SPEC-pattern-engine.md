# Module Specification: pattern-engine

Status: Base scope approved for implementation under the user's 2026-10-24 instruction to continue the build automatically. Amendment approved by the user on 2026-10-27; implementation authorized.

## Outcome

Turn validated daily or calendar-complete weekly bars into explainable, causal pattern events. A pattern can be a candidate before it is confirmed, so results must preserve labels such as `W底，未确认`; absence of a breakout is not absence of a pattern. This module reports shape evidence and lifecycle state only. It does not decide whether the user should buy or sell.

## Inputs and boundaries

- The active `caisen-rules-v6` catalog contains exactly ten rules. The source audit keeps all twelve book candidates, but `w_bottom_spring` and `false_breakout_head_shoulders` are retired from current recognition. Their old IDs remain displayable in legacy reports only.
- Consume normalized `MarketDataResult` or a DataFrame carrying the same provenance/quality attributes. Respect `segment_id`; never construct a pattern across a data break.
- Daily and weekly bars are analyzed independently. Formal weekly analysis receives only calendar-complete bars; provisional weekly data may be shown as preview and cannot generate a formal pattern.
- Candidate output must distinguish pivot extreme date from pivot confirmation date. A pivot is knowable only after the configured right-side bars have elapsed.
- Only rules present in `caisen/rules/patterns.v2.json` can become active recognitions. The catalog is authoritative; unsupported or retired source-audit candidates are never emitted as recognized rules. A spring/fake-breakout event requires a prior consolidation box with four alternating boundary touches, then a close beyond the frozen boundary and a close back inside. The current box geometry numbers are versioned engineering starts and remain manually reviewed; they are not attributed to the book. Flags use five confirmed swing points and search the full 500-bar analysis window; older results retain their recorded rule version.
- No probability, buy/sell action, or claim of empirical accuracy is produced.

## Lifecycle contract

Each event has a stable identity and append-only observations with one current state:

`forming` → `price_confirmed` → `volume_confirmed` or `waiting_for_retest` → `invalidated` / `target_reached` / `followthrough_confirmed`.

The display layer maps rule ID plus state to labels such as `W底，未确认` and `W底，价格确认，量能未验证`. A candidate with only one pivot or incomplete topology is not a pattern result; the minimum evidenced topology is included with its unmet confirmation conditions. State history records event date and evidence. Recomputing a historical prefix must not use any later bar to select its pivots, neckline, state, or target. Later invalidation is appended as a terminal event; prior events remain immutable for the current pattern history and chart.

Before price confirmation, a bullish candidate becomes terminal `invalidated` on the first close below its frozen structural invalidation level; a bearish candidate becomes terminal on the first close above it. The latest event per rule remains in the current analysis payload so the UI can show it in completed history, but terminal states do not count as current buy/hold candidates.

A projected neckline that later leaves the price range spanned by its frozen pivots (including crossing zero) also ends the candidate at the first such bar. The terminal event and its frozen geometry remain available to current history/chart views. A neckline already on the wrong side of its final anchor remains rejected at formation.

## Causality and geometry

- Pivot detection records both `extreme_index/date` and `known_index/date`; right-side confirmation delay is explicit. ATR thresholds use values known at the candidate/confirmation time, without backfilling future ATR values. If ATR is missing or non-positive at a pivot, that pivot is skipped with no future-value substitution.
- Frozen neckline/boundary parameters use only anchors known at the time. Horizontal and sloped lines are permitted where the active rule card permits them.
- A candidate is rejected when its neckline at the final pivot is on the wrong side of that pivot (the last top/bottom never reached the required boundary), when it later crosses zero or leaves the frozen pivot price range, or when its first measured target is not a positive price. An invalid target means the book rule is not satisfied and no pattern candidate is emitted, per the user's clarification about 688216.
- Head-and-shoulders candidates use a versioned engineering shoulder tolerance: the absolute difference between the two shoulder prices divided by the head-to-neckline height at the head must be at most 7%. This is a fixed engineering parameter, not a book rule or validated accuracy claim; the retired human-label workflow will not continue tuning it.
- A confirmed breakout is a close beyond the frozen boundary at or after `max(pivot.known_index)` for all required anchors. A close that occurred before the final pivot became knowable cannot confirm that pattern. The record captures the first eligible confirmation index/date and does not move it when new bars arrive.
- After confirmation, the first target touch is recorded as a terminal `target_reached` event with the target price and date. For OHLC bars where the close invalidates the structure and the high/low also touches the target, invalidation wins because intrabar order is unknown. A reached target remains visible as completed historical structure but cannot create a current entry or exit decision.
- A spring that later closes above the frozen box upper boundary and a false breakout that later closes below the frozen box lower boundary receive terminal `followthrough_confirmed` events. They remain in history, leave the current-pattern list, and do not create a new entry/exit action by themselves.
- Segment boundaries terminate a lifecycle; they cannot be bridged by pivots, target measurements, weekly aggregation, or retest logic.
- Parent/derived relationships are emitted only when both IDs and derivation evidence are stable. Duplicate detections of one event are deduplicated; distinct rule IDs may coexist on the same evidence.
- W/M and converging-triangle candidates must have a maximum-to-minimum adjacent leg-duration ratio of 2.5 or less. W-bottom candidates also require the absolute difference between the lowest and highest trough to be no more than 15% of the higher trough. These are fixed engineering parameters, not numbers stated by the book or statistically validated accuracy claims. They remain unchanged by the calibration-tool retirement.
- A shared pivot topology can satisfy more than one shape family. Such competing labels remain visible with their evidence, but receive `manual_review_required` and cannot generate an actionable buy plan until reviewed. The same evidence does not count as independent confirmations.
- Flag candidates use five confirmed pivots, an impulse lasting at least four bars, a countertrend correction between 12% and 50% of the impulse, and a flag range no greater than 60% of the impulse. These are versioned engineering starting points, not numbers claimed by the book. A flag boundary break confirms the structure; the prior impulse extreme separately unlocks the measured projection.

## Result contract

Every pattern includes ID, rule ID/version, data version, code/kind/timeframe, direction, lifecycle state, formed date, last observation date, extreme anchors and their known dates, neckline/boundaries, breakout/confirmation date if any, invalidation, target measurements and whether each is unlocked, volume evidence/verification, source rule reference, warnings, and optional parent/derived IDs. Pattern IDs derive from instrument, timeframe, rule ID and frozen anchor dates, not future bars.

Incomplete adjustment, volume, registry, or calendar provenance is attached to evidence and remains visible; the engine does not silently treat unknown as verified. Thresholds such as ATR multiples and volume comparisons are labeled engineering parameters and are versioned separately from source rules.

## Recognition scope

Use only the ten active rule IDs in the versioned catalog and their catalog-defined topology. Standard supported shapes are automatic; compound/ambiguous variants are returned with `manual_review_required` only if the catalog includes that recognition, otherwise omitted with a support reason in diagnostics. The two retired composite IDs may appear when reading legacy v14 records but never in new v15 analyses. The engine must not infer a compound head-and-shoulders false breakout from a generic neckline reclaim.

## Verification

1. For each supported rule: positive, near-miss negative, false breakout, later invalidation and nested/overlapping fixtures.
2. Prefix audit: compute on every prefix containing a candidate; append future rows; previous event IDs, anchor prices/dates, known dates and historical states remain identical.
3. ATR/pivot causality: append extreme future volatility and prove earlier pivot thresholds and known dates do not change; a missing ATR at a candidate pivot must not be backfilled from later rows.
4. Segmentation: no event anchors, confirmation, retest or target can cross a data break.
5. Daily and weekly independence; no incomplete weekly preview creates a formal event.
6. Human calibration tools and records are removed by the `manual-review-retirement` module. The active rule set remains versioned; no human-label accuracy or calibration claims are produced.
7. A head-and-shoulders candidate with a non-positive projected target is rejected just like other pattern families; no pattern candidate or plan is emitted.
8. Run pattern tests and the full Caisen test suite.

## 2026-10-27 approved amendment

- Remove arbitrary `asof`/`as_of` arguments from pattern detection and its public analysis flow. The engine operates on the latest available local history.
- Preserve pivot known dates, event occurrence/confirmation/terminal dates, append-only lifecycle events and prefix-based causality checks. These ensure future bars cannot rewrite already-known earlier structures without exposing historical replay to users.
- Ended pattern events remain part of the latest detail result so the chart can render and navigate to their frozen anchors and terminal evidence.
- Remove calibration-specific accuracy claims and artifacts; rule thresholds remain unchanged by this feature-retirement scope.

## Limits

Recognition metrics are uncalibrated; no human-label accuracy is claimed. A visual similarity or backtest outcome is not ground truth for whether a chart pattern existed. This module cannot claim a signal is actionable or profitable.

## 2026-11-08 approved amendment: unified daily four-pattern instances

This amendment governs the daily scan and daily Backtrader research path for the approved fixed-universe backtest. It supersedes conflicting older recognition-scope and lifecycle statements above for these four bullish pattern types only: `w_bottom`, `inverse_head_shoulders`, `triangle_bottom`, and `bull_flag`. It does not rewrite or delete legacy records or historical research artifacts.

### Shared immutable instance and event contract

- Recognition produces one stable pattern-instance identity from instrument, timeframe, pattern type, and the frozen anchor dates/roles. Adding later bars cannot rename the instance or change its frozen anchors/boundary.
- `structure_key` is derived from code, kind, timeframe, type, and dated anchor roles only; bar indices and pivot-scale labels are excluded. `boundary_version_id` captures the frozen resistance line, dated pressure-test groups, and the lower frozen boundary for triangles. `instance_id` hashes both values, so a distinct, fully validated neckline on the same dated anchors is a separate tradable instance.
- Pair every validated, fully frozen boundary with the structure as a separate instance. Merge only scale copies of the same boundary. Distinct validated boundary versions on the same anchor set remain independently trackable and may each emit one first-buy event after their own freeze date. A later line does not overwrite or cancel an earlier line's pending or confirmed lifecycle. An identical consumed line cannot re-enter, and no line inherits another line's breakout or invalidation.
- Suppress a nested entry only from an eligible, unsuppressed larger outer structure toward a smaller child whose full structure-anchor interval is strictly contained within the parent's known structural interval. A structurally valid `forming` or `confirmed` main parent may suppress a child known by the decision time; `signal_eligible=false` means its own neckline has not broken yet and does not make the structure invalid. Use the parent's explicit first-trough and structure-end dates; if a forming parent has no recorded end date, its interval may extend through the current observation. Internal, already-suppressed, or explicitly unselected candidates cannot act as parents; `invalidated` and `target_reached` parents release suppression. Two W-bottom candidates sharing a low anchor are overlapping alternatives, not an inferred parent/child pair. Other shared trough anchors alone do not imply parent/child suppression, and the smaller structure cannot suppress its enclosing parent.
- Frozen geometry stores dated anchor roles/prices and frozen resistance/support boundaries. Pivot scale, sliding indices, and neckline-test indices remain separate `multiscale_evidence`; they do not alter geometry equality. Same-structure/same-boundary scale evidence is merged into one event snapshot.
- Each observation is an append-only immutable state event with `instance_id`, `event_id`, `state`, `event_date`, frozen geometry reference, evidence, and rule/engine versions. Consumers derive current state by selecting the latest event; no consumer edits an earlier event.
- Event serialization exposes `structure_key`, `boundary_version_id`, `paired_at`, `state_at`, `selection_reason`, `recognition_schema_version`, and `state_schema_version` directly. Consumers must not need to decode pattern evidence to associate an instance with its selected frozen boundary or schema.
- The daily scan and daily Backtrader research consume the same causal recognition event semantics and must agree on instance IDs, first valid breakout dates, and state at an identical input prefix. A scanner-only relabel or a separate backtest-only entry detector is not allowed.
- Permitted current lifecycle labels are `forming`, `confirmed`, `focus`, `invalidated`, and `target_reached`. A fully evidenced but unbroken structure remains `forming`; only the first valid close breakout of that instance's frozen boundary can produce `confirmed`. Invalidation and target completion are terminal for that instance's entry generation. A retreat after breakout may be represented as `focus`, but never creates a second BUY event for the same boundary instance.
- Backtest emits at most one BUY event per pattern instance, on its first valid breakout date. The simulated fill remains on the next available daily open according to the existing broker execution contract. Repeated closes above the boundary, retests, or later retreats cannot create another BUY for that instance.
- An instance in `focus` after breakout remains visible for review but cannot re-enter the order stream. Recognition/state events and execution orders are distinct records.
- A close that crosses above a frozen boundary but remains below the required ATR buffer emits an append-only `event_type=weak_crossing` observation with lifecycle `state=forming` and `entry_eligible=false`. It remains a waiting event, not a BUY or a terminal state.
- A valid buffered crossing before the six/twelve-month maturity gate emits `event_type=early_breakout_consumed` with `state=forming` and `entry_eligible=false`. The frozen boundary phase is consumed at that first crossing; if the structure later closes below its invalidation boundary, it is terminal and cannot re-enter after maturity.
- A W-bottom neckline must be associated with the rebound between each pair of trough anchors: the rebound peak must itself be a selected independent test or lie within the frozen line's touch tolerance. Full-history line coverage alone does not establish that the line belongs to the W structure. This is part of `caisen-main-bottoms-v13` and applies to every symbol and prefix.
- For W candidates, two causal independent pressure tests may both be known before the first trough, or may complete after it, or span both periods. The full paired W and rebound association must be known before freezing the instance; a post-first-trough pressure touch is not mandatory.
- A repeated crossing of one unchanged frozen boundary never creates another BUY. Touch tolerance is used to judge price/line contact, not to merge separate frozen lines: any changed frozen-line parameters on the same trough anchors require their own causal paired tests, freeze date, and new boundary-version instance; that instance may create a new buy event on a fresh post-freeze crossing even while the earlier instance remains live. A terminal instance cannot be reactivated by replaying its old boundary.
- Parent/child suppression follows strict structural time containment only: an eligible, selected outer structure can suppress a contained child, independent of whether the outer structure used a finer pivot scale. Scale labels do not reverse parenthood, and a child cannot suppress its enclosing parent.

### Geometry and eligibility

- Only the four types named in this amendment are eligible for this daily research run.
- Pressure-point reversal distance is 2 ATR. For bottom structures, minimum elapsed formation duration is six calendar months on daily bars and twelve calendar months on weekly bars. This amendment's backtest and scan comparison is daily; weekly results remain separate and cannot confirm a daily instance.
- W-bottom and inverse-head-and-shoulders candidates are rejected when the fall from the first effective pressure touch to observation exceeds 20%. Triangle bottoms have no 20% decline cap. Bull flags use their catalogued impulse/correction geometry and are not assigned that bottom-pattern decline cap.
- A slightly rising frozen bottom boundary is allowed up to 5%; the comparison must use the paired frozen support/pressure boundary from the same instance. The breakout condition is a daily close at least 1 ATR above that frozen boundary, using only ATR known before the current bar; retain the minimum two-tick floor where instrument metadata requires it.
- Existing safety constraints remain: candidates cannot bridge data segments; missing/non-positive ATR is not filled from future values; a historical frozen boundary cannot be relabeled or overwritten, while a materially different causal boundary creates a separately versioned instance; confirmation occurs no earlier than all required anchors being knowable.

### Fixed research delivery

- The end-to-end run uses the checked-in fixed universe manifest `daily_800_stock_200_etf_v1.json` (800 stocks and 200 ETFs); do not select or omit accounts based on expected outcomes.
- Each account is processed with the same recognition events that are exposed to the scan path, then passed to Backtrader under the run's recorded trading assumptions. Report all 1,000 selected accounts, including no-data and failed accounts with explicit status/reason.
- Produce a complete, linked HTML report from the completed run artifacts. Report fixed-universe counts, completion/failure reasons, event/state counts, first-breakout entry counts, trade/accounting metrics, data coverage and run/version provenance. A partially completed process or a progress snapshot is not an accepted 1,000-account result.

## 2026-11-08 approved lifecycle clarification

- Add `target_reached` as a terminal lifecycle state for a confirmed instance. On a later daily bar, record it when that bar's high reaches the first measured shape target. This state is descriptive; it does not alter the fixed Backtrader 8% stop / 20% take-profit exit rules.
- Lifecycle maintenance begins on the daily bar after the first confirmation event. Do not use the confirmation bar's earlier high or low to retroactively emit target or invalidation state.
- If a later bar closes at or below the frozen structural invalidation level and its high also reaches the measured target, record `invalidated`; OHLC alone cannot establish intrabar order, and invalidation has priority.
- `current_instances()` and all detail consumers map the latest immutable event state onto the displayed Pattern while preserving the event's frozen geometry. When a support breach removes a candidate from the active set, append its invalidation event before removal if a frozen instance was already recorded.

## 2026-11-11 neckline-coverage and boundary-revision clarification

- For the experimental boundary-first daily path, a candidate resistance line is checked against the selected structure's anchor pivots, not every intervening candle high. The structure anchors must be known before the boundary/structure pair freezes, and each anchor must remain below the line within the contemporaneous ATR/tick tolerance. The prior full-interval coverage check remains available as a reference mode and must not silently govern the boundary-first path.
- An ordinary intervening high that crosses a proposed resistance line does not by itself reject an anchor-only candidate. A selected structural high above the line still rejects it. Independent pressure tests, 2 ATR pressure-point confirmation, causal availability, slope and duration constraints remain in force.
- One structural low group may have multiple materially different frozen neckline versions. Each version is a distinct, immutable pattern instance. A new version is eligible for at most one new BUY, and only on a valid breakout after that version freezes; earlier crossings cannot be backfilled. Re-evaluating or reusing the exact same frozen line cannot re-trigger an entry.
- Parent/child suppression is directional: only a known, valid outer parent can suppress a strictly contained child. Overlapping alternatives and shared-low structures do not suppress each other in either direction. A later parent cannot retroactively cancel an earlier child event.
- When two different pattern families share at least two structural low anchors, treat them as alternative readings of the same lows rather than suppressing the contained family by date window alone. One shared low does not exempt a genuinely contained child from suppression.
- Parent/child containment uses the structural low-anchor span (first trough through latest structural low), never a live `last_observed_date` or lifecycle `structure_end_date`. An old confirmed structure cannot suppress a later formation simply because its observation window was extended.
- A newly paired boundary may freeze on the same bar that makes the structure knowable. It becomes eligible to confirm only on a later bar; the final-prefix snapshot must retain that frozen forming instance instead of dropping it.
- A new frozen neckline version on the same low anchors is a distinct lifecycle and may emit one new buy after its own freeze. Re-crossing the exact same frozen line cannot emit again; a replacement line cannot inherit an earlier line's breakout or invalidation.
- Distinct validated frozen lines on the same structural anchors remain independently active; adding a better-ranked line does not silently retire an earlier pending line. Only exact same-line scale/test copies are merged, and once a line instance has confirmed, a geometrically identical line cannot create a duplicate entry just by acquiring additional test metadata.
- Parent suppression is strictly one-way. A candidate already marked as a child (`parent_id` or `execution_suppression.suppressed`) cannot itself become a suppressing parent. The suppression pass resolves the parent hierarchy before assigning children, so a middle structure suppressed by an outer main structure cannot cascade suppression to its own descendants. Containment comparisons are based on structure anchors, and the enclosing parent is never suppressed by its child.
- The anchor-only mode currently runs in the daily `boundary_first` recognizer. Market scan and Backtrader select this same session in their normal daily paths; code-level wiring is covered, while symbol-by-symbol prefix parity for the updated mode remains a required validation before full-run acceptance.
- Different validated frozen necklines on the same structural lows are independent instances and may each produce one buy signal on their own first post-freeze valid breakout. Repeated crossing of the exact same frozen line remains consumed and cannot produce a second buy.
- The implementation carrying the one-way hierarchy resolution and independent same-anchor boundary revision behavior is `caisen-daily-continuous-v15` / `caisen-daily-boundary-first-v16`. These versions are part of signal-cache identity; earlier cache entries are not evidence for the revised behavior and must not be reused for its acceptance run.
