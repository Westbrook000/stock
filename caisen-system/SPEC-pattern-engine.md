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

## 2026-11-12 structure-boundary lifecycle amendment

This amendment applies to the four bullish research entries W bottom, inverse head-and-shoulders, triangle bottom, and bull flag. It supersedes conflicting lifecycle text above only for these entries and keeps the original ten-rule catalog and the other six rules unchanged.

- A tradable instance is the immutable pairing of instrument, timeframe, family, structure anchor dates, one frozen boundary version, line tests, structure/boundary known dates, and append-only events. Candidate pressure lines can be shared as proposals; a pattern instance always owns its own copied line.
- W and inverse head-and-shoulders may use confirmed pressure touches before and after the structure inside the daily five-year / weekly ten-year observation window. The selected line must still geometrically match the structure: W lines must cover the intervening rebound; inverse head-and-shoulders lines must have independent evidence on both reaction legs. External pressure evidence cannot substitute for these required internal relations. Triangle rails remain tied to triangle pivots; bull flag uses its existing channel geometry.
- The pressure search is anchored to the current observation date, not to the first trough or left shoulder. No additional one-year-before-structure cutoff is applied; all candidate touches inside the applicable five-year / ten-year window are eligible, subject to the shape-specific internal association and coverage checks.
- Daily bottom structures mature after six calendar months and weekly bottom structures after twelve, counted from the first structural trough. Pressure pivots are confirmed after a 2 ATR reversal. Independent tests retain the existing minimum gaps and departure requirements. Formation coverage uses all highs from the first selected touch group through freeze, with the existing `max(2 ticks, 0.5 ATR14)` tolerance.
- A selected boundary is ranked using only evidence knowable at that time: independent test count, test span, normalized residual, latest test, then stable anchor order. Breakout price and future trading outcome do not influence boundary selection. Readiness is `max(structure_known_at, line_frozen_at)`.
- The first close after readiness above the instance's frozen boundary plus `max(previous completed ATR14, 2 ticks)` confirms the instance, subject to the shape's unchanged geometry and maturity rules. A buffered crossing before maturity consumes that boundary phase and cannot be moved forward to create a later first signal. After confirmation, the first close below that same instance's projected frozen line invalidates it. Neither invalidated nor expired instances return to waiting.
- A genuinely new line can establish a new instance only under a new boundary identity; after an instance fails, the replacement must include at least two independently confirmed pressure tests formed after the failure. A later touch alone, scale relabel, or micro-adjustment does not revive the old event.
- Scanner and daily Backtrader signals consume the same daily append-only event stream. Weekly scanner analysis is independent and does not affect the daily backtest. Backtrader entry and exits remain next-session open, 8% stop, 20% target, zero costs, and end-of-period liquidation.
- Daily and weekly scans/details now use the same append-session engine on separate completed-bar streams. Weekly structure search is ten calendar years, maturity is twelve calendar months, and neckline test spacing is two complete weekly bars. An unfinished week remains preview data and cannot create a formal weekly event.
- Nested suppression is directional and structural: identical confirmed anchor dates across ATR scales are one structure, not parent/child pairs. A distinct parent can suppress a main-pattern child only when its known time interval and frozen price envelope both contain the child; temporal overlap alone is insufficient.
- Historical research uses the fixed 100-symbol cohort for first paired validation against exact Git `748d3d01`, cutoff 2026-10-30. Only after event and ledger parity checks may the fixed 800-stock/200-ETF cohort run. The 60-minute cold-run goal is measured from data loading through HTML report writing; a miss is reported as a miss.
- The previous structure-boundary baseline uses source rules `caisen-main-bottoms-v12-structure-boundary`, neckline selector `joint_structure_boundary_v2`, pattern rules `caisen-v25-structure-boundary-v2`, and append-session engine `caisen-joint-pattern-session-v4` (`joint_append_session_v4` recognizer). These identifiers must be stored with each new scan and research run; historical v8/v9 reports retain their original identifiers.

Versioned implementation work is experimental until the paired validation completes. Old scans, reports, and cached signals must keep their original engine/rules identifiers and must not be relabeled as this lifecycle.

### 颈线低位检查（v14）

W 底、头肩底及收敛三角形底部在日线与周线共用此检查。以最早结构低点之前的全部可用同周期行情为基准（不含该低点当根 K 线），上限为历史最低价加上历史高低区间的 50%。每条候选上沿取首末结构低点之间的中点线价（两个投影价的平均），须为正且不超过该上限；缺少有效历史区间时不输出该形态。底点本身的低位或前高回落条件继续独立计算。

候选线在形态几何匹配后、形成区间覆盖与选线排序前分别检查；不合格的候选不会影响其他线。一组结构没有合格上沿时不输出形态。旗形不使用该门槛。选线证据记录中间价、历史范围与上限，规则版本为 `caisen-main-bottoms-v14-neckline-low-position-50`；历史报告保留原口径。

### 扫描短窗口和失效（v15）

扫描、详情截至观察日期，仅送入最近 5 个自然年的日线或 10 个自然年的完整周线，窗口之外只提供低位所用高低价摘要。日线回测保留全部可用历史。两条路径共享失效规则：未确认结构移出搜索窗口、旗形超过 500 根或三角形错过 50%–75% 窗口时，写入一次终态事件和原因。历史确认日与边界保留。

`WATCH` 是观察标签：最新完整 K 线的收盘严格低于同一已成型实例的冻结边界，价差不超过前一根完整 K 线 ATR14 的 3 倍。日周独立；可考虑买入优先；关注不产生交易信号。

### 颈线相对高度（v17）

本规则替换 v14/v15 的颈线历史区间低位门槛。W 底取两底间最高已确认反弹当根的冻结颈线价，以两底均价为参考；头肩底、收敛三角形底部取最低结构低点当根的冻结上沿价，以该点价格为参考。高度比例为（边界价 − 参考价）÷ 参考价，必须大于零且不超过 100%（v16 为 50%）。每条候选线分别检查，通过者才继续参与覆盖检验与排序。底点自身的低位规则、过去行情摘要、旗形和突破条件保持原样。

### 过早突破与候选颈线接替

每条候选颈线独立核查自身首次有效越线。日线未满六个月、周线未满十二个月即有效越线时，记录该边界的失效事件，分别保留发生日与识别记录日。该线不再参与关注或触发入场。同组结构点仍可匹配通过原有几何、覆盖、独立测试和排序条件的不同边界。新边界有独立标识，以自身可知时间及首次有效越线决定确认。
