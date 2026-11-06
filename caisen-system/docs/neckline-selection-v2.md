# Multi touch pressure boundaries

This document records the earlier v6 contract. For the current full-formation envelope and W/IHS 20% falling-line limit, see [neckline-envelope-v7.md](neckline-envelope-v7.md).

The main bottom recognizer now emits rule version `caisen-v20-main-bottom-v3` and source rule set `caisen-main-bottoms-v6`.
Its neckline selection evidence uses `pattern_matched_pressure_envelope_v4`.

## Selection contract

1. Build candidate boundaries from causally confirmed resistance pivots. A boundary may be horizontal, falling, or rise no more than the configured cumulative 5% across its independent tests. A small rising line is flattened and tested again as horizontal.
2. Deduplicate touches by pivot and require the configured trading-session separation plus a one ATR retreat between successive tests.
3. Check every raw bar high from the pressure segment start through the latest touch's confirmation date. The permitted excess is `max(2 ticks, 0.5 ATR14)` using ATR known at each bar. A boundary with any larger excursion is excluded.
4. Match the surviving line to the complete pattern geometry. W bottoms must cover the intervening rim. Inverse head and shoulders must use the lowest raw low in the full structure as the head; a right shoulder cannot supersede an earlier significant right-side trough. The recognizer keeps asymmetric shoulders within the configured 50% depth gap, requires each shoulder to rise at least one ATR above the head, and requires independent pressure evidence on both reaction legs. Triangles must match their upper boundary while keeping their separate converging support boundary.
5. Rank valid candidates by independent test count descending, test span descending, normalized residual ascending, latest known test descending, and anchor dates ascending. Price height, current close, future target, and later trade result never rank a line.
6. Freeze the line when its final supporting test is known. Track the later date when the full pattern geometry becomes known separately. A complete structure remains a waiting candidate until a subsequent close first reaches the buffered breakout threshold. The recognition date itself cannot create an order, and discovering an already-broken line does not create a retroactive signal.

For an inverse head-and-shoulders candidate, the recognizer keeps the latest significant right shoulder known before a breakout and the nearest valid left shoulder before the head. The same head/right pair with alternate left anchors is treated as competing geometry, not automatically as a parent that suppresses the user's intended structure. A nested candidate only suppresses a child when the parent uses an equal or coarser pivot scale and was already known at the child's action date.

The detail chart draws the interval with verified high coverage as a solid boundary and draws its frozen extension as dashed. The selection evidence includes ordered viable line alternatives and a date-bounded sample of rejection reasons. Historical invalidated attempts keep their own line and event date.

## Verified case anchors

- `600108` daily: W low points on 2024-02-06 and 2024-09-18; selected pressure around 3.32; confirmed on 2026-01-09.
- `516630` daily: W structure confirmed on 2024-10-08; four independent tests; the 2024-03-21 touch is about 0.973 on the gently falling line.
- `159718` daily: the corrected shoulders are 2023-08-22 and 2025-01-13 around the 2024-04-19 head. The structure became knowable on 2025-02-10. Its frozen pressure boundary uses independent tests on both reaction legs and later pre-breakout retests; it froze on 2025-04-28. The buy trigger is unchanged: a close must exceed the frozen neckline by the prior session's ATR. It confirmed on 2025-05-29; the next-session entry is 2025-05-30. The chart and fixed Backtrader replay are in [159718-neckline-review.html](../reports/159718-neckline-review.html).

Earlier loss-case matching reports were produced under older shoulder rules and are not evidence for v3/v6. The 159718 fixed 8% stop / 20% target comparison uses actual Backtrader broker replays with zero fees. It compares the old shoulder anchors and their July 2025 trigger with the corrected February/May structure and trigger. This one-trade result is a regression example, not evidence of overall strategy performance.

## Validation boundary

Automated tests cover independent touch grouping, horizontal-versus-falling competition, whole-interval high excursions, date freezing, raw-bar head selection, right-shoulder selection, and the 159718 breakout date. This example is development evidence only. No independent 120-case accuracy sample or full rolling strategy evaluation has been rerun for v3/v6.
