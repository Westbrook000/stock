# Module Specification: decision-engine

Status: Base scope approved for implementation under the user's 2026-10-24 instruction to continue the build automatically. Amendment approved by the user on 2026-10-27; implementation authorized.

## User outcome

Translate current daily and weekly pattern evidence into a clear action label, an auditable plan, and a deterministic scan order. The scan must put the best current buy candidates first, while showing why each ticker qualified or did not. Ranking is a rule priority, not an estimated probability or promise of return.

## Actions and position semantics

- `CONSIDER_BUY`: a confirmed bullish setup passes structural risk and reward-space gates at the reference price.
- `WAIT`: evidence is forming, conflicts across timeframes, the price/space gate fails, or the candidate needs a trigger/retest.
- `HOLD`: a valid bullish structure exists but there is no fresh entry; this does not assume a saved user position.
- `SELL_IF_HELD`: confirmed bearish evidence or structural invalidation is a sell warning for an existing long position. For a user without a position, show “不买入 / 规避”，not a short-sale instruction.
- `INSUFFICIENT_DATA`: no formal usable structure can be evaluated or required bars/calendar are absent.

No broker order is sent. The engine has no position ledger in this module; it must never claim that a user should sell shares it does not know they hold.

## Buy plan

- Only `w_bottom`, `inverse_head_shoulders`, `bull_flag`, and `triangle_bottom` can create buy plans. The other six active structures remain visible as context or risk alerts but never enter buy ranking.
- The stop is derived from the pattern's frozen structural invalidation level and a documented buffer. Do not move the stop to satisfy a risk cap.
- Maximum structural risk defaults to 7% from reference entry; first unlocked target must offer at least 2R. Fees, commission and slippage are zero for the user's requested acceptance baseline.
- The plan returns reference entry, stop, risk percentage, first target, R multiple, maximum allowed entry and explicit rejection reason codes.
- If no first target is unlocked, the reward-space test fails; a conditional second-wave target cannot substitute for it.
- A forming member of the four-shape buy list remains visible with its formed label and returns `WAIT` until its own frozen neckline or structure boundary is closed through. A forming bullish pattern with an existing position may retain `HOLD` position semantics. Price-confirmed buy-list patterns with unverified volume remain visible for research, but their trade plan is `WAIT` with `volume_data_unverified`; unverifiable volume/corporate-action provenance cannot produce a `CONSIDER_BUY` result.

## Multi-timeframe synthesis

Daily and weekly analyses remain independent. Same-direction confirmed structures reinforce; conflicting confirmed directions move the symbol to `WAIT` and the result explains the conflict. Missing calendar-complete weekly data is marked unknown, never interpreted as neutral confirmation. One instrument appears once in the full scan with its strongest buy plan and all timeframe evidence attached.

## Deterministic scan priority

Sort by these keys in order:

1. `CONSIDER_BUY` passing all hard gates before every other action.
2. More confirmed timeframes before one; same-direction agreement before conflict.
3. First-target R multiple descending, capped for ordering so extreme targets do not dominate.
4. Lower structural risk ascending.
5. Freshness of the confirmation descending.
6. Stable code then kind ascending for ties.

Expose every input key and rejection reason. Do not expose an opaque score or call rank a probability. Non-buy actions follow in order: `WAIT`, `HOLD`, `SELL_IF_HELD`, `INSUFFICIENT_DATA` unless a product-level exit view explicitly filters sells.

## Verification

- Table-driven cases for buy, wait, hold, sell-if-held, insufficient data, daily/weekly agreement/conflict, over-7% risk, under-2R target space, locked target, and unknown volume quality.
- Zero-cost plan arithmetic reproduces exact risk and R calculations.
- Ranking tests cover each tie-break and stable determinism.
- Missing weekly calendar remains unknown and cannot improve a candidate's rank.
- No result contains a probability field or submits an order.
- Run decision-plan tests and the complete Caisen suite.

## Limits

These are screening rules, not proven investment advice or profitable strategy evidence. Human pattern calibration tools and labels are being retired; no accuracy estimate is claimed. Strategy-effectiveness testing remains deferred.
