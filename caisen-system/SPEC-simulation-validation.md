# Module Specification: simulation-validation

Status: approved for implementation under the user's 2026-10-24 instruction to continue the build automatically.

## Outcome

Use Backtrader as the sole simulation broker/execution engine for this phase. Verify order timing, filled/rejected status, position quantity, zero-fee cash accounting, and end-of-data valuation. Strategy return quality is explicitly deferred.

## Execution baseline

- Initial cash: 100,000 CNY by default.
- Commission, minimum commission, and slippage: exactly zero.
- Signal is known after the close of its signal bar; entry is attempted once at the next daily bar's open. The open must remain above structural stop, below first target and at/below maximum allowed entry; otherwise record a rejection and do not retry that signal.
- Whole-lot sizing uses current cash and configured lot size. A symbol/timeframe can hold at most one position at a time.
- Place the structural stop before the first target in the bracket so a bar touching both is handled conservatively as stop-first. Report this rule.
- For this execution-validation phase, close the full position at the first unlocked target. More advanced partial exits and conditional second-wave management remain future behavior; strategy quality is not evaluated here.
- Positions open at the end of data are marked to the final close and shown separately from realized cash; they are not silently treated as closed trades.

## Accounting

Return starting cash, ending cash, available cash, marked position value, equity, completed trades, open position, order lifecycle events, rejected signals and invariant checks. Assert `ending equity = available cash + marked open position value` and zero commissions. Preserve order references and dates so results can be reconciled to input bars. Daily data is the execution feed for both daily and weekly signals; incomplete weekly bars cannot create signals.

## Verification

- A deterministic fixture checks next-open execution, structural stop/target bracket, no retry after invalid open, lot/cash cap, and zero-fee arithmetic using Backtrader.
- A stop/target same-bar fixture verifies stop-first behavior.
- An open end-of-data position reconciles its marked equity.
- No-calendar weekly input produces zero formal weekly signals.
- Run simulation tests and all Caisen tests. Do not report profitability metrics as evidence of validity.
