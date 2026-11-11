# V8 restoration audit and research ledger

Status: incomplete source recovery; production unchanged. No new 100-account reproduction was run.

## Completed

- Archived source, tests, scripts, docs and uncommitted Git patch.
- APFS-cloned complete reports and research folders; path/size/mtime inventories match.
- Checked Git history, reflog, stash and unreachable objects; no exact v8 commit found.
- No active old research backtest process observed at preflight. Stale running manifests are not proof of a live process.

## Missing

- Exact matching v8 rule catalog: snapshot-v8 contains empty directories
- Complete v8 execution and dependency source provenance
- Reproduction of historical signals, orders and ledgers on the paired 100 accounts

## Research findings

|Finding|Evidence and limits|Disposition|
|---|---|---|
|Directional suppression|Bidirectional and cascading suppression are candidate bugs.|Separate experiment after baseline restoration.|
|Same troughs, new neckline|A genuinely different boundary can have its own first signal after availability.|User-approved research rule; not mixed into v8.|
|Triangle apex|Changing upper boundaries requires recalculating apex and 50%-75% breakout window.|000008 smoke only; not 100-account evidence.|
|Dynamic neckline V2|100 accounts: win rate 34.47%, return 302.19%, worst account drawdown 90.20%.|Stored v12 structures only; not a complete new recognizer.|
|Trade removal|Matched 100: v8 3780 trades/33.7566% wins/+61.3449%; v12 1669/33.6129%/+27.1279%.|Fewer trades did not establish better accuracy.|
|Performance|Reuse arrays, pressure pairs and interval calculations; parallelize by symbol.|Require identical signals and ledgers; report cold and warm timings separately.|

## Recovery gate

Do not rename v9 as v8. Recover exact catalog and matching dependencies, then reproduce historical signal dates, frozen boundaries, orders and account ledgers before switching scan consumers. If only reconstructed candidate code is available, label it unverified until parity is demonstrated.

## Remaining

- Complete source recovery and paired-100 reproduction.
- Daily/weekly scan and cache migration.
- Independent v8-based experiments and new Backtrader reports.
