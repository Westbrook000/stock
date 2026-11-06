# Formation envelope and falling-neckline limits

User-confirmed on 2026-11-02.

- W bottoms and inverse head-and-shoulders: the resistance line may fall at most 20% from its first effective pressure touch through the observation / confirmation bar. A line's test-to-test fall alone is insufficient: its later extension also counts.
- Triangle bottoms and bullish flags have no falling-line percentage cap. Existing rising-line, convergence and flag geometry requirements remain.
- Check every raw high from the earlier of structure start / pressure start through the later of structure-known / line-frozen. Preserve the existing tolerance `max(2 ticks, 0.5 contemporaneous ATR14)`.
- After this formation interval, weak line crossings remain breakout attempts. The unchanged close trigger is the line plus `max(2 ticks, previous completed ATR14)`.
- Scanner/detail main-bottom construction and the daily incremental cached-breakout route use the same guards. Flag correction phases also require full upper-channel coverage.
- Historical confirmed events retain their original boundary and confirmation after invalidation; absence of a current eligible replacement must not erase them.

Versions: `caisen-main-bottoms-v7`, `pattern_matched_pressure_envelope_v5`, `caisen-v21-main-bottom-v1`, daily engine `caisen-daily-continuous-v3`. Archived v2 pages and trades remain archived results, not v3 replays. Existing v2 performance benchmarks do not automatically qualify as v3 benchmarks.

Reported 600108 failures:

| Confirmation | Failure | Frozen-line evidence |
| --- | --- | --- |
| 2014-08-04 | Excessive fall | 2013-11-28 pressure 9.52; confirmation line about 6.1479, fall about 35.4% |
| 2014-08-11 | Excessive fall | Same pressure segment; confirmation line about 6.0469, fall about 36.5% |
| 2021-09-15 | Formation not covered | 2020-08-19 high 4.43 versus line 3.795; structure only known on 2021-07-26 |

These are rejection regressions for the reported boundaries, not symbol/date exceptions. Other supported boundaries must still be evaluated generically. Focused tests also cover the exact 20% boundary, triangle/flag exemption, wicks inside tolerance, post-formation weak crossings, and cached signal rejection. Full recognition-quality statistics and strategy performance require separate reruns.
