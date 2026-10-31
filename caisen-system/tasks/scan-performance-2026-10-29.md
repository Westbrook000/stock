# Market scan worker benchmark — 2026-10-29

## Method and inputs

Ran the same first 10 locally available stock CSVs in `data/stocks`, through full `1d` + `1w` analysis on macOS with the project Python 3.9 environment. The serial and process runs called the same top-level analyzer; the process run used a `spawn` pool capped at 10 workers. Thread and process results were compared to serial results as Python values. A separate second round repeated serial and process with the explicit market-root argument now used by spawned workers.

| Code | Available data |
|---|---|
| 000001 | 1991-04-03 to 2026-10-29 |
| 000002 | 1991-01-29 to 2026-10-29 |
| 000006 | 1992-04-27 to 2026-10-29 |
| 000007 | 1992-04-13 to 2026-10-29 |
| 000008 | 1992-05-07 to 2026-10-29 |
| 000009 | 1991-06-25 to 2026-10-29 |
| 000010 | 1995-10-27 to 2026-10-29 |
| 000011 | 1992-03-30 to 2026-10-29 |
| 000012 | 1992-02-28 to 2026-10-29 |
| 000014 | 1992-06-02 to 2026-10-29 |

## Results

| Round | Serial | 10 threads | 10 spawn processes | Equality | Peak process-tree RSS |
|---|---:|---:|---:|---|---:|
| 1 | 22.485 s | 24.109 s | 6.422 s | thread and process outputs exactly equal serial | ~1,323.9 MB |
| 2 | 21.599 s | not repeated | 6.321 s | process output exactly equal serial | ~1,260.7 MB |

The RSS figure sums sampled `ps` resident sizes for the parent and child PIDs; macOS shared pages can be counted more than once, so treat it as an approximate upper bound. During both measurements the user's separate full scan (PID 88762) was still running and competing for CPU. These 10-symbol timings show the worker-level difference; they do not predict full-universe runtime or memory exactly.

## Runtime behavior

The scan now uses spawn-based processes for analysis only. The coordinator keeps SQLite writes, result ordering, and progress updates. At most 20 futures are submitted at once for a 10-worker pool. FastAPI lifespan shutdown signals active scan coordinators and cancels queued work; completed rows and patterns are retained and runs are marked `interrupted`. A coordinator-level spawn test confirms it exits, keeps database rows consistent with the in-memory partial results, and leaves no child workers after in-flight work drains.
