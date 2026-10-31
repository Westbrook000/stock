# Causal Pattern Path Migration

## Boundary in this release

The current analysis, scan, decision, and fixed Backtrader v2 paths use only
the causal main-structure engine. The migrated families are W bottom, inverse
head-and-shoulders, and triangle bottom. Unsupported families stay in the
ten-family rule catalog and are reported as `unsupported` with reason code
`not_migrated_to_causal_main_engine`.

| Family | New path status |
| --- | --- |
| W bottom | Supported |
| Inverse head-and-shoulders | Supported |
| Triangle bottom | Supported |
| Spring | Unsupported |
| Bull flag | Unsupported |
| Bear flag | Unsupported |
| M top | Unsupported |
| False breakout | Unsupported |
| Head-and-shoulders | Unsupported |
| Triangle top | Unsupported |

The frozen v1 Backtrader strategy remains an archival replay path. It does not
feed current analysis, scans, decisions, or v2 simulations.

## Staged migration

1. Keep this manifest synchronized with the rule catalog and runtime
   `migration_status()` response. Fail tests if a catalog family lacks a
   status.
2. For each unsupported family, specify causal geometry, confirmation,
   invalidation, timeframes, and required evidence before implementation.
3. Add deterministic synthetic and real-history recognition cases, including
   as-of-prefix checks, then add family routing to analysis and scans.
4. Add buy or sell decision policy only after recognition acceptance. Keep
   unsupported families out of current decisions until that stage passes.
5. Add Backtrader v2 replay coverage and separate family-level reports. Do
   not treat frozen v1 replay results as evidence for v2 behavior.

Every stage must preserve daily and weekly outputs as independent evidence.
A qualifying timeframe can independently produce a buy recommendation; a
missing or unsupported family on the other timeframe does not veto it.
