# Six-team 2026 scouting pilot

Requested directly by the user: extract Washington State (265), California (25), Maryland (120), South Carolina (2579), Oklahoma (201), and North Carolina (153); build raw, cleaned, processed, feature datasets and populated report visuals from 2026 team boxes, player boxes, shots and rosters before 2027 results arrive.

## Approach

Reuse the existing Python/standalone HTML scouting architecture in an independent `pilot` command and state folder. This provides reusable stages and an inspectable interactive report without injecting historical player results into the active 2027 collector. Modifying the live season or replacing it with a separate frontend would create unnecessary season/roster ambiguity. The scoped pilot is the implementation choice for this request; its concrete output remains reviewable before any production integration.

## Data contract

All four requested files are from the exact `espn_womens_college_basketball_*` SDV release tags, season-ending year 2026 (2025-26). Supporting 2026 schedules/team membership validate games, opponents and IDs. Preserve downloaded national bytes and checksums, then extract the six featured teams; retain opponent context separately for reciprocal box checks and denominator calculations. Save original subset columns in raw/extracted. Clean uses string IDs and complete validated game groups. Processed enriches game/player records; features aggregate numerators and denominators. Both CSV and Parquet are available for clean/processed/features. Every table has explicit grain and fingerprints in a manifest. Identical rebuild produces zero table changes.

Roster displays union the official 2026 roster with box participants, joined by team/season/player ID; box-only players remain visible and labeled. No 2026 player is silently treated as a 2027 roster member. Missing roster metadata stays null.

Shots have no unique source event ID. Preserve repeated identical events as a multiset with deterministic internal row IDs, never drop identical free throws. Retain FT records in clean data but exclude them from field-goal spatial denominators. A team-game's FGA/FGM/3PA/3PM must match its validated box before shots enter features. Player attribution reconciles to corresponding player boxes. Incomplete games are withheld, not rescaled. Bad/sentinel coordinates do not erase otherwise valid attempts; show mapped versus total attempts. Use source raw coordinates after verified conversion, 94x50 foot court and basket 5.25 feet from baseline. Recompute rates from aggregate counts; zero denominators produce null.

## Report

The three requested pairings are confirmed in the 2027 schedule for November 2, 2026 Pacific: Maryland-South Carolina 09:00 PST; Oklahoma-North Carolina 11:30 PST; Washington State-California 19:00 PST. These are matchup context only. Numeric ranking labels are user supplied, not current NET. Statistics and rosters remain explicitly 2025-26 historical reference.

Sections: paired team/four-factor comparison; team game/rolling trends; player production and roster table; made/missed shot court with team/player/outcome filters; shot-zone volume and efficiency with denominators; selected player game log; complete source/validation view and data exports. Season and last-five windows change appropriate data together. No national percentiles, availability, tactical causality or 2027 player projections are invented. A changed roster is not inferred as a transfer from absence alone.

## Acceptance

All six teams have actual source-backed data. Hand-checked feature formulas, duplicate-FT preservation, order-stable shot IDs, incomplete shot group withholding, roster union, season isolation and unchanged rebuild have tests. Every dashboard section uses processed/features outputs; desktop/mobile controls, all three matchups, player switching, exports and console are exercised in a real browser. Preserve the original dirty checkout and active M/W/F workflow. Build/review a feature branch; do not merge or replace production without the user's concrete-result approval.
