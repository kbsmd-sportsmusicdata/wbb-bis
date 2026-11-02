# Six-team 2026 scouting pilot

This is the historical development pilot requested for Washington State-California, Maryland-South Carolina, and Oklahoma-North Carolina. It uses season-ending year 2026 (2025-26) and labels the November 2, 2026 opening matchups as context. It does not replace the active 2027 recurring report.

## Rebuild

Use Python 3.12 with requirements-scouting.txt installed. From the repository root:

~~~sh
python -m wbb_scouting pilot --season 2026
~~~

The first command discovers the exact women's college basketball release tags, downloads and checksums all four requested files plus supporting schedules/team membership, and writes the independent data/scouting/pilot_2026 folder.

To reproduce a build from the already verified national snapshots:

~~~sh
python -m wbb_scouting pilot --season 2026 --local-input data/scouting/pilot_2026/raw
~~~

The pilot refuses --publish. The active Monday/Wednesday/Friday workflow and production season are unchanged.

## Dataset layers

- raw: six original national snapshots and their downloaded SHA-256/source metadata. raw/extracted contains only the six featured teams' four original-column datasets.
- clean: string IDs, validated box groups, roster records and shot rows. Opponent boxes and player records are retained in separate support tables for validation. Schedules and team references supply authoritative identities and game times.
- processed: team_games with opponent-derived denominators, player_games, roster union and shot_events.
- features: team_summary, player_summary, team_trends and shot_zones. Season and team-last-five windows are separate rows.
- quality: shot-group reconciliation and box quarantines.

The 22 tables each have CSV and Parquet versions. manifest.json records rows, columns, grain, logical hashes, file checksums and changed-table flags. An unchanged real rebuild reported changed_tables=0. report_data.json is an inspectable intermediate; the HTML binds data reloaded from the saved processed and feature files.

## Verified six-team coverage

| Team | Team games | Player box rows | Roster snapshot | Historical players | Accepted shot games |
|---|---:|---:|---:|---:|---:|
| Washington State | 34 | 408 | 12 | 12 | 33 / 34 |
| California | 36 | 468 | 12 | 13 | 34 / 36 |
| Maryland | 33 | 495 | 13 | 15 | 33 / 33 |
| South Carolina | 40 | 467 | 8 | 12 | 40 / 40 |
| Oklahoma | 34 | 374 | 10 | 11 | 31 / 34 |
| North Carolina | 36 | 504 | 12 | 14 | 36 / 36 |

Totals: 213 featured team-game rows, 2,716 player-game rows, 67 source roster records, 77 historical roster/box participants, and 17,182 shot records including free throws. Those team-games represent 209 unique games because the featured teams sometimes played one another. Opponent context is additional and is not included in these featured counts.

All featured team and player box groups pass the existing identity, score and player-minute checks. Six team-game shot groups are withheld: three have incomplete/missing aggregate attempts, and three have player attribution discrepancies. The exact game IDs, expected/observed attempts and reasons are in quality/shot_groups.csv. Team/player box metrics use their full validated samples; shot features use 207 of 213 team-games. Never scale a partial shot game to fill missing attempts.

## Shot identities and coordinates

The ESPN shots release has no unique source event ID. It is a multiset: identical free throws can be separate actual attempts. Clean source_row_id is an internal fingerprint plus occurrence counter, stable under input reordering and preserving repeated events. It is not an ESPN event identifier. A game's shot rows may be replaced as a complete validated group during later integration; an append-only union of guessed event IDs is unsafe.

Only 2- and 3-point attempts enter shooting features. FGA, FGM, 3PA and 3PM must reconcile at both team and player grain. DNP box placeholders with blank stats do not imply missing attempts. An unmatched shooter or mismatch withholds the group.

Court coordinates use ESPN raw court width and basket-relative length, transformed to a 50-foot-wide court with the basket 5.25 feet from the baseline. Small negative basket-relative lengths can still be in bounds. Missing/sentinel coordinates are retained as attempts but omitted from location measures. All featured field-goal coordinates passed physical bounds in this build; invalid free-throw sentinels are not plotted. The half-court chart labels backcourt attempts omitted from its view, while zone denominators retain all accepted field goals. Geographic zones are source-coordinate approximations, not tactical play classifications.

## Report interactions and boundaries

The dashboard includes paired team/four-factor profiles; rolling five-game team ratings; player production; roster and production; a made/missed shot map; zone shares and efficiency; historical game details and player logs; coverage, source files and table links.

Matchup/window controls affect the paired team summaries and relevant detail views. Focus team and inspected player control the detail sections. The roster table intentionally retains the whole team's rotation; inspecting a player changes shooting and game history. Outcome/value controls are local to the map, so zone FG% retains makes and misses. On a last-five view, player appearances come from the team's selected games.

The roster is a union by season/team/player ID. Ten box participants are absent from the downloaded roster snapshot and are labeled box-only. That label does not establish a transfer, injury, availability or current eligibility. A player with no appearances retains unavailable per-appearance rates.

Rates use summed numerators/denominators and the existing documented 0.44 box possession estimate. There are no national percentiles, live predictions, inferred lineup tactics or current-roster projections. Ranking numbers are user-supplied matchup labels, not verified NET ranks.

## Portable delivery

Open index.html in the local preview or extract six_team_pilot_2026.zip and open its index.html in a browser. The archive includes the four selected-team raw extracts, every clean/processed/feature/quality CSV and Parquet table, manifest, configuration and source provenance. The full national originals remain in the source cache (the local raw folder by default, or the supplied --local-input folder) and at the source URLs; they are not duplicated in the portable bundle.

Report JSON and player/map CSV actions export the selected report scope. Dataset links refer to files in the extracted bundle. The report has no external script, font or API dependency.

## Sources

The exact SDV release families are:

- ESPN WBB team boxes: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_team_boxscores
- ESPN WBB player boxes: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_player_boxscores
- ESPN WBB shots: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_shots
- ESPN WBB rosters: https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_rosters

The original NCAA-family RAPM file is separate and is not used by this pilot. Source inventory, published digests and source row counts are recorded in raw/sources.json.
