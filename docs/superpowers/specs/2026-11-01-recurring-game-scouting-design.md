# Recurring WBB pregame and postgame scouting

Status: approved November 1, 2026, with Monday/Wednesday/Friday refresh cadence; collection, scheduling, and publication are not activated.
Date: November 1, 2026. User-facing timezone: America/Los_Angeles.

## Intent and agreed requirements

Build a recurring scouting dashboard with team and player evidence before and after games. The report cohort is the union of the NCAA women's NET top 50 and every current-season ACC team. A team meeting both conditions appears once. The approved refresh days are Monday, Wednesday, and Friday; postgame reports become available on the next scheduled check, subject to source publication. The refresh runs on GitHub independently of the user's Mac. Published SDV dataset downloads are the selected starting approach.

Repeated ingestion must not duplicate games, player box scores, or PBP events. It must also incorporate legitimate source corrections. Collection must preserve evidence of exactly what each report used.

## Current project and constraints

The existing repository is `kbsmd-sportsmusicdata/wbb-bis`, with default branch `main`. The connected account has write access; the repository is public. The proposal keeps code and public-data reports in that repository. Access-controlled scouting notes or a private deployment would require a separate sharing decision before publication.

The inspected local checkout has substantial unrelated work. Preserve it. `config.py` targets season 2026 (2025-26) and fixed completed-season files. `scripts/07_build_dashboard.py` stops at `NotImplementedError`. Existing scripts expect ESPN-style identifiers and columns. There is no existing `.github` workflow directory in this checkout. The new recurring workflow must be independently runnable; it must not invoke the unfinished dashboard step or overwrite the completed-season configuration, approved reports, or historical source files.

The repository uses Python, pandas, numpy, and pyarrow. New work should extend that stack and keep its metric conventions. No database service or additional paid storage service is required for version one.

## Approaches considered and selection

1. **Published SDV season files, scheduled on GitHub — selected.** Straightforward ingestion with coherent source snapshots, broad opponent coverage, and correction handling. Availability depends on SDV publication timing; a morning report cannot guarantee data SDV has not published.
2. **Published files plus game-level source requests.** Appropriate for reports shortly after the final buzzer, but adds request scheduling, source reconciliation, and maintenance. The user chose morning reports and then Monday/Wednesday/Friday refreshes, so this is deferred.
3. **Run the same collector on the user's Mac.** Reuses the local environment but depends on machine availability. The user chose GitHub execution instead.

## Approved defaults and cadence adjustment

The user approved the design and defaults, changing the refresh frequency from daily to Monday, Wednesday, and Friday. The following cadence-dependent details apply that adjustment:

- Start with 2026-27, stored under season-ending year `2027`; retain 2026 as a separately labeled historical reference.
- Run at 06:17 Pacific on Monday, Wednesday, and Friday, with an 08:17 Pacific retry on those days only when the primary check had failed or had pending publications. Successful complete primary runs do not trigger another source refresh. Include a manual refresh button. Use timezone-aware GitHub scheduling and one non-overlapping publication job.
- Collect published Division I coverage broadly; restrict report selection to NET top 50 plus ACC. Broad collection supplies opponents and national comparison baselines without repeated cohort-specific backfills.
- Before current-season NET exists, use a verified prior-season NET snapshot for a provisional top-50 selection, with its season and publication date displayed. Current-season NET remains unavailable, rather than inheriting old numeric ranks. If no verified prior-season snapshot is available, show ACC teams and an explicit pending top-50 selection. Switch to dated current-season NET once published.
- Deliver an interactive, standalone HTML dashboard and machine-readable report exports as GitHub Release assets. This gives a durable downloadable dashboard without requiring a hosting account. A continuously hosted browser URL and access controls are a later delivery decision.

NCAA states that NET first appears in December. Do not fabricate a current-season November NET. ACC membership is season-specific and comes from a checked team/conference reference, not names or last year's roster files.

## Sources and dataset boundaries

### Required for the initial recurring workflow

| Dataset | Source family | Purpose |
| --- | --- | --- |
| Schedule/results | SDV ESPN WBB releases | Upcoming opponents, tip times, game status, expected completed games, final score |
| Team game box scores | Same ESPN family | Team performance, opponent results, denominator checks |
| Player game box scores | Same ESPN family | Rotation, minutes, player production and recent form |
| Play-by-play | Same ESPN family | Supported event and shooting splits, scoring sequences and game context |
| Rosters/team references | Same ESPN family plus checked conference reference | Player/team identifiers and current-season ACC membership |
| Women's NET | Official NCAA ranking publication | Dated ranking snapshots and report cohort selection |

Discover actual published assets and their schemas during implementation; do not construct unverified filenames and treat a successful HTTP request as proof of season coverage. All input sources have a manifest entry with provider, release tag, canonical URL, season, asset metadata, retrieval time, checksum, row count, and schema fingerprint.

SDV's `ncaa_wbb_*` datasets, including the RAPM file downloaded in this chat, are a separate stats.ncaa.org-based family. Keep them in their own namespace. Add possessions, lineups, shot charts, and RAPM only through validated identifier mappings and documented coverage. They are enhancements, not prerequisites for a traditional box-score report. Do not substitute NCAA IDs for ESPN IDs or mix model estimates with box-derived measures under one label.

An update can require downloading an entire changed season file. Incremental storage does not imply the upstream server offers row-level incremental downloads. Check release metadata and use conditional requests where supported; unchanged content must not generate duplicate stored snapshots.

## Canonical storage and reconciliation

Store canonical tables as Parquet, partitioned by source family, dataset, season, and game month. GitHub Releases retain durable data assets and manifests between otherwise temporary GitHub Actions runners. Each successful run's manifest can reference unchanged assets from earlier runs; only changed partitions and genuinely new raw source snapshots need another upload. Keep original downloaded bytes for changed source versions, referenced by checksum. Split any asset approaching the release file-size limit.

Code, source definitions, metric definitions, and tests belong in normal Git history. Large downloaded and generated datasets do not. GitHub caches and expiring Actions artifacts may accelerate or diagnose a run but are never the only persistent store.

### Identity and update rules

Validate actual unique keys in each published schema before enabling a dataset. Candidate logical identities are:

| Table | Candidate identity |
| --- | --- |
| Schedule | source family + season + game ID |
| Team box | source family + season + game ID + team ID |
| Player box | source family + season + game ID + team ID + player ID |
| PBP | source family + season + game ID + source event ID, when reliable |
| Rosters | source family + season + team ID + player ID |
| NET | ranking source + ranking season + publication date + canonical team ID |

Identifiers remain strings; display names are not identities. Use explicit adapters for column changes, missing identity, and event order. Never invent an event key from clock time alone. Conflicting duplicates inside a source snapshot are a validation problem, not a last-row-wins decision.

For each validated game/dataset group:

1. A new game adds records.
2. Identical normalized records make no changes.
3. A changed, validated game replaces that game's dataset group in a staging copy. This accommodates corrected values and events removed or renumbered upstream.
4. Absence of an entire previously stored game from a new season file never automatically deletes it. Flag an unexplained coverage regression. Partition replacement requires a complete incoming game group with dataset-specific checks; uncertain groups are quarantined.

Record added, changed, unchanged, removed-within-replaced-game, rejected rows, and affected games. A repeated identical input must produce zero logical mutations. Preserve prior manifests and source versions so a correction is inspectable and reversible.

## Validation, publication, and freshness

Stage downloads and canonical changes away from the last good published state. Reject unreadable files, unexpected season/source identity, required-column loss, null keys, conflicting duplicate keys, unresolved team mapping, and invalid numeric/stat relationships. Validate the women's sport and Division I scope explicitly. Do not treat non-D1 opponents as D1 comparison teams; retain eligible D1 teams' games against non-D1 opponents with an explicit opponent-division flag and separate comparison eligibility.

For completed games, compare expected schedule coverage with team boxes, player boxes, and PBP separately. Require two reciprocal team box rows when that source contract applies. Reconcile team/player point totals and final score, allowing only documented source-specific exceptions. Check PBP's final score and scoring-event totals where fields support those checks; unsupported reconciliation remains flagged. Do not sum all `score_value` entries as points: the source can carry an attempted shot's value on misses.

Publish current status per game and per dataset: complete, partial, pending source publication, or failed validation. Missing PBP must withhold PBP-dependent conclusions while allowing a clearly labeled box-score-only postgame report. Preserve last good data after failure and display its actual age. A source outage must not silently turn into a zero-game success or erase a previous season.

The dashboard and exports carry last attempted refresh, last successful validation, latest usable game date by dataset, expected completed games, available games, missing game IDs, ranking date, report cutoff, source snapshot IDs, and sample sizes. A maximum game date alone is not proof of complete coverage. Prior-year history must never satisfy current-season freshness checks.

Upload a run's assets to a new run-specific draft release; verify checksums and references before publishing its manifest and marking the release available. One job owns state publication. Select the latest valid scouting manifest explicitly rather than assuming the repository's generic latest release is scouting state. Never replace the only known-good assets in place. Failed publication leaves the previous published state available.

Unchanged source data can still produce a small new refresh-status/report manifest referencing the same data assets. If publication itself fails, GitHub Actions records the failure; the older dashboard's timestamp and age indicator show that it is stale. No automatic historical cleanup in the first version. Revisit retention and external storage after measuring actual size.

## Scouting report behavior

### Cohort and chronology

Save dated cohort membership at each scheduled or manual refresh with inclusion reason: NET top 50, ACC, or both. Preserve the ranking publication date separately from the collection date. Keep reports for teams that later leave the top 50. Historical reports retain the ranking/cohort snapshot used when issued. Current selectors use the newest valid cohort snapshot, with stale or provisional membership clearly marked.

Pregame reports use only games completed before their cutoff and before the target game's tip. Store report generation time, planned tip time, input snapshot references, and calculation version. Save issued reports unchanged; a corrected report is a new labeled revision. A retrospective report generated after a game is labeled reconstructed, not presented as a pregame report that actually existed then. Opponent ranks and other contextual inputs must respect the same as-known cutoff, or be labeled unavailable/reconstructed.

Issuance policy with the approved cadence: the first successful primary refresh issues a pregame edition for each cohort matchup scheduled to tip before the next Monday/Wednesday/Friday primary check. Compute that boundary in America/Los_Angeles, allowing 48-hour and weekend 72-hour windows and daylight-saving changes rather than a fixed 24-hour lookahead. Display when the report's inputs were current. A same-day retry issues a new revision only when relevant data, cohort/ranking context, or the scheduled matchup has changed; an identical rerun does not create another edition. A manual refresh uses the same next-primary-check boundary. Enforce the actual known tip-time cutoff again before issuance, including reschedules. The postgame comparison references the latest qualifying issued pregame edition from before tip. If none exists, compute a clearly labeled reconstructed baseline from earlier games and do not imply it was an issued pregame report. Save partial pregame editions with their unavailable sections so later improvements remain attributable. Reports for off-day games arrive at the next scheduled check; do not promise next-morning delivery every day.

Proposed bootstrap policy: ingest all available current-season statistics and the separately labeled 2026 reference, but do not mass-backfill historical pregame reports. On the first run, generate postgame reports for eligible games from the preceding seven calendar days, labeled with their actual generation time and reconstructed baseline where needed. Thereafter, generate postgame reports for newly validated completed cohort games, including delayed source arrivals, regardless of how long publication took. Keep the corresponding source delay visible. Older report backfill is an explicit later request; older game statistics remain available for calculations and exploration.

### Initial pregame content

- Upcoming matchup, venue, date/time, and opponent identity.
- Season-to-date and last-five-completed-game team and player production with games/minutes/attempts denominators.
- Scoring efficiency, shooting mix, turnovers, rebounding, and free-throw pressure; distinguish own performance from opponent allowances.
- Recent rotation/minutes and documented matchup differences supported by the available tables.
- Coverage and small-sample caveats. DNP or low minutes alone does not establish injury or availability.

### Initial postgame content

- Final result, team and player box scores, and game-level validation status.
- Performance versus each team's pregame baseline, including shooting, turnovers, rebounding, and free throws.
- Player minutes and production changes, with explicit comparison periods.
- PBP-supported shooting/scoring context only when its coverage passes validation.
- Evidence-linked observations. Do not infer defensive schemes, play types, or possession causes that the source cannot establish.

Zero current-season games produces a legitimate empty-season state. Prior-season context is labeled separately. Missing attempts or denominators produce unavailable rates, not zero efficiency.

### Metric conventions

Follow `docs/metric_conventions.md`. Box-derived possession and true-shooting estimates use the existing 0.44 free-throw coefficient and expose their formula. For each team, start with `FGA - OREB + TOV + 0.44 * FTA`; estimated game possessions are the mean of the two teams' estimates, with an invalid or nonpositive estimate withheld. Offensive rating is `100 * team_points / estimated_game_possessions`; the corresponding defensive rating uses opponent points and the same denominator. Label these as estimated rather than possession-engine measurements. The initial Four Factors are eFG% = `(FGM + 0.5 * 3PM) / FGA`, turnover rate = `TOV / estimated_game_possessions`, offensive rebound rate = `OREB / (OREB + opponent_DREB)`, and free-throw rate = `FTA / FGA`. True shooting is `points / (2 * (FGA + 0.44 * FTA))`. These are the proposed version-one definitions; rates with unavailable or zero denominators are unavailable.

Do not combine estimated possessions with possession-engine counts in one comparison. Four Factors and rolling rates use aggregate numerators/denominators, not an unweighted average of per-game percentages. Maintain a metric dictionary and comparison eligibility flags.

## Operations and rollout

GitHub Actions runs scheduled and manually requested refreshes on the default branch. Primary checks run Monday, Wednesday, and Friday at 06:17 America/Los_Angeles; 08:17 on those same days is a conditional retry only for a failed primary check or pending source publications. The retry's gate must inspect the latest primary attempt and its persisted status; a missing primary attempt is also actionable. No routine Tuesday, Thursday, Saturday, or Sunday checks. Exact start times are best effort; GitHub can delay scheduled jobs. Detect missing expected publications and report gaps rather than claiming a guaranteed morning completion time. Calculate next-run and stale-state indicators against this schedule, so the normal weekend interval is not incorrectly labeled a missed daily run. Public repositories can have schedules disabled after prolonged inactivity; document that operational limitation and expose last-run age.

Use the repository-scoped Actions token with only necessary permissions for releases. Keep credentials out of code, manifests, logs, and HTML. Fetch public source data without requiring the user's personal machine. Normal success/no-change runs are quiet; failures and actionable coverage gaps are visible in run summaries and standard GitHub workflow notifications. Sending email or chat messages requires a separate request.

Build in this order after design review:

1. Source inventory, explicit schema adapters, ranking/team reference validation, and first-season availability audit.
2. Repeatable ingestion and durable manifests, with correction and failure behavior verified using representative inputs.
3. Scouting calculations and archived pregame/postgame report payloads.
4. Interactive HTML dashboard and exports; verify actual desktop/mobile interactions and missing-data states.
5. GitHub workflow and release publication; verify a real initial run and an unchanged rerun before calling the recurring system operational.

The user approves a concrete source/coverage and report result before activation. This design does not enable a schedule, publish a dashboard, or declare current-season data ready. Deployment timing depends on sources actually publishing current-season games.

## Acceptance evidence

- Selected cohort is NET ranks 1-50 union season-correct ACC membership, with no duplicate teams and truthful provisional handling.
- Identical repeated input yields zero additions, updates, or removals.
- A new game is added once. A correction updates the intended game and any dependent metrics without keeping a duplicate obsolete event.
- Conflicting keys, truncated games, schema drift, incorrect season/sport, missing team mappings, and failed reconciliation retain last good state with an explicit failure or quarantine result.
- Expected-versus-available coverage identifies a missing game even when a later game exists.
- Pregame calculations exclude the target game and all future information; issued reports retain their original snapshots.
- Monday/Wednesday/Friday scheduling, conditional retry behavior, weekend pregame coverage, and timezone/DST boundaries are verified. Normal weekends do not trigger false missed-run warnings.
- A fresh GitHub runner restores durable state without any local files or cache.
- Partial source updates produce truthful report availability, and a failed release upload cannot corrupt the previous state.
- Dashboard selectors, cohort reasons, pregame/postgame views, date cutoffs, sample sizes, exports, and mobile layout work in a real browser.
- A verified cloud run and unchanged rerun produce saved logs, row/game counters, manifests, output links, and evidence of the active schedule. A saved workflow file alone is not completion.

## Builder-owned choices and limits

The builder may choose internal module layout, exact schemas after live profiling, partition sizing, compatible dependency pins, small test fixtures, chart library, visual styling, and bounded retry intervals. These must preserve the requirements above. Browser hosting, private access, notification channels beyond GitHub's existing settings, live-game reports, automated lineup reconstruction, and advanced model integration are later scope decisions.

## References inspected

- Existing `README.md`, `pipeline.py`, `config.py`, `data/raw/SOURCES.md`, `scripts/01_player_box_processing.py`, `scripts/02_pbp_player_processing.py`, `scripts/05_merge_scouting.py`, `scripts/07_build_dashboard.py`, and `docs/metric_conventions.md` in this checkout.
- [wehoop ESPN WBB release loaders](https://wehoop.sportsdataverse.org/reference/load_wbb_game_rosters.html).
- [wehoop NCAA WBB dataset family and identifiers](https://wehoop.sportsdataverse.org/reference/load_ncaa_wbb_lineups.html).
- [SDV publication status](https://www.sportsdataverse.org/status), inspected November 1, 2026. At inspection the ESPN WBB publisher was flagged failing and multiple NCAA-family products were through season 2026; this status can change and is not a substitute for file-level coverage checks.
- [NCAA explanation of NET and December release](https://www.ncaa.org/news/march-basketball/media-center-how-do-net-rankings-work-in-ncaa-tournament-selection/).
- [GitHub scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
- [GitHub Release assets and storage limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). The documented per-asset limit is under 2 GiB; verify actual proposed asset sizes.
