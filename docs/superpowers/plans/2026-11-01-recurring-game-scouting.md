# Recurring WBB Game Scouting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-validated, correction-aware scouting dashboard for the NCAA women's NET top 50 plus ACC, refreshed on GitHub on Monday, Wednesday, and Friday.

**Architecture:** Add an independent `wbb_scouting` Python package beside the existing completed-season pipeline. Download and validate SDV season snapshots, reconcile game groups into partitioned Parquet, calculate reports with preserved cutoffs, and retain durable state and downloadable HTML in versioned GitHub Releases. A primary M/W/F check and conditional same-day retry use one publication job; existing historical artifacts remain untouched.

**Tech Stack:** Python 3.11+, pandas, numpy, pyarrow, standard-library HTTP/JSON/hashlib/zoneinfo, pytest, GitHub Actions, standalone HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-11-01-recurring-game-scouting-design.md` (approved November 1 with M/W/F amendment).

## Global Constraints

- Cohort: NCAA women's NET ranks 1-50 union every season-correct ACC team; deduplicate by canonical team identifier.
- Primary: Monday/Wednesday/Friday at 06:17 America/Los_Angeles. Retry: 08:17 on those days only after failed/missing primary or pending publication. Manual runs remain possible.
- Pregame coverage extends to the next primary check, including weekend games and timezone/DST changes. Postgame reports arrive on the next scheduled check, subject to source publication.
- Current season `2027` (2026-27), separate historical reference `2026`; prior-year data never establishes current-year freshness.
- Identical input produces zero logical mutations. Corrected games replace validated complete game groups; absent games never disappear automatically.
- Preserve source families and string identifiers. No name-only or float-based identity joins.
- Box-derived possession/TS calculations use 0.44 and the formulas in the spec. Missing denominators produce unavailable values.
- Save report inputs, ranking/cohort date, creation/cutoff time, and revisions. No future-game information in pregame calculations.
- Public code/data releases in `kbsmd-sportsmusicdata/wbb-bis`; no new paid services. Existing project files and unrelated local work remain untouched.
- Raw bytes, canonical Parquet, and report snapshots persist in Releases, not solely Actions caches or expiring artifacts. Individual assets must stay below 2 GiB; choose a lower practical shard cap.
- Verify concrete source coverage and output before activation. A saved workflow is not an operating recurring system.

## Review Focus

1. Source games disappear or PBP events are renumbered: preserve last good state or replace only a validated complete game group (Tasks 2-3).
2. NCAA NET is not published or official delivery is blocked: use a dated verified prior-year selection or ACC-only pending state, never fabricated current NET (Tasks 2 and 4).
3. The Friday-to-Monday interval crosses daylight-saving changes: cover all intervening games without daily missed-run warnings (Tasks 1 and 5).
4. A current-season box file is missing while previous-season files exist: report current-season pending coverage, not an empty success or historical fallback disguised as current (Tasks 2, 5 and 8).
5. A cloud runner has no cache, or a release upload fails partway: restore state via verified manifests and preserve the previous published run (Tasks 7-8).

## File map

| File | Responsibility |
| --- | --- |
| `wbb_scouting/__init__.py`, `__main__.py`, `cli.py` | Package, command entry point, orchestration |
| `wbb_scouting/calendar.py` | M/W/F boundaries, conditional retry and age calculations |
| `wbb_scouting/sources.py`, `adapters.py` | Asset discovery/download and explicit source-schema normalization |
| `wbb_scouting/validation.py` | Dataset/game contracts, reconciliation and coverage |
| `wbb_scouting/store.py` | Game reconciliation, canonical partitions, content hashes and manifests |
| `wbb_scouting/references.py` | Dated NET, team/conference identity and cohort selection |
| `wbb_scouting/reports.py`, `metrics.py` | As-of report calculations, editions and revisions |
| `wbb_scouting/dashboard.py`, `templates/dashboard.html` | Self-contained interactive dashboard and exports |
| `wbb_scouting/github_state.py` | Durable state restore and draft-release publication |
| `config/scouting_sources.json`, `config/scouting_teams_2027.json` | Verified release definitions and season-specific team/conference references |
| `tests/scouting/` | Representative fixtures and behavioral tests |
| `.github/workflows/wbb-scouting.yml` | Scheduled/manual workflow and validation entry point |
| `requirements-scouting.txt`, `.gitignore` | Compatible scoped dependencies and exclusion of local generated state |
| `docs/recurring_scouting/README.md`, `source_inventory.md` | Operation instructions, verified source matrix and limitations |

## Task 1: Calendar and independent entry point

**Files:** `wbb_scouting/{__init__,__main__,calendar,cli}.py`, `tests/scouting/test_calendar.py`, `requirements-scouting.txt`.

**Interfaces:** `next_primary_check(now: datetime) -> datetime` returns an aware UTC instant strictly after `now`; `refresh_window(now: datetime) -> tuple[datetime, datetime]`; `retry_required(now: datetime, primary_attempt: dict | None) -> bool`; CLI commands `audit`, `refresh`, `build` accept explicit season, repository, local state path and an injected clock for tests.

- [ ] Write tests: Friday 2026-11-06 at 06:17 Pacific ends at Monday 2026-11-09 06:17 Pacific; Thursday uses Friday; UTC boundary follows local weekdays; spring/fall DST preserve 06:17 local time; retry skips complete successful primary and runs for missing/failed/pending primary.
- [ ] Run `/usr/bin/python3 -m pytest tests/scouting/test_calendar.py -q`; confirm failure before implementation.
- [ ] Implement timezone-aware scheduling with `zoneinfo`, explicit UTC storage, scoped requirements, and CLI argument validation. Do not alter existing `config.py` or `pipeline.py`.
- [ ] Rerun calendar tests and `python -m wbb_scouting --help`; verify no network or writes from help.
- [ ] Review and commit only Task 1 files when a suitable isolated checkout/branch is available.

## Task 2: Live source inventory and normalization

**Files:** `sources.py`, `adapters.py`, `config/scouting_sources.json`, `docs/recurring_scouting/source_inventory.md`, `tests/scouting/test_sources.py`, representative small fixtures.

**Interfaces:** `discover_assets(dataset: str, season: int, client: object) -> list[dict]`; `download_asset(asset: dict, destination: Path, client: object) -> dict`; `normalize_snapshot(dataset: str, frame: DataFrame, source: dict) -> DataFrame`. Metadata contains provider, family, dataset, season, asset ID/name, canonical URL, updated time, byte size, SHA-256 and schema fingerprint. Normalized identifiers are strings.

- [ ] Write tests for exact asset/season discovery, 404 pending publication, 304 reuse, HTTP failures, corrupt Parquet, wrong women's league/season, null IDs, large IDs above float precision, participant-column aliasing, and schema drift.
- [ ] Run the source tests and confirm expected missing implementation failures.
- [ ] Inspect SDV release metadata and read real published schemas for schedules, team/player boxes, PBP and rosters. Record actual fields, key grain, division coverage, game coverage, dates and availability for 2027 plus a separately labeled 2026 reference.
- [ ] Implement explicit aliases supported by those inspected schemas, validation of metadata, streamed atomic downloads with bounded retries, and file fingerprint reuse. A blocked/unavailable source yields a structured pending/error result, not a blank successful snapshot.
- [ ] Implement and verify official NET retrieval only against an accessible official publication and its own date/season. The official NCAA.com page currently presents a browser verification challenge; record that if it persists and support validated official saved snapshots without inventing current NET. Do not quietly substitute a third-party ranking.
- [ ] Run source tests and a read-only live `audit`; save the source inventory and coverage result. Capture any unavailable current-season files explicitly.
- [ ] Review and commit source definitions, adapters, tests and evidence notes only; leave large downloaded data outside Git history.

## Task 3: Validated game reconciliation and persistent local state

**Files:** `validation.py`, `store.py`, `.gitignore`, `tests/scouting/test_store.py`, `tests/scouting/test_validation.py`.

**Interfaces:** `validate_snapshot(dataset: str, frame: DataFrame, context: dict) -> dict` returns accepted game groups and structured quarantine reasons; `reconcile_games(existing: DataFrame, incoming: DataFrame, keys: list[str], accepted_games: set[str]) -> tuple[DataFrame, dict]`; `write_partitions(frame: DataFrame, dataset: str, state_dir: Path) -> list[dict]`; `load_manifest(path: Path) -> dict`; `write_manifest(manifest: dict, path: Path) -> None` writes atomically.

- [ ] Write tests for unchanged rerun, new game, corrected stat, removed/renumbered PBP event, conflicting duplicates, incoming missing game, incomplete game, reciprocal team rows, player/team/final-score disagreement, attempted missed-shot values, and a later game masking an earlier coverage gap.
- [ ] Run these tests to establish failures.
- [ ] Implement exact normalized record fingerprints and staging. Partition by family/dataset/season/month; deduplicate byte-identical records, quarantine conflicting keys, replace validated groups, and retain absent games with a coverage regression warning.
- [ ] Implement season-specific expected-versus-observed coverage, supported score reconciliation, D1 eligibility/opponent-division flags and a structured status per dataset/game. Do not conflate missing PBP with unusable validated boxes.
- [ ] Record inserted, changed, unchanged, removed-within-replacement, quarantined rows and affected game IDs. Preserve snapshot fingerprints, run IDs and prior manifests. Guard raw assets against the release size limit using deterministic byte-part splitting when necessary.
- [ ] Run store/validation tests; perform a real initial ingest and unchanged rerun, checking printed counters and actual stored row/key counts.
- [ ] Review and commit implementation/tests, excluding generated state.

## Task 4: Dated team references, NET and cohort

**Files:** `references.py`, `config/scouting_teams_2027.json`, `tests/scouting/test_references.py`.

**Interfaces:** `load_team_reference(path: Path, season: int) -> DataFrame`; `validate_net(frame: DataFrame, publication: dict, teams: DataFrame) -> DataFrame`; `select_cohort(net: DataFrame | None, teams: DataFrame, as_of: datetime, season: int, historical_net: DataFrame | None = None) -> DataFrame` returns team IDs, names, inclusion reason, ranking source/season/date and selection status.

- [ ] Write tests for NET 1-50 union ACC, overlapping teams once, ACC team outside top 50, unresolved aliases, stale ranking, ranking published after cutoff, new-season ACC membership, and unavailable current NET with dated verified prior-year or ACC-only fallback.
- [ ] Run tests before implementation.
- [ ] Verify season-correct ACC membership and stable IDs from authoritative conference/team references; preserve provenance and source dates in the reference file.
- [ ] Implement strict reference/NET validation and cohort selection; previous-season ranks can select provisional teams but never populate a current-season NET field. Save cohort snapshots by refresh time and preserve source publication time.
- [ ] Run reference tests and inspect the real selected/provisional cohort with explicit counts, unresolved mappings and inclusion reasons.
- [ ] Review and commit reference data with its provenance, implementation and tests.

## Task 5: Metrics and preserved report editions

**Files:** `metrics.py`, `reports.py`, `tests/scouting/test_reports.py`, `tests/scouting/test_metrics.py`.

**Interfaces:** `team_metrics(team_box: DataFrame) -> dict`; `player_metrics(player_box: DataFrame) -> list[dict]`; `build_reports(tables: dict[str, DataFrame], cohort: DataFrame, now: datetime, previous_reports: list[dict], snapshots: dict, bootstrap: bool = False) -> list[dict]`. A report has identity, mode, game/team IDs, generation/cutoff/tip times, source references, cohort/ranking reference, edition fingerprint, status, denominators and supported observations.

- [ ] Write tests with hand-calculated box totals for the spec's eFG%, estimated possessions, ratings, turnover/rebound/FT rates and TS%; test zero/missing denominators and aggregate last-five rates.
- [ ] Write report tests for no target/future-game leakage, no future NET, Friday-to-Monday matchup coverage, changed matchup revisions, unchanged input producing no duplicate edition, early/late tip rescheduling, first-run seven-day postgames and late source arrivals.
- [ ] Test using the latest issued pregame edition before tip for postgame baselines; use a labeled reconstructed baseline if absent. Test empty season, stale source, missing PBP with valid boxes, partial pregame editions and source-corrected revisions preserving earlier editions.
- [ ] Run tests and confirm failures, then implement the explicit formulas and report functions with a common injected clock and immutable prior editions.
- [ ] Run all metric/report/calendar tests, inspect real source-backed report payloads, and confirm current-season gaps remain visible. Historical 2026 examples must be labeled historical/reconstructed and cannot stand in for current-season evidence.
- [ ] Review and commit calculations, payload generation and tests.

## Task 6: Interactive downloadable dashboard and exports

**Files:** `dashboard.py`, `templates/dashboard.html`, `tests/scouting/test_dashboard.py`.

**Interfaces:** `render_dashboard(reports: list[dict], manifest: dict, destination: Path) -> None`; `write_exports(reports: list[dict], destination: Path) -> list[Path]`. HTML includes its own report payload, styles and interaction code; no server or API key required to open it.

- [ ] Write focused tests for safe embedded JSON/HTML escaping, serializable nulls/large string IDs, export cutoff/source fields, and preservation of partial/pending statuses.
- [ ] Run tests, then implement team/matchup/date and pregame/postgame selectors, cohort reasons, source/coverage indicators, small-sample denominators, concise team/player comparisons, printable report, and filtered CSV/JSON downloads.
- [ ] Compute stale/missed-run messaging from the next expected M/W/F primary boundary rather than 24 hours. Report age remains visible even when no run is missed.
- [ ] Run targeted tests and real browser checks at desktop and mobile sizes: selection state, mode changes, populated and empty/partial views, exports, printing and console errors. Use actual current coverage; any synthetic fixture preview is explicitly labeled and separate from operational outputs.
- [ ] Review the concrete dashboard and data coverage with the user before activation, as the approved spec requires.
- [ ] Commit builder/template/tests and keep generated dashboard/state out of normal source history.

## Task 7: Durable GitHub Release state

**Files:** `github_state.py`, `tests/scouting/test_github_state.py`, `docs/recurring_scouting/README.md`.

**Interfaces:** `restore_state(repository: str, destination: Path, client: object) -> dict | None`; `publish_run(repository: str, manifest: dict, files: list[Path], client: object, commit_sha: str) -> dict`. Restore selects a valid published scouting run by explicit prefix/schema/status and validates all referenced data assets; publication uses a new draft release and promotes it only after verification.

- [ ] Write tests for fresh-runner restoration, no-cache execution, unrelated latest release, invalid latest manifest, prior valid fallback, unchanged-asset references, corrupted/missing asset, interrupted upload, concurrent publication rejection and files exceeding the configured shard threshold.
- [ ] Run tests, then implement GitHub API pagination, repository-scoped token use, checksum validation, immutable asset references, bounded API retries and explicit error propagation. Never expose token-bearing headers or temporary signed URLs in logs/manifests.
- [ ] Verify partial failure cannot replace the last good published state; an invalid persisted asset fails closed rather than silently starting a new empty season.
- [ ] Run mocked API tests and local restore/publish simulations. After concrete user review, perform a real draft/verified publication using only authorized project artifacts and verify downloads.
- [ ] Document public release visibility, file references, restoration, storage growth, no automatic deletion and recovery instructions; commit implementation/tests/docs.

## Task 8: Cloud orchestration, schedule and completion evidence

**Files:** `cli.py`, `.github/workflows/wbb-scouting.yml`, `tests/scouting/test_refresh.py`, operations README, scoped source inventory.

**Interfaces:** `refresh(config: dict, now: datetime, trigger: str, clients: dict) -> dict` orchestrates restore, retry gate, inventory, downloads, validation, canonical reconciliation, cohort, reports, dashboard and publication. It writes a human-readable run summary and machine-readable manifest with attempted/successful times, all dataset/game counters, coverage gaps and output references.

- [ ] Write integration tests with injected sources/storage: first ingest, identical rerun, new game, corrected game, pending 2027 boxes with 2026 available, missing primary attempt, retry after partial failure, no retry after complete success, and failed publication retaining prior state.
- [ ] Run integration tests, then implement orchestration and explicit exit/status handling. A pending publication is recorded as pending; schema/validation/network failures must not look like a successful empty refresh.
- [ ] Add a workflow on `workflow_dispatch` and two timezone-aware M/W/F schedules (06:17 primary and 08:17 conditional retry). Use pinned compatible action versions, Python 3.11+, scoped requirements, minimal token permissions and workflow concurrency with no cancellation of an in-progress publisher. Keep activation gated until the source-backed preview has been approved.
- [ ] Read primary attempt status from GitHub workflow metadata plus persisted run status so a missing or publication-failed primary still triggers the retry. Limit successful/no-change noise to run summaries and standard GitHub notification settings.
- [ ] Run the complete focused scouting test suite and relevant existing repo tests; inspect all output and `git diff --check`. Perform a clean-environment installation to check dependency reproducibility.
- [ ] Open a reviewable branch/PR containing only this feature, attach it to the chat, and provide the dashboard/coverage preview. Never stage unrelated user work.
- [ ] Following the authorized activation step, verify a real GitHub run and an unchanged rerun from clean runners, source/gap counters, release restoration, HTML/export URLs, effective M/W/F workflow schedule and conditional retry behavior. If required current-season source data is unavailable, show that exact limitation and do not claim complete current-season reporting.
- [ ] Mark completed plan checkboxes based on current-run evidence; record unresolved external blockers separately. A deployed workflow without verification does not satisfy completion.

## Plan self-review

- All approved spec sections map to Tasks 1-8; source enhancements remain explicitly deferred.
- Interfaces use one `DataFrame` table convention, dictionary manifests/report payloads and UTC-aware clocks; no task relies on a differently named undefined interface.
- All five Review Focus items have owning behavioral tests in the tasks above.
- Calendar amendment covers both job cadence and report lookahead; the weekend is intentionally longer than a daily freshness window.
- Operational activation remains after concrete result review. Current source availability and GitHub publishing capability are verified during execution, not assumed from this plan.
