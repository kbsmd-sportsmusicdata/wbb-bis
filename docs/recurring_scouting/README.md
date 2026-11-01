# Recurring WBB scouting

This independent pipeline collects the SDV ESPN women's college basketball family. It does not run the older unfinished dashboard pipeline or modify its season-2026 configuration.

## Operator commands

Use Python 3.12 and install \`requirements-scouting.txt\`. Run from the repository root:

\`\`\`sh
python -m wbb_scouting audit --season 2027
python -m wbb_scouting refresh --season 2027
python -m wbb_scouting build --season 2027
\`\`\`

The initial delivery is a downloaded, self-contained \`index.html\`, with full \`reports.json\`, \`reports.csv\`, and \`players.csv\` exports. The interactive view includes the past 14 days and all upcoming reports; every edition remains in exports and earlier release dashboards. The HTML needs no server, login, API key, or network after download.

## Durable state and recovery

Public GitHub Releases tagged \`scouting-v1-2027-*\` retain canonical Parquet partitions, original changed source bytes, report editions, and checksummed manifests. These files are publicly visible, like this repository. Do not add private scouting notes or credentials.

A fresh runner enumerates scouting releases for the requested season, independently of generic latest releases or caches. It verifies the manifest checksum and every referenced asset. A damaged latest release can fall back to an older verified checkpoint; if scouting checkpoints exist but none verify, collection fails closed instead of pretending to start empty.

Each publisher creates a new draft, uploads only new data, downloads uploaded bytes to verify SHA-256, checks its parent head again, then publishes. Unchanged partitions and raw versions retain immutable links to earlier releases. Sources over 512 MiB are split, checked individually, and reassembled and checked on restore. Workflow concurrency provides the primary serialization guard; a stale parent rejects a competing publisher. GitHub does not offer a transactional compare-and-swap release promotion, so do not run a second publisher outside this workflow.

Failed uploads leave a draft and preserve the prior published state. Inspect the failed Actions run, resolve the cause, and rerun; never overwrite or delete the last good release. No automated deletion is enabled. Storage grows with source revisions, report history, and changed partitions. Review measured growth before adding retention or external storage. Actions artifacts and caches are convenience copies, never the authoritative store.

## Coverage and evidence

All stats use ESPN identifiers as strings; NCAA-family RAPM/lineups remain separate. Reconciliation replaces only validated complete game groups, preserves absent historical games, and records additions, corrections, removals within a replaced group, quarantines, and gaps. A zero-change rerun must have zero additions, updates and removals.

The report cohort is checked current-season ACC membership union verified NCAA NET ranks 1–50. Without a verified current or prior-season NET snapshot, selection is explicitly ACC-only with top-50 selection pending. Source availability is checked separately for schedule, boxes, PBP and rosters. Prior-season history never satisfies current-season freshness.

Team and player box reports can remain available while PBP is pending or withheld. Rate denominators, game counts, source checksums, actual coverage, report cutoffs and baseline provenance travel with each edition. Source corrections create new editions; issued pregame reports remain immutable. Historical examples are labeled reconstructed and never presented as current-season production.

See [source inventory](source_inventory.md), [metric conventions](../metric_conventions.md), and the [approved design](../superpowers/specs/2026-11-01-recurring-game-scouting-design.md).

## Schedule and activation

The publisher workflow is `.github/workflows/wbb-scouting.yml`. Monday, Wednesday, Friday checks run at 06:17 America/Los_Angeles, with an 08:17 conditional retry on those same days. The retry checks both GitHub's actual latest primary workflow result and the persisted primary status; a missing primary, failed publication, or pending source can trigger collection. A complete successful primary skips it.

The approved design requires review of a concrete source/coverage preview before activation. Until review, repository Actions variable WBB_SCOUTING_ENABLED is unset and the publisher is disabled. After approval, merge the reviewed PR to the default branch, set WBB_SCOUTING_ENABLED=true, then use Actions → WBB scouting → Run workflow. The scoped Actions token publishes release assets. Confirm a real run, fresh-runner restore, and an unchanged rerun before calling the service operational.

GitHub schedules run only from the default branch, can be delayed, and public repository schedules can be disabled after 60 days without repository activity. Next-check/overdue warnings use the M/W/F calendar, including the weekend and daylight saving; report age is displayed separately. Concurrency serializes publishers without cancelling an in-progress publication. Success/no-change details go to run summaries and failures to standard GitHub notifications. No separate messages are sent.

PR checks run tests and build a source-backed preview with read-only permissions. Their expiring artifact is for review, not durable operational storage. They cannot publish or enable the schedule.

## Official NET setup

The official NCAA page is requested each refresh. Browser verification, unpublished tables, missing dates, or unresolved school mappings remain visibly pending. Never substitute another ranking or silently use an undated copy.

Reviewed saved official snapshots may be placed in config/scouting_net/*.json (or --net-dir). Each has a "publication" object with season, publication_date, first_seen_at, official NCAA source_url, verified:true, id_namespace:"espn"; its "rows" contain checked team_id and net_rank. Include a complete verified top 50 with unique IDs/ranks. A separate school-map.json maps reviewed official school slugs/names to ESPN IDs for HTML retrieval. Prior-year rankings only select a visibly provisional cohort; their numeric ranks never appear as current NET.

## Reproducible local verification

--local-input /path/to/verified/downloads reads explicit SDV-named snapshots. Original bytes remain archived and labeled local inputs. The live NET page may still be checked. --now accepts an explicit timezone-aware timestamp. Keep retrospective examples separate and never publish a historical clock as a current operational run.

To disable publication set WBB_SCOUTING_ENABLED=false. Preserve published releases while investigating. A new season requires a checked season-specific team/conference reference and reviewed workflow season.
