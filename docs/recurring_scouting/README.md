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
