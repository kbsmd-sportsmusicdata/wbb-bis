# Role Archetypes + Roster Construction Summary

Run timestamp: 2026-04-14 17:22
Season: 2026

## Canonical Taxonomy Preservation

The script preserves the original C0-C5 role IDs and role names from clustering metadata.
Existing archived player assignments are kept as-is; only missing players are inferred.

- Players with preserved archival assignment: 269
- Players inferred via centroid matching: 134
- Total players in feature table: 403
- Players with tournament seed context: 403

## Role Distribution

- C0: Stretch Defender — 36 players, avg confidence 0.646
- C2: Perimeter Contributor — 111 players, avg confidence 0.782
- C3: Shot Creator — 121 players, avg confidence 0.812
- C4: Two-Way Elite Wing — 8 players, avg confidence 0.666
- C5: Interior Anchor — 127 players, avg confidence 0.796

## Target Role Mix Baseline

Baseline profile is computed from top-seeded teams (seed <= 4), with fallback to all seeded teams.

- Stretch Defender: 0.052 share
- Complete Performer: 0.000 share
- Perimeter Contributor: 0.236 share
- Shot Creator: 0.392 share
- Two-Way Elite Wing: 0.024 share
- Interior Anchor: 0.297 share

## Output Files

- /Users/krystalbeasley/projects/wbb-bis/analysis/role_archetypes/role_archetype_assignments_2026.csv
- /Users/krystalbeasley/projects/wbb-bis/analysis/role_archetypes/role_cluster_summary_2026.csv
- /Users/krystalbeasley/projects/wbb-bis/analysis/roster_construction/team_role_balance_report_2026.csv
- /Users/krystalbeasley/projects/wbb-bis/analysis/roster_construction/role_gap_recommendations_2026.csv