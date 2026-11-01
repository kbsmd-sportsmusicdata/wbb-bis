# SDV scouting source inventory

Inspected November 1, 2026 using live GitHub release metadata and downloaded Parquet bytes. This inventory is evidence of this audit, not an evergreen freshness guarantee.

| Dataset | Season | Published asset | Rows | Result |
| --- | --- | --- | ---: | --- |
| Schedule | 2027 | wbb_schedule_2027.parquet | 5,814 | Full table normalized; all games currently scheduled, none completed |
| Rosters | 2027 | rosters_2027.parquet | 9,680 | Full table normalized; includes non-D1 schools, so team reference filtering is required |
| Team membership | 2027 | wbb_team_group_seasons_2027.parquet | 363 | All tagged wbb:d1; 18 current-season ACC teams |
| Team boxes | 2027 | Not published in live asset list | — | Pending source publication |
| Player boxes | 2027 | Not published in live asset list | — | Pending source publication |
| PBP | 2027 | Not published in live asset list | — | Pending source publication |
| Team boxes | 2026 | team_box_2026.parquet | 12,058 | Full table normalized; 6,029 games; historical reference only |
| Player boxes | 2026 | player_box_2026.parquet | 168,228 | Two placeholder DNP rows have null athlete IDs in two games; strict identity gate detects them. Validation must quarantine affected game groups rather than invent IDs |
| PBP | 2026 | play_by_play_2026.parquet | 2,824,090 | Metadata read; first row group (122,787 events / 276 games) normalized. This is not a full-file PBP completeness certification |

Schedule range: November 1, 2026 through March 7, 2027 UTC. Current-season boxes cannot be substituted with previous-season observations. Real files have ESPN-style league 54 identifiers; stats.ncaa.org identifiers remain separate.

Actual release metadata carries asset SHA-256 digests. The downloader verifies those digests, original Parquet readability and schema fingerprints, and reuses already verified identical bytes. No large downloaded files were added to Git.

Observed schema adapters: schedule home_id/away_id -> home_team_id/away_team_id; completion state -> completed; athlete_id -> player_id; PBP id -> event_id; team_score -> points. All canonical identifiers are strings. Attempted shot values are not scoring points on misses.

## NET access

The official NCAA.com women's NET page currently presents browser verification instead of a readable table. Current-season NET is not yet published at this preseason stage. The implementation must retain a visible pending/provisional selection and support validated official saved snapshots. Do not silently switch to another ranking provider or label a manually supplied prior-year CSV as a verified current NCAA publication.

## Source links

- [Schedule release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_schedules)
- [Team boxes](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_team_boxscores)
- [Player boxes](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_player_boxscores)
- [PBP](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_pbp)
- [Rosters](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_womens_college_basketball_rosters)
- [Dated team membership](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/wbb_groups)
- [Official women's NET](https://www.ncaa.com/rankings/basketball-women/d1/ncaa-womens-basketball-net-rankings)

## Real reconciliation pilot

The historical California-and-opponents pilot validated 36 season-2026 games / 72 team-box rows against the same season's final schedule, with no quarantined games. Initial ingestion inserted 72 rows. The identical rerun inserted 0, updated 0, removed 0 and retained 72 unchanged rows. Five monthly canonical partitions were written under temporary pilot state. This is a bounded historical reconciliation check, not current-season coverage or a national completeness certification.
