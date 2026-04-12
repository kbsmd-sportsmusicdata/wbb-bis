# BIS Pipeline — Data Dictionary
### WBB Basketball Intelligence System · 2025-26 Season

> **Usage:** This file lives at `docs/data_dictionary.md` in the `wbb-bis` repo. It documents every table produced by the pipeline — column names, types, formulas, and source scripts. Update this file whenever a script adds or removes columns.

---

## Table of Contents

1. [Key Identifiers](#1-key-identifiers)
2. [Constants & Lookup Tables](#2-constants--lookup-tables)
3. [Tier 1 — Processed Tables](#3-tier-1--processed-tables)
   - 3a. `player_box_advanced_metrics.csv`
   - 3b. `player_game_log_enriched.csv`
   - 3c. `pbp_player_metrics.csv`
   - 3d. `lineup_stints_raw.csv`
   - 3e. `player_onoff_metrics.csv`
   - 3f. `postseason_stints_raw.csv`
   - 3g. `postseason_onoff_metrics.csv`
4. [Tier 2 — Merge Tables](#4-tier-2--merge-tables)
   - 4a. `player_scouting_tournament68.csv`
   - 4b. `player_scouting_top50.csv`
   - 4c. `player_feature_table_2026.csv`
5. [Categorical & Label Columns](#5-categorical--label-columns)
6. [Column Naming Conventions](#6-column-naming-conventions)
7. [Derived Metric Formulas](#7-derived-metric-formulas)

---

## 1. Key Identifiers

These columns are the primary join keys across all pipeline tables. Never join on player name or team name — both vary across sources.

| Field | Type | Description |
|---|---|---|
| `athlete_id` | int | ESPN athlete ID (via wehoop). Primary player key across all tables. |
| `team_id` | int | ESPN team ID. Primary team key across all tables. |
| `game_id` | int | ESPN game ID. Unique per game across all seasons. |
| `season` | int | Year the season **ends** — e.g., 2026 = the 2025-26 season. |
| `athlete_display_name` | str | Player full name as returned by ESPN. Display only — not a join key. |
| `team_location` | str | City/school name as returned by ESPN (e.g., "South Carolina"). Display only. |
| `team_name` | str | Mascot name as returned by ESPN (e.g., "Gamecocks"). Display only. |

---

## 2. Constants & Lookup Tables

### Season Type Codes (ESPN / wehoop)

| Code | Meaning |
|---|---|
| 1 | Preseason |
| 2 | Regular season |
| 3 | Postseason / NCAA Tournament |
| 4 | Off-season |

Pipeline scripts filter on `season_type == 2` for regular season and use postseason-only parquets (no filter needed) for tournament data.

### NET Quadrant Definitions

Quadrant assignments depend on **game location** (home/away/neutral) and **opponent NET rank**.

| Quadrant | Home game (opp NET) | Neutral | Away game (opp NET) |
|---|---|---|---|
| Q1 | 1–30 | 1–50 | 1–75 |
| Q2 | 31–75 | 51–100 | 76–135 |
| Q3 | 76–160 | 101–200 | 136–240 |
| Q4 | 161+ | 201+ | 241+ |

> **Note:** The pipeline currently uses simplified Q1–Q4 cutoffs (30 / 75 / 160) applied uniformly regardless of game location. Location-adjusted thresholds are the `NET_Q*_MAX` constants in `config.py` and can be updated there.

### Minutes Tiers

| Tier | Threshold | Label |
|---|---|---|
| Star | 28+ MPG | `"Star"` |
| Rotation | 20–27.9 MPG | `"Rotation"` |
| Bench | 10–19.9 MPG | `"Bench"` |
| Deep Bench | < 10 MPG | `"Deep Bench"` |

Minimum qualifying threshold for inclusion in most analyses: `MIN_MINUTES_SEASON = 50` total minutes (from `config.py`).

---

## 3. Tier 1 — Processed Tables

### 3a. `player_box_advanced_metrics.csv`

**Script:** `scripts/01_player_box_processing.py`
**Inputs:** `player_box_2026.parquet`, `schedule_filtered.csv`, `net_rankings.csv`, `d1_player_benchmarks_2025.csv`
**Rows:** ~1 row per player (season aggregate); filtered to D1 players meeting minimum minutes threshold.

#### Identity & Counting Stats

| Column | Type | Description |
|---|---|---|
| `athlete_id` | int | ESPN athlete ID |
| `team_id` | int | ESPN team ID |
| `athlete_display_name` | str | Player full name |
| `athlete_position_abbreviation` | str | Position code (G, F, C, etc.) |
| `athlete_position_name` | str | Position full name |
| `team_location` | str | School name |
| `team_name` | str | Mascot name |
| `season` | int | Season year (2026) |
| `games_played` | int | Total games appeared in |
| `minutes` | float | Total minutes played |
| `points` | int | Total points |
| `rebounds` | int | Total rebounds (offensive + defensive) |
| `offensive_rebounds` | int | Total offensive rebounds |
| `defensive_rebounds` | int | Total defensive rebounds |
| `assists` | int | Total assists |
| `steals` | int | Total steals |
| `blocks` | int | Total blocks |
| `turnovers` | int | Total turnovers |
| `fouls` | int | Total personal fouls |
| `fg_made` | int | Field goals made |
| `fg_attempted` | int | Field goals attempted |
| `three_made` | int | Three-pointers made |
| `three_attempted` | int | Three-pointers attempted |
| `ft_made` | int | Free throws made |
| `ft_attempted` | int | Free throws attempted |

#### Per-Game Rates (`_pg` suffix)

| Column | Formula |
|---|---|
| `points_pg` | `points / games_played` |
| `rebounds_pg` | `rebounds / games_played` |
| `assists_pg` | `assists / games_played` |
| `steals_pg` | `steals / games_played` |
| `blocks_pg` | `blocks / games_played` |
| `turnovers_pg` | `turnovers / games_played` |
| `minutes_pg` | `minutes / games_played` |

#### Per-40 Rates (`_per40` suffix)

| Column | Formula |
|---|---|
| `points_per40` | `points × (40 / minutes)` |
| `rebounds_per40` | `rebounds × (40 / minutes)` |
| `assists_per40` | `assists × (40 / minutes)` |
| `steals_per40` | `steals × (40 / minutes)` |
| `blocks_per40` | `blocks × (40 / minutes)` |

#### Shooting Splits

| Column | Formula |
|---|---|
| `fg_pct` | `fg_made / fg_attempted` |
| `three_pct` | `three_made / three_attempted` |
| `ft_pct` | `ft_made / ft_attempted` |
| `true_shooting_pct` | `points / (2 × (fg_attempted + 0.44 × ft_attempted))` |
| `three_attempt_rate` | `three_attempted / fg_attempted` |
| `ft_attempt_rate` | `ft_attempted / fg_attempted` |

#### Advanced & Efficiency

| Column | Formula / Description |
|---|---|
| `usage_proxy` | `(fg_attempted + 0.44 × ft_attempted + turnovers) / team_poss` — estimated possession usage share |
| `pointdiff` | Average point differential in games the player appeared in (PPG − opponent PPG); reflects team strength in their games |
| `assist_to_turnover` | `assists / turnovers` |

#### NET Quadrant Records

| Column | Description |
|---|---|
| `q1_wins` | Wins vs. Q1 opponents |
| `q1_losses` | Losses vs. Q1 opponents |
| `q2_wins` | Wins vs. Q2 opponents |
| `q2_losses` | Losses vs. Q2 opponents |
| `q3_wins` | Wins vs. Q3 opponents |
| `q3_losses` | Losses vs. Q3 opponents |
| `q4_wins` | Wins vs. Q4 opponents |
| `q4_losses` | Losses vs. Q4 opponents |
| `q1_win_pct` | `q1_wins / (q1_wins + q1_losses)` |
| `q1q2_wins` | Combined Q1+Q2 wins |
| `q1q2_win_pct` | `q1q2_wins / (q1q2_wins + q1_losses + q2_losses)` |

#### Percentile Ranks (`_pct_rank` suffix)

Percentile ranks are computed relative to all D1 players meeting the minimum minutes threshold. 100th percentile = best in the dataset.

| Column | Based on |
|---|---|
| `points_pg_pct_rank` | `points_pg` |
| `true_shooting_pct_rank` | `true_shooting_pct` |
| `assists_pg_pct_rank` | `assists_pg` |
| `rebounds_pg_pct_rank` | `rebounds_pg` |
| `usage_proxy_pct_rank` | `usage_proxy` |
| `steals_pg_pct_rank` | `steals_pg` |
| `blocks_pg_pct_rank` | `blocks_pg` |

#### Label Columns

| Column | Values | Description |
|---|---|---|
| `minutes_tier` | `"Star"`, `"Rotation"`, `"Bench"`, `"Deep Bench"` | Playing time classification |
| `half_adjustment_label` | `"Closer"`, `"Fader"`, `"Steady"` | Whether player's production improves, drops, or holds in 2H vs 1H |

---

### 3b. `player_game_log_enriched.csv`

**Script:** `scripts/01_player_box_processing.py`
**Inputs:** Same as 3a.
**Rows:** 1 row per player per game (game-level, not aggregated). Same columns as 3a counting stats plus:

| Column | Type | Description |
|---|---|---|
| `game_date` | date | Date of game |
| `opponent_id` | int | Opponent ESPN team ID |
| `opponent_name` | str | Opponent team name |
| `home_away` | str | `"home"`, `"away"`, or `"neutral"` |
| `result` | str | `"W"` or `"L"` |
| `team_score` | int | Player's team final score |
| `opponent_score` | int | Opponent final score |
| `opponent_net_rank` | int | Opponent NET rank on game date |
| `net_quadrant` | str | `"Q1"`, `"Q2"`, `"Q3"`, `"Q4"` (using simplified uniform thresholds) |
| `half` | int | 1 or 2 (rows are split by half for half-adjustment analysis) |

---

### 3c. `pbp_player_metrics.csv`

**Script:** `scripts/02_pbp_player_processing.py`
**Inputs:** `play_by_play_YYYYMMDD.parquet` (regular season rows only, `season_type == 2`)
**Rows:** 1 row per player (season aggregate from PBP events). Only players with shot attempts are included.

#### Identity

| Column | Type | Description |
|---|---|---|
| `athlete_id` | int | ESPN athlete ID |
| `team_id` | int | ESPN team ID |
| `athlete_display_name` | str | Player name (from PBP event records) |

#### Shot Zone Counts

| Column | Description |
|---|---|
| `paint_fga` | Field goal attempts classified as paint shots (≤ 8 ft from basket) |
| `paint_fgm` | Paint field goals made |
| `paint_fg_pct` | `paint_fgm / paint_fga` |
| `midrange_fga` | Mid-range attempts (inside arc, outside paint) |
| `midrange_fgm` | Mid-range makes |
| `midrange_fg_pct` | `midrange_fgm / midrange_fga` |
| `corner_three_fga` | Corner three-point attempts |
| `corner_three_fgm` | Corner three makes |
| `corner_three_fg_pct` | `corner_three_fgm / corner_three_fga` |
| `above_break_three_fga` | Above-the-break three-point attempts |
| `above_break_three_fgm` | Above-the-break three makes |
| `above_break_three_fg_pct` | `above_break_three_fgm / above_break_three_fga` |
| `total_shots` | Total shot attempts from PBP |

#### Shot Diet Shares

| Column | Formula |
|---|---|
| `paint_share` | `paint_fga / total_shots` |
| `midrange_share` | `midrange_fga / total_shots` |
| `corner_three_share` | `corner_three_fga / total_shots` |
| `above_break_three_share` | `above_break_three_fga / total_shots` |
| `three_share` | `(corner_three_fga + above_break_three_fga) / total_shots` |

#### Creation & Playmaking

| Column | Formula / Description |
|---|---|
| `assisted_fg_rate` | `assisted_fgm / total_fgm` (min 10 FGM) — proportion of made shots that were assisted; lower = more self-creation |
| `self_created_pct` | `1 - assisted_fg_rate` |
| `creation_share` | Estimated share of team possessions where player created the shot (includes assisted passes + unassisted FGM) |
| `potential_assists` | Passes that directly led to a field goal attempt (made or missed) |

#### Clutch Performance

| Column | Description |
|---|---|
| `clutch_fga` | FGA in clutch situations (score within 5, final 5 minutes) |
| `clutch_fgm` | FGM in clutch situations |
| `clutch_fg_pct` | `clutch_fgm / clutch_fga` |
| `clutch_minutes` | Minutes played in clutch situations |

#### Label Columns

| Column | Values | Description |
|---|---|---|
| `shot_creator_type` | `"Rim Finisher"`, `"Sniper"`, `"Mid-Range"`, `"Balanced"` | Primary shot profile based on zone shares |
| `playmaking_style` | `"Pass-First"`, `"Scoring Playmaker"`, `"Score-First"`, `"Off-Ball"` | Derived from creation_share and assist rates |

---

### 3d. `lineup_stints_raw.csv`

**Script:** `scripts/03_lineup_stints_regular.py`
**Inputs:** `play_by_play.parquet` (regular season), `net_rankings.csv`
**Rows:** 1 row per lineup stint — each record represents one continuous stretch of a fixed 5-player lineup on the court.

| Column | Type | Description |
|---|---|---|
| `game_id` | int | ESPN game ID |
| `team_id` | int | ESPN team ID |
| `period` | int | Game period (1 or 2 for regulation halves) |
| `lineup` | str | Pipe-delimited sorted athlete_ids of the 5 on-court players (e.g., `"12345\|23456\|..."\`) |
| `start_time` | float | Clock time (seconds remaining in period) at stint start |
| `end_time` | float | Clock time at stint end |
| `duration_sec` | float | `start_time - end_time` — length of stint in seconds |
| `pts_scored` | int | Points scored by this lineup during the stint |
| `pts_allowed` | int | Points allowed by this lineup during the stint |
| `poss_for` | float | Estimated offensive possessions (`FGA + 0.44×FTA + TOV - OREB`) |
| `poss_against` | float | Estimated defensive possessions (opponent's `poss_for` during stint) |

---

### 3e. `player_onoff_metrics.csv`

**Script:** `scripts/03_lineup_stints_regular.py`
**Inputs:** `lineup_stints_raw.csv` (derived)
**Rows:** 1 row per player (regular season on/off splits across all stints).

| Column | Type | Description |
|---|---|---|
| `athlete_id` | int | ESPN athlete ID |
| `player_name` | str | Player display name |
| `team_id` | int | ESPN team ID |
| `team_name` | str | Team name |
| `team_location` | str | School name |
| `athlete_position_abbreviation` | str | Position code |
| `athlete_position_name` | str | Position full name |
| `games` | int | Games with at least one stint |
| `stints_on` | int | Number of stints player was on the court |
| `stints_off` | int | Number of stints player was off the court |
| `on_minutes` | float | Total minutes on court |
| `off_minutes` | float | Total minutes off court |
| `poss_on` | float | Total estimated possessions while player was on court |
| `poss_off` | float | Total estimated possessions while player was off court |
| `on_off_rtg` | float | Offensive rating while on court (points per 100 possessions) |
| `on_def_rtg` | float | Defensive rating while on court (opponent points per 100 possessions; lower = better defense) |
| `on_net_rtg` | float | Net rating while on court (`on_off_rtg - on_def_rtg`) |
| `off_off_rtg` | float | Offensive rating while off court |
| `off_def_rtg` | float | Defensive rating while off court |
| `off_net_rtg` | float | Net rating while off court |
| `net_rtg_diff` | float | `on_net_rtg - off_net_rtg` — key two-way impact signal; positive = team is better with player on court |

---

### 3f. `postseason_stints_raw.csv`

**Script:** `scripts/04_lineup_stints_postseason.py`
**Inputs:** `postseason/march_pbp_2026.parquet`, `tournament_bracket.csv`
**Rows:** Same schema as `lineup_stints_raw.csv` but tournament games only.

Schema is identical to `lineup_stints_raw.csv`. See section 3d.

---

### 3g. `postseason_onoff_metrics.csv`

**Script:** `scripts/04_lineup_stints_postseason.py`
**Inputs:** `postseason_stints_raw.csv` (derived), `tournament_bracket.csv`
**Rows:** 1 row per player (tournament on/off splits). All columns use `tourney_` prefix to avoid collision in feature table merges.

| Column | Type | Description |
|---|---|---|
| `athlete_id` | int | ESPN athlete ID — join key to other tables |
| `player_name` | str | Player display name |
| `team_id` | int | ESPN team ID |
| `tourney_games` | int | Tournament games played |
| `tourney_stints` | int | Tournament stints on court |
| `tourney_on_sec` | float | Total seconds on court in tournament |
| `tourney_poss_on` | float | Estimated possessions while on court (tournament) |
| `tourney_poss_off` | float | Estimated possessions while off court (tournament) |
| `tourney_off_rtg` | float | Offensive rating while on court (tournament) |
| `tourney_def_rtg` | float | Defensive rating while on court (tournament) |
| `tourney_net_rtg` | float | Net rating while on court (tournament) |
| `tourney_net_rtg_diff` | float | `tourney_net_rtg - off_net_rtg_tourney` — tournament on/off impact |

---

## 4. Tier 2 — Merge Tables

### 4a. `player_scouting_tournament68.csv`

**Script:** `scripts/05_merge_scouting.py`
**Inputs:** `player_box_advanced_metrics.csv`, `pbp_player_metrics.csv`, `tournament_bracket.csv`
**Rows:** 1 row per player on a tournament team (68 teams × ~13 roster spots = ~884 rows, filtered to meaningful minutes).

Contains all columns from `player_box_advanced_metrics.csv` and `pbp_player_metrics.csv` joined on `athlete_id`, filtered to players whose team appeared in the 68-team tournament field. Also includes:

| Column | Type | Description |
|---|---|---|
| `seed` | int | Team's tournament seed (1–16) |
| `region` | str | Tournament region (e.g., "Albany", "Portland") |
| `first_game_opponent` | str | First-round opponent team name |
| `first_game_result` | str | `"W"` or `"L"` (if tournament has concluded) |
| `rounds_won` | int | Number of tournament rounds the team won |

---

### 4b. `player_scouting_top50.csv`

**Script:** `scripts/05_merge_scouting.py`
**Inputs:** Same as 4a.
**Rows:** Top 50 players by a composite production score across the full D1 dataset.

Same schema as `player_scouting_tournament68.csv` but not filtered to tournament teams. Includes players from non-tournament programs who ranked among the top 50 overall.

---

### 4c. `player_feature_table_2026.csv`

**Script:** `scripts/06_merge_feature_table.py`
**Inputs:** `player_onoff_metrics.csv` (base), `player_box_advanced_metrics.csv`, `pbp_player_metrics.csv`, `postseason_onoff_metrics.csv`, `wbb_rosters_2025_26.csv`, `tournament_bracket.csv`
**Rows:** 403 rows — one per player from R32 tournament teams (based on `player_onoff_metrics.csv` as the base).

This is the ML-ready wide table. Contains all columns from all upstream tables (with merge suffix artifacts cleaned up) plus the following cross-table derived features:

#### Cross-Table Derived Features

| Column | Formula | Description |
|---|---|---|
| `net_rtg_reg_to_tourney` | `tourney_net_rtg - on_net_rtg` | Impact delta: did the player perform better or worse on/off in the tournament vs. regular season? Positive = improved under pressure |
| `weighted_production` | `true_shooting_pct × points_per40` | Volume-efficiency composite — rewards efficient high-usage players |
| `two_way_flag` | `1 if (net_rtg_diff > 2 AND on_net_rtg > 0) else 0` | Binary flag: player is measurably positive on both ends of the court |
| `shot_diet` | categorical | Derived from zone shares — see section 5 |

#### Recruiting Columns (from `player_recruit_rankings_20212026.csv`, name-based join)

> **Join note:** This file has no ESPN `athlete_id`. The join is performed on normalised `athlete_display_name` ↔ `PLAYER_NAME`. Expected match rate: 50–70% of R32 players. Unmatched players receive `NaN` for rank/grade columns and `0` for all flag columns.

| Column | Type | Description |
|---|---|---|
| `recruit_rank` | float (→ int) | ESPN W national recruit rank in the player's signing class (1 = #1 nationally). `NaN` = unranked or not in database. |
| `recruit_grade` | float | ESPN recruit grade (0–100 scale). Higher = higher-rated prospect. |
| `recruit_class_year` | int | Year the player was recruited / signed (e.g., 2023 = class of 2023). |
| `is_top25_recruit` | int | `1` if `recruit_rank ≤ 25`, else `0`. |
| `is_top100_recruit` | int | `1` if `recruit_rank ≤ 100`, else `0`. |
| `is_ranked_recruit` | int | `1` if any ESPN ranking exists for this player, else `0` (walk-ons, unrated transfers, and players outside the 2021–2026 window will be `0`). |

#### Roster Columns (from `wbb_rosters_2025_26.csv`)

| Column | Type | Description |
|---|---|---|
| `height_in` | int | Player height in inches |
| `class_year` | str | `"FR"`, `"SO"`, `"JR"`, `"SR"`, `"GR"` (graduate) |
| `hometown_city` | str | Player hometown city |
| `hometown_state` | str | Player hometown state |
| `transfer_flag` | int | `1` if player transferred from another program, `0` if native recruit |

#### Tournament Columns (from `tournament_bracket.csv`)

| Column | Type | Description |
|---|---|---|
| `seed` | int | Tournament seed |
| `region` | str | Tournament region |

---

## 5. Categorical & Label Columns

### `shot_diet` (in `player_feature_table_2026.csv`)

Derived from `paint_share` and `three_share` in `pbp_player_metrics.csv`.

| Value | Condition |
|---|---|
| `"Paint-Dominant"` | `paint_share > 0.50` |
| `"Three-Dominant"` | `three_share > 0.45` |
| `"Three-Leaning"` | `three_share > 0.35` |
| `"Balanced"` | All other cases |

### `shot_creator_type` (in `pbp_player_metrics.csv`)

| Value | Condition |
|---|---|
| `"Rim Finisher"` | `paint_share > 0.55 AND assisted_fg_rate > 0.60` |
| `"Sniper"` | `three_share > 0.50 AND assisted_fg_rate > 0.70` |
| `"Mid-Range"` | `midrange_share > 0.35` |
| `"Balanced"` | All other cases |

### `playmaking_style` (in `pbp_player_metrics.csv`)

| Value | Condition |
|---|---|
| `"Pass-First"` | `creation_share > 0.25 AND assisted_fg_rate > 0.60` |
| `"Scoring Playmaker"` | `creation_share > 0.20 AND assisted_fg_rate < 0.55` |
| `"Score-First"` | `creation_share < 0.15 AND self_created_pct > 0.50` |
| `"Off-Ball"` | `creation_share < 0.10 AND assisted_fg_rate > 0.75` |

### `half_adjustment_label` (in `player_box_advanced_metrics.csv`)

Comparing 2H vs. 1H per-40 point production:

| Value | Condition |
|---|---|
| `"Closer"` | 2H points_per40 > 1H points_per40 by ≥ 10% |
| `"Fader"` | 2H points_per40 < 1H points_per40 by ≥ 10% |
| `"Steady"` | Within 10% either direction |

### Recruiting Tier Flags (in `player_feature_table_2026.csv`)

| Column | Value | Condition |
|---|---|---|
| `is_top25_recruit` | `1` | `recruit_rank` is not null AND ≤ 25 |
| `is_top25_recruit` | `0` | Ranked 26+ or unranked |
| `is_top100_recruit` | `1` | `recruit_rank` is not null AND ≤ 100 |
| `is_top100_recruit` | `0` | Ranked 101+ or unranked |
| `is_ranked_recruit` | `1` | Any ESPN recruit rank exists in the 2021–2026 database |
| `is_ranked_recruit` | `0` | Walk-on, unrated transfer, or recruit class outside the database window |

### `minutes_tier` (in `player_box_advanced_metrics.csv`)

| Value | `minutes_pg` threshold |
|---|---|
| `"Star"` | ≥ 28 |
| `"Rotation"` | 20–27.9 |
| `"Bench"` | 10–19.9 |
| `"Deep Bench"` | < 10 |

---

## 6. Column Naming Conventions

| Pattern | Example | Meaning |
|---|---|---|
| No suffix | `points`, `rebounds` | Raw season totals |
| `_pg` | `points_pg` | Per game |
| `_per40` | `points_per40` | Per 40 minutes |
| `_rtg` | `on_net_rtg` | Per 100 possessions (efficiency rating) |
| `_pct` | `fg_pct`, `paint_fg_pct` | Percentage (0.0–1.0) |
| `_pct_rank` | `points_pg_pct_rank` | Percentile rank vs. D1 (0–100, higher = better) |
| `_share` | `paint_share` | Proportion of total (0.0–1.0) |
| `_flag` | `two_way_flag`, `transfer_flag` | Binary indicator (0 or 1) |
| `_type`, `_style`, `_label`, `_tier` | `shot_creator_type` | Categorical / label columns |
| `tourney_` prefix | `tourney_net_rtg` | Postseason / NCAA Tournament metric |
| `q1_`, `q2_`, `q3_`, `q4_` prefix | `q1_wins` | NET quadrant-specific record |
| `on_`, `off_` prefix | `on_net_rtg`, `off_def_rtg` | Lineup on/off court splits |

---

## 7. Derived Metric Formulas

Quick reference for formulas used across pipeline scripts.

| Metric | Formula |
|---|---|
| True Shooting % | `PTS / (2 × (FGA + 0.44 × FTA))` |
| Usage Proxy | `(FGA + 0.44×FTA + TOV) / team_poss` |
| Offensive Rating | `(pts_scored / poss_for) × 100` |
| Defensive Rating | `(pts_allowed / poss_against) × 100` (lower = better defense) |
| Net Rating | `off_rtg - def_rtg` |
| Net Rating Diff | `on_net_rtg - off_net_rtg` |
| Estimated Possessions | `FGA + 0.44×FTA + TOV - OREB` |
| Weighted Production | `true_shooting_pct × points_per40` |
| Assist-to-Turnover | `assists / turnovers` |
| Assisted FG Rate | `assisted_fgm / total_fgm` (min 10 FGM) |
| Paint Share | `paint_fga / total_shots` |
| Three Share | `(corner_three_fga + above_break_three_fga) / total_shots` |
| Reg→Tourney Delta | `tourney_net_rtg - on_net_rtg` |

---

*Last updated: April 2026 · Season: 2025-26 (season integer: 2026)*
*For pipeline questions, see `docs/pipeline_dependency_map.md`.*
