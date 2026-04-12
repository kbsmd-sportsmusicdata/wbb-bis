# BIS Pipeline — Dependency Map & Rebuild Order

*Last updated: 2026-04-10*

---

## Roster Analytics Folder — Status Check

**No sync issue.** The "missing" files (player_season_analytic_2026_top25.csv, team_season_analytic_2026_top25.csv, team_box_2026.parquet, play_by_play_2026.parquet) exist and are complete — they live in the separate **`WBB/Roster Analytics/data/`** folder, which is its own earlier project. Those files will need to be incorporated into the BIS GitHub repo structure, but nothing is lost or corrupted.

---

## Pipeline Architecture — Three-Tier Model

```
TIER 0: RAW SOURCES          (scraped / static / manual inputs)
   ↓
TIER 1: PROCESSED TABLES     (one script per output)
   ↓
TIER 2: MERGED / FEATURE     (merge scripts combining Tier 1)
   ↓
TIER 3: DASHBOARDS & REPORTS (HTML output, final deliverables)
```

---

## TIER 0 — Raw Source Datasets

These are inputs only. No script produces them — they come from wehoop/ESPN scrapes, Kaggle, or manual pulls.

| File | Location | Current Max Date | Status |
|---|---|---|---|
| `player_box_2026.parquet` | `Player Roles + Breakouts/` | **March 15, 2026** | ⚠️ Needs update |
| `play_by_play_20260327.parquet` | `Player Roles + Breakouts/` | **March 27, 2026** | ⚠️ Needs update (Final Four + Championship missing) |
| `wbb_schedule_2026.parquet` | `Player Roles + Breakouts/` | April 5, 2026 | ⚠️ Verify game results complete |
| `schedule_filtered_20260309.csv` | `Player Roles + Breakouts/` | **March 13, 2026** | ⚠️ Needs update |
| `net_rankings_manual_20260316.csv` | `Player Roles + Breakouts/` | **March 15, 2026** | ⚠️ Needs final pull |
| `wbb_rosters_2025_26.csv` | `roster_data/` | Season 2025-26 | ✅ Complete (verify portal changes) |
| `d1_player_benchmarks_2025.csv` | `Player Roles + Breakouts/` | Static | ✅ Complete |
| `tournament_bracket.csv` | `Player Roles + Breakouts/` | Static (68 teams) | ✅ Complete |
| `tournament_storylines.csv` | `Player Roles + Breakouts/` | Static | ✅ Update with final outcomes |
| `Postseason Data/player_box_2026_march.parquet` | `Postseason Data/` | **March 24, 2026** | ⚠️ Needs Elite 8 + Final Four + Championship |
| `Postseason Data/march_pbp_2026.parquet` | `Postseason Data/` | **March 23, 2026** | ⚠️ Needs Elite 8 + Final Four + Championship |
| `Postseason Data/clean_player_box_march.parquet` | `Postseason Data/` | **March 23, 2026** | ⚠️ Needs update |
| `Postseason Data/clean_conference_pbp.parquet` | `Postseason Data/` | March 9, 2026 | ✅ Complete (conf tourneys only) |
| `Draft/hist_draft_2014_2024_full_raw.csv` | `Draft/` | 2024 | ✅ Historical, static |
| `Draft/draft_prospects_2026.csv` | `Draft/` | 2026 class | ✅ Complete |
| `roster_data/player_recruit_rankings_20212026.csv` | `roster_data/` | 2026-27 class | ✅ Complete |
| `polls_historical_analytics copy.csv` | `Player Roles + Breakouts/` | Season 2026 | ✅ Complete |
| Kaggle datasets (`WNCAATourney*`, `WRegularSeason*`, etc.) | `Kaggle Datasets copy/` | Historical | ✅ Static |

---

## TIER 1 — Processed Tables (Script → Output)

### Script 1: `player_box_processing.py`

**Run via:** `update_pipeline.py` (Step 1) or directly

**Inputs:**
- `player_box_2026.parquet` ← **must be updated first**
- `schedule_filtered_YYYYMMDD.csv` ← **must be updated first**
- `net_rankings_YYYYMMDD.csv` ← **must be updated first**
- `d1_player_benchmarks_2025.csv` (optional — skipped if missing)

**Outputs:**
- `player_box_advanced_metrics.csv` — season-level advanced metrics per player (4,475 rows). Feeds _merge_scouting.py and player_feature_table.
- `player_game_log_enriched.csv` — game-level log with rolling features, L5/L10/L15 windows, opponent context (118,736 rows).

**What it does:** 10-stage pipeline — cleans box data, joins game context, computes Game Score/PPP, builds rolling/trend features, aggregates to season, computes rate metrics, benchmarks against D1 averages, labels archetypes, builds position percentiles, flags breakout signals.

---

### Script 2: `pbp_player_processing.py`

**Run via:** `update_pipeline.py` (Step 2) or directly

**Inputs:**
- `play_by_play_20260327.parquet` ← **must be updated first**

**Outputs:**
- `pbp_player_metrics.csv` — PBP-derived player metrics per season (8,039 rows): shot zones (paint/midrange/three), creation metrics, clutch splits, game-phase efficiency, hustle stats, categorical labels (Shot Creator Type, Playmaking Style, Half Adjustment).

**What it does:** Classifies all 2.6M+ PBP rows into shot events → aggregates shot profiles per player → computes assist/creation metrics → game phase splits → hustle event counts → merges all, applies labels, exports.

---

### Script 3: `lineup_stints_pipeline.py`

**Run via:** Standalone (not in `update_pipeline.py`)

**Inputs:**
- `play_by_play_20260327.parquet` (hardcoded path — update `DATA_PATH` constant)
- `net_rankings_20260302.csv` (hardcoded path — update `RANKS_PATH` constant)

**Filter:** `season_type == 2` (regular season only, as currently written)

**Outputs:**
- `lineup_stints_raw.csv` — stint-level data (25,609 rows): every lineup change tracked with duration, pts scored/allowed, possession estimates, shot quality, game state context. Max date: **March 9, 2026** (regular season + conf tourney only)
- `player_onoff_metrics.csv` — player-level on/off aggregation (403 players): on/off ratings per 100 poss, net rating differential, half splits, game-state splits, opponent quality splits, lineup summaries.

**What it does:** Reconstructs lineups from substitution events for 32 tournament teams → tracks every lineup stint → aggregates per-100 offensive/defensive/net ratings → computes on/off differentials and contextual splits.

> **Note for postseason version:** `postseason_stints_raw.csv` and `postseason_onoff_metrics.csv` were generated by running this same logic on `Postseason Data/march_pbp_2026.parquet` with `season_type` filter removed or changed. **There is currently no saved version of this modified script** — this needs to be formalized as a second script for the GitHub repo.

---

### Script 4: `build_visualization.py`

**Run via:** Standalone

**Inputs:**
- `Kaggle Datasets copy/WNCAATourneySeeds.csv`
- `Kaggle Datasets copy/WNCAATourneyCompactResults.csv`
- `Kaggle Datasets copy/WTeams.csv`
- `Kaggle Datasets copy/WTeamConferences.csv`
- `Kaggle Datasets copy/WTeamSpellings.csv`

**Outputs:**
- `wbb_conference_funnel.html` (likely) — conference tournament history visualization using historical Kaggle data.

**What it does:** Builds historical conference → tournament pipeline visualization from 2003–present Kaggle data.

---

## TIER 2 — Merged / Feature Tables

### Script 5: `_merge_scouting.py`

**Run via:** `update_pipeline.py` (Step 3) or directly

**Inputs:**
- `player_box_advanced_metrics.csv` ← Tier 1 output
- `pbp_player_metrics.csv` ← Tier 1 output
- `tournament_bracket.csv` ← Tier 0 static

**Outputs:**
- `player_scouting_tournament68.csv` — full scouting profiles for all 68 tournament teams (757 rows): box + PBP metrics merged.
- `player_scouting_top50.csv` — same merge filtered to top-50 NET teams (527 rows): used by main dashboard.

**What it does:** Filters box metrics to tournament or top-50 NET teams, left-joins PBP metrics on athlete_id, drops duplicate columns.

---

### Script 6: ⚠️ `merge_feature_table.py` — **NEEDS TO BE BUILT**

**Purpose:** Produces `player_feature_table_2026.csv` — the ML-ready single wide table (403 rows) combining all player-level metrics from Tier 1 sources.

**Inputs (inferred from column structure):**
- `player_box_advanced_metrics.csv`
- `pbp_player_metrics.csv`
- `player_onoff_metrics.csv`
- `wbb_rosters_2025_26.csv` (for identity fields: position, height, hometown, class)

**Outputs:**
- `player_feature_table_2026.csv` — wide feature table for archetype modeling and role classification

**Status:** No script currently exists for this file. Was likely produced ad-hoc. Needs to be written and formalized for the GitHub repo.

---

### Script 7: ⚠️ `postseason_lineup_pipeline.py` — **NEEDS TO BE BUILT**

**Purpose:** Postseason version of the lineup stints pipeline, producing postseason on/off metrics.

**Inputs:**
- `Postseason Data/march_pbp_2026.parquet` (needs update through Championship)
- `net_rankings_manual_20260316.csv` (or updated version)

**Outputs:**
- `postseason_stints_raw.csv` — tournament stint-level data (max currently March 27)
- `postseason_onoff_metrics.csv` — player-level postseason on/off (337 players)

**Status:** ✅ `postseason_lineup_pipeline.py` written 2026-04-10. Loads teams dynamically from `tournament_bracket.csv` (no hardcoded IDs), uses `config.py` paths throughout, removes `season_type` filter, outputs to `postseason_stints_raw.csv` + `postseason_onoff_metrics.csv`. Re-run once `march_pbp_2026.parquet` is updated through Championship.

---

## TIER 3 — Dashboard / Report Outputs

| Output File | Script | Inputs |
|---|---|---|
| `wbb_player_dashboard.html` | ⚠️ `build_dashboard_v2.py` — **CONFIRMED MISSING from Drive** | player_scouting_top50.csv + schedule + NET + tournament files |
| `wbb_tournament68_dashboard.html` | ⚠️ likely `build_dashboard_v2.py` variant — also missing | player_scouting_tournament68.csv |
| `wbb_conference_funnel.html` | `build_visualization.py` | Kaggle historical datasets |

> **Note on `build_dashboard_v2.py`:** Searched all `.py` files across the entire WBB Google Drive folder — confirmed not saved anywhere. The script ran in a prior session but was never written to Drive. `update_pipeline.py` expects it at `WBB/build_dashboard_v2.py` (one level up from the working folder). This script **must be rebuilt** for the GitHub repo. Draft dashboard scripts are out of scope for the BIS pipeline (separate portfolio project).

---

## Rebuild Order (after source data is updated)

Run these in strict sequence — each step depends on the one above it.

```
STEP 0  ── Deliver updated raw data:
            • player_box_2026.parquet          (through Championship, ~Apr 5)
            • play_by_play_YYYYMMDD.parquet    (through Championship)
            • Postseason Data/player_box_2026_march.parquet  (same)
            • Postseason Data/march_pbp_2026.parquet         (same)
            • schedule_filtered_YYYYMMDD.csv   (final schedule + results)
            • net_rankings_YYYYMMDD.csv        (final NET pull)

STEP 1  ── python player_box_processing.py
            → rebuilds: player_box_advanced_metrics.csv
                        player_game_log_enriched.csv

STEP 2  ── python pbp_player_processing.py
            → rebuilds: pbp_player_metrics.csv

STEP 3  ── python lineup_stints_pipeline.py       (regular season filter)
            → rebuilds: lineup_stints_raw.csv
                        player_onoff_metrics.csv

STEP 4  ── python postseason_lineup_pipeline.py   (⚠️ needs to be written)
            → rebuilds: postseason_stints_raw.csv
                        postseason_onoff_metrics.csv

STEP 5  ── python _merge_scouting.py
            → rebuilds: player_scouting_tournament68.csv
                        player_scouting_top50.csv

STEP 6  ── python merge_feature_table.py          (⚠️ needs to be written)
            → rebuilds: player_feature_table_2026.csv

STEP 7  ── python build_dashboard_v2.py           (locate + add to repo)
            → rebuilds: wbb_player_dashboard.html
                        wbb_tournament68_dashboard.html
```

**Or use the orchestrator:** `update_pipeline.py` runs Steps 1–2–5–7 automatically (Steps 3, 4, and 6 run separately).

---

## GitHub Repo Structure (Proposed)

```
wbb-bis/
├── README.md
├── requirements.txt
├── .gitignore                     # data/, outputs/ (large files via LFS or excluded)
│
├── data/                          # Raw source data (gitignored or LFS)
│   ├── raw/
│   │   ├── player_box_2026.parquet
│   │   ├── play_by_play_2026.parquet
│   │   ├── schedule_filtered_YYYYMMDD.csv
│   │   ├── net_rankings_YYYYMMDD.csv
│   │   └── postseason/
│   │       ├── march_pbp_2026.parquet
│   │       └── player_box_2026_march.parquet
│   ├── static/                    # Static / reference datasets (can commit)
│   │   ├── tournament_bracket.csv
│   │   ├── tournament_storylines.csv
│   │   ├── d1_player_benchmarks_2025.csv
│   │   ├── wbb_rosters_2025_26.csv
│   │   ├── player_recruit_rankings_20212026.csv
│   │   ├── hist_draft_2014_2024_full_raw.csv
│   │   └── kaggle/                # Historical Kaggle datasets
│   └── processed/                 # Tier 1-2 outputs (gitignored, regenerated)
│
├── scripts/
│   ├── 01_player_box_processing.py
│   ├── 02_pbp_player_processing.py
│   ├── 03_lineup_stints_regular.py
│   ├── 04_lineup_stints_postseason.py    ← needs to be written
│   ├── 05_merge_scouting.py
│   ├── 06_merge_feature_table.py         ← needs to be written
│   ├── 07_build_dashboard.py             ← locate build_dashboard_v2.py
│   ├── 08_build_visualization.py
│   └── utils/
│       └── advanced_metrics.py           ← utility library
│
├── dashboards/                    # Final HTML outputs
│   ├── wbb_player_dashboard.html
│   ├── wbb_tournament68_dashboard.html
│   ├── wbb_conference_funnel.html
│   └── draft/
│       ├── wbb_draft_rankings_2026.html
│       └── wbb_draft_prospect_landscape_2026.html
│
├── pipeline.py                    # Orchestrator (refactored update_pipeline.py)
│
└── docs/
    ├── pipeline_dependency_map.md (this file)
    └── data_dictionary.md
```

---

## Scripts That Need to Be Written / Located

| Priority | Script | What It Does | Inputs | Outputs | Status |
|---|---|---|---|---|---|
| ✅ Done | `config.py` | Centralizes all file paths + constants | — | — | Written 2026-04-10 |
| ✅ Done | `postseason_lineup_pipeline.py` | Postseason on/off from tournament PBP | march_pbp_2026.parquet + bracket | postseason_stints_raw.csv, postseason_onoff_metrics.csv | Written 2026-04-10 |
| 🔴 High | `merge_feature_table.py` | Wide ML-ready player feature table | box_advanced + pbp_metrics + onoff + rosters | player_feature_table_2026.csv | Not written |
| 🔴 High | `build_dashboard.py` | Main player + tournament68 dashboard HTML | player_scouting_top50.csv + tournament + schedule + NET | wbb_player_dashboard.html, wbb_tournament68_dashboard.html | **Confirmed missing from Drive — must be rebuilt** |

---

## Notes for BIS GitHub Build

- `advanced_metrics.py` is a **utility/reference library**, not a pipeline runner — move to `scripts/utils/` and import from other scripts as needed.
- `update_pipeline.py` should be refactored as `pipeline.py` at the repo root — clean up hardcoded session paths in `lineup_stints_pipeline.py` before committing.
- Consider adding a `config.py` to centralize all file path constants so they don't need to be patched per-run.
- Data files >50MB should use Git LFS or be documented in README with download instructions (wehoop GitHub release URLs are already in `pbp_player_processing.py`).
