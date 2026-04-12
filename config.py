"""
BIS Pipeline — Central Configuration
=====================================
All file paths and constants live here. Import this module in every script
so that nothing has machine-specific or session-specific paths baked in.

Usage:
    from config import PATHS, CONSTANTS, validate_inputs
    # or import specific names:
    from config import PLAYER_BOX_FILE, SCHEDULE_FILE, CONSTANTS

To update for new data:  change only the dated filenames in the
"Live / updatable files" section below — everything downstream auto-updates.

Author:  Krystal B Creative — Sports Analytics Portfolio
Updated: 2026-04-11
"""

import sys
from pathlib import Path

# =============================================================================
# ROOT RESOLUTION
# =============================================================================
# REPO_ROOT resolves to the directory containing this config.py file,
# regardless of operating system or where the repo is cloned.
# Every path below is built relative to this — no hardcoded machine paths.

REPO_ROOT = Path(__file__).resolve().parent

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

RAW_DIR         = REPO_ROOT / "data" / "raw"
POSTSEASON_DIR  = REPO_ROOT / "data" / "raw" / "postseason"
STATIC_DIR      = REPO_ROOT / "data" / "static"
PROCESSED_DIR   = REPO_ROOT / "data" / "processed"
ROSTER_DIR      = REPO_ROOT / "data" / "static" / "rosters"
RECRUITING_DIR  = REPO_ROOT / "data" / "static" / "recruiting"
POLLS_DIR       = REPO_ROOT / "data" / "static" / "polls"
KAGGLE_DIR      = REPO_ROOT / "data" / "static" / "kaggle"
DASHBOARD_DIR   = REPO_ROOT / "dashboards"


# =============================================================================
# SEASON METADATA
# =============================================================================

SEASON        = 2026          # wehoop season integer (year the season ends)
SEASON_LABEL  = "2025-26"     # human-readable label


# =============================================================================
# TIER 0 — RAW SOURCE FILES
# ─────────────────────────────────────────────────────────────────────────────
# WHEN NEW DATA ARRIVES: update only the dated filenames here.
# Everything downstream picks up the change automatically.
# =============================================================================

# ── Live / updatable files ────────────────────────────────────────────────────
PLAYER_BOX_FILE  = RAW_DIR / "player_box_2026_final.parquet"
PBP_FILE         = RAW_DIR / "play_by_play_20260327.parquet"
SCHEDULE_FILE    = RAW_DIR / "schedule_filtered_20260405.csv"     
NET_FILE         = RAW_DIR / "net_rankings_manual_20260405.csv"
SCHEDULE_FILE_FULL    = RAW_DIR / "wbb_schedule_2026_final.parquet"     
TEAM_BOX_FILE_FULL    = RAW_DIR / "team_box_2026_final.parquet"     


# ── Postseason raw files ──────────────────────────────────────────────────────
POSTSEASON_PBP_FILE  = POSTSEASON_DIR / "march_pbp_2026.parquet"
POSTSEASON_BOX_FILE  = POSTSEASON_DIR / "player_box_2026_march.parquet"
POSTSEASON_CLEAN_BOX = POSTSEASON_DIR / "clean_player_box_march.parquet"
CONF_PBP_FILE        = POSTSEASON_DIR / "clean_conference_pbp.parquet"

# ── Static / reference files (rarely change) ─────────────────────────────────
BENCHMARKS_FILE    = STATIC_DIR  / "d1_player_benchmarks_2025.csv"
TOURNAMENT_BRACKET = STATIC_DIR  / "tournament_bracket.csv"
TOURNAMENT_STORIES = STATIC_DIR  / "tournament_storylines.csv"
ROSTER_FILE        = ROSTER_DIR  / "wbb_rosters_2025_26.csv"
RECRUIT_RANKINGS   = RECRUITING_DIR / "player_recruit_rankings_20212026.csv"

# ── Kaggle historical datasets ────────────────────────────────────────────────
KAGGLE_SEEDS       = KAGGLE_DIR / "WNCAATourneySeeds.csv"
KAGGLE_TOURNEY     = KAGGLE_DIR / "WNCAATourneyCompactResults.csv"
KAGGLE_TOURNEY_DET = KAGGLE_DIR / "WNCAATourneyDetailedResults.csv"
KAGGLE_REG         = KAGGLE_DIR / "WRegularSeasonCompactResults.csv"
KAGGLE_REG_DET     = KAGGLE_DIR / "WRegularSeasonDetailedResults.csv"
KAGGLE_TEAMS       = KAGGLE_DIR / "WTeams.csv"
KAGGLE_CONFS       = KAGGLE_DIR / "WTeamConferences.csv"
KAGGLE_SPELLINGS   = KAGGLE_DIR / "WTeamSpellings.csv"

# ── Dashboard config (used by 08_build_visualization.py) ─────────────────────
DASHBOARD_CONF = DASHBOARD_DIR / "wbb_conference_funnel.html"


# =============================================================================
# TIER 1 — PROCESSED OUTPUT FILES
# =============================================================================

# player_box_processing.py  →
PLAYER_BOX_ADVANCED = PROCESSED_DIR / "player_box_advanced_metrics.csv"
PLAYER_GAME_LOG     = PROCESSED_DIR / "player_game_log_enriched.csv"

# pbp_player_processing.py  →
PBP_PLAYER_METRICS  = PROCESSED_DIR / "pbp_player_metrics.csv"

# lineup_stints_pipeline.py  → (regular season)
LINEUP_STINTS_RAW   = PROCESSED_DIR / "lineup_stints_raw.csv"
PLAYER_ONOFF        = PROCESSED_DIR / "player_onoff_metrics.csv"

# postseason_lineup_pipeline.py  →
POSTSEASON_STINTS   = PROCESSED_DIR / "postseason_stints_raw.csv"
POSTSEASON_ONOFF    = PROCESSED_DIR / "postseason_onoff_metrics.csv"


# =============================================================================
# TIER 2 — MERGED / FEATURE OUTPUT FILES
# =============================================================================

PLAYER_SCOUTING_68   = PROCESSED_DIR / "player_scouting_tournament68.csv"
PLAYER_SCOUTING_50   = PROCESSED_DIR / "player_scouting_top50.csv"
PLAYER_FEATURE_TABLE = PROCESSED_DIR / "player_feature_table_2026.csv"


# =============================================================================
# TIER 3 — DASHBOARD OUTPUT FILES
# =============================================================================

DASHBOARD_MAIN    = DASHBOARD_DIR / "wbb_player_dashboard.html"
DASHBOARD_T68     = DASHBOARD_DIR / "wbb_tournament68_dashboard.html"
DASHBOARD_SWEET16 = DASHBOARD_DIR / "wbb_sweet16_dashboard.html"


# =============================================================================
# CONSTANTS — SHARED ACROSS SCRIPTS
# =============================================================================

CONSTANTS = {

    # ── player_box_processing.py ─────────────────────────────────────────────
    "MIN_MINUTES_SEASON":    50,         # Minimum season minutes to include a player
    "MIN_MINUTES_GAME":       5,         # Minimum game minutes to count toward rolling stats
    "MIN_GAMES_PERCENTILE":  10,         # Minimum games played for percentile ranking
    "ROLLING_WINDOWS":  [5, 10, 15],     # L5, L10, L15 rolling windows
    "USG_THRESHOLD_HIGH":  0.22,         # ~top 40% usage (quadrant split)
    "TS_THRESHOLD_HIGH":   0.52,         # ~top 40% TS% (quadrant split)
    "MINUTES_TIERS": {
        "Star":       28,
        "Rotation":   20,
        "Bench":      10,
        "Deep Bench":  0,
    },

    # ── pbp_player_processing.py ─────────────────────────────────────────────
    "MIN_FGM_ASSISTED":  10,             # Min made FGs for assisted rate to be reliable
    "CLUTCH_QUARTER":     4,             # Quarter for clutch window
    "CLUTCH_SECONDS":   300,             # Seconds remaining = last 5 minutes
    "CLUTCH_MARGIN":      5,             # Max scoring margin to be "clutch"

    # ── lineup_stints pipelines (regular + postseason) ────────────────────────
    "MIN_STINT_PLAYERS":  4,             # Minimum on-court players to record a stint
    "MIN_POSSESSIONS":    5,             # Minimum possessions for per-100 ratings
    "SEASON_TYPE_REGULAR":    2,         # ESPN season_type code — regular season
    "SEASON_TYPE_POSTSEASON": 3,         # ESPN season_type code — NCAA tournament

    # ── NET ranking tier boundaries (NCAA quadrant system) ───────────────────
    "NET_Q1_MAX":   30,
    "NET_Q2_MAX":   75,
    "NET_Q3_MAX":  160,
}


# =============================================================================
# CONVENIENCE GROUPINGS  (for scripts that need to iterate over file sets)
# =============================================================================

# Files that must exist before any pipeline step can run
REQUIRED_INPUTS = [
    PLAYER_BOX_FILE,
    PBP_FILE,
    SCHEDULE_FILE,
    NET_FILE,
    TOURNAMENT_BRACKET,
]

# Files that are optional (pipeline degrades gracefully if missing)
OPTIONAL_INPUTS = [
    BENCHMARKS_FILE,
    TOURNAMENT_STORIES,
    POSTSEASON_PBP_FILE,
    POSTSEASON_BOX_FILE,
    RECRUIT_RANKINGS,
]


# =============================================================================
# VALIDATION HELPER
# =============================================================================

def validate_inputs(required: list, optional: list = None) -> bool:
    """
    Check that required files exist before running a pipeline step.
    Prints a status line for each file.

    Args:
        required: List of Path objects that must exist to proceed.
        optional: List of Path objects that are nice-to-have (warns if absent).

    Returns:
        True if all required files are present, False otherwise.

    Example:
        import sys
        from config import validate_inputs, PLAYER_BOX_FILE, SCHEDULE_FILE, NET_FILE
        if not validate_inputs([PLAYER_BOX_FILE, SCHEDULE_FILE, NET_FILE]):
            sys.exit(1)
    """
    print("\n── Input file validation ──────────────────────────────────────────")
    all_present = True

    for f in required:
        f = Path(f)
        if f.exists():
            print(f"  ✅  {f.name}")
        else:
            print(f"  ❌  {f.name}  (REQUIRED — file missing)")
            all_present = False

    for f in (optional or []):
        f = Path(f)
        if f.exists():
            print(f"  ✅  {f.name}  (optional)")
        else:
            print(f"  ⚠️   {f.name}  (optional — will be skipped if needed)")

    if not all_present:
        print("\n  ❌  Cannot proceed — one or more required files are missing.\n")
    else:
        print("\n  All required files present.\n")

    return all_present


# =============================================================================
# QUICK SANITY CHECK  (run this file directly to verify paths on a new machine)
# =============================================================================

if __name__ == "__main__":
    print(f"\nBIS Pipeline Config — {SEASON_LABEL} Season")
    print(f"REPO_ROOT : {REPO_ROOT}\n")

    all_files = {
        "Raw inputs": [
            PLAYER_BOX_FILE, PBP_FILE, SCHEDULE_FILE, NET_FILE,
            POSTSEASON_PBP_FILE, POSTSEASON_BOX_FILE,
            BENCHMARKS_FILE, TOURNAMENT_BRACKET, ROSTER_FILE,
        ],
        "Tier 1 outputs": [
            PLAYER_BOX_ADVANCED, PLAYER_GAME_LOG, PBP_PLAYER_METRICS,
            LINEUP_STINTS_RAW, PLAYER_ONOFF, POSTSEASON_STINTS, POSTSEASON_ONOFF,
        ],
        "Tier 2 outputs": [
            PLAYER_SCOUTING_68, PLAYER_SCOUTING_50, PLAYER_FEATURE_TABLE,
        ],
        "Tier 3 outputs": [
            DASHBOARD_MAIN, DASHBOARD_T68, DASHBOARD_CONF,
        ],
    }

    for section, paths in all_files.items():
        print(f"  {section}:")
        for p in paths:
            status = "✅" if Path(p).exists() else "--"
            print(f"    {status}  {Path(p).name}")
        print()

    sys.exit(0)
