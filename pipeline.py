"""
WBB Dashboard — One-Click Update Pipeline
==========================================
Run this script whenever you have new data to refresh the entire dashboard.

WHAT IT DOES (in order):
  1. Validates all required input files are present
  2. Runs 01_player_box_processing.py  → regenerates player_box_advanced_metrics.csv
                                          and player_game_log_enriched.csv
  3. Runs 02_pbp_player_processing.py  → regenerates pbp_player_metrics.csv
  4. Merges box + PBP metrics          → creates player_scouting_tournament68.csv
                                          and player_scouting_top50.csv
  5. Runs 07_build_dashboard.py        → rebuilds wbb_player_dashboard.html

To run lineup stints (slow; run separately after raw PBP is updated):
  python scripts/03_lineup_stints_regular.py
  python scripts/04_lineup_stints_postseason.py

FILES YOU MUST PROVIDE BEFORE RUNNING:
  ✅ data/raw/player_box_2026.parquet
  ✅ data/raw/play_by_play_YYYYMMDD.parquet
  ✅ data/raw/schedule_filtered_YYYYMMDD.csv   — update SCHEDULE_FILE in config.py
  ✅ data/raw/net_rankings_YYYYMMDD.csv        — update NET_FILE in config.py

Author: Krystal B Creative — Sports Analytics Portfolio
Date:   2026-04-11
"""

import subprocess
import sys
from pathlib import Path

# =============================================================================
# CONFIG IMPORTS — all paths come from config.py
# =============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    REPO_ROOT,
    PLAYER_BOX_FILE, PBP_FILE, SCHEDULE_FILE, NET_FILE,
    BENCHMARKS_FILE, TOURNAMENT_BRACKET, TOURNAMENT_STORIES,
    validate_inputs,
)

SCRIPTS_DIR = REPO_ROOT / "scripts"

# ── Behaviour flags ───────────────────────────────────────────────────────────
SKIP_BENCHMARKS = not BENCHMARKS_FILE.exists()   # auto-skip if file missing
REBUILD_PBP     = PBP_FILE.exists()              # skip PBP step if parquet missing
EXPAND_TO_68    = True                           # create tournament68 scouting file


# =============================================================================
# HELPERS
# =============================================================================

def run(label: str, script: Path, extra_args: list = None):
    """Run a Python script with the current interpreter, print result."""
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  CMD : {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n❌  Step failed: {label}  (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"✅  {label} — complete")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    print("\n" + "╔"+"═"*62+"╗")
    print("║   WBB DASHBOARD — ONE-CLICK UPDATE PIPELINE              ║")
    print("╚"+"═"*62+"╝")

    # ── Step 0: validate inputs ───────────────────────────────────────────────
    if not validate_inputs(
        required=[PLAYER_BOX_FILE, SCHEDULE_FILE, NET_FILE, TOURNAMENT_BRACKET],
        optional=[PBP_FILE, BENCHMARKS_FILE, TOURNAMENT_STORIES],
    ):
        sys.exit(1)

    # ── Step 1: player box processing → box_advanced_metrics + game_log ───────
    if SKIP_BENCHMARKS:
        print("⚠️  Benchmarks file missing — breakout signals will be skipped.")
        print("   (drop d1_player_benchmarks_2025.csv into data/static/ to enable)")
    run("Player Box Processing (box metrics + game log)",
        SCRIPTS_DIR / "01_player_box_processing.py")

    # ── Step 2: PBP processing → pbp_player_metrics ───────────────────────────
    if REBUILD_PBP:
        run("PBP Processing (shot zones + clutch metrics)",
            SCRIPTS_DIR / "02_pbp_player_processing.py")
    else:
        print("\n⚠️  Skipping PBP step — play_by_play parquet not found in data/raw/.")
        print("   Existing pbp_player_metrics.csv will be used as-is.")

    # ── Step 3: merge box + PBP → scouting file ───────────────────────────────
    if EXPAND_TO_68:
        run("Merge Box + PBP → tournament68 scouting file",
            SCRIPTS_DIR / "05_merge_scouting.py")
    else:
        print("\n⚠️  Skipping scouting merge — using existing player_scouting_top50.csv")

    # ── Step 4: rebuild dashboard ─────────────────────────────────────────────
    run("Build Dashboard HTML", SCRIPTS_DIR / "07_build_dashboard.py")

    print("\n" + "╔"+"═"*62+"╗")
    print("║   ✅  PIPELINE COMPLETE — dashboard refreshed!            ║")
    print("╚"+"═"*62+"╝\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()
