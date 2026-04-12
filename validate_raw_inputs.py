"""
validate_raw_inputs.py — BIS Pipeline Data Quality Check
=========================================================
Run this BEFORE and AFTER updating raw data files to confirm that each
dataset contains complete 2025-26 season coverage.

Checks per file:
  ✅ File exists and is readable
  ✅ Row count is within expected range (not empty or suspiciously small)
  ✅ Season 2026 rows are present
  ✅ Max game/event date reaches end of season (≥ cutoff date per file)
  ✅ Required columns are present

Usage:
    python validate_raw_inputs.py

Author:  Krystal B Creative — Sports Analytics Portfolio
"""

import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import date

import pandas as pd

# ── Config import ─────────────────────────────────────────────────────────────
try:
    from config import (
        PLAYER_BOX_FILE, PBP_FILE, SCHEDULE_FILE, NET_FILE,
        POSTSEASON_PBP_FILE, POSTSEASON_BOX_FILE,
        TOURNAMENT_BRACKET, ROSTER_FILE,
        SEASON,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        PLAYER_BOX_FILE, PBP_FILE, SCHEDULE_FILE, NET_FILE,
        POSTSEASON_PBP_FILE, POSTSEASON_BOX_FILE,
        TOURNAMENT_BRACKET, ROSTER_FILE,
        SEASON,
    )


# =============================================================================
# SEASON COVERAGE TARGETS
# =============================================================================
# What "complete" looks like for the 2025-26 season.
# Regular season ended ~March 9, 2026.
# NCAA Tournament ended April 6, 2026 (National Championship).

REGULAR_SEASON_END = date(2026, 3, 9)    # last regular season game
TOURNEY_END        = date(2026, 4, 6)    # national championship date
MIN_SEASON_DATE    = date(2025, 11, 1)   # first game of 2025-26 season


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

passed = 0
failed = 0
warned = 0

def _ok(msg):
    global passed
    passed += 1
    print(f"    ✅  {msg}")

def _fail(msg):
    global failed
    failed += 1
    print(f"    ❌  {msg}")

def _warn(msg):
    global warned
    warned += 1
    print(f"    ⚠️   {msg}")

def _header(label, path, required=True):
    tag = "(required)" if required else "(optional)"
    print(f"\n  {'─'*56}")
    print(f"  {label}  {tag}")
    print(f"  {path.name}")
    print(f"  {'─'*56}")
    if not path.exists():
        if required:
            _fail(f"File not found — pipeline cannot run without this file")
        else:
            _warn(f"File not found — pipeline will skip this source")
        return False
    return True

def _find_date_col(df, candidates):
    """Return the first column from candidates that exists in df, or None."""
    return next((c for c in candidates if c in df.columns), None)

def _max_date(df, col):
    """Parse a date column and return the max as a date object."""
    try:
        return pd.to_datetime(df[col], errors='coerce').dt.date.max()
    except Exception:
        return None

def _check_date_coverage(df, col_candidates, cutoff, label):
    col = _find_date_col(df, col_candidates)
    if col is None:
        _warn(f"No date column found (tried: {col_candidates}) — skipping date check")
        return
    mx = _max_date(df, col)
    if mx is None:
        _warn(f"Could not parse dates in '{col}'")
        return
    if mx >= cutoff:
        _ok(f"Max {col}: {mx}  (≥ {cutoff} target ✓)")
    else:
        _fail(f"Max {col}: {mx}  — expected ≥ {cutoff}  ← DATA INCOMPLETE")

def _check_season(df, season_col_candidates, season_val):
    col = _find_date_col(df, season_col_candidates)
    if col is None:
        _warn(f"No season column found — skipping season filter check")
        return
    n = (df[col] == season_val).sum()
    if n > 0:
        _ok(f"Season {season_val} rows: {n:,}")
    else:
        _fail(f"No rows with season == {season_val}  ← wrong season or wrong file")

def _check_cols(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        _fail(f"Missing expected columns: {missing}")
    else:
        _ok(f"All expected columns present")

def _check_rows(df, min_rows, max_rows=None):
    n = len(df)
    if n < min_rows:
        _fail(f"Row count: {n:,}  — suspiciously low (expected ≥ {min_rows:,})")
    elif max_rows and n > max_rows:
        _warn(f"Row count: {n:,}  — unusually high (expected ≤ {max_rows:,}); verify filter")
    else:
        _ok(f"Row count: {n:,}  (within expected range)")


# =============================================================================
# FILE-BY-FILE VALIDATION
# =============================================================================

print(f"\n{'='*60}")
print(f"  BIS Pipeline — Raw Data Validation")
print(f"  Season: {SEASON}  (2025-26)")
print(f"  Regular season target date : {REGULAR_SEASON_END}")
print(f"  Tournament target date     : {TOURNEY_END}")
print(f"{'='*60}")


# ── 1. Player Box Scores ──────────────────────────────────────────────────────
if _header("1. PLAYER BOX SCORES", PLAYER_BOX_FILE, required=True):
    try:
        df = pd.read_parquet(PLAYER_BOX_FILE)
        _check_rows(df, min_rows=50_000, max_rows=500_000)
        _check_season(df, ['season', 'Season'], SEASON)
        _check_date_coverage(df,
            ['game_date', 'game_date_time', 'date'],
            REGULAR_SEASON_END,
            "last regular season game")
        _check_cols(df, ['athlete_id', 'team_id', 'game_id', 'minutes',
                         'points', 'rebounds', 'assists'])
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 2. Play-by-Play ───────────────────────────────────────────────────────────
if _header("2. PLAY-BY-PLAY", PBP_FILE, required=True):
    try:
        df = pd.read_parquet(PBP_FILE)
        _check_rows(df, min_rows=500_000, max_rows=10_000_000)
        _check_season(df, ['season', 'Season'], SEASON)
        _check_date_coverage(df,
            ['game_date', 'start_date_time', 'start_date', 'date'],
            REGULAR_SEASON_END,
            "last regular season game")
        _check_cols(df, ['game_id', 'team_id', 'athlete_id_1',
                         'type_id', 'period_display_value'])
        # Confirm it contains season_type == 2 (regular season) rows
        if 'season_type' in df.columns:
            n_reg = (df['season_type'] == 2).sum()
            n_post = (df['season_type'] == 3).sum()
            if n_reg > 0:
                _ok(f"Regular season rows (season_type=2): {n_reg:,}")
            else:
                _fail("No regular season rows (season_type=2) found")
            if n_post > 0:
                _ok(f"Postseason rows (season_type=3): {n_post:,}")
            else:
                _warn("No postseason rows in this file — postseason pipeline needs separate file")
        else:
            _warn("No 'season_type' column — could not verify regular/postseason split")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 3. Schedule ───────────────────────────────────────────────────────────────
if _header("3. SCHEDULE", SCHEDULE_FILE, required=True):
    try:
        df = pd.read_csv(SCHEDULE_FILE, low_memory=False)
        _check_rows(df, min_rows=3_000, max_rows=20_000)
        _check_season(df, ['season', 'Season'], SEASON)
        _check_date_coverage(df,
            ['game_date', 'date', 'start_date'],
            REGULAR_SEASON_END,
            "last regular season game")
        _check_cols(df, ['game_id', 'home_team_id', 'away_team_id'])
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 4. NET Rankings ───────────────────────────────────────────────────────────
if _header("4. NET RANKINGS", NET_FILE, required=True):
    try:
        df = pd.read_csv(NET_FILE)
        _check_rows(df, min_rows=200, max_rows=5_000)
        # NET rankings are a snapshot file — check run_date is recent
        _check_date_coverage(df,
            ['run_date', 'date', 'pull_date', 'snapshot_date'],
            REGULAR_SEASON_END,
            "end-of-regular-season snapshot")
        _check_cols(df, ['team_id', 'net_rank'])
        # Confirm all D1 teams are represented (should be ~360)
        if 'team_id' in df.columns:
            n_teams = df['team_id'].nunique()
            if n_teams >= 300:
                _ok(f"Teams in NET file: {n_teams}  (full D1 field ✓)")
            else:
                _warn(f"Teams in NET file: {n_teams}  — expected ~360 for full D1")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 5. Postseason PBP ─────────────────────────────────────────────────────────
if _header("5. POSTSEASON PLAY-BY-PLAY", POSTSEASON_PBP_FILE, required=False):
    try:
        df = pd.read_parquet(POSTSEASON_PBP_FILE)
        _check_rows(df, min_rows=10_000, max_rows=500_000)
        _check_date_coverage(df,
            ['game_date', 'start_date_time', 'start_date', 'date'],
            TOURNEY_END,
            "national championship game (Apr 6)")
        _check_cols(df, ['game_id', 'team_id'])
        if 'season_type' in df.columns:
            n_post = (df['season_type'] == 3).sum()
            if n_post > 0:
                _ok(f"Postseason rows (season_type=3): {n_post:,}")
            else:
                _warn("season_type=3 not found — verify this is the postseason file")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 6. Postseason Player Box ──────────────────────────────────────────────────
if _header("6. POSTSEASON PLAYER BOX", POSTSEASON_BOX_FILE, required=False):
    try:
        df = pd.read_parquet(POSTSEASON_BOX_FILE)
        _check_rows(df, min_rows=500, max_rows=10_000)
        _check_date_coverage(df,
            ['game_date', 'date'],
            TOURNEY_END,
            "national championship game (Apr 6)")
        _check_cols(df, ['athlete_id', 'team_id', 'game_id'])
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 7. Tournament Bracket (static — spot-check) ───────────────────────────────
if _header("7. TOURNAMENT BRACKET", TOURNAMENT_BRACKET, required=True):
    try:
        df = pd.read_csv(TOURNAMENT_BRACKET)
        n_teams = len(df)
        if n_teams == 68:
            _ok(f"Team count: {n_teams}  (full 68-team field ✓)")
        elif n_teams == 64:
            _warn(f"Team count: {n_teams}  — may be missing play-in teams (expected 68)")
        elif n_teams < 32:
            _fail(f"Team count: {n_teams}  — too few teams; file may be incomplete")
        else:
            _ok(f"Team count: {n_teams}")
        _check_cols(df, ['seed', 'region'])
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 8. Rosters (optional spot-check) ─────────────────────────────────────────
if _header("8. ROSTERS 2025-26", ROSTER_FILE, required=False):
    try:
        df = pd.read_csv(ROSTER_FILE, low_memory=False)
        _check_rows(df, min_rows=3_000, max_rows=20_000)
        # Roster file should cover current season
        if 'season' in df.columns:
            _check_season(df, ['season'], SEASON)
        else:
            _warn("No 'season' column — could not verify season coverage")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# =============================================================================
# SUMMARY
# =============================================================================

total = passed + failed + warned
print(f"\n{'='*60}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  ✅  Passed : {passed}")
print(f"  ❌  Failed : {failed}")
print(f"  ⚠️   Warned : {warned}")
print(f"  {'─'*40}")

if failed == 0 and warned == 0:
    print(f"  🟢  All checks passed — data is ready for pipeline.py")
elif failed == 0:
    print(f"  🟡  No failures, but review warnings above before running pipeline.py")
else:
    print(f"  🔴  {failed} check(s) failed — resolve before running pipeline.py")
    print(f"\n  Common fixes:")
    print(f"    • 'DATA INCOMPLETE' on a date check → re-download from wehoop with final data")
    print(f"    • 'wrong season' → confirm you pulled season=2026 from wehoop")
    print(f"    • 'File not found' on a required file → update filename in config.py")
    print(f"    • 'Missing columns' → file schema may have changed in a wehoop release")

print(f"\n  To update config.py filenames:")
print(f"    Edit SCHEDULE_FILE and NET_FILE (and PBP_FILE if re-dated)")
print(f"    Then re-run:  python validate_raw_inputs.py")
print(f"    Then run:     python pipeline.py")
print()
