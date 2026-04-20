"""
validate_pipeline_outputs.py — BIS Pipeline Output Quality Check
=================================================================
Run this AFTER pipeline.py (and after lineup stints + merge) to confirm
that every processed file was rebuilt correctly with full 2025-26 coverage.

Checks per file:
  ✅ File exists
  ✅ Row count is within expected range
  ✅ Key columns are present
  ✅ Max date / season coverage reaches end of season where applicable
  ✅ No all-NaN columns (merge artifacts)
  ✅ Spot-check known anchors (e.g., R32 player count = 403)

Usage:
    python validate_pipeline_outputs.py

Author:  Krystal B Creative — Sports Analytics Portfolio
"""

import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

# ── Config import ─────────────────────────────────────────────────────────────
try:
    from config import (
        PLAYER_BOX_ADVANCED, PLAYER_GAME_LOG,
        PBP_PLAYER_METRICS,
        LINEUP_STINTS_RAW, PLAYER_ONOFF,
        POSTSEASON_STINTS, POSTSEASON_ONOFF,
        PLAYER_SCOUTING_68, PLAYER_SCOUTING_50,
        PLAYER_FEATURE_TABLE,
        SEASON,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        PLAYER_BOX_ADVANCED, PLAYER_GAME_LOG,
        PBP_PLAYER_METRICS,
        LINEUP_STINTS_RAW, PLAYER_ONOFF,
        POSTSEASON_STINTS, POSTSEASON_ONOFF,
        PLAYER_SCOUTING_68, PLAYER_SCOUTING_50,
        PLAYER_FEATURE_TABLE,
        SEASON,
    )


# =============================================================================
# SEASON TARGETS
# =============================================================================

REGULAR_SEASON_END = date(2026, 3, 9)
TOURNEY_END        = date(2026, 4, 6)
R32_PLAYER_COUNT   = 403       # expected rows in player_onoff_metrics (R32 teams)
R32_TEAM_COUNT     = 32        # expected unique teams in R32 on/off output
T68_PLAYER_COUNT   = (600, 1000)   # expected range for tournament68 scouting file


# =============================================================================
# HELPERS
# =============================================================================

passed = 0
failed = 0
warned = 0

def _ok(msg):
    global passed; passed += 1
    print(f"    ✅  {msg}")

def _fail(msg):
    global failed; failed += 1
    print(f"    ❌  {msg}")

def _warn(msg):
    global warned; warned += 1
    print(f"    ⚠️   {msg}")

def _header(label, path, required=True):
    tag = "(required)" if required else "(optional)"
    print(f"\n  {'─'*56}")
    print(f"  {label}  {tag}")
    print(f"  {path.name}")
    print(f"  {'─'*56}")
    if not path.exists():
        if required:
            _fail("File not found — this script did not produce output")
        else:
            _warn("File not found — optional step was skipped")
        return False
    return True

def _check_rows(df, min_rows, max_rows=None, label="rows"):
    n = len(df)
    if n < min_rows:
        _fail(f"Row count: {n:,}  — too low (expected ≥ {min_rows:,})")
    elif max_rows and n > max_rows:
        _warn(f"Row count: {n:,}  — unusually high (expected ≤ {max_rows:,}); verify filters")
    else:
        _ok(f"Row count: {n:,}  ✓")
    return n

def _check_cols(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        _fail(f"Missing expected columns: {missing}")
    else:
        _ok(f"All key columns present")

def _check_nan_cols(df, label=""):
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        _warn(f"All-NaN columns (possible merge failure): {all_nan[:8]}"
              + ("  ..." if len(all_nan) > 8 else ""))
    else:
        _ok(f"No all-NaN columns")

def _check_date_coverage(df, col_candidates, cutoff, label):
    col = next((c for c in col_candidates if c in df.columns), None)
    if col is None:
        _warn(f"No date column found (tried: {col_candidates})")
        return
    try:
        mx = pd.to_datetime(df[col], errors='coerce').dt.date.max()
    except Exception:
        _warn(f"Could not parse dates in '{col}'")
        return
    if mx is None:
        _warn(f"All dates in '{col}' are null")
    elif mx >= cutoff:
        _ok(f"Max {col}: {mx}  (≥ {cutoff} target ✓)")
    else:
        _fail(f"Max {col}: {mx}  — expected ≥ {cutoff}  ← OUTPUT MAY USE STALE DATA")

def _check_unique(df, col, expected_min, label):
    if col not in df.columns:
        _warn(f"Column '{col}' not found — skipping {label} check")
        return
    n = df[col].nunique()
    if n >= expected_min:
        _ok(f"Unique {label}: {n}  (≥ {expected_min} ✓)")
    else:
        _fail(f"Unique {label}: {n}  — expected ≥ {expected_min}  ← possible filter or join issue")

def _check_no_negatives(df, cols):
    for col in cols:
        if col not in df.columns:
            continue
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            _warn(f"'{col}' has {n_neg} negative values — verify metric calculation")


# =============================================================================
# OUTPUT FILE CHECKS
# =============================================================================

print(f"\n{'='*60}")
print(f"  BIS Pipeline — Output Validation")
print(f"  Season: {SEASON}  (2025-26)")
print(f"{'='*60}")


# ── 1. Player Box Advanced Metrics ───────────────────────────────────────────
if _header("1. PLAYER BOX ADVANCED METRICS", PLAYER_BOX_ADVANCED, required=True):
    try:
        df = pd.read_csv(PLAYER_BOX_ADVANCED)
        _check_rows(df, min_rows=1_500, max_rows=8_000)
        _check_unique(df, 'team_id', 300, "D1 teams")
        _check_cols(df, ['athlete_id', 'team_id', 'games_played', 'minutes',
                         'true_shooting_pct', 'points_pg', 'net_rtg_diff'
                         if 'net_rtg_diff' in df.columns else 'points_per40'])
        _check_nan_cols(df)
        _check_no_negatives(df, ['games_played', 'minutes', 'points'])
        # Spot-check: no player with 0 minutes
        if 'minutes' in df.columns:
            zero_min = (df['minutes'] == 0).sum()
            if zero_min > 0:
                _warn(f"{zero_min} players with 0 minutes — check MIN_MINUTES_SEASON filter")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 2. Player Game Log ────────────────────────────────────────────────────────
if _header("2. PLAYER GAME LOG ENRICHED", PLAYER_GAME_LOG, required=True):
    try:
        df = pd.read_csv(PLAYER_GAME_LOG)
        _check_rows(df, min_rows=100_000, max_rows=1_000_000)
        _check_date_coverage(df,
            ['game_date', 'date'],
            REGULAR_SEASON_END,
            "last regular season game")
        _check_cols(df, ['athlete_id', 'team_id', 'game_id', 'game_date'])
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 3. PBP Player Metrics ─────────────────────────────────────────────────────
if _header("3. PBP PLAYER METRICS", PBP_PLAYER_METRICS, required=True):
    try:
        df = pd.read_csv(PBP_PLAYER_METRICS)
        _check_rows(df, min_rows=1_000, max_rows=8_000)
        _check_cols(df, ['athlete_id', 'team_id', 'paint_fga', 'three_share',
                         'assisted_fg_rate'])
        _check_nan_cols(df)
        # paint_share + midrange_share + three_share should sum to ~1.0
        if all(c in df.columns for c in ['paint_share', 'three_share']):
            share_sum = df['paint_share'].fillna(0) + df['three_share'].fillna(0)
            out_of_range = ((share_sum > 1.05) | (share_sum < 0)).sum()
            if out_of_range > 0:
                _warn(f"{out_of_range} players have shot zone shares summing outside [0, 1.05]")
            else:
                _ok(f"Shot zone shares sum correctly (within [0, 1.05])")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 4. Lineup Stints (Regular Season) ────────────────────────────────────────
if _header("4. LINEUP STINTS RAW (regular season)", LINEUP_STINTS_RAW, required=True):
    try:
        df = pd.read_csv(LINEUP_STINTS_RAW)
        _check_rows(df, min_rows=50_000, max_rows=1_000_000)
        _check_cols(df, ['game_id', 'team_id', 'lineup', 'duration_sec',
                         'pts_scored', 'pts_allowed'])
        _check_no_negatives(df, ['duration_sec', 'pts_scored', 'pts_allowed'])
        if 'duration_sec' in df.columns:
            med_dur = df['duration_sec'].median()
            if med_dur < 5 or med_dur > 120:
                _warn(f"Median stint duration: {med_dur:.1f}s — expected 20–60s range")
            else:
                _ok(f"Median stint duration: {med_dur:.1f}s  ✓")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 5. Player On/Off Metrics (Regular Season) ─────────────────────────────────
if _header("5. PLAYER ON/OFF METRICS (regular season)", PLAYER_ONOFF, required=True):
    try:
        df = pd.read_csv(PLAYER_ONOFF)
        n = _check_rows(df, min_rows=380, max_rows=450)
        # Key anchor: R32 teams = 32, R32 players ~= 403
        if abs(n - R32_PLAYER_COUNT) <= 20:
            _ok(f"Player count {n} is close to expected R32 anchor ({R32_PLAYER_COUNT}) ✓")
        else:
            _warn(f"Player count {n} differs from expected ~{R32_PLAYER_COUNT} — "
                  f"verify team filter in lineup_stints_pipeline")
        _check_unique(df, 'team_id', R32_TEAM_COUNT, "R32 teams")
        _check_cols(df, ['athlete_id', 'team_id', 'on_net_rtg', 'off_net_rtg',
                         'net_rtg_diff', 'on_minutes', 'poss_on'])
        _check_nan_cols(df)
        # Sanity: net_rtg_diff distribution should be roughly centered on 0
        if 'net_rtg_diff' in df.columns:
            mean_diff = df['net_rtg_diff'].mean()
            if abs(mean_diff) > 10:
                _warn(f"Mean net_rtg_diff: {mean_diff:.2f} — unusually far from 0; check calculation")
            else:
                _ok(f"Mean net_rtg_diff: {mean_diff:.2f}  (expected ~0 across the roster ✓)")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 6. Postseason Stints ──────────────────────────────────────────────────────
if _header("6. POSTSEASON STINTS RAW", POSTSEASON_STINTS, required=False):
    try:
        df = pd.read_csv(POSTSEASON_STINTS)
        _check_rows(df, min_rows=2_000, max_rows=100_000)
        _check_cols(df, ['game_id', 'team_id', 'lineup', 'duration_sec'])
        # Should have 32–68 unique teams (first round through championship)
        _check_unique(df, 'team_id', 32, "tournament teams")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 7. Postseason On/Off ──────────────────────────────────────────────────────
if _header("7. POSTSEASON ON/OFF METRICS", POSTSEASON_ONOFF, required=False):
    try:
        df = pd.read_csv(POSTSEASON_ONOFF)
        _check_rows(df, min_rows=300, max_rows=1_500)
        _check_cols(df, ['athlete_id', 'tourney_net_rtg', 'tourney_net_rtg_diff',
                         'tourney_games', 'tourney_poss_on'])
        _check_nan_cols(df)
        # All tourney_games should be ≥ 1 and ≤ 6
        if 'tourney_games' in df.columns:
            out = ((df['tourney_games'] < 1) | (df['tourney_games'] > 6)).sum()
            if out > 0:
                _warn(f"{out} players with tourney_games outside [1, 6] — verify game filter")
            else:
                _ok(f"tourney_games values all within [1, 6]  ✓")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 8. Scouting — Tournament 68 ───────────────────────────────────────────────
if _header("8. PLAYER SCOUTING — TOURNAMENT 68", PLAYER_SCOUTING_68, required=False):
    try:
        df = pd.read_csv(PLAYER_SCOUTING_68)
        _check_rows(df, min_rows=T68_PLAYER_COUNT[0], max_rows=T68_PLAYER_COUNT[1])
        _check_unique(df, 'team_id', 60, "tournament teams")
        _check_cols(df, ['athlete_id', 'team_id', 'seed', 'region'])
        _check_nan_cols(df)
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 9. Scouting — Top 50 ─────────────────────────────────────────────────────
if _header("9. PLAYER SCOUTING — TOP 50", PLAYER_SCOUTING_50, required=False):
    try:
        df = pd.read_csv(PLAYER_SCOUTING_50)
        _check_rows(df, min_rows=40, max_rows=60)
        _check_cols(df, ['athlete_id', 'team_id'])
    except Exception as e:
        _fail(f"Could not read file: {e}")


# ── 10. Player Feature Table (ML-ready) ───────────────────────────────────────
if _header("10. PLAYER FEATURE TABLE (ML-ready)", PLAYER_FEATURE_TABLE, required=True):
    try:
        df = pd.read_csv(PLAYER_FEATURE_TABLE)
        n = _check_rows(df, min_rows=380, max_rows=450)
        if abs(n - R32_PLAYER_COUNT) <= 20:
            _ok(f"Player count {n} matches R32 anchor ({R32_PLAYER_COUNT}) ✓")
        else:
            _warn(f"Player count {n} — expected ~{R32_PLAYER_COUNT} for R32 teams")
        # Column count sanity: should be wide (50+ cols after all merges)
        if len(df.columns) >= 50:
            _ok(f"Column count: {len(df.columns)}  (wide table confirmed ✓)")
        else:
            _warn(f"Column count: {len(df.columns)}  — may be missing merge layers (expected 50+)")
        # Check key column groups are present
        _check_cols(df, ['athlete_id', 'seed', 'region', 'on_net_rtg',
                         'net_rtg_diff', 'true_shooting_pct', 'weighted_production',
                         'two_way_flag'])
        _check_nan_cols(df)
        # Recruiting join: how many players matched?
        if 'recruit_rank' in df.columns:
            matched   = df['recruit_rank'].notna().sum()
            pct = matched / len(df) * 100
            if pct >= 40:
                _ok(f"Recruiting join matched: {matched} / {len(df)} players  ({pct:.0f}%)")
            else:
                _warn(f"Recruiting join matched only {matched} / {len(df)} ({pct:.0f}%) — "
                      f"check name normalisation in merge_feature_table.py")
        # Tourney columns: confirm postseason data flowed in
        tourney_cols = [c for c in df.columns if c.startswith('tourney_')]
        if tourney_cols:
            filled = df[tourney_cols[0]].notna().sum()
            _ok(f"Postseason on/off joined: {filled} players have tourney_ data")
        else:
            _warn("No tourney_ columns found — postseason_onoff_metrics.csv may not have been merged")
        # Cross-table derived features
        for feat in ['net_rtg_reg_to_tourney', 'weighted_production', 'two_way_flag', 'shot_diet']:
            if feat in df.columns:
                n_valid = df[feat].notna().sum()
                _ok(f"'{feat}' computed for {n_valid} players")
            else:
                _warn(f"'{feat}' not found — cross-table derived step may have failed")
    except Exception as e:
        _fail(f"Could not read file: {e}")


# =============================================================================
# SUMMARY
# =============================================================================

total = passed + failed + warned
print(f"\n{'='*60}")
print(f"  OUTPUT VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  ✅  Passed : {passed}")
print(f"  ❌  Failed : {failed}")
print(f"  ⚠️   Warned : {warned}")
print(f"  {'─'*40}")

if failed == 0 and warned == 0:
    print(f"  🟢  All output checks passed — data/processed/ is ready")
    print(f"      Safe to commit and begin analysis/ work")
elif failed == 0:
    print(f"  🟡  No failures — review warnings before committing")
    print(f"      Minor issues won't block analysis but should be investigated")
else:
    print(f"  🔴  {failed} check(s) failed — do not commit data/processed/ yet")
    print(f"\n  Common fixes:")
    print(f"    • 'OUTPUT MAY USE STALE DATA' → raw file wasn't fully updated; re-run validate_raw_inputs.py")
    print(f"    • 'too low' row count → check script filter (MIN_MINUTES_SEASON, season_type, team filter)")
    print(f"    • 'missing merge layers' on feature table → re-run scripts 03–06 in order")
    print(f"    • 'All-NaN columns' → join key mismatch; check athlete_id coercion in merge_feature_table.py")
    print(f"\n  Rebuild order if outputs are wrong:")
    print(f"    python scripts/01_player_box_processing.py")
    print(f"    python scripts/02_pbp_player_processing.py")
    print(f"    python scripts/03_lineup_stints_regular.py")
    print(f"    python scripts/04_lineup_stints_postseason.py")
    print(f"    python scripts/05_merge_scouting.py")
    print(f"    python scripts/06_merge_feature_table.py")

print()
