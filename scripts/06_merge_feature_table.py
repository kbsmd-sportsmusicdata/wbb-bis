"""
Script 6 — Merge Feature Table
================================
Builds player_feature_table_2026.csv — the single wide, ML-ready table that
combines every player-level metric computed across the pipeline into one row
per player.

This is the primary input for archetype modeling, similarity scoring, and
role-fit analysis in the BIS Decision Layer (Step 4).

Merge strategy (left-join chain):
  Base   : player_onoff_metrics.csv          — 403 players (R32 teams, regular season)
       +   player_box_advanced_metrics.csv   — season box stats & efficiency metrics
       +   pbp_player_metrics.csv            — shot zones, creation, clutch, hustle
       +   postseason_onoff_metrics.csv      — tournament on/off (tourney_ prefix cols)
       +   wbb_rosters_2025_26.csv           — identity: position, height, class, hometown
       +   player_recruit_rankings.csv       — ESPN recruit rank + grade (name-based join)
       +   tournament_bracket.csv            — seed, region, bracket context

Join key: athlete_id throughout, except recruiting (name-based, no ESPN ID in file).
Roster join uses athlete_id where available; falls back to name-matching with
a warning (some wehoop rosters have ID mismatches).

Output: player_feature_table_2026.csv  (one row per player)

Author:  Krystal B Creative — Sports Analytics Portfolio
Date:    2026-04-10
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Config import ─────────────────────────────────────────────────────────────
try:
    from config import (
        PLAYER_ONOFF, PLAYER_BOX_ADVANCED, PBP_PLAYER_METRICS,
        POSTSEASON_ONOFF, ROSTER_FILE, RECRUIT_RANKINGS, TOURNAMENT_BRACKET,
        PLAYER_FEATURE_TABLE, PROCESSED_DIR, RECRUITING_DIR, SEASON, validate_inputs,
    )
except ImportError:
    # Allow running from scripts/ subdirectory
    sys.path.insert(0, str(REPO_ROOT))
    from config import (
        PLAYER_ONOFF, PLAYER_BOX_ADVANCED, PBP_PLAYER_METRICS,
        POSTSEASON_ONOFF, ROSTER_FILE, RECRUIT_RANKINGS, TOURNAMENT_BRACKET,
        PLAYER_FEATURE_TABLE, PROCESSED_DIR, RECRUITING_DIR, SEASON, validate_inputs,
    )

TEAM_SEASON_BASE = PROCESSED_DIR / f"team_season_analytic_{SEASON}.csv"
TEAM_SEASON_ENRICHED = PROCESSED_DIR / f"team_season_analytic_{SEASON}_top25_enriched.csv"
TEAM_STYLE_HISTORY = REPO_ROOT / "analysis" / "conference_efficiency" / "team_style_efficiency_2021_2026.csv"
RECRUIT_DRAFT_FILE = RECRUITING_DIR / "player_recruit_to_draft_analysis.csv"


# =============================================================================
# HELPER FUNCTIONS  (defined here — used in merge steps below)
# =============================================================================

def _drop_dupes(df):
    """
    Drop columns with _x/_y or other merge-artifact suffixes in place.
    Keeps the left-table version (_x) and drops right-table duplicates (_y).
    Also drops suffixed copies from named merge suffixes (_box, _pbp, etc.)
    only when the unsuffixed base column already exists.
    """
    drop_cols = []
    for col in df.columns:
        if col.endswith('_y'):
            drop_cols.append(col)
        elif col.endswith(('_box', '_pbp', '_post', '_roster', '_rec', '_brkt')):
            base = col.rsplit('_', 1)[0]
            if base in df.columns:
                drop_cols.append(col)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    # Rename any _x survivors to their base name
    rename_map = {c: c[:-2] for c in df.columns if c.endswith('_x')}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)


def to_int_id(series):
    """Coerce an ID column to nullable Int64, dropping non-numeric values."""
    return pd.to_numeric(series, errors='coerce').astype('Int64')


# =============================================================================
# STEP 0 — VALIDATE INPUTS
# =============================================================================

if not validate_inputs(
    required=[PLAYER_ONOFF, PLAYER_BOX_ADVANCED, TOURNAMENT_BRACKET],
    optional=[PBP_PLAYER_METRICS, POSTSEASON_ONOFF, ROSTER_FILE, RECRUIT_RANKINGS, TEAM_SEASON_BASE, TEAM_SEASON_ENRICHED, TEAM_STYLE_HISTORY, RECRUIT_DRAFT_FILE],
):
    sys.exit(1)


# =============================================================================
# STEP 1 — LOAD ALL SOURCES
# =============================================================================

print("Loading pipeline outputs...")

# Base: on/off metrics (403 players, R32 teams) — this defines the player set
onoff = pd.read_csv(PLAYER_ONOFF)
print(f"  player_onoff_metrics:          {len(onoff):>5,} rows  |  {len(onoff.columns)} cols")

# Season box stats + derived metrics
box = pd.read_csv(PLAYER_BOX_ADVANCED)
print(f"  player_box_advanced_metrics:   {len(box):>5,} rows  |  {len(box.columns)} cols")

# PBP-derived metrics (optional)
pbp = None
if PBP_PLAYER_METRICS.exists():
    pbp = pd.read_csv(PBP_PLAYER_METRICS)
    print(f"  pbp_player_metrics:            {len(pbp):>5,} rows  |  {len(pbp.columns)} cols")
else:
    print(f"  ⚠️  pbp_player_metrics.csv not found — shot zone / clutch cols will be missing")

# Postseason on/off (optional — may not be fully rebuilt yet)
post = None
if POSTSEASON_ONOFF.exists():
    post = pd.read_csv(POSTSEASON_ONOFF)
    print(f"  postseason_onoff_metrics:      {len(post):>5,} rows  |  {len(post.columns)} cols")
else:
    print(f"  ⚠️  postseason_onoff_metrics.csv not found — tourney_ cols will be missing")

# Rosters (optional — for identity enrichment)
roster = None
if ROSTER_FILE.exists():
    roster = pd.read_csv(ROSTER_FILE, low_memory=False)
    print(f"  wbb_rosters_2025_26:           {len(roster):>5,} rows  |  {len(roster.columns)} cols")
else:
    print(f"  ⚠️  wbb_rosters_2025_26.csv not found — position/height/class cols will be missing")

# Recruiting ranks (optional — name-based join, no ESPN ID in file)
recruit = None
if RECRUIT_RANKINGS.exists():
    recruit = pd.read_csv(RECRUIT_RANKINGS)
    print(f"  player_recruit_rankings:       {len(recruit):>5,} rows  |  {len(recruit.columns)} cols")
else:
    print(f"  ⚠️  player_recruit_rankings.csv not found — recruit_rank cols will be missing")

# Tournament bracket (seed, region)
bracket = pd.read_csv(TOURNAMENT_BRACKET)
print(f"  tournament_bracket:            {len(bracket):>5,} rows  |  {len(bracket.columns)} cols")

# Team context (optional — conference + efficiency style context)
team_base = None
if TEAM_SEASON_BASE.exists():
    team_base = pd.read_csv(TEAM_SEASON_BASE)
    print(f"  {TEAM_SEASON_BASE.name:<30} {len(team_base):>5,} rows  |  {len(team_base.columns)} cols")
else:
    print(f"  ⚠️  {TEAM_SEASON_BASE.name} not found — full conference context will be missing")

team_enriched = None
if TEAM_SEASON_ENRICHED.exists():
    team_enriched = pd.read_csv(TEAM_SEASON_ENRICHED)
    print(f"  {TEAM_SEASON_ENRICHED.name:<30} {len(team_enriched):>5,} rows  |  {len(team_enriched.columns)} cols")
else:
    print(f"  ⚠️  {TEAM_SEASON_ENRICHED.name} not found — offensive/defensive efficiency + pace may be missing")

team_style_hist = None
if TEAM_STYLE_HISTORY.exists():
    team_style_hist = pd.read_csv(TEAM_STYLE_HISTORY, low_memory=False)
    print(f"  {TEAM_STYLE_HISTORY.name:<30} {len(team_style_hist):>5,} rows  |  {len(team_style_hist.columns)} cols")
else:
    print(f"  ⚠️  {TEAM_STYLE_HISTORY.name} not found — fallback team style metrics unavailable")

# Recruit-to-draft context (optional)
recruit_draft = None
if RECRUIT_DRAFT_FILE.exists():
    recruit_draft = pd.read_csv(RECRUIT_DRAFT_FILE)
    print(f"  {RECRUIT_DRAFT_FILE.name:<30} {len(recruit_draft):>5,} rows  |  {len(recruit_draft.columns)} cols")
else:
    print(f"  ⚠️  {RECRUIT_DRAFT_FILE.name} not found — draft-proxy context will be missing")

print()


# =============================================================================
# STEP 2 — NORMALISE IDs
# =============================================================================

onoff['athlete_id'] = to_int_id(onoff['athlete_id'])
box['athlete_id']   = to_int_id(box['athlete_id'])

if pbp is not None:
    pbp['athlete_id'] = to_int_id(pbp['athlete_id'])
if post is not None:
    post['athlete_id'] = to_int_id(post['athlete_id'])

# Normalise roster athlete_id if present
roster_id_col = None
if roster is not None:
    id_candidates = ['athlete_id', 'espn_id', 'player_id', 'id']
    roster_id_col = next((c for c in id_candidates if c in roster.columns), None)
    if roster_id_col:
        roster['athlete_id'] = to_int_id(roster[roster_id_col])
    else:
        print("  ⚠️  No numeric ID column found in roster — will attempt name-based join")


# =============================================================================
# STEP 3 — MERGE BOX STATS
# =============================================================================

print("Merging box advanced metrics...")

onoff_cols   = set(onoff.columns)
box_new_cols = [c for c in box.columns if c not in onoff_cols or c == 'athlete_id']

feature = onoff.merge(
    box[box_new_cols],
    on='athlete_id',
    how='left',
    suffixes=('', '_box'),
)
_drop_dupes(feature)
print(f"  After box merge:  {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 4 — MERGE PBP METRICS
# =============================================================================

if pbp is not None:
    print("Merging PBP player metrics...")
    pbp_new_cols = [c for c in pbp.columns
                    if c not in set(feature.columns) or c == 'athlete_id']
    feature = feature.merge(
        pbp[pbp_new_cols],
        on='athlete_id',
        how='left',
        suffixes=('', '_pbp'),
    )
    _drop_dupes(feature)
    print(f"  After PBP merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 5 — MERGE POSTSEASON ON/OFF
# =============================================================================

if post is not None:
    print("Merging postseason on/off metrics...")
    # postseason cols are already prefixed tourney_ — minimal collision risk
    post_new_cols = [c for c in post.columns
                     if c not in set(feature.columns) or c in ('athlete_id', 'team_id')]
    feature = feature.merge(
        post[post_new_cols],
        on='athlete_id',
        how='left',
        suffixes=('', '_post'),
    )
    _drop_dupes(feature)
    print(f"  After postseason merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 6 — MERGE ROSTER IDENTITY
# =============================================================================
# Adds position, height, class year, hometown, transfer status.
# Only pulls columns that add new information (identity fields not in box data).

if roster is not None:
    print("Merging roster identity fields...")

    ROSTER_KEEP = [
        'athlete_id',
        # Identity
        'athlete_display_name', 'athlete_short_name',
        'athlete_jersey', 'athlete_headshot',
        # Position (may differ from box data position field)
        'athlete_position_name', 'athlete_position_abbreviation',
        # Physical
        'athlete_height', 'athlete_weight',
        # Class
        'athlete_year', 'athlete_eligibility_year',
        # Status
        'transfer_portal', 'is_transfer', 'prior_school',
        # Geography
        'athlete_hometown', 'athlete_home_state', 'athlete_home_country',
        'athlete_high_school',
    ]

    roster_cols_available = [c for c in ROSTER_KEEP if c in roster.columns]
    missing_roster_cols   = [c for c in ROSTER_KEEP if c not in roster.columns]
    if missing_roster_cols:
        print(f"  ⚠️  Roster columns not found (skipped): {missing_roster_cols}")

    if roster_id_col and 'athlete_id' in roster_cols_available:
        # Clean join on athlete_id
        roster_sub = roster[roster_cols_available].drop_duplicates('athlete_id')
        roster_new_cols = [c for c in roster_cols_available
                           if c not in set(feature.columns) or c == 'athlete_id']
        feature = feature.merge(
            roster_sub[roster_new_cols],
            on='athlete_id',
            how='left',
            suffixes=('', '_roster'),
        )
        _drop_dupes(feature)
        matched = feature['athlete_display_name'].notna().sum() \
            if 'athlete_display_name' in feature.columns else '?'
        print(f"  Roster matched: {matched} of {len(feature)} players")

    else:
        # Fallback: name-based join (less reliable — warns on mismatches)
        name_col_box  = next((c for c in feature.columns
                              if 'name' in c.lower() and 'display' in c.lower()), None)
        name_col_rost = next((c for c in roster.columns
                              if 'name' in c.lower() and 'display' in c.lower()), None)
        if name_col_box and name_col_rost:
            print(f"  ⚠️  Falling back to name join: '{name_col_box}' ↔ '{name_col_rost}'")
            feature[name_col_box]  = feature[name_col_box].str.strip().str.lower()
            roster[name_col_rost]  = roster[name_col_rost].str.strip().str.lower()
            roster_sub = roster.rename(columns={name_col_rost: name_col_box})
            roster_new_cols = [c for c in roster_cols_available
                               if c not in set(feature.columns) or c == name_col_box]
            feature = feature.merge(
                roster_sub[roster_new_cols + [name_col_box]].drop_duplicates(name_col_box),
                on=name_col_box,
                how='left',
                suffixes=('', '_roster'),
            )
            _drop_dupes(feature)
        else:
            print("  ⚠️  Could not find name columns for fallback join — roster fields skipped")

    print(f"  After roster merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 6.5 — MERGE RECRUITING RANKS
# =============================================================================
# The recruiting file (player_recruit_rankings_20212026.csv) has no ESPN athlete_id,
# so this join is name-based. We normalise to lowercase-stripped display names and
# deduplicate by keeping the player's most recent recruiting class record.
#
# Expected match rate: ~50–70% of R32 players (recruits outside the ranking window,
# walk-ons, and transfers from outside the D1 system will be unmatched → NaN → 0).

if recruit is not None:
    print("Merging recruiting rank data (name-based join)...")

    # Identify the name column in the feature table
    name_col = next(
        (c for c in ['athlete_display_name', 'player_name'] if c in feature.columns), None
    )

    if name_col is None:
        print("  ⚠️  No display name column found in feature table — recruiting join skipped")
    else:
        # Build normalised join key
        recruit['_name_key'] = recruit['PLAYER_NAME'].str.strip().str.lower()
        feature['_name_key'] = feature[name_col].str.strip().str.lower()

        # Deduplicate: if a player appears in multiple recruiting classes (e.g., reclassified),
        # keep the most recent record. Tie-break: best (lowest) rank.
        recruit_dedup = (
            recruit
            .sort_values(['RECRUITING_YEAR', 'ESPNW_PLAYER_RECRUIT_RANK'],
                         ascending=[False, True])
            .drop_duplicates('_name_key', keep='first')
        )

        RECRUIT_KEEP = [
            '_name_key',
            'ESPNW_PLAYER_RECRUIT_RANK',
            'ESPN_GRADE',
            'RECRUITING_YEAR',
        ]
        # Only keep columns that are actually present (guard against schema changes)
        recruit_keep_available = [c for c in RECRUIT_KEEP if c in recruit_dedup.columns]
        recruit_sub = recruit_dedup[recruit_keep_available].rename(columns={
            'ESPNW_PLAYER_RECRUIT_RANK': 'recruit_rank',
            'ESPN_GRADE':                'recruit_grade',
            'RECRUITING_YEAR':           'recruit_class_year',
        })

        feature = feature.merge(recruit_sub, on='_name_key', how='left', suffixes=('', '_rec'))
        feature.drop(columns=['_name_key'], inplace=True)
        _drop_dupes(feature)

        matched  = feature['recruit_rank'].notna().sum()
        unranked = len(feature) - matched
        print(f"  Recruiting matched:  {matched} of {len(feature)} players  "
              f"({matched / len(feature) * 100:.0f}%)")
        if unranked:
            print(f"  Unmatched ({unranked} players) → recruit_rank = NaN  "
                  f"(treated as unranked in derived features)")

    print(f"  After recruiting merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 7 — MERGE BRACKET CONTEXT (seed, region)
# =============================================================================

print("Merging tournament bracket context...")

bracket_id_col = next(
    (c for c in ['team_id', 'TeamID', 'espn_team_id'] if c in bracket.columns), None
)
bracket_name_col = next(
    (c for c in ['team_location', 'team_name', 'TeamName'] if c in bracket.columns), None
)

BRACKET_KEEP = ['seed', 'region', 'bracket_display']
bracket_keep_available = [c for c in BRACKET_KEEP if c in bracket.columns]

if bracket_id_col and 'team_id' in feature.columns:
    bracket_sub = bracket[[bracket_id_col] + bracket_keep_available].rename(
        columns={bracket_id_col: 'team_id'}
    )
    bracket_sub['team_id'] = to_int_id(bracket_sub['team_id'])
    feature = feature.merge(
        bracket_sub.drop_duplicates('team_id'),
        on='team_id',
        how='left',
        suffixes=('', '_brkt'),
    )
elif bracket_name_col and 'team_name' in feature.columns:
    bracket_sub = bracket[[bracket_name_col] + bracket_keep_available].rename(
        columns={bracket_name_col: 'team_name'}
    )
    feature = feature.merge(
        bracket_sub.drop_duplicates('team_name'),
        on='team_name',
        how='left',
        suffixes=('', '_brkt'),
    )
else:
    print("  ⚠️  Could not join bracket — no matching team ID or name column found")

_drop_dupes(feature)
print(f"  After bracket merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 7B — MERGE TEAM CONTEXT (conference + style)
# =============================================================================

if team_base is not None or team_enriched is not None:
    print("Merging team context...")
    if 'team_id' in feature.columns:
        feature['team_id'] = to_int_id(feature['team_id'])

    if team_base is not None and 'team_id' in team_base.columns and 'team_id' in feature.columns:
        team_base['team_id'] = to_int_id(team_base['team_id'])
        base_keep = [c for c in ['team_id', 'conference', 'division', 'team_state'] if c in team_base.columns]
        if base_keep:
            feature = feature.merge(
                team_base[base_keep].drop_duplicates('team_id'),
                on='team_id',
                how='left',
                suffixes=('', '_teambase')
            )
            _drop_dupes(feature)

    if team_enriched is not None and 'team_id' in team_enriched.columns and 'team_id' in feature.columns:
        team_enriched['team_id'] = to_int_id(team_enriched['team_id'])
        enrich_keep = [
            c for c in [
                'team_id',
                'offensive_eff', 'defensive_eff', 'net_eff', 'pace',
                'weeks_in_top25', 'top25_win_pct', 'best_rank'
            ] if c in team_enriched.columns
        ]
        if enrich_keep:
            feature = feature.merge(
                team_enriched[enrich_keep].drop_duplicates('team_id'),
                on='team_id',
                how='left',
                suffixes=('', '_teamenr')
            )
            _drop_dupes(feature)

    # Fallback: fill missing style metrics from full team style history (raw-team-box derived).
    if team_style_hist is not None and 'team_id' in team_style_hist.columns and 'team_id' in feature.columns:
        team_style_hist['team_id'] = to_int_id(team_style_hist['team_id'])
        if 'season' in team_style_hist.columns:
            team_style_hist['season'] = pd.to_numeric(team_style_hist['season'], errors='coerce')
            team_style_hist = team_style_hist[team_style_hist['season'] == SEASON].copy()
        style_keep = [c for c in ['team_id', 'offensive_eff', 'defensive_eff', 'net_eff', 'pace'] if c in team_style_hist.columns]
        if style_keep:
            feature = feature.merge(
                team_style_hist[style_keep].drop_duplicates('team_id').rename(
                    columns={c: f"{c}_stylefb" for c in style_keep if c != 'team_id'}
                ),
                on='team_id',
                how='left',
            )
            for c in ['offensive_eff', 'defensive_eff', 'net_eff', 'pace']:
                fb = f"{c}_stylefb"
                if c in feature.columns and fb in feature.columns:
                    feature[c] = feature[c].fillna(feature[fb])
            drop_fb = [c for c in feature.columns if c.endswith('_stylefb')]
            if drop_fb:
                feature.drop(columns=drop_fb, inplace=True)
            _drop_dupes(feature)

    print(f"  After team context merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 7C — MERGE RECRUIT-TO-DRAFT CONTEXT
# =============================================================================

if recruit_draft is not None and 'athlete_id' in recruit_draft.columns:
    print("Merging recruit-to-draft context...")
    recruit_draft['athlete_id'] = to_int_id(recruit_draft['athlete_id'])
    draft_keep = [
        c for c in [
            'athlete_id', 'in_draft_2026',
            'draft_prob_early', 'draft_prob_mid', 'draft_prob_late',
            'impact_delta', 'recruit_rank'
        ] if c in recruit_draft.columns
    ]
    if draft_keep:
        feature = feature.merge(
            recruit_draft[draft_keep].drop_duplicates('athlete_id'),
            on='athlete_id',
            how='left',
            suffixes=('', '_draft')
        )
        _drop_dupes(feature)
    print(f"  After draft context merge: {len(feature):,} rows  |  {len(feature.columns)} cols")


# =============================================================================
# STEP 8 — COMPUTE CROSS-TABLE FEATURES
# =============================================================================
# Derived columns that require data from multiple sources (computed after all merges).

print("\nComputing cross-table features...")

# Regular season → tournament performance delta
if 'on_net_rtg' in feature.columns and 'tourney_net_rtg' in feature.columns:
    feature['net_rtg_reg_to_tourney'] = (
        feature['tourney_net_rtg'] - feature['on_net_rtg']
    ).round(2)
    print("  ✓ net_rtg_reg_to_tourney (regular → tournament delta)")

# Scoring efficiency composite: TS% × per-40 points (weighted production score)
if 'ts_pct' in feature.columns and 'true_shooting_pct' not in feature.columns:
    feature['true_shooting_pct'] = feature['ts_pct']
if 'pts_per40' in feature.columns and 'points_per40' not in feature.columns:
    feature['points_per40'] = feature['pts_per40']

ts_col = next((c for c in ['true_shooting_pct', 'ts_pct'] if c in feature.columns), None)
pts_col = next((c for c in ['points_per40', 'pts_per40'] if c in feature.columns), None)
if ts_col and pts_col:
    feature['weighted_production'] = (
        feature[ts_col].fillna(0) * feature[pts_col].fillna(0)
    ).round(3)
    print(f"  ✓ weighted_production ({ts_col} × {pts_col})")

# Two-way impact flag: positive net_rtg_diff AND positive on_net_rtg
if 'net_rtg_diff' in feature.columns and 'on_net_rtg' in feature.columns:
    feature['two_way_flag'] = (
        (feature['net_rtg_diff'] > 2) & (feature['on_net_rtg'] > 0)
    ).astype(int)
    print("  ✓ two_way_flag (net_rtg_diff > 2 AND on_net_rtg > 0)")

# Shot diet: three-heavy vs paint-heavy (classification for archetype modeling)
if 'paint_share' in feature.columns and 'three_share' in feature.columns:
    conditions = [
        feature['paint_share'] >= 0.45,
        feature['three_share'] >= 0.40,
        (feature['three_share'] >= 0.30) & (feature['paint_share'] < 0.40),
    ]
    choices = ['Paint-Dominant', 'Three-Dominant', 'Three-Leaning']
    feature['shot_diet'] = np.select(conditions, choices, default='Balanced')
    print("  ✓ shot_diet (Paint-Dominant / Three-Dominant / Three-Leaning / Balanced)")

# Recruiting tier flags
# NaN recruit_rank → unranked (outside top-600 national class or not in database)
# 0 is not a valid rank, so 0 will not trip the tier flags
if 'recruit_rank' in feature.columns:
    rr = feature['recruit_rank']
    feature['is_top25_recruit']  = (rr.notna() & (rr <= 25)).astype(int)
    feature['is_top100_recruit'] = (rr.notna() & (rr <= 100)).astype(int)
    feature['is_ranked_recruit'] = rr.notna().astype(int)   # 1 = any ESPN ranking on record
    print("  ✓ is_top25_recruit, is_top100_recruit, is_ranked_recruit")

# Season label
feature['season'] = SEASON


# =============================================================================
# STEP 9 — COLUMN ORDERING & CLEANUP
# =============================================================================

print("\nOrdering and cleaning columns...")

PRIORITY_COLS = [
    # Identity
    'athlete_id', 'athlete_display_name', 'athlete_short_name',
    'team_id', 'team_name', 'team_location',
    'athlete_position_abbreviation', 'athlete_position_name',
    'athlete_jersey', 'athlete_headshot',
    'athlete_height', 'athlete_weight',
    'athlete_year', 'athlete_eligibility_year',
    'is_transfer', 'transfer_portal', 'prior_school',
    'athlete_hometown', 'athlete_home_state',
    'athlete_high_school',
    # Recruiting context
    'recruit_rank', 'recruit_grade', 'recruit_class_year',
    'is_top25_recruit', 'is_top100_recruit', 'is_ranked_recruit',
    # Tournament context
    'seed', 'region', 'bracket_display',
    # Team style context
    'conference', 'offensive_eff', 'defensive_eff', 'net_eff', 'pace',
    # Draft proxy context
    'in_draft_2026', 'draft_prob_early', 'draft_prob_mid', 'draft_prob_late', 'impact_delta',
    'season',
]

existing_priority = [c for c in PRIORITY_COLS if c in feature.columns]
remaining_cols    = [c for c in feature.columns if c not in PRIORITY_COLS]
feature = feature[existing_priority + remaining_cols]

# Round all float columns to 3 decimal places for readability
float_cols = feature.select_dtypes(include='float').columns
feature[float_cols] = feature[float_cols].round(3)

# Report any remaining duplicate-like columns
dup_suffixes = [c for c in feature.columns
                if c.endswith(('_x', '_y', '_box', '_pbp', '_post', '_roster', '_rec', '_brkt'))]
if dup_suffixes:
    print(f"  ⚠️  Residual duplicate columns detected — review and drop manually: {dup_suffixes}")


# =============================================================================
# STEP 10 — EXPORT
# =============================================================================

feature.to_csv(PLAYER_FEATURE_TABLE, index=False)

print(f"\n{'='*60}")
print(f"  FEATURE TABLE COMPLETE")
print(f"{'='*60}")
print(f"  ✅  {PLAYER_FEATURE_TABLE.name}")
print(f"      {len(feature):,} players  ×  {len(feature.columns)} columns")
print(f"\n  Column groups:")
print(f"    Identity / roster fields : {len([c for c in feature.columns if any(k in c for k in ['athlete', 'roster', 'school', 'hometown', 'high_school'])])}")
print(f"    Recruiting context       : {len([c for c in feature.columns if any(k in c for k in ['recruit_', 'is_top', 'is_ranked'])])}")
print(f"    Box advanced metrics     : {len([c for c in feature.columns if any(k in c for k in ['ppg', 'per40', '_pct', 'usage', 'ts_', 'game_score'])])}")
print(f"    PBP-derived metrics      : {len([c for c in feature.columns if any(k in c for k in ['paint_', 'three_', 'clutch', 'assisted', 'creation'])])}")
print(f"    On/off (regular season)  : {len([c for c in feature.columns if any(k in c for k in ['on_net', 'off_net', 'net_rtg', 'net_rtg_diff'])])}")
print(f"    Postseason on/off        : {len([c for c in feature.columns if c.startswith('tourney_')])}")
print(f"    Tournament context       : {len([c for c in feature.columns if c in ['seed', 'region', 'bracket_display']])}")
print(f"    Cross-table derived      : {len([c for c in feature.columns if c in ['net_rtg_reg_to_tourney', 'weighted_production', 'two_way_flag', 'shot_diet']])}")

print(f"\n  Sample (first 5 rows, key columns):")
id_cols = [c for c in ['athlete_display_name', 'team_name', 'seed', 'recruit_rank',
                        'on_net_rtg', 'net_rtg_diff', 'tourney_net_rtg'] if c in feature.columns]
print(feature[id_cols].head().to_string(index=False))
print()
