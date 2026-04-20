"""
Postseason Lineup Stints & On/Off Pipeline
==========================================
NCAA WBB 2026 — NCAA Tournament (First Round through Championship)
Tiers 1-4: Core On/Off, Lineup-Level, Shot Quality, Contextual Splits

Derived from lineup_stints_pipeline.py. Key differences vs. regular season version:
  • Input:  Postseason Data/march_pbp_2026.parquet  (tournament PBP only)
  • No season_type filter — all rows in the postseason parquet are tournament games
  • Focal teams loaded dynamically from tournament_bracket.csv  (no hardcoded IDs)
  • Output: postseason_stints_raw.csv   (stint-level)
            postseason_onoff_metrics.csv (player-level)

All file paths come from config.py — no hardcoded machine or session paths.

Input files required:
  ✅ config.POSTSEASON_PBP_FILE      (march_pbp_2026.parquet)
  ✅ config.NET_FILE                 (net_rankings_*.csv)
  ✅ config.TOURNAMENT_BRACKET       (tournament_bracket.csv)

Author:  Krystal B Creative — Sports Analytics Portfolio
Date:    2026-04-10
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Config import ─────────────────────────────────────────────────────────────
# All paths and shared constants live in config.py; nothing hardcoded here.
try:
    from config import (
        POSTSEASON_PBP_FILE, NET_FILE, TOURNAMENT_BRACKET,
        POSTSEASON_STINTS, POSTSEASON_ONOFF,
        CONSTANTS, validate_inputs,
    )
except ImportError:
    print("ERROR: config.py not found. Run this script from the repo root,")
    print("       or add the repo root to PYTHONPATH.")
    sys.exit(1)

# ── Pull shared constants ─────────────────────────────────────────────────────
MIN_STINT_PLAYERS = CONSTANTS["MIN_STINT_PLAYERS"]   # 4
MIN_POSSESSIONS   = CONSTANTS["MIN_POSSESSIONS"]     # 5
NET_Q1_MAX        = CONSTANTS["NET_Q1_MAX"]          # 30
NET_Q2_MAX        = CONSTANTS["NET_Q2_MAX"]          # 75
NET_Q3_MAX        = CONSTANTS["NET_Q3_MAX"]          # 160

# ── Columns to load from parquet (saves ~60% memory) ─────────────────────────
LOAD_COLS = [
    'game_id', 'sequence_number', 'game_play_number',
    'type_text', 'text', 'scoring_play', 'score_value',
    'away_score', 'home_score', 'shooting_play', 'points_attempted',
    'coordinate_x', 'coordinate_y',
    'period_number', 'half', 'game_date',
    'start_game_seconds_remaining', 'end_game_seconds_remaining',
    'clock_minutes', 'clock_seconds',
    'home_team_id', 'away_team_id',
    'home_team_name', 'away_team_name',
    'team_id', 'athlete_id_1',
    'game_spread', 'home_favorite',
    # NOTE: season_type intentionally omitted — postseason parquet has no reg-season rows
]


# =============================================================================
# HELPERS  (identical to regular season version)
# =============================================================================

def shot_zone(row):
    """Classify a shot into paint / midrange / three."""
    tt = row['type_text']
    pa = row['points_attempted']
    if tt in ('LayUpShot', 'DunkShot', 'TipShot'):
        return 'paint'
    if pa == 3:
        return 'three'
    x, y = row['coordinate_x'], row['coordinate_y']
    if pd.notna(x) and pd.notna(y) and abs(x) < 1e6 and abs(y) < 1e6:
        dist = np.sqrt(x**2 + y**2)
        if dist <= 12:
            return 'paint'
    return 'midrange'


def estimate_possessions(events_df, team_id):
    """
    Estimate possessions for a team during a stint.
    Formula: FGA + 0.44*FTA + TOV - OREB
    """
    team = events_df[events_df['team_id'] == team_id]
    fga  = team['shooting_play'].sum()
    fta  = team['type_text'].isin(['MadeFreeThrow', 'MissedFreeThrow']).sum()
    tov  = team['type_text'].isin(['Lost Ball Turnover', 'Bad Pass Turnover', 'Turnover']).sum()
    oreb = team['type_text'].isin(['Offensive Rebound']).sum()
    return max(fga + 0.44 * fta + tov - oreb, 0)


def opp_quality_tier(opp_id, net_lookup):
    """Map a team_id to its NET quadrant label."""
    rank = net_lookup.get(opp_id)
    if rank is None:
        return 'unknown'
    rank = int(rank)
    if rank <= NET_Q1_MAX: return 'Q1'
    if rank <= NET_Q2_MAX: return 'Q2'
    if rank <= NET_Q3_MAX: return 'Q3'
    return 'Q4'


def classify_game_state(score_diff):
    """Classify game context by scoring margin."""
    abs_diff = abs(score_diff)
    if abs_diff <= 5:   return 'close'
    if abs_diff <= 10:  return 'competitive'
    return 'blowout'


# =============================================================================
# STEP 0 — VALIDATE INPUTS
# =============================================================================

if not validate_inputs(
    required=[POSTSEASON_PBP_FILE, NET_FILE, TOURNAMENT_BRACKET],
):
    sys.exit(1)


# =============================================================================
# STEP 1 — LOAD TOURNAMENT TEAMS FROM BRACKET
# =============================================================================
# Dynamically loads team IDs from tournament_bracket.csv so the script doesn't
# need to be updated if the bracket composition changes.

print("Loading tournament teams from bracket...")
bracket_df = pd.read_csv(TOURNAMENT_BRACKET)

# The bracket CSV uses team_location as the display name; we need ESPN team_ids.
# Try common column names for the numeric team ID.
_id_col_candidates = ['team_id', 'TeamID', 'espn_team_id', 'wehoop_team_id']
_bracket_id_col = next((c for c in _id_col_candidates if c in bracket_df.columns), None)

if _bracket_id_col:
    TOURNEY_TEAM_IDS = set(bracket_df[_bracket_id_col].dropna().astype(int).tolist())
    print(f"  Loaded {len(TOURNEY_TEAM_IDS)} tournament teams from bracket (via '{_bracket_id_col}')")
else:
    # Fallback: no numeric ID column — use team names to filter after loading PBP
    # (less efficient but functional; bracket CSV may only have team_location strings)
    TOURNEY_TEAM_IDS = None
    TOURNEY_TEAM_NAMES = set(bracket_df['team_location'].dropna().unique())
    print(f"  No numeric team_id column in bracket — will filter by team name ({len(TOURNEY_TEAM_NAMES)} teams)")
    print(f"  Note: add a 'team_id' column to tournament_bracket.csv for faster filtering.")


# =============================================================================
# STEP 2 — LOAD POSTSEASON PBP DATA
# =============================================================================

print(f"\nLoading postseason PBP from {POSTSEASON_PBP_FILE.name}...")

# Only load columns we need — keeps memory footprint manageable
available_cols = pd.read_parquet(POSTSEASON_PBP_FILE, columns=[]).columns.tolist() \
    if hasattr(pd, 'read_parquet') else []
load_cols = [c for c in LOAD_COLS if c in available_cols] if available_cols else LOAD_COLS

try:
    pbp = pd.read_parquet(POSTSEASON_PBP_FILE, columns=load_cols)
except TypeError:
    # Older pyarrow versions may not support selective column reads
    pbp = pd.read_parquet(POSTSEASON_PBP_FILE)
    pbp = pbp[[c for c in load_cols if c in pbp.columns]]

print(f"  Rows loaded: {len(pbp):,} | Games: {pbp['game_id'].nunique():,}")
print(f"  Date range: {pbp['game_date'].min()} → {pbp['game_date'].max()}")

# ── No season_type filter — all rows in the postseason parquet are tournament games ──
# (Unlike lineup_stints_pipeline.py which had: pbp = pbp[pbp['season_type'] == 2])

# Ensure integer types for ID columns
for col in ['home_team_id', 'away_team_id', 'team_id']:
    if col in pbp.columns:
        pbp[col] = pbp[col].astype('Int64')

pbp = pbp.sort_values(['game_id', 'sequence_number']).reset_index(drop=True)

# ── Filter to tournament teams ──────────────────────────────────────────────
if TOURNEY_TEAM_IDS:
    in_tourney = pbp['home_team_id'].isin(TOURNEY_TEAM_IDS) | pbp['away_team_id'].isin(TOURNEY_TEAM_IDS)
    pbp = pbp[in_tourney].copy()
else:
    in_tourney = pbp['home_team_name'].isin(TOURNEY_TEAM_NAMES) | pbp['away_team_name'].isin(TOURNEY_TEAM_NAMES)
    pbp = pbp[in_tourney].copy()
    TOURNEY_TEAM_IDS = set(
        pbp['home_team_id'].dropna().astype(int).tolist() +
        pbp['away_team_id'].dropna().astype(int).tolist()
    )

print(f"  After tournament team filter: {len(pbp):,} rows, {pbp['game_id'].nunique():,} games")


# =============================================================================
# STEP 3 — LOAD NET RANKINGS
# =============================================================================

print(f"\nLoading NET rankings from {NET_FILE.name}...")
try:
    net_df  = pd.read_csv(NET_FILE)
    id_col  = next((c for c in net_df.columns if 'team_id' in c.lower()), None)
    rk_col  = next((c for c in net_df.columns if 'net' in c.lower() and 'rank' in c.lower()), None)
    if id_col and rk_col:
        net_lookup = net_df.set_index(id_col)[rk_col].to_dict()
        print(f"  NET ranks loaded for {len(net_lookup):,} teams")
    else:
        print(f"  ⚠️  NET rank columns not found (available: {list(net_df.columns)[:8]}) — skipping")
        net_lookup = {}
except Exception as e:
    print(f"  ⚠️  Could not load NET rankings: {e}")
    net_lookup = {}


# =============================================================================
# STEP 4 — BUILD LINEUP STINTS PER GAME PER TEAM
# =============================================================================
# Logic is identical to the regular season pipeline.
# Reconstructs lineups from substitution events and flushes a stint record
# each time the lineup changes or a period ends.

print("\nBuilding postseason lineup stints...")
all_stints = []

game_groups  = pbp.groupby('game_id', sort=False)
total_games  = pbp['game_id'].nunique()

for g_idx, (game_id, gdf) in enumerate(game_groups):
    if g_idx % 10 == 0:
        print(f"  Processing game {g_idx + 1}/{total_games}...")

    gdf = gdf.sort_values('sequence_number').reset_index(drop=True)

    home_id    = int(gdf['home_team_id'].iloc[0]) if pd.notna(gdf['home_team_id'].iloc[0]) else None
    away_id    = int(gdf['away_team_id'].iloc[0]) if pd.notna(gdf['away_team_id'].iloc[0]) else None
    game_date  = gdf['game_date'].iloc[0]
    game_spread = gdf['game_spread'].iloc[0] if 'game_spread' in gdf.columns else None

    if home_id is None or away_id is None:
        continue

    tourney_in_game = [t for t in [home_id, away_id] if t in TOURNEY_TEAM_IDS]

    for focal_team in tourney_in_game:
        opp_team   = away_id if focal_team == home_id else home_id
        is_home    = (focal_team == home_id)
        team_name  = gdf['home_team_name'].iloc[0] if is_home else gdf['away_team_name'].iloc[0]

        # ── Reconstruct lineup from substitution events ──────────────────────
        on_court = set()
        stint_start_idx = 0
        stint_start_sec = gdf['start_game_seconds_remaining'].iloc[0]

        # Seed starting lineup from first 5 unique players who appear before first sub
        period_subs = gdf[
            (gdf['type_text'] == 'Substitution') & (gdf['team_id'] == focal_team)
        ]
        first_sub_idx = period_subs.index[0] if len(period_subs) > 0 else len(gdf)

        pre_sub_events = gdf.loc[:first_sub_idx, :]
        pre_sub_players = (
            pre_sub_events[
                (pre_sub_events['team_id'] == focal_team) &
                (pre_sub_events['type_text'] != 'Substitution') &
                (pre_sub_events['athlete_id_1'].notna())
            ]['athlete_id_1'].astype(int).unique().tolist()
        )
        on_court = set(pre_sub_players[:5])

        def flush_stint(end_idx, end_sec):
            """Record a completed stint to all_stints."""
            if not on_court or len(on_court) < MIN_STINT_PLAYERS:
                return
            stint_events = gdf.iloc[stint_start_idx:end_idx + 1]
            duration_sec = max(stint_start_sec - end_sec, 0)

            pts_scored  = stint_events.loc[
                (stint_events['scoring_play'] == True) & (stint_events['team_id'] == focal_team),
                'score_value'
            ].sum()
            pts_allowed = stint_events.loc[
                (stint_events['scoring_play'] == True) &
                (stint_events['team_id'] != focal_team) &
                (stint_events['team_id'].notna()),
                'score_value'
            ].sum()

            poss_for     = estimate_possessions(stint_events, focal_team)
            poss_against = estimate_possessions(stint_events, opp_team)

            # Game state at stint start
            try:
                start_row  = gdf.iloc[stint_start_idx]
                score_diff = (
                    int(start_row['home_score'] or 0) - int(start_row['away_score'] or 0)
                    if is_home else
                    int(start_row['away_score'] or 0) - int(start_row['home_score'] or 0)
                )
            except Exception:
                score_diff = 0

            half = int(stint_events['half'].mode().iloc[0]) if (len(stint_events) > 0 and 'half' in stint_events) else 1

            # Shot quality for focal team
            focal_shots = stint_events[
                (stint_events['team_id'] == focal_team) &
                (stint_events['shooting_play'] == True) &
                (stint_events['coordinate_x'].notna()) &
                (stint_events['coordinate_x'].abs() < 1e6)
            ].copy()

            n_shots = len(focal_shots)
            if n_shots > 0:
                focal_shots['zone'] = focal_shots.apply(shot_zone, axis=1)
                paint_fga    = (focal_shots['zone'] == 'paint').sum()
                midrange_fga = (focal_shots['zone'] == 'midrange').sum()
                three_fga    = (focal_shots['zone'] == 'three').sum()
                avg_shot_dist = np.sqrt(
                    focal_shots['coordinate_x']**2 + focal_shots['coordinate_y']**2
                ).mean()
            else:
                paint_fga = midrange_fga = three_fga = 0
                avg_shot_dist = np.nan

            all_stints.append({
                'game_id':       game_id,
                'game_date':     game_date,
                'team_id':       focal_team,
                'team_name':     team_name,
                'opp_team_id':   opp_team,
                'is_home':       is_home,
                'lineup':        frozenset(on_court),
                'lineup_str':    '|'.join(str(p) for p in sorted(on_court)),
                'n_players':     len(on_court),
                'duration_sec':  duration_sec,
                'half':          half,
                'game_state':    classify_game_state(score_diff),
                'opp_quality':   opp_quality_tier(opp_team, net_lookup),
                'pts_scored':    pts_scored,
                'pts_allowed':   pts_allowed,
                'poss_for':      poss_for,
                'poss_against':  poss_against,
                'n_shots':       n_shots,
                'paint_fga':     paint_fga,
                'midrange_fga':  midrange_fga,
                'three_fga':     three_fga,
                'avg_shot_dist': avg_shot_dist,
                'stint_start_idx': stint_start_idx,
            })

        # ── Walk through plays, tracking lineup changes ────────────────────
        for i, row in gdf.iterrows():
            row_idx = gdf.index.get_loc(i)

            if (row['type_text'] == 'Substitution' and
                    pd.notna(row['team_id']) and int(row['team_id']) == focal_team):
                flush_stint(row_idx, row['start_game_seconds_remaining'] or 0)
                stint_start_idx = row_idx
                stint_start_sec = row['start_game_seconds_remaining'] or 0

                pid = int(row['athlete_id_1']) if pd.notna(row['athlete_id_1']) else None
                if pid is None:
                    continue
                txt = str(row['text']).lower()
                if 'subbing in' in txt or 'entering' in txt:
                    on_court.add(pid)
                elif 'subbing out' in txt or 'leaving' in txt:
                    on_court.discard(pid)

            elif row['type_text'] == 'End Period':
                flush_stint(row_idx, row['end_game_seconds_remaining'] or 0)
                on_court = set()
                stint_start_idx = row_idx + 1
                stint_start_sec = row['end_game_seconds_remaining'] or 0

                # Re-seed lineup from next period's first players
                next_rows     = gdf.iloc[row_idx + 1:]
                next_sub_idxs = next_rows[
                    (next_rows['type_text'] == 'Substitution') &
                    (next_rows['team_id'] == focal_team)
                ].index
                next_sub_loc  = (
                    gdf.index.get_loc(next_sub_idxs[0])
                    if len(next_sub_idxs) > 0
                    else len(gdf) - 1
                )
                period_pre = gdf.iloc[row_idx + 1:next_sub_loc]
                new_starters = (
                    period_pre[
                        (period_pre['team_id'] == focal_team) &
                        (period_pre['type_text'] != 'Substitution') &
                        (period_pre['athlete_id_1'].notna())
                    ]['athlete_id_1'].astype(int).unique().tolist()
                )
                on_court = set(new_starters[:5])

        # Flush final stint
        if len(gdf) > 0 and len(on_court) >= MIN_STINT_PLAYERS:
            last_row = gdf.iloc[-1]
            flush_stint(len(gdf) - 1, last_row['end_game_seconds_remaining'] or 0)

print(f"  Total postseason stints collected: {len(all_stints):,}")


# =============================================================================
# STEP 5 — BUILD STINT DATAFRAME
# =============================================================================

print("\nBuilding stint dataframe...")
stints_df = pd.DataFrame(all_stints)
stints_df = stints_df[stints_df['n_players'] >= MIN_STINT_PLAYERS].copy()
print(f"  Stints with {MIN_STINT_PLAYERS}+ players: {len(stints_df):,}")

# Per-100 ratings at stint level
def safe_per100(pts, poss):
    return (pts / poss * 100) if poss > 1 else np.nan

stints_df['off_rtg']   = stints_df.apply(lambda r: safe_per100(r['pts_scored'],  r['poss_for']),     axis=1)
stints_df['def_rtg']   = stints_df.apply(lambda r: safe_per100(r['pts_allowed'], r['poss_against']), axis=1)
stints_df['net_rtg']   = stints_df['off_rtg'] - stints_df['def_rtg']
stints_df['plus_minus'] = stints_df['pts_scored'] - stints_df['pts_allowed']

# Shot quality ratios
total_fga = (stints_df['paint_fga'] + stints_df['midrange_fga'] + stints_df['three_fga']).replace(0, np.nan)
stints_df['paint_pct']    = stints_df['paint_fga']    / total_fga
stints_df['midrange_pct'] = stints_df['midrange_fga'] / total_fga
stints_df['three_pct']    = stints_df['three_fga']    / total_fga

# Save stint-level output (drop non-serializable columns)
stint_save = stints_df.copy()
stint_save['lineup'] = stint_save['lineup_str']
stint_save = stint_save.drop(columns=['stint_start_idx'], errors='ignore')
stint_save.to_csv(POSTSEASON_STINTS, index=False)
print(f"  ✅  Saved {POSTSEASON_STINTS.name} ({len(stint_save):,} rows)")


# =============================================================================
# STEP 6 — PLAYER-LEVEL ON/OFF AGGREGATION
# =============================================================================

print("\nComputing player-level postseason on/off metrics...")

# Expand stints to player-level (one row per player per stint they're in)
player_rows = []
for _, stint in stints_df.iterrows():
    for pid in stint['lineup']:
        player_rows.append({
            'athlete_id': pid,
            'team_id':    stint['team_id'],
            'team_name':  stint['team_name'],
            'game_id':    stint['game_id'],
            **{k: stint[k] for k in [
                'duration_sec', 'half', 'game_state', 'opp_quality',
                'pts_scored', 'pts_allowed', 'poss_for', 'poss_against',
                'plus_minus', 'n_shots', 'paint_fga', 'midrange_fga',
                'three_fga', 'avg_shot_dist', 'paint_pct', 'midrange_pct', 'three_pct'
            ]}
        })

on_df = pd.DataFrame(player_rows)
print(f"  On-court player-stint rows: {len(on_df):,}")

# Off-court: all team stints where a given player is NOT in the lineup
player_teams  = on_df[['athlete_id', 'team_id', 'team_name']].drop_duplicates()
team_stints   = stints_df.copy()

print("  Computing off-court aggregates...")
off_rows = []
for player_id, tid in player_teams[['athlete_id', 'team_id']].values:
    team_s = team_stints[team_stints['team_id'] == tid]
    off_s  = team_s[~team_s['lineup'].apply(lambda lp: player_id in lp)]
    if len(off_s) > 0:
        off_rows.append({
            'athlete_id':       player_id,
            'team_id':          tid,
            'off_poss_for':     off_s['poss_for'].sum(),
            'off_poss_against': off_s['poss_against'].sum(),
            'off_pts_scored':   off_s['pts_scored'].sum(),
            'off_pts_allowed':  off_s['pts_allowed'].sum(),
            'off_plus_minus':   off_s['plus_minus'].sum(),
        })

off_df = pd.DataFrame(off_rows)
print(f"  Off-court rows built: {len(off_df):,}")


# =============================================================================
# STEP 7 — AGGREGATE ON-COURT METRICS
# =============================================================================

print("\nAggregating on-court metrics...")

def per100(pts_col, poss_col, df, min_poss=None):
    min_poss = min_poss or MIN_POSSESSIONS
    return np.where(df[poss_col] > min_poss, df[pts_col] / df[poss_col] * 100, np.nan)

agg_base = on_df.groupby(['athlete_id', 'team_id', 'team_name']).agg(
    tourney_games     =('game_id',       'nunique'),
    tourney_stints    =('game_id',       'count'),     # stints, not games
    tourney_on_sec    =('duration_sec',  'sum'),
    on_poss_for       =('poss_for',      'sum'),
    on_poss_against   =('poss_against',  'sum'),
    on_pts_scored     =('pts_scored',    'sum'),
    on_pts_allowed    =('pts_allowed',   'sum'),
    on_plus_minus     =('plus_minus',    'sum'),
    total_on_shots    =('n_shots',       'sum'),
    on_paint_fga      =('paint_fga',     'sum'),
    on_midrange_fga   =('midrange_fga',  'sum'),
    on_three_fga      =('three_fga',     'sum'),
    on_avg_shot_dist  =('avg_shot_dist', 'mean'),
).reset_index()

agg_base['tourney_on_min']  = (agg_base['tourney_on_sec'] / 60).round(1)
agg_base['tourney_off_rtg'] = per100('on_pts_scored',  'on_poss_for',     agg_base)
agg_base['tourney_def_rtg'] = per100('on_pts_allowed', 'on_poss_against', agg_base)
agg_base['tourney_net_rtg'] = agg_base['tourney_off_rtg'] - agg_base['tourney_def_rtg']

# Shot quality proportions
total_on = (
    agg_base['on_paint_fga'] + agg_base['on_midrange_fga'] + agg_base['on_three_fga']
).replace(0, np.nan)
agg_base['tourney_paint_pct']    = (agg_base['on_paint_fga']    / total_on).round(3)
agg_base['tourney_midrange_pct'] = (agg_base['on_midrange_fga'] / total_on).round(3)
agg_base['tourney_three_pct']    = (agg_base['on_three_fga']    / total_on).round(3)


# =============================================================================
# STEP 8 — CONTEXTUAL SPLITS
# =============================================================================

print("Computing contextual splits...")

def split_agg(split_col, split_val, label):
    sub = on_df[on_df[split_col] == split_val].groupby(['athlete_id', 'team_id']).agg(
        **{f'{label}_poss_for':     ('poss_for',    'sum'),
           f'{label}_poss_against': ('poss_against','sum'),
           f'{label}_pts_scored':   ('pts_scored',  'sum'),
           f'{label}_pts_allowed':  ('pts_allowed', 'sum')}
    ).reset_index()
    sub[f'{label}_off_rtg'] = np.where(sub[f'{label}_poss_for']     > 3,
                                       sub[f'{label}_pts_scored']   / sub[f'{label}_poss_for']     * 100, np.nan)
    sub[f'{label}_def_rtg'] = np.where(sub[f'{label}_poss_against'] > 3,
                                       sub[f'{label}_pts_allowed']  / sub[f'{label}_poss_against'] * 100, np.nan)
    sub[f'{label}_net_rtg'] = sub[f'{label}_off_rtg'] - sub[f'{label}_def_rtg']
    keep = ['athlete_id', 'team_id',
            f'{label}_off_rtg', f'{label}_def_rtg', f'{label}_net_rtg']
    return sub[keep]

h1  = split_agg('half',       1,             'tourney_h1')
h2  = split_agg('half',       2,             'tourney_h2')
cls = split_agg('game_state', 'close',       'tourney_close')
cmp = split_agg('game_state', 'competitive', 'tourney_comp')
blw = split_agg('game_state', 'blowout',     'tourney_blowout')
q1  = split_agg('opp_quality','Q1',          'tourney_q1_opp')
q2  = split_agg('opp_quality','Q2',          'tourney_q2_opp')


# =============================================================================
# STEP 9 — LINEUP-LEVEL METRICS
# =============================================================================

print("Computing lineup-level metrics...")

lineup_agg = stints_df.groupby(['team_id', 'lineup_str']).agg(
    lineup_stints      =('plus_minus',   'count'),
    lineup_sec         =('duration_sec', 'sum'),
    lineup_poss_for    =('poss_for',     'sum'),
    lineup_poss_ag     =('poss_against', 'sum'),
    lineup_pts_scored  =('pts_scored',   'sum'),
    lineup_pts_allowed =('pts_allowed',  'sum'),
    lineup_plus_minus  =('plus_minus',   'sum'),
).reset_index()

lineup_agg['lineup_min_est'] = (lineup_agg['lineup_sec'] / 60).round(1)
lineup_agg['lineup_off_rtg'] = np.where(
    lineup_agg['lineup_poss_for'] > MIN_POSSESSIONS,
    lineup_agg['lineup_pts_scored'] / lineup_agg['lineup_poss_for'] * 100, np.nan)
lineup_agg['lineup_net_rtg'] = np.where(
    lineup_agg['lineup_poss_ag'] > MIN_POSSESSIONS,
    lineup_agg['lineup_plus_minus'] / lineup_agg['lineup_poss_for'] * 100, np.nan)

# Best lineup per player
player_lineup_best = []
for _, p_row in agg_base[['athlete_id', 'team_id']].iterrows():
    pid, tid = p_row['athlete_id'], p_row['team_id']
    plup = lineup_agg[
        (lineup_agg['team_id'] == tid) &
        (lineup_agg['lineup_str'].str.contains(str(pid)))
    ]
    if len(plup) == 0:
        player_lineup_best.append({'athlete_id': pid, 'team_id': tid,
                                   'tourney_top_lineup_net': np.nan,
                                   'tourney_top_lineup_min': np.nan,
                                   'tourney_n_lineups': 0})
        continue
    best = plup.nlargest(1, 'lineup_min_est', keep='first')
    player_lineup_best.append({
        'athlete_id':             pid,
        'team_id':                tid,
        'tourney_n_lineups':      len(plup),
        'tourney_top_lineup_min': best['lineup_min_est'].iloc[0],
        'tourney_top_lineup_net': best['lineup_net_rtg'].iloc[0],
    })
lineup_player_df = pd.DataFrame(player_lineup_best)


# =============================================================================
# STEP 10 — MERGE EVERYTHING & EXPORT
# =============================================================================

print("\nMerging all postseason metrics...")

final = agg_base.copy()

# Off-court ratings
off_df['off_off_rtg'] = np.where(off_df['off_poss_for']     > MIN_POSSESSIONS,
                                  off_df['off_pts_scored']  / off_df['off_poss_for']     * 100, np.nan)
off_df['off_def_rtg'] = np.where(off_df['off_poss_against'] > MIN_POSSESSIONS,
                                  off_df['off_pts_allowed'] / off_df['off_poss_against'] * 100, np.nan)
off_df['off_net_rtg'] = off_df['off_off_rtg'] - off_df['off_def_rtg']

final = final.merge(
    off_df[['athlete_id', 'team_id',
            'off_off_rtg', 'off_def_rtg', 'off_net_rtg', 'off_plus_minus']],
    on=['athlete_id', 'team_id'], how='left'
)

# On/Off differentials (key signal: positive = player improves team)
final['tourney_net_rtg_diff'] = final['tourney_net_rtg'] - final['off_net_rtg']
final['tourney_off_rtg_diff'] = final['tourney_off_rtg'] - final['off_off_rtg']
final['tourney_def_rtg_diff'] = final['tourney_def_rtg'] - final['off_def_rtg']

# Contextual splits
for df_split in [h1, h2, cls, cmp, blw, q1, q2]:
    final = final.merge(df_split, on=['athlete_id', 'team_id'], how='left')

# Lineup summaries
final = final.merge(lineup_player_df, on=['athlete_id', 'team_id'], how='left')

# Round floats
float_cols = final.select_dtypes(include='float').columns
final[float_cols] = final[float_cols].round(2)

# Backward-compatible alias used by downstream validators/consumers
if 'on_poss_for' in final.columns and 'tourney_poss_on' not in final.columns:
    final['tourney_poss_on'] = final['on_poss_for']

# Export
final.to_csv(POSTSEASON_ONOFF, index=False)

print(f"\n{'='*55}")
print(f"  POSTSEASON PIPELINE COMPLETE")
print(f"{'='*55}")
print(f"  ✅  {POSTSEASON_STINTS.name}  →  {len(stint_save):,} stints")
print(f"  ✅  {POSTSEASON_ONOFF.name}   →  {len(final):,} players, {len(final.columns)} cols")
print(f"\n  Top 10 players by postseason net rating differential:")
preview = (
    final[['athlete_id', 'team_name', 'tourney_net_rtg',
           'off_net_rtg', 'tourney_net_rtg_diff', 'tourney_on_min']]
    .dropna(subset=['tourney_net_rtg_diff'])
    .nlargest(10, 'tourney_net_rtg_diff')
)
print(preview.to_string(index=False))
print()
