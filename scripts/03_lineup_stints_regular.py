"""
Lineup Stints & On/Off Pipeline
NCAA WBB 2026 — Regular Season, Round of 32 Teams
Tiers 1-4: Core On/Off, Lineup-Level, Shot Quality, Contextual Splits

Output: player_onoff_metrics.csv (player-level)
        lineup_stints_raw.csv   (stint-level, for reference)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from config import PBP_FILE, NET_FILE, LINEUP_STINTS_RAW, PLAYER_ONOFF, CONSTANTS

DATA_PATH  = PBP_FILE
RANKS_PATH = NET_FILE
OUT_DIR    = LINEUP_STINTS_RAW.parent

# Round of 32 team IDs (identified from postseason PBP March 22-23 R32 games)
R32_TEAM_IDS = {
    130,   # Michigan
    152,   # NC State
    201,   # Oklahoma
    127,   # Michigan State
    99,    # LSU
    2641,  # Texas Tech
    2628,  # TCU
    264,   # Washington
    150,   # Duke
    239,   # Baylor
    251,   # Texas
    2483,  # Oregon
    153,   # North Carolina
    120,   # Maryland
    135,   # Minnesota
    145,   # Ole Miss
    194,   # Ohio State
    87,    # Notre Dame
    2294,  # Iowa
    258,   # Virginia
    2579,  # South Carolina
    30,    # USC
    41,    # UConn
    183,   # Syracuse
    277,   # West Virginia
    96,    # Kentucky
    238,   # Vanderbilt
    356,   # Illinois
    26,    # UCLA
    197,   # Oklahoma State
    97,    # Louisville
    333,   # Alabama
}

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
    'season_type', 'game_spread', 'home_favorite'
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def shot_zone(row):
    """Classify shot into paint / midrange / three."""
    tt = row['type_text']
    pa = row['points_attempted']

    if tt in ('LayUpShot', 'DunkShot', 'TipShot'):
        return 'paint'
    if pa == 3:
        return 'three'

    # Use coordinates for JumpShot paint vs midrange
    x = row['coordinate_x']
    y = row['coordinate_y']
    if pd.notna(x) and pd.notna(y) and abs(x) < 1e6 and abs(y) < 1e6:
        # Basket estimated at |x|~38-42, y~0 in this coordinate system
        # Paint = within roughly 8ft of basket along y, 6ft wide
        # Proxy: |y| <= 8 and near wing/center (not corner 3)
        dist = np.sqrt(x**2 + y**2)
        if dist <= 12:
            return 'paint'
    return 'midrange'


def estimate_possessions(events_df, team_id):
    """
    Estimate possessions for a team during a stint using standard formula:
    Poss = FGA + 0.44*FTA + TOV - OREB
    Uses only events belonging to the team.
    """
    team = events_df[events_df['team_id'] == team_id]
    fga  = team['shooting_play'].sum()
    # Free throw attempts: MadeFreeThrow or MissedFreeThrow
    fta  = team['type_text'].isin(['MadeFreeThrow', 'MissedFreeThrow']).sum()
    tov  = team['type_text'].isin([
        'Lost Ball Turnover', 'Bad Pass Turnover', 'Turnover'
    ]).sum()
    oreb = team['type_text'].isin(['Offensive Rebound']).sum()
    poss = fga + 0.44 * fta + tov - oreb
    return max(poss, 0)


def get_opponent_id(game_row, team_id):
    if game_row['home_team_id'] == team_id:
        return game_row['away_team_id']
    return game_row['home_team_id']


def classify_game_state(score_diff):
    """Classify game state based on scoring margin."""
    abs_diff = abs(score_diff)
    if abs_diff <= 5:
        return 'close'
    elif abs_diff <= 10:
        return 'competitive'
    else:
        return 'blowout'


# ─────────────────────────────────────────────
# STEP 1: LOAD & FILTER
# ─────────────────────────────────────────────
print("Loading PBP data...")
pbp = pd.read_parquet(DATA_PATH, columns=LOAD_COLS)

# Regular season only
pbp = pbp[pbp['season_type'] == 2].copy()
print(f"  Regular season rows: {len(pbp):,}")

# Filter to games involving at least one R32 team
pbp['home_team_id'] = pbp['home_team_id'].astype('Int64')
pbp['away_team_id'] = pbp['away_team_id'].astype('Int64')
pbp['team_id']      = pbp['team_id'].astype('Int64')

in_r32 = pbp['home_team_id'].isin(R32_TEAM_IDS) | pbp['away_team_id'].isin(R32_TEAM_IDS)
pbp = pbp[in_r32].copy()
print(f"  After R32 team filter: {len(pbp):,} rows, {pbp['game_id'].nunique():,} games")

# Sort
pbp = pbp.sort_values(['game_id', 'sequence_number']).reset_index(drop=True)

# ─────────────────────────────────────────────
# STEP 2: LOAD OPPONENT QUALITY (NET RANKINGS)
# ─────────────────────────────────────────────
print("Loading NET rankings...")
try:
    net_df = pd.read_csv(RANKS_PATH)
    # Find team_id and NET rank columns
    id_col  = next((c for c in net_df.columns if 'team_id' in c.lower()), None)
    rk_col  = next((c for c in net_df.columns if 'net' in c.lower() and 'rank' in c.lower()), None)
    if id_col and rk_col:
        net_lookup = net_df.set_index(id_col)[rk_col].to_dict()
        print(f"  Loaded NET ranks for {len(net_lookup)} teams")
    else:
        print(f"  NET rank columns not found (cols: {list(net_df.columns)[:8]}), skipping")
        net_lookup = {}
except Exception as e:
    print(f"  Could not load NET rankings: {e}")
    net_lookup = {}

def opp_quality_tier(opp_id, net_lookup):
    rank = net_lookup.get(opp_id, None)
    if rank is None:
        return 'unknown'
    rank = int(rank)
    if rank <= 30:   return 'Q1'
    if rank <= 75:   return 'Q2'
    if rank <= 160:  return 'Q3'
    return 'Q4'

# ─────────────────────────────────────────────
# STEP 3: BUILD LINEUP STINTS PER GAME PER TEAM
# ─────────────────────────────────────────────
print("Building lineup stints...")

all_stints = []

game_groups = pbp.groupby('game_id', sort=False)
total_games = pbp['game_id'].nunique()

for g_idx, (game_id, gdf) in enumerate(game_groups):
    if g_idx % 200 == 0:
        print(f"  Processing game {g_idx+1}/{total_games}...")

    gdf = gdf.sort_values('sequence_number').reset_index(drop=True)

    home_id = int(gdf['home_team_id'].iloc[0]) if pd.notna(gdf['home_team_id'].iloc[0]) else None
    away_id = int(gdf['away_team_id'].iloc[0]) if pd.notna(gdf['away_team_id'].iloc[0]) else None
    game_date = gdf['game_date'].iloc[0]
    game_spread = gdf['game_spread'].iloc[0] if 'game_spread' in gdf.columns else None

    if home_id is None or away_id is None:
        continue

    # Process each R32 team playing in this game
    r32_in_game = [t for t in [home_id, away_id] if t in R32_TEAM_IDS]

    for focal_team in r32_in_game:
        opp_team = away_id if focal_team == home_id else home_id
        is_home  = (focal_team == home_id)
        team_name = gdf['home_team_name'].iloc[0] if is_home else gdf['away_team_name'].iloc[0]

        # ── Reconstruct lineup from substitution events ──
        # Strategy: seed lineup from first-appearing players in each period,
        # then track subs to maintain a running 5-man set.

        on_court = set()  # current lineup for focal_team
        prev_lineup = None
        stint_start_idx = 0
        stint_start_sec  = gdf['start_game_seconds_remaining'].iloc[0]

        # Pre-pass: gather all player IDs who appear for this team (non-sub)
        team_events = gdf[gdf['team_id'] == focal_team]
        all_team_players = set(
            team_events[team_events['type_text'] != 'Substitution']['athlete_id_1'].dropna().astype(int).tolist()
        )

        # Seed starting lineup from first 5 unique players who appear before first sub
        period_subs = gdf[
            (gdf['type_text'] == 'Substitution') &
            (gdf['team_id'] == focal_team)
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
            """Record a completed stint."""
            if not on_court or len(on_court) < 3:
                return  # skip incomplete lineups
            stint_events = gdf.iloc[stint_start_idx:end_idx + 1]
            duration_sec = max(stint_start_sec - end_sec, 0)

            # Points scored / allowed
            pts_scored  = stint_events.loc[
                (stint_events['scoring_play'] == True) &
                (stint_events['team_id'] == focal_team), 'score_value'
            ].sum()
            pts_allowed = stint_events.loc[
                (stint_events['scoring_play'] == True) &
                (stint_events['team_id'] != focal_team) &
                (stint_events['team_id'].notna()), 'score_value'
            ].sum()

            poss_for     = estimate_possessions(stint_events, focal_team)
            poss_against = estimate_possessions(stint_events, opp_team)

            # Score differential at start of stint (for game-state context)
            try:
                start_row = gdf.iloc[stint_start_idx]
                if is_home:
                    score_diff = int(start_row['home_score'] or 0) - int(start_row['away_score'] or 0)
                else:
                    score_diff = int(start_row['away_score'] or 0) - int(start_row['home_score'] or 0)
            except:
                score_diff = 0

            # Half
            half = int(stint_events['half'].mode().iloc[0]) if len(stint_events) > 0 and 'half' in stint_events else 1

            # Shot quality metrics for focal team
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
                'game_id':        game_id,
                'game_date':      game_date,
                'team_id':        focal_team,
                'team_name':      team_name,
                'opp_team_id':    opp_team,
                'is_home':        is_home,
                'lineup':         frozenset(on_court),
                'lineup_str':     '|'.join(str(p) for p in sorted(on_court)),
                'n_players':      len(on_court),
                'duration_sec':   duration_sec,
                'half':           half,
                'game_state':     classify_game_state(score_diff),
                'pts_scored':     pts_scored,
                'pts_allowed':    pts_allowed,
                'poss_for':       poss_for,
                'poss_against':   poss_against,
                'n_shots':        n_shots,
                'paint_fga':      paint_fga,
                'midrange_fga':   midrange_fga,
                'three_fga':      three_fga,
                'avg_shot_dist':  avg_shot_dist,
                'stint_start_idx': stint_start_idx,
            })

        # ── Walk through plays tracking lineup changes ──
        for i, row in gdf.iterrows():
            row_idx = gdf.index.get_loc(i)

            if row['type_text'] == 'Substitution' and pd.notna(row['team_id']) and int(row['team_id']) == focal_team:
                # Flush current stint before changing lineup
                flush_stint(row_idx, row['start_game_seconds_remaining'] or 0)
                stint_start_idx = row_idx
                stint_start_sec  = row['start_game_seconds_remaining'] or 0

                pid = int(row['athlete_id_1']) if pd.notna(row['athlete_id_1']) else None
                if pid is None:
                    continue

                # Determine if subbing in or out from text
                txt = str(row['text']).lower()
                if 'subbing in' in txt or 'entering' in txt:
                    on_court.add(pid)
                elif 'subbing out' in txt or 'leaving' in txt:
                    on_court.discard(pid)
                else:
                    # Can't tell — skip to avoid corrupting lineup
                    pass

            # Handle period transitions — reset to empty and re-seed
            elif row['type_text'] == 'End Period':
                flush_stint(row_idx, row['end_game_seconds_remaining'] or 0)
                on_court = set()  # will be re-seeded at next period start
                stint_start_idx = row_idx + 1
                stint_start_sec  = row['end_game_seconds_remaining'] or 0

                # Re-seed from next period's first players
                next_rows = gdf.iloc[row_idx + 1:]
                next_sub_idxs = next_rows[
                    (next_rows['type_text'] == 'Substitution') &
                    (next_rows['team_id'] == focal_team)
                ].index
                next_sub_loc = gdf.index.get_loc(next_sub_idxs[0]) if len(next_sub_idxs) > 0 else len(gdf) - 1
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
        if len(gdf) > 0 and len(on_court) >= 3:
            last_row = gdf.iloc[-1]
            flush_stint(len(gdf) - 1, last_row['end_game_seconds_remaining'] or 0)

print(f"  Total stints collected: {len(all_stints):,}")

# ─────────────────────────────────────────────
# STEP 4: BUILD STINT DATAFRAME
# ─────────────────────────────────────────────
print("Building stint dataframe...")
stints_df = pd.DataFrame(all_stints)

# Filter stints with valid possessions / duration
stints_df = stints_df[stints_df['n_players'] >= 4].copy()
print(f"  Stints with 4+ players: {len(stints_df):,}")

# Attach opponent quality
stints_df['opp_quality'] = stints_df['opp_team_id'].apply(
    lambda x: opp_quality_tier(x, net_lookup)
)

# Compute per-100 ratings at stint level
def safe_per100(pts, poss):
    return (pts / poss * 100) if poss > 1 else np.nan

stints_df['off_rtg'] = stints_df.apply(lambda r: safe_per100(r['pts_scored'],  r['poss_for']),     axis=1)
stints_df['def_rtg'] = stints_df.apply(lambda r: safe_per100(r['pts_allowed'], r['poss_against']), axis=1)
stints_df['net_rtg'] = stints_df['off_rtg'] - stints_df['def_rtg']
stints_df['plus_minus'] = stints_df['pts_scored'] - stints_df['pts_allowed']

# Shot quality ratios
total_fga = (stints_df['paint_fga'] + stints_df['midrange_fga'] + stints_df['three_fga']).replace(0, np.nan)
stints_df['paint_pct']    = stints_df['paint_fga']    / total_fga
stints_df['midrange_pct'] = stints_df['midrange_fga'] / total_fga
stints_df['three_pct']    = stints_df['three_fga']    / total_fga

# Save stint-level (lighter version — no lineup object)
stint_save = stints_df.copy()
stint_save['lineup'] = stint_save['lineup_str']
stint_save = stint_save.drop(columns=['stint_start_idx'], errors='ignore')
stint_save.to_csv(LINEUP_STINTS_RAW, index=False)
print(f"  Saved lineup_stints_raw.csv ({len(stint_save):,} rows)")

# ─────────────────────────────────────────────
# STEP 5: PLAYER-LEVEL ON/OFF AGGREGATION
# ─────────────────────────────────────────────
print("Computing player-level on/off metrics...")

def weighted_rtg(pts, poss, min_poss=5):
    """Return per-100 rating; NaN if too few possessions."""
    total_poss = poss.sum()
    if total_poss < min_poss:
        return np.nan
    return pts.sum() / total_poss * 100

# Expand stints to player level (one row per player per stint they're IN)
player_rows = []
for _, stint in stints_df.iterrows():
    for pid in stint['lineup']:
        player_rows.append({
            'athlete_id': pid,
            'team_id':    stint['team_id'],
            'team_name':  stint['team_name'],
            'game_id':    stint['game_id'],
            'stint_is_on': True,
            **{k: stint[k] for k in [
                'duration_sec', 'half', 'game_state', 'opp_quality',
                'pts_scored', 'pts_allowed', 'poss_for', 'poss_against',
                'plus_minus', 'n_shots', 'paint_fga', 'midrange_fga',
                'three_fga', 'avg_shot_dist', 'paint_pct', 'midrange_pct', 'three_pct'
            ]}
        })

on_df = pd.DataFrame(player_rows)
print(f"  On-court player-stint rows: {len(on_df):,}")

# Build "off" stints: all stints for a team where a given player is NOT in the lineup
# For each team, get all team stints; for each player on that team, off = stints where they're absent

# First get all team-game-player combinations
player_teams = on_df[['athlete_id', 'team_id', 'team_name']].drop_duplicates()

# For off-court, use team stints
team_stints = stints_df.copy()

print("  Computing off-court aggregates (this may take a moment)...")
off_rows = []
for (player_id, tid) in player_teams[['athlete_id', 'team_id']].values:
    team_s = team_stints[team_stints['team_id'] == tid]
    off_s  = team_s[~team_s['lineup'].apply(lambda lp: player_id in lp)]
    if len(off_s) > 0:
        off_rows.append({
            'athlete_id':      player_id,
            'team_id':         tid,
            'off_poss_for':    off_s['poss_for'].sum(),
            'off_poss_against':off_s['poss_against'].sum(),
            'off_pts_scored':  off_s['pts_scored'].sum(),
            'off_pts_allowed': off_s['pts_allowed'].sum(),
            'off_plus_minus':  off_s['plus_minus'].sum(),
        })

off_df = pd.DataFrame(off_rows)
print(f"  Off-court rows built: {len(off_df):,}")

# ─────────────────────────────────────────────
# STEP 6: AGGREGATE ON-COURT METRICS
# ─────────────────────────────────────────────
print("Aggregating on-court metrics...")

agg_base = on_df.groupby(['athlete_id', 'team_id', 'team_name']).agg(
    games_played    =('game_id',       'nunique'),
    total_stints    =('stint_is_on',   'count'),
    on_sec          =('duration_sec',  'sum'),
    on_poss_for     =('poss_for',      'sum'),
    on_poss_against =('poss_against',  'sum'),
    on_pts_scored   =('pts_scored',    'sum'),
    on_pts_allowed  =('pts_allowed',   'sum'),
    on_plus_minus   =('plus_minus',    'sum'),
    # Shot quality
    total_on_shots  =('n_shots',       'sum'),
    on_paint_fga    =('paint_fga',     'sum'),
    on_midrange_fga =('midrange_fga',  'sum'),
    on_three_fga    =('three_fga',     'sum'),
    on_avg_shot_dist=('avg_shot_dist', 'mean'),
).reset_index()

# Per-100 ratings
def per100(pts_col, poss_col, df):
    return np.where(df[poss_col] > 5, df[pts_col] / df[poss_col] * 100, np.nan)

agg_base['on_off_rtg'] = per100('on_pts_scored',  'on_poss_for',     agg_base)
agg_base['on_def_rtg'] = per100('on_pts_allowed', 'on_poss_against', agg_base)
agg_base['on_net_rtg'] = agg_base['on_off_rtg'] - agg_base['on_def_rtg']
agg_base['on_min_est'] = (agg_base['on_sec'] / 60).round(1)

# Shot quality
total_on = (agg_base['on_paint_fga'] + agg_base['on_midrange_fga'] + agg_base['on_three_fga']).replace(0, np.nan)
agg_base['on_paint_pct']    = (agg_base['on_paint_fga']    / total_on).round(3)
agg_base['on_midrange_pct'] = (agg_base['on_midrange_fga'] / total_on).round(3)
agg_base['on_three_pct']    = (agg_base['on_three_fga']    / total_on).round(3)

# ─────────────────────────────────────────────
# STEP 7: CONTEXTUAL SPLITS (Tier 4)
# ─────────────────────────────────────────────
print("Computing contextual splits...")

def split_agg(split_col, split_val, label):
    sub = on_df[on_df[split_col] == split_val].groupby(['athlete_id', 'team_id']).agg(
        **{f'{label}_poss_for':    ('poss_for',    'sum'),
           f'{label}_poss_against':('poss_against','sum'),
           f'{label}_pts_scored':  ('pts_scored',  'sum'),
           f'{label}_pts_allowed': ('pts_allowed', 'sum'),
        }
    ).reset_index()
    sub[f'{label}_off_rtg'] = np.where(sub[f'{label}_poss_for']     > 3,
                                       sub[f'{label}_pts_scored']  / sub[f'{label}_poss_for']     * 100, np.nan)
    sub[f'{label}_def_rtg'] = np.where(sub[f'{label}_poss_against'] > 3,
                                       sub[f'{label}_pts_allowed'] / sub[f'{label}_poss_against'] * 100, np.nan)
    sub[f'{label}_net_rtg'] = sub[f'{label}_off_rtg'] - sub[f'{label}_def_rtg']
    keep = ['athlete_id', 'team_id', f'{label}_off_rtg', f'{label}_def_rtg', f'{label}_net_rtg']
    return sub[keep]

h1  = split_agg('half',       1,        'h1')
h2  = split_agg('half',       2,        'h2')
cls = split_agg('game_state', 'close',  'close')
cmp = split_agg('game_state', 'competitive', 'comp')
blw = split_agg('game_state', 'blowout','blowout')
q1  = split_agg('opp_quality','Q1',     'q1_opp')
q2  = split_agg('opp_quality','Q2',     'q2_opp')

# ─────────────────────────────────────────────
# STEP 8: LINEUP-LEVEL METRICS (Tier 2)
# ─────────────────────────────────────────────
print("Computing lineup-level metrics...")

lineup_agg = stints_df.groupby(['team_id', 'lineup_str']).agg(
    lineup_stints    =('plus_minus',   'count'),
    lineup_sec       =('duration_sec', 'sum'),
    lineup_poss_for  =('poss_for',     'sum'),
    lineup_poss_ag   =('poss_against', 'sum'),
    lineup_pts_scored=('pts_scored',   'sum'),
    lineup_pts_allowed=('pts_allowed', 'sum'),
    lineup_plus_minus=('plus_minus',   'sum'),
).reset_index()

lineup_agg['lineup_min_est'] = (lineup_agg['lineup_sec'] / 60).round(1)
lineup_agg['lineup_off_rtg'] = np.where(
    lineup_agg['lineup_poss_for'] > 5,
    lineup_agg['lineup_pts_scored'] / lineup_agg['lineup_poss_for'] * 100, np.nan)
lineup_agg['lineup_net_rtg'] = np.where(
    lineup_agg['lineup_poss_ag'] > 5,
    lineup_agg['lineup_plus_minus'] / lineup_agg['lineup_poss_for'] * 100, np.nan)

# Best and most-used lineup per player (for player-level summary)
player_lineup_best = []
for _, p_row in agg_base[['athlete_id', 'team_id']].iterrows():
    pid = p_row['athlete_id']
    tid = p_row['team_id']
    plup = lineup_agg[
        (lineup_agg['team_id'] == tid) &
        (lineup_agg['lineup_str'].str.contains(str(pid)))
    ]
    if len(plup) == 0:
        player_lineup_best.append({'athlete_id': pid, 'team_id': tid,
                                   'top_lineup_net_rtg': np.nan, 'top_lineup_min': np.nan, 'n_lineups': 0})
        continue
    best = plup.nlargest(1, 'lineup_min_est', keep='first')  # most-used lineup
    player_lineup_best.append({
        'athlete_id':       pid,
        'team_id':          tid,
        'n_lineups':        len(plup),
        'top_lineup_min':   best['lineup_min_est'].iloc[0],
        'top_lineup_net_rtg': best['lineup_net_rtg'].iloc[0],
    })

lineup_player_df = pd.DataFrame(player_lineup_best)

# ─────────────────────────────────────────────
# STEP 9: MERGE EVERYTHING
# ─────────────────────────────────────────────
print("Merging all metrics...")

final = agg_base.copy()

# Off-court ratings
off_df['off_off_rtg'] = np.where(off_df['off_poss_for']     > 5,
                                  off_df['off_pts_scored']  / off_df['off_poss_for']     * 100, np.nan)
off_df['off_def_rtg'] = np.where(off_df['off_poss_against'] > 5,
                                  off_df['off_pts_allowed'] / off_df['off_poss_against'] * 100, np.nan)
off_df['off_net_rtg'] = off_df['off_off_rtg'] - off_df['off_def_rtg']

final = final.merge(off_df[['athlete_id', 'team_id',
                              'off_off_rtg', 'off_def_rtg', 'off_net_rtg',
                              'off_plus_minus']], on=['athlete_id','team_id'], how='left')

# On/Off differential (the key number)
final['net_rtg_diff']  = final['on_net_rtg']  - final['off_net_rtg']
final['off_rtg_diff']  = final['on_off_rtg']  - final['off_off_rtg']
final['def_rtg_diff']  = final['on_def_rtg']  - final['off_def_rtg']

# Contextual splits
for df_split in [h1, h2, cls, cmp, blw, q1, q2]:
    final = final.merge(df_split, on=['athlete_id', 'team_id'], how='left')

# Lineup summaries
final = final.merge(lineup_player_df, on=['athlete_id', 'team_id'], how='left')


# Backward-compatible aliases used by downstream consumers
if 'on_min_est' in final.columns and 'on_minutes' not in final.columns:
    final['on_minutes'] = final['on_min_est']
if 'on_poss_for' in final.columns and 'poss_on' not in final.columns:
    final['poss_on'] = final['on_poss_for']

# ─────────────────────────────────────────────
# STEP 10: EXPORT
# ─────────────────────────────────────────────
# Round floats
float_cols = final.select_dtypes(include='float').columns
final[float_cols] = final[float_cols].round(2)

final.to_csv(PLAYER_ONOFF, index=False)

print(f"\n{'='*50}")
print(f"✓ player_onoff_metrics.csv → {len(final):,} players, {len(final.columns)} columns")
print(f"✓ lineup_stints_raw.csv   → {len(stints_df):,} stints")
print(f"\nTop 10 players by net rating differential:")
preview = final[['athlete_id','team_name','on_net_rtg','off_net_rtg','net_rtg_diff','on_min_est']].dropna(
    subset=['net_rtg_diff']).nlargest(10, 'net_rtg_diff')
print(preview.to_string(index=False))
print("Done.")
