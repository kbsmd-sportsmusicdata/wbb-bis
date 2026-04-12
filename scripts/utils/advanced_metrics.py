"""
WBB Scout Reports - Advanced Metrics (Play-by-Play Derived)
============================================================
Metrics that require play-by-play data for accurate calculation.

Data Source: wehoop-wbb-data PBP releases
Reference: https://github.com/sportsdataverse/wehoop-wbb-data

These metrics provide deeper insights than box score data alone:
- On/Off splits and net rating
- Shot zone efficiency
- Transition vs half-court splits
- Points off turnovers
- Second chance points
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# wehoop PBP data URL pattern
WEHOOP_PBP_BASE = "https://github.com/sportsdataverse/wehoop-wbb-data/releases/download"

# Play type classifications (based on ESPN type_text field)
SHOT_TYPES = {
    'layup': ['LayUpShot', 'layup', 'Layup'],
    'dunk': ['Dunk', 'dunk'],
    'jumper': ['JumpShot', 'jumper', 'Jumper', 'Jump Shot'],
    'three': ['Three Point', 'ThreePointShot', '3PT', 'three'],
    'free_throw': ['FreeThrow', 'Free Throw', 'free throw'],
}

TRANSITION_INDICATORS = [
    'fastbreak', 'fast break', 'transition', 'in transition',
    'coast to coast', 'outlet'
]

TURNOVER_TYPES = [
    'Turnover', 'turnover', 'steal', 'Steal', 'lost ball', 'bad pass',
    'traveling', 'offensive foul', '3 second', 'shot clock'
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pbp_data(season: int = 2025,
                  local_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load play-by-play data from wehoop releases or local file.

    Args:
        season: Season year (e.g., 2025 for 2024-25 season)
        local_path: Optional local file path

    Returns:
        DataFrame with PBP data
    """
    if local_path and Path(local_path).exists():
        print(f"Loading PBP from local file: {local_path}")
        return pd.read_parquet(local_path)

    # Try wehoop releases
    url_patterns = [
        f"{WEHOOP_PBP_BASE}/espn_womens_college_basketball_pbp/wbb_pbp_{season}.parquet",
        f"{WEHOOP_PBP_BASE}/wbb_pbp/wbb_pbp_{season}.parquet",
        f"{WEHOOP_PBP_BASE}/wbb_pbp/pbp_{season}.parquet",
    ]

    for url in url_patterns:
        try:
            print(f"Trying: {url}")
            df = pd.read_parquet(url)
            print(f"✓ Loaded {len(df)} PBP rows")
            return df
        except Exception as e:
            print(f"  ✗ {e}")

    print("ERROR: Could not load PBP data")
    return pd.DataFrame()


# =============================================================================
# ON/OFF CALCULATIONS
# =============================================================================

def calculate_on_off_rating(pbp_df: pd.DataFrame,
                            player_id: int,
                            team_id: int) -> Dict[str, float]:
    """
    Calculate On/Off ratings for a player.

    Requires PBP data with:
    - Lineup/substitution tracking
    - Score changes per possession

    Args:
        pbp_df: Play-by-play DataFrame
        player_id: Player's athlete_id
        team_id: Player's team_id

    Returns:
        Dictionary with on_ortg, on_drtg, off_ortg, off_drtg, on_off_diff
    """
    # Filter to relevant game(s)
    team_pbp = pbp_df[pbp_df['team_id'] == team_id].copy()

    if len(team_pbp) == 0:
        return {
            'on_ortg': np.nan, 'on_drtg': np.nan,
            'off_ortg': np.nan, 'off_drtg': np.nan,
            'on_off_diff': np.nan
        }

    # Check for lineup data
    if 'participants' not in team_pbp.columns and 'home_lineup' not in team_pbp.columns:
        print("Warning: PBP data missing lineup information for On/Off calculation")
        return {
            'on_ortg': np.nan, 'on_drtg': np.nan,
            'off_ortg': np.nan, 'off_drtg': np.nan,
            'on_off_diff': np.nan
        }

    # Determine if player is on court for each play
    # This depends on the specific PBP data structure
    # ESPN PBP may have 'participants' or separate lineup columns

    # Placeholder logic - actual implementation depends on data structure
    on_plays = team_pbp  # TODO: Filter to plays where player is on court
    off_plays = pd.DataFrame()  # TODO: Filter to plays where player is off court

    # Calculate ratings for on-court plays
    on_pts_scored = _sum_points_scored(on_plays, team_id)
    on_pts_allowed = _sum_points_allowed(on_plays, team_id)
    on_poss = _estimate_possessions_from_pbp(on_plays, team_id)

    # Calculate ratings for off-court plays
    off_pts_scored = _sum_points_scored(off_plays, team_id)
    off_pts_allowed = _sum_points_allowed(off_plays, team_id)
    off_poss = _estimate_possessions_from_pbp(off_plays, team_id)

    # Compute ratings
    on_ortg = 100 * on_pts_scored / on_poss if on_poss > 0 else 0
    on_drtg = 100 * on_pts_allowed / on_poss if on_poss > 0 else 0
    off_ortg = 100 * off_pts_scored / off_poss if off_poss > 0 else 0
    off_drtg = 100 * off_pts_allowed / off_poss if off_poss > 0 else 0

    on_net = on_ortg - on_drtg
    off_net = off_ortg - off_drtg

    return {
        'on_ortg': on_ortg,
        'on_drtg': on_drtg,
        'on_net': on_net,
        'off_ortg': off_ortg,
        'off_drtg': off_drtg,
        'off_net': off_net,
        'on_off_diff': on_net - off_net
    }


def calculate_lineup_stats(pbp_df: pd.DataFrame,
                           team_id: int,
                           min_possessions: int = 10) -> pd.DataFrame:
    """
    Calculate stats for all 5-man lineup combinations.

    Args:
        pbp_df: Play-by-play DataFrame
        team_id: Team ID to analyze
        min_possessions: Minimum possessions for inclusion

    Returns:
        DataFrame with lineup combinations and their stats
    """
    # Placeholder - actual implementation depends on PBP structure
    lineup_stats = []

    # Group by unique lineup combinations
    # Calculate: minutes, possessions, ORtg, DRtg, Net

    return pd.DataFrame(lineup_stats)


# =============================================================================
# SHOT ZONE EFFICIENCY
# =============================================================================

def calculate_shot_zone_efficiency(pbp_df: pd.DataFrame,
                                   player_id: Optional[int] = None,
                                   team_id: Optional[int] = None) -> Dict[str, Dict]:
    """
    Calculate shooting efficiency by shot zone/type.

    Shot zones (based on available data):
    - Paint (layups, dunks)
    - Mid-range (2PT jumpers)
    - Three-point
    - Free throws

    Args:
        pbp_df: Play-by-play DataFrame with shot data
        player_id: Optional player filter
        team_id: Optional team filter

    Returns:
        Dictionary with zone efficiency stats
    """
    df = pbp_df.copy()

    # Apply filters
    if player_id:
        df = df[df.get('athlete_id', df.get('shooter_id', -1)) == player_id]
    if team_id:
        df = df[df['team_id'] == team_id]

    # Classify shots by zone
    zones = {
        'paint': {'made': 0, 'attempted': 0, 'pps': 0.0},
        'midrange': {'made': 0, 'attempted': 0, 'pps': 0.0},
        'three': {'made': 0, 'attempted': 0, 'pps': 0.0},
        'free_throw': {'made': 0, 'attempted': 0, 'pps': 0.0},
    }

    # Check for type_text column (ESPN PBP format)
    if 'type_text' not in df.columns:
        print("Warning: PBP data missing type_text column for shot classification")
        return zones

    for _, play in df.iterrows():
        play_type = str(play.get('type_text', '')).lower()
        scoring_play = play.get('scoring_play', False)
        score_value = play.get('score_value', 0)

        # Classify by shot type
        if any(t.lower() in play_type for t in SHOT_TYPES['layup'] + SHOT_TYPES['dunk']):
            zone = 'paint'
            points = 2
        elif any(t.lower() in play_type for t in SHOT_TYPES['three']):
            zone = 'three'
            points = 3
        elif any(t.lower() in play_type for t in SHOT_TYPES['free_throw']):
            zone = 'free_throw'
            points = 1
        elif any(t.lower() in play_type for t in SHOT_TYPES['jumper']):
            zone = 'midrange'
            points = 2
        else:
            continue

        zones[zone]['attempted'] += 1
        if scoring_play or score_value > 0:
            zones[zone]['made'] += 1

    # Calculate efficiency
    for zone in zones:
        if zones[zone]['attempted'] > 0:
            zones[zone]['fg_pct'] = zones[zone]['made'] / zones[zone]['attempted']
            if zone == 'three':
                zones[zone]['pps'] = 3 * zones[zone]['fg_pct']
            elif zone == 'free_throw':
                zones[zone]['pps'] = 1 * zones[zone]['fg_pct']
            else:
                zones[zone]['pps'] = 2 * zones[zone]['fg_pct']

    return zones


# =============================================================================
# ASSISTED VS UNASSISTED
# =============================================================================

def calculate_assisted_rate(pbp_df: pd.DataFrame,
                            player_id: Optional[int] = None,
                            team_id: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate assisted vs unassisted field goal rates.

    Args:
        pbp_df: Play-by-play DataFrame
        player_id: Optional player filter
        team_id: Optional team filter

    Returns:
        Dictionary with assisted/unassisted rates
    """
    df = pbp_df.copy()

    if player_id:
        df = df[df.get('athlete_id', df.get('shooter_id', -1)) == player_id]
    if team_id:
        df = df[df['team_id'] == team_id]

    # Filter to made field goals
    made_fgs = df[
        (df.get('scoring_play', False) == True) &
        (df.get('type_text', '').str.lower().str.contains('shot|layup|dunk', na=False))
    ]

    if len(made_fgs) == 0:
        return {
            'total_fgm': 0,
            'assisted_fgm': 0,
            'unassisted_fgm': 0,
            'assisted_rate': 0.0,
            'unassisted_rate': 0.0
        }

    # Check for assist indicator
    # ESPN PBP typically includes assist info in 'text' or separate column
    assisted = made_fgs[
        made_fgs.get('text', '').str.lower().str.contains('assist', na=False) |
        made_fgs.get('assist', False) == True
    ]

    total_fgm = len(made_fgs)
    assisted_fgm = len(assisted)
    unassisted_fgm = total_fgm - assisted_fgm

    return {
        'total_fgm': total_fgm,
        'assisted_fgm': assisted_fgm,
        'unassisted_fgm': unassisted_fgm,
        'assisted_rate': assisted_fgm / total_fgm if total_fgm > 0 else 0.0,
        'unassisted_rate': unassisted_fgm / total_fgm if total_fgm > 0 else 0.0
    }


# =============================================================================
# TRANSITION VS HALF-COURT
# =============================================================================

def calculate_transition_efficiency(pbp_df: pd.DataFrame,
                                    team_id: Optional[int] = None) -> Dict[str, Dict]:
    """
    Calculate efficiency in transition vs half-court possessions.

    Args:
        pbp_df: Play-by-play DataFrame
        team_id: Optional team filter

    Returns:
        Dictionary with transition and half-court stats
    """
    df = pbp_df.copy()

    if team_id:
        df = df[df['team_id'] == team_id]

    results = {
        'transition': {
            'possessions': 0, 'points': 0, 'ppp': 0.0,
            'fga': 0, 'fgm': 0, 'efg_pct': 0.0
        },
        'halfcourt': {
            'possessions': 0, 'points': 0, 'ppp': 0.0,
            'fga': 0, 'fgm': 0, 'efg_pct': 0.0
        }
    }

    # Classify possessions by tempo
    # This requires sequence_number or time elapsed data
    if 'text' not in df.columns:
        return results

    for _, play in df.iterrows():
        play_text = str(play.get('text', '')).lower()
        is_transition = any(ind in play_text for ind in TRANSITION_INDICATORS)

        category = 'transition' if is_transition else 'halfcourt'

        # Count shots
        play_type = str(play.get('type_text', '')).lower()
        if 'shot' in play_type or 'layup' in play_type or 'dunk' in play_type:
            results[category]['fga'] += 1
            if play.get('scoring_play', False):
                results[category]['fgm'] += 1
                score_val = play.get('score_value', 2)
                results[category]['points'] += score_val

    # Calculate efficiency
    for cat in ['transition', 'halfcourt']:
        if results[cat]['fga'] > 0:
            fg_pct = results[cat]['fgm'] / results[cat]['fga']
            results[cat]['efg_pct'] = fg_pct  # Simplified, would need 3PM for true eFG
            results[cat]['ppp'] = results[cat]['points'] / results[cat]['fga']

    return results


# =============================================================================
# POINTS OFF TURNOVERS
# =============================================================================

def calculate_points_off_turnovers(pbp_df: pd.DataFrame,
                                   team_id: int) -> Dict[str, float]:
    """
    Calculate points scored off opponent turnovers.

    Args:
        pbp_df: Play-by-play DataFrame
        team_id: Team ID to calculate for

    Returns:
        Dictionary with POT stats
    """
    df = pbp_df.copy()

    # Find opponent turnovers
    opp_turnovers = df[
        (df['team_id'] != team_id) &
        (df.get('type_text', '').str.lower().str.contains('|'.join(TURNOVER_TYPES).lower(), na=False))
    ]

    # Find scoring plays immediately after turnovers
    # This requires sequence tracking
    pot_points = 0
    pot_possessions = len(opp_turnovers)

    if 'sequence_number' in df.columns:
        for _, tov_play in opp_turnovers.iterrows():
            game_id = tov_play.get('game_id')
            seq = tov_play.get('sequence_number')

            # Look for next scoring play by team in same sequence
            next_plays = df[
                (df['game_id'] == game_id) &
                (df['sequence_number'] == seq + 1) &
                (df['team_id'] == team_id) &
                (df.get('scoring_play', False) == True)
            ]

            if len(next_plays) > 0:
                pot_points += next_plays['score_value'].sum()

    return {
        'points_off_turnovers': pot_points,
        'turnover_opportunities': pot_possessions,
        'pot_ppp': pot_points / pot_possessions if pot_possessions > 0 else 0.0
    }


# =============================================================================
# SECOND CHANCE POINTS
# =============================================================================

def calculate_second_chance_points(pbp_df: pd.DataFrame,
                                   team_id: int) -> Dict[str, float]:
    """
    Calculate second chance points (after offensive rebounds).

    Args:
        pbp_df: Play-by-play DataFrame
        team_id: Team ID to calculate for

    Returns:
        Dictionary with second chance stats
    """
    df = pbp_df.copy()
    df = df[df['team_id'] == team_id]

    # Find offensive rebounds
    off_rebounds = df[
        df.get('type_text', '').str.lower().str.contains('offensive rebound|off reb', na=False)
    ]

    scp_points = 0
    scp_possessions = len(off_rebounds)

    if 'sequence_number' in df.columns:
        for _, oreb_play in off_rebounds.iterrows():
            game_id = oreb_play.get('game_id')
            seq = oreb_play.get('sequence_number')

            # Look for scoring in same or next sequence
            scoring_plays = df[
                (df['game_id'] == game_id) &
                (df['sequence_number'].isin([seq, seq + 1])) &
                (df.get('scoring_play', False) == True)
            ]

            if len(scoring_plays) > 0:
                scp_points += scoring_plays['score_value'].sum()

    return {
        'second_chance_points': scp_points,
        'offensive_rebounds': scp_possessions,
        'scp_ppp': scp_points / scp_possessions if scp_possessions > 0 else 0.0
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _sum_points_scored(pbp_df: pd.DataFrame, team_id: int) -> int:
    """Sum points scored by team in PBP data."""
    scoring_plays = pbp_df[
        (pbp_df['team_id'] == team_id) &
        (pbp_df.get('scoring_play', False) == True)
    ]
    return scoring_plays.get('score_value', pd.Series([0])).sum()


def _sum_points_allowed(pbp_df: pd.DataFrame, team_id: int) -> int:
    """Sum points allowed by team in PBP data."""
    opp_scoring = pbp_df[
        (pbp_df['team_id'] != team_id) &
        (pbp_df.get('scoring_play', False) == True)
    ]
    return opp_scoring.get('score_value', pd.Series([0])).sum()


def _estimate_possessions_from_pbp(pbp_df: pd.DataFrame, team_id: int) -> int:
    """Estimate possessions from PBP sequence data."""
    if 'sequence_number' in pbp_df.columns:
        team_sequences = pbp_df[pbp_df['team_id'] == team_id]['sequence_number'].nunique()
        return team_sequences
    return len(pbp_df[pbp_df['team_id'] == team_id]) // 5  # Rough estimate


# =============================================================================
# BATCH CALCULATION
# =============================================================================

def calculate_all_pbp_metrics(pbp_df: pd.DataFrame,
                              player_id: Optional[int] = None,
                              team_id: Optional[int] = None) -> Dict:
    """
    Calculate all available PBP-derived metrics.

    Args:
        pbp_df: Play-by-play DataFrame
        player_id: Optional player to analyze
        team_id: Optional team to analyze

    Returns:
        Dictionary with all calculated metrics
    """
    results = {}

    if team_id:
        results['shot_zones'] = calculate_shot_zone_efficiency(pbp_df, player_id, team_id)
        results['assisted_rate'] = calculate_assisted_rate(pbp_df, player_id, team_id)
        results['transition'] = calculate_transition_efficiency(pbp_df, team_id)
        results['points_off_turnovers'] = calculate_points_off_turnovers(pbp_df, team_id)
        results['second_chance'] = calculate_second_chance_points(pbp_df, team_id)

    if player_id and team_id:
        results['on_off'] = calculate_on_off_rating(pbp_df, player_id, team_id)

    return results


# =============================================================================
# DATA EXPORT
# =============================================================================

def export_pbp_metrics_summary(pbp_df: pd.DataFrame,
                               team_id: int,
                               output_path: Path = None) -> pd.DataFrame:
    """
    Export a summary of PBP-derived metrics for a team.

    Args:
        pbp_df: Play-by-play DataFrame
        team_id: Team to analyze
        output_path: Optional path to save CSV

    Returns:
        DataFrame with metric summary
    """
    metrics = calculate_all_pbp_metrics(pbp_df, team_id=team_id)

    # Flatten to DataFrame
    rows = []

    # Shot zones
    for zone, stats in metrics.get('shot_zones', {}).items():
        rows.append({
            'category': 'shot_zone',
            'metric': zone,
            'value': stats.get('pps', 0),
            'made': stats.get('made', 0),
            'attempted': stats.get('attempted', 0)
        })

    # Transition
    for cat, stats in metrics.get('transition', {}).items():
        rows.append({
            'category': 'tempo',
            'metric': cat,
            'value': stats.get('ppp', 0),
            'made': stats.get('fgm', 0),
            'attempted': stats.get('fga', 0)
        })

    # Points off turnovers
    pot = metrics.get('points_off_turnovers', {})
    rows.append({
        'category': 'secondary',
        'metric': 'points_off_turnovers',
        'value': pot.get('pot_ppp', 0),
        'made': pot.get('points_off_turnovers', 0),
        'attempted': pot.get('turnover_opportunities', 0)
    })

    # Second chance
    scp = metrics.get('second_chance', {})
    rows.append({
        'category': 'secondary',
        'metric': 'second_chance_points',
        'value': scp.get('scp_ppp', 0),
        'made': scp.get('second_chance_points', 0),
        'attempted': scp.get('offensive_rebounds', 0)
    })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Saved PBP metrics to {output_path}")

    return df
