#!/usr/bin/env python3
"""
Build WBB Conference History visualization (2022-2026)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from config import KAGGLE_DIR, STATIC_DIR, DASHBOARD_CONF

import pandas as pd
import json
import re
import numpy as np
from collections import defaultdict

# Base paths resolved via config.py
MAIN_DIR = STATIC_DIR

# Load data
seeds_df = pd.read_csv(KAGGLE_DIR / "WNCAATourneySeeds.csv")
results_df = pd.read_csv(KAGGLE_DIR / "WNCAATourneyCompactResults.csv")
teams_df = pd.read_csv(KAGGLE_DIR / "WTeams.csv")
team_conferences_df = pd.read_csv(KAGGLE_DIR / "WTeamConferences.csv")
spellings_df = pd.read_csv(KAGGLE_DIR / "WTeamSpellings.csv")
conferences_df = pd.read_csv(KAGGLE_DIR / "Conferences.csv")
bracket_2026_df = pd.read_csv(MAIN_DIR / "tournament_bracket.csv")

print("Data loaded successfully")

# Create team name to ID mapping
team_name_to_id = {}
for _, row in teams_df.iterrows():
    team_name_to_id[row['TeamName'].lower().strip()] = row['TeamID']

# Add alternate spellings
for _, row in spellings_df.iterrows():
    team_name_to_id[row['TeamNameSpelling'].lower().strip()] = row['TeamID']

# Normalize special cases
team_name_to_id['southern california'] = team_name_to_id.get('usc', None)
team_name_to_id['w illinois'] = team_name_to_id.get('western illinois', None)
team_name_to_id['ca baptist'] = team_name_to_id.get('california baptist', None)
team_name_to_id['s dakota st'] = team_name_to_id.get('south dakota state', None)
team_name_to_id['fdu'] = team_name_to_id.get('fairleigh dickinson', None)
team_name_to_id['miami oh'] = team_name_to_id.get('miami (oh)', None)
team_name_to_id['uc san diego'] = team_name_to_id.get('uc san diego', None)
team_name_to_id['oklahoma st'] = team_name_to_id.get('oklahoma state', None)

print("\nTeam name mapping sample:")
for key in list(team_name_to_id.keys())[:5]:
    print(f"  {key} -> {team_name_to_id[key]}")

# Conference name normalization
def normalize_conference(abbrev):
    """Normalize conference abbreviations"""
    if pd.isna(abbrev) or abbrev is None:
        return 'Other'
    abbrev = str(abbrev).lower().strip()

    power_confs = {
        'big_ten': 'Big Ten',
        'sec': 'SEC',
        'acc': 'ACC',
        'big_twelve': 'Big 12',
        'big_12': 'Big 12',
        'big_east': 'Big East',
        'pac_twelve': 'Pac-12',
        'pac_ten': 'Pac-12',
        'a_ten': 'Atlantic 10',
    }

    for key, val in power_confs.items():
        if key == abbrev:
            return val

    # Check if in the normalized forms already
    if abbrev in ['big ten', 'sec', 'acc', 'big 12', 'big east', 'pac-12', 'atlantic 10']:
        return abbrev.title() if abbrev != 'sec' else 'SEC'

    return 'Other'

# Get conference for a team in a given season
def get_team_conference(team_id, season):
    """Get the conference for a team in a given season"""
    matches = team_conferences_df[
        (team_conferences_df['TeamID'] == team_id) &
        (team_conferences_df['Season'] == season)
    ]
    if len(matches) > 0:
        return normalize_conference(matches.iloc[0]['ConfAbbrev'])
    return 'Other'

def get_teams_at_round(season, day_nums):
    """Get all team IDs that won games in the given day numbers"""
    relevant_results = results_df[
        (results_df['Season'] == season) &
        (results_df['DayNum'].isin(day_nums))
    ]
    return set(relevant_results['WTeamID'].unique())

# Process 2022-2025 (Kaggle data)
year_data = {}

for season in range(2022, 2026):
    print(f"\n=== Processing {season} ===")

    # Get all seeded teams (Field of 68)
    seeds_this_season = seeds_df[seeds_df['Season'] == season]
    field_of_68_ids = set(seeds_this_season['TeamID'].unique())
    print(f"Field of 68: {len(field_of_68_ids)} teams")

    # Get advancing teams
    r32_ids = get_teams_at_round(season, [137, 138])
    s16_ids = get_teams_at_round(season, [139, 140])
    e8_ids = get_teams_at_round(season, [144, 145])
    f4_ids = get_teams_at_round(season, [146, 147])
    champ_ids = get_teams_at_round(season, [153])

    print(f"R32: {len(r32_ids)}, S16: {len(s16_ids)}, E8: {len(e8_ids)}, F4: {len(f4_ids)}, Champ: {len(champ_ids)}")

    # Build conference breakdowns for each round
    rounds = {
        'Field of 68': field_of_68_ids,
        'R32': r32_ids,
        'S16': s16_ids,
        'E8': e8_ids,
        'F4': f4_ids,
    }

    conf_breakdown = {}
    for round_name, team_ids in rounds.items():
        conf_counts = defaultdict(int)
        teams_by_conf = defaultdict(list)

        for team_id in team_ids:
            conf = get_team_conference(team_id, season)
            conf_counts[conf] += 1
            team_name = teams_df[teams_df['TeamID'] == team_id]['TeamName'].iloc[0] if len(teams_df[teams_df['TeamID'] == team_id]) > 0 else f"Team {team_id}"
            teams_by_conf[conf].append({
                'name': team_name,
                'id': team_id,
                'seed': seeds_this_season[seeds_this_season['TeamID'] == team_id]['Seed'].iloc[0] if len(seeds_this_season[seeds_this_season['TeamID'] == team_id]) > 0 else 'N/A'
            })

        conf_breakdown[round_name] = {
            'counts': dict(conf_counts),
            'teams': {conf: sorted(teams, key=lambda x: x['name']) for conf, teams in teams_by_conf.items()}
        }

    year_data[season] = conf_breakdown

    # Print summary
    for round_name, data in conf_breakdown.items():
        print(f"  {round_name}: {data['counts']}")

# Process 2026 manually
print("\n=== Processing 2026 ===")
bracket_2026 = bracket_2026_df.copy()

# Build 2026 team mapping
def find_team_id_2026(team_name):
    """Find TeamID for a 2026 bracket team"""
    search_name = team_name.lower().strip()

    # Direct match
    if search_name in team_name_to_id:
        return team_name_to_id[search_name]

    # Try partial matches
    for key, team_id in team_name_to_id.items():
        if key in search_name or search_name in key:
            return team_id

    return None

# Get unique teams from bracket
bracket_teams = bracket_2026['team_location'].unique()
bracket_team_confs = {}
for team_name in bracket_teams:
    conf = bracket_2026[bracket_2026['team_location'] == team_name]['conference'].iloc[0]
    bracket_team_confs[team_name] = normalize_conference(conf)

print(f"2026 Field of 68: {len(bracket_teams)} teams")
print(f"Sample conferences: {dict(list(bracket_team_confs.items())[:5])}")

# Build 2026 rounds from provided data
teams_2026_r32 = ['UConn', 'North Carolina', 'Notre Dame', 'Vanderbilt', 'UCLA', 'Minnesota', 'Duke', 'LSU',
                  'Texas', 'Kentucky', 'Louisville', 'Michigan', 'South Carolina', 'Oklahoma', 'TCU', 'Virginia',
                  'Iowa', 'Maryland', 'Ohio State', 'Illinois', 'Syracuse', 'Baylor', 'Texas Tech',
                  'Oklahoma State', 'Ole Miss', 'Oregon', 'West Virginia', 'Alabama', 'NC State', 'Washington', 'Michigan State']
teams_2026_s16 = ['UConn', 'North Carolina', 'Notre Dame', 'Vanderbilt', 'UCLA', 'Minnesota', 'Duke', 'LSU',
                  'Texas', 'Kentucky', 'Louisville', 'Michigan', 'South Carolina', 'Oklahoma', 'TCU', 'Virginia']
teams_2026_e8 = ['UConn', 'Notre Dame', 'UCLA', 'Duke', 'Texas', 'Michigan', 'South Carolina', 'TCU']
teams_2026_f4 = ['UConn', 'UCLA', 'Texas', 'South Carolina']

def get_confs_2026(team_list):
    conf_counts = defaultdict(int)
    teams_by_conf = defaultdict(list)

    for team_name in team_list:
        conf = bracket_team_confs.get(team_name, 'Other')
        conf_counts[conf] += 1
        teams_by_conf[conf].append({
            'name': team_name,
            'seed': bracket_2026[bracket_2026['team_location'] == team_name]['seed'].iloc[0]
        })

    return {
        'counts': dict(conf_counts),
        'teams': {conf: sorted(teams, key=lambda x: x['name']) for conf, teams in teams_by_conf.items()}
    }

field_2026_confs = defaultdict(int)
for team in bracket_teams:
    conf = bracket_team_confs[team]
    field_2026_confs[conf] += 1

year_data[2026] = {
    'Field of 68': {
        'counts': dict(field_2026_confs),
        'teams': {conf: [{'name': t, 'seed': bracket_2026[bracket_2026['team_location'] == t]['seed'].iloc[0]}
                        for t in bracket_teams if bracket_team_confs[t] == conf]
                 for conf in bracket_team_confs.values()}
    },
    'R32': get_confs_2026(teams_2026_r32),
    'S16': get_confs_2026(teams_2026_s16),
    'E8': get_confs_2026(teams_2026_e8),
    'F4': get_confs_2026(teams_2026_f4),
}

# Print all breakdowns
print("\n" + "="*80)
print("CONFERENCE BREAKDOWN SUMMARY")
print("="*80)

for season in sorted(year_data.keys()):
    print(f"\n{season}")
    print("-" * 80)
    for round_name in ['Field of 68', 'R32', 'S16', 'E8', 'F4']:
        data = year_data[season].get(round_name, {})
        counts = data.get('counts', {})
        print(f"  {round_name:12} | {', '.join(f'{conf}: {count}' for conf, count in sorted(counts.items()))}")

# Calculate survival rates
print("\n" + "="*80)
print("SURVIVAL RATES (% of teams advancing)")
print("="*80)

for season in sorted(year_data.keys()):
    print(f"\n{season}")
    print("-" * 80)

    field_count = sum(year_data[season]['Field of 68']['counts'].values())

    for conf in sorted(set().union(*[year_data[season].get(r, {}).get('counts', {}).keys()
                                     for r in ['Field of 68', 'R32', 'S16', 'E8', 'F4']])):
        percentages = []
        for round_name in ['Field of 68', 'R32', 'S16', 'E8', 'F4']:
            count = year_data[season].get(round_name, {}).get('counts', {}).get(conf, 0)
            if count > 0:
                pct = count
                percentages.append(f"{round_name}: {count}")

        if percentages:
            print(f"  {conf:15} | {', '.join(percentages)}")

# Convert numpy types to native Python types
def convert_to_native(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, str):
        return obj
    else:
        return obj

# Save data to JSON
output_json = {
    'years': {}
}

for season in sorted(year_data.keys()):
    season_data = year_data[season]
    output_json['years'][str(season)] = {}

    for round_name in ['Field of 68', 'R32', 'S16', 'E8', 'F4']:
        round_data = season_data.get(round_name, {})
        output_json['years'][str(season)][round_name] = convert_to_native(round_data)

# Write JSON for embedding
json_str = json.dumps(output_json, indent=2)
print(f"\n\nJSON data size: {len(json_str)} bytes")

# Save to temp file for reference
with open('/tmp/wbb_conf_data.json', 'w') as f:
    json.dump(output_json, f, indent=2)

print(f"JSON saved to /tmp/wbb_conf_data.json")
