"""
build_player_box_json.py
========================
Generates player_box_with_game_type.json — a game-log JSON for all
players in the parquet, with every row tagged by game_type:

  game_type values:
    "tournament"      — season_type == 3
    "conference"      — season_type == 2 + both teams in same conference
    "regular_season"  — season_type == 2, non-conference game
                        (also used where opponent conference is unknown)

Conference identification strategy:
  1. Pull team_location → conf from dashboard_slim_k6_enriched.json (8 confs covered)
  2. Supplement with hardcoded full membership for all D1 major conferences
  3. Conference game = opponent_team_location resolves to same conf as player's team

Output: analysis/player_archetypes/Player Archetypes/player_box_with_game_type.json
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

PLAYER_BOX_CANDIDATES = [
    Path(os.getenv("PLAYER_BOX_FILE", "")),
    REPO_ROOT / "data" / "raw" / "player_box_2026_final.parquet",
    REPO_ROOT / "data" / "player_box_2026.parquet",
]
JSON_IN_CANDIDATES = [
    Path(os.getenv("DASHBOARD_SLIM_JSON", "")),
    SCRIPT_DIR / "dashboard_slim_k6_enriched.json",
]
OUT_PATH = Path(
    os.getenv("PLAYER_BOX_JSON_OUT", str(SCRIPT_DIR / "player_box_with_game_type.json"))
)

PLAYER_BOX = next((p for p in PLAYER_BOX_CANDIDATES if p and p.is_file()), None)
JSON_IN = next((p for p in JSON_IN_CANDIDATES if p and p.is_file()), None)
if PLAYER_BOX is None or JSON_IN is None:
    raise FileNotFoundError(
        "Required input file missing. Set PLAYER_BOX_FILE/DASHBOARD_SLIM_JSON env vars "
        "or place files at expected default paths."
    )

# ── Comprehensive 2025-26 WBB conference membership ─────────────────────────
# team_location strings as they appear in ESPN data
CONFERENCE_MAP = {}

def _reg(conf, teams):
    for t in teams:
        CONFERENCE_MAP[t] = conf

_reg("SEC", [
    "Alabama","Arkansas","Auburn","Florida","Georgia","Kentucky","LSU",
    "Mississippi State","Missouri","Ole Miss","South Carolina","Tennessee",
    "Texas","Texas A&M","Vanderbilt","Oklahoma"
])
_reg("Big Ten", [
    "Illinois","Indiana","Iowa","Maryland","Michigan","Michigan State",
    "Minnesota","Nebraska","Northwestern","Ohio State","Oregon","Penn State",
    "Purdue","Rutgers","UCLA","USC","Washington","Wisconsin"
])
_reg("ACC", [
    "Boston College","California","Clemson","Duke","Florida State",
    "Georgia Tech","Louisville","Miami","NC State","North Carolina",
    "Notre Dame","Pittsburgh","SMU","Stanford","Syracuse",
    "Virginia","Virginia Tech","Wake Forest","Cal"
])
_reg("Big 12", [
    "Arizona","Arizona State","Baylor","BYU","Cincinnati","Colorado",
    "Houston","Iowa State","Kansas","Kansas State","Oklahoma State",
    "TCU","Texas Tech","UCF","Utah","West Virginia"
])
_reg("Big East", [
    "Butler","Connecticut","Creighton","DePaul","Georgetown",
    "Marquette","Providence","Seton Hall","St. John's","Villanova","Xavier","UConn"
])
_reg("Mountain West", [
    "Air Force","Boise State","Colorado State","Fresno State","Hawaii",
    "Nevada","New Mexico","San Diego State","UNLV","Utah State","Wyoming"
])
_reg("American Athletic", [
    "Charlotte","East Carolina","FAU","Memphis","North Texas",
    "Rice","South Florida","Temple","Tulane","Tulsa","UAB","Wichita State",
    "Florida Atlantic"
])
_reg("Atlantic 10", [
    "Davidson","Dayton","Duquesne","Fordham","George Mason","George Washington",
    "La Salle","Loyola Chicago","Massachusetts","Rhode Island","Richmond",
    "Saint Joseph's","Saint Louis","VCU"
])
_reg("Sun Belt", [
    "Appalachian State","Arkansas State","Coastal Carolina","Georgia Southern",
    "Georgia State","James Madison","Louisiana","Marshall","Old Dominion",
    "South Alabama","Southern Miss","Texas State","Troy","UL Monroe"
])
_reg("Conference USA", [
    "FIU","Jacksonville State","Liberty","Louisiana Tech","Middle Tennessee",
    "New Mexico State","Sam Houston","UTEP","Western Kentucky"
])
_reg("Missouri Valley", [
    "Belmont","Bradley","Drake","Evansville","Illinois State","Indiana State",
    "Loyola","Missouri State","Northern Iowa","Southern Illinois","UIC",
    "Valparaiso"
])
_reg("Pac-12", ["Oregon State","Washington State"])  # remaining Pac-12 holdovers
_reg("West Coast", [
    "BYU","Gonzaga","Loyola Marymount","Pacific","Pepperdine","Portland",
    "San Diego","San Francisco","Santa Clara","St. Mary's"
])
_reg("Ivy League", [
    "Brown","Columbia","Cornell","Dartmouth","Harvard","Penn","Princeton","Yale"
])
_reg("Patriot League", [
    "American","Army","Boston University","Bucknell","Colgate","Holy Cross",
    "Lafayette","Lehigh","Navy","Loyola Maryland"
])
_reg("Atlantic Sun", [
    "Austin Peay","Bellarmine","Eastern Kentucky","Florida Gulf Coast",
    "Jacksonville","Kennesaw State","Lipscomb","North Alabama","Northern Kentucky",
    "Queens","Stetson","UTRGV"
])
_reg("Horizon League", [
    "Cleveland State","Detroit Mercy","Green Bay","IUPUI","Milwaukee",
    "Northern Kentucky","Oakland","Purdue Fort Wayne","Robert Morris",
    "Wright State","Youngstown State"
])
_reg("MAAC", [
    "Canisius","Fairfield","Iona","Manhattan","Marist","Monmouth",
    "Niagara","Quinnipiac","Rider","Saint Peter's","Siena"
])
_reg("MAC", [
    "Akron","Ball State","Bowling Green","Buffalo","Central Michigan",
    "Eastern Michigan","Kent State","Miami (OH)","Northern Illinois",
    "Ohio","Toledo","Western Michigan"
])
_reg("Mountain West", [  # already registered above, just alias
])
_reg("Northeast", [
    "Central Connecticut","Fairleigh Dickinson","LIU","Long Island University",
    "Merrimack","Sacred Heart","Saint Francis","Stonehill","Wagner"
])
_reg("Ohio Valley", [
    "Eastern Illinois","Morehead State","Murray State","Southeast Missouri State",
    "Tennessee-Martin","Tennessee State","Tennessee Tech"
])
_reg("Southern", [
    "Chattanooga","East Tennessee State","Furman","Mercer","Samford",
    "UNC Greensboro","VMI","Western Carolina","Wofford"
])
_reg("Southland", [
    "Houston Baptist","Lamar","McNeese","Nicholls","Northwestern State",
    "Southeastern Louisiana","Texas A&M-Commerce","UIW"
])
_reg("SWAC", [
    "Alabama A&M","Alabama State","Alcorn State","Arkansas-Pine Bluff",
    "Bethune-Cookman","Florida A&M","Grambling","Jackson State",
    "Mississippi Valley State","Prairie View A&M","Southern","Texas Southern"
])
_reg("MEAC", [
    "Coppin State","Delaware State","Howard","Maryland-Eastern Shore",
    "Morgan State","Norfolk State","North Carolina A&T","North Carolina Central",
    "South Carolina State"
])
_reg("Big South", [
    "Campbell","Charleston Southern","Elon","Gardner-Webb","High Point",
    "Longwood","Presbyterians","Radford","UNC Asheville","Winthrop"
])
_reg("CAA", [
    "Campbell","Charleston","Delaware","Drexel","Elon","Hofstra",
    "Monmouth","Northeast","Stony Brook","Towson","UNC Wilmington","William & Mary"
])
_reg("WAC", [
    "Cal Baptist","Dixie State","Abilene Christian","Grand Canyon",
    "Sacramento State","Southern Utah","Utah Tech","Utah Valley","Portland State"
])
_reg("Summit League", [
    "Denver","Kansas City","North Dakota","North Dakota State","Oral Roberts",
    "Omaha","South Dakota","South Dakota State","St. Thomas","Western Illinois"
])
_reg("Big West", [
    "Cal Poly","Cal State Bakersfield","Cal State Fullerton","Cal State Northridge",
    "Long Beach State","UC Davis","UC Irvine","UC Riverside","UC San Diego",
    "UC Santa Barbara","UCSB"
])
_reg("America East", [
    "Albany","Binghamton","Hartford","Maine","Maryland-Baltimore County",
    "New Hampshire","NJIT","Stony Brook","UMass Lowell","Vermont"
])
_reg("Colonial Athletic", [
    "Drexel","Elon","Hampton","Hofstra","Monmouth","Northeastern",
    "Stony Brook","Towson","William & Mary"
])

# ── Load JSON: conf cross-reference + archetype lookup ───────────────────────
with open(JSON_IN) as f:
    slim = json.load(f)

# Cluster id → role name
CL_NAMES = {int(k): v['name'] for k, v in slim.get('cl', {}).items()}

# Override/supplement conference map with JSON ground truth
for p in slim['p']:
    if p.get('aid') and p.get('conf') and p.get('tm'):
        CONFERENCE_MAP[p['tm']] = p['conf']

# Build aid → {athlete_cluster, athlete_role} lookup
# Covers 393 Top-50 NET players; all others get null
archetype_map = {}
for p in slim['p']:
    aid = p.get('aid')
    if aid is not None:
        cl = p.get('cl')
        archetype_map[int(aid)] = {
            'athlete_cluster': int(cl) if cl is not None else None,
            'athlete_role':    CL_NAMES.get(int(cl)) if cl is not None else None
        }

print(f"Conference map entries: {len(CONFERENCE_MAP)}")
print(f"Archetype map entries:  {len(archetype_map)}")

# ── Load parquet ─────────────────────────────────────────────────────────────
print("Loading parquet...")
pb = pd.read_parquet(PLAYER_BOX)

# Include RS + tournament; exclude preseason
pb = pb[pb['season_type'].isin([2, 3])].copy()
print(f"  Total rows (RS+TOURN): {len(pb):,}")

# ── Tag game_type ────────────────────────────────────────────────────────────
print("Classifying games...")
team_conf = pb["team_location"].map(CONFERENCE_MAP)
opp_conf = pb["opponent_team_location"].map(CONFERENCE_MAP)
is_tournament = pb["season_type"] == 3
is_conference = (
    (pb["season_type"] == 2)
    & team_conf.notna()
    & opp_conf.notna()
    & (team_conf == opp_conf)
)
pb["game_type"] = np.select(
    [is_tournament, is_conference],
    ["tournament", "conference"],
    default="regular_season",
)

dist = pb['game_type'].value_counts()
print(f"\ngame_type distribution:")
for k, v in dist.items():
    print(f"  {k}: {v:,} rows")

# ── Select output columns ────────────────────────────────────────────────────
KEEP_COLS = [
    'game_id', 'game_date', 'season_type', 'game_type',
    'athlete_id', 'athlete_display_name', 'athlete_jersey',
    'athlete_position_abbreviation',
    'team_id', 'team_location', 'team_name',
    'home_away', 'team_winner', 'team_score', 'opponent_team_score',
    'opponent_team_id', 'opponent_team_location', 'opponent_team_name',
    'starter', 'did_not_play', 'minutes',
    'field_goals_made', 'field_goals_attempted',
    'three_point_field_goals_made', 'three_point_field_goals_attempted',
    'free_throws_made', 'free_throws_attempted',
    'offensive_rebounds', 'defensive_rebounds', 'rebounds',
    'assists', 'steals', 'blocks', 'turnovers', 'fouls', 'points'
]
pb_out = pb[KEEP_COLS].copy()

# Coerce types for JSON serialisation
pb_out['game_id']     = pb_out['game_id'].astype(str)
pb_out['game_date']   = pb_out['game_date'].astype(str)
pb_out['athlete_id']  = pb_out['athlete_id'].where(pb_out['athlete_id'].notna(), None)
pb_out['team_id']     = pb_out['team_id'].astype(str)
pb_out['opponent_team_id'] = pb_out['opponent_team_id'].astype(str)
pb_out['season_type'] = pb_out['season_type'].astype(int)
pb_out['team_score']  = pd.to_numeric(pb_out['team_score'], errors='coerce')
pb_out['opponent_team_score'] = pd.to_numeric(pb_out['opponent_team_score'], errors='coerce')

# Convert to records
print("\nConverting to JSON records...")
import math
records = pb_out.where(pb_out.notna(), None).to_dict(orient='records')

# ── Join archetype fields onto every row ──────────────────────────────────────
# athlete_cluster (int 0–5) and athlete_role (string) for Top-50 dashboard players.
# All other athletes get null — this is expected and documented in meta.
print("Joining archetype fields...")
arch_patched = 0
for row in records:
    raw_aid = row.get('athlete_id')
    if raw_aid is None or (isinstance(raw_aid, float) and math.isnan(raw_aid)):
        aid_int = None
    else:
        try:
            aid_int = int(raw_aid)
        except (ValueError, TypeError):
            aid_int = None
    info = archetype_map.get(aid_int, {'athlete_cluster': None, 'athlete_role': None})
    row['athlete_cluster'] = info['athlete_cluster']
    row['athlete_role']    = info['athlete_role']
    if info['athlete_cluster'] is not None:
        arch_patched += 1

print(f"  Rows with archetype: {arch_patched:,} / {len(records):,}")

# Build output structure
output = {
    "meta": {
        "generated": "2026-03-24",
        "season": 2026,
        "source": "ESPN via sportradar / hoopR parquet",
        "season_types_included": [2, 3],
        "game_type_legend": {
            "tournament":     "season_type==3 (NCAA Tournament / postseason)",
            "conference":     "season_type==2, both teams confirmed same conference",
            "regular_season": "season_type==2, non-conference OR opponent conf unknown"
        },
        "archetype_fields": {
            "athlete_cluster": "GMM cluster id 0-5; null for non-dashboard players",
            "athlete_role":    "Role archetype name (e.g. 'Stretch Defender'); null for non-dashboard players",
            "coverage":        f"{len(archetype_map)} players (Top-50 NET teams only)"
        },
        "conference_identification": {
            "method": "team_location string matched to hardcoded D1 conference membership + JSON ground truth override",
            "note": "Games where opponent conference is unknown default to regular_season",
            "conferences_covered": sorted(set(CONFERENCE_MAP.values()))
        },
        "total_rows": len(records),
        "row_counts": {k: int(v) for k, v in dist.items()}
    },
    "games": records
}

print(f"Writing {len(records):,} rows → {OUT_PATH}")
with open(OUT_PATH, 'w') as f:
    json.dump(output, f, separators=(',', ':'))

size_mb = Path(OUT_PATH).stat().st_size / 1024 / 1024
print(f"Done — {size_mb:.1f} MB")

# ── Quick conference accuracy check ─────────────────────────────────────────
print("\nConference game sample (RS only):")
conf_sample = pb[
    (pb['game_type'] == 'conference') & 
    (pb['team_location'].isin(CONFERENCE_MAP))
][['game_date','team_location','opponent_team_location']].drop_duplicates('game_id').head(8)
print(conf_sample.to_string(index=False))

print("\nConfidence check — same-conf opponent rate for known teams:")
rs_known = pb[
    (pb['season_type'] == 2) &
    (pb['team_location'].isin(CONFERENCE_MAP)) &
    (pb['opponent_team_location'].isin(CONFERENCE_MAP))
]
total_known = len(rs_known)
conf_match  = (rs_known['game_type'] == 'conference').sum()
print(f"  Of {total_known:,} games where both teams known: {conf_match:,} tagged conference ({100*conf_match/total_known:.1f}%)")
