"""
Script 7 — Build Player Dashboard HTML
=======================================
STATUS: ⚠️  This script needs to be rebuilt.

build_dashboard_v2.py was not saved to Google Drive and cannot be recovered.
The output file (wbb_player_dashboard.html) still exists in dashboards/ and
can be used as a reference for what this script should produce.

Inputs (from config.py):
    PLAYER_SCOUTING_50   → player_scouting_top50.csv
    PLAYER_SCOUTING_68   → player_scouting_tournament68.csv
    SCHEDULE_FILE        → schedule_filtered_YYYYMMDD.csv
    NET_FILE             → net_rankings_YYYYMMDD.csv
    TOURNAMENT_BRACKET   → tournament_bracket.csv
    TOURNAMENT_STORIES   → tournament_storylines.csv

Outputs:
    DASHBOARD_MAIN  → dashboards/wbb_player_dashboard.html
    DASHBOARD_T68   → dashboards/wbb_tournament68_dashboard.html

Author:  Krystal B Creative — Sports Analytics Portfolio
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PLAYER_SCOUTING_50, PLAYER_SCOUTING_68,
    SCHEDULE_FILE, NET_FILE,
    TOURNAMENT_BRACKET, TOURNAMENT_STORIES,
    DASHBOARD_MAIN, DASHBOARD_T68,
    validate_inputs,
)

if not validate_inputs(
    required=[PLAYER_SCOUTING_50, SCHEDULE_FILE, NET_FILE, TOURNAMENT_BRACKET],
    optional=[PLAYER_SCOUTING_68, TOURNAMENT_STORIES],
):
    sys.exit(1)

# TODO: Implement dashboard builder
# Reference: dashboards/wbb_player_dashboard.html shows the expected output structure
raise NotImplementedError(
    "build_dashboard.py needs to be rebuilt. "
    "See dashboards/wbb_player_dashboard.html for the expected output. "
    "Use the data:build-dashboard skill in Cowork mode to regenerate."
)
