"""Explicit ESPN WBB schema adapters; never join on display names."""
from __future__ import annotations
import math
import numbers
import pandas as pd

FAMILY = "sdv_espn_wbb"
KEYS = {
    "schedule":["source_family","season","game_id"],
    "team_box":["source_family","season","game_id","team_id"],
    "player_box":["source_family","season","game_id","team_id","player_id"],
    "pbp":["source_family","season","game_id","event_id"],
    "rosters":["source_family","season","team_id","player_id"],
    "teams":["source_family","season","team_id"],
}
STAT_COLUMNS = ["points","minutes","field_goals_made","field_goals_attempted",
    "three_point_field_goals_made","three_point_field_goals_attempted",
    "free_throws_made","free_throws_attempted","offensive_rebounds","defensive_rebounds",
    "rebounds","assists","steals","blocks","turnovers","fouls"]

def identifier(value):
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        if not math.isfinite(value) or not float(value).is_integer():
            raise ValueError("Nonintegral source identifier")
        if abs(value) > 2 ** 53:
            raise ValueError("Source identifier has unsafe float precision")
        return str(int(value))
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "<na>"):
        return pd.NA
    return text

def boolean(value):
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        if value.lower() in ("true","1"):
            return True
        if value.lower() in ("false","0"):
            return False
        raise ValueError("Unrecognized source boolean")
    if value in (True, False, 0, 1):
        return bool(value)
    raise ValueError("Unrecognized source boolean")

def normalize_snapshot(dataset: str, frame: pd.DataFrame, source: dict) -> pd.DataFrame:
    if source.get("family") != FAMILY:
        raise ValueError("Unexpected source family")
    if source.get("league", "womens_college_basketball") != "womens_college_basketball":
        raise ValueError("Unexpected source league")
    expected = int(source["season"])
    result = frame.copy()
    for column in ("uid", "team_uid", "home_uid", "away_uid"):
        if column in result:
            leagues = result[column].astype("string").str.extract(r"~l:(\d+)", expand=False).dropna()
            if not leagues.eq("54").all():
                raise ValueError("Unexpected women's basketball league identifier")
    if "season" not in result:
        raise ValueError("Missing required season")
    seasons = pd.to_numeric(result["season"], errors="coerce")
    if not seasons.eq(expected).all():
        raise ValueError("Unexpected source season")
    result["season"] = expected
    result["source_family"] = FAMILY
    aliases = {
        "schedule":{"id":"game_id","home_id":"home_team_id","away_id":"away_team_id",
                    "date":"game_date_time","status_type_completed":"completed"},
        "player_box":{"athlete_id":"player_id","athlete_display_name":"player_name"},
        "team_box":{"team_score":"points","opponent_team_score":"opponent_points","total_rebounds":"rebounds"},
        "pbp":{"id":"event_id","participants_0_athlete_id":"athlete_id_1","participants_1_athlete_id":"athlete_id_2"},
        "rosters":{"athlete_id":"player_id","display_name":"player_name"},
    }.get(dataset,{})
    for old, new in aliases.items():
        if old in result and new not in result:
            result = result.rename(columns={old:new})
        elif old in result and new in result and old in ("id", "athlete_id"):
            left, right = result[old].map(identifier), result[new].map(identifier)
            if not left.fillna("").eq(right.fillna("")).all():
                raise ValueError("Conflicting source identity aliases")
    for column in result:
        if column.endswith("_id") or column in ("id","event_id","sequence_number","game_play_number"):
            result[column] = result[column].map(identifier).astype("string")
    required = [c for c in KEYS[dataset] if c not in result]
    if required:
        raise ValueError("Missing required columns: " + ", ".join(required))
    if result[KEYS[dataset]].isna().any().any():
        raise ValueError("Missing source identity")
    if dataset in ("schedule","team_box","player_box","pbp"):
        if "game_date_time" not in result:
            if "game_date" not in result:
                raise ValueError("Missing required game date")
            result["game_date_time"] = result["game_date"]
        timestamps = pd.to_datetime(result["game_date_time"], utc=True, errors="coerce", format="mixed")
        if timestamps.isna().any():
            raise ValueError("Invalid game timestamp")
        # Calendar-based season check is an additional source identity guard.
        if not ((timestamps.dt.year == expected) | (timestamps.dt.year == expected - 1)).all():
            raise ValueError("Game timestamp outside requested season years")
        result["game_date_time"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "game_date" not in result:
            result["game_date"] = timestamps.dt.strftime("%Y-%m-%d")
    for column in ("completed","did_not_play","scoring_play","shooting_play","home_winner","away_winner"):
        if column in result:
            result[column] = result[column].map(boolean).astype("boolean")
    if dataset == "schedule" and "completed" not in result:
        raise ValueError("Missing schedule completion state")
    for column in STAT_COLUMNS + ["opponent_points","home_score","away_score","score_value"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if dataset in ("team_box","player_box") and "points" not in result:
        raise ValueError("Missing box-score points")
    if dataset == "team_box" and "total_turnovers" in result:
        result["turnovers"] = pd.to_numeric(result["total_turnovers"], errors="coerce")
    # Raw nested fields are retained as source bytes; only scalar fields enter canonical tables.
    keep = [c for c in result if not result[c].map(lambda v:isinstance(v,(dict,list))).any()]
    result = result[keep].drop_duplicates().reset_index(drop=True)
    if result.duplicated(KEYS[dataset]).any():
        raise ValueError("Conflicting duplicate source keys")
    return result
