"""Bind saved pilot tables to a portable, historical scouting report."""
from pathlib import Path
from .store import clean_json,json_text

def _records(frame,columns=None):
    if columns is not None:frame=frame[[c for c in columns if c in frame]]
    return clean_json(frame.to_dict("records"))

def pilot_payload(processed,features,quality,config,manifest):
    shots=processed["shot_events"]
    shots=shots[shots.is_field_goal & shots.eligible_for_features]
    shot_columns=["source_row_id","season","game_id","team_id","player_id","player_name","game_date_time",
                  "is_field_goal","eligible_for_features","location_valid","court_x","court_y","made","score_value","shot_zone"]
    player_columns=["season","game_id","game_date_time","team_id","opponent_team_id","opponent_team_display_name",
        "player_id","player_name","minutes","points","rebounds","assists","field_goals_made","field_goals_attempted",
        "three_point_field_goals_made","three_point_field_goals_attempted","ts_pct","did_not_play","jersey","position"]
    return clean_json({"schema_version":1,"season":config["season"],"label":config["label"],
        "built_at":manifest["built_at"],"data_through":manifest["source_data_latest_game_at"],
        "teams":config["teams"],"matchups":config["matchups"],"zone_definitions":config.get("zone_definitions",{}),
        "datasets":manifest["tables"],
        "sources":[{k:s.get(k) for k in ("dataset","name","tag","sha256","row_count","bytes","browser_download_url","updated_at")} for s in manifest["sources"]],
        "coverage":{"validated_shot_groups":manifest["shot_groups_validated"],"withheld_shot_groups":manifest["shot_groups_withheld"]},
        "tables":{**{k:_records(v) for k,v in features.items()},"roster":_records(processed["roster"]),
                  "player_games":_records(processed["player_games"],player_columns),"shots":_records(shots,shot_columns),
                  "shot_groups":_records(quality["shot_groups"])}})

def render_pilot_dashboard(payload,destination):
    text=json_text(payload).replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    template=(Path(__file__).parent/"templates/pilot_dashboard.html").read_text()
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(template.replace("__PILOT_PAYLOAD__",text))
