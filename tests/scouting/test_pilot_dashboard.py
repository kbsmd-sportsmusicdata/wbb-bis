import json
from pathlib import Path
import pandas as pd
from .test_pilot_features import clean_fixture,config


def payload():
    from wbb_scouting.pilot_features import process_pilot,build_pilot_features
    from wbb_scouting.pilot_dashboard import pilot_payload
    c=config();c["label"]="2025–26 historical pilot";c["matchups"]=[]
    clean=clean_fixture();p=process_pilot(clean,c);f=build_pilot_features(p,clean,c)
    quality={"shot_groups":pd.DataFrame([{"game_id":"g1","team_id":"A","status":"validated","reason":"","expected_fga":2,"observed_fga":2}]),"box_quarantine":pd.DataFrame()}
    m={"built_at":"2026-11-01T20:00:00Z","source_data_latest_game_at":"2026-03-01T20:00:00Z","tables":{"features.team_summary":{"rows":2,"path":"features/team_summary.parquet","csv_path":"features/team_summary.csv","logical_sha256":"a"*64}},"sources":[],"shot_groups_validated":1,"shot_groups_withheld":0}
    return pilot_payload(p,f,quality,c,m)


def test_payload_binds_saved_feature_and_processed_grains_and_keeps_history():
    p=payload()
    assert p["season"]==2026
    assert p["label"]=="2025–26 historical pilot"
    assert len(p["tables"]["team_summary"])==2
    assert len(p["tables"]["roster"])==2
    assert len(p["tables"]["shots"])==2
    assert all(r["is_field_goal"] and r["eligible_for_features"] for r in p["tables"]["shots"])
    assert p["datasets"]["features.team_summary"]["logical_sha256"]=="a"*64


def test_dashboard_safe_embedding_and_sections(tmp_path):
    from wbb_scouting.pilot_dashboard import render_pilot_dashboard
    p=payload();p["teams"][0]["name"]="</script><script>alert(1)</script>"
    target=tmp_path/"index.html";render_pilot_dashboard(p,target)
    text=target.read_text()
    assert "\\u003c/script\\u003e" in text
    assert "</script><script>alert(1)" not in text
    for label in ["Matchup profile","Rolling five-game ratings","Player production","Shot map","Shooting zones","Roster and production","Game detail","Dataset layers"]:
        assert label in text
    assert "2025–26" in text
    assert "id=\"pilot-data\"" in text
