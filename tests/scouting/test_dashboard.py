import json,re
from pathlib import Path
from datetime import datetime,timezone
from wbb_scouting.dashboard import render_dashboard,write_exports,dashboard_payload

def report():
 return {"report_id":"r","edition_id":"r-v1","edition":1,"mode":"pregame","season":2027,"game_id":"9007199254740993","team_id":"25","team_name":"Team </script><script>alert(1)</script>","opponent_name":"Other","tip_at":"2026-11-01T17:00:00+00:00","generated_at":"2026-11-01T09:00:00+00:00","cutoff_at":"2026-11-01T09:00:00+00:00","status":"historical_context_only","source_references":{"team_box":[{"sha256":"abc","games":["2026:1"]}]},"profile":{"season":{"games":0,"efg_pct":None},"coverage":{}}}

def test_safe_embedded_json_and_large_identifiers(tmp_path):
 path=tmp_path/"index.html";render_dashboard([report()],{"attempted_at":"2026-11-01T09:00:00+00:00","cohort":[]},path)
 text=path.read_text()
 assert "</script><script>alert" not in text
 data=json.loads(re.search(r'<script id="scouting-data" type="application/json">(.*?)</script>',text,re.S).group(1))
 assert data["reports"][0]["game_id"]=="9007199254740993"
 assert data["reports"][0]["profile"]["season"]["efg_pct"] is None
 assert data["reports"][0]["status"]=="historical_context_only"
 assert "https://cdn" not in text

def test_all_exported_editions_keep_cutoff_and_sources(tmp_path):
 paths=write_exports([report()],tmp_path)
 assert {p.name for p in paths}=={"reports.json","reports.csv","players.csv"}
 rows=json.loads((tmp_path/"reports.json").read_text())
 assert rows[0]["cutoff_at"]==report()["cutoff_at"]
 text=(tmp_path/"reports.csv").read_text()
 assert "9007199254740993" in text and "source_references" in text and "abc" in text

def test_window_keeps_editions_and_all_upcoming_but_full_export_keeps_archive(tmp_path):
 old={**report(),"tip_at":"2026-09-01T17:00:00+00:00","report_id":"old"}
 latest={**report(),"edition_id":"r-v2","edition":2}
 m={"attempted_at":"2026-11-01T09:00:00+00:00","cohort":[]}
 p=dashboard_payload([old,report(),latest],m)
 assert len(p["reports"])==2 and p["archive_count"]==1
 write_exports([old,report(),latest],tmp_path)
 assert len(json.loads((tmp_path/"reports.json").read_text()))==3

def test_refresh_health_respects_weekend_not_daily():
 m={"attempted_at":"2026-10-30T13:17:00+00:00","cohort":[]}
 p=dashboard_payload([],m)
 assert p["next_primary_at"]=="2026-11-02T14:17:00+00:00"
 assert p["refresh_health"]=="on_schedule"

def test_report_csv_includes_team_metrics(tmp_path):
 r=report();r["profile"]["season"]["efg_pct"]=0.5
 write_exports([r],tmp_path)
 import csv
 rows=list(csv.DictReader((tmp_path/"reports.csv").open()))
 assert rows[0]["season_efg_pct"]=="0.5"
