"""Self-contained report view and complete machine-readable archives."""
from __future__ import annotations
import csv,json
from datetime import timedelta
from pathlib import Path
from .calendar import parse_time,next_primary_check,schedule_health
from .store import clean_json,json_text

def dashboard_payload(reports,manifest):
    anchor=parse_time(manifest["attempted_at"])
    visible=[r for r in reports if parse_time(r["tip_at"])>=anchor-timedelta(days=14)]
    return clean_json({"reports":visible,"cohort":manifest.get("cohort",[]),
        "attempted_at":manifest["attempted_at"],"status":manifest.get("status","partial"),
        "pending":manifest.get("pending",[]),"coverage":manifest.get("coverage",{}),
        "sources":manifest.get("sources",{}),"repository":manifest.get("repository"),
        "next_primary_at":next_primary_check(anchor).isoformat(),
        "refresh_health":schedule_health(anchor,anchor),
        "archive_count":len(reports)-len(visible),"total_editions":len(reports),
        "window_label":"Reports from the past 14 days and all upcoming reports",
        "preview_label":manifest.get("preview_label"),
        "calculation_version":"scouting-0.1.0"})

def render_dashboard(reports,manifest,destination):
    payload=json_text(dashboard_payload(reports,manifest))
    payload=payload.replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    template=(Path(__file__).parent/"templates/dashboard.html").read_text()
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(template.replace("__SCOUTING_PAYLOAD__",payload))

def write_exports(reports,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    (destination/"reports.json").write_text(json_text(reports))
    fields=["edition_id","mode","season","game_id","team_id","team_name","opponent_id","opponent_name","tip_at","generated_at","cutoff_at","status","baseline_origin","calculation_version","source_references"]
    with (destination/"reports.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in reports:
            writer.writerow({k:json_text(r.get(k)) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in fields})
    player_fields=["edition_id","cutoff_at","window","team_id","player_id","player_name","games_played","team_games","minutes_per_game","points_per_game","rebounds_per_game","assists_per_game","ts_pct"]
    with (destination/"players.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=player_fields,extrasaction="ignore");writer.writeheader()
        for r in reports:
            for label,rows in (("season",r.get("profile",{}).get("players",[])),("last_five",r.get("profile",{}).get("last_five_players",[])),("game",r.get("game_players",[]))):
                for row in rows:writer.writerow({**row,"edition_id":r["edition_id"],"cutoff_at":r["cutoff_at"],"window":label})
    return [destination/name for name in ("reports.json","reports.csv","players.csv")]
