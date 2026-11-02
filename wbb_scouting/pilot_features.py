"""Inspectable game-level processing and denominator-preserving pilot features."""
from __future__ import annotations
import pandas as pd
from .metrics import attach_opponents,team_metrics,player_metrics
from .store import clean_json

WINDOWS=("season","last_five")
ZONE_COLUMNS=["season","team_id","window","player_id","shot_zone","attempts","makes","fg_pct","attempt_share","points_per_attempt"]


def _ratio(a,b):
    return a/b if pd.notna(a) and pd.notna(b) and b>0 else None


def roster_union(rosters,boxes):
    rows=[]
    roster={(int(r["season"]),str(r["team_id"]),str(r["player_id"])):r for r in rosters.to_dict("records")}
    history={}
    for key,g in boxes.groupby(["season","team_id","player_id"],sort=True):
        history[(int(key[0]),str(key[1]),str(key[2]))]=g.sort_values(["game_date_time","game_id"]).iloc[-1].to_dict()
    for key in sorted(set(roster)|set(history)):
        official=roster.get(key,{});box=history.get(key,{})
        def pick(*values):
            return next((v for v in values if v is not None and pd.notna(v)),None)
        rows.append({"season":key[0],"team_id":key[1],"player_id":key[2],
            "player_name":pick(box.get("player_name"),official.get("player_name"),"Player "+key[2]),
            "jersey":pick(official.get("jersey"),box.get("athlete_jersey")),
            "position":pick(official.get("position_abbreviation"),box.get("athlete_position_abbreviation")),
            "height":official.get("height"),"experience":official.get("experience_display_value"),
            "roster_status":"roster_and_box" if official and box else "roster_only" if official else "box_only",
            "roster_member":bool(official),"box_participant":bool(box)})
    return pd.DataFrame(rows)


def process_pilot(clean,config):
    ids={t["team_id"] for t in config["teams"]}
    games=attach_opponents(clean["support_team_box"])
    games=games[games.team_id.isin(ids)].sort_values(["team_id","game_date_time","game_id"]).copy()
    # Reuse reconciled opponent statistics before restricting the featured cohort.
    records=[]
    for row in games.to_dict("records"):
        metrics=team_metrics(pd.DataFrame([row]))
        records.append({**row,**metrics,"net_rating":(metrics["offensive_rating"]-metrics["defensive_rating"])
                        if metrics["offensive_rating"] is not None and metrics["defensive_rating"] is not None else None,
                        "margin":row["points"]-row["opponent_points"]})
    games=pd.DataFrame(records)
    pg=clean["player_box"].copy()
    pg["ts_pct"]=[_ratio(p,2*(f+.44*t)) for p,f,t in zip(pg.points,pg.field_goals_attempted,pg.free_throws_attempted)]
    pg["fg_pct"]=[_ratio(m,a) for m,a in zip(pg.field_goals_made,pg.field_goals_attempted)]
    roster=roster_union(clean["rosters"],pg)
    identity=roster[["season","team_id","player_id","jersey","position","height","roster_status"]]
    pg=pg.merge(identity,on=["season","team_id","player_id"],how="left",validate="many_to_one")
    shots=clean["shots"].copy()
    names=roster[["season","team_id","player_id","player_name"]]
    shots=shots.merge(names,on=["season","team_id","player_id"],how="left",validate="many_to_one")
    return {"team_games":games,"player_games":pg,"roster":roster,"shot_events":shots}


def _zone_rows(shots,team,window,season):
    eligible=shots[shots.is_field_goal & shots.eligible_for_features]
    if eligible.empty:return []
    rows=[]
    buckets=[("",eligible),*[(str(pid),g) for pid,g in eligible.groupby("player_id",sort=True)]]
    for pid,group in buckets:
        for zone,g in group.groupby("shot_zone",sort=True):
            attempts=len(g);makes=int(g.made.sum())
            points=float((g.made.astype(int)*g.score_value).sum())
            rows.append({"season":season,"team_id":team,"window":window,"player_id":pid,
                "shot_zone":zone,"attempts":attempts,"makes":makes,"fg_pct":_ratio(makes,attempts),
                "attempt_share":_ratio(attempts,len(group)),"points_per_attempt":_ratio(points,attempts)})
    return rows


def build_pilot_features(processed,clean,config):
    teams=[];players=[];zones=[];trends=[]
    boxes=processed["team_games"];pg=processed["player_games"];roster=processed["roster"];shots=processed["shot_events"]
    for spec in config["teams"]:
        tid=spec["team_id"]
        own=boxes[boxes.team_id.eq(tid)].sort_values(["game_date_time","game_id"])
        own_players=pg[pg.team_id.eq(tid)]
        team_roster=roster[roster.team_id.eq(tid)]
        for window in WINDOWS:
            group=own.tail(5) if window=="last_five" else own
            ids=set(group.game_id)
            selected=own_players[own_players.game_id.isin(ids)]
            sh=shots[shots.team_id.eq(tid)&shots.game_id.isin(ids)]
            field=sh[sh.is_field_goal&sh.eligible_for_features]
            m=team_metrics(group)
            m.update({"season":config["season"],"team_id":tid,"team_name":spec["name"],"window":window,
                "wins":int(group.points.gt(group.opponent_points).sum()),"losses":int(group.points.lt(group.opponent_points).sum()),
                "fg_pct":_ratio(group.field_goals_made.sum(),group.field_goals_attempted.sum()),
                "net_rating":m["offensive_rating"]-m["defensive_rating"] if m["offensive_rating"] is not None and m["defensive_rating"] is not None else None,
                "shot_games":int(field.game_id.nunique()),"shot_expected_games":len(group),
                "shot_attempts":len(field),"located_attempts":int(field.location_valid.sum())})
            teams.append(m)
            totals={r["player_id"]:r for r in player_metrics(selected)}
            minutes=pd.to_numeric(selected.minutes,errors="coerce").sum()
            for record in team_roster.to_dict("records"):
                feature=totals.get(record["player_id"],{"games_played":0,"minutes_per_game":None,"points_per_game":None,
                    "rebounds_per_game":None,"assists_per_game":None,"ts_pct":None,"minutes_total":0,"points_total":0})
                mine=selected[selected.player_id.eq(record["player_id"])]
                row={**record,**feature,"window":window,"team_games":len(group),
                    "minutes_share":_ratio(feature.get("minutes_total",0),minutes),
                    "points_share":_ratio(feature.get("points_total",0),m["points_total"]),
                    "field_goals_made":mine.field_goals_made.sum(min_count=1),
                    "field_goals_attempted":mine.field_goals_attempted.sum(min_count=1),
                    "three_point_field_goals_made":mine.three_point_field_goals_made.sum(min_count=1),
                    "three_point_field_goals_attempted":mine.three_point_field_goals_attempted.sum(min_count=1)}
                row["fg_pct"]=_ratio(row["field_goals_made"],row["field_goals_attempted"])
                row["three_pct"]=_ratio(row["three_point_field_goals_made"],row["three_point_field_goals_attempted"])
                players.append(row)
            zones.extend(_zone_rows(sh,tid,window,config["season"]))
        for index,row in enumerate(own.to_dict("records")):
            rolling=team_metrics(own.iloc[max(0,index-4):index+1])
            trends.append({"season":config["season"],"team_id":tid,"game_id":row["game_id"],"game_date_time":row["game_date_time"],
                "game_number":index+1,"opponent_team_id":row["opponent_team_id"],
                "opponent_name":row.get("opponent_team_display_name",row["opponent_team_id"]),
                "points":row["points"],"opponent_points":row["opponent_points"],"margin":row["margin"],
                "offensive_rating":row["offensive_rating"],"defensive_rating":row["defensive_rating"],
                "rolling_games":rolling["games"],"rolling_offensive_rating":rolling["offensive_rating"],
                "rolling_defensive_rating":rolling["defensive_rating"]})
    return {"team_summary":pd.DataFrame(clean_json(teams)),"player_summary":pd.DataFrame(clean_json(players)),
            "shot_zones":pd.DataFrame(zones,columns=ZONE_COLUMNS),"team_trends":pd.DataFrame(trends)}
