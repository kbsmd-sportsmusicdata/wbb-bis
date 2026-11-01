"""Immutable pregame/postgame editions with strict game-time cutoffs."""
from __future__ import annotations
from datetime import datetime,timedelta
import pandas as pd
from .calendar import aware_utc,next_primary_check,parse_time
from .metrics import attach_opponents,team_metrics,player_metrics,pbp_summary
from .store import clean_json,fingerprint

EMPTY=pd.DataFrame()

def _season_table(tables,name,season):
    frame=tables.get(name,EMPTY)
    if frame.empty or "season" not in frame:return frame.iloc[:0].copy()
    mask=pd.to_numeric(frame.season,errors="coerce").eq(season)
    return frame if mask.all() else frame.loc[mask]

def _team_rows(frame,team_id,cutoff):
    if frame.empty:return frame.copy()
    return frame[frame.team_id.astype(str).eq(str(team_id)) & (pd.to_datetime(frame.game_date_time,utc=True)<cutoff)].copy()


def _chronological(tables,name,season):
    frame=_season_table(tables,name,season)
    schedule=_season_table(tables,"schedule",season)
    if frame.empty or schedule.empty:return frame
    frame=frame.copy()
    times=schedule.drop_duplicates("game_id").set_index("game_id").game_date_time
    frame["game_date_time"]=frame.game_id.map(times)
    return frame[frame.game_date_time.notna()]


def _profile(tables,team_id,season,cutoff,exclude_game_id=None):
    boxes=_chronological(tables,"team_box",season)
    if len(boxes) and exclude_game_id is not None:boxes=boxes[~boxes.game_id.astype(str).eq(str(exclude_game_id))]
    boxes=boxes[pd.to_datetime(boxes.game_date_time,utc=True)<cutoff] if len(boxes) else boxes
    own=_team_rows(attach_opponents(boxes),team_id,cutoff)
    own=own.sort_values("game_date_time") if len(own) else own
    last=own.tail(5)
    games=set(own.game_id.astype(str)) if len(own) else set()
    recent=set(last.game_id.astype(str)) if len(last) else set()
    players=_chronological(tables,"player_box",season)
    players=_team_rows(players,team_id,cutoff)
    if len(players):players=players[players.game_id.astype(str).isin(games)]
    recent_players=players[players.game_id.astype(str).isin(recent)] if len(players) else players
    events=_chronological(tables,"pbp",season)
    if len(events):
        events=events[events.game_id.astype(str).isin(games)&(pd.to_datetime(events.game_date_time,utc=True)<cutoff)]
    schedule=_season_table(tables,"schedule",season)
    expected=set()
    if len(schedule):
        eligible=(schedule.home_team_id.astype(str).eq(str(team_id))|schedule.away_team_id.astype(str).eq(str(team_id)))
        complete=schedule.completed.fillna(False).astype(bool)
        before=pd.to_datetime(schedule.game_date_time,utc=True)<cutoff
        expected=set(schedule.loc[eligible&complete&before,"game_id"].astype(str))
    logs=[]
    for row in own.to_dict("records"):
        logs.append({"game_id":str(row["game_id"]),"game_at":row["game_date_time"],
                     "points":row.get("points"),"opponent_points":row.get("opponent_points"),
                     "opponent_id":row.get("opponent_team_id")})
    return clean_json({"season":team_metrics(own),"last_five":team_metrics(last),
        "players":player_metrics(players),"last_five_players":player_metrics(recent_players),
        "game_log":logs,"pbp":pbp_summary(events,team_id),
        "input_game_ids":{"team_box":sorted(games),"player_box":sorted(set(players.game_id.astype(str))) if len(players) else [],"pbp":sorted(set(events.game_id.astype(str))) if len(events) else []},
        "coverage":{"expected_games":len(expected),"team_box_games":len(games),
                    "player_box_games":int(players.game_id.nunique()) if len(players) else 0,
                    "pbp_games":int(events.game_id.nunique()) if len(events) else 0,
                    "missing_box_game_ids":sorted(expected-games)}})

def _cohort_context(row):
    fields=("inclusion_reason","selection_status","net_rank","ranking_date","ranking_season","ranking_source_url")
    return clean_json({field:row.get(field) for field in fields})

def _historical(tables,team,season,cutoff):
    boxes=_chronological(tables,"team_box",season-1)
    own=_team_rows(attach_opponents(boxes),team,cutoff)
    return {"season":season-1,"metrics":team_metrics(own),"game_ids":sorted(set(own.game_id.astype(str))) if len(own) else [],
            "label":"Historical team context; not a current-roster projection"}


def _report_sources(snapshots,profile,opponent_profile,historical,opponent_historical,season,game_id,mode,game_metrics,game_players,scoring):
    game_sources=snapshots.get("game_sources")
    if game_sources is None:
        return snapshots.get("dataset_snapshots",{})
    used={}
    def add(dataset,year,ids):
        used.setdefault(dataset,set()).update(str(year)+":"+str(value) for value in ids)
    add("schedule",season,[game_id])
    for item in (profile,opponent_profile):
        for dataset,ids in item.get("input_game_ids",{}).items():
            add(dataset,season,ids)
    for item in (historical,opponent_historical):
        add("team_box",item["season"],item.get("game_ids",[]))
    if mode=="postgame":
        if game_metrics and game_metrics["games"]:add("team_box",season,[game_id])
        if game_players:add("player_box",season,[game_id])
        if scoring is not None:add("pbp",season,[game_id])
    refs={}
    for dataset,keys in used.items():
        grouped={}
        for key in sorted(keys):
            checksum=game_sources.get(dataset,{}).get(key)
            if checksum:grouped.setdefault(checksum,[]).append(key)
        if grouped:
            refs[dataset]=[{"sha256":checksum,"games":ids} for checksum,ids in sorted(grouped.items())]
    for name,checksum in snapshots.get("dataset_snapshots",{}).items():
        if name.split(":")[-1] not in ("team_box","player_box","pbp","schedule","net"):
            refs[name]=checksum
    refs.update(snapshots.get("active_net_snapshot",{}))
    return refs

def build_reports(tables,cohort,now,previous_reports,snapshots,bootstrap=False):
    now=aware_utc(now)
    season=int(cohort.season.iloc[0]) if len(cohort) else int(snapshots.get("season",2027))
    schedule=_season_table(tables,"schedule",season)
    result=list(previous_reports)
    if schedule.empty:return result
    cohort_lookup={str(row["team_id"]):row for row in cohort.to_dict("records")}
    horizon=next_primary_check(now)
    boxes=_season_table(tables,"team_box",season)
    enriched=attach_opponents(boxes)
    players=_season_table(tables,"player_box",season)
    events=_season_table(tables,"pbp",season)
    for game in schedule.sort_values("game_date_time").to_dict("records"):
        tip=pd.to_datetime(game["game_date_time"],utc=True).to_pydatetime()
        completed=bool(game.get("completed",False))
        if not completed and now<tip<horizon:
            mode="pregame"
        elif completed and tip<=now:
            if bootstrap and tip<now-timedelta(days=7):continue
            mode="postgame"
        else:
            continue
        game_id=str(game["game_id"])
        for side,other in (("home","away"),("away","home")):
            team,opponent=str(game[side+"_team_id"]),str(game[other+"_team_id"])
            prior_for_team=[r for r in previous_reports if r.get("team_id")==team and r.get("game_id")==game_id and r.get("season")==season]
            if team not in cohort_lookup and not (mode=="postgame" and any(r["mode"]=="pregame" for r in prior_for_team)):
                continue
            if mode=="postgame" and not bootstrap and snapshots.get("new_game_ids") is not None:
                already_issued=any(r["mode"] in ("pregame","postgame") for r in prior_for_team)
                if game_id not in set(map(str,snapshots["new_game_ids"])) and not already_issued:
                    continue
            member=cohort_lookup.get(team,{})
            team_name=member.get("team_name") or game.get(side+"_display_name") or game.get(side+"_location") or team
            opponent_name=cohort_lookup.get(opponent,{}).get("team_name") or game.get(other+"_display_name") or game.get(other+"_location") or opponent
            cutoff=now if mode=="pregame" else tip
            profile=_profile(tables,team,season,cutoff,game_id)
            opponent_profile=_profile(tables,opponent,season,cutoff,game_id)
            historical=_historical(tables,team,season,cutoff)
            opponent_historical=_historical(tables,opponent,season,cutoff)
            baseline=None
            opponent_baseline=None
            baseline_sources={}
            baseline_origin=None
            baseline_report_id=None
            game_metrics=None
            game_players=[]
            scoring=None
            if mode=="postgame":
                issued=[r for r in prior_for_team if r["mode"]=="pregame" and parse_time(r["generated_at"])<tip]
                issued.sort(key=lambda r:r["generated_at"])
                if issued:
                    baseline=issued[-1]["profile"];opponent_baseline=issued[-1]["opponent_profile"];baseline_sources=issued[-1].get("source_references",{});baseline_origin="issued_pregame";baseline_report_id=issued[-1]["edition_id"]
                else:
                    baseline=profile;opponent_baseline=opponent_profile;baseline_origin="reconstructed"
                target=enriched[enriched.game_id.astype(str).eq(game_id)&enriched.team_id.astype(str).eq(team)] if len(enriched) else enriched
                game_metrics=team_metrics(target)
                game_players=player_metrics(players[players.game_id.astype(str).eq(game_id)&players.team_id.astype(str).eq(team)]) if len(players) else []
                game_events=events[events.game_id.astype(str).eq(game_id)] if len(events) else events
                scoring=pbp_summary(game_events,team)
                status="awaiting_box_scores" if game_metrics["games"]==0 else "box_score_only" if scoring is None else "complete" if game_players else "partial_player_box"
            else:
                expected=profile["coverage"]["expected_games"]
                available=profile["season"]["games"]
                status=("awaiting_box_scores" if expected and not available else
                        "historical_context_only" if not available and historical["metrics"]["games"] else
                        "awaiting_first_game" if not available else
                        "partial_baseline" if available<expected or profile["coverage"]["player_box_games"]<available else "ready")
            roster=_season_table(tables,"rosters",season)
            roster_rows=[]
            if len(roster):
                cols=[c for c in ("player_id","player_name","position_abbreviation","experience_display_value") if c in roster]
                roster_rows=roster.loc[roster.team_id.astype(str).eq(team),cols].to_dict("records")
            material=clean_json({"mode":mode,"season":season,"game_id":game_id,"team_id":team,"team_name":team_name,
                "opponent_id":opponent,"opponent_name":opponent_name,"home_away":side,"tip_at":tip.isoformat(),
                "venue":game.get("venue_full_name"),"status":status,"profile":profile,"opponent_profile":opponent_profile,
                "historical_reference":historical,"opponent_historical_reference":opponent_historical,"cohort":_cohort_context(member) if member else prior_for_team[-1].get("cohort",{}),
                "baseline":baseline,"opponent_baseline":opponent_baseline,"baseline_source_references":baseline_sources,"baseline_origin":baseline_origin,"baseline_report_id":baseline_report_id,
                "game_metrics":game_metrics,"game_players":game_players,"pbp":scoring,"roster":roster_rows,
                "source_references":_report_sources(snapshots,profile,opponent_profile,historical,opponent_historical,season,game_id,mode,game_metrics,game_players,scoring)})
            material["calculation_version"]="scouting-0.1.0"
            digest=fingerprint({k:v for k,v in material.items() if k!="source_references"})
            report_id=fingerprint({"mode":mode,"game_id":game_id,"team_id":team,"season":season})[:20]
            editions=[r for r in result if r.get("report_id")==report_id]
            if editions and editions[-1].get("edition_fingerprint")==digest:
                continue
            report={**material,"report_id":report_id,"edition_id":report_id+"-v"+str(len(editions)+1)+"-"+digest[:12],
                    "edition":len(editions)+1,"edition_fingerprint":digest,
                    "generated_at":now.isoformat(),"cutoff_at":now.isoformat(),"baseline_cutoff_at":cutoff.isoformat(),
                    "calculation_version":"scouting-0.1.0",
                    "baseline_label":"Issued pregame" if baseline_origin=="issued_pregame" else "Reconstructed from earlier games" if mode=="postgame" else "Before scheduled tip"}
            result.append(report)
    return clean_json(result)
