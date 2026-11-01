"""Fail-closed identity, score and schedule coverage validation."""
from __future__ import annotations
from datetime import datetime,timezone
import pandas as pd
from .adapters import normalize_snapshot,identifier,KEYS,STAT_COLUMNS

def prepare_snapshot(dataset,raw,source):
    if dataset not in ("schedule","team_box","player_box","pbp"):
        return normalize_snapshot(dataset,raw,source),{}
    frame=raw.copy()
    game_column="game_id" if "game_id" in frame else "id"
    if game_column not in frame:
        raise ValueError("Missing game identity")
    games=frame[game_column].map(identifier)
    if games.isna().any():
        raise ValueError("Unaddressable null game identity")
    aliases={"player_id":"athlete_id","event_id":"id","home_team_id":"home_id","away_team_id":"away_id"}
    raw_keys=[]
    for key in KEYS[dataset]:
        if key=="source_family":
            continue
        column=key if key in frame else aliases.get(key,key)
        if column not in frame:
            raise ValueError("Missing required source identity column: "+key)
        raw_keys.append(column)
    quarantine={}
    key_frame=frame[raw_keys].copy()
    for column in raw_keys:
        if column!="season":
            key_frame[column]=key_frame[column].map(identifier)
    for game in games[key_frame.isna().any(axis=1)].unique():
        quarantine[str(game)]=["null source identity"]
    duplicates=key_frame.duplicated(raw_keys,keep=False)
    if duplicates.any():
        subset=frame.loc[duplicates]
        scalar=[c for c in subset if not subset[c].map(lambda v:isinstance(v,(dict,list))).any()]
        for _,rows in subset.groupby(raw_keys,dropna=False):
            if len(rows[scalar].drop_duplicates())>1:
                for game in rows[game_column].map(identifier).unique():
                    quarantine[str(game)]=["conflicting duplicate source keys"]
    keep=~games.isin(quarantine)
    return normalize_snapshot(dataset,frame.loc[keep].copy(),source),quarantine

def _stat_issues(dataset,group):
    errors=[]
    for column in STAT_COLUMNS:
        if column in group:
            values=pd.to_numeric(group[column],errors="coerce")
            if values.dropna().lt(0).any():
                errors.append("negative "+column)
    for made,attempted in (("field_goals_made","field_goals_attempted"),
        ("three_point_field_goals_made","three_point_field_goals_attempted"),
        ("free_throws_made","free_throws_attempted")):
        if made in group and attempted in group and (group[made]>group[attempted]).any():
            errors.append(made+" exceeds attempts")
    if "three_point_field_goals_made" in group and "field_goals_made" in group:
        if (group.three_point_field_goals_made>group.field_goals_made).any():
            errors.append("three-pointers exceed made field goals")
    if "points" in group:
        playing=~group.get("did_not_play",pd.Series(False,index=group.index)).fillna(False).astype(bool)
        if group.loc[playing,"points"].isna().any():
            errors.append("missing playing points")
        required=["field_goals_made","three_point_field_goals_made","free_throws_made"]
        if all(c in group for c in required):
            calculated=2*group.field_goals_made+group.three_point_field_goals_made+group.free_throws_made
            compared=playing & group[required].notna().all(axis=1)
            if (calculated[compared]!=group.loc[compared,"points"]).any():
                errors.append("points disagree with shooting totals")
    return errors


def _player_completeness(player_rows,team_row,event,previous):
    """Whole-minute source rounding permits at most half a minute per participant."""
    errors=[]
    values=pd.to_numeric(player_rows.get("minutes",pd.Series(dtype=float)),errors="coerce")
    playing=~player_rows.get("did_not_play",pd.Series(False,index=player_rows.index)).fillna(False).astype(bool)
    if "minutes" not in player_rows or values.loc[playing].isna().any():
        return ["player minutes completeness unavailable"]
    total=values.sum(min_count=1)
    expected=pd.to_numeric(pd.Series([team_row.get("minutes")]),errors="coerce").iloc[0]
    if pd.isna(expected) or expected<=0:
        final=pd.to_numeric(pd.Series([event.get("status_period") if event else None]),errors="coerce").iloc[0]
        regulation=pd.to_numeric(pd.Series([event.get("format_regulation_periods") if event else None]),errors="coerce").iloc[0]
        if pd.isna(final) or pd.isna(regulation) or regulation not in (2,4) or final<regulation or final%1:
            return ["game duration unavailable for player completeness"]
        expected=200+25*(final-regulation)
    participants=int(values.gt(0).sum())
    tolerance=max(.5,participants*.5)
    if pd.isna(total) or abs(total-expected)>tolerance:
        errors.append("player minutes disagree with team duration")
    if len(previous):
        prior=previous[previous.team_id.astype(str).eq(str(team_row["team_id"]))]
        prior_minutes=pd.to_numeric(prior.get("minutes",pd.Series(dtype=float)),errors="coerce")
        missing=set(prior.loc[prior_minutes.gt(0),"player_id"].astype(str))-set(player_rows.player_id.astype(str))
        if missing and (pd.isna(total) or abs(total-expected)>.01):
            errors.append("participating player removed without complete minute reconciliation")
    return errors


def validate_snapshot(dataset,frame,context):
    season=int(context["season"])
    schedule=context.get("schedule",pd.DataFrame())
    teams=context.get("teams",pd.DataFrame())
    now=pd.to_datetime(context.get("now",datetime.now(timezone.utc)),utc=True)
    accepted=set()
    quarantine={}
    if len(frame) and not pd.to_numeric(frame.season,errors="coerce").eq(season).all():
        raise ValueError("Unexpected canonical season")
    expected=set()
    schedule_lookup={}
    if len(schedule):
        schedule=schedule[pd.to_numeric(schedule.season,errors="coerce").eq(season)].copy()
        if len(teams):
            d1=set(teams.loc[teams.get("is_d1",pd.Series(True,index=teams.index)).fillna(False),"team_id"].astype(str))
            eligible=schedule.home_team_id.astype(str).isin(d1)|schedule.away_team_id.astype(str).isin(d1)
            schedule=schedule.loc[eligible]
        schedule_lookup={str(row["game_id"]):row for row in schedule.to_dict("records")}
        final=schedule.completed.fillna(False).astype(bool)&(pd.to_datetime(schedule.game_date_time,utc=True)<=now)
        expected=set(schedule.loc[final,"game_id"].astype(str))
    if "game_id" not in frame:
        return {"accepted_games":set(),"quarantine":{},"coverage":{"expected_games":len(expected),"available_games":0,"missing_game_ids":sorted(expected),"latest_usable_game_at":None},"reference_valid":True}
    for game,group in frame.groupby("game_id",sort=False):
        game=str(game)
        errors=[]
        if group[KEYS[dataset]].isna().any().any() or group.duplicated(KEYS[dataset]).any():
            errors.append("invalid or duplicate canonical identity")
        event=schedule_lookup.get(game)
        if event is None:
            errors.append("game absent from eligible schedule")
        if event is not None and (pd.isna(event["completed"]) or not bool(event["completed"])):
            errors.append("schedule game not final")
        if event is not None and pd.to_datetime(event["game_date_time"],utc=True)>now:
            errors.append("schedule game is in the future")
        if dataset in ("team_box","player_box"):
            errors+=_stat_issues(dataset,group)
        if dataset=="team_box":
            if len(group)!=2 or group.team_id.nunique()!=2:
                errors.append("team box requires two reciprocal teams")
            else:
                by_team={str(row["team_id"]):row for row in group.to_dict("records")}
                for team,row in by_team.items():
                    opponent=next(t for t in by_team if t!=team)
                    if str(row.get("opponent_team_id",opponent))!=opponent:
                        errors.append("nonreciprocal opponent identity")
                    if pd.notna(row.get("opponent_points")) and row["opponent_points"]!=by_team[opponent]["points"]:
                        errors.append("nonreciprocal final score")
                if event is not None:
                    for side in ("home","away"):
                        tid=str(event[side+"_team_id"])
                        if tid not in by_team or by_team[tid]["points"]!=event[side+"_score"]:
                            errors.append("team points disagree with final schedule")
        elif dataset=="player_box":
            boxes=context.get("team_box",pd.DataFrame())
            expected_boxes=boxes[boxes.game_id.astype(str).eq(game)] if len(boxes) else pd.DataFrame()
            if len(expected_boxes)!=2:
                errors.append("validated team boxes unavailable")
            else:
                if set(group.team_id.astype(str))!=set(expected_boxes.team_id.astype(str)):
                    errors.append("player teams disagree with matchup")
                for row in expected_boxes.to_dict("records"):
                    player_rows=group[group.team_id.astype(str).eq(str(row["team_id"]))]
                    previous=context.get("previous_player_box",pd.DataFrame())
                    if len(previous):previous=previous[previous.game_id.astype(str).eq(game)]
                    errors+=_player_completeness(player_rows,row,event,previous)
                    points=player_rows.points.sum(min_count=1)
                    if pd.isna(points) or points!=row["points"]:
                        errors.append("player points disagree with team box")
        elif dataset=="pbp":
            required=["home_score","away_score","scoring_play","score_value","team_id"]
            if any(c not in group for c in required) or event is None:
                errors.append("PBP score reconciliation unsupported")
            else:
                ordered=group.copy()
                order_column="game_play_number" if "game_play_number" in ordered else "sequence_number"
                if order_column not in ordered:
                    errors.append("PBP event order unavailable")
                else:
                    ordered["_order"]=pd.to_numeric(ordered[order_column],errors="coerce")
                    if ordered._order.isna().any() or ordered._order.duplicated().any():
                        errors.append("PBP event order invalid")
                    else:
                        last=ordered.sort_values("_order").iloc[-1]
                        if last.home_score!=event["home_score"] or last.away_score!=event["away_score"]:
                            errors.append("PBP final score disagrees with schedule")
                scoring=group[group.scoring_play.fillna(False).astype(bool)]
                if not set(scoring.team_id.astype(str)).issubset({str(event["home_team_id"]),str(event["away_team_id"])}):
                    errors.append("PBP scoring team outside matchup")
                if group.scoring_play.isna().any():
                    errors.append("PBP scoring flags missing")
                for side in ("home","away"):
                    tid=str(event[side+"_team_id"])
                    total=scoring.loc[scoring.team_id.astype(str).eq(tid),"score_value"].sum(min_count=1)
                    if pd.isna(total) or total!=event[side+"_score"]:
                        errors.append("PBP scoring-event total disagrees with schedule")
        if errors:
            quarantine[game]=sorted(set(errors))
        else:
            accepted.add(game)
    available=accepted&expected
    dates=frame.loc[frame.game_id.astype(str).isin(available),"game_date_time"] if "game_date_time" in frame else pd.Series(dtype=str)
    return {"accepted_games":accepted,"quarantine":quarantine,
            "coverage":{"expected_games":len(expected),"available_games":len(available),
                        "missing_game_ids":sorted(expected-available),
                        "latest_usable_game_at":None if dates.empty else dates.max()}}

def annotate_division(frame,teams):
    """Keep reference coverage distinct from an asserted non-D1 classification."""
    out=frame.copy()
    known=set(teams.loc[teams["is_d1"].fillna(False),"team_id"].astype(str)) if len(teams) else set()
    for column,label in (("team_id","team_division"),("opponent_team_id","opponent_division"),
                         ("home_team_id","home_division"),("away_team_id","away_division")):
        if column in out:
            out[label]=out[column].astype(str).map(lambda value:"I" if value in known else "UNKNOWN_NEEDS_REVIEW")
    if "team_division" in out and "opponent_division" in out:
        out["comparison_eligible"]=out.team_division.eq("I")&out.opponent_division.eq("I")
    elif "home_division" in out and "away_division" in out:
        out["comparison_eligible"]=out.home_division.eq("I")&out.away_division.eq("I")
        if "away_non_div1_team" in out:
            from .adapters import boolean
            explicit=out.away_non_div1_team.map(boolean).astype("boolean").fillna(False).astype(bool)
            out.loc[explicit,"away_division"]="NON_D1"
            out.loc[explicit,"comparison_eligible"]=False
    else:
        out["comparison_eligible"]=False
    return out
