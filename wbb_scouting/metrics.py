"""Explicit box-derived measures; rates use aggregate denominators."""
from __future__ import annotations
import pandas as pd
from .store import clean_json

BOX_COLUMNS=["points","field_goals_made","field_goals_attempted","three_point_field_goals_made",
    "three_point_field_goals_attempted","free_throws_made","free_throws_attempted",
    "offensive_rebounds","defensive_rebounds","turnovers","rebounds","assists","minutes"]

def attach_opponents(frame):
    if frame.empty:return frame.copy()
    keys=["source_family","season","game_id","team_id"]
    columns=keys+[c for c in BOX_COLUMNS if c in frame]
    lookup=frame[columns].rename(columns={"team_id":"opponent_team_id",**{c:"opponent_"+c for c in BOX_COLUMNS if c in frame}})
    remove=[c for c in lookup if c.startswith("opponent_") and c!="opponent_team_id" and c in frame]
    return frame.drop(columns=remove).merge(lookup,on=["source_family","season","game_id","opponent_team_id"],how="left",validate="many_to_one")

def _sum(frame,column):
    if frame.empty or column not in frame:return None
    values=pd.to_numeric(frame[column],errors="coerce")
    return None if values.isna().any() else float(values.sum())

def _ratio(numerator,denominator):
    if numerator is None or denominator is None or denominator<=0:return None
    return numerator/denominator

def _add(*values):
    return None if any(v is None for v in values) else sum(values)

def _scale(value,multiplier):
    return None if value is None else value*multiplier

def team_metrics(frame):
    if len(frame) and frame.team_id.nunique()>1:
        raise ValueError("team_metrics requires one team")
    games=int(frame.game_id.nunique()) if len(frame) else 0
    sums={c:_sum(frame,c) for c in BOX_COLUMNS}
    opp={c:_sum(frame,"opponent_"+c) for c in BOX_COLUMNS}
    def estimate(values):
        if any(values.get(c) is None for c in ("field_goals_attempted","offensive_rebounds","turnovers","free_throws_attempted")):return None
        value=values["field_goals_attempted"]-values["offensive_rebounds"]+values["turnovers"]+.44*values["free_throws_attempted"]
        return value if value>0 else None
    own_poss,opp_poss=estimate(sums),estimate(opp)
    possessions=(own_poss+opp_poss)/2 if own_poss is not None and opp_poss is not None else None
    efg_numerator=_add(sums["field_goals_made"],_scale(sums["three_point_field_goals_made"],.5))
    ts_denominator=_scale(_add(sums["field_goals_attempted"],_scale(sums["free_throws_attempted"],.44)),2)
    allowed_efg=_add(opp["field_goals_made"],_scale(opp["three_point_field_goals_made"],.5))
    result={"games":games,"points_total":sums["points"],"points_per_game":_ratio(sums["points"],games),
        "opponent_points_per_game":_ratio(opp["points"],games),
        "field_goals_attempted":sums["field_goals_attempted"],"three_point_attempts":sums["three_point_field_goals_attempted"],
        "free_throw_attempts":sums["free_throws_attempted"],
        "efg_pct":_ratio(efg_numerator,sums["field_goals_attempted"]),
        "ts_pct":_ratio(sums["points"],ts_denominator),
        "three_point_attempt_rate":_ratio(sums["three_point_field_goals_attempted"],sums["field_goals_attempted"]),
        "free_throw_rate":_ratio(sums["free_throws_attempted"],sums["field_goals_attempted"]),
        "turnover_rate":_ratio(sums["turnovers"],possessions),
        "offensive_rebound_rate":_ratio(sums["offensive_rebounds"],_add(sums["offensive_rebounds"],opp["defensive_rebounds"])),
        "estimated_possessions":possessions,
        "offensive_rating":_scale(_ratio(sums["points"],possessions),100),
        "defensive_rating":_scale(_ratio(opp["points"],possessions),100),
        "opponent_efg_pct":_ratio(allowed_efg,opp["field_goals_attempted"]),
        "opponent_turnover_rate":_ratio(opp["turnovers"],possessions),
        "opponent_offensive_rebound_rate":_ratio(opp["offensive_rebounds"],_add(opp["offensive_rebounds"],sums["defensive_rebounds"])),
        "opponent_free_throw_rate":_ratio(opp["free_throws_attempted"],opp["field_goals_attempted"]),
        "method":"Box-derived estimates; 0.44 free-throw coefficient"}
    return clean_json(result)

def player_metrics(frame):
    if frame.empty:return []
    result=[]
    team_games=frame.groupby(["season","team_id"]).game_id.nunique().to_dict()
    playing=frame.copy()
    if "did_not_play" in playing:
        playing=playing[~playing.did_not_play.fillna(False).astype(bool)]
    if "minutes" in playing:
        playing=playing[pd.to_numeric(playing.minutes,errors="coerce").gt(0)]
    for (season,team,player),group in playing.groupby(["season","team_id","player_id"],sort=True):
        games=int(group.game_id.nunique())
        sums={c:_sum(group,c) for c in BOX_COLUMNS}
        ts_denominator=_scale(_add(sums["field_goals_attempted"],_scale(sums["free_throws_attempted"],.44)),2)
        result.append(clean_json({"player_id":str(player),"player_name":str(group.get("player_name",pd.Series(str(player),index=group.index)).iloc[-1]),
            "team_id":str(team),"season":int(season),"games_played":games,"team_games":int(team_games[(season,team)]),
            "minutes_total":sums["minutes"],"minutes_per_game":_ratio(sums["minutes"],games),
            "points_total":sums["points"],"points_per_game":_ratio(sums["points"],games),
            "rebounds_per_game":_ratio(sums["rebounds"],games),"assists_per_game":_ratio(sums["assists"],games),
            "ts_pct":_ratio(sums["points"],ts_denominator),"field_goals_attempted":sums["field_goals_attempted"]}))
    return sorted(result,key=lambda row:-(row["minutes_per_game"] or 0))

def pbp_summary(frame,team_id):
    if frame.empty:return None
    own=frame[frame.team_id.astype(str).eq(str(team_id))] if "team_id" in frame else frame.iloc[:0]
    if "scoring_play" not in own or "score_value" not in own:return None
    scoring=own[own.scoring_play.fillna(False).astype(bool)]
    points_by_period=None
    if "period_number" in scoring:
        points_by_period={str(int(period)):float(group.score_value.sum()) for period,group in scoring.groupby("period_number") if pd.notna(period)}
    return clean_json({"games":int(frame.game_id.nunique()),"events":len(own),"scoring_events":len(scoring),
                       "points_by_period":points_by_period,"scope":"Validated scoring events only"})
