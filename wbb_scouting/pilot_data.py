"""Six-team historical extraction; source identities and shot multiplicity survive."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .adapters import FAMILY,identifier,boolean,normalize_snapshot
from .sources import sha256_file
from .store import fingerprint,clean_json
from .validation import prepare_snapshot,validate_snapshot

SHOT_COUNTS={"field_goals_attempted":"fga","field_goals_made":"fgm",
             "three_point_field_goals_attempted":"three_pa","three_point_field_goals_made":"three_pm"}


def load_pilot_sources(raw_dir,config):
    import json
    root=Path(raw_dir)
    provenance=json.loads((root/"sources.json").read_text())
    sources={s["name"]:s for s in provenance["sources"]}
    frames={}
    for dataset,spec in config["sources"].items():
        source=sources.get(spec["name"])
        if not source or source.get("tag")!=spec["tag"] or source.get("season")!=config["season"] or source.get("family")!=FAMILY:
            raise ValueError("Unverified pilot source family/tag/season: "+dataset)
        expected="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"+spec["tag"]+"/"+spec["name"]
        if source.get("browser_download_url")!=expected:
            raise ValueError("Unexpected pilot source URL")
        path=root/spec["name"]
        if sha256_file(path)!=source["sha256"]:
            raise ValueError("Pilot source checksum mismatch: "+dataset)
        frames[dataset]=pd.read_parquet(path)
    return frames,provenance


def normalize_shots(frame,season):
    out=frame.copy()
    required={"season","game_id","team_id","athlete_id_1","period_number","clock_display_value",
              "type_text","score_value","scoring_play","coordinate_x_raw","coordinate_y_raw"}
    if not required.issubset(out):
        raise ValueError("Missing shot columns: "+", ".join(sorted(required-set(out))))
    if not pd.to_numeric(out.season,errors="coerce").eq(season).all():
        raise ValueError("Unexpected shot season")
    for col in ("game_id","team_id","athlete_id_1","athlete_id_2","type_id"):
        if col in out:out[col]=out[col].map(identifier).astype("string")
    if out[["game_id","team_id","athlete_id_1"]].isna().any().any():
        raise ValueError("Missing shot identity")
    out["season"]=season
    out["scoring_play"]=out.scoring_play.map(boolean).astype("boolean")
    out["score_value"]=pd.to_numeric(out.score_value,errors="coerce")
    out["period_number"]=pd.to_numeric(out.period_number,errors="coerce")
    if out.scoring_play.isna().any() or not out.score_value.isin([1,2,3]).all():
        raise ValueError("Invalid shot value/outcome")
    if (out.period_number.isna()|out.period_number.lt(1)|(out.period_number%1).ne(0)).any():
        raise ValueError("Invalid shot period")
    if out.clock_display_value.isna().any():
        raise ValueError("Missing shot clock")
    original=sorted(out.columns)
    out["_signature"]=[fingerprint(row) for row in clean_json(out[original].to_dict("records"))]
    out=out.sort_values("_signature",kind="stable").reset_index(drop=True)
    ordinal=out.groupby("_signature",sort=False).cumcount().astype(str)
    out["source_row_id"]=out["_signature"]+":"+ordinal
    out=out.drop(columns="_signature")
    out["source_family"]=FAMILY
    out["player_id"]=out.athlete_id_1
    out["is_field_goal"]=out.score_value.isin([2,3])
    out["made"]=out.scoring_play.astype(bool)
    x=pd.to_numeric(out.coordinate_x_raw,errors="coerce")
    y=pd.to_numeric(out.coordinate_y_raw,errors="coerce")
    # ESPN raw x is court width; raw y is measured from the basket.
    # Unlike a missing-coordinate sentinel, a small negative y can be in bounds.
    out["location_valid"]=out.is_field_goal & x.between(0,50) & y.between(-5.25,88.75)
    out["court_x"]=(x-25).where(out.location_valid)
    out["court_y"]=(y+5.25).where(out.location_valid)
    distance=np.sqrt((x-25)**2+y**2)
    out["shot_distance_ft"]=distance.where(out.location_valid)
    zone=np.select([out.score_value.eq(3),distance.le(4),
                    (x-25).abs().le(8)&y.between(-5.25,13.75)],
                   ["three","rim","paint"],default="midrange")
    out["shot_zone"]=pd.Series(zone,index=out.index).where(out.location_valid,"unlocated")
    return out.sort_values("source_row_id",kind="stable").reset_index(drop=True)


def _counts(frame):
    field=frame.loc[frame.is_field_goal]
    return {"fga":len(field),"fgm":int(field.made.sum()),
            "three_pa":int(field.score_value.eq(3).sum()),
            "three_pm":int((field.score_value.eq(3)&field.made).sum())}


def assess_shots(shots,team_boxes,player_boxes):
    """Require exact box denominators; missing or misattributed groups stay visible."""
    rows=[];accepted=set()
    groups={(str(g),str(t)):f for (g,t),f in shots.groupby(["game_id","team_id"],sort=False)}
    player_groups={(str(g),str(t)):f for (g,t),f in player_boxes.groupby(["game_id","team_id"],sort=False)}
    boxes={(str(r["game_id"]),str(r["team_id"])):r for r in team_boxes.to_dict("records")}
    for key in sorted(set(boxes)|set(groups)):
        field=groups.get(key,shots.iloc[:0]);observed=_counts(field);box=boxes.get(key);reasons=[]
        if box is None:reasons.append("validated team box unavailable")
        elif any(pd.isna(box.get(stat)) or observed[name]!=box[stat] for stat,name in SHOT_COUNTS.items()):
            reasons.append("field-goal counts disagree with team box")
        player_rows=player_groups.get(key,player_boxes.iloc[:0])
        if player_rows.empty:
            reasons.append("validated player box unavailable")
        else:
            by_player={str(pid):_counts(group) for pid,group in field.groupby("player_id")}
            expected={str(r["player_id"]):r for r in player_rows.to_dict("records")}
            for pid in sorted(set(expected)|set(by_player)):
                own=by_player.get(pid,{"fga":0,"fgm":0,"three_pa":0,"three_pm":0});p=expected.get(pid)
                if p is not None and own["fga"]==0 and p.get("did_not_play") is True:
                    continue
                if p is None or any(pd.isna(p.get(stat)) or own[name]!=p[stat] for stat,name in SHOT_COUNTS.items()):
                    reasons.append("field-goal attribution disagrees with player box");break
        status="withheld" if reasons else "validated"
        if not reasons:accepted.add(key)
        rows.append({"game_id":key[0],"team_id":key[1],"status":status,"reason":"; ".join(reasons),
                     "expected_fga":box.get("field_goals_attempted") if box else None,
                     "observed_fga":observed["fga"],"observed_fgm":observed["fgm"],
                     "located_fga":int((field.is_field_goal&field.location_valid).sum())})
    return pd.DataFrame(rows),accepted


def clean_pilot(raw,config,now):
    season=config["season"];ids={t["team_id"] for t in config["teams"]}
    source={"family":FAMILY,"league":"womens_college_basketball","season":season}
    teams=normalize_snapshot("teams",raw["teams"],source)
    teams["is_d1"]=teams.subdivision_id.eq("wbb:d1")
    schedule=normalize_snapshot("schedule",raw["schedule"],source)
    schedule=schedule[schedule.home_team_id.isin(ids)|schedule.away_team_id.isin(ids)].copy()
    game_ids=set(schedule.game_id)
    context={};quarantine=[]
    for name in ("team_box","player_box"):
        subset=raw[name][raw[name].game_id.map(identifier).isin(game_ids)].copy()
        normalized,identity_bad=prepare_snapshot(name,subset,source)
        validation=validate_snapshot(name,normalized,{"season":season,"teams":teams,"schedule":schedule,
            "team_box":context.get("team_box",pd.DataFrame()),"now":now})
        for gid,reasons in {**identity_bad,**validation["quarantine"]}.items():
            quarantine.append({"dataset":name,"game_id":gid,"reason":"; ".join(reasons)})
        context[name]=normalized[normalized.game_id.isin(validation["accepted_games"])].copy()
    clean={name:frame[frame.team_id.isin(ids)].copy() for name,frame in context.items()}
    roster=raw["rosters"][raw["rosters"].team_id.map(identifier).isin(ids)]
    clean["rosters"]=normalize_snapshot("rosters",roster,source)
    shot_raw=raw["shots"][raw["shots"].team_id.map(identifier).isin(ids)]
    shots=normalize_shots(shot_raw,season)
    shots=shots.merge(schedule[["game_id","game_date_time"]],on="game_id",how="left",validate="many_to_one")
    shot_quality,accepted=assess_shots(shots,clean["team_box"],clean["player_box"])
    shots["eligible_for_features"]=[(g,t) in accepted for g,t in zip(shots.game_id,shots.team_id)]
    clean["shots"]=shots
    clean["schedule"]=schedule
    clean["support_team_box"]=context["team_box"]
    clean["support_player_box"]=context["player_box"]
    clean["teams"]=teams[teams.team_id.isin(ids)].copy()
    return clean,{"shot_groups":shot_quality,"box_quarantine":pd.DataFrame(quarantine,columns=["dataset","game_id","reason"])}
