from datetime import datetime,timezone
import pandas as pd
from .helpers import team_boxes,players,pbp,schedule

def clock(value):
    return datetime.fromisoformat(value.replace("Z","+00:00"))

def cohort():
    return pd.DataFrame([{"team_id":"A","team_name":"Alpha","season":2027,"conference":"ACC","is_d1":True,"inclusion_reason":"ACC","selection_status":"pending_net","net_rank":None,"ranking_date":None,"ranking_season":2027},{"team_id":"B","team_name":"Beta","season":2027,"conference":"Other","is_d1":True,"inclusion_reason":"NET top 50","selection_status":"current_net","net_rank":10,"ranking_date":"2026-11-01","ranking_season":2027}])

def tables():
    upcoming=schedule("g2").assign(game_date_time="2026-11-03T20:00:00Z",game_date="2026-11-03",completed=False,home_score=0,away_score=0)
    return {"schedule":pd.concat([schedule(),upcoming],ignore_index=True),"team_box":team_boxes(),"player_box":players(),"pbp":pbp(),"rosters":pd.DataFrame()}

def test_pregame_excludes_target_game_and_future_information():
    from wbb_scouting.reports import build_reports
    t=tables()
    future=team_boxes("g2").assign(game_date_time="2026-11-03T20:00:00Z",points=999)
    t["team_box"]=pd.concat([t["team_box"],future],ignore_index=True)
    result=build_reports(t,cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    report=next(r for r in result if r["mode"]=="pregame" and r["team_id"]=="A")
    assert report["profile"]["season"]["games"]==1
    assert report["profile"]["season"]["points_per_game"]==10
    assert report["game_id"]=="g2"

def test_unchanged_refresh_preserves_existing_edition_and_cutoff():
    from wbb_scouting.reports import build_reports
    first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    again=build_reports(tables(),cohort(),clock("2026-11-02T16:17:00Z"),first,{},bootstrap=False)
    assert len(again)==len(first)
    assert sum(r["mode"]=="pregame" for r in first)==2
    assert again==first

def test_postgame_uses_last_issued_pregame_baseline_and_keeps_it():
    from wbb_scouting.reports import build_reports
    first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    t=tables()
    t["schedule"].loc[t["schedule"].game_id.eq("g2"),["completed","home_score","away_score"]]=[True,10,8]
    t["team_box"]=pd.concat([t["team_box"],team_boxes("g2").assign(game_date_time="2026-11-03T20:00:00Z")],ignore_index=True)
    t["player_box"]=pd.concat([t["player_box"],players("g2").assign(game_date_time="2026-11-03T20:00:00Z")],ignore_index=True)
    result=build_reports(t,cohort(),clock("2026-11-04T14:17:00Z"),first,{},bootstrap=False)
    post=next(r for r in result if r["mode"]=="postgame" and r["game_id"]=="g2" and r["team_id"]=="A")
    assert post["baseline_origin"]=="issued_pregame"
    assert post["baseline"]["season"]["games"]==1
    assert post["status"]=="box_score_only"
    assert all(r in result for r in first)

def test_weekend_games_are_prepared_before_next_primary_check():
    from wbb_scouting.reports import build_reports
    t=tables()
    t["schedule"]=schedule("weekend").assign(game_date_time="2026-11-08T20:00:00Z",game_date="2026-11-08",completed=False)
    result=build_reports(t,cohort(),clock("2026-11-06T14:17:00Z"),[],{},bootstrap=False)
    assert len(result)==2
    assert all(r["game_id"]=="weekend" for r in result)

def test_empty_current_season_preserves_historical_context_as_separate():
    from wbb_scouting.reports import build_reports
    t=tables();t["schedule"]=t["schedule"][t["schedule"].game_id.eq("g2")]
    t["team_box"]=team_boxes().assign(season=2026,game_date_time="2026-03-01T20:00:00Z")
    t["player_box"]=pd.DataFrame();t["pbp"]=pd.DataFrame()
    result=build_reports(t,cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    r=result[0]
    assert r["profile"]["season"]["games"]==0
    assert r["historical_reference"]["season"]==2026
    assert r["historical_reference"]["metrics"]["games"]==1
    assert r["status"]=="historical_context_only"

def test_bootstrap_postgames_and_missing_pbp_are_explicit():
    from wbb_scouting.reports import build_reports
    t=tables();t["pbp"]=pd.DataFrame()
    result=build_reports(t,cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=True)
    post=next(r for r in result if r["mode"]=="postgame")
    assert post["baseline_origin"]=="reconstructed"
    assert post["status"]=="box_score_only"
    assert post["pbp"] is None

def test_late_arriving_completed_game_is_reported_after_bootstrap_window():
    from wbb_scouting.reports import build_reports
    result=build_reports(tables(),cohort(),clock("2026-11-20T14:17:00Z"),[],{},bootstrap=False)
    assert any(r["mode"]=="postgame" and r["game_id"]=="g1" for r in result)

def test_changed_tip_creates_revision_without_overwriting_original():
    from wbb_scouting.reports import build_reports
    first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    t=tables();t["schedule"].loc[t["schedule"].game_id.eq("g2"),"game_date_time"]="2026-11-03T21:00:00Z"
    changed=build_reports(t,cohort(),clock("2026-11-02T16:17:00Z"),first,{},bootstrap=False)
    assert len(changed)==len(first)+2 and all(r in changed for r in first)

def test_past_tip_cannot_be_issued_as_new_pregame():
    from wbb_scouting.reports import build_reports
    t=tables();t["schedule"]=t["schedule"][t["schedule"].game_id.eq("g2")].assign(game_date_time="2026-11-02T13:00:00Z")
    result=build_reports(t,cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    assert not any(r["mode"]=="pregame" for r in result)

def test_bootstrap_does_not_mass_backfill_old_postgames():
    from wbb_scouting.reports import build_reports
    result=build_reports(tables(),cohort(),clock("2026-11-20T14:17:00Z"),[],{},bootstrap=True)
    assert not any(r["mode"]=="postgame" and r["game_id"]=="g1" for r in result)

def test_prepared_team_still_gets_postgame_after_leaving_current_cohort():
    from wbb_scouting.reports import build_reports
    first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    t=tables()
    t["schedule"].loc[t["schedule"].game_id.eq("g2"),["completed","home_score","away_score"]]=[True,10,8]
    t["team_box"]=pd.concat([t["team_box"],team_boxes("g2").assign(game_date_time="2026-11-03T20:00:00Z")],ignore_index=True)
    t["player_box"]=pd.concat([t["player_box"],players("g2").assign(game_date_time="2026-11-03T20:00:00Z")],ignore_index=True)
    reduced=cohort().iloc[:1]
    result=build_reports(t,reduced,clock("2026-11-04T14:17:00Z"),first,{},bootstrap=False)
    assert any(r["mode"]=="postgame" and r["game_id"]=="g2" and r["team_id"]=="B" for r in result)

def test_corrected_postgame_creates_new_edition_without_rewriting_old():
    from wbb_scouting.reports import build_reports
    t=tables();t["pbp"]=pd.DataFrame()
    first=build_reports(t,cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=True)
    t["team_box"].loc[t["team_box"].team_id.eq("A"),["points","field_goals_made"]]=[12,5]
    t["team_box"].loc[t["team_box"].team_id.eq("B"),"opponent_points"]=12
    t["schedule"].loc[t["schedule"].game_id.eq("g1"),"home_score"]=12
    again=build_reports(t,cohort(),clock("2026-11-02T16:17:00Z"),first,{},bootstrap=False)
    editions=[r for r in again if r["mode"]=="postgame" and r["game_id"]=="g1" and r["team_id"]=="A"]
    assert len(editions)==2
    assert editions[0]["game_metrics"]["points_total"]==10
    assert editions[1]["game_metrics"]["points_total"]==12

def test_pbp_period_scoring_uses_scoring_flags():
    from wbb_scouting.metrics import pbp_summary
    events=pbp()
    events["period_number"]=[1,1,1,1,2,2,2]
    miss=events.iloc[-1].copy();miss["event_id"]="miss";miss["scoring_play"]=False;miss["score_value"]=3
    result=pbp_summary(pd.concat([events,pd.DataFrame([miss])],ignore_index=True),"A")
    assert result["points_by_period"]=={"1":5,"2":5}

def test_return_to_earlier_tip_still_creates_a_new_current_edition():
    from wbb_scouting.reports import build_reports
    first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    t=tables();t["schedule"].loc[t["schedule"].game_id.eq("g2"),"game_date_time"]="2026-11-03T21:00:00Z"
    second=build_reports(t,cohort(),clock("2026-11-02T16:17:00Z"),first,{},bootstrap=False)
    reverted=build_reports(tables(),cohort(),clock("2026-11-03T14:17:00Z"),second,{},bootstrap=False)
    editions=[r for r in reverted if r["mode"]=="pregame" and r["team_id"]=="A"]
    assert len(editions)==3 and len({r["edition_id"] for r in editions})==3
    assert editions[-1]["tip_at"].startswith("2026-11-03T20:00:00")

def test_refresh_does_not_backfill_old_games_already_seen_at_bootstrap():
    from wbb_scouting.reports import build_reports
    initial=build_reports(tables(),cohort(),clock("2026-11-20T14:17:00Z"),[],{},bootstrap=True)
    next_run=build_reports(tables(),cohort(),clock("2026-11-23T14:17:00Z"),initial,{"new_game_ids":[]},bootstrap=False)
    assert not any(r["mode"]=="postgame" and r["game_id"]=="g1" for r in next_run)

def test_postgame_cutoff_distinguishes_report_time_from_baseline_time():
    from wbb_scouting.reports import build_reports
    result=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=True)
    r=next(r for r in result if r["mode"]=="postgame")
    assert r["cutoff_at"]=="2026-11-02T14:17:00+00:00"
    assert r["baseline_cutoff_at"]=="2026-11-01T20:00:00+00:00"

def test_retained_game_keeps_its_original_source_snapshot_reference():
    import json
    from wbb_scouting.reports import build_reports
    snapshots={"game_sources":{"team_box":{"2027:g1":"original-snapshot","2027:other":"unrelated-new-snapshot"}},"dataset_snapshots":{"team_box":"unrelated-new-snapshot"}}
    result=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],snapshots,bootstrap=False)
    pre=next(r for r in result if r["mode"]=="pregame")
    encoded=json.dumps(pre["source_references"])
    assert "original-snapshot" in encoded
    assert "unrelated-new-snapshot" not in encoded

def test_historical_comparison_keeps_both_teams_in_the_same_reference_season():
    from wbb_scouting.reports import build_reports
    t=tables();t["schedule"]=t["schedule"][t["schedule"].game_id.eq("g2")]
    t["team_box"]=team_boxes().assign(season=2026,game_date_time="2026-03-01T20:00:00Z")
    t["player_box"]=pd.DataFrame();t["pbp"]=pd.DataFrame()
    result=build_reports(t,cohort(),clock("2026-11-02T14:17:00Z"),[],{},bootstrap=False)
    r=next(r for r in result if r["team_id"]=="A")
    assert r["opponent_historical_reference"]["season"]==r["historical_reference"]["season"]==2026
    assert r["opponent_historical_reference"]["metrics"]["points_per_game"]==8

def test_unrelated_snapshot_change_does_not_reissue_unchanged_report():
    from wbb_scouting.reports import build_reports
    first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{"dataset_snapshots":{"team_box":"old"}},bootstrap=False)
    second=build_reports(tables(),cohort(),clock("2026-11-02T16:17:00Z"),first,{"dataset_snapshots":{"team_box":"new"}},bootstrap=False)
    assert second==first

def test_postgame_preserves_issued_opponent_baseline_and_sources():
 from wbb_scouting.reports import build_reports
 first=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{"dataset_snapshots":{"team_box":"original"}},bootstrap=False)
 t=tables();t["schedule"].loc[t["schedule"].game_id.eq("g2"),["completed","home_score","away_score"]]=[True,10,8]
 t["team_box"]=pd.concat([t["team_box"],team_boxes("g2").assign(game_date_time="2026-11-03T20:00:00Z")],ignore_index=True)
 result=build_reports(t,cohort(),clock("2026-11-04T14:17:00Z"),first,{},bootstrap=False)
 post=next(r for r in result if r["mode"]=="postgame" and r["game_id"]=="g2" and r["team_id"]=="A")
 assert post["opponent_baseline"]["season"]["points_per_game"]==8
 assert post["baseline_source_references"]["team_box"]=="original"

def test_rescheduled_target_and_other_future_games_cannot_enter_baseline():
 from wbb_scouting.reports import build_reports
 t=tables()
 t["schedule"].loc[t["schedule"].game_id.eq("g2"),["completed","home_score","away_score"]]=[True,10,8]
 future=schedule("g3").assign(game_date_time="2026-11-04T20:00:00Z",completed=True)
 t["schedule"]=pd.concat([t["schedule"],future],ignore_index=True)
 # Source timestamps predate the revised official schedule for both games.
 t["team_box"]=pd.concat([team_boxes(),team_boxes("g2"),team_boxes("g3")],ignore_index=True)
 result=build_reports(t,cohort(),clock("2026-11-05T14:17:00Z"),[],{},bootstrap=True)
 target=next(r for r in result if r["team_id"]=="A" and r["game_id"]=="g2")
 assert target["baseline"]["input_game_ids"]["team_box"]==["g1"]
 assert target["baseline"]["season"]["games"]==1

def test_future_net_source_is_not_cited_by_pending_cohort_report():
 from wbb_scouting.reports import build_reports
 result=build_reports(tables(),cohort(),clock("2026-11-02T14:17:00Z"),[],{"game_sources":{},"dataset_snapshots":{"2027:net":"future-snapshot"},"active_net_snapshot":{}},bootstrap=False)
 assert all("2027:net" not in r["source_references"] for r in result)
