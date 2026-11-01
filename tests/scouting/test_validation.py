from wbb_scouting.validation import validate_snapshot
import pandas as pd
from .helpers import team_boxes,players,pbp,schedule

def context():
    return {"season":2027,"schedule":schedule(),"team_box":team_boxes(),"player_box":players(),"now":"2026-11-02T14:17:00Z"}

def test_reciprocal_team_rows_and_scores_validate():
    from wbb_scouting.validation import validate_snapshot
    result=validate_snapshot("team_box",team_boxes(),context())
    assert result["accepted_games"]=={"g1"}
    assert not result["quarantine"]

def test_one_sided_or_negative_box_is_quarantined():
    from wbb_scouting.validation import validate_snapshot
    assert not validate_snapshot("team_box",team_boxes().iloc[:1],context())["accepted_games"]
    assert not validate_snapshot("team_box",team_boxes().assign(field_goals_attempted=-1),context())["accepted_games"]

def test_team_points_must_match_schedule_and_shooting_totals():
    from wbb_scouting.validation import validate_snapshot
    result=validate_snapshot("team_box",team_boxes().assign(points=99),context())
    assert "g1" in result["quarantine"]

def test_player_point_totals_reconcile_to_team_box():
    from wbb_scouting.validation import validate_snapshot
    assert validate_snapshot("player_box",players(),context())["accepted_games"]=={"g1"}
    assert not validate_snapshot("player_box",players().assign(points=1),context())["accepted_games"]

def test_scoring_flag_prevents_misses_being_counted_as_points():
    from wbb_scouting.validation import validate_snapshot
    events=pbp()
    miss=events.iloc[-1].copy()
    miss["event_id"]="miss";miss["game_play_number"]="9";miss["scoring_play"]=False;miss["score_value"]=3
    result=validate_snapshot("pbp",pd.concat([events,pd.DataFrame([miss])],ignore_index=True),context())
    assert result["accepted_games"]=={"g1"}

def test_truncated_pbp_game_is_quarantined():
    from wbb_scouting.validation import validate_snapshot
    result=validate_snapshot("pbp",pbp().iloc[:-1],context())
    assert not result["accepted_games"]

def test_coverage_finds_earlier_missing_game():
    from wbb_scouting.validation import validate_snapshot
    ctx=context();ctx["schedule"]=schedule("g1","g2")
    result=validate_snapshot("team_box",team_boxes("g2"),ctx)
    assert result["coverage"]["expected_games"]==2
    assert result["coverage"]["available_games"]==1
    assert result["coverage"]["missing_game_ids"]==["g1"]

def test_raw_null_player_id_quarantines_only_affected_game():
    from wbb_scouting.validation import prepare_snapshot
    raw=pd.DataFrame({"game_id":["g1","g2"],"season":[2027,2027],"team_id":["A","A"],"athlete_id":["p1",None],"points":[10,None],"did_not_play":[False,True],"game_date_time":["2026-11-01T20:00:00Z"]*2})
    out,quarantine=prepare_snapshot("player_box",raw,{"season":2027,"family":"sdv_espn_wbb"})
    assert out.game_id.tolist()==["g1"]
    assert "g2" in quarantine

def test_raw_conflicting_keys_quarantine_game_instead_of_last_row_wins():
    from wbb_scouting.validation import prepare_snapshot
    raw=pd.DataFrame({"game_id":["g1","g1"],"season":[2027,2027],"team_id":["A","A"],"athlete_id":["p1","p1"],"points":[10,12],"game_date_time":["2026-11-01T20:00:00Z"]*2})
    out,quarantine=prepare_snapshot("player_box",raw,{"season":2027,"family":"sdv_espn_wbb"})
    assert out.empty and "g1" in quarantine

def test_division_reference_flags_keep_unknown_opponents_explicit():
    from wbb_scouting.validation import annotate_division
    known=pd.DataFrame({"team_id":["A"],"is_d1":[True]})
    out=annotate_division(team_boxes(),known)
    row=out[out.team_id.eq("A")].iloc[0]
    assert row.team_division=="I"
    assert row.opponent_division=="UNKNOWN_NEEDS_REVIEW"
    assert not row.comparison_eligible

def test_player_third_team_is_quarantined():
 frame=players();extra=frame.iloc[[0]].copy();extra["team_id"]="C";extra["player_id"]="p3";extra["points"]=0;extra["field_goals_made"]=0;extra["three_point_field_goals_made"]=0;extra["free_throws_made"]=0
 result=validate_snapshot("player_box",pd.concat([frame,extra]),{"season":2027,"schedule":schedule(),"team_box":team_boxes(),"now":"2026-11-02T00:00:00Z"})
 assert not result["accepted_games"]

def test_pbp_unknown_scoring_team_is_quarantined():
 frame=pbp();extra=frame.iloc[[0]].copy();extra["team_id"]="C";extra["event_id"]="99";extra["game_play_number"]="99";extra["home_score"]=10;extra["away_score"]=8
 result=validate_snapshot("pbp",pd.concat([frame,extra]),{"season":2027,"schedule":schedule(),"now":"2026-11-02T00:00:00Z"})
 assert not result["accepted_games"]

def test_completed_future_game_is_withheld():
 result=validate_snapshot("team_box",team_boxes(),{"season":2027,"schedule":schedule(),"now":"2026-11-01T19:00:00Z"})
 assert not result["accepted_games"]

def test_absent_dataset_still_lists_expected_missing_games():
 result=validate_snapshot("team_box",pd.DataFrame(),{"season":2027,"schedule":schedule(),"now":"2026-11-02T00:00:00Z"})
 assert result["coverage"]["expected_games"]==1
 assert result["coverage"]["missing_game_ids"]==["g1"]
