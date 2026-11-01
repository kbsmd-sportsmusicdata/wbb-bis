import pandas as pd
import pytest
from .helpers import team_boxes,players

def test_team_rates_match_hand_calculated_denominators():
    from wbb_scouting.metrics import attach_opponents,team_metrics
    frame=attach_opponents(team_boxes())
    m=team_metrics(frame[frame.team_id.eq("A")])
    assert m["games"]==1 and m["points_per_game"]==10
    assert m["efg_pct"]==pytest.approx(4.5/8)
    assert m["ts_pct"]==pytest.approx(10/(2*(8+.44*2)))
    assert m["estimated_possessions"]==pytest.approx(9.88)
    assert m["offensive_rating"]==pytest.approx(100*10/9.88)
    assert m["defensive_rating"]==pytest.approx(100*8/9.88)
    assert m["turnover_rate"]==pytest.approx(2/9.88)
    assert m["offensive_rebound_rate"]==pytest.approx(1/5)
    assert m["free_throw_rate"]==pytest.approx(2/8)

def test_aggregate_efficiency_is_weighted_by_attempts():
    from wbb_scouting.metrics import team_metrics
    f=team_boxes().iloc[:1].copy()
    f["field_goals_attempted"]=10;f["field_goals_made"]=5;f["three_point_field_goals_made"]=2
    second=f.assign(game_id="g2",field_goals_attempted=2,field_goals_made=0,three_point_field_goals_made=0)
    result=team_metrics(pd.concat([f,second],ignore_index=True))
    assert result["efg_pct"]==pytest.approx(6/12)

def test_empty_and_missing_denominators_are_unavailable():
    from wbb_scouting.metrics import team_metrics
    assert team_metrics(pd.DataFrame())["efg_pct"] is None
    f=team_boxes().iloc[:1].assign(field_goals_attempted=0)
    assert team_metrics(f)["efg_pct"] is None
    assert team_metrics(f)["estimated_possessions"] is None

def test_dnp_does_not_become_a_played_game_or_injury_claim():
    from wbb_scouting.metrics import player_metrics
    f=players().iloc[:1].copy()
    dnp=f.assign(game_id="g2",did_not_play=True,minutes=None,points=None)
    result=player_metrics(pd.concat([f,dnp],ignore_index=True))
    assert result[0]["games_played"]==1 and result[0]["team_games"]==2
    assert result[0]["minutes_per_game"]==40
    assert "injury" not in result[0]
