import pandas as pd
import pytest
from .helpers import team_boxes,players
from .test_pilot_data import shot_rows


def clean_fixture():
    from wbb_scouting.pilot_data import normalize_shots
    b=team_boxes().assign(season=2026,game_date_time="2026-03-01T20:00:00Z")
    p=players().assign(season=2026,game_date_time="2026-03-01T20:00:00Z")
    s=normalize_shots(shot_rows(),2026).assign(team_id="A",player_id="p1",game_id="g1",
        game_date_time="2026-03-01T20:00:00Z",eligible_for_features=True)
    return {"team_box":b[b.team_id.eq("A")],"player_box":p[p.team_id.eq("A")],
            "support_team_box":b,"rosters":pd.DataFrame([{"season":2026,"team_id":"A","player_id":"p0","player_name":"Roster Only","jersey":"0","position_abbreviation":"G"}]),"shots":s}


def config():
    return {"season":2026,"teams":[{"team_id":"A","name":"Alpha","color":"#123456"}]}


def test_opponent_rows_outside_featured_cohort_supply_correct_denominators():
    from wbb_scouting.pilot_features import process_pilot,build_pilot_features
    clean=clean_fixture();processed=process_pilot(clean,config())
    assert set(processed["team_games"].team_id)=={"A"}
    assert processed["team_games"].opponent_points.iloc[0]==8
    features=build_pilot_features(processed,clean,config())
    summary=features["team_summary"].query("window == 'season'").iloc[0]
    assert summary.offensive_rating==pytest.approx(100*10/9.88)
    assert summary.offensive_rebound_rate==pytest.approx(1/5)


def test_roster_union_keeps_box_only_and_roster_only_players():
    from wbb_scouting.pilot_features import process_pilot,build_pilot_features
    clean=clean_fixture();p=process_pilot(clean,config())
    roster=p["roster"]
    assert set(roster.player_id)=={"p0","p1"}
    assert roster.set_index("player_id").loc["p1","roster_status"]=="box_only"
    assert roster.set_index("player_id").loc["p0","roster_status"]=="roster_only"
    f=build_pilot_features(p,clean,config())["player_summary"].query("window == 'season'").set_index("player_id")
    assert f.loc["p1","games_played"]==1
    assert f.loc["p0","games_played"]==0
    assert pd.isna(f.loc["p0","points_per_game"])


def test_shot_features_exclude_withheld_groups_and_free_throws():
    from wbb_scouting.pilot_features import process_pilot,build_pilot_features
    clean=clean_fixture();p=process_pilot(clean,config())
    f=build_pilot_features(p,clean,config())
    zones=f["shot_zones"].query("window == 'season' and player_id == ''")
    assert zones.attempts.sum()==2
    assert zones.makes.sum()==1
    clean["shots"]["eligible_for_features"]=False
    p=process_pilot(clean,config());f=build_pilot_features(p,clean,config())
    assert f["shot_zones"].attempts.sum()==0
    assert f["team_summary"].shot_games.sum()==0


def test_team_window_limits_player_features_to_same_last_five_games():
    from wbb_scouting.pilot_features import process_pilot,build_pilot_features
    clean=clean_fixture();all_boxes=[];all_players=[]
    for i in range(6):
        date=f"2026-03-{i+1:02}T20:00:00Z"
        all_boxes.append(team_boxes("g"+str(i)).assign(season=2026,game_date_time=date))
        all_players.append(players("g"+str(i)).iloc[:1].assign(season=2026,game_date_time=date,player_id="p1" if i==0 else "p2"))
    clean["support_team_box"]=pd.concat(all_boxes,ignore_index=True)
    clean["team_box"]=clean["support_team_box"].query("team_id == 'A'")
    clean["player_box"]=pd.concat(all_players,ignore_index=True)
    p=process_pilot(clean,config());f=build_pilot_features(p,clean,config())
    recent=f["player_summary"].query("window == 'last_five'").set_index("player_id")
    assert recent.loc["p1","games_played"]==0
    assert recent.loc["p2","games_played"]==5
    assert f["team_summary"].query("window == 'last_five'").games.iloc[0]==5


def test_team_features_aggregate_counts_not_game_percentages():
    from wbb_scouting.pilot_features import process_pilot,build_pilot_features
    clean=clean_fixture()
    b1=clean["support_team_box"].copy();b1.loc[b1.team_id.eq("A"),["field_goals_attempted","field_goals_made","three_point_field_goals_made"]]=[10,5,2]
    b2=b1.assign(game_id="g2",game_date_time="2026-03-02T20:00:00Z")
    b2.loc[b2.team_id.eq("A"),["field_goals_attempted","field_goals_made","three_point_field_goals_made"]]=[2,0,0]
    clean["support_team_box"]=pd.concat([b1,b2],ignore_index=True)
    clean["team_box"]=clean["support_team_box"].query("team_id == 'A'")
    p=process_pilot(clean,config());f=build_pilot_features(p,clean,config())
    assert f["team_summary"].query("window == 'season'").efg_pct.iloc[0]==pytest.approx(6/12)
