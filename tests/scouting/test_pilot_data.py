import pandas as pd
import pytest


def shot_rows():
    base={"season":2026,"game_id":4011,"team_id":25,"athlete_id_1":5001,
          "period_number":1,"clock_display_value":"9:20","type_id":558,
          "coordinate_x":38.75,"coordinate_y":0.0,"coordinate_x_raw":25.0,"coordinate_y_raw":3.0}
    return pd.DataFrame([
        {**base,"type_text":"LayUpShot","score_value":2,"scoring_play":True},
        {**base,"type_text":"JumpShot","clock_display_value":"8:45","score_value":3,"scoring_play":False},
        {**base,"type_text":"MadeFreeThrow","clock_display_value":"7:00","score_value":1,"scoring_play":True},
        {**base,"type_text":"MadeFreeThrow","clock_display_value":"7:00","score_value":1,"scoring_play":True},
    ])


def boxes():
    return pd.DataFrame([{"season":2026,"game_id":"4011","team_id":"25",
        "field_goals_attempted":2,"field_goals_made":1,
        "three_point_field_goals_attempted":1,"three_point_field_goals_made":0}])


def player_boxes():
    return boxes().assign(player_id="5001")


def test_identical_free_throws_are_preserved_and_order_does_not_change_ids():
    from wbb_scouting.pilot_data import normalize_shots
    a=normalize_shots(shot_rows(),2026)
    b=normalize_shots(shot_rows().sample(frac=1,random_state=7),2026)
    assert len(a)==4
    assert a.source_row_id.nunique()==4
    assert list(a.source_row_id)==list(b.source_row_id)
    assert int(a.is_field_goal.sum())==2
    assert a.team_id.eq("25").all()


def test_wrong_season_and_missing_shooter_are_rejected():
    from wbb_scouting.pilot_data import normalize_shots
    with pytest.raises(ValueError,match="season"):
        normalize_shots(shot_rows().assign(season=2027),2026)
    with pytest.raises(ValueError,match="identity"):
        normalize_shots(shot_rows().assign(athlete_id_1=None),2026)


def test_invalid_coordinates_retain_attempt_and_allow_valid_behind_basket_location():
    from wbb_scouting.pilot_data import normalize_shots
    raw=shot_rows()
    raw.loc[0,["coordinate_x_raw","coordinate_y_raw"]]=[-214748340,-214748365]
    raw.loc[1,["coordinate_x_raw","coordinate_y_raw"]]=[25,-5]
    out=normalize_shots(raw,2026)
    assert len(out)==4
    assert not out.loc[out.type_text.eq("LayUpShot"),"location_valid"].any()
    assert out.loc[out.type_text.eq("JumpShot"),"location_valid"].all()
    assert out.loc[out.type_text.eq("JumpShot"),"court_y"].iloc[0]==.25


def test_incomplete_shot_group_is_withheld_without_erasing_clean_records():
    from wbb_scouting.pilot_data import normalize_shots,assess_shots
    shots=normalize_shots(shot_rows(),2026)
    quality,accepted=assess_shots(shots,boxes(),player_boxes())
    assert accepted=={("4011","25")}
    assert quality.iloc[0]["observed_fga"]==2
    partial=shots[~shots.type_text.eq("JumpShot")]
    quality,accepted=assess_shots(partial,boxes(),player_boxes())
    assert accepted==set()
    assert quality.iloc[0]["status"]=="withheld"
    assert "field-goal" in quality.iloc[0]["reason"]
    assert len(partial)==3


def test_missing_shot_game_and_wrong_player_attribution_are_visible():
    from wbb_scouting.pilot_data import normalize_shots,assess_shots
    shots=normalize_shots(shot_rows(),2026)
    quality,accepted=assess_shots(shots.iloc[:0],boxes(),player_boxes())
    assert not accepted and quality.iloc[0]["observed_fga"]==0
    quality,accepted=assess_shots(shots.assign(player_id="999"),boxes(),player_boxes())
    assert not accepted
    assert "player" in quality.iloc[0]["reason"]


def test_coordinate_conversion_matches_source_home_and_away_convention():
    from wbb_scouting.pilot_data import normalize_shots
    out=normalize_shots(shot_rows(),2026)
    row=out.loc[out.type_text.eq("LayUpShot")].iloc[0]
    assert row.court_x==0
    assert row.court_y==8.25
    assert row.shot_zone=="rim"


def test_dnp_placeholder_without_shooting_stats_does_not_withhold_complete_game():
    from wbb_scouting.pilot_data import normalize_shots,assess_shots
    shots=normalize_shots(shot_rows(),2026)
    dnp={"game_id":"4011","team_id":"25","player_id":"5002","did_not_play":True,
         "field_goals_attempted":None,"field_goals_made":None,
         "three_point_field_goals_attempted":None,"three_point_field_goals_made":None}
    players=pd.concat([player_boxes(),pd.DataFrame([dnp])],ignore_index=True)
    quality,accepted=assess_shots(shots,boxes(),players)
    assert accepted=={("4011","25")}
