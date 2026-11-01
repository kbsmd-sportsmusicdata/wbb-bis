import pandas as pd

def team_boxes(game="g1"):
    common={"source_family":"sdv_espn_wbb","season":2027,"game_id":game,"game_date_time":"2026-11-01T20:00:00Z","game_date":"2026-11-01","field_goals_attempted":8,"three_point_field_goals_made":1,"three_point_field_goals_attempted":3,"free_throws_made":1,"free_throws_attempted":2,"offensive_rebounds":1,"defensive_rebounds":4,"turnovers":2}
    return pd.DataFrame([{**common,"team_id":"A","opponent_team_id":"B","points":10,"opponent_points":8,"field_goals_made":4},{**common,"team_id":"B","opponent_team_id":"A","points":8,"opponent_points":10,"field_goals_made":3}])

def schedule(*games):
    return pd.DataFrame([{"source_family":"sdv_espn_wbb","season":2027,"game_id":g,"game_date_time":"2026-11-01T20:00:00Z","game_date":"2026-11-01","home_team_id":"A","away_team_id":"B","completed":True,"home_score":10,"away_score":8} for g in (games or ("g1",))])

def players(game="g1"):
    f=team_boxes(game).drop(columns=["opponent_points"])
    f["player_id"]=["p1","p2"]
    f["player_name"]=["Player One","Player Two"]
    f["minutes"]=40
    f["did_not_play"]=False
    return f

def pbp(game="g1"):
    rows=[]
    h=a=0
    for i,(team,points) in enumerate([("A",2),("B",2),("A",3),("B",3),("A",2),("B",3),("A",3)]):
        h+=points if team=="A" else 0
        a+=points if team=="B" else 0
        rows.append({"source_family":"sdv_espn_wbb","season":2027,"game_id":game,"game_date_time":"2026-11-01T20:00:00Z","event_id":str(i),"game_play_number":str(i),"team_id":team,"home_team_id":"A","away_team_id":"B","home_score":h,"away_score":a,"scoring_play":True,"score_value":points})
    return pd.DataFrame(rows)
