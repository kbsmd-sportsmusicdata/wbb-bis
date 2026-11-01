from datetime import datetime,timezone
import json
import pandas as pd
import pytest

def teams():
    return pd.DataFrame([{"team_id":str(i),"team_name":"Team "+str(i),"season":2027,"conference":"ACC" if i==3 else "Other","is_d1":True} for i in range(1,61)]+[{"team_id":"999","team_name":"ACC Extra","season":2027,"conference":"ACC","is_d1":True}])

def publication(season=2027,date="2026-12-04"):
    return {"season":season,"publication_date":date,"source_url":"https://www.ncaa.com/rankings/basketball-women/d1/ncaa-womens-basketball-net-rankings","verified":True,"id_namespace":"espn"}

def net(season=2027,date="2026-12-04"):
    from wbb_scouting.references import validate_net
    return validate_net(pd.DataFrame({"team_id":[str(i) for i in range(1,61)],"net_rank":list(range(1,61))}),publication(season,date),teams())

def test_net_top50_union_acc_without_duplicate_teams():
    from wbb_scouting.references import select_cohort
    cohort=select_cohort(net(),teams(),datetime(2026,12,5,tzinfo=timezone.utc),2027)
    assert len(cohort)==51 and cohort.team_id.is_unique
    assert cohort.set_index("team_id").loc["3","inclusion_reason"]=="NET top 50 + ACC"
    assert cohort.set_index("team_id").loc["999","inclusion_reason"]=="ACC"
    assert "51" not in set(cohort.team_id)

def test_unpublished_net_produces_honest_acc_only_pending_selection():
    from wbb_scouting.references import select_cohort
    cohort=select_cohort(None,teams(),datetime(2026,11,1,tzinfo=timezone.utc),2027)
    assert set(cohort.team_id)=={"3","999"}
    assert cohort.selection_status.eq("pending_net").all()
    assert cohort.net_rank.isna().all()

def test_prior_year_net_selects_provisional_teams_without_current_rank():
    from wbb_scouting.references import select_cohort
    cohort=select_cohort(None,teams(),datetime(2026,11,1,tzinfo=timezone.utc),2027,historical_net=net(2026,"2026-04-05"))
    assert len(cohort)==51
    assert cohort.net_rank.isna().all()
    assert cohort.ranking_season.eq(2026).all()
    assert cohort.selection_status.eq("provisional_prior_season").all()

def test_future_ranking_cannot_change_pregame_cohort():
    from wbb_scouting.references import select_cohort
    cohort=select_cohort(net(),teams(),datetime(2026,12,1,tzinfo=timezone.utc),2027)
    assert set(cohort.team_id)=={"3","999"}

@pytest.mark.parametrize("issue",["unverified","namespace","unknown_team","duplicate_rank","missing_top50","duplicate_team","nonintegral_rank"])
def test_net_rejects_ambiguous_or_unreviewed_snapshots(issue):
    from wbb_scouting.references import validate_net
    raw=pd.DataFrame({"team_id":[str(i) for i in range(1,61)],"net_rank":list(range(1,61))})
    meta=publication()
    if issue=="unverified":meta["verified"]=False
    if issue=="namespace":meta["id_namespace"]="ncaa"
    if issue=="unknown_team":raw.loc[0,"team_id"]="unknown"
    if issue=="duplicate_rank":raw.loc[0,"net_rank"]=2
    if issue=="missing_top50":raw=raw.iloc[1:]
    if issue=="duplicate_team":raw.loc[0,"team_id"]="2"
    if issue=="nonintegral_rank":raw["net_rank"]=raw["net_rank"].astype(float);raw.loc[0,"net_rank"]=1.5
    with pytest.raises(ValueError):
        validate_net(raw,meta,teams())

def test_team_reference_is_season_specific(tmp_path):
    from wbb_scouting.references import load_team_reference
    path=tmp_path/"teams.json"
    path.write_text(json.dumps({"season":2027,"source_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/wbb_groups","teams":teams().to_dict("records")}))
    loaded=load_team_reference(path,2027)
    assert len(loaded)==61
    with pytest.raises(ValueError,match="season"):
        load_team_reference(path,2026)

def test_stale_current_ranking_is_dated_and_flagged():
    from wbb_scouting.references import select_cohort
    cohort=select_cohort(net(),teams(),datetime(2026,12,12,tzinfo=timezone.utc),2027)
    assert cohort.selection_status.eq("stale_net").all()
    assert cohort.ranking_date.eq("2026-12-04").all()

def test_official_net_parser_rejects_challenge_instead_of_empty_success():
    from wbb_scouting.references import parse_net_document
    from wbb_scouting.sources import SourceUnavailable
    with pytest.raises(SourceUnavailable,match="verification"):
        parse_net_document(b"JavaScript is disabled. We need to verify that you are not a robot.","text/html",publication())

def test_future_publication_date_cannot_be_backdated_by_first_seen_time():
    from wbb_scouting.references import validate_net
    raw=pd.DataFrame({"team_id":[str(i) for i in range(1,61)],"net_rank":list(range(1,61))})
    meta=publication(date="2026-12-08")
    meta["first_seen_at"]="2026-12-05T14:17:00Z"
    with pytest.raises(ValueError,match="publication"):
        validate_net(raw,meta,teams())

def test_official_html_parser_uses_explicit_reviewed_school_mapping():
    from wbb_scouting.references import parse_net_document
    html=b'<table><tr><th>Rank</th><th>School</th></tr><tr><td>1</td><td><a href="/schools/example">Example University</a></td></tr></table>'
    meta=publication()
    meta["school_id_map"]={"example":"1"}
    out=parse_net_document(html,"text/html",meta)
    assert out.team_id.tolist()==["1"] and out.net_rank.tolist()==["1"]
