import hashlib
import pandas as pd
import pytest

class FakeClient:
    def __init__(self, assets=None, payload=b"", status=200):
        self.assets, self.payload, self.status, self.calls = assets or [], payload, status, 0
    def get_json(self, url):
        return {"assets": self.assets}
    def download(self, url, path, headers=None):
        self.calls += 1
        if self.status != 304:
            path.write_bytes(self.payload)
        return self.status, {}

def test_discovery_selects_exact_published_season():
    from wbb_scouting.sources import discover_assets
    client = FakeClient([{"name":"player_box_2026.parquet"}, {"name":"player_box_2027.parquet"}])
    assert [a["name"] for a in discover_assets("player_box",2027,client)] == ["player_box_2027.parquet"]
    assert discover_assets("player_box",2028,client) == []

def test_download_is_verified_and_unchanged_bytes_are_reused(tmp_path):
    from wbb_scouting.sources import download_asset
    file = tmp_path/"sample.parquet"
    pd.DataFrame({"season":[2027]}).to_parquet(file,index=False)
    payload = file.read_bytes()
    file.unlink()
    checksum = hashlib.sha256(payload).hexdigest()
    client = FakeClient(payload=payload)
    asset = {"browser_download_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/test/sample.parquet","digest":"sha256:"+checksum}
    first = download_asset(asset,file,client)
    second = download_asset(asset,file,client)
    assert first["sha256"] == checksum
    assert second["reused"] is True
    assert client.calls == 1

def test_download_rejects_corrupt_or_wrong_checksum_bytes(tmp_path):
    from wbb_scouting.sources import download_asset
    asset={"browser_download_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/test/a.parquet"}
    with pytest.raises(ValueError,match="Parquet"):
        download_asset(asset,tmp_path/"a.parquet",FakeClient(payload=b"not parquet"))
    assert not (tmp_path/"a.parquet").exists()

def test_http_304_requires_valid_previous_file(tmp_path):
    from wbb_scouting.sources import download_asset
    asset={"browser_download_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/test/a.parquet"}
    with pytest.raises(ValueError,match="304"):
        download_asset(asset,tmp_path/"missing.parquet",FakeClient(status=304))

def test_player_ids_preserve_large_integer_precision():
    from wbb_scouting.adapters import normalize_snapshot
    raw=pd.DataFrame({"game_id":["001"],"season":[2027],"team_id":["25"],"athlete_id":[401414502101958017],"game_date_time":["2026-11-01T20:00:00Z"],"points":[10]})
    out=normalize_snapshot("player_box",raw,{"season":2027,"family":"sdv_espn_wbb"})
    assert out.loc[0,"player_id"] == "401414502101958017"
    assert out.loc[0,"game_id"] == "001"
    assert out.loc[0,"source_family"] == "sdv_espn_wbb"

def test_unsafe_float_identifiers_are_rejected():
    from wbb_scouting.adapters import normalize_snapshot
    raw=pd.DataFrame({"game_id":["1"],"season":[2027],"team_id":["25"],"athlete_id":[float(401414502101958017)],"game_date_time":["2026-11-01T20:00:00Z"],"points":[10]})
    with pytest.raises(ValueError,match="precision"):
        normalize_snapshot("player_box",raw,{"season":2027,"family":"sdv_espn_wbb"})

@pytest.mark.parametrize("change",["wrong_season","missing_key","wrong_family","conflicting_duplicate"])
def test_adapter_rejects_contract_violations(change):
    from wbb_scouting.adapters import normalize_snapshot
    raw=pd.DataFrame({"game_id":["1"],"season":[2027],"team_id":["25"],"athlete_id":["99"],"game_date_time":["2026-11-01T20:00:00Z"],"points":[10]})
    source={"season":2027,"family":"sdv_espn_wbb"}
    if change=="wrong_season": raw["season"]=2026
    if change=="missing_key": raw["athlete_id"]=None
    if change=="wrong_family": source["family"]="nba"
    if change=="conflicting_duplicate": raw=pd.concat([raw,raw.assign(points=12)],ignore_index=True)
    with pytest.raises(ValueError):
        normalize_snapshot("player_box",raw,source)

def test_schedule_aliases_and_boolean_strings_are_normalized():
    from wbb_scouting.adapters import normalize_snapshot
    raw=pd.DataFrame({"id":["1"],"season":[2027],"home_id":["25"],"away_id":["26"],"date":["2026-11-01T20:00:00Z"],"status_type_completed":["false"],"home_score":[0],"away_score":[0]})
    out=normalize_snapshot("schedule",raw,{"season":2027,"family":"sdv_espn_wbb"})
    assert out.loc[0,"home_team_id"]=="25"
    assert not out.loc[0,"completed"]
    assert out.loc[0,"game_id"]=="1"

def test_adapter_rejects_wrong_womens_league():
    from wbb_scouting.adapters import normalize_snapshot
    raw=pd.DataFrame({"game_id":["1"],"season":[2027],"team_id":["25"],"athlete_id":["99"],"game_date_time":["2026-11-01T20:00:00Z"],"points":[10],"team_uid":["s:40~l:46~t:25"]})
    with pytest.raises(ValueError,match="league"):
        normalize_snapshot("player_box",raw,{"season":2027,"family":"sdv_espn_wbb"})

def test_pbp_participant_aliases_are_explicit():
    from wbb_scouting.adapters import normalize_snapshot
    raw=pd.DataFrame({"id":[401414502101958017],"game_id":["1"],"season":[2027],"game_date_time":["2026-11-01T20:00:00Z"],"participants_0_athlete_id":["99"],"participants_1_athlete_id":["98"]})
    out=normalize_snapshot("pbp",raw,{"season":2027,"family":"sdv_espn_wbb"})
    assert out.loc[0,"athlete_id_1"]=="99"
    assert out.loc[0,"event_id"]=="401414502101958017"

def test_missing_asset_is_pending_but_other_errors_propagate():
    from wbb_scouting.sources import discover_assets, SourceUnavailable
    class Missing:
        def get_json(self,url): raise SourceUnavailable("not yet published")
    class Broken:
        def get_json(self,url): raise RuntimeError("upstream failed")
    assert discover_assets("player_box",2027,Missing())==[]
    with pytest.raises(RuntimeError,match="upstream failed"):
        discover_assets("player_box",2027,Broken())

def test_download_rejects_checksum_mismatch_before_replacement(tmp_path):
    from wbb_scouting.sources import download_asset
    p=tmp_path/"valid.parquet"
    pd.DataFrame({"season":[2027]}).to_parquet(p,index=False)
    payload=p.read_bytes()
    p.unlink()
    asset={"browser_download_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/test/valid.parquet","digest":"sha256:"+"0"*64}
    with pytest.raises(ValueError,match="checksum"):
        download_asset(asset,p,FakeClient(payload=payload))
    assert not p.exists()
