import pandas as pd
import pytest


def test_written_stages_have_stable_order_independent_hashes_and_zero_changes(tmp_path):
    from wbb_scouting.pilot import write_stage
    frame=pd.DataFrame([{"team_id":"25","points":8},{"team_id":"265","points":10}])
    first=write_stage({"example":frame},tmp_path,"features",{})
    again=write_stage({"example":frame.iloc[::-1]},tmp_path,"features",first)
    assert first["features.example"]["logical_sha256"]==again["features.example"]["logical_sha256"]
    assert again["features.example"]["changed"] is False
    assert again["features.example"]["rows"]==2
    assert (tmp_path/"features/example.parquet").exists()
    assert (tmp_path/"features/example.csv").exists()


def test_pilot_command_is_independent_and_refuses_production_publication():
    from wbb_scouting.cli import main,parser
    assert parser().parse_args(["pilot","--season","2026"]).command=="pilot"
    with pytest.raises(SystemExit):
        main(["pilot","--season","2026","--publish"])


def test_duplicate_grain_is_rejected_before_overwriting_previous_stage(tmp_path):
    from wbb_scouting.pilot import write_stage
    good=pd.DataFrame([{"season":2026,"team_id":"25","window":"season","games":2}])
    write_stage({"team_summary":good},tmp_path,"features",{})
    original=(tmp_path/"features/team_summary.parquet").read_bytes()
    with pytest.raises(ValueError,match="Duplicate table grain"):
        write_stage({"team_summary":pd.concat([good,good])},tmp_path,"features",{})
    assert (tmp_path/"features/team_summary.parquet").read_bytes()==original


def test_portable_bundle_contains_dashboard_and_each_saved_dataset(tmp_path):
    import json,zipfile
    from wbb_scouting.pilot import package_pilot
    write={"index.html":"<h1>Pilot</h1>","features/x.csv":"team_id\n25\n","features/x.parquet":"example","raw/sources.json":"{}"}
    for name,text in write.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    manifest={"tables":{"features.x":{"path":"features/x.parquet","csv_path":"features/x.csv"}}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    archive=package_pilot(tmp_path,{"season":2026,"label":"Historical"})
    with zipfile.ZipFile(archive) as z:
        assert {"index.html","manifest.json","README.txt","config.json","features/x.csv","features/x.parquet","raw/sources.json"}.issubset(set(z.namelist()))
        assert "2026" in z.read("README.txt").decode()


def test_naive_pilot_clock_is_rejected_before_any_collection(tmp_path):
    from wbb_scouting.pilot import run_pilot
    with pytest.raises(ValueError,match="timezone"):
        run_pilot({"season":2026},tmp_path/"new",now="2026-11-01T12:00:00")
    assert not (tmp_path/"new").exists()

@pytest.mark.parametrize("stale_destination", [False, True])
def test_external_source_cache_carries_verified_provenance_into_new_bundle(tmp_path, monkeypatch, stale_destination):
    import json, zipfile
    from wbb_scouting import pilot
    from .test_pilot_features import clean_fixture, config
    from .test_pilot_data import shot_rows
    clean=clean_fixture()
    raw={"team_box":clean["team_box"],"player_box":clean["player_box"].rename(columns={"player_id":"athlete_id"}),
         "rosters":clean["rosters"].rename(columns={"player_id":"athlete_id"}),"shots":shot_rows()}
    provenance={"downloaded_at":"2026-11-01T00:00:00Z","sources":[{"name":"verified.parquet","sha256":"verified"}]}
    cache=tmp_path/"source-cache";cache.mkdir()
    (cache/"sources.json").write_text(json.dumps(provenance))
    destination=tmp_path/"new-output"
    if stale_destination:
        (destination/"raw").mkdir(parents=True)
        (destination/"raw/sources.json").write_text('{"sources":[{"name":"stale.parquet"}]}')
    monkeypatch.setattr(pilot,"load_pilot_sources",
        lambda path, cfg: (raw,json.loads((path/"sources.json").read_text())))
    quality={"shot_groups":pd.DataFrame([{"game_id":"g1","team_id":"A","status":"validated","reason":""}]),
             "box_quarantine":pd.DataFrame(columns=["dataset","game_id","reason"])}
    monkeypatch.setattr(pilot,"clean_pilot",lambda *args:(clean,quality))
    cfg={**config(),"label":"Historical 2025-26","matchups":[]}
    result=pilot.run_pilot(cfg,destination,now="2026-11-02T10:00:00Z",raw_dir=cache)
    assert result["sources"]==provenance["sources"]
    assert json.loads((destination/"raw/sources.json").read_text())==provenance
    with zipfile.ZipFile(destination/"six_team_pilot_2026.zip") as bundle:
        assert json.loads(bundle.read("raw/sources.json"))==provenance
