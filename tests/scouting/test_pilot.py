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
