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
