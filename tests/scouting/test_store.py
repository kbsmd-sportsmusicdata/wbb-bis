import pandas as pd
import pytest
from .helpers import team_boxes, pbp

def test_unchanged_reingest_has_zero_logical_mutations():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=team_boxes()
    result,counts=reconcile_games(original,original.copy(),KEYS["team_box"],{"g1"})
    assert len(result)==2
    assert (counts["inserted"],counts["updated"],counts["removed"])==(0,0,0)
    assert counts["unchanged"]==2

def test_new_game_is_added_once_and_correction_replaces_it():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=team_boxes()
    incoming=pd.concat([original,team_boxes("g2")],ignore_index=True)
    result,counts=reconcile_games(original,incoming,KEYS["team_box"],{"g1","g2"})
    assert len(result)==4 and counts["inserted"]==2
    corrected=incoming.copy()
    corrected.loc[(corrected.game_id=="g2")&(corrected.team_id=="A"),"points"]=11
    fixed,counts=reconcile_games(result,corrected,KEYS["team_box"],{"g1","g2"})
    assert len(fixed)==4 and counts["updated"]==1
    assert fixed.loc[(fixed.game_id=="g2")&(fixed.team_id=="A"),"points"].item()==11

def test_removed_or_renumbered_event_does_not_keep_obsolete_row():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=pbp()
    replacement=original.iloc[:-1].copy()
    replacement.loc[replacement.index[-1],"event_id"]="replacement"
    out,counts=reconcile_games(original,replacement,KEYS["pbp"],{"g1"})
    assert set(out.event_id)==set(replacement.event_id)
    assert counts["removed"]==2 and counts["inserted"]==1

def test_absent_or_quarantined_game_never_erases_previous_state():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=pd.concat([team_boxes("g1"),team_boxes("g2")],ignore_index=True)
    incoming=team_boxes("g2").assign(points=999)
    out,counts=reconcile_games(original,incoming,KEYS["team_box"],set())
    assert out.points.tolist()==original.points.tolist()
    assert counts["missing_games"]==["g1"]

def test_conflicting_duplicate_keys_are_rejected():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    incoming=pd.concat([team_boxes(),team_boxes().assign(points=999)],ignore_index=True)
    with pytest.raises(ValueError,match="duplicate"):
        reconcile_games(team_boxes(),incoming,KEYS["team_box"],{"g1"})

def test_partition_and_manifest_roundtrip_is_verified(tmp_path):
    from wbb_scouting.store import write_partitions,write_manifest,load_manifest
    files=write_partitions(team_boxes(),"team_box",tmp_path)
    assert len(files)==1 and files[0]["rows"]==2
    manifest={"schema_version":1,"partitions":files,"season":2027}
    target=tmp_path/"manifest.json"
    write_manifest(manifest,target)
    assert load_manifest(target)["partitions"][0]["sha256"]==files[0]["sha256"]
    original=(tmp_path/files[0]["path"]).read_bytes()
    files_again=write_partitions(team_boxes(),"team_box",tmp_path)
    assert files_again[0]["sha256"]==files[0]["sha256"]


def test_empty_incoming_snapshot_preserves_all_prior_games():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=team_boxes()
    out,counts=reconcile_games(original,pd.DataFrame(),KEYS["team_box"],set())
    assert len(out)==2 and counts["missing_games"]==["g1"]
    assert counts["inserted"]==counts["updated"]==counts["removed"]==0

def test_real_parquet_date_values_have_stable_record_fingerprints():
    from datetime import date
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=team_boxes().assign(game_date=date(2026,11,1))
    out,counts=reconcile_games(pd.DataFrame(),original,KEYS["team_box"],{"g1"})
    again,repeat=reconcile_games(out,original,KEYS["team_box"],{"g1"})
    assert len(again)==2 and counts["inserted"]==2
    assert repeat["inserted"]==repeat["updated"]==repeat["removed"]==0

def test_initial_ingest_preserves_booleans_and_nulls_for_unchanged_rerun():
    from wbb_scouting.store import reconcile_games
    from wbb_scouting.adapters import KEYS
    original=team_boxes().assign(team_winner=[True,False],optional_stat=pd.NA)
    first,_=reconcile_games(pd.DataFrame(),original,KEYS["team_box"],{"g1"})
    second,counts=reconcile_games(first,original,KEYS["team_box"],{"g1"})
    assert counts["updated"]==0
    assert first.team_winner.tolist()==[True,False]
    assert all(type(v) is bool for v in first.team_winner.tolist())

def test_large_asset_parts_reassemble_exact_original_bytes(tmp_path):
    from wbb_scouting.store import split_asset
    original=tmp_path/"large.bin"
    original.write_bytes(bytes(range(250))*3)
    parts=split_asset(original,tmp_path/"parts",max_bytes=128)
    assert all(item["bytes"]<=128 for item in parts)
    assert b"".join(__import__("pathlib").Path(item["path"]).read_bytes() for item in parts)==original.read_bytes()
