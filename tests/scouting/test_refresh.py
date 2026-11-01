from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import pytest
from wbb_scouting.runner import refresh,should_retry
from wbb_scouting.sources import sha256_file
from wbb_scouting.store import load_manifest,load_tables
from tests.scouting.helpers import team_boxes,players,pbp,schedule

NOW=datetime(2026,11,2,14,17,tzinfo=timezone.utc)
def setup(tmp_path):
 config={"season":2027,"state_dir":tmp_path/"state","repository":"owner/repo","team_reference":pd.DataFrame([{"season":2027,"team_id":"A","team_name":"Alpha","conference":"ACC","is_d1":True},{"season":2027,"team_id":"B","team_name":"Beta","conference":"Other","is_d1":True}])}
 data={"schedule":schedule(),"team_box":team_boxes(),"player_box":players(),"pbp":pbp(),"rosters":pd.DataFrame([{"season":2027,"team_id":"A","player_id":"p1","player_name":"Player One"}]),"teams":config["team_reference"].assign(source_family="sdv_espn_wbb")}
 def collect(dataset,season,destination):
  if season!=2027 or dataset not in data:return None
  destination.parent.mkdir(parents=True,exist_ok=True);data[dataset].to_parquet(destination,index=False)
  return {"season":season,"dataset":dataset,"family":"sdv_espn_wbb","league":"womens_college_basketball","name":dataset+".parquet","source_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/example/"+dataset+".parquet","sha256":sha256_file(destination),"local_path":str(destination),"row_count":len(data[dataset])}
 clients={"collect":collect,"net":lambda *args:(None,None),"restore":lambda *args:None}
 return config,data,clients

def mutations(m):return sum(sum(c.get(k,0) for k in ("inserted","updated","removed")) for c in m["counters"].values())

def test_first_ingest_and_identical_rerun(tmp_path):
 cfg,data,clients=setup(tmp_path)
 one=refresh(cfg,NOW,"manual",clients)
 two=refresh(cfg,NOW,"manual",clients)
 assert one["coverage"]["2027:team_box"]["available_games"]==1
 assert mutations(one)>0 and mutations(two)==0
 assert len(two["reports"])==len(one["reports"])
 assert (cfg["state_dir"]/"output/index.html").exists()

def test_new_game_added_once_and_correction_is_replacement(tmp_path):
 cfg,data,clients=setup(tmp_path);first=refresh(cfg,NOW,"manual",clients)
 for name,make in [("schedule",schedule),("team_box",team_boxes),("player_box",players),("pbp",pbp)]:data[name]=pd.concat([data[name],make("g2")],ignore_index=True)
 second=refresh(cfg,NOW,"manual",clients);assert second["counters"]["2027:team_box"]["inserted"]==2
 data["team_box"].loc[data["team_box"].team_id.eq("A"),"assists"]=3
 third=refresh(cfg,NOW,"manual",clients);assert third["counters"]["2027:team_box"]["updated"]==2
 assert len(load_tables(third,cfg["state_dir"])["team_box"])==4
 assert mutations(refresh(cfg,NOW,"manual",clients))==0

def test_pending_current_boxes_never_borrow_historical_freshness(tmp_path):
 cfg,data,clients=setup(tmp_path)
 for n in ("team_box","player_box","pbp"):del data[n]
 result=refresh(cfg,NOW,"manual",clients)
 assert result["coverage"]["2027:team_box"]["available_games"]==0
 assert result["status"]=="partial" and "2027:team_box" in " ".join(result["pending"])
 assert result["reports"][0]["status"]=="awaiting_box_scores"

def test_invalid_incoming_keeps_previous_groups_and_records_failure(tmp_path):
 cfg,data,clients=setup(tmp_path);refresh(cfg,NOW,"manual",clients)
 data["team_box"].loc[0,"points"]=999
 result=refresh(cfg,NOW,"manual",clients)
 assert result["errors"] and result["dataset_status"]["2027:team_box"]["status"]=="failed_validation"
 assert load_tables(result,cfg["state_dir"])["team_box"].points.max()==10

def test_changed_schedule_withholds_incompatible_retained_boxes(tmp_path):
 cfg,data,clients=setup(tmp_path);refresh(cfg,NOW,"manual",clients)
 data["schedule"].loc[0,"home_score"]=12
 result=refresh(cfg,NOW,"manual",clients)
 assert result["coverage"]["2027:team_box"]["available_games"]==0
 assert result["reports"][-1]["status"]=="awaiting_box_scores"

def test_publication_failure_retains_last_local_manifest(tmp_path):
 cfg,data,clients=setup(tmp_path);refresh(cfg,NOW,"manual",clients)
 original=(cfg["state_dir"]/"manifest.json").read_bytes()
 cfg["publish"]=True
 def fail(*args):raise OSError("upload failed")
 clients["publish"]=fail
 with pytest.raises(OSError):refresh(cfg,NOW,"primary",clients)
 assert (cfg["state_dir"]/"manifest.json").read_bytes()==original

class Runs:
 def __init__(self,rows):self.rows=rows
 def get_json(self,url):return {"workflow_runs":self.rows}
def test_retry_uses_workflow_metadata_and_persisted_status():
 now=datetime(2026,11,2,16,17,tzinfo=timezone.utc)
 good={"id":123,"display_title":"WBB scouting primary","created_at":NOW.isoformat(),"status":"completed","conclusion":"success"}
 complete={"primary_attempt":{"run_id":"123","attempted_at":NOW.isoformat(),"status":"complete","pending":[]}}
 assert should_retry("owner/repo",now,complete,Runs([]))
 assert not should_retry("owner/repo",now,complete,Runs([good]))
 assert should_retry("owner/repo",now,{**complete,"primary_attempt":{**complete["primary_attempt"],"pending":["PBP"]}},Runs([good]))
 assert should_retry("owner/repo",now,complete,Runs([{**good,"conclusion":"failure"}]))

def test_retry_skip_performs_no_collection(tmp_path):
 cfg,data,clients=setup(tmp_path);clients["retry_required"]=lambda *args:False
 result=refresh(cfg,datetime(2026,11,2,16,17,tzinfo=timezone.utc),"retry",clients)
 assert result["status"]=="skipped" and not (cfg["state_dir"]/"manifest.json").exists()

def test_player_snapshot_reprocessed_when_team_boxes_arrive_later(tmp_path):
 cfg,data,clients=setup(tmp_path);boxes=data.pop("team_box")
 one=refresh(cfg,NOW,"manual",clients)
 assert one["coverage"]["2027:player_box"]["available_games"]==0
 data["team_box"]=boxes
 two=refresh(cfg,NOW,"manual",clients)
 assert two["coverage"]["2027:player_box"]["available_games"]==1
 assert two["reports"][-1]["game_players"]

def test_source_regression_stays_flagged_on_identical_rerun(tmp_path):
 cfg,data,clients=setup(tmp_path)
 for name,make in [("schedule",schedule),("team_box",team_boxes),("player_box",players),("pbp",pbp)]:data[name]=pd.concat([data[name],make("g2")],ignore_index=True)
 refresh(cfg,NOW,"manual",clients)
 data["team_box"]=team_boxes()
 two=refresh(cfg,NOW,"manual",clients)
 assert any("absent from source" in s for s in two["pending"])
 three=refresh(cfg,NOW,"manual",clients)
 assert any("absent from source" in s for s in three["pending"])

def test_reviewed_net_original_bytes_are_archived_with_lineage(tmp_path):
 import json
 from wbb_scouting.runner import _net_snapshots
 from .test_references import teams,publication
 from wbb_scouting.sources import SourceUnavailable
 folder=tmp_path/"net";folder.mkdir();state=tmp_path/"state";state.mkdir()
 value={"publication":publication(),"rows":[{"team_id":str(i),"net_rank":i} for i in range(1,61)]}
 path=folder/"official.json";path.write_text(json.dumps(value))
 class Blocked:
  def _open(self,url):raise SourceUnavailable("browser verification")
 current,prior,snapshots,sources=_net_snapshots({"season":2027,"net_dir":folder},datetime(2026,12,5,tzinfo=timezone.utc),teams(),Blocked(),state,{})
 assert len(current)==60 and prior is None
 assert sources and snapshots[0]["sha256"] in sources
 source=next(iter(sources.values()))
 assert (state/source["path"]).read_bytes()==path.read_bytes()

def test_current_asset_reuses_restored_raw_bytes_by_published_checksum(tmp_path):
 from wbb_scouting.runner import collect_source
 from wbb_scouting.sources import source_config
 state=tmp_path/"state";raw=state/"raw";raw.mkdir(parents=True)
 path=raw/"temporary.parquet";team_boxes().to_parquet(path,index=False);sha=sha256_file(path);path.rename(raw/(sha+".parquet"))
 spec=source_config()["datasets"]["team_box"]
 asset={"name":"team_box_2027.parquet","digest":"sha256:"+sha,"browser_download_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"+spec["tag"]+"/team_box_2027.parquet"}
 class Cached:
  def get_json(self,url):return {"assets":[asset]}
  def download(self,*args):raise AssertionError("unchanged restored source downloaded again")
 result=collect_source("team_box",2027,state/"incoming/2027-team_box.parquet",Cached())
 assert result["reused"] and result["sha256"]==sha
