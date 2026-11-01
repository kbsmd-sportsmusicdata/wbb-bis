import json,hashlib
from pathlib import Path
import pytest
from wbb_scouting.github_state import restore_state,publish_run,StateError
from wbb_scouting.sources import sha256_file

REPO="owner/repo"
class Fake:
 def __init__(self):self.releases=[];self.contents={};self.next=1;self.fail=False;self.race=False
 def get_json(self,url):
  if "releases?" in url:return list(reversed(self.releases)) if "page=1&" in url else []
  raise AssertionError(url)
 def request(self,method,url,body):
  if method=="POST":
   row={"id":self.next,"tag_name":body["tag_name"],"draft":True,"assets":[],"upload_url":"https://uploads.github.com/repos/"+REPO+"/releases/"+str(self.next)+"/assets{?name,label}"}
   self.next+=1;self.releases.append(row);return row
  if method=="PATCH":
   row=next(r for r in self.releases if url.endswith("/"+str(r["id"])));row.update(body);return row
  raise AssertionError(method)
 def upload(self,url,name,path):
  if self.fail and name!="index.html":raise OSError("interrupted upload")
  row=self.releases[-1];asset={"name":name,"url":"https://api.github.com/repos/"+REPO+"/releases/assets/"+str(len(self.contents)+1),"browser_download_url":"https://github.com/"+REPO+"/releases/download/"+row["tag_name"]+"/"+name}
  self.contents[asset["url"]]=path.read_bytes();self.contents[asset["browser_download_url"]]=path.read_bytes()
  row["assets"].append(asset);return asset
 def download(self,url,path,headers=None):
  if url not in self.contents:raise OSError("missing asset")
  path.write_bytes(self.contents[url]);return 200,{}
def manifest():return {"schema_version":1,"season":2027,"attempted_at":"2026-11-02T14:17:00+00:00","status":"partial","partitions":[],"sources":{},"parent_release":None}
def setup(tmp_path):
 d=tmp_path/"input";d.mkdir();p=d/"x.parquet";p.write_bytes(b"canonical");h=d/"index.html";h.write_text("dashboard")
 m=manifest();m["partitions"]=[{"dataset":"team_box","path":"objects/x.parquet","sha256":sha256_file(p),"bytes":p.stat().st_size}]
 return m,[p,h]

def test_fresh_runner_restores_all_files_without_cache(tmp_path):
 c=Fake();m,files=setup(tmp_path);published=publish_run(REPO,m,files,c,"a"*40)
 restored=restore_state(REPO,tmp_path/"fresh",c)
 assert restored["partitions"][0]["sha256"]==m["partitions"][0]["sha256"]
 assert (tmp_path/"fresh/objects/x.parquet").read_bytes()==b"canonical"
 assert restored["state_release"]==published["tag_name"]
 assert not published["draft"]

def test_unrelated_latest_release_is_ignored_and_unchanged_assets_reused(tmp_path):
 c=Fake();m,files=setup(tmp_path);first=publish_run(REPO,m,files,c,"a"*40)
 restored=restore_state(REPO,tmp_path/"fresh",c);c.releases.append({"id":999,"tag_name":"other","draft":False,"assets":[]})
 m2={**restored,"parent_release":restored["state_head"],"attempted_at":"2026-11-04T14:17:00+00:00"}
 second=publish_run(REPO,m2,[],c,"a"*40)
 assert all(a["name"]!="x.parquet" for a in second["assets"])
 assert restore_state(REPO,tmp_path/"fresh2",c)["partitions"][0]["url"]==restored["partitions"][0]["url"]

def test_invalid_latest_falls_back_but_all_invalid_fails_closed(tmp_path):
 c=Fake();m,files=setup(tmp_path);first=publish_run(REPO,m,files,c,"a"*40)
 c.releases.append({"tag_name":"scouting-v1-2027-bad","draft":False,"assets":[{"name":"manifest.json","browser_download_url":"https://github.com/"+REPO+"/releases/download/scouting-v1-2027-bad/manifest.json"}]})
 assert restore_state(REPO,tmp_path/"fresh",c)["state_release"]==first["tag_name"]
 c.releases=c.releases[1:]
 with pytest.raises(StateError):restore_state(REPO,tmp_path/"none",c)

def test_corrupt_or_missing_assets_never_silently_bootstrap(tmp_path):
 c=Fake();m,files=setup(tmp_path);publish_run(REPO,m,files,c,"a"*40)
 url=next(u for u in c.contents if u.endswith("/x.parquet"));c.contents[url]=b"corrupt"
 with pytest.raises(StateError):restore_state(REPO,tmp_path/"fresh",c)
 assert not (tmp_path/"fresh/manifest.json").exists()

def test_upload_failure_leaves_previous_release_published(tmp_path):
 c=Fake();m,files=setup(tmp_path);first=publish_run(REPO,m,files,c,"a"*40);c.fail=True
 m["parent_release"]=first["tag_name"]
 with pytest.raises(OSError):publish_run(REPO,m,files,c,"a"*40)
 assert not c.releases[0]["draft"] and c.releases[-1]["draft"]

def test_stale_parent_rejects_concurrent_publisher(tmp_path):
 c=Fake();m,files=setup(tmp_path);publish_run(REPO,m,files,c,"a"*40)
 with pytest.raises(StateError,match="concurrent"):publish_run(REPO,m,files,c,"a"*40)

def test_unsafe_references_are_rejected(tmp_path):
 c=Fake();m,files=setup(tmp_path);m["partitions"][0]["path"]="../outside"
 with pytest.raises(StateError):publish_run(REPO,m,files,c,"a"*40)

def test_large_raw_source_is_sharded_and_reassembled(tmp_path):
 c=Fake();m,files=setup(tmp_path);raw=tmp_path/"raw.parquet";raw.write_bytes(b"0123456789"*10)
 sha=sha256_file(raw);m["sources"]={sha:{"path":"raw/"+sha+".parquet","sha256":sha,"bytes":100}}
 publish_run(REPO,m,files+[raw],c,"a"*40,max_bytes=31)
 out=tmp_path/"restored";restored=restore_state(REPO,out,c)
 assert (out/("raw/"+sha+".parquet")).read_bytes()==raw.read_bytes()
 assert len(restored["sources"][sha]["parts"])==4

def test_no_matching_published_state_is_only_safe_bootstrap(tmp_path):
 assert restore_state(REPO,tmp_path/"fresh",Fake()) is None

def test_changed_dashboard_replaces_only_current_artifact_reference(tmp_path):
 c=Fake();m,files=setup(tmp_path);first=publish_run(REPO,m,files,c,"a"*40)
 old=restore_state(REPO,tmp_path/"restored",c)
 files[-1].write_text("updated dashboard")
 next_m={**old,"parent_release":old["state_head"],"attempted_at":"2026-11-04T14:17:00+00:00"}
 publish_run(REPO,next_m,[files[-1]],c,"a"*40)
 restored=restore_state(REPO,tmp_path/"new",c)
 assert len([r for r in restored["artifacts"] if r["path"]=="output/index.html"])==1
 assert (tmp_path/"new/output/index.html").read_text()=="updated dashboard"

def test_redirect_does_not_forward_authorization_to_storage_host():
 from urllib.request import Request
 from wbb_scouting.sources import SafeRedirectHandler
 request=Request("https://api.github.com/repos/owner/repo/releases/assets/1",headers={"Authorization":"Bearer test-secret"})
 redirected=SafeRedirectHandler().redirect_request(request,None,302,"Found",{},"https://release-assets.githubusercontent.com/asset")
 assert not redirected.has_header("Authorization")
