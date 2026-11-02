"""Immutable GitHub Release checkpoints; caches are never authoritative."""
from __future__ import annotations
import copy,hashlib,http.client,json,re,shutil,tempfile,uuid
from pathlib import Path,PurePosixPath
from urllib.parse import urlparse,quote,unquote
from .sources import HTTPClient,sha256_file,SourceUnavailable
from .store import json_text,write_manifest,split_asset

PREFIX="scouting-v1-"
class StateError(RuntimeError):pass

def _repo(repository):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",repository):raise StateError("Invalid repository")
    return "https://api.github.com/repos/"+repository

class GitHubClient(HTTPClient):
    def request(self,method,url,body):
        parsed=urlparse(url)
        if parsed.scheme!="https" or parsed.hostname!="api.github.com":raise StateError("Unexpected GitHub API endpoint")
        if not self.token:raise StateError("GitHub publication requires the repository Actions token")
        conn=http.client.HTTPSConnection(parsed.hostname,timeout=120)
        payload=json_text(body).encode()
        try:
            conn.request(method,parsed.path+("?"+parsed.query if parsed.query else ""),payload,
                {"Authorization":"Bearer "+self.token,"Accept":"application/vnd.github+json","User-Agent":"wbb-bis-scouting","Content-Type":"application/json"})
            response=conn.getresponse();data=response.read()
            if response.status not in (200,201):raise StateError("GitHub mutation failed with HTTP "+str(response.status))
            return json.loads(data)
        finally:conn.close()
    def upload(self,url,name,path):
        base=url.split("{")[0];parsed=urlparse(base)
        if parsed.scheme!="https" or parsed.hostname!="uploads.github.com":raise StateError("Unexpected GitHub upload endpoint")
        if not self.token:raise StateError("GitHub publication requires the repository Actions token")
        conn=http.client.HTTPSConnection(parsed.hostname,timeout=120)
        try:
            conn.putrequest("POST",parsed.path+"?name="+quote(name))
            for k,v in {"Authorization":"Bearer "+self.token,"Accept":"application/vnd.github+json","User-Agent":"wbb-bis-scouting","Content-Type":"application/octet-stream","Content-Length":str(path.stat().st_size)}.items():conn.putheader(k,v)
            conn.endheaders()
            with path.open("rb") as handle:
                while block:=handle.read(1024*1024):conn.send(block)
            response=conn.getresponse();data=response.read()
            if response.status!=201:raise StateError("GitHub asset upload failed with HTTP "+str(response.status))
            return json.loads(data)
        finally:conn.close()

def _releases(repository,client):
    result=[]
    for page in range(1,101):
        rows=client.get_json(_repo(repository)+"/releases?page="+str(page)+"&per_page=100")
        if not isinstance(rows,list):raise StateError("Invalid release listing")
        result.extend(rows)
        if len(rows)<100:return result
    raise StateError("Release pagination exceeded safety bound")

def _candidates(repository,client,season=None):
    prefix=PREFIX+(str(season)+"-" if season else "")
    rows=[r for r in _releases(repository,client) if str(r.get("tag_name","")).startswith(prefix) and not r.get("draft") and not r.get("prerelease")]
    return sorted(rows,key=lambda r:(r.get("published_at") or r.get("created_at") or "",r.get("id",0)),reverse=True)

def _safe_path(value):
    path=PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in value:raise StateError("Unsafe state path")
    return str(path)

def _safe_url(repository,url):
    prefix="https://github.com/"+repository+"/releases/download/"+PREFIX
    if not isinstance(url,str) or not url.startswith(prefix) or urlparse(url).query:raise StateError("Unsafe immutable asset reference")
    return url

def _asset_url(repository,tag,name):
    _repo(repository)
    return _safe_url(repository,"https://github.com/"+repository+"/releases/download/"+quote(tag,safe="")+"/"+quote(name,safe=""))

def _repair_draft_urls(repository,manifest,release,assets):
    # Early v1 checkpoints saved GitHub's pre-publication untagged URLs.
    # Rebind only to this release's same-name, same-digest published asset.
    pattern=re.escape("https://github.com/"+repository+"/releases/download/")+r"untagged-[a-fA-F0-9]+/([^/?#]+)"
    for ref in _references(manifest):
        for target in [ref,*ref.get("parts",[])]:
            match=re.fullmatch(pattern,target.get("url",""))
            if not match:continue
            name=unquote(match.group(1));asset=assets.get(name)
            canonical=_asset_url(repository,release["tag_name"],name)
            if not asset or asset.get("browser_download_url")!=canonical or asset.get("digest")!="sha256:"+_checksum(target["sha256"]):
                raise StateError("Unverified legacy draft asset reference")
            if target.get("bytes") is not None and asset.get("size")!=target["bytes"]:
                raise StateError("Legacy draft asset size mismatch")
            target["url"]=canonical

def _checksum(value):
    if not isinstance(value,str) or not re.fullmatch("[a-f0-9]{64}",value):raise StateError("Invalid SHA-256")
    return value

def _references(manifest):
    yield from manifest.get("partitions",[])
    yield from manifest.get("artifacts",[])
    for source in manifest.get("sources",{}).values():
        if "path" in source:yield source

def _restore_references(manifest):
    yield from manifest.get("partitions",[])
    yield from manifest.get("artifacts",[])
    active=set(manifest.get("dataset_snapshots",{}).values())
    for sha,source in manifest.get("sources",{}).items():
        if "path" in source and ("dataset_snapshots" not in manifest or sha in active):yield source

def _download_ref(repository,ref,root,client):
    relative=_safe_path(ref["path"]);checksum=_checksum(ref["sha256"])
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if ref.get("parts"):
        with path.open("wb") as output:
            for index,part in enumerate(ref["parts"]):
                target=root/("chunk-"+str(index))
                client.download(_safe_url(repository,part["url"]),target)
                if sha256_file(target)!=_checksum(part["sha256"]):raise StateError("Raw part checksum mismatch")
                with target.open("rb") as incoming:shutil.copyfileobj(incoming,output)
                target.unlink()
    else:client.download(_safe_url(repository,ref["url"]),path)
    if sha256_file(path)!=checksum:raise StateError("Persisted asset checksum mismatch")
    if ref.get("bytes") is not None and path.stat().st_size!=ref["bytes"]:raise StateError("Persisted asset size mismatch")

def restore_state(repository,destination,client,season=None):
    destination=Path(destination);candidates=_candidates(repository,client,season)
    if not candidates:return None
    destination.parent.mkdir(parents=True,exist_ok=True)
    for release in candidates:
        try:
            with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
                staging=Path(folder)
                assets={a["name"]:a for a in release.get("assets",[])}
                client.download(_safe_url(repository,assets["manifest.json"]["browser_download_url"]),staging/"manifest.json")
                client.download(_safe_url(repository,assets["manifest.sha256"]["browser_download_url"]),staging/"manifest.sha256")
                expected=(staging/"manifest.sha256").read_text().strip()
                if sha256_file(staging/"manifest.json")!=_checksum(expected):raise StateError("Manifest checksum mismatch")
                manifest=json.loads((staging/"manifest.json").read_text())
                if manifest.get("schema_version")!=1 or manifest.get("status") not in ("complete","partial"):
                    raise StateError("Unsupported or failed checkpoint")
                if season and manifest.get("season")!=season:raise StateError("Unexpected checkpoint season")
                _repair_draft_urls(repository,manifest,release,assets)
                for ref in _references(manifest):
                    _safe_path(ref["path"]);_checksum(ref["sha256"])
                    if ref.get("url"):_safe_url(repository,ref["url"])
                for ref in _restore_references(manifest):_download_ref(repository,ref,staging,client)
                manifest.update(state_release=release["tag_name"],state_head=candidates[0]["tag_name"])
                destination.mkdir(parents=True,exist_ok=True)
                for ref in _restore_references(manifest):
                    target=destination/_safe_path(ref["path"]);target.parent.mkdir(parents=True,exist_ok=True)
                    (staging/ref["path"]).replace(target)
                write_manifest(manifest,destination/"manifest.json")
                return manifest
        except (OSError,ValueError,KeyError,StateError,SourceUnavailable):
            continue
    raise StateError("Published scouting checkpoints exist but none can be verified; refusing empty bootstrap")

def publish_run(repository,manifest,files,client,commit_sha,max_bytes=512*1024*1024):
    base=_repo(repository)
    if manifest.get("schema_version")!=1 or manifest.get("status") not in ("complete","partial"):raise StateError("Only validated checkpoints can publish")
    if not re.fullmatch("[a-fA-F0-9]{40}",commit_sha):raise StateError("Exact code commit is required")
    parent=manifest.get("parent_release")
    def check_head():
        rows=_candidates(repository,client,manifest["season"])
        if (rows[0]["tag_name"] if rows else None)!=parent:raise StateError("Rejected concurrent publication; restore current head first")
    check_head()
    m=copy.deepcopy(manifest)
    paths={sha256_file(Path(p)):Path(p) for p in files}
    output_paths={"output/"+Path(p).name for p in files}
    m["artifacts"]=[ref for ref in m.get("artifacts",[]) if ref["path"] not in output_paths]
    for ref in _references(m):
        _safe_path(ref["path"]);_checksum(ref["sha256"])
        if ref.get("url"):_safe_url(repository,ref["url"])
        elif not ref.get("parts") and ref["sha256"] not in paths:raise StateError("Missing file for new checkpoint reference")
    tag=PREFIX+str(m["season"])+"-"+re.sub("[^0-9]","",m["attempted_at"])[:20]+"-"+uuid.uuid4().hex[:8]
    release=client.request("POST",base+"/releases",{"tag_name":tag,"target_commitish":commit_sha,"name":"WBB scouting · "+m["attempted_at"],"draft":True,"make_latest":"false","body":"Verified public-data scouting checkpoint. Open index.html after downloading; full report exports and checksummed source data are attached."})
    uploaded={}
    with tempfile.TemporaryDirectory() as folder:
        temporary=Path(folder)
        def upload(path):
            checksum=sha256_file(path)
            if checksum in uploaded:return uploaded[checksum]
            asset=client.upload(release["upload_url"],path.name,path)
            verify=temporary/("verify-"+checksum)
            client.download(asset["url"],verify,{"Accept":"application/octet-stream"})
            if sha256_file(verify)!=checksum:raise StateError("Uploaded asset verification failed")
            verify.unlink();uploaded[checksum]=_asset_url(repository,tag,path.name)
            return uploaded[checksum]
        # Reuse old immutable URLs; upload only new references.
        for ref in _references(m):
            if ref.get("url") or ref.get("parts"):continue
            path=paths[ref["sha256"]]
            chunks=split_asset(path,temporary/"shards",max_bytes)
            if len(chunks)>1:
                ref["parts"]=[{"sha256":p["sha256"],"bytes":p["bytes"],"url":upload(Path(p["path"]))} for p in chunks]
            else:ref["url"]=upload(path)
        # User-facing files are always attached under stable names in this run.
        referenced={r["sha256"] for r in _references(m)}
        for path in map(Path,files):
            if sha256_file(path) not in referenced:
                url=upload(path)
                m.setdefault("artifacts",[]).append({"path":"output/"+path.name,"sha256":sha256_file(path),"bytes":path.stat().st_size,"url":url})
        m.update(state_release=tag,state_head=tag,code_commit=commit_sha)
        mpath=temporary/"manifest.json";write_manifest(m,mpath)
        upload(mpath)
        digest=temporary/"manifest.sha256";digest.write_text(sha256_file(mpath)+"\n");upload(digest)
        check_head()
        published=client.request("PATCH",base+"/releases/"+str(release["id"]),{"draft":False,"make_latest":"false"})
    return {**published,"manifest":m}
