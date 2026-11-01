"""One reproducible refresh, independent of the legacy pipeline."""
from __future__ import annotations
import copy,json,os,shutil,subprocess
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.error import HTTPError,URLError
import pandas as pd
import pyarrow.parquet as pq
from .adapters import KEYS,normalize_snapshot
from .calendar import PACIFIC,aware_utc,parse_time,retry_required
from .sources import HTTPClient,SourceUnavailable,source_config,discover_assets,download_asset,sha256_file
from .store import clean_json,json_text,fingerprint,reconcile_games,write_partitions,write_manifest,load_manifest,load_tables
from .validation import prepare_snapshot,validate_snapshot,annotate_division
from .references import load_team_reference,select_cohort,validate_net,parse_net_document,NET_URL
from .reports import build_reports
from .dashboard import render_dashboard,write_exports
from .github_state import GitHubClient,restore_state,publish_run

ADAPTER_VERSION="scouting-normalization-2"
EMPTY=pd.DataFrame()
DATASETS=("teams","schedule","team_box","player_box","pbp","rosters")

def _year(frame,season):
    return frame.loc[frame.season.eq(season)].copy() if len(frame) else frame.copy()

def should_retry(repository,now,previous,client):
    if not retry_required(now,None):return False
    try:
        payload=client.get_json("https://api.github.com/repos/"+repository+"/actions/workflows/wbb-scouting.yml/runs?event=schedule&per_page=100")
    except SourceUnavailable:return True
    today=aware_utc(now).astimezone(PACIFIC).date()
    primaries=[r for r in payload.get("workflow_runs",[]) if r.get("display_title")=="WBB scouting primary" and parse_time(r["created_at"]).astimezone(PACIFIC).date()==today]
    if not primaries:return True
    primary=max(primaries,key=lambda r:r["created_at"])
    persisted=previous.get("primary_attempt") if previous else None
    if primary.get("status")!="completed" or primary.get("conclusion")!="success":return True
    if not persisted or str(primary["id"])!=str(persisted.get("run_id")):return True
    return retry_required(now,persisted)

def collect_source(dataset,season,destination,client,local_input=None):
    if local_input:
        spec=source_config()["datasets"][dataset];path=Path(local_input)/spec["pattern"].format(season=season)
        if not path.exists():return None
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        pf=pq.ParquetFile(destination)
        return {"dataset":dataset,"season":season,"family":"sdv_espn_wbb","league":"womens_college_basketball",
            "name":path.name,"local_path":str(destination),"sha256":sha256_file(destination),"row_count":pf.metadata.num_rows,
            "schema_fingerprint":fingerprint(str(pf.schema_arrow)),"source_url":"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"+spec["tag"]+"/"+path.name,"local_input":True}
    assets=discover_assets(dataset,season,client)
    if not assets:return None
    expected=(assets[0].get("digest") or "").removeprefix("sha256:")
    cached=destination.parent.parent/"raw"/(expected+".parquet")
    chosen=cached if expected and cached.exists() and sha256_file(cached)==expected else destination
    result=download_asset(assets[0],chosen,client)
    result["source_url"]=assets[0]["browser_download_url"]
    return result

def _net_snapshots(config,now,teams,client,state_dir,previous):
    """Retain reviewed snapshots; a browser challenge is pending, never a guessed rank."""
    verified=copy.deepcopy(previous.get("net_snapshots",[]))
    net_sources={}
    def archive(payload,suffix,publication):
        import hashlib
        sha=hashlib.sha256(payload).hexdigest();path=Path(state_dir)/"raw"/(sha+suffix)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        net_sources[sha]={"sha256":sha,"path":str(path.relative_to(state_dir)),"bytes":len(payload),"name":path.name,"dataset":"net","provider":"NCAA","source_url":publication["source_url"],"season":publication["season"],"publication_date":publication["publication_date"],"retrieved_at":now.isoformat()}
        return sha
    folder=Path(config.get("net_dir",Path(__file__).resolve().parents[1]/"config/scouting_net"))
    if folder.exists():
        for path in sorted(folder.glob("*.json")):
            if path.name=="school-map.json":continue
            value=json.loads(path.read_text())
            rows=validate_net(pd.DataFrame(value["rows"]),value["publication"],teams)
            item={"rows":clean_json(rows.to_dict("records")),"publication":value["publication"],"sha256":archive(path.read_bytes(),".json",value["publication"])}
            if fingerprint(item) not in {fingerprint(v) for v in verified}:verified.append(item)
    # The official page is attempted each run. HTML requires a reviewed NCAA-school to ESPN map.
    mapping=folder/"school-map.json"
    try:
        with client._open(NET_URL) as response:
            payload=response.read(4*1024*1024)
        metadata={"source_url":NET_URL,"verified":True,"id_namespace":"espn","first_seen_at":now.isoformat(),
            "school_id_map":json.loads(mapping.read_text()) if mapping.exists() else {}}
        raw=parse_net_document(payload,"text/html",metadata)
        # Require an explicit dated publication embedded in the actual page.
        import re
        match=re.search(rb'Last updated[^<]{0,80}?([A-Z][a-z]+ \d{1,2}, \d{4})',payload,re.I)
        if match:
            date=pd.to_datetime(match.group(1).decode()).date()
            metadata.update(publication_date=date.isoformat(),season=date.year+1 if date.month>=8 else date.year)
            rows=validate_net(raw,metadata,teams)
            item={"rows":clean_json(rows.to_dict("records")),"publication":metadata,"sha256":archive(payload,".html",metadata)}
            if fingerprint(item) not in {fingerprint(v) for v in verified}:verified.append(item)
    except (SourceUnavailable,HTTPError,URLError,TimeoutError):
        pass
    current=prior=None
    for item in sorted(verified,key=lambda v:v["publication"]["publication_date"]):
        rows=pd.DataFrame(item["rows"])
        rows=validate_net(rows,item["publication"],teams)
        if rows.ranking_season.iloc[0]==config["season"]:current=rows
        elif rows.ranking_season.iloc[0]==config["season"]-1:prior=rows
    return current,prior,verified,net_sources

def _read_raw(path,dataset):
    if dataset!="pbp":return pd.read_parquet(path)
    # Keep original bytes intact; project only scalar fields needed by this version.
    wanted={"id","event_id","game_id","season","game_date","game_date_time","uid","team_id","home_team_id","away_team_id",
        "game_play_number","sequence_number","home_score","away_score","score_value","scoring_play","shooting_play",
        "period_number","text","type_text","type_id","clock_display_value","participants_0_athlete_id","participants_1_athlete_id"}
    return pd.read_parquet(path,columns=[c for c in pq.ParquetFile(path).schema_arrow.names if c in wanted])

def refresh(config,now,trigger,clients=None):
    clients=clients or {};now=aware_utc(now);season=int(config["season"]);state=Path(config["state_dir"])
    state.mkdir(parents=True,exist_ok=True);client=clients.get("http") or GitHubClient()
    previous=None
    if config.get("restore_remote"):
        previous=clients.get("restore",restore_state)(config["repository"],state,client,season)
    if previous is None and (state/"manifest.json").exists():previous=load_manifest(state/"manifest.json")
    previous=previous or {}
    gate=clients.get("retry_required",should_retry)
    if trigger=="retry" and not gate(config["repository"],now,previous,client):return {"status":"skipped","reason":"Primary was complete; no source refresh required"}
    tables=load_tables(previous,state) if previous else {}
    m={**copy.deepcopy(previous),"schema_version":1,"season":season,"repository":config["repository"],
        "attempted_at":now.isoformat(),"trigger":trigger,"parent_release":previous.get("state_head"),
        "adapter_version":ADAPTER_VERSION,"status":"partial","pending":[],"errors":[],"coverage":{},"counters":{},"artifacts":[]}
    for key in ("sources","dataset_snapshots","game_sources","dataset_status","ingest_quarantine"):m.setdefault(key,{})
    checked=config.get("team_reference")
    if checked is None:checked=load_team_reference(Path(__file__).resolve().parents[1]/("config/scouting_teams_"+str(season)+".json"),season)
    checked=checked.copy();checked["source_family"]="sdv_espn_wbb"
    collect=clients.get("collect") or (lambda dataset,year,path:collect_source(dataset,year,path,client,config.get("local_input")))
    new_games=set()
    for year in (season-1,season):
        for dataset in DATASETS:
            if year!=season and dataset in ("pbp","rosters"):continue
            key=str(year)+":"+dataset
            old=_year(tables.get(dataset,EMPTY),year)
            try:
                metadata=collect(dataset,year,state/"incoming"/(str(year)+"-"+dataset+".parquet"))
                if metadata is None:
                    m["dataset_status"][key]={"status":"pending","last_validated_at":m["dataset_status"].get(key,{}).get("last_validated_at")}
                    if year==season:m["pending"].append(key+": source not published")
                    continue
                checksum=metadata["sha256"];path=Path(metadata["local_path"]);raw_path=state/"raw"/(checksum+".parquet")
                raw_path.parent.mkdir(exist_ok=True)
                if not raw_path.exists():path.replace(raw_path)
                if sha256_file(raw_path)!=checksum:raise ValueError("Raw archive checksum mismatch")
                safe={k:metadata[k] for k in ("name","season","dataset","family","league","source_url","row_count","schema_fingerprint","tag","updated_at","bytes","local_input") if k in metadata}
                m["sources"].setdefault(checksum,{**safe,"sha256":checksum,"retrieved_at":now.isoformat(),"path":str(raw_path.relative_to(state)),"bytes":raw_path.stat().st_size})
                dependencies=["teams"] if dataset=="schedule" else ["teams","schedule","team_box"] if dataset in ("player_box","pbp") else ["teams","schedule"] if dataset=="team_box" else ["teams"] if dataset=="rosters" else []
                dep_signature=fingerprint({d:m["dataset_snapshots"].get(str(year)+":"+d) for d in dependencies})
                unchanged=(m["dataset_snapshots"].get(key)==checksum and previous.get("adapter_version")==ADAPTER_VERSION and m["dataset_status"].get(key,{}).get("dependency_signature")==dep_signature)
                if unchanged:
                    if m["dataset_status"][key].get("missing_games"):
                        m["pending"].append(key+": previously stored games absent from source")
                    m["counters"][key]={"inserted":0,"updated":0,"removed":0,"unchanged":len(old),"affected_games":[]}
                    continue
                incoming,identity_bad=prepare_snapshot(dataset,_read_raw(raw_path,dataset),metadata)
                if incoming.empty and not old.empty:raise ValueError("Empty snapshot would erase prior coverage")
                if dataset=="teams":
                    if "is_d1" not in incoming:incoming["is_d1"]=incoming.subdivision_id.eq("wbb:d1")
                    if "conference" not in incoming:incoming["conference"]=incoming.conference_id.map(lambda v:"ACC" if v=="wbb:acc" else str(v))
                    incoming=incoming[incoming.is_d1.eq(True)].copy()
                    if year==season:
                        actual=set(incoming.loc[incoming.conference.eq("ACC"),"team_id"])
                        expected=set(checked.loc[checked.conference.eq("ACC"),"team_id"])
                        if actual!=expected or set(incoming.team_id)!=set(checked.team_id):raise ValueError("Team/conference membership changed; reference review required")
                        incoming=checked.copy()
                teams=_year(tables.get("teams",EMPTY),year)
                if year==season:teams=checked
                if dataset=="schedule":
                    if incoming[["home_team_id","away_team_id","completed"]].isna().any().any():raise ValueError("Missing schedule matchup/status")
                    if incoming.home_team_id.eq(incoming.away_team_id).any():raise ValueError("Invalid schedule matchup")
                    if len(teams):
                        d1=set(teams.loc[teams.is_d1.eq(True),"team_id"].astype(str))
                        incoming=incoming[incoming.home_team_id.isin(d1)|incoming.away_team_id.isin(d1)]
                    accepted=set(incoming.game_id.astype(str));bad=identity_bad
                elif dataset in ("team_box","player_box","pbp"):
                    context={"season":year,"schedule":_year(tables.get("schedule",EMPTY),year),"teams":teams,"team_box":_year(tables.get("team_box",EMPTY),year),"previous_player_box":old if dataset=="player_box" else EMPTY,"now":now.isoformat()}
                    validation=validate_snapshot(dataset,incoming,context)
                    accepted=validation["accepted_games"];bad={**identity_bad,**validation["quarantine"]}
                else:
                    if dataset=="rosters" and len(teams):incoming=incoming[incoming.team_id.isin(set(teams.loc[teams.is_d1.eq(True),"team_id"]))]
                    accepted=None;bad=identity_bad
                if len(teams) and dataset in ("schedule","team_box","player_box","pbp"):
                    incoming=annotate_division(incoming,teams)
                if "game_id" in incoming:
                    merged,counters=reconcile_games(old,incoming,KEYS[dataset],accepted)
                    for game in counters["affected_games"]:m["game_sources"].setdefault(dataset,{})[str(year)+":"+game]=checksum
                    if year==season and dataset in ("team_box","player_box","pbp"):new_games.update(counters["affected_games"])
                    if counters.get("missing_games"):m["pending"].append(key+": previously stored games absent from source")
                else:
                    merged=incoming;counters={"inserted":max(0,len(incoming)-len(old)),"updated":0 if fingerprint(clean_json(old.to_dict("records")))==fingerprint(clean_json(incoming.to_dict("records"))) else min(len(old),len(incoming)),"removed":max(0,len(old)-len(incoming)),"unchanged":0,"affected_games":[]}
                if len(teams) and dataset in ("schedule","team_box","player_box","pbp"):merged=annotate_division(merged,teams)
                other=tables.get(dataset,EMPTY)
                other=other.loc[~other.season.eq(year)] if len(other) else other
                pieces=[v for v in (other,merged) if len(v)]
                tables[dataset]=pd.concat(pieces,ignore_index=True) if len(pieces)>1 else pieces[0] if pieces else merged
                m["dataset_snapshots"][key]=checksum;m["counters"][key]=counters;m["ingest_quarantine"][key]=bad
                m["dataset_status"][key]={"status":"failed_validation" if bad else "validated","last_validated_at":now.isoformat(),"raw_sha256":checksum,"quarantined_games":len(bad),"dependency_signature":dep_signature,"missing_games":counters.get("missing_games",[])}
            except (ValueError,KeyError,OSError,HTTPError,URLError,SourceUnavailable) as exc:
                m["dataset_status"][key]={"status":"failed","last_validated_at":previous.get("dataset_status",{}).get(key,{}).get("last_validated_at")}
                m["errors"].append(key+": "+type(exc).__name__)
                m["pending"].append(key+": collection/validation failed")
    # Revalidate retained groups against the latest schedule and team boxes before analytics.
    report_tables={k:v.copy() for k,v in tables.items() if k not in ("team_box","player_box","pbp")}
    for dataset in ("team_box","player_box","pbp"):
        pieces=[]
        for year in (season-1,season):
            frame=_year(tables.get(dataset,EMPTY),year)
            context={"season":year,"schedule":_year(tables.get("schedule",EMPTY),year),"teams":checked if year==season else _year(tables.get("teams",EMPTY),year),"team_box":_year(report_tables.get("team_box",EMPTY),year),"now":now.isoformat()}
            v=validate_snapshot(dataset,frame,context)
            m["coverage"][str(year)+":"+dataset]=v["coverage"]
            if len(frame):pieces.append(frame[frame.game_id.astype(str).isin(v["accepted_games"])])
            key=str(year)+":"+dataset
            bad=m["ingest_quarantine"].get(key,{})
            if year==season:
                if bad or v["quarantine"]:
                    m["errors"].append(key+": failed game validation")
                    m["pending"].append(key+": quarantined games")
                if v["coverage"]["missing_game_ids"]:m["pending"].append(key+": completed games missing")
        report_tables[dataset]=pd.concat(pieces,ignore_index=True) if pieces else EMPTY.copy()
    if _year(report_tables.get("schedule",EMPTY),season).empty:raise ValueError("No verified current-season schedule; refusing empty success")
    if "net" in clients:
        net,prior=clients["net"](config,now,checked)
    else:
        net,prior,net_snapshots,net_sources=_net_snapshots(config,now,checked,client,state,previous);m["net_snapshots"]=net_snapshots
        for sha,source in net_sources.items():m["sources"].setdefault(sha,source)
        for item in net_snapshots:
            if item.get("sha256"):m["dataset_snapshots"][str(item["publication"]["season"])+":net"]=item["sha256"]
    cohort=select_cohort(net,checked,now,season,prior);m["cohort"]=clean_json(cohort.to_dict("records"))
    if cohort.selection_status.isin(["pending_net","stale_net","provisional_prior_season"]).any():m["pending"].append("Verified current-season NCAA NET")
    m["active_net_snapshot"]={}
    if len(cohort) and cohort.ranking_date.notna().any():
        year=int(cohort.ranking_season.iloc[0]);date=str(cohort.ranking_date.iloc[0])
        for item in m.get("net_snapshots",[]):
            pub=item["publication"]
            if pub["season"]==year and pub["publication_date"]==date and item.get("sha256"):
                m["active_net_snapshot"][str(year)+":net"]=item["sha256"]
    m["new_game_ids"]=sorted(new_games)
    m["reports"]=build_reports(report_tables,cohort,now,previous.get("reports",[]),m,bootstrap=not bool(previous))
    m["partitions"]=[]
    for name,frame in tables.items():m["partitions"].extend(write_partitions(frame,name,state))
    # Retain immutable locations for content-identical objects.
    old_refs={p["sha256"]:p for p in previous.get("partitions",[])}
    for part in m["partitions"]:
        if part["sha256"] in old_refs:
            for field in ("url","parts"):
                if field in old_refs[part["sha256"]]:part[field]=old_refs[part["sha256"]][field]
    m["pending"]=sorted(set(m["pending"]));m["errors"]=sorted(set(m["errors"]))
    m["status"]="partial" if m["pending"] or m["errors"] else "complete"
    if not m["errors"]:m["last_successful_validation_at"]=now.isoformat()
    if trigger=="primary":m["primary_attempt"]={"run_id":os.environ.get("GITHUB_RUN_ID"),"attempted_at":now.isoformat(),"status":m["status"],"pending":m["pending"]}
    m=clean_json(m);output=state/"output";render_dashboard(m["reports"],m,output/"index.html")
    exports=write_exports(m["reports"],output)
    summary=output/"summary.json";write_manifest({k:m[k] for k in ("schema_version","season","status","attempted_at","pending","errors","counters","coverage")},summary)
    files=[state/p["path"] for p in m["partitions"] if not p.get("url") and not p.get("parts")]
    files += [state/s["path"] for s in m["sources"].values() if s.get("path") and not s.get("url") and not s.get("parts")]
    files += [output/"index.html",*exports,summary]
    if config.get("publish"):
        commit=config.get("commit_sha") or os.environ.get("GITHUB_SHA") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        published=clients.get("publish",publish_run)(config["repository"],m,files,client,commit)
        m=published["manifest"]
    write_manifest(m,state/"manifest.json")
    return m

def run_command(args):
    now=parse_time(args.now) if args.now else datetime.now(timezone.utc)
    config=vars(args).copy()
    config["restore_remote"]=bool(args.publish)
    config["net_dir"]=getattr(args,"net_dir",None) or Path("config/scouting_net")
    try:
        if args.command=="audit":
            client=HTTPClient();rows={}
            for year in (args.season-1,args.season):
                for dataset in DATASETS:rows[str(year)+":"+dataset]=[{k:a.get(k) for k in ("name","size","digest","updated_at","browser_download_url")} for a in discover_assets(dataset,year,client)]
            print(json_text(rows,indent=2));return 0
        if args.command=="build":
            m=load_manifest(args.state_dir/"manifest.json");load_tables(m,args.state_dir)
            render_dashboard(m["reports"],m,args.state_dir/"output/index.html");write_exports(m["reports"],args.state_dir/"output");return 0
        m=refresh(config,now,args.trigger)
        summary={k:m.get(k) for k in ("status","attempted_at","state_release","pending","errors","counters","coverage")}
        print(json_text(summary,indent=2))
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as handle:handle.write("## WBB scouting refresh\n\nStatus: "+m["status"]+"\n\n```json\n"+json_text(summary,indent=2)+"\n```\n")
        return 1 if m.get("errors") else 0
    except Exception as exc:
        # Do not expose credential-bearing HTTP objects or signed URLs.
        print("Scouting refresh failed: "+type(exc).__name__+". Previous published state remains available.")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as handle:handle.write("WBB scouting refresh failed: "+type(exc).__name__+". Inspect prior published cutoff and rerun after resolving the cause.\n")
        return 1
