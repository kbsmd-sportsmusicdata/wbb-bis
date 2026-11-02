"""Independent historical pilot; it cannot publish into the live season."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from .adapters import identifier
from .sources import HTTPClient,download_asset,sha256_file
from .store import json_text,fingerprint,clean_json
from .pilot_data import load_pilot_sources,clean_pilot
from .pilot_features import process_pilot,build_pilot_features

DEFAULT_CONFIG=Path(__file__).resolve().parents[1]/"config/scouting_pilot_2026.json"
GRAINS={"team_box":["season","game_id","team_id"],"player_box":["season","game_id","team_id","player_id"],
    "rosters":["season","team_id","player_id"],"shots":["source_row_id"],"team_games":["season","game_id","team_id"],
    "player_games":["season","game_id","team_id","player_id"],"roster":["season","team_id","player_id"],
    "shot_events":["source_row_id"],"team_summary":["season","team_id","window"],
    "player_summary":["season","team_id","player_id","window"],"shot_zones":["season","team_id","player_id","window","shot_zone"],
    "team_trends":["season","team_id","game_id"],"shot_groups":["game_id","team_id"],
    "box_quarantine":["dataset","game_id"],"schedule":["season","game_id"],"teams":["season","team_id"]}


def download_sources(config,raw_dir,client=None):
    client=client or HTTPClient();root=Path(raw_dir);root.mkdir(parents=True,exist_ok=True);sources=[]
    for dataset,spec in config["sources"].items():
        release=client.get_json("https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/"+spec["tag"])
        matches=[a for a in release["assets"] if a["name"]==spec["name"]]
        if len(matches)!=1:raise ValueError("Pilot source missing or ambiguous: "+dataset)
        asset={**matches[0],"dataset":dataset,"tag":spec["tag"],"season":config["season"],
               "family":"sdv_espn_wbb","league":"womens_college_basketball"}
        result=download_asset(asset,root/spec["name"],client)
        sources.append({k:v for k,v in result.items() if k not in ("columns","uploader","local_path")})
    provenance={"downloaded_at":datetime.now(timezone.utc).isoformat(),"sources":sources}
    (root/"sources.json").write_text(json_text(provenance,indent=2))
    return provenance


def write_stage(tables,root,stage,previous):
    """Row-order-independent logical hashes distinguish real changes from rebuilds."""
    root=Path(root);folder=root/stage;folder.mkdir(parents=True,exist_ok=True);refs={}
    for name,frame in sorted(tables.items()):
        frame=frame.reindex(sorted(frame.columns),axis=1).copy()
        if len(frame):
            key=[fingerprint(r) for r in clean_json(frame.to_dict("records"))]
            frame=frame.assign(_sort_key=key).sort_values("_sort_key",kind="stable").drop(columns="_sort_key").reset_index(drop=True)
        grain=GRAINS.get(name.removeprefix("support_"),[])
        note=""
        if stage.startswith("raw/"):
            grain=["athlete_id" if c=="player_id" else c for c in grain]
            if name=="shots":
                grain=[];note="Source row multiset; identical attempts are preserved because no source event ID exists"
        if grain and not set(grain).issubset(frame):raise ValueError("Missing table grain: "+stage+"."+name)
        if grain and frame[grain].isna().any().any():raise ValueError("Null table grain: "+stage+"."+name)
        if grain and frame.duplicated(grain).any():raise ValueError("Duplicate table grain: "+stage+"."+name)
        logical=fingerprint(clean_json(frame.to_dict("records")))
        parquet=folder/(name+".parquet");csv=folder/(name+".csv")
        frame.to_parquet(parquet,index=False,compression="zstd");frame.to_csv(csv,index=False)
        ident=stage+"."+name;prior=previous.get(ident,{})
        refs[ident]={"rows":len(frame),"columns":list(frame.columns),"grain":grain,"grain_note":note,
            "path":str(parquet.relative_to(root)),"csv_path":str(csv.relative_to(root)),
            "logical_sha256":logical,"parquet_sha256":sha256_file(parquet),"csv_sha256":sha256_file(csv),
            "changed":prior.get("logical_sha256")!=logical}
    return refs


def read_stage(root,stage,names):
    return {name:pd.read_parquet(Path(root)/stage/(name+".parquet")) for name in names}



def package_pilot(root,config):
    """Portable selected-team data and HTML; national originals remain source-linked."""
    import zipfile
    root=Path(root);manifest=json.loads((root/"manifest.json").read_text())
    name="six_team_pilot_"+str(config["season"])+".zip"
    archive=root/name;temporary=root/(name+".tmp")
    paths={"index.html","manifest.json","raw/sources.json"}
    for ref in manifest["tables"].values():paths.update((ref["path"],ref["csv_path"]))
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
        for relative in sorted(paths):output.write(root/relative,relative)
        output.writestr("config.json",json_text(config,indent=2))
        output.writestr("README.txt","Six-team historical scouting pilot, season "+str(config["season"])+
            "\nExtract this folder and open index.html in a browser. The report works offline; CSV and Parquet files are saved in their named layers.\n"+
            "raw/extracted contains the four original-column six-team extracts. National source snapshots are retained in the local build folder; source URLs and SHA-256 checksums are listed in raw/sources.json.\n"+
            "Opponent support rows are separate from the six featured teams. Incomplete shot groups remain documented in quality/shot_groups.csv and are excluded from shooting features.\n"+
            "Roster box-only labels describe snapshot coverage; they do not establish current eligibility or availability.\n")
    temporary.replace(archive)
    return archive


def run_pilot(config,output,now=None,raw_dir=None):
    from .calendar import parse_time
    now=parse_time(now).isoformat() if now else datetime.now(timezone.utc).isoformat()
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    raw_root=Path(raw_dir) if raw_dir else root/"raw"
    if raw_dir is None:download_sources(config,raw_root)
    raw,provenance=load_pilot_sources(raw_root,config)
    previous=json.loads((root/"manifest.json").read_text()) if (root/"manifest.json").exists() else {}
    prior=previous.get("tables",{});tables={}
    featured={t["team_id"] for t in config["teams"]}
    extracted={name:frame[frame.team_id.map(identifier).isin(featured)].copy()
               for name,frame in raw.items() if name in ("team_box","player_box","shots","rosters")}
    tables.update(write_stage(extracted,root,"raw/extracted",prior))
    clean,quality=clean_pilot(raw,config,now)
    tables.update(write_stage(clean,root,"clean",prior))
    tables.update(write_stage(quality,root,"quality",prior))
    processed=process_pilot(clean,config);tables.update(write_stage(processed,root,"processed",prior))
    features=build_pilot_features(processed,clean,config);tables.update(write_stage(features,root,"features",prior))
    # Render/report inputs are reloaded from the saved processed and feature outputs.
    processed=read_stage(root,"processed",processed)
    features=read_stage(root,"features",features)
    manifest={"schema_version":1,"kind":"historical_scouting_pilot","season":config["season"],"label":config["label"],
        "built_at":now,"teams":config["teams"],"matchups":config["matchups"],"sources":provenance["sources"],"tables":tables,
        "changed_tables":sum(r["changed"] for r in tables.values()),
        "source_data_latest_game_at":clean["team_box"].game_date_time.max(),
        "shot_groups_validated":int(quality["shot_groups"].status.eq("validated").sum()),
        "shot_groups_withheld":int(quality["shot_groups"].status.ne("validated").sum()),
        "box_quarantines":len(quality["box_quarantine"]),
        "historical_roster_warning":"2025–26 roster and game participants; no current-season eligibility or availability is inferred.",
        "pipeline_version":"six-team-pilot-1"}
    manifest=clean_json(manifest)
    (root/"manifest.json").write_text(json_text(manifest,indent=2))
    report={"config":config,"manifest":manifest,
            "processed":{k:clean_json(v.to_dict("records")) for k,v in processed.items()},
            "features":{k:clean_json(v.to_dict("records")) for k,v in features.items()},
            "quality":{k:clean_json(v.to_dict("records")) for k,v in quality.items()}}
    (root/"report_data.json").write_text(json_text(report))
    from .pilot_dashboard import pilot_payload,render_pilot_dashboard
    render_pilot_dashboard(pilot_payload(processed,features,quality,config,manifest),root/"index.html")
    package_pilot(root,config)
    return manifest


def run_pilot_command(args):
    config=json.loads((args.pilot_config or DEFAULT_CONFIG).read_text())
    if args.season!=config["season"]:raise ValueError("Pilot season differs from its explicit source configuration")
    output=args.state_dir
    if output==Path("data/scouting"):output=output/("pilot_"+str(config["season"]))
    result=run_pilot(config,output,args.now,raw_dir=args.local_input)
    print(json_text({k:result[k] for k in ("kind","season","changed_tables","shot_groups_validated","shot_groups_withheld","box_quarantines")},indent=2))
    return 0
