"""Correction-aware game groups and content-addressed Parquet state."""
from __future__ import annotations
from datetime import datetime, date
import hashlib
import json
import math
import numbers
from pathlib import Path
import numpy as np
import pandas as pd
from .sources import sha256_file

def clean_json(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value=value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value

def json_text(value, indent=None):
    return json.dumps(clean_json(value),sort_keys=True,ensure_ascii=False,allow_nan=False,indent=indent)

def fingerprint(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()

def frame_records(frame):
    return clean_json(frame.reindex(sorted(frame.columns),axis=1).to_dict("records"))

def reconcile_games(existing, incoming, keys, accepted_games):
    existing=existing.copy()
    incoming=incoming.drop_duplicates().copy()
    if incoming.duplicated(keys).any():
        raise ValueError("Conflicting duplicate canonical keys")
    if len(existing) and existing.duplicated(keys).any():
        raise ValueError("Conflicting duplicate stored keys")
    if len(incoming) and incoming[keys].isna().any().any():
        raise ValueError("Null canonical key")
    columns=sorted(set(existing.columns)|set(incoming.columns))
    existing=existing.reindex(columns=columns)
    incoming=incoming.reindex(columns=columns)
    old_games=set(existing["game_id"].astype(str)) if "game_id" in existing else set()
    new_games=set(incoming["game_id"].astype(str)) if "game_id" in incoming else set()
    accepted=set(map(str,accepted_games))
    replacement=incoming[incoming["game_id"].astype(str).isin(accepted)].copy()
    before=existing[existing["game_id"].astype(str).isin(accepted)].copy()
    def row_map(frame):
        return {tuple(row[k] for k in keys):fingerprint(row) for row in frame_records(frame)}
    old_map,new_map=row_map(before),row_map(replacement)
    overlap=set(old_map)&set(new_map)
    inserted=len(set(new_map)-set(old_map))
    removed=len(set(old_map)-set(new_map))
    updated=sum(old_map[k]!=new_map[k] for k in overlap)
    changed_games=set()
    for frame in (before,replacement):
        for row in frame_records(frame):
            k=tuple(row[c] for c in keys)
            if old_map.get(k)!=new_map.get(k):
                changed_games.add(str(row["game_id"]))
    kept=existing[~existing["game_id"].astype(str).isin(accepted)]
    pieces=[part for part in (kept,replacement) if not part.empty]
    out=(pd.concat(pieces,ignore_index=True) if len(pieces)>1 else pieces[0].copy() if pieces else incoming.iloc[:0].copy())
    out=out.sort_values(keys,kind="stable").reset_index(drop=True)
    return out,{"inserted":inserted,"updated":updated,"removed":removed,
                "unchanged":len(existing)-removed-updated,
                "affected_games":sorted(changed_games),"missing_games":sorted(old_games-new_games)}

def write_partitions(frame, dataset, state_dir):
    state_dir=Path(state_dir)
    objects=state_dir/"objects"
    objects.mkdir(parents=True,exist_ok=True)
    if frame.empty:
        return []
    working=frame.copy()
    if "game_date_time" in working:
        working["_month"]=pd.to_datetime(working.game_date_time,utc=True).dt.strftime("%Y-%m")
    else:
        working["_month"]="reference"
    parts=[]
    for (family,season,month),group in working.groupby(["source_family","season","_month"],dropna=False,sort=True):
        group=group.drop(columns="_month").reset_index(drop=True)
        from .adapters import KEYS
        group=group.sort_values(KEYS[dataset],kind="stable").reset_index(drop=True)
        temporary=objects/(dataset+"-"+str(season)+"-"+str(month)+".stage.parquet")
        group.to_parquet(temporary,index=False,compression="zstd")
        checksum=sha256_file(temporary)
        target=objects/(checksum+".parquet")
        if target.exists():
            temporary.unlink()
        else:
            temporary.replace(target)
        item={"dataset":dataset,"family":str(family),"season":int(season),"month":str(month),
              "path":str(target.relative_to(state_dir)),"sha256":checksum,"rows":len(group),"bytes":target.stat().st_size}
        if "game_date_time" in group:
            item.update(first_game_at=group.game_date_time.min(),last_game_at=group.game_date_time.max())
        parts.append(item)
    return parts

def write_manifest(manifest, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json_text(manifest,indent=2)+"\n")
    temporary.replace(path)

def load_manifest(path):
    value=json.loads(Path(path).read_text())
    if value.get("schema_version")!=1:
        raise ValueError("Unsupported scouting manifest schema")
    if not isinstance(value.get("partitions",[]),list):
        raise ValueError("Invalid partition manifest")
    return value

def load_tables(manifest,state_dir):
    groups={}
    for part in manifest.get("partitions",[]):
        path=Path(state_dir)/part["path"]
        if not path.resolve().is_relative_to(Path(state_dir).resolve()):
            raise ValueError("Invalid partition path")
        if not path.exists() or sha256_file(path)!=part["sha256"]:
            raise ValueError("Stored partition checksum mismatch")
        groups.setdefault(part["dataset"],[]).append(pd.read_parquet(path))
    return {name:pd.concat(frames,ignore_index=True) for name,frames in groups.items()}

def split_asset(path,directory,max_bytes=512*1024*1024):
    path,directory=Path(path),Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if path.stat().st_size<=max_bytes:
        return [{"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size}]
    parts=[]
    with path.open("rb") as handle:
        index=0
        while block:=handle.read(max_bytes):
            destination=directory/(path.name+".part-"+str(index).zfill(4))
            destination.write_bytes(block)
            parts.append({"path":str(destination),"sha256":sha256_file(destination),"bytes":len(block)})
            index+=1
    return parts
