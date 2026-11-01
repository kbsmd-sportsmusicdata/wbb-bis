"""Dated NCAA NET snapshots and season-correct team identity."""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import pandas as pd
from .adapters import identifier
from .calendar import aware_utc,parse_time,schedule_health
from .sources import SourceUnavailable

NET_URL="https://www.ncaa.com/rankings/basketball-women/d1/ncaa-womens-basketball-net-rankings"

def load_team_reference(path:Path,season:int)->pd.DataFrame:
    payload=json.loads(Path(path).read_text())
    if payload.get("season")!=season:
        raise ValueError("Team reference season mismatch")
    frame=pd.DataFrame(payload["teams"])
    required={"team_id","team_name","conference","is_d1"}
    if not required.issubset(frame):
        raise ValueError("Missing team reference fields")
    frame["team_id"]=frame.team_id.map(identifier).astype("string")
    if frame.team_id.isna().any() or frame.team_id.duplicated().any():
        raise ValueError("Ambiguous team reference")
    if "season" in frame and not frame.season.eq(season).all():
        raise ValueError("Team row season mismatch")
    if not frame.is_d1.map(lambda value:isinstance(value,bool)).all():
        raise ValueError("Invalid division reference flag")
    frame["season"]=season
    frame.attrs["source_url"]=payload.get("source_url")
    return frame

def validate_net(frame:pd.DataFrame,publication:dict,teams:pd.DataFrame)->pd.DataFrame:
    host=urlparse(publication.get("source_url","")).hostname or ""
    if not (host=="ncaa.com" or host.endswith(".ncaa.com") or host=="ncaa.org" or host.endswith(".ncaa.org")):
        raise ValueError("NET publication must identify an official NCAA source")
    if publication.get("verified") is not True or publication.get("id_namespace")!="espn":
        raise ValueError("Reviewed official NET provenance and checked ESPN ID mapping are required")
    required={"team_id","net_rank"}
    if not required.issubset(frame):
        raise ValueError("NET requires checked team identifiers and ranks")
    result=frame.copy()
    result["team_id"]=result.team_id.map(identifier).astype("string")
    ranks=pd.to_numeric(result.net_rank,errors="coerce")
    if ranks.isna().any() or (ranks%1!=0).any() or ranks.lt(1).any():
        raise ValueError("NET ranks must be positive integers")
    if ranks.duplicated().any() or result.team_id.isna().any() or result.team_id.duplicated().any():
        raise ValueError("Conflicting NET identities/ranks")
    if not set(range(1,51)).issubset(set(ranks)):
        raise ValueError("NET top 50 coverage is incomplete")
    unknown=set(result.team_id)-set(teams.team_id.astype(str))
    if unknown:
        raise ValueError("NET has unresolved team identifiers: "+", ".join(sorted(unknown)))
    season=int(publication["season"])
    if "season" in result and not pd.to_numeric(result.season,errors="coerce").eq(season).all():
        raise ValueError("NET season disagrees with publication")
    date=pd.to_datetime(publication["publication_date"],errors="raise").date()
    if date.year not in (season-1,season):
        raise ValueError("NET publication outside season years")
    available=publication.get("published_at") or publication.get("first_seen_at") or date.isoformat()+"T23:59:59+00:00"
    available=parse_time(available)
    if date > available.date():
        raise ValueError("NET publication date is after its availability timestamp")
    if publication.get("first_seen_at"):
        available=max(available,parse_time(publication["first_seen_at"]))
    result["net_rank"]=ranks.astype(int)
    result["ranking_date"]=date.isoformat()
    result["ranking_season"]=season
    result["ranking_available_at"]=available.isoformat()
    result["ranking_source_url"]=publication["source_url"]
    result["ranking_verified"]=True
    return result

def _usable(snapshot,season,as_of):
    if snapshot is None or snapshot.empty:
        return False
    required={"ranking_season","ranking_date","ranking_available_at","ranking_verified","team_id","net_rank"}
    if not required.issubset(snapshot):
        raise ValueError("Unvalidated NET snapshot")
    return (snapshot.ranking_verified.eq(True).all()
            and snapshot.ranking_season.eq(season).all()
            and (pd.to_datetime(snapshot.ranking_available_at,utc=True)<=as_of).all())

def select_cohort(net,teams,as_of,season,historical_net=None):
    now=aware_utc(as_of)
    eligible=teams[teams.season.eq(season)&teams.is_d1.eq(True)].copy()
    status="pending_net"
    chosen=None
    if _usable(net,season,now):
        chosen=net
        seen=parse_time(chosen.ranking_available_at.iloc[0])
        status="stale_net" if schedule_health(seen,now)=="overdue" else "current_net"
    elif _usable(historical_net,season-1,now):
        chosen=historical_net
        status="provisional_prior_season"
    top_ids=set(chosen.loc[chosen.net_rank.le(50),"team_id"].astype(str)) if chosen is not None else set()
    acc_ids=set(eligible.loc[eligible.conference.eq("ACC"),"team_id"].astype(str))
    result=eligible[eligible.team_id.astype(str).isin(top_ids|acc_ids)].copy()
    result["inclusion_reason"]=result.team_id.map(lambda tid:"NET top 50 + ACC" if tid in top_ids and tid in acc_ids else "NET top 50" if tid in top_ids else "ACC")
    result["selection_status"]=status
    result["net_rank"]=pd.NA
    result["ranking_date"]=None
    result["ranking_season"]=season
    result["ranking_source_url"]=None
    result["selection_at"]=now.isoformat()
    if chosen is not None:
        lookup=chosen.set_index("team_id")
        result["ranking_date"]=str(chosen.ranking_date.iloc[0])
        result["ranking_season"]=int(chosen.ranking_season.iloc[0])
        result["ranking_source_url"]=str(chosen.ranking_source_url.iloc[0])
        result["reference_rank"]=result.team_id.map(lookup.net_rank)
        if status!="provisional_prior_season":
            result["net_rank"]=result.team_id.map(lookup.net_rank).astype("Int64")
    return result.sort_values(["conference","team_name"],kind="stable").reset_index(drop=True)

class _RankTable(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None;self.slug=None
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if tag=="table":
            self.table=[]
        elif tag=="tr" and self.table is not None:
            self.row=[]
        elif tag in ("th","td") and self.row is not None:
            self.cell=[]
        elif tag=="a" and self.cell is not None and "/schools/" in attributes.get("href",""):
            self.slug=attributes["href"].rstrip("/").split("/")[-1]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ("th","td") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()));self.cell=None
        elif tag=="tr" and self.row is not None:
            self.table.append((self.row,self.slug));self.row=None;self.slug=None
        elif tag=="table" and self.table is not None:
            self.tables.append(self.table);self.table=None

def parse_net_document(payload,content_type,publication):
    text=payload.decode("utf-8-sig",errors="strict")
    if any(marker in text.lower() for marker in ("javascript is disabled","verify that you are not a robot","captcha")):
        raise SourceUnavailable("Official NET browser verification prevents retrieval")
    if "csv" in content_type:
        return pd.read_csv(BytesIO(payload),dtype={"team_id":"string"})
    if "json" in content_type:
        value=json.loads(text)
        return pd.DataFrame(value["rows"] if isinstance(value,dict) else value)
    parser=_RankTable();parser.feed(text)
    mapping=publication.get("school_id_map",{})
    for table in parser.tables:
        if not table:continue
        headers=[re.sub(r"[^a-z0-9]","",name.lower()) for name in table[0][0]]
        rank_column=next((i for i,h in enumerate(headers) if h in ("rank","net","netrank")),None)
        school_column=next((i for i,h in enumerate(headers) if h in ("school","team")),None)
        if rank_column is None or school_column is None:continue
        rows=[]
        for cells,slug in table[1:]:
            if len(cells)<=max(rank_column,school_column):continue
            school=cells[school_column]
            team_id=mapping.get(slug) or mapping.get(school)
            if not team_id:
                raise SourceUnavailable("Official NET school identity mapping requires review")
            rows.append({"team_id":str(team_id),"net_rank":cells[rank_column],"school":school})
        if rows:return pd.DataFrame(rows)
    raise SourceUnavailable("Official NET table/publication is unavailable")
