"""
Script 13 — WPBA Fit Context Builder
====================================
Builds context layers and composite scores for WPBA pathway evaluation.

Outputs:
- analysis/conference_efficiency/team_style_efficiency_2021_2026.csv
- analysis/player_development/program_fr_sr_track_2021_2026.csv
- analysis/wpba/wpba_fit_context_2026.csv
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Optional
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (  # noqa: E402
    RAW_DIR,
    PROCESSED_DIR,
    TEAM_BOX_FILE_FULL,
    POLLS_DIR,
    RECRUITING_DIR,
    SEASON,
)


HISTORICAL_DIR = RAW_DIR / "historical"
ROSTER_DIR = REPO_ROOT / "data" / "static" / "rosters"
_KAGGLE_ENV = os.getenv("KAGGLE_DATASET_DIR")
KAGGLE_CANDIDATE_DIRS = [REPO_ROOT / "data" / "static" / "kaggle"]
if _KAGGLE_ENV:
    KAGGLE_CANDIDATE_DIRS.append(Path(_KAGGLE_ENV))

TEAM_STYLE_OUT = REPO_ROOT / "analysis" / "conference_efficiency" / "team_style_efficiency_2021_2026.csv"
PROGRAM_TRACK_OUT = REPO_ROOT / "analysis" / "player_development" / "program_fr_sr_track_2021_2026.csv"
WPBA_OUT = REPO_ROOT / "analysis" / "wpba" / f"wpba_fit_context_{SEASON}.csv"

TEAM_SEASON_FILE = PROCESSED_DIR / f"team_season_analytic_{SEASON}.csv"
HIST_PLAYER_AGG_FILE = PROCESSED_DIR / "player_box_multiseason_agg_2021_2026.csv"
ROLE_ASSIGNMENTS_FILE = REPO_ROOT / "analysis" / "role_archetypes" / f"role_archetype_assignments_{SEASON}.csv"
DEV_PROFILES_FILE = REPO_ROOT / "analysis" / "player_development" / f"player_development_profiles_{SEASON}.csv"

RECRUIT_DRAFT_CANDIDATES = [
    RECRUITING_DIR / "player_recruit_to_draft_analysis.csv",
]
_RECRUIT_DRAFT_ENV = os.getenv("RECRUIT_DRAFT_FILE")
if _RECRUIT_DRAFT_ENV:
    RECRUIT_DRAFT_CANDIDATES.append(Path(_RECRUIT_DRAFT_ENV))

HIST_DRAFT_CONFERENCE_CANDIDATES = [
    RECRUITING_DIR / "historical_draft_data_2014_2025.xlsx",
    Path.home() / "Downloads" / "historical_draft_data_2014_2025.xlsx",
]
_HIST_DRAFT_ENV = os.getenv("HIST_DRAFT_FILE")
if _HIST_DRAFT_ENV:
    HIST_DRAFT_CONFERENCE_CANDIDATES.append(Path(_HIST_DRAFT_ENV))

MASTER_PLAYER_MAP_CANDIDATES = [
    ROSTER_DIR / "master_player_season_mapping_2026.csv",
    Path.home() / "Downloads" / "master_player_season_mapping_2026.csv",
]
_MASTER_MAP_ENV = os.getenv("MASTER_PLAYER_MAP_FILE")
if _MASTER_MAP_ENV:
    MASTER_PLAYER_MAP_CANDIDATES.append(Path(_MASTER_MAP_ENV))

PAC12_REALIGN_TEAM_NORMS = {
    "arizona",
    "arizonastate",
    "california",
    "colorado",
    "oregon",
    "oregonstate",
    "stanford",
    "ucla",
    "usc",
    "utah",
    "washington",
    "washingtonstate",
}


def _to_int_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _norm_team(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
        .str.strip()
    )


def _norm_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
        .str.strip()
    )


def _parse_season_end(value: object) -> Optional[int]:
    s = str(value)
    m = re.search(r"(20\d{2})\s*[-/]\s*(\d{2,4})", s)
    if m:
        start = int(m.group(1))
        end = m.group(2)
        if len(end) == 2:
            return int(str(start)[:2] + end)
        return int(end)
    m2 = re.search(r"\b(20\d{2})\b", s)
    if m2:
        return int(m2.group(1))
    return None


def _canon_class_stage(value: object) -> float:
    s = str(value).upper().strip()
    if s in {"", "NAN", "NONE"}:
        return np.nan
    if "FRESHMAN" in s or s.startswith("FR") or "FIRST YEAR" in s or s.startswith("FY"):
        return 1.0
    if "SOPHOMORE" in s or s.startswith("SO"):
        return 2.0
    if "JUNIOR" in s or s.startswith("JR"):
        return 3.0
    if "SENIOR" in s or s.startswith("SR"):
        return 4.0
    if "GRADUATE" in s or "FIFTH YEAR" in s or "SIXTH YEAR" in s or s.startswith("GR"):
        return 5.0
    return np.nan


def _load_master_player_map() -> Optional[pd.DataFrame]:
    src = next((p for p in MASTER_PLAYER_MAP_CANDIDATES if p.exists()), None)
    if src is None:
        print("Master player-season mapping file not found; using roster fallback joins only.")
        return None

    m = pd.read_csv(src, low_memory=False)
    required = {"season", "athlete_id", "team_id", "athlete_display_name", "standardized_team_name"}
    if not required.issubset(set(m.columns)):
        print(f"Master mapping missing required cols; found {list(m.columns)}")
        return None

    m["season"] = pd.to_numeric(m["season"], errors="coerce")
    m["athlete_id"] = _to_int_id(m["athlete_id"])
    m["team_id"] = _to_int_id(m["team_id"])
    m["map_name_norm"] = _norm_name(m["athlete_display_name"])
    m["map_team_norm"] = _norm_team(m["standardized_team_name"])

    keep = [
        c
        for c in [
            "season",
            "athlete_id",
            "team_id",
            "map_name_norm",
            "map_team_norm",
            "conference_name",
            "conference_short_name",
            "groups_id_wehoop_espn",
            "standardized_team_name",
        ]
        if c in m.columns
    ]
    m = m[keep].drop_duplicates(subset=["season", "athlete_id", "team_id"])
    print(f"Loaded master player-season mapping: {src} ({len(m):,} rows)")
    return m


def _build_master_team_conference_crosswalk(master_map: Optional[pd.DataFrame]) -> pd.DataFrame:
    if master_map is None or len(master_map) == 0:
        return pd.DataFrame()

    cols = [c for c in ["team_id", "standardized_team_name", "conference_name", "conference_short_name"] if c in master_map.columns]
    if "team_id" not in cols or "conference_name" not in cols:
        return pd.DataFrame()

    mm = master_map[cols].copy()
    mm = mm.dropna(subset=["team_id", "conference_name"]).copy()
    if len(mm) == 0:
        return pd.DataFrame()

    if "standardized_team_name" in mm.columns:
        mm["map_team_norm"] = _norm_team(mm["standardized_team_name"])
    else:
        mm["map_team_norm"] = pd.NA

    # Choose most frequent conference per team_id from the source mapping.
    team_conf = (
        mm.groupby(["team_id", "conference_name"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["team_id", "n"], ascending=[True, False])
        .drop_duplicates(subset=["team_id"])
        .rename(columns={"conference_name": "conference_master"})
    )

    team_norm = (
        mm.dropna(subset=["map_team_norm", "conference_name"])
        .groupby(["map_team_norm", "conference_name"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["map_team_norm", "n"], ascending=[True, False])
        .drop_duplicates(subset=["map_team_norm"])
        .rename(columns={"conference_name": "conference_master_teamnorm"})
    )

    # Representative normalized team name by team_id for optional name-based fallback.
    team_id_norm = (
        mm.dropna(subset=["team_id", "map_team_norm"])
        .groupby(["team_id", "map_team_norm"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["team_id", "n"], ascending=[True, False])
        .drop_duplicates(subset=["team_id"])
        .drop(columns=["n"])
    )

    out = team_conf.merge(team_id_norm, on="team_id", how="left")
    out = out.merge(team_norm[["map_team_norm", "conference_master_teamnorm"]], on="map_team_norm", how="left")
    return out


def _load_historical_draft_conference_lookup() -> pd.DataFrame:
    src = next((p for p in HIST_DRAFT_CONFERENCE_CANDIDATES if p.exists()), None)
    if src is None:
        return pd.DataFrame()

    try:
        xls = pd.ExcelFile(src)
        preferred_sheets = ["historical_draft_data", "draft_historical_2014_2025"]
        sheet = next((s for s in preferred_sheets if s in xls.sheet_names), xls.sheet_names[0])
        d = pd.read_excel(src, sheet_name=sheet)
    except Exception as exc:
        print(f"Failed to read historical draft workbook for conference backfill: {exc}")
        return pd.DataFrame()

    team_col = next((c for c in ["COLLEGE", "college", "team", "school", "TEAM"] if c in d.columns), None)
    conf_col = next(
        (c for c in ["COLLEGE CONFERENCE", "college_conference", "conference", "CONFERENCE"] if c in d.columns),
        None,
    )
    year_col = next((c for c in ["YEAR DRAFTED", "year_drafted", "draft_year"] if c in d.columns), None)

    if team_col is None or conf_col is None:
        print("Historical draft workbook missing team/conference columns; skipping draft conference backfill.")
        return pd.DataFrame()

    keep_cols = [team_col, conf_col] + ([year_col] if year_col else [])
    d = d[keep_cols].copy()
    d = d.rename(columns={team_col: "college_team", conf_col: "conference_draft_hist"})
    d = d.dropna(subset=["college_team", "conference_draft_hist"]).copy()

    if year_col is not None:
        d["draft_year"] = pd.to_numeric(d[year_col], errors="coerce")
        # Pre-realignment recent era only to align with 2021-2024 conference setup.
        d = d[d["draft_year"].between(2018, 2024, inclusive="both")].copy()

    d["team_norm_draft"] = _norm_team(d["college_team"])
    d["conference_draft_hist"] = d["conference_draft_hist"].astype(str).str.strip()
    d = d[(d["team_norm_draft"] != "") & (d["conference_draft_hist"] != "")].copy()

    if len(d) == 0:
        return pd.DataFrame()

    lookup = (
        d.groupby(["team_norm_draft", "conference_draft_hist"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["team_norm_draft", "n"], ascending=[True, False])
        .drop_duplicates(subset=["team_norm_draft"])
        [["team_norm_draft", "conference_draft_hist"]]
    )
    print(f"Loaded historical draft conference lookup: {src} ({len(lookup):,} teams)")
    return lookup


def _load_roster_longitudinal() -> pd.DataFrame:
    files = sorted(ROSTER_DIR.glob("wbb_rosters_*.csv"))
    if not files:
        return pd.DataFrame()

    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["season_num"] = d.get("season", pd.Series(index=d.index)).map(_parse_season_end)
        d["name_norm"] = _norm_name(d.get("name", pd.Series("", index=d.index)))
        d["team_norm"] = _norm_team(d.get("team", pd.Series("", index=d.index)))
        class_raw = d.get("year_clean", d.get("year", pd.Series("", index=d.index)))
        d["class_year_clean"] = class_raw.astype(str).str.upper().str.strip()
        d["class_stage"] = d["class_year_clean"].map(_canon_class_stage)
        parts.append(d[["season_num", "name_norm", "team_norm", "class_year_clean", "class_stage"]])

    roster = pd.concat(parts, ignore_index=True)
    roster = roster.dropna(subset=["season_num", "name_norm"])
    roster["season_num"] = pd.to_numeric(roster["season_num"], errors="coerce")
    roster = roster[roster["season_num"].notna()].copy()
    roster["season_num"] = roster["season_num"].astype(int)
    return roster


def _first_nonnull(series: pd.Series) -> object:
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[0]


def _mode_or_first(series: pd.Series) -> object:
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    vc = s.value_counts(dropna=True)
    if len(vc) == 0:
        return s.iloc[0]
    return vc.index[0]


def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return vals.rank(pct=True, method="average", ascending=ascending) * 100.0


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = vals.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (vals - vals.mean()) / std


def _load_kaggle_team_conference_lookup() -> pd.DataFrame:
    src_dir = next(
        (
            d
            for d in KAGGLE_CANDIDATE_DIRS
            if (d / "WTeamConferences.csv").exists()
            and (d / "Conferences.csv").exists()
            and (d / "WTeams.csv").exists()
            and (d / "WTeamSpellings.csv").exists()
        ),
        None,
    )
    if src_dir is None:
        return pd.DataFrame()

    tc = pd.read_csv(src_dir / "WTeamConferences.csv", low_memory=False)
    conf = pd.read_csv(src_dir / "Conferences.csv", low_memory=False)
    wt = pd.read_csv(src_dir / "WTeams.csv", low_memory=False)
    ws = pd.read_csv(src_dir / "WTeamSpellings.csv", low_memory=False)

    tc["Season"] = pd.to_numeric(tc["Season"], errors="coerce")
    tc["TeamID"] = pd.to_numeric(tc["TeamID"], errors="coerce").astype("Int64")
    tc["ConfAbbrev"] = tc["ConfAbbrev"].astype(str).str.strip().str.lower()

    conf["ConfAbbrev"] = conf["ConfAbbrev"].astype(str).str.strip().str.lower()
    conf["Description"] = conf["Description"].astype(str).str.strip()

    wt["TeamID"] = pd.to_numeric(wt["TeamID"], errors="coerce").astype("Int64")
    ws["TeamID"] = pd.to_numeric(ws["TeamID"], errors="coerce").astype("Int64")

    wt["team_norm"] = _norm_team(wt["TeamName"])
    ws["team_norm"] = _norm_team(ws["TeamNameSpelling"])

    team_norm_map = (
        pd.concat(
            [
                wt[["TeamID", "team_norm", "TeamName"]].rename(columns={"TeamName": "team_name_kaggle"}),
                ws[["TeamID", "team_norm"]].assign(team_name_kaggle=pd.NA),
            ],
            ignore_index=True,
        )
        .dropna(subset=["TeamID", "team_norm"])
        .drop_duplicates()
    )
    canonical_name = wt[["TeamID", "TeamName"]].rename(columns={"TeamName": "team_name_kaggle"}).drop_duplicates("TeamID")
    team_norm_map = team_norm_map.merge(canonical_name, on="TeamID", how="left", suffixes=("", "_canon"))
    team_norm_map["team_name_kaggle"] = team_norm_map["team_name_kaggle_canon"].fillna(team_norm_map["team_name_kaggle"])
    team_norm_map = team_norm_map.drop(columns=["team_name_kaggle_canon"], errors="ignore")

    lookup = (
        tc.merge(conf[["ConfAbbrev", "Description"]], on="ConfAbbrev", how="left")
        .merge(team_norm_map, on="TeamID", how="left")
        .dropna(subset=["Season", "team_norm"])
    )
    if len(lookup) == 0:
        return pd.DataFrame()

    lookup = (
        lookup.groupby(["Season", "team_norm"], dropna=False)
        .agg(
            conference_kaggle=("Description", _mode_or_first),
            conf_abbrev_kaggle=("ConfAbbrev", _mode_or_first),
            team_name_kaggle=("team_name_kaggle", _first_nonnull),
            kaggle_team_id=("TeamID", _first_nonnull),
        )
        .reset_index()
        .rename(columns={"Season": "season"})
    )

    print(
        f"Loaded Kaggle conference/team lookup: {src_dir} "
        f"({len(lookup):,} season-team name rows)"
    )
    return lookup


def _load_team_box_sources() -> pd.DataFrame:
    files = sorted(HISTORICAL_DIR.glob("team_box_*.parquet"))
    if TEAM_BOX_FILE_FULL.exists() and TEAM_BOX_FILE_FULL not in files:
        files.append(TEAM_BOX_FILE_FULL)

    if not files:
        raise SystemExit("No team_box historical files found.")

    frames = []
    print("Loading team_box history...")
    for f in files:
        df = pd.read_parquet(f)
        if "season" not in df.columns:
            season_guess = pd.to_numeric("".join(ch for ch in f.stem if ch.isdigit())[-4:], errors="coerce")
            df["season"] = season_guess
        df["source_file"] = f.name
        frames.append(df)
        seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique().tolist())
        print(f"  {f.name:<24} rows={len(df):>6,} seasons={seasons}")

    out = pd.concat(frames, ignore_index=True)
    print(f"Total team_box rows loaded: {len(out):,}")
    return out


def build_team_style_efficiency() -> pd.DataFrame:
    raw = _load_team_box_sources().copy()

    for c in [
        "season",
        "season_type",
        "team_id",
        "team_score",
        "opponent_team_score",
        "field_goals_attempted",
        "three_point_field_goals_attempted",
        "free_throws_attempted",
        "offensive_rebounds",
        "turnovers",
        "total_turnovers",
        "assists",
        "field_goals_made",
        "fouls",
        "total_rebounds",
    ]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Regular season only for stable conference style baselines.
    if "season_type" in raw.columns:
        raw = raw[(raw["season_type"] == 2) | raw["season_type"].isna()].copy()

    raw["team_id"] = _to_int_id(raw.get("team_id", pd.Series(dtype="float64")))
    raw = raw[raw["team_id"].notna()].copy()

    tov_col = "total_turnovers" if "total_turnovers" in raw.columns else "turnovers"

    fga = raw.get("field_goals_attempted", pd.Series(0, index=raw.index)).fillna(0)
    fta = raw.get("free_throws_attempted", pd.Series(0, index=raw.index)).fillna(0)
    oreb = raw.get("offensive_rebounds", pd.Series(0, index=raw.index)).fillna(0)
    tov = raw.get(tov_col, pd.Series(0, index=raw.index)).fillna(0)

    raw["possessions_est"] = (fga + 0.44 * fta - oreb + tov).clip(lower=1)

    team_cols = [
        "team_id",
        "season",
        "team_location",
        "team_name",
        "team_display_name",
        "team_short_display_name",
    ]
    team_cols = [c for c in team_cols if c in raw.columns]

    agg = (
        raw.groupby(team_cols, dropna=False)
        .agg(
            games=("game_id", "nunique") if "game_id" in raw.columns else ("team_id", "size"),
            pts_total=("team_score", "sum"),
            opp_pts_total=("opponent_team_score", "sum"),
            poss_total=("possessions_est", "sum"),
            fga_total=("field_goals_attempted", "sum"),
            fg3a_total=("three_point_field_goals_attempted", "sum"),
            fta_total=("free_throws_attempted", "sum"),
            ast_total=("assists", "sum"),
            fgm_total=("field_goals_made", "sum"),
            tov_total=(tov_col, "sum"),
            fouls_total=("fouls", "sum"),
            oreb_total=("offensive_rebounds", "sum"),
            reb_total=("total_rebounds", "sum"),
        )
        .reset_index()
    )

    poss_safe = agg["poss_total"].replace(0, np.nan)
    fga_safe = agg["fga_total"].replace(0, np.nan)
    fgm_safe = agg["fgm_total"].replace(0, np.nan)
    g_safe = agg["games"].replace(0, np.nan)

    agg["offensive_eff"] = agg["pts_total"] * 100 / poss_safe
    agg["defensive_eff"] = agg["opp_pts_total"] * 100 / poss_safe
    agg["net_eff"] = agg["offensive_eff"] - agg["defensive_eff"]
    agg["pace"] = agg["poss_total"] / g_safe

    agg["three_rate"] = agg["fg3a_total"] / fga_safe
    agg["ft_rate"] = agg["fta_total"] / fga_safe
    agg["assist_rate"] = agg["ast_total"] / fgm_safe
    agg["turnover_rate"] = agg["tov_total"] / poss_safe
    agg["fouls_per100"] = agg["fouls_total"] * 100 / poss_safe
    agg["oreb_rate"] = agg["oreb_total"] / agg["reb_total"].replace(0, np.nan)

    # Physicality proxy: fouls + glass pressure.
    agg["physicality_index"] = 0.65 * _zscore(agg["fouls_per100"]) + 0.35 * _zscore(agg["oreb_rate"])

    # Conference enrichment.
    agg["conference"] = pd.NA
    agg["team_norm"] = _norm_team(agg["team_location"]).replace("", pd.NA)
    agg["team_location_standardized"] = agg["team_location"]

    # Kaggle reference mapping (season + normalized team name) for conference backfill and name standardization.
    kaggle_lookup = _load_kaggle_team_conference_lookup()
    if len(kaggle_lookup) > 0:
        agg = agg.merge(kaggle_lookup, on=["season", "team_norm"], how="left")
        agg["conference"] = agg["conference"].fillna(agg["conference_kaggle"])
        agg["team_location_standardized"] = agg["team_name_kaggle"].fillna(agg["team_location_standardized"])
        agg.drop(columns=["conference_kaggle", "conf_abbrev_kaggle", "team_name_kaggle"], inplace=True, errors="ignore")

    polls_file = POLLS_DIR / "polls_historical_analytics.csv"
    if polls_file.exists():
        polls = pd.read_csv(polls_file, low_memory=False)
        for c in ["season"]:
            if c in polls.columns:
                polls[c] = pd.to_numeric(polls[c], errors="coerce")
        polls = polls[[c for c in ["season", "team", "conference"] if c in polls.columns]].dropna(subset=["season", "team"])
        polls["team_norm"] = _norm_team(polls["team"])
        polls = polls.drop_duplicates(subset=["season", "team_norm"]) 
        agg = agg.merge(
            polls[["season", "team_norm", "conference"]].rename(columns={"conference": "conference_polls"}),
            on=["season", "team_norm"],
            how="left",
        )
        agg["conference"] = agg["conference"].fillna(agg["conference_polls"])
        agg.drop(columns=["conference_polls"], inplace=True)

    if TEAM_SEASON_FILE.exists():
        tcur = pd.read_csv(TEAM_SEASON_FILE, low_memory=False)
        if "team_id" in tcur.columns and "conference" in tcur.columns:
            tcur["team_id"] = _to_int_id(tcur["team_id"])
            tcur = tcur[["team_id", "conference"]].drop_duplicates("team_id")
            agg = agg.merge(tcur.rename(columns={"conference": "conference_2026"}), on="team_id", how="left")
            season_num = pd.to_numeric(agg["season"], errors="coerce")
            current_align_mask = season_num.isin([2025, SEASON])
            agg.loc[current_align_mask, "conference"] = agg.loc[current_align_mask, "conference"].fillna(
                agg.loc[current_align_mask, "conference_2026"]
            )
            agg.drop(columns=["conference_2026"], inplace=True)

    # Master mapping enrichment (first-priority by team_id), with conference realignment handling:
    # - 2025/2026 use current alignments from master mapping.
    # - 2021-2024 keep Pac-12-era conference label for realigned Pac-12 programs.
    master_map = _load_master_player_map()
    team_conf = _build_master_team_conference_crosswalk(master_map)
    if len(team_conf) > 0:
        if "team_id" in team_conf.columns:
            team_conf["team_id"] = _to_int_id(team_conf["team_id"])

        merge_cols = [c for c in ["team_id", "map_team_norm", "conference_master", "conference_master_teamnorm"] if c in team_conf.columns]
        agg = agg.merge(team_conf[merge_cols], on="team_id", how="left")

        # Fallback conference from team name normalization for any unresolved team_id match.
        if "conference_master_teamnorm" in agg.columns:
            agg["conference_master"] = agg["conference_master"].fillna(agg["conference_master_teamnorm"])

        pac12_moved = agg["team_norm"].isin(PAC12_REALIGN_TEAM_NORMS)
        season_num = pd.to_numeric(agg["season"], errors="coerce")
        is_legacy_era = season_num.between(2021, 2024, inclusive="both")
        is_new_align_era = season_num >= 2025

        # 2025+ : use mapped conference where current conference is missing.
        mask_new = is_new_align_era & agg["conference"].isna()
        agg.loc[mask_new, "conference"] = agg.loc[mask_new, "conference_master"]

        # 2021-2024 : keep legacy Pac-12 label for moved programs.
        mask_legacy_pac = is_legacy_era & pac12_moved & agg["conference"].isna()
        agg.loc[mask_legacy_pac, "conference"] = "Pac-12 Conference"

        # 2021-2024 : for non-realigned teams, mapped conference can safely backfill missing values.
        mask_legacy_other = is_legacy_era & (~pac12_moved) & agg["conference"].isna()
        agg.loc[mask_legacy_other, "conference"] = agg.loc[mask_legacy_other, "conference_master"]

        agg.drop(columns=["map_team_norm", "conference_master", "conference_master_teamnorm"], inplace=True, errors="ignore")

    # Historical draft conference lookup backfill: only for 2021-2024 missing values.
    draft_lookup = _load_historical_draft_conference_lookup()
    if len(draft_lookup) > 0:
        agg = agg.merge(draft_lookup, left_on="team_norm", right_on="team_norm_draft", how="left")
        season_num = pd.to_numeric(agg["season"], errors="coerce")
        missing_before = int((season_num.between(2021, 2024, inclusive="both") & agg["conference"].isna()).sum())
        fill_mask = season_num.between(2021, 2024, inclusive="both") & agg["conference"].isna()
        agg.loc[fill_mask, "conference"] = agg.loc[fill_mask, "conference_draft_hist"]
        missing_after = int((season_num.between(2021, 2024, inclusive="both") & agg["conference"].isna()).sum())
        print(
            "Draft historical conference backfill (2021-2024): "
            f"filled {missing_before - missing_after:,} rows; remaining missing {missing_after:,}"
        )
        agg.drop(columns=["team_norm_draft", "conference_draft_hist"], inplace=True, errors="ignore")

    agg = agg.drop(columns=["team_norm"], errors="ignore")

    TEAM_STYLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg = agg.sort_values(["season", "team_location"]).reset_index(drop=True)
    agg.to_csv(TEAM_STYLE_OUT, index=False)

    print(f"Saved team style history: {TEAM_STYLE_OUT}")
    print(f"  Rows: {len(agg):,} | seasons: {sorted(agg['season'].dropna().astype(int).unique().tolist())}")
    return agg


def build_program_fr_sr_track() -> pd.DataFrame:
    if not HIST_PLAYER_AGG_FILE.exists():
        raise SystemExit(f"Missing required historical player aggregate file: {HIST_PLAYER_AGG_FILE}")

    df = pd.read_csv(HIST_PLAYER_AGG_FILE, low_memory=False)
    keep = [
        "athlete_id",
        "athlete_display_name",
        "season",
        "team_id",
        "team_location",
        "pts_per40",
        "ast_per40",
        "tov_per40",
        "ts_pct",
        "usage_proxy",
        "minutes_total",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df["athlete_id"] = _to_int_id(df["athlete_id"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["team_id"] = _to_int_id(df.get("team_id", pd.Series(dtype="float64")))
    df = df.dropna(subset=["athlete_id", "season", "team_location", "athlete_display_name"]).copy()
    df["name_norm"] = _norm_name(df["athlete_display_name"])
    df["team_norm"] = _norm_team(df["team_location"])

    master_map = _load_master_player_map()
    if master_map is not None:
        df = df.merge(master_map, on=["season", "athlete_id", "team_id"], how="left")
        master_hit = float(df.get("map_name_norm", pd.Series(index=df.index)).notna().mean())
        print(f"Master mapping join coverage: {master_hit:.1%}")
        df["join_name_norm"] = df.get("map_name_norm").fillna(df["name_norm"])
        df["join_team_norm"] = df.get("map_team_norm").fillna(df["team_norm"])
    else:
        df["join_name_norm"] = df["name_norm"]
        df["join_team_norm"] = df["team_norm"]

    roster = _load_roster_longitudinal()
    if roster.empty:
        print("Roster longitudinal files missing; falling back to year-in-program proxy only.")
        df["class_stage"] = np.nan
        df["class_match_source"] = pd.NA
    else:
        # Stage 1 (first priority): season + mapped name + mapped team.
        exact = (
            roster.groupby(["season_num", "name_norm", "team_norm"], dropna=False)
            .agg(
                class_stage=("class_stage", _first_nonnull),
                class_year_clean=("class_year_clean", _first_nonnull),
            )
            .reset_index()
        )
        stage1 = df.merge(
            exact,
            left_on=["season", "join_name_norm", "join_team_norm"],
            right_on=["season_num", "name_norm", "team_norm"],
            how="left",
            suffixes=("", "_r"),
        )
        stage1["class_match_source"] = np.where(stage1["class_stage"].notna(), "exact_team", pd.NA)

        # Stage 2 fallback: season + mapped name when unique in roster for that season.
        unresolved = stage1[stage1["class_stage"].isna()].copy()
        if len(unresolved) > 0:
            name_counts = roster.groupby(["season_num", "name_norm"], dropna=False).size().reset_index(name="n")
            uniq = roster.merge(name_counts, on=["season_num", "name_norm"], how="left")
            uniq = uniq[uniq["n"] == 1].copy()
            uniq = (
                uniq.groupby(["season_num", "name_norm"], dropna=False)
                .agg(
                    class_stage=("class_stage", _first_nonnull),
                    class_year_clean=("class_year_clean", _first_nonnull),
                )
                .reset_index()
            )
            fb = unresolved.merge(
                uniq,
                left_on=["season", "join_name_norm"],
                right_on=["season_num", "name_norm"],
                how="left",
                suffixes=("", "_fb"),
            )
            idx = fb.index
            take = fb["class_stage_fb"].notna()
            stage1.loc[idx[take], "class_stage"] = fb.loc[take, "class_stage_fb"].values
            stage1.loc[idx[take], "class_year_clean"] = fb.loc[take, "class_year_clean_fb"].values
            stage1.loc[idx[take], "class_match_source"] = "unique_name_fallback"

        df = stage1.drop(columns=["season_num", "name_norm_r", "team_norm_r"], errors="ignore")
        class_cov = float(df["class_stage"].notna().mean())
        print(f"Roster class-year join coverage (before proxy): {class_cov:.1%}")

    # Year-in-program proxy fallback (only where class_stage is missing).
    df = df.sort_values(["athlete_id", "team_location", "season"]).copy()
    df["year_in_program"] = df.groupby(["athlete_id", "team_location"]).cumcount() + 1
    df["year_in_program"] = df["year_in_program"].clip(upper=5)
    df["class_stage_final"] = pd.to_numeric(df.get("class_stage"), errors="coerce")
    miss = df["class_stage_final"].isna()
    df.loc[miss, "class_stage_final"] = df.loc[miss, "year_in_program"]
    df.loc[miss, "class_match_source"] = "year_in_program_proxy"
    df["class_stage_final"] = df["class_stage_final"].clip(lower=1, upper=5)

    metrics = [c for c in ["pts_per40", "ast_per40", "tov_per40", "ts_pct", "usage_proxy"] if c in df.columns]

    # Program x class-stage aggregates.
    py = (
        df.groupby(["team_location", "class_stage_final"], dropna=False)
        .agg(
            players=("athlete_id", "nunique"),
            avg_minutes=("minutes_total", "mean"),
            **{f"avg_{m}": (m, "mean") for m in metrics},
        )
        .reset_index()
    )
    py["class_stage_final"] = pd.to_numeric(py["class_stage_final"], errors="coerce").astype("Int64")

    # Wide table for deltas.
    wide_parts = []
    for y in [1, 2, 3, 4, 5]:
        chunk = py[py["class_stage_final"] == y].copy()
        rename = {c: f"{c}_y{y}" for c in chunk.columns if c not in ["team_location", "class_stage_final"]}
        chunk = chunk.rename(columns=rename).drop(columns=["class_stage_final"])
        wide_parts.append(chunk)

    prog = wide_parts[0]
    for chunk in wide_parts[1:]:
        prog = prog.merge(chunk, on="team_location", how="outer")

    # Delta metrics.
    def add_delta(metric: str, y_to: int) -> None:
        c1 = f"avg_{metric}_y1"
        c2 = f"avg_{metric}_y{y_to}"
        if c1 in prog.columns and c2 in prog.columns:
            prog[f"{metric}_y1_y{y_to}_delta"] = prog[c2] - prog[c1]

    for m in metrics:
        for y_to in [2, 3, 4]:
            add_delta(m, y_to)

    # Composite FR->SR proxy score (y1->y4 emphasis where available).
    pos_terms = [c for c in ["pts_per40_y1_y4_delta", "ast_per40_y1_y4_delta", "ts_pct_y1_y4_delta", "usage_proxy_y1_y4_delta"] if c in prog.columns]
    neg_terms = [c for c in ["tov_per40_y1_y4_delta"] if c in prog.columns]

    score = pd.Series(0.0, index=prog.index)
    if pos_terms:
        score += sum(_zscore(prog[c]).fillna(0) for c in pos_terms) / len(pos_terms)
    if neg_terms:
        score -= sum(_zscore(prog[c]).fillna(0) for c in neg_terms) / len(neg_terms)
    prog["program_fr_sr_track_score"] = score
    prog["program_fr_sr_track_score_0_100"] = _pct_rank(score, ascending=True).round(2)

    # Coverage diagnostics for transparency.
    coverage = (
        df.groupby("class_match_source", dropna=False)["athlete_id"]
        .nunique()
        .rename("players")
        .reset_index()
        .sort_values("players", ascending=False)
    )
    total_players = max(1, int(df["athlete_id"].nunique()))
    for _, r in coverage.iterrows():
        src = str(r["class_match_source"])
        pct = 100.0 * float(r["players"]) / total_players
        print(f"  Class-stage source {src}: {int(r['players']):,} players ({pct:.1f}%)")

    PROGRAM_TRACK_OUT.parent.mkdir(parents=True, exist_ok=True)
    prog = prog.sort_values("team_location").reset_index(drop=True)
    prog.to_csv(PROGRAM_TRACK_OUT, index=False)

    print(f"Saved program FR->SR track table: {PROGRAM_TRACK_OUT}")
    print(f"  Rows (programs): {len(prog):,}")
    return prog


def _load_recruit_draft() -> Optional[pd.DataFrame]:
    src = next((p for p in RECRUIT_DRAFT_CANDIDATES if p.exists()), None)
    if src is None:
        print("Recruit-to-draft file not found in known locations; continuing without draft context.")
        return None
    df = pd.read_csv(src, low_memory=False)
    if "athlete_id" in df.columns:
        df["athlete_id"] = _to_int_id(df["athlete_id"])
    print(f"Loaded recruit-to-draft context: {src} ({len(df):,} rows)")
    return df


def build_wpba_fit(team_style: pd.DataFrame, program_track: pd.DataFrame) -> pd.DataFrame:
    if not ROLE_ASSIGNMENTS_FILE.exists():
        raise SystemExit(f"Missing role assignments file: {ROLE_ASSIGNMENTS_FILE}")
    if not DEV_PROFILES_FILE.exists():
        raise SystemExit(f"Missing development profiles file: {DEV_PROFILES_FILE}")

    role = pd.read_csv(ROLE_ASSIGNMENTS_FILE, low_memory=False)
    dev = pd.read_csv(DEV_PROFILES_FILE, low_memory=False)

    role["athlete_id"] = _to_int_id(role["athlete_id"])
    dev["athlete_id"] = _to_int_id(dev["athlete_id"])

    # Lean development subset for stable join shape.
    dev_keep = [
        "athlete_id",
        "readiness_index",
        "development_status",
        "yoy_growth_score",
        "pts_per40_yoy_delta",
        "ts_pct_yoy_delta",
        "usage_proxy_yoy_delta",
        "game_score_trend",
        "ts_trend",
        "consistency_recent_std",
        "flag_skill_growth",
        "flag_consistency_improving",
        "flag_clutch_trend_up",
        "flag_tough_opponent_ready",
    ]
    dev_keep = [c for c in dev_keep if c in dev.columns]
    dev = dev[dev_keep].drop_duplicates(subset=["athlete_id"])

    out = role.merge(dev, on="athlete_id", how="left", suffixes=("", "_dev"))

    # Team style context for configured season.
    ts = team_style.copy()
    ts = ts[pd.to_numeric(ts.get("season"), errors="coerce") == SEASON].copy()
    if "team_id" in ts.columns and "team_id" in out.columns:
        ts["team_id"] = _to_int_id(ts["team_id"])
        out["team_id"] = _to_int_id(out["team_id"])
        ts_keep = [c for c in ["team_id", "conference", "offensive_eff", "defensive_eff", "pace", "physicality_index"] if c in ts.columns]
        out = out.merge(ts[ts_keep].drop_duplicates("team_id"), on="team_id", how="left")

    # Program FR->SR context by team_location.
    if "team_location" in out.columns and "team_location" in program_track.columns:
        prog_keep = [c for c in ["team_location", "program_fr_sr_track_score", "program_fr_sr_track_score_0_100"] if c in program_track.columns]
        out = out.merge(program_track[prog_keep].drop_duplicates("team_location"), on="team_location", how="left")

    # Recruit-draft context.
    recruit_draft = _load_recruit_draft()
    if recruit_draft is not None:
        keep = [
            c
            for c in [
                "athlete_id",
                "in_draft_2026",
                "draft_prob_early",
                "draft_prob_mid",
                "draft_prob_late",
                "impact_delta",
                "recruit_rank",
            ]
            if c in recruit_draft.columns
        ]
        out = out.merge(recruit_draft[keep].drop_duplicates("athlete_id"), on="athlete_id", how="left", suffixes=("", "_draft"))

    # -------------------- Archetype score (0-100) --------------------
    role_conf = pd.to_numeric(out.get("role_confidence"), errors="coerce") * 100.0
    ts_pctile = pd.to_numeric(out.get("ts_pct_pctile_pos"), errors="coerce")
    usage_pctile = pd.to_numeric(out.get("usage_proxy_pctile_pos"), errors="coerce")
    stocks_mix = (
        pd.to_numeric(out.get("stl_per40_pctile_pos"), errors="coerce")
        + pd.to_numeric(out.get("blk_per40_pctile_pos"), errors="coerce")
    ) / 2.0
    ast_tov_pctile = pd.to_numeric(out.get("ast_to_tov_pctile_pos"), errors="coerce")

    archetype_score = (
        0.45 * role_conf.fillna(50)
        + 0.20 * ts_pctile.fillna(50)
        + 0.15 * usage_pctile.fillna(50)
        + 0.10 * stocks_mix.fillna(50)
        + 0.10 * ast_tov_pctile.fillna(50)
    )
    out["archetype_score_0_100"] = archetype_score.clip(0, 100).round(2)

    # -------------------- Development trajectory score (0-100) --------------------
    readiness_pct = _pct_rank(pd.to_numeric(out.get("readiness_index"), errors="coerce"), ascending=True)
    yoy_pct = _pct_rank(pd.to_numeric(out.get("yoy_growth_score"), errors="coerce"), ascending=True)
    trend_pct = _pct_rank(pd.to_numeric(out.get("game_score_trend"), errors="coerce"), ascending=True)
    ts_trend_pct = _pct_rank(pd.to_numeric(out.get("ts_trend"), errors="coerce"), ascending=True)
    consistency_pct = _pct_rank(pd.to_numeric(out.get("consistency_recent_std"), errors="coerce"), ascending=False)

    dev_score = (
        0.34 * readiness_pct.fillna(50)
        + 0.26 * yoy_pct.fillna(50)
        + 0.18 * trend_pct.fillna(50)
        + 0.12 * ts_trend_pct.fillna(50)
        + 0.10 * consistency_pct.fillna(50)
    )
    out["development_trajectory_score_0_100"] = dev_score.clip(0, 100).round(2)

    # -------------------- Conference/program context score (0-100) --------------------
    # Conference-relative player efficiency.
    ts_player = pd.to_numeric(out.get("ts_pct"), errors="coerce")
    on_net = pd.to_numeric(out.get("on_net_rtg"), errors="coerce")
    net_diff = pd.to_numeric(out.get("net_rtg_diff"), errors="coerce")

    group_keys = [c for c in ["conference", "athlete_position_abbreviation"] if c in out.columns]
    if len(group_keys) == 2:
        out["conf_pos_ts_avg"] = out.groupby(group_keys)["ts_pct"].transform("mean")
    elif "conference" in out.columns:
        out["conf_pos_ts_avg"] = out.groupby("conference")["ts_pct"].transform("mean")
    else:
        out["conf_pos_ts_avg"] = np.nan

    out["ts_plus_vs_conf"] = ts_player - pd.to_numeric(out["conf_pos_ts_avg"], errors="coerce")

    ts_plus_pct = _pct_rank(pd.to_numeric(out["ts_plus_vs_conf"], errors="coerce"), ascending=True)
    on_net_pct = _pct_rank(on_net, ascending=True)
    net_diff_pct = _pct_rank(net_diff, ascending=True)
    prog_track_pct = pd.to_numeric(out.get("program_fr_sr_track_score_0_100"), errors="coerce")

    context_score = (
        0.36 * ts_plus_pct.fillna(50)
        + 0.28 * on_net_pct.fillna(50)
        + 0.20 * net_diff_pct.fillna(50)
        + 0.16 * prog_track_pct.fillna(50)
    )
    out["conference_program_context_score_0_100"] = context_score.clip(0, 100).round(2)

    # -------------------- Undervalued bump + final WPBA fit --------------------
    recruit_rank = pd.to_numeric(out.get("recruit_rank"), errors="coerce")
    in_draft_raw = out.get("in_draft_2026")
    if in_draft_raw is None:
        in_draft = pd.Series(False, index=out.index, dtype=bool)
    else:
        in_draft = pd.Series(in_draft_raw, index=out.index).map(lambda v: bool(v) if pd.notna(v) else False)

    undervalued_flag = ((recruit_rank > 100) | recruit_rank.isna()) & in_draft
    out["undervalued_bump"] = np.where(undervalued_flag, 5.0, 0.0)

    wpba_score = (
        0.40 * out["archetype_score_0_100"]
        + 0.35 * out["development_trajectory_score_0_100"]
        + 0.25 * out["conference_program_context_score_0_100"]
        + out["undervalued_bump"]
    )
    out["wpba_pathway_fit_score_0_100"] = wpba_score.clip(0, 100).round(2)

    # Tier labels for delivery use.
    out["wpba_fit_band"] = pd.cut(
        out["wpba_pathway_fit_score_0_100"],
        bins=[-np.inf, 45, 60, 75, np.inf],
        labels=["Monitor", "Viable", "Strong", "Priority"],
    ).astype(str)

    # Final columns.
    final_cols = [
        "athlete_id",
        "athlete_display_name",
        "team_location",
        "conference",
        "athlete_position_abbreviation",
        "role_code",
        "role_name",
        "role_confidence",
        "development_status",
        "readiness_index",
        "yoy_growth_score",
        "ts_plus_vs_conf",
        "program_fr_sr_track_score_0_100",
        "in_draft_2026",
        "draft_prob_early",
        "draft_prob_mid",
        "draft_prob_late",
        "archetype_score_0_100",
        "development_trajectory_score_0_100",
        "conference_program_context_score_0_100",
        "undervalued_bump",
        "wpba_pathway_fit_score_0_100",
        "wpba_fit_band",
    ]
    final_cols = [c for c in final_cols if c in out.columns]

    out = out[final_cols].copy()
    out = out.sort_values("wpba_pathway_fit_score_0_100", ascending=False).reset_index(drop=True)

    WPBA_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WPBA_OUT, index=False)

    print(f"Saved WPBA fit context: {WPBA_OUT}")
    print(f"  Rows: {len(out):,}")
    print("  Top 5:")
    print(out[[c for c in ["athlete_display_name", "team_location", "role_name", "wpba_pathway_fit_score_0_100"] if c in out.columns]].head(5).to_string(index=False))
    return out


def main() -> None:
    team_style = build_team_style_efficiency()
    program_track = build_program_fr_sr_track()
    _ = build_wpba_fit(team_style=team_style, program_track=program_track)


if __name__ == "__main__":
    main()
