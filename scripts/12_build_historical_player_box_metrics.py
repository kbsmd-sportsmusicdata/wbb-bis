"""
Script 12 — Historical Player Box Aggregation + Percentiles + YoY
===================================================================
Builds multi-season player aggregates from raw player_box parquet files,
then computes position-level weighted percentiles and YoY deltas.

Primary use case: feed YoY development signals into Script 11.

Inputs:
- data/raw/historical/player_box_*.parquet
- data/raw/player_box_2026_final.parquet (from config PLAYER_BOX_FILE)

Outputs:
- data/processed/player_box_multiseason_agg_2021_2026.csv
- data/processed/player_box_multiseason_yoy_2021_2026.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PLAYER_BOX_FILE, PROCESSED_DIR, RAW_DIR, SEASON


HISTORICAL_DIR = RAW_DIR / "historical"
AGG_OUT = PROCESSED_DIR / "player_box_multiseason_agg_2021_2026.csv"
YOY_OUT = PROCESSED_DIR / "player_box_multiseason_yoy_2021_2026.csv"


def _weighted_percentile(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Weighted percentile in [0, 1], aligned to original row index."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0)

    out = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = v.notna() & (w > 0)
    if valid.sum() == 0:
        return out

    tmp = pd.DataFrame({"v": v[valid], "w": w[valid]})
    tmp = tmp.sort_values(["v", "w"], kind="mergesort")

    cw = tmp["w"].cumsum()
    total_w = tmp["w"].sum()
    p = cw / total_w if total_w > 0 else np.nan
    out.loc[tmp.index] = p
    return out


def _load_sources() -> pd.DataFrame:
    files: List[Path] = sorted(HISTORICAL_DIR.glob("player_box_*.parquet"))

    # Include current configured season file if present and not already in historical dir.
    if PLAYER_BOX_FILE.exists() and PLAYER_BOX_FILE not in files:
        files.append(PLAYER_BOX_FILE)

    if not files:
        raise SystemExit("No player_box parquet files found in historical/ or config PLAYER_BOX_FILE.")

    parts = []
    print("Loading player_box parquet files...")
    for f in files:
        df = pd.read_parquet(f)
        src = f.name
        if "season" not in df.columns:
            # Conservative fallback from filename if missing.
            try:
                season_guess = int("".join(ch for ch in f.stem if ch.isdigit())[-4:])
            except Exception:
                season_guess = np.nan
            df["season"] = season_guess
        df["source_file"] = src
        parts.append(df)
        seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique().tolist())
        print(f"  {src:<35} rows={len(df):>8,} seasons={seasons}")

    raw = pd.concat(parts, ignore_index=True)
    print(f"Total loaded rows: {len(raw):,}")
    return raw


def _clean_filter(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    for c in [
        "minutes",
        "points",
        "rebounds",
        "offensive_rebounds",
        "defensive_rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "fouls",
        "field_goals_made",
        "field_goals_attempted",
        "three_point_field_goals_made",
        "three_point_field_goals_attempted",
        "free_throws_made",
        "free_throws_attempted",
        "season",
    ]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    if "did_not_play" in work.columns:
        work = work[~work["did_not_play"].fillna(False)].copy()

    # Regular season only for stable YoY comparisons unless season_type missing.
    if "season_type" in work.columns:
        st = pd.to_numeric(work["season_type"], errors="coerce")
        work = work[(st == 2) | st.isna()].copy()

    work = work[work["minutes"].fillna(0) > 0].copy()

    # Keep only usable athlete ids.
    work["athlete_id"] = pd.to_numeric(work.get("athlete_id"), errors="coerce").astype("Int64")
    work = work[work["athlete_id"].notna()].copy()

    # Fill identity fallbacks.
    if "athlete_position_abbreviation" in work.columns:
        work["athlete_position_abbreviation"] = (
            work["athlete_position_abbreviation"].fillna("UNK").astype(str).str.upper().str.strip().replace({"": "UNK"})
        )
    else:
        work["athlete_position_abbreviation"] = "UNK"

    if "team_location" not in work.columns and "team_short_display_name" in work.columns:
        work["team_location"] = work["team_short_display_name"]

    if "starter" in work.columns:
        work["starter"] = pd.to_numeric(work["starter"], errors="coerce").fillna(0)
    else:
        work["starter"] = 0

    work["season"] = pd.to_numeric(work["season"], errors="coerce").astype("Int64")
    work = work[work["season"].notna()].copy()

    print(f"Rows after filters (played + regular season): {len(work):,}")
    print(f"Seasons in cleaned data: {sorted(work['season'].dropna().astype(int).unique().tolist())}")
    return work


def _aggregate_season(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "athlete_id",
        "season",
        "athlete_display_name",
        "team_id",
        "team_location",
        "team_short_display_name",
        "athlete_position_abbreviation",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            games_played=("game_id", "nunique") if "game_id" in df.columns else ("minutes", "size"),
            games_started=("starter", "sum"),
            minutes_total=("minutes", "sum"),
            pts_total=("points", "sum"),
            reb_total=("rebounds", "sum"),
            oreb_total=("offensive_rebounds", "sum"),
            dreb_total=("defensive_rebounds", "sum"),
            ast_total=("assists", "sum"),
            stl_total=("steals", "sum"),
            blk_total=("blocks", "sum"),
            tov_total=("turnovers", "sum"),
            fouls_total=("fouls", "sum"),
            fgm=("field_goals_made", "sum"),
            fga=("field_goals_attempted", "sum"),
            fg3m=("three_point_field_goals_made", "sum"),
            fg3a=("three_point_field_goals_attempted", "sum"),
            ftm=("free_throws_made", "sum"),
            fta=("free_throws_attempted", "sum"),
        )
        .reset_index()
    )

    # Rate stats.
    gp = agg["games_played"].replace(0, np.nan)
    mins = agg["minutes_total"].replace(0, np.nan)

    agg["mpg"] = agg["minutes_total"] / gp
    agg["ppg"] = agg["pts_total"] / gp
    agg["rpg"] = agg["reb_total"] / gp
    agg["apg"] = agg["ast_total"] / gp
    agg["spg"] = agg["stl_total"] / gp
    agg["bpg"] = agg["blk_total"] / gp
    agg["tovpg"] = agg["tov_total"] / gp

    for total_col, out_col in [
        ("pts_total", "pts_per40"),
        ("reb_total", "reb_per40"),
        ("oreb_total", "oreb_per40"),
        ("dreb_total", "dreb_per40"),
        ("ast_total", "ast_per40"),
        ("stl_total", "stl_per40"),
        ("blk_total", "blk_per40"),
        ("tov_total", "tov_per40"),
        ("fouls_total", "fouls_per40"),
    ]:
        agg[out_col] = (agg[total_col] / mins) * 40.0

    # Shooting & advanced.
    agg["fg_pct"] = np.where(agg["fga"] > 0, agg["fgm"] / agg["fga"], np.nan)
    agg["fg3_pct"] = np.where(agg["fg3a"] > 0, agg["fg3m"] / agg["fg3a"], np.nan)
    agg["ft_pct"] = np.where(agg["fta"] > 0, agg["ftm"] / agg["fta"], np.nan)

    agg["threepar"] = np.where(agg["fga"] > 0, agg["fg3a"] / agg["fga"], np.nan)
    agg["fta_rate"] = np.where(agg["fga"] > 0, agg["fta"] / agg["fga"], np.nan)

    agg["efg_pct"] = np.where(agg["fga"] > 0, (agg["fgm"] + 0.5 * agg["fg3m"]) / agg["fga"], np.nan)
    tsa = agg["fga"] + 0.44 * agg["fta"]
    agg["ts_pct"] = np.where(tsa > 0, agg["pts_total"] / (2 * tsa), np.nan)

    agg["usage_load"] = agg["fga"] + 0.44 * agg["fta"] + agg["tov_total"]
    agg["usage_proxy"] = agg["usage_load"] / mins

    agg["ast_to_tov"] = np.where(agg["tov_total"] > 0, agg["ast_total"] / agg["tov_total"], np.nan)
    agg["foul_rate"] = agg["fouls_total"] / mins

    poss_est = agg["fga"] + 0.44 * agg["fta"] + agg["tov_total"] + agg["ast_total"]
    agg["ast_pct_proxy"] = np.where(poss_est > 0, agg["ast_total"] / poss_est, np.nan)
    agg["tov_pct_proxy"] = np.where(poss_est > 0, agg["tov_total"] / poss_est, np.nan)

    agg["oreb_pct_proxy"] = np.where(agg["reb_total"] > 0, agg["oreb_total"] / agg["reb_total"], np.nan)
    agg["dreb_pct_proxy"] = np.where(agg["reb_total"] > 0, agg["dreb_total"] / agg["reb_total"], np.nan)

    # Reliability filter similar to existing pipeline.
    agg = agg[agg["minutes_total"] >= 50].copy()

    return agg


def _add_position_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "usage_proxy",
        "ts_pct",
        "efg_pct",
        "ast_per40",
        "tov_per40",
        "stl_per40",
        "blk_per40",
        "reb_per40",
        "pts_per40",
    ]

    out = df.copy()

    group_cols = ["season", "athlete_position_abbreviation"]
    for m in metrics:
        pct_col = f"{m}_pctile_pos"
        out[pct_col] = np.nan
        for _, idx in out.groupby(group_cols).groups.items():
            grp = out.loc[idx]
            out.loc[idx, pct_col] = _weighted_percentile(grp[m], grp["minutes_total"])

    return out


def _add_yoy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["athlete_id", "season"])
    out["prev_season"] = out.groupby("athlete_id")["season"].shift(1)

    yoy_metrics = [
        "minutes_total",
        "mpg",
        "pts_per40",
        "reb_per40",
        "ast_per40",
        "stl_per40",
        "blk_per40",
        "tov_per40",
        "fouls_per40",
        "ts_pct",
        "efg_pct",
        "usage_proxy",
        "ast_to_tov",
        "pts_per40_pctile_pos",
        "ts_pct_pctile_pos",
        "usage_proxy_pctile_pos",
    ]
    yoy_metrics = [m for m in yoy_metrics if m in out.columns]

    for m in yoy_metrics:
        prev = out.groupby("athlete_id")[m].shift(1)
        out[f"{m}_prev"] = prev
        out[f"{m}_yoy_delta"] = out[m] - prev
        out[f"{m}_yoy_pct"] = np.where(prev.abs() > 1e-9, (out[m] - prev) / prev.abs(), np.nan)

    # Convenience composite.
    components = [
        c
        for c in [
            "pts_per40_yoy_delta",
            "ts_pct_yoy_delta",
            "usage_proxy_yoy_delta",
            "ast_per40_yoy_delta",
            "reb_per40_yoy_delta",
            "tov_per40_yoy_delta",
        ]
        if c in out.columns
    ]
    for c in components:
        out[f"z_{c}"] = out.groupby("season")[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) not in [0, np.nan] else 1.0))

    pos = [c for c in ["z_pts_per40_yoy_delta", "z_ts_pct_yoy_delta", "z_ast_per40_yoy_delta", "z_reb_per40_yoy_delta"] if c in out.columns]
    neg = [c for c in ["z_tov_per40_yoy_delta"] if c in out.columns]

    score = 0
    if pos:
        score += sum(out[c].fillna(0) for c in pos) / len(pos)
    if neg:
        score -= sum(out[c].fillna(0) for c in neg) / len(neg)
    out["yoy_growth_score"] = score

    return out


def main() -> None:
    raw = _load_sources()
    clean = _clean_filter(raw)

    agg = _aggregate_season(clean)
    agg = _add_position_percentiles(agg)
    yoy = _add_yoy(agg)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(AGG_OUT, index=False)
    yoy.to_csv(YOY_OUT, index=False)

    seasons = sorted(yoy["season"].dropna().astype(int).unique().tolist())
    print("\n=== Script 12 Complete ===")
    print(f"Seasons covered: {seasons}")
    print(f"Players (season rows): {len(yoy):,}")
    print(f"Saved: {AGG_OUT}")
    print(f"Saved: {YOY_OUT}")

    # Quick check for current configured season.
    cur = yoy[yoy["season"] == SEASON]
    print(f"Configured season ({SEASON}) rows: {len(cur):,}")


if __name__ == "__main__":
    main()
