"""
Script 2 – Play-by-Play Player Processing Pipeline
====================================================
Derives possession-level and event-level metrics from ESPN play-by-play
data that cannot be computed from box scores alone.

OPTIMIZED for memory: uses vectorized pandas operations throughout
(no Python-level row iteration on 2.6M+ rows).

Input:  play_by_play_2026_sdv_espn.parquet
Output: pbp_player_metrics.csv  (one row per player-season)

Metrics derived:
  ── Shot Profile ──
    • Shot zone distribution (paint / midrange / three / FT)
    • Shot zone efficiency (FG% and PPS by zone)
    • Shot distance profile (avg distance, distance std)

  ── Creation & Playmaking ──
    • Assisted FG rate (% of made FGs that were assisted)
    • Unassisted FG rate (self-created scoring)
    • Assist-to-creation ratio
    • Points created via assists (estimated)

  ── Game Phase Efficiency ──
    • 1st half vs 2nd half scoring splits
    • Clutch-window performance (Q4 last 5 min, close game)

  ── Transition vs Half-Court ──
    • Transition scoring rate (from text parsing)

  ── Secondary & Hustle ──
    • PBP-verified steals, blocks, offensive rebounds
    • Stocks generation rate

  ── Categorical Labels ──
    • Shot creator type (Rim Finisher / Mid-Range / Sniper / Balanced)
    • Playmaking style (Pass-First / Scoring Playmaker / Score-First / Off-Ball)
    • Half adjustment label (Closer / Fader / Steady)

Author:  Krystal B Creative — Sports Analytics Portfolio
Date:    2026-03-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PBP_FILE, PROCESSED_DIR

MIN_FGM_ASSISTED = 10    # Minimum made FGs for assisted rate reliability

CLUTCH_QUARTER = 4
CLUTCH_SECONDS_REMAINING = 300
CLUTCH_MARGIN = 5


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_pbp_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load play-by-play data from parquet.
    Only loads columns we actually need to minimize memory footprint.
    """
    fpath = path or PBP_FILE

    # Only read columns we need (saves ~60% memory)
    needed_cols = [
        "game_id", "season", "type_text", "text",
        "scoring_play", "score_value", "shooting_play",
        "team_id", "athlete_id_1", "athlete_id_2",
        "period_number", "end_quarter_seconds_remaining",
        "home_score", "away_score",
        "sequence_number", "game_play_number",
    ]

    print(f"Loading PBP data from {fpath.name} (selective columns)...")
    df = pd.read_parquet(fpath, columns=needed_cols)

    # Downcast numeric types to save memory
    for col in ["game_id", "season", "score_value", "period_number",
                "home_score", "away_score", "sequence_number", "game_play_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    print(f"  Rows: {len(df):,}  |  Games: {df['game_id'].nunique():,}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    return df


# =============================================================================
# 2. SHOT ZONE CLASSIFICATION  (fully vectorized)
# =============================================================================

def classify_and_tag_shots(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to shooting plays and classify each into a shot zone.
    All operations are vectorized — no row-level iteration.

    Returns a DataFrame of shooting plays with added columns:
      shot_zone, shot_distance, is_transition, is_assisted
    """
    print("Classifying shot zones (vectorized)...")

    shots = pbp_df[pbp_df["shooting_play"] == True].copy()
    print(f"  Shooting plays: {len(shots):,}")

    # --- Shot zone ---
    # Default: midrange for 2-point jumpers
    shots["shot_zone"] = "midrange"

    # Paint shots: LayUpShot, DunkShot, TipShot
    paint_mask = shots["type_text"].isin(["LayUpShot", "DunkShot", "TipShot"])
    shots.loc[paint_mask, "shot_zone"] = "paint"

    # Free throws
    ft_mask = shots["type_text"] == "MadeFreeThrow"
    shots.loc[ft_mask, "shot_zone"] = "free_throw"

    # Three-pointers (by score_value)
    three_mask = shots["score_value"] == 3
    shots.loc[three_mask, "shot_zone"] = "three"

    # --- Shot distance from text ---
    shots["shot_distance"] = (
        shots["text"]
        .str.extract(r"(\d+)-foot", expand=False)
        .astype(float)
    )

    # --- Transition flag ---
    text_lower = shots["text"].fillna("").str.lower()
    transition_pattern = "fast break|fastbreak|transition|coast to coast|outlet"
    shots["is_transition"] = text_lower.str.contains(transition_pattern, regex=True).astype(int)

    # --- Assisted flag (athlete_id_2 present on made FGs, excluding FTs) ---
    shots["is_assisted"] = (
        (shots["scoring_play"] == True) &
        (shots["athlete_id_2"].notna()) &
        (shots["shot_zone"] != "free_throw")
    ).astype(int)

    zone_dist = shots["shot_zone"].value_counts().to_dict()
    print(f"  Zones: {zone_dist}")
    return shots


# =============================================================================
# 3. SHOT PROFILE AGGREGATION  (vectorized groupby)
# =============================================================================

def aggregate_shot_profiles(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shot zone stats per player-season using vectorized groupby.

    For each zone: FGA, FGM, FG%, PPS, share of total shots.
    Plus: avg distance, assisted rate, transition share.
    """
    print("Aggregating shot profiles (vectorized)...")

    df = shots_df.copy()
    gk = ["athlete_id_1", "season"]

    # --- Total shots per player ---
    total = df.groupby(gk).agg(
        pbp_total_shots=("game_id", "size"),
        avg_shot_distance=("shot_distance", "mean"),
        shot_distance_std=("shot_distance", "std"),
        transition_shots=("is_transition", "sum"),
    ).reset_index()

    total["transition_shot_share"] = total["transition_shots"] / total["pbp_total_shots"].clip(lower=1)

    # --- Per-zone stats ---
    zones = ["paint", "midrange", "three", "free_throw"]
    for zone in zones:
        zone_df = df[df["shot_zone"] == zone]
        zone_agg = zone_df.groupby(gk).agg(
            fga=("game_id", "size"),
            fgm=("scoring_play", "sum"),
        ).reset_index()
        zone_agg.columns = gk + [f"{zone}_fga", f"{zone}_fgm"]

        zone_agg[f"{zone}_fg_pct"] = np.where(
            zone_agg[f"{zone}_fga"] > 0,
            zone_agg[f"{zone}_fgm"] / zone_agg[f"{zone}_fga"],
            np.nan
        )
        # Points per shot
        multiplier = {"paint": 2, "midrange": 2, "three": 3, "free_throw": 1}[zone]
        zone_agg[f"{zone}_pps"] = np.where(
            zone_agg[f"{zone}_fga"] > 0,
            multiplier * zone_agg[f"{zone}_fgm"] / zone_agg[f"{zone}_fga"],
            0
        )

        total = total.merge(zone_agg, on=gk, how="left")
        total[f"{zone}_fga"] = total[f"{zone}_fga"].fillna(0).astype(int)
        total[f"{zone}_fgm"] = total[f"{zone}_fgm"].fillna(0).astype(int)

    # Zone shares
    for zone in zones:
        total[f"{zone}_share"] = total[f"{zone}_fga"] / total["pbp_total_shots"].clip(lower=1)

    # --- Assisted / unassisted (FGs only, no FTs) ---
    fg_only = df[df["shot_zone"] != "free_throw"]
    fg_made = fg_only[fg_only["scoring_play"] == True]

    assist_agg = fg_made.groupby(gk).agg(
        fg_made_total=("game_id", "size"),
        assisted_fgm=("is_assisted", "sum"),
    ).reset_index()
    assist_agg["unassisted_fgm"] = assist_agg["fg_made_total"] - assist_agg["assisted_fgm"]
    assist_agg["assisted_fg_rate"] = np.where(
        assist_agg["fg_made_total"] >= MIN_FGM_ASSISTED,
        assist_agg["assisted_fgm"] / assist_agg["fg_made_total"],
        np.nan
    )
    assist_agg["unassisted_fg_rate"] = np.where(
        assist_agg["fg_made_total"] >= MIN_FGM_ASSISTED,
        assist_agg["unassisted_fgm"] / assist_agg["fg_made_total"],
        np.nan
    )

    total = total.merge(assist_agg, on=gk, how="left")

    # Rename key column
    total.rename(columns={"athlete_id_1": "athlete_id"}, inplace=True)

    print(f"  Shot profiles: {len(total):,} player-seasons")
    return total


# =============================================================================
# 4. ASSIST / CREATION METRICS  (vectorized)
# =============================================================================

def aggregate_assist_metrics(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute assist and creation metrics.
    athlete_id_2 = the assister on made field goals.
    """
    print("Computing assist/creation metrics (vectorized)...")

    # Assisted scoring plays (no FTs)
    assists = pbp_df[
        (pbp_df["scoring_play"] == True) &
        (pbp_df["athlete_id_2"].notna()) &
        (pbp_df["type_text"] != "MadeFreeThrow")
    ]

    assist_agg = assists.groupby(["athlete_id_2", "season"]).agg(
        pbp_assists=("game_id", "size"),
        points_created_via_ast=("score_value", "sum"),
    ).reset_index()
    assist_agg.rename(columns={"athlete_id_2": "athlete_id"}, inplace=True)

    # Own FGM for ratio
    own_fg = pbp_df[
        (pbp_df["scoring_play"] == True) &
        (pbp_df["type_text"] != "MadeFreeThrow") &
        (pbp_df["athlete_id_1"].notna())
    ]
    own_fgm = own_fg.groupby(["athlete_id_1", "season"]).size().reset_index(name="own_fgm")
    own_fgm.rename(columns={"athlete_id_1": "athlete_id"}, inplace=True)

    result = assist_agg.merge(own_fgm, on=["athlete_id", "season"], how="outer").fillna(0)
    result["ast_to_fg_ratio"] = np.where(
        result["own_fgm"] > 0,
        result["pbp_assists"] / result["own_fgm"], 0
    )

    print(f"  Assist metrics: {len(result):,} player-seasons")
    return result


# =============================================================================
# 5. GAME PHASE EFFICIENCY  (vectorized)
# =============================================================================

def aggregate_game_phase_metrics(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scoring efficiency by game phase using vectorized operations.
    Phases: first_half, second_half, clutch, overtime.
    """
    print("Computing game phase efficiency (vectorized)...")

    shots = pbp_df[
        (pbp_df["shooting_play"] == True) &
        (pbp_df["athlete_id_1"].notna())
    ].copy()

    # Classify phase
    margin = (shots["home_score"] - shots["away_score"]).abs()

    shots["game_phase"] = "second_half"  # default for periods 3-4
    shots.loc[shots["period_number"] <= 2, "game_phase"] = "first_half"
    shots.loc[shots["period_number"] >= 5, "game_phase"] = "overtime"

    # Clutch: Q4, <=5 min remaining, margin <=5
    clutch_mask = (
        (shots["period_number"] == CLUTCH_QUARTER) &
        (shots["end_quarter_seconds_remaining"].fillna(9999) <= CLUTCH_SECONDS_REMAINING) &
        (margin <= CLUTCH_MARGIN)
    )
    shots.loc[clutch_mask, "game_phase"] = "clutch"

    # Flag three-pointers made for eFG calc
    shots["is_three_made"] = (
        (shots["scoring_play"] == True) & (shots["score_value"] == 3)
    ).astype(int)

    gk = ["athlete_id_1", "season"]
    phases = ["first_half", "second_half", "clutch", "overtime"]

    # Start with player-season skeleton
    all_players = shots.groupby(gk).size().reset_index(name="_count")[gk]

    for phase in phases:
        phase_shots = shots[shots["game_phase"] == phase].copy()

        # Pre-compute points (score_value only for scoring plays, else 0)
        phase_shots["pts_scored"] = np.where(
            phase_shots["scoring_play"] == True, phase_shots["score_value"], 0
        )

        pa = phase_shots.groupby(gk).agg(
            fga=("game_id", "size"),
            fgm=("scoring_play", "sum"),
            pts=("pts_scored", "sum"),
            three_m=("is_three_made", "sum"),
            games=("game_id", "nunique"),
        ).reset_index()

        pa[f"{phase}_fg_pct"] = np.where(pa["fga"] >= 5, pa["fgm"] / pa["fga"], np.nan)
        pa[f"{phase}_efg"] = np.where(
            pa["fga"] >= 5,
            (pa["fgm"] + 0.5 * pa["three_m"]) / pa["fga"],
            np.nan
        )

        pa = pa.rename(columns={
            "fga": f"{phase}_fga",
            "fgm": f"{phase}_fgm",
            "pts": f"{phase}_pts",
            "games": f"{phase}_games",
        })
        pa.drop(columns=["three_m"], inplace=True)

        all_players = all_players.merge(pa, on=gk, how="left")

    # Clutch-specific aggregations
    all_players["clutch_shot_attempts"] = all_players.get("clutch_fga", 0).fillna(0)
    all_players["clutch_games"] = all_players.get("clutch_games", 0).fillna(0)

    all_players.rename(columns={"athlete_id_1": "athlete_id"}, inplace=True)

    print(f"  Phase metrics: {len(all_players):,} player-seasons")
    return all_players


# =============================================================================
# 6. HUSTLE / EVENT COUNT METRICS  (vectorized)
# =============================================================================

def aggregate_hustle_metrics(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PBP-verified event counts per player using vectorized pivot.
    """
    print("Computing hustle/event metrics (vectorized)...")

    events = pbp_df[pbp_df["athlete_id_1"].notna()].copy()

    # Pivot: count each type_text per player-season
    target_types = ["Steal", "Block Shot", "Offensive Rebound", "Defensive Rebound",
                    "Lost Ball Turnover"]
    filtered = events[events["type_text"].isin(target_types)]

    counts = filtered.groupby(["athlete_id_1", "season", "type_text"]).size().unstack(fill_value=0)

    result = counts.reset_index()
    result.rename(columns={
        "athlete_id_1": "athlete_id",
        "Steal": "pbp_steals",
        "Block Shot": "pbp_blocks",
        "Offensive Rebound": "pbp_oreb",
        "Defensive Rebound": "pbp_dreb",
        "Lost Ball Turnover": "pbp_turnovers",
    }, inplace=True)

    # Ensure all columns exist
    for col in ["pbp_steals", "pbp_blocks", "pbp_oreb", "pbp_dreb", "pbp_turnovers"]:
        if col not in result.columns:
            result[col] = 0

    print(f"  Hustle metrics: {len(result):,} player-seasons")
    return result


# =============================================================================
# 7. MERGE & LABELS
# =============================================================================

def merge_and_label(shot_profiles: pd.DataFrame,
                    assist_metrics: pd.DataFrame,
                    phase_metrics: pd.DataFrame,
                    hustle_metrics: pd.DataFrame) -> pd.DataFrame:
    """Merge all PBP metric tables and apply categorical labels."""
    print("Merging all PBP metrics...")

    result = shot_profiles.copy()
    for other in [assist_metrics, phase_metrics, hustle_metrics]:
        result = result.merge(other, on=["athlete_id", "season"], how="outer")

    # Fill count NaNs with 0
    fill_cols = [c for c in result.columns if any(
        kw in c for kw in ["_fga", "_fgm", "_pts", "total", "assists", "steals",
                           "blocks", "oreb", "dreb", "turnovers", "transition",
                           "clutch_shot", "clutch_games"]
    )]
    for col in fill_cols:
        if col in result.columns and result[col].dtype in [np.float64, np.int64, float, int]:
            result[col] = result[col].fillna(0)

    print(f"  Merged: {len(result):,} rows, {len(result.columns)} columns")

    # --- Categorical Labels ---
    print("Applying PBP-based labels...")

    # Shot Creator Type (by dominant zone)
    conditions = [
        result.get("paint_share", 0) >= 0.45,
        result.get("three_share", 0) >= 0.45,
        (result.get("midrange_share", 0) >= 0.30) & (result.get("three_share", 0) < 0.35),
    ]
    choices = ["Rim Finisher", "Sniper", "Mid-Range"]
    result["shot_creator_type"] = np.select(conditions, choices, default="Balanced")

    # Playmaking Style
    ast_fg = result.get("ast_to_fg_ratio", 0).fillna(0)
    unast = result.get("unassisted_fg_rate", 0).fillna(0)
    conditions = [
        ast_fg >= 1.0,
        (ast_fg >= 0.5) & (unast >= 0.40),
        unast >= 0.50,
    ]
    choices = ["Pass-First", "Scoring Playmaker", "Score-First"]
    result["playmaking_style"] = np.select(conditions, choices, default="Off-Ball")

    # Half Adjustment Label
    fh_pct = result.get("first_half_fg_pct", np.nan)
    sh_pct = result.get("second_half_fg_pct", np.nan)
    result["second_half_boost"] = np.where(
        pd.notna(fh_pct) & pd.notna(sh_pct), sh_pct - fh_pct, np.nan
    )
    conditions = [
        result["second_half_boost"] > 0.05,
        result["second_half_boost"] < -0.05,
    ]
    choices = ["Closer", "Fader"]
    result["half_adjustment_label"] = np.select(conditions, choices, default="Steady")

    print(f"  Labels applied: shot_creator_type, playmaking_style, half_adjustment_label")
    return result


# =============================================================================
# 8. EXPORT
# =============================================================================

def export_pbp_metrics(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Export final PBP player metrics."""
    out = output_dir or PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / "pbp_player_metrics.csv"
    df.to_csv(fpath, index=False)
    print(f"  ✓ Exported: {fpath.name}  ({len(df):,} rows, {len(df.columns)} cols)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """Execute the full PBP processing pipeline."""
    print("=" * 70)
    print("  PLAY-BY-PLAY PLAYER PROCESSING PIPELINE")
    print("  WBB Player Role Profiles + Breakout Detection")
    print("=" * 70)
    print()

    # 1. Load
    pbp_df = load_pbp_data()
    print()

    # 2. Classify shots
    shots_df = classify_and_tag_shots(pbp_df)
    print()

    # 3. Shot profiles
    shot_profiles = aggregate_shot_profiles(shots_df)
    print()

    # 4. Assist metrics
    assist_metrics = aggregate_assist_metrics(pbp_df)
    print()

    # 5. Game phase metrics
    phase_metrics = aggregate_game_phase_metrics(pbp_df)
    print()

    # 6. Hustle metrics
    hustle_metrics = aggregate_hustle_metrics(pbp_df)
    print()

    # 7. Merge & label
    result = merge_and_label(shot_profiles, assist_metrics, phase_metrics, hustle_metrics)
    print()

    # 8. Export
    export_pbp_metrics(result)

    print()
    print("=" * 70)
    print("  PBP PIPELINE COMPLETE")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_pipeline()
