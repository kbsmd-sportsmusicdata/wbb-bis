"""
Script 1 – Player Box Processing Pipeline
==========================================
Transforms raw game-level player box scores into a season-level
analytics-ready dataset with advanced metrics for role classification
and breakout detection.

Input:  player_box_2026_sdv_espn.parquet
        schedule_filtered_20260302.csv  (game context + opponent ranks)
        net_rankings_20260302.csv       (team NET rankings)
        d1_player_benchmarks_2025.csv   (prior-season percentile benchmarks)

Output: player_box_advanced_metrics.csv  (one row per player-season)
        player_game_log_enriched.csv     (game-level with rolling features)

Pipeline stages:
    1. Load & clean raw data
    2. Join game context (opponent NET, home/away, rest days)
    3. Compute game-level derived metrics (Game Score, individual PPP, etc.)
    4. Compute rolling / trend features (L5, L10, L15, slope, z-score)
    5. Aggregate to season level
    6. Compute advanced rate & efficiency metrics
    7. Add context / benchmark metrics
    8. Add categorical / labeling metrics
    9. Compute position-weighted percentiles
   10. Compute breakout signals against prior-season benchmarks
   11. Export

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

from config import (
    PLAYER_BOX_FILE,
    NET_FILE,
    SCHEDULE_FILE,
    BENCHMARKS_FILE,
    PROCESSED_DIR,
)

MIN_MINUTES_SEASON = 50       # Minimum season minutes for inclusion
MIN_MINUTES_GAME = 5          # Minimum game minutes to count in rolling stats
MIN_GAMES_PERCENTILE = 10     # Minimum games for percentile ranking
ROLLING_WINDOWS = [5, 10, 15] # L5, L10, L15 rolling windows

# Minutes-per-game tier thresholds
MINUTES_TIERS = {
    'Star': 28,
    'Rotation': 20,
    'Bench': 10,
    'Deep Bench': 0
}

# Usage-Efficiency quadrant thresholds (will be calibrated from data)
USG_THRESHOLD_HIGH = 0.22  # ~top 40% usage
TS_THRESHOLD_HIGH = 0.52   # ~top 40% TS%


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_player_box(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and clean player box score data from parquet.

    Returns a DataFrame with one row per player-game, filtered to
    players who actually played (did_not_play == False, minutes > 0).
    """
    fpath = path or PLAYER_BOX_FILE
    print(f"Loading player box scores from {fpath.name}...")
    df = pd.read_parquet(fpath)
    print(f"  Raw rows: {len(df):,}")

    # Filter to players who actually played
    df = df[
        (df["did_not_play"] == False) &
        (df["minutes"].notna()) &
        (df["minutes"] > 0)
    ].copy()
    print(f"  After filtering DNP/0-min: {len(df):,}")

    # Ensure numeric types
    numeric_cols = [
        "minutes", "field_goals_made", "field_goals_attempted",
        "three_point_field_goals_made", "three_point_field_goals_attempted",
        "free_throws_made", "free_throws_attempted",
        "offensive_rebounds", "defensive_rebounds", "rebounds",
        "assists", "steals", "blocks", "turnovers", "fouls", "points"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Parse game date
    df["game_date"] = pd.to_datetime(df["game_date"])

    return df


def load_net_rankings(path: Optional[Path] = None) -> pd.DataFrame:
    """Load NET rankings and create team-to-NET lookup."""
    fpath = path or NET_FILE
    print(f"Loading NET rankings from {fpath.name}...")
    net = pd.read_csv(fpath)
    print(f"  Teams loaded: {len(net)}")
    return net


def load_schedule(path: Optional[Path] = None) -> pd.DataFrame:
    """Load filtered schedule for game context."""
    fpath = path or SCHEDULE_FILE
    print(f"Loading schedule from {fpath.name}...")
    sched = pd.read_csv(fpath)
    sched["game_date"] = pd.to_datetime(sched["date"]).dt.date
    print(f"  Schedule rows: {len(sched):,}")
    return sched


def load_benchmarks(path: Optional[Path] = None) -> pd.DataFrame:
    """Load prior-season D1 benchmarks for breakout comparison."""
    fpath = path or BENCHMARKS_FILE
    if not fpath.exists():
        print(f"  ⚠️  Benchmarks file not found ({fpath.name}) — breakout signals will use intra-season percentiles.")
        return pd.DataFrame()
    print(f"Loading 2024-25 benchmarks from {fpath.name}...")
    bench = pd.read_csv(fpath)
    print(f"  Benchmark rows: {len(bench)}")
    return bench


# =============================================================================
# 2. GAME CONTEXT ENRICHMENT
# =============================================================================

def enrich_game_context(box_df: pd.DataFrame,
                        net_df: pd.DataFrame,
                        sched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game-level context columns:
      - opponent_net_rank:  opponent's NET ranking
      - team_net_rank:      player's team NET ranking
      - rest_days:          days since player's last game
      - is_back_to_back:    True if game is on consecutive day
      - location:           Home / Away / Neutral
      - game_result:        W or L for the player's team
      - point_diff:         team score minus opponent score
      - is_close_game:      True if final margin <= 5 (clutch proxy)
    """
    print("Enriching game context...")
    df = box_df.copy()

    # --- Opponent NET rank ---
    # Build team abbreviation -> NET rank mapping
    net_lookup = net_df.set_index("team")["net_rank"].to_dict()

    # Map opponent display name to NET rank
    # We'll try mapping by opponent_team_location which matches NET 'team' col
    df["opponent_net_rank"] = (
        df["opponent_team_location"]
        .map(net_lookup)
        .fillna(df["opponent_team_display_name"].map(net_lookup))
        .fillna(200)  # Unranked default
    )

    # Map player's own team NET rank
    df["team_net_rank"] = (
        df["team_location"]
        .map(net_lookup)
        .fillna(df["team_display_name"].map(net_lookup))
        .fillna(200)
    )

    # --- Rest days (days since player's last game) ---
    df = df.sort_values(["athlete_id", "game_date"])
    df["prev_game_date"] = df.groupby("athlete_id")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days
    df["rest_days"] = df["rest_days"].fillna(7)  # First game of season default
    df["is_back_to_back"] = (df["rest_days"] <= 1).astype(int)

    # --- Location ---
    df["location"] = df["home_away"].map({"home": "Home", "away": "Away"}).fillna("Neutral")

    # --- Game result & margin ---
    df["game_result"] = df["team_winner"].map({True: "W", False: "L"})
    df["point_diff"] = df["team_score"] - df["opponent_team_score"]
    df["is_close_game"] = (df["point_diff"].abs() <= 5).astype(int)

    # --- Opponent NET weight for adjusted stats ---
    # Higher weight for tougher opponents (inverse rank, normalized)
    max_rank = df["opponent_net_rank"].max()
    df["opp_net_weight"] = 1 + (max_rank - df["opponent_net_rank"]) / max_rank

    print(f"  Context columns added. Shape: {df.shape}")
    return df


# =============================================================================
# 3. GAME-LEVEL DERIVED METRICS
# =============================================================================

def compute_game_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-game derived metrics before aggregation.

    New metrics:
      - game_score:          Hollinger Game Score
      - individual_ppp:      Points per individual possession
      - efg_pct_game:        eFG% for this game
      - ts_pct_game:         TS% for this game
      - stocks:              Steals + Blocks
      - is_double_double:    Boolean flag
    """
    print("Computing game-level derived metrics...")

    # --- Hollinger Game Score ---
    # GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB
    #        + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
    df["game_score"] = (
        df["points"]
        + 0.4 * df["field_goals_made"]
        - 0.7 * df["field_goals_attempted"]
        - 0.4 * (df["free_throws_attempted"] - df["free_throws_made"])
        + 0.7 * df["offensive_rebounds"]
        + 0.3 * df["defensive_rebounds"]
        + df["steals"]
        + 0.7 * df["assists"]
        + 0.7 * df["blocks"]
        - 0.4 * df["fouls"]
        - df["turnovers"]
    )

    # --- Individual Points Per Possession ---
    # PPP = PTS / (FGA + 0.44*FTA + TOV)
    indiv_poss = df["field_goals_attempted"] + 0.44 * df["free_throws_attempted"] + df["turnovers"]
    df["individual_ppp"] = np.where(indiv_poss > 0, df["points"] / indiv_poss, 0)

    # --- Game-level eFG% ---
    df["efg_pct_game"] = np.where(
        df["field_goals_attempted"] > 0,
        (df["field_goals_made"] + 0.5 * df["three_point_field_goals_made"]) / df["field_goals_attempted"],
        0
    )

    # --- Game-level TS% ---
    tsa = df["field_goals_attempted"] + 0.44 * df["free_throws_attempted"]
    df["ts_pct_game"] = np.where(tsa > 0, df["points"] / (2 * tsa), 0)

    # --- Stocks (STL + BLK) ---
    df["stocks"] = df["steals"] + df["blocks"]

    # --- Double-Double detection ---
    # Check pairs among points, rebounds, assists, steals, blocks
    stat_cols = ["points", "rebounds", "assists", "steals", "blocks"]
    df["double_double_count"] = (df[stat_cols] >= 10).sum(axis=1)
    df["is_double_double"] = (df["double_double_count"] >= 2).astype(int)

    # --- Opponent-adjusted Game Score ---
    # Weight game score by opponent quality
    df["adj_game_score"] = df["game_score"] * df["opp_net_weight"]

    # --- Per-40 single-game metrics ---
    minutes_safe = df["minutes"].clip(lower=1)
    df["pts_per40_game"] = df["points"] / minutes_safe * 40
    df["reb_per40_game"] = df["rebounds"] / minutes_safe * 40
    df["ast_per40_game"] = df["assists"] / minutes_safe * 40

    print(f"  Game-level metrics computed. New columns: game_score, individual_ppp, adj_game_score, etc.")
    return df


# =============================================================================
# 4. ROLLING / TREND FEATURES (BREAKOUT DETECTION ENGINE)
# =============================================================================

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages and trend features for breakout detection.

    For each rolling window (L5, L10, L15):
      - Rolling mean of Game Score, TS%, usage proxy
      - Improvement z-score:  (rolling_mean - season_mean) / season_std
      - Trend slope:          linear regression slope of Game Score over window

    These features detect "is this player surging right now?"
    Think of it like a stock chart — the rolling averages are moving
    averages and the z-score is like a Bollinger Band signal.
    """
    print("Computing rolling / trend features...")

    # Sort by player and date
    df = df.sort_values(["athlete_id", "game_date"]).copy()

    # Only compute rolling stats for games with meaningful minutes
    mask = df["minutes"] >= MIN_MINUTES_GAME

    for window in ROLLING_WINDOWS:
        suffix = f"_L{window}"

        # Rolling mean of Game Score
        df[f"game_score{suffix}"] = (
            df.groupby("athlete_id")["game_score"]
            .transform(lambda x: x.rolling(window, min_periods=max(3, window // 2)).mean())
        )

        # Rolling mean of TS%
        df[f"ts_pct{suffix}"] = (
            df.groupby("athlete_id")["ts_pct_game"]
            .transform(lambda x: x.rolling(window, min_periods=max(3, window // 2)).mean())
        )

        # Rolling mean of individual PPP
        df[f"individual_ppp{suffix}"] = (
            df.groupby("athlete_id")["individual_ppp"]
            .transform(lambda x: x.rolling(window, min_periods=max(3, window // 2)).mean())
        )

    # --- Season-level mean/std for z-score computation ---
    season_stats = df[mask].groupby("athlete_id")["game_score"].agg(["mean", "std"]).reset_index()
    season_stats.columns = ["athlete_id", "gs_season_mean", "gs_season_std"]
    season_stats["gs_season_std"] = season_stats["gs_season_std"].replace(0, np.nan)
    df = df.merge(season_stats, on="athlete_id", how="left")

    # --- Improvement Z-Score (L10 vs season) ---
    # Positive z-score = player is performing above their own season baseline
    df["improvement_zscore_L10"] = np.where(
        df["gs_season_std"].notna() & (df["gs_season_std"] > 0),
        (df["game_score_L10"] - df["gs_season_mean"]) / df["gs_season_std"],
        0
    )

    # --- Trend Slope (linear regression of Game Score over last 10 games) ---
    def rolling_slope(series, window=10):
        """Compute slope of linear fit over rolling window."""
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(window)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            if len(y) == window and not np.any(np.isnan(y)):
                # Least squares slope: sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)
                x_centered = x - x.mean()
                slope = np.dot(x_centered, y - y.mean()) / np.dot(x_centered, x_centered)
                result.iloc[i] = slope
        return result

    df["game_score_slope_L10"] = (
        df.groupby("athlete_id")["game_score"]
        .transform(lambda x: rolling_slope(x, window=10))
    )

    # --- Consistency Score (CV of Game Score over last 10) ---
    df["game_score_std_L10"] = (
        df.groupby("athlete_id")["game_score"]
        .transform(lambda x: x.rolling(10, min_periods=5).std())
    )
    df["consistency_score_L10"] = np.where(
        df["game_score_L10"].abs() > 0.5,
        df["game_score_std_L10"] / df["game_score_L10"].abs(),
        np.nan
    )
    # Lower consistency score = more consistent performer
    # Invert so higher = better: consistency_rating = 1 / (1 + CV)
    df["consistency_rating_L10"] = 1 / (1 + df["consistency_score_L10"].fillna(1))

    # Clean up temp columns
    df.drop(columns=["gs_season_mean", "gs_season_std"], inplace=True, errors="ignore")

    print(f"  Rolling features computed for windows: {ROLLING_WINDOWS}")
    return df


# =============================================================================
# 5. SEASON AGGREGATION
# =============================================================================

def aggregate_to_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate game-level data to one row per player-season.

    Includes:
      - Counting stat totals and per-game / per-40 rates
      - Shooting totals for efficiency calculations
      - Context summaries (avg opponent NET, home/away splits, etc.)
      - Rolling feature snapshots (most recent L10/L5 values)
      - Game Score season stats (mean, std, max)
    """
    print("Aggregating to season level...")

    # Most recent game per player (for rolling feature snapshot)
    most_recent = df.sort_values("game_date").groupby("athlete_id").tail(1)
    rolling_snapshot = most_recent[[
        "athlete_id",
        "game_score_L5", "game_score_L10", "game_score_L15",
        "ts_pct_L5", "ts_pct_L10", "ts_pct_L15",
        "individual_ppp_L5", "individual_ppp_L10", "individual_ppp_L15",
        "improvement_zscore_L10",
        "game_score_slope_L10",
        "consistency_rating_L10",
    ]].set_index("athlete_id")

    # --- Main aggregation ---
    agg = df.groupby(["athlete_id", "athlete_display_name", "team_id",
                       "team_location", "team_short_display_name",
                       "team_abbreviation",
                       "athlete_position_name", "athlete_position_abbreviation",
                       "season"]).agg(

        # Playing time
        games_played=("game_id", "nunique"),
        games_started=("starter", "sum"),
        minutes_total=("minutes", "sum"),

        # Counting stats
        pts_total=("points", "sum"),
        reb_total=("rebounds", "sum"),
        oreb_total=("offensive_rebounds", "sum"),
        dreb_total=("defensive_rebounds", "sum"),
        ast_total=("assists", "sum"),
        stl_total=("steals", "sum"),
        blk_total=("blocks", "sum"),
        tov_total=("turnovers", "sum"),
        fouls_total=("fouls", "sum"),

        # Shooting totals
        fgm=("field_goals_made", "sum"),
        fga=("field_goals_attempted", "sum"),
        fg3m=("three_point_field_goals_made", "sum"),
        fg3a=("three_point_field_goals_attempted", "sum"),
        ftm=("free_throws_made", "sum"),
        fta=("free_throws_attempted", "sum"),

        # Context aggregations
        avg_opponent_net=("opponent_net_rank", "mean"),
        games_vs_top25=("opponent_net_rank", lambda x: (x <= 25).sum()),
        games_vs_top50=("opponent_net_rank", lambda x: (x <= 50).sum()),
        home_games=("location", lambda x: (x == "Home").sum()),
        away_games=("location", lambda x: (x == "Away").sum()),
        back_to_back_games=("is_back_to_back", "sum"),
        avg_rest_days=("rest_days", "mean"),
        close_games=("is_close_game", "sum"),
        wins=("game_result", lambda x: (x == "W").sum()),

        # Game Score season distribution
        game_score_mean=("game_score", "mean"),
        game_score_std=("game_score", "std"),
        game_score_max=("game_score", "max"),
        game_score_min=("game_score", "min"),

        # Adj Game Score
        adj_game_score_mean=("adj_game_score", "mean"),

        # Double-doubles
        double_doubles=("is_double_double", "sum"),

        # Stocks per game raw
        stocks_total=("stocks", "sum"),

        # Win/Loss context
        pts_in_wins=("points", lambda x: x[df.loc[x.index, "game_result"] == "W"].sum()),
        pts_in_losses=("points", lambda x: x[df.loc[x.index, "game_result"] == "L"].sum()),
        games_won=("game_result", lambda x: (x == "W").sum()),
        games_lost=("game_result", lambda x: (x == "L").sum()),

        # Opponent-adjusted stats
        adj_pts_total=("points", lambda x: (x * df.loc[x.index, "opp_net_weight"]).sum()),
        adj_reb_total=("rebounds", lambda x: (x * df.loc[x.index, "opp_net_weight"]).sum()),
        adj_ast_total=("assists", lambda x: (x * df.loc[x.index, "opp_net_weight"]).sum()),

        # Team NET rank (take mode/first)
        team_net_rank=("team_net_rank", "first"),

    ).reset_index()

    # --- Join rolling snapshot ---
    agg = agg.merge(rolling_snapshot, left_on="athlete_id", right_index=True, how="left")

    print(f"  Season-level rows: {len(agg):,}")
    return agg


# =============================================================================
# 6. ADVANCED RATE & EFFICIENCY METRICS
# =============================================================================

def compute_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all rate, efficiency, and advanced metrics from season totals.

    Includes everything from the original R script PLUS new additions.
    Avoids duplicating what's already in advanced_metrics.py (PBP-derived).
    """
    print("Computing advanced rate & efficiency metrics...")

    # --- Safe denominators ---
    min_total = df["minutes_total"].clip(lower=1)
    fga_safe = df["fga"].clip(lower=1)
    gp_safe = df["games_played"].clip(lower=1)

    # ===================== Per-Game Rates =====================
    df["mpg"] = df["minutes_total"] / gp_safe
    df["ppg"] = df["pts_total"] / gp_safe
    df["rpg"] = df["reb_total"] / gp_safe
    df["apg"] = df["ast_total"] / gp_safe
    df["spg"] = df["stl_total"] / gp_safe
    df["bpg"] = df["blk_total"] / gp_safe
    df["tovpg"] = df["tov_total"] / gp_safe

    # ===================== Per-40 Rates =====================
    df["pts_per40"] = df["pts_total"] / min_total * 40
    df["reb_per40"] = df["reb_total"] / min_total * 40
    df["oreb_per40"] = df["oreb_total"] / min_total * 40
    df["dreb_per40"] = df["dreb_total"] / min_total * 40
    df["ast_per40"] = df["ast_total"] / min_total * 40
    df["stl_per40"] = df["stl_total"] / min_total * 40
    df["blk_per40"] = df["blk_total"] / min_total * 40
    df["tov_per40"] = df["tov_total"] / min_total * 40
    df["stocks_per40"] = df["stocks_total"] / min_total * 40

    # ===================== Shooting Percentages =====================
    df["fg_pct"] = np.where(df["fga"] > 0, df["fgm"] / df["fga"], 0)
    df["fg3_pct"] = np.where(df["fg3a"] > 0, df["fg3m"] / df["fg3a"], 0)
    df["ft_pct"] = np.where(df["fta"] > 0, df["ftm"] / df["fta"], 0)

    # ===================== Advanced Shooting =====================
    # eFG%
    df["efg_pct"] = np.where(df["fga"] > 0,
        (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"], 0)

    # TS%
    tsa = df["fga"] + 0.44 * df["fta"]
    df["ts_pct"] = np.where(tsa > 0, df["pts_total"] / (2 * tsa), 0)

    # Three-Point Attempt Rate
    df["three_par"] = np.where(df["fga"] > 0, df["fg3a"] / df["fga"], 0)

    # Free Throw Rate
    df["fta_rate"] = np.where(df["fga"] > 0, df["fta"] / df["fga"], 0)

    # ===================== Playmaking / Turnover =====================
    # Assist-to-Turnover Ratio
    df["ast_to_tov"] = np.where(df["tov_total"] > 0,
        df["ast_total"] / df["tov_total"], df["ast_total"])

    # Usage proxy (player-level, without team totals)
    usage_load = df["fga"] + 0.44 * df["fta"] + df["tov_total"]
    df["usage_proxy"] = usage_load / min_total

    # Assist percentage approximation
    poss_est = df["fga"] + 0.44 * df["fta"] + df["tov_total"] + df["ast_total"]
    df["ast_pct"] = np.where(poss_est > 0, df["ast_total"] / poss_est, 0)

    # Turnover percentage approximation
    df["tov_pct"] = np.where(poss_est > 0, df["tov_total"] / poss_est, 0)

    # ===================== Rebounding Shares =====================
    reb_safe = df["reb_total"].clip(lower=1)
    df["oreb_share"] = df["oreb_total"] / reb_safe
    df["dreb_share"] = df["dreb_total"] / reb_safe

    # ===================== NEW: Points Per Individual Possession =====================
    # More precise than PPG for comparing scoring efficiency across usage levels
    indiv_poss = df["fga"] + 0.44 * df["fta"] + df["tov_total"]
    df["pts_per_indiv_poss"] = np.where(indiv_poss > 0,
        df["pts_total"] / indiv_poss, 0)

    # ===================== NEW: Assist-to-Usage Ratio =====================
    # Measures playmaking output relative to possessions consumed
    # High = efficient playmaker, Low = ball-dominant scorer
    df["ast_to_usage"] = np.where(df["usage_proxy"] > 0,
        (df["ast_total"] / min_total) / df["usage_proxy"], 0)

    # ===================== NEW: Approximate Box Plus/Minus =====================
    # Simplified BPM using position-adjusted coefficients
    # Inspired by basketball-reference BPM methodology
    # Coefficients tuned for women's college basketball
    df["approx_bpm"] = (
        0.123 * df["pts_per40"]
        + 0.119 * df["reb_per40"]
        + 0.253 * df["ast_per40"]
        + 0.466 * df["stl_per40"]
        + 0.466 * df["blk_per40"]
        - 0.197 * df["tov_per40"]
        - 0.099 * (df["fga"] / min_total * 40)  # penalize volume
        + 0.350 * df["efg_pct"]
        - 5.0  # Centering constant (league average ~ 0)
    )

    # ===================== NEW: Versatility Index =====================
    # Measures breadth of contribution across categories
    # Scoring + Rebounding + Assisting + Defending (stocks)
    # Each component normalized to per-40 scale, then combined
    # A high versatility index = contributes across many areas
    df["versatility_index"] = (
        df["pts_per40"].clip(upper=30) / 30 * 25     # Scoring (max ~30 per40)
        + df["reb_per40"].clip(upper=15) / 15 * 25   # Rebounding (max ~15 per40)
        + df["ast_per40"].clip(upper=10) / 10 * 25   # Assisting (max ~10 per40)
        + df["stocks_per40"].clip(upper=6) / 6 * 25  # Defending (max ~6 per40)
    )

    # ===================== NEW: Minutes Share =====================
    # What fraction of team minutes does this player command?
    # Proxy for team importance / role weight
    # Team plays 200 min/game (5 players x 40 min), so per-game share:
    df["minutes_share"] = df["mpg"] / 40  # 1.0 = plays every minute

    # ===================== NEW: Win/Loss Efficiency Splits =====================
    df["ppg_in_wins"] = np.where(df["games_won"] > 0,
        df["pts_in_wins"] / df["games_won"], 0)
    df["ppg_in_losses"] = np.where(df["games_lost"] > 0,
        df["pts_in_losses"] / df["games_lost"], 0)
    df["win_loss_ppg_diff"] = df["ppg_in_wins"] - df["ppg_in_losses"]

    # ===================== NEW: Opponent-Adjusted Per-Game Rates =====================
    # These give more credit for performing against tough opponents
    df["adj_ppg"] = df["adj_pts_total"] / gp_safe
    df["adj_rpg"] = df["adj_reb_total"] / gp_safe
    df["adj_apg"] = df["adj_ast_total"] / gp_safe

    # ===================== NEW: Double-Double Rate =====================
    df["double_double_rate"] = df["double_doubles"] / gp_safe

    # ===================== NEW: Starter Consistency Rate =====================
    df["starter_rate"] = df["games_started"] / gp_safe

    # ===================== NEW: Win Rate =====================
    df["win_rate"] = df["wins"] / gp_safe

    print(f"  Advanced metrics computed. Total columns: {len(df.columns)}")
    return df


# =============================================================================
# 7. CATEGORICAL / LABELING METRICS
# =============================================================================

def apply_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorical labels that directly support role classification.

    Labels:
      - minutes_tier:         Star / Rotation / Bench / Deep Bench
      - usage_efficiency_quad: Elite Creator / Volume Scorer / Efficient Role / Low Impact
      - three_level_scorer:    True if produces above threshold from all three zones
      - scoring_profile:       Primary Scorer / Secondary Scorer / Floor Spacer /
                               Non-Scorer / Balanced
    """
    print("Applying categorical labels...")

    # --- Minutes Tier ---
    conditions = [
        df["mpg"] >= MINUTES_TIERS["Star"],
        df["mpg"] >= MINUTES_TIERS["Rotation"],
        df["mpg"] >= MINUTES_TIERS["Bench"],
    ]
    choices = ["Star", "Rotation", "Bench"]
    df["minutes_tier"] = np.select(conditions, choices, default="Deep Bench")

    # --- Usage-Efficiency Quadrant ---
    # Calibrate thresholds from data medians for players with enough minutes
    qualified = df[df["minutes_total"] >= MIN_MINUTES_SEASON]
    if len(qualified) > 50:
        usg_med = qualified["usage_proxy"].median()
        ts_med = qualified["ts_pct"].median()
    else:
        usg_med = USG_THRESHOLD_HIGH
        ts_med = TS_THRESHOLD_HIGH

    conditions = [
        (df["usage_proxy"] >= usg_med) & (df["ts_pct"] >= ts_med),
        (df["usage_proxy"] >= usg_med) & (df["ts_pct"] < ts_med),
        (df["usage_proxy"] < usg_med) & (df["ts_pct"] >= ts_med),
    ]
    choices = ["Elite Creator", "Volume Scorer", "Efficient Role"]
    df["usage_eff_quadrant"] = np.select(conditions, choices, default="Low Impact")

    # --- Scoring Profile ---
    # Based on 3PAr and FTA rate
    conditions = [
        (df["ppg"] >= 15) & (df["usage_proxy"] >= usg_med),
        (df["ppg"] >= 8) & (df["ppg"] < 15),
        (df["three_par"] >= 0.40) & (df["fg3_pct"] >= 0.30),
        (df["ppg"] < 5),
    ]
    choices = ["Primary Scorer", "Secondary Scorer", "Floor Spacer", "Non-Scorer"]
    df["scoring_profile"] = np.select(conditions, choices, default="Balanced")

    # --- Three-Level Scorer Flag ---
    # Produces from paint (2PT non-3), midrange/3PT, and FT line
    # Simplified: FG% > 40%, 3P% > 28%, FT% > 65%, with meaningful attempts
    df["three_level_scorer"] = (
        (df["fg_pct"] > 0.40) &
        (df["fg3_pct"] > 0.28) &
        (df["fg3a"] >= 20) &  # At least ~1 attempt per game
        (df["ft_pct"] > 0.65) &
        (df["fta"] >= 30)
    ).astype(int)

    # --- Defensive Profile ---
    conditions = [
        (df["stocks_per40"] >= 3.5) & (df["blk_per40"] >= 1.5),
        (df["stocks_per40"] >= 3.5) & (df["stl_per40"] >= 1.5),
        (df["stocks_per40"] >= 2.5),
    ]
    choices = ["Rim Protector", "Perimeter Disruptor", "Solid Defender"]
    df["defensive_profile"] = np.select(conditions, choices, default="Neutral")

    print(f"  Labels applied: minutes_tier, usage_eff_quadrant, scoring_profile, defensive_profile")
    return df


# =============================================================================
# 8. POSITION-WEIGHTED PERCENTILES
# =============================================================================

def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minutes-weighted percentile rankings within position groups
    and across all players.

    Uses rank-based percentiles weighted by minutes played so that
    players with more playing time anchor the distribution — similar
    to how batting average leaders need a minimum plate appearances.
    """
    print("Computing position-weighted percentiles...")

    metrics_to_rank = [
        "pts_per40", "reb_per40", "ast_per40", "stl_per40", "blk_per40",
        "efg_pct", "ts_pct", "usage_proxy", "ast_to_tov", "stocks_per40",
        "game_score_mean", "approx_bpm", "versatility_index",
        "pts_per_indiv_poss", "adj_game_score_mean",
    ]

    # Only rank players meeting minimum thresholds
    qualified = df["minutes_total"] >= MIN_MINUTES_SEASON

    for metric in metrics_to_rank:
        # Overall percentile
        col_name = f"{metric}_pctile"
        df[col_name] = np.nan
        df.loc[qualified, col_name] = df.loc[qualified, metric].rank(pct=True)

        # Position percentile
        col_name_pos = f"{metric}_pctile_pos"
        df[col_name_pos] = np.nan
        df.loc[qualified, col_name_pos] = (
            df.loc[qualified]
            .groupby("athlete_position_abbreviation")[metric]
            .rank(pct=True)
        )

    print(f"  Percentiles computed for {len(metrics_to_rank)} metrics (overall + by position)")
    return df


# =============================================================================
# 9. BREAKOUT SIGNALS (vs Prior-Season Benchmarks)
# =============================================================================

def compute_breakout_signals(df: pd.DataFrame,
                             benchmarks: pd.DataFrame) -> pd.DataFrame:
    """
    Compare current season metrics to 2024-25 D1 benchmarks to flag
    potential breakout performances.

    Breakout signals:
      - benchmark_delta_*:     Current value minus benchmark median/p75
      - tournament_readiness:  Composite index of recent form + efficiency +
                               opponent strength + consistency

    The benchmarks file has percentile distributions by position for key
    metrics like efg_pct, ts_pct, pts_per40, reb_per40, ast_per40, ast_tov.
    """
    print("Computing breakout signals against 2024-25 benchmarks...")

    # --- Tournament Readiness Index (always computed, benchmarks not required) ---
    # Composite score: recent form + efficiency + schedule strength + consistency
    # Each component scaled 0-25, total range 0-100
    zscore_capped = df["improvement_zscore_L10"].clip(-3, 3)
    df["tri_form"]        = ((zscore_capped + 3) / 6) * 25
    df["tri_efficiency"]  = df.get("ts_pct_pctile", df["ts_pct"].rank(pct=True)).fillna(0.5) * 25
    max_net = df["avg_opponent_net"].max()
    df["tri_schedule"]    = ((max_net - df["avg_opponent_net"]) / max_net * 25).clip(0, 25)
    df["tri_consistency"] = df["consistency_rating_L10"].fillna(0.5) * 25
    df["tournament_readiness_index"] = (
        df["tri_form"] + df["tri_efficiency"] + df["tri_schedule"] + df["tri_consistency"]
    ).round(2)
    df.drop(columns=["tri_form", "tri_efficiency", "tri_schedule", "tri_consistency"],
            inplace=True, errors="ignore")
    print(f"  Tournament Readiness Index range: "
          f"{df['tournament_readiness_index'].min():.1f} – {df['tournament_readiness_index'].max():.1f}")

    # If benchmarks unavailable or wrong schema, skip cross-season comparisons
    if benchmarks is None or benchmarks.empty or "position" not in benchmarks.columns:
        if benchmarks is not None and not benchmarks.empty:
            print("  ⚠️  Benchmarks schema mismatch (expected 'position' column) — skipping cross-season delta columns.")
        else:
            print("  ⚠️  No benchmarks — skipping cross-season delta columns.")
        return df

    # Parse benchmarks into a lookup: {(metric, position): {p50: val, p75: val, ...}}
    bench_lookup = {}
    for _, row in benchmarks.iterrows():
        metric = row["metric"]
        position = row["position"]
        bench_lookup[(metric, position)] = {
            "p25": row.get("p25", np.nan),
            "p50": row.get("p50", np.nan),
            "p75": row.get("p75", np.nan),
            "p90": row.get("p90", np.nan),
            "mean": row.get("mean", np.nan),
        }

    # Map current-season metrics to benchmark metric names
    metric_mapping = {
        "efg_pct": "efg_pct",
        "ts_pct": "ts_pct",
        "pts_per40": "pts_per40",
        "reb_per40": "reb_per40",
        "ast_per40": "ast_per40",
        "ast_to_tov": "ast_tov",
    }

    # Simplified position mapping (G, F, C, or "all")
    def map_position(pos_abbr):
        if pos_abbr in ("G", "PG", "SG"):
            return "Guard"
        elif pos_abbr in ("F", "SF", "PF"):
            return "Forward"
        elif pos_abbr in ("C",):
            return "Center"
        return "all"

    df["bench_position"] = df["athlete_position_abbreviation"].apply(map_position)

    for current_col, bench_metric in metric_mapping.items():
        p75_col = f"{current_col}_bench_p75"
        delta_col = f"{current_col}_vs_bench_p75"

        df[p75_col] = df["bench_position"].apply(
            lambda pos: bench_lookup.get((bench_metric, pos), {}).get("p75",
                         bench_lookup.get((bench_metric, "all"), {}).get("p75", np.nan))
        )
        df[delta_col] = df[current_col] - df[p75_col]

    # Clean up temp columns from benchmark deltas
    df.drop(columns=["bench_position"] +
                     [c for c in df.columns if c.endswith("_bench_p75")],
            inplace=True, errors="ignore")

    print(f"  Breakout signals computed.")
    return df


# =============================================================================
# 10. EXPORT
# =============================================================================

def export_datasets(season_df: pd.DataFrame,
                    game_df: pd.DataFrame,
                    output_dir: Optional[Path] = None):
    """
    Export final datasets:
      1. player_box_advanced_metrics.csv  (season-level, one row per player)
      2. player_game_log_enriched.csv     (game-level with rolling features)
    """
    out = output_dir or PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    print("Exporting datasets...")

    # --- Season-level export ---
    season_path = out / "player_box_advanced_metrics.csv"
    season_df.to_csv(season_path, index=False)
    print(f"  ✓ Season-level: {season_path.name}  ({len(season_df):,} rows, {len(season_df.columns)} cols)")

    # --- Game-level export (select key columns to keep file manageable) ---
    game_cols = [
        "game_id", "game_date", "season",
        "athlete_id", "athlete_display_name",
        "team_abbreviation", "team_location",
        "opponent_team_abbreviation", "opponent_team_location",
        "opponent_net_rank", "location", "game_result", "point_diff",
        "minutes", "points", "rebounds", "assists", "steals", "blocks", "turnovers",
        "field_goals_made", "field_goals_attempted",
        "three_point_field_goals_made", "three_point_field_goals_attempted",
        "free_throws_made", "free_throws_attempted",
        "game_score", "adj_game_score", "individual_ppp",
        "efg_pct_game", "ts_pct_game", "stocks",
        "is_double_double", "is_close_game", "rest_days", "is_back_to_back",
        "game_score_L5", "game_score_L10", "game_score_L15",
        "ts_pct_L5", "ts_pct_L10",
        "improvement_zscore_L10", "game_score_slope_L10",
        "consistency_rating_L10",
    ]
    # Only export columns that exist
    game_cols_exist = [c for c in game_cols if c in game_df.columns]
    game_path = out / "player_game_log_enriched.csv"
    game_df[game_cols_exist].to_csv(game_path, index=False)
    print(f"  ✓ Game-level: {game_path.name}  ({len(game_df):,} rows, {len(game_cols_exist)} cols)")

    # --- Top-50 NET filtered scouting file (used by dashboard main tab) ---
    if "team_net_rank" in season_df.columns:
        top50 = season_df[season_df["team_net_rank"] <= 50].copy()
        top50_path = out / "player_scouting_top50.csv"
        top50.to_csv(top50_path, index=False)
        print(f"  ✓ Top-50 scouting: {top50_path.name}  ({len(top50):,} rows, {top50['team_location'].nunique()} teams)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """Execute the full player box processing pipeline."""
    print("=" * 70)
    print("  PLAYER BOX PROCESSING PIPELINE")
    print("  WBB Player Role Profiles + Breakout Detection")
    print("=" * 70)
    print()

    # 1. Load data
    box_df = load_player_box()
    net_df = load_net_rankings()
    sched_df = load_schedule()
    benchmarks = load_benchmarks()
    print()

    # 2. Enrich with game context
    box_df = enrich_game_context(box_df, net_df, sched_df)
    print()

    # 3. Game-level derived metrics
    box_df = compute_game_level_metrics(box_df)
    print()

    # 4. Rolling / trend features
    box_df = compute_rolling_features(box_df)
    print()

    # 5. Aggregate to season
    season_df = aggregate_to_season(box_df)
    print()

    # 6. Advanced rate & efficiency metrics
    season_df = compute_advanced_metrics(season_df)
    print()

    # 7. Categorical labels
    season_df = apply_categorical_labels(season_df)
    print()

    # 8. Percentiles
    season_df = compute_percentiles(season_df)
    print()

    # 9. Breakout signals
    season_df = compute_breakout_signals(season_df, benchmarks)
    print()

    # 10. Filter to meaningful sample sizes
    season_df_full = season_df.copy()
    season_df = season_df[season_df["minutes_total"] >= MIN_MINUTES_SEASON].copy()
    print(f"Final dataset: {len(season_df):,} players (min {MIN_MINUTES_SEASON} minutes)")
    print()

    # 11. Export
    export_datasets(season_df, box_df)

    print()
    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    return season_df, box_df


if __name__ == "__main__":
    season_df, game_df = run_pipeline()
