"""
Script 11 — Player Development (YoY-Aware)
==========================================
Builds player development profiles by combining:
1) Within-season trajectory from player_game_log_enriched.csv
2) Year-over-year deltas from multi-season player box aggregation

Inputs
------
Required:
- data/processed/player_game_log_enriched.csv

Optional:
- data/processed/player_feature_table_2026.csv
- data/processed/player_box_multiseason_yoy_2021_2026.csv
- analysis/role_archetypes/role_archetype_assignments_2026.csv

Outputs
-------
- analysis/player_development/player_development_profiles_2026.csv
- analysis/player_development/player_development_leaderboard_2026.csv
- analysis/player_development/player_development_flags_2026.csv
- analysis/player_development/player_development_summary_2026.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PLAYER_FEATURE_TABLE, PLAYER_GAME_LOG, REPO_ROOT, SEASON, validate_inputs


# =============================================================================
# PATHS
# =============================================================================

DEV_OUT_DIR = REPO_ROOT / "analysis" / "player_development"
ROLE_ASSIGNMENTS_FILE = REPO_ROOT / "analysis" / "role_archetypes" / "role_archetype_assignments_2026.csv"
HIST_YOY_FILE = REPO_ROOT / "data" / "processed" / "player_box_multiseason_yoy_2021_2026.csv"

PROFILES_OUT = DEV_OUT_DIR / "player_development_profiles_2026.csv"
LEADERBOARD_OUT = DEV_OUT_DIR / "player_development_leaderboard_2026.csv"
FLAGS_OUT = DEV_OUT_DIR / "player_development_flags_2026.csv"
SUMMARY_OUT = DEV_OUT_DIR / "player_development_summary_2026.md"


# =============================================================================
# HELPERS
# =============================================================================

def _to_int_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sigma = x.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sigma


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _safe_std(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").std(ddof=0))


def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num / den)


def _build_intervention_focus(row: pd.Series) -> str:
    if pd.notna(row.get("ts_recent")) and row["ts_recent"] < 0.50:
        return "Shot quality and finishing efficiency"
    if pd.notna(row.get("tov_per40_recent")) and row["tov_per40_recent"] > 3.0:
        return "Decision-making and turnover reduction"
    if pd.notna(row.get("consistency_recent_std")) and row["consistency_recent_std"] > 6.5:
        return "Game-to-game consistency"
    if pd.notna(row.get("tough_vs_easy_game_score_gap")) and row["tough_vs_easy_game_score_gap"] < -1.0:
        return "Performance against top-50 opponents"
    if pd.notna(row.get("ts_pct_yoy_delta")) and row["ts_pct_yoy_delta"] < -0.01:
        return "Efficiency recovery versus last season"
    if pd.notna(row.get("pts_per40_yoy_delta")) and row["pts_per40_yoy_delta"] < -1.5:
        return "Role confidence and offensive assertiveness"
    return "Maintain current development trajectory"


def _build_status(score: float) -> str:
    if pd.isna(score):
        return "Insufficient Sample"
    if score >= 0.8:
        return "Ascending"
    if score >= 0.1:
        return "Stable+"
    if score >= -0.6:
        return "Mixed"
    return "Needs Intervention"


def _fmt(v: float, nd: int = 2) -> str:
    return "NA" if pd.isna(v) else f"{v:.{nd}f}"


# =============================================================================
# MAIN
# =============================================================================

print("\n== Script 11: Player Development (YoY-Aware) ==")

if not validate_inputs(required=[PLAYER_GAME_LOG], optional=[PLAYER_FEATURE_TABLE, ROLE_ASSIGNMENTS_FILE, HIST_YOY_FILE]):
    raise SystemExit(1)

DEV_OUT_DIR.mkdir(parents=True, exist_ok=True)

log = pd.read_csv(PLAYER_GAME_LOG, low_memory=False)
log["athlete_id"] = _to_int_id(log["athlete_id"])
log["game_date"] = pd.to_datetime(log["game_date"], errors="coerce")
log = log[log["athlete_id"].notna()].copy()
log = log.sort_values(["athlete_id", "game_date", "game_id"], na_position="last")

print(f"Loaded game log: {len(log):,} rows x {len(log.columns)} columns")

records: List[Dict] = []

for athlete_id, g in log.groupby("athlete_id", sort=False):
    g = g.dropna(subset=["game_date"]).copy()
    if g.empty:
        continue

    n_games = len(g)
    window = 5 if n_games >= 10 else max(3, n_games // 2)
    early = g.head(window)
    late = g.tail(window)

    half = n_games // 2
    first_half = g.iloc[:half] if half > 0 else g
    second_half = g.iloc[half:] if half > 0 else g

    close_games = g[g.get("is_close_game", 0) == 1]
    close_first = first_half[first_half.get("is_close_game", 0) == 1]
    close_second = second_half[second_half.get("is_close_game", 0) == 1]

    tough = g[pd.to_numeric(g.get("opponent_net_rank"), errors="coerce") <= 50]
    easy = g[pd.to_numeric(g.get("opponent_net_rank"), errors="coerce") >= 100]

    name = g["athlete_display_name"].dropna().iloc[-1] if "athlete_display_name" in g.columns and g["athlete_display_name"].notna().any() else "Unknown"
    team = g["team_location"].dropna().iloc[-1] if "team_location" in g.columns and g["team_location"].notna().any() else np.nan
    team_abbr = g["team_abbreviation"].dropna().iloc[-1] if "team_abbreviation" in g.columns and g["team_abbreviation"].notna().any() else np.nan

    game_score_early = _safe_mean(early, "game_score")
    game_score_recent = _safe_mean(late, "game_score")
    game_score_trend = game_score_recent - game_score_early if pd.notna(game_score_recent) and pd.notna(game_score_early) else np.nan

    ts_early = _safe_mean(early, "ts_pct_game")
    ts_recent = _safe_mean(late, "ts_pct_game")
    ts_trend = ts_recent - ts_early if pd.notna(ts_recent) and pd.notna(ts_early) else np.nan

    ppp_early = _safe_mean(early, "individual_ppp")
    ppp_recent = _safe_mean(late, "individual_ppp")
    ppp_trend = ppp_recent - ppp_early if pd.notna(ppp_recent) and pd.notna(ppp_early) else np.nan

    consistency_full_std = _safe_std(g, "game_score")
    consistency_recent_std = _safe_std(late, "game_score")
    consistency_improvement = (
        consistency_full_std - consistency_recent_std
        if pd.notna(consistency_full_std) and pd.notna(consistency_recent_std)
        else np.nan
    )

    tough_game_score = _safe_mean(tough, "game_score")
    easy_game_score = _safe_mean(easy, "game_score")
    tough_vs_easy_gap = (
        tough_game_score - easy_game_score
        if pd.notna(tough_game_score) and pd.notna(easy_game_score)
        else np.nan
    )

    close_first_mean = _safe_mean(close_first, "game_score")
    close_second_mean = _safe_mean(close_second, "game_score")
    close_trend = (
        close_second_mean - close_first_mean
        if pd.notna(close_first_mean) and pd.notna(close_second_mean)
        else np.nan
    )

    minutes_total = float(pd.to_numeric(g.get("minutes"), errors="coerce").fillna(0).sum()) if "minutes" in g.columns else np.nan
    minutes_recent = _safe_mean(late, "minutes")
    availability_rate = _safe_ratio(float((pd.to_numeric(g.get("minutes"), errors="coerce") >= 10).sum()), float(n_games))

    tov_recent = _safe_mean(late, "turnovers")
    min_recent = _safe_mean(late, "minutes")
    tov_per40_recent = (tov_recent / min_recent * 40.0) if pd.notna(tov_recent) and pd.notna(min_recent) and min_recent > 0 else np.nan

    records.append(
        {
            "athlete_id": athlete_id,
            "athlete_display_name": name,
            "team_location": team,
            "team_abbreviation": team_abbr,
            "games_played": n_games,
            "window_games": window,
            "minutes_total": minutes_total,
            "minutes_recent": minutes_recent,
            "availability_rate_10plus_min": availability_rate,
            "game_score_early": game_score_early,
            "game_score_recent": game_score_recent,
            "game_score_trend": game_score_trend,
            "ts_early": ts_early,
            "ts_recent": ts_recent,
            "ts_trend": ts_trend,
            "ppp_early": ppp_early,
            "ppp_recent": ppp_recent,
            "ppp_trend": ppp_trend,
            "consistency_full_std": consistency_full_std,
            "consistency_recent_std": consistency_recent_std,
            "consistency_improvement": consistency_improvement,
            "tough_game_score": tough_game_score,
            "easy_game_score": easy_game_score,
            "tough_vs_easy_game_score_gap": tough_vs_easy_gap,
            "close_games_n": int(len(close_games)),
            "close_game_score_first_half": close_first_mean,
            "close_game_score_second_half": close_second_mean,
            "close_game_score_trend": close_trend,
            "tov_per40_recent": tov_per40_recent,
        }
    )

profiles = pd.DataFrame(records)
if profiles.empty:
    raise SystemExit("No player development profiles generated. Check player_game_log_enriched.csv.")

# Optional context from feature table.
if PLAYER_FEATURE_TABLE.exists():
    feat = pd.read_csv(PLAYER_FEATURE_TABLE, low_memory=False)
    feat["athlete_id"] = _to_int_id(feat["athlete_id"])

    keep = [
        "athlete_id",
        "seed",
        "region",
        "team_id",
        "athlete_position_abbreviation",
        "recruit_rank",
        "on_net_rtg",
        "net_rtg_diff",
    ]
    keep = [c for c in keep if c in feat.columns]
    feat = feat[keep].drop_duplicates(subset=["athlete_id"])
    profiles = profiles.merge(feat, on="athlete_id", how="left")

# YoY context from historical aggregation.
if HIST_YOY_FILE.exists():
    yoy = pd.read_csv(HIST_YOY_FILE, low_memory=False)
    yoy["athlete_id"] = _to_int_id(yoy["athlete_id"])
    yoy["season"] = pd.to_numeric(yoy.get("season"), errors="coerce")
    yoy_cur = yoy[yoy["season"] == SEASON].copy()

    keep = [
        "athlete_id",
        "prev_season",
        "yoy_growth_score",
        "pts_per40_yoy_delta",
        "reb_per40_yoy_delta",
        "ast_per40_yoy_delta",
        "tov_per40_yoy_delta",
        "ts_pct_yoy_delta",
        "efg_pct_yoy_delta",
        "usage_proxy_yoy_delta",
        "pts_per40_pctile_pos_yoy_delta",
        "ts_pct_pctile_pos_yoy_delta",
        "usage_proxy_pctile_pos_yoy_delta",
    ]
    keep = [c for c in keep if c in yoy_cur.columns]
    yoy_cur = yoy_cur[keep].drop_duplicates(subset=["athlete_id"])

    profiles = profiles.merge(yoy_cur, on="athlete_id", how="left")
    print(f"Joined YoY context for season {SEASON}: {profiles['prev_season'].notna().sum():,} players with prior-season history")
else:
    print("  ⚠️  Historical YoY file not found; YoY features skipped.")

# Role context.
if ROLE_ASSIGNMENTS_FILE.exists():
    roles = pd.read_csv(ROLE_ASSIGNMENTS_FILE, low_memory=False)
    roles["athlete_id"] = _to_int_id(roles["athlete_id"])
    keep = ["athlete_id", "role_code", "role_name", "role_confidence", "confidence_band", "assignment_source"]
    keep = [c for c in keep if c in roles.columns]
    roles = roles[keep].drop_duplicates(subset=["athlete_id"])
    profiles = profiles.merge(roles, on="athlete_id", how="left")

# Composite readiness with YoY component.
profiles["z_recent_game_score"] = _zscore(profiles["game_score_recent"]).fillna(0)
profiles["z_game_score_trend"] = _zscore(profiles["game_score_trend"]).fillna(0)
profiles["z_recent_ts"] = _zscore(profiles["ts_recent"]).fillna(0)
profiles["z_consistency"] = _zscore(-profiles["consistency_recent_std"]).fillna(0)
profiles["z_tough_context"] = _zscore(profiles["tough_game_score"]).fillna(0)
profiles["z_availability"] = _zscore(profiles["availability_rate_10plus_min"]).fillna(0)
profiles["z_yoy_growth"] = _zscore(profiles.get("yoy_growth_score", pd.Series(np.nan, index=profiles.index))).fillna(0)

profiles["sample_weight"] = np.minimum(1.0, profiles["games_played"] / 20.0)
profiles["readiness_index"] = (
    0.22 * profiles["z_recent_game_score"]
    + 0.16 * profiles["z_game_score_trend"]
    + 0.16 * profiles["z_recent_ts"]
    + 0.12 * profiles["z_consistency"]
    + 0.10 * profiles["z_tough_context"]
    + 0.08 * profiles["z_availability"]
    + 0.16 * profiles["z_yoy_growth"]
)
profiles["readiness_index"] = profiles["readiness_index"] * profiles["sample_weight"]

# Flags.
profiles["flag_skill_growth"] = (
    pd.to_numeric(profiles["game_score_trend"], errors="coerce") >= 1.0
) & (
    pd.to_numeric(profiles["ts_trend"], errors="coerce") >= 0.02
)
profiles["flag_consistency_improving"] = pd.to_numeric(profiles["consistency_improvement"], errors="coerce") >= 0.5
profiles["flag_clutch_trend_up"] = pd.to_numeric(profiles["close_game_score_trend"], errors="coerce") >= 0.5
profiles["flag_tough_opponent_ready"] = pd.to_numeric(profiles["tough_vs_easy_game_score_gap"], errors="coerce") >= -0.25

profiles["flag_yoy_efficiency_up"] = pd.to_numeric(profiles.get("ts_pct_yoy_delta"), errors="coerce") >= 0.01
profiles["flag_yoy_volume_up"] = pd.to_numeric(profiles.get("pts_per40_yoy_delta"), errors="coerce") >= 1.0
profiles["flag_yoy_usage_eff_balanced"] = (
    pd.to_numeric(profiles.get("usage_proxy_yoy_delta"), errors="coerce") >= -0.005
) & (
    pd.to_numeric(profiles.get("ts_pct_yoy_delta"), errors="coerce") >= -0.005
)

profiles["development_status"] = profiles["readiness_index"].map(_build_status)
profiles["intervention_focus"] = profiles.apply(_build_intervention_focus, axis=1)

profiles["improvement_profile"] = profiles.apply(
    lambda r: (
        f"{r['development_status']}: GS trend {_fmt(r.get('game_score_trend'))}, "
        f"TS trend {_fmt(r.get('ts_trend'), 3)}, YoY TS {_fmt(r.get('ts_pct_yoy_delta'), 3)}, "
        f"YoY PTS/40 {_fmt(r.get('pts_per40_yoy_delta'))}"
    ),
    axis=1,
)

profiles = profiles.sort_values("readiness_index", ascending=False).reset_index(drop=True)
profiles.to_csv(PROFILES_OUT, index=False)
print(f"Saved profiles: {PROFILES_OUT}")

leaderboard_cols = [
    "athlete_id",
    "athlete_display_name",
    "team_location",
    "team_abbreviation",
    "role_code",
    "role_name",
    "games_played",
    "readiness_index",
    "development_status",
    "game_score_recent",
    "game_score_trend",
    "ts_recent",
    "ts_trend",
    "yoy_growth_score",
    "pts_per40_yoy_delta",
    "ts_pct_yoy_delta",
    "usage_proxy_yoy_delta",
    "intervention_focus",
]
leaderboard_cols = [c for c in leaderboard_cols if c in profiles.columns]

leaderboard = profiles[leaderboard_cols].copy()
leaderboard.insert(0, "development_rank", np.arange(1, len(leaderboard) + 1))
leaderboard.to_csv(LEADERBOARD_OUT, index=False)
print(f"Saved leaderboard: {LEADERBOARD_OUT}")

flags_cols = [
    "athlete_id",
    "athlete_display_name",
    "team_location",
    "role_code",
    "role_name",
    "flag_skill_growth",
    "flag_consistency_improving",
    "flag_clutch_trend_up",
    "flag_tough_opponent_ready",
    "flag_yoy_efficiency_up",
    "flag_yoy_volume_up",
    "flag_yoy_usage_eff_balanced",
    "development_status",
    "intervention_focus",
    "improvement_profile",
]
flags_cols = [c for c in flags_cols if c in profiles.columns]
flags = profiles[flags_cols].copy()
flags.to_csv(FLAGS_OUT, index=False)
print(f"Saved flags/report layer: {FLAGS_OUT}")

# Summary markdown.
now = datetime.now().strftime("%Y-%m-%d %H:%M")
ascending = int((profiles["development_status"] == "Ascending").sum())
intervention = int((profiles["development_status"] == "Needs Intervention").sum())
yoy_available = int(profiles.get("prev_season", pd.Series(dtype=float)).notna().sum())

top5 = profiles[[c for c in ["athlete_display_name", "team_location", "readiness_index", "development_status", "ts_pct_yoy_delta", "pts_per40_yoy_delta"] if c in profiles.columns]].head(5)

lines = [
    "# Player Development Summary (YoY-Aware)",
    "",
    f"Run timestamp: {now}",
    f"Season: {SEASON}",
    f"Players profiled: {len(profiles)}",
    f"Players with YoY history: {yoy_available}",
    "",
    "## Status Mix",
    "",
    f"- Ascending: {ascending}",
    f"- Needs Intervention: {intervention}",
    "",
    "## Top 5 Readiness",
    "",
]

for _, r in top5.iterrows():
    lines.append(
        f"- {r.get('athlete_display_name', 'Unknown')} ({r.get('team_location', 'N/A')}): "
        f"{_fmt(r.get('readiness_index'), 3)} [{r.get('development_status', 'N/A')}], "
        f"YoY TS {_fmt(r.get('ts_pct_yoy_delta'), 3)}, YoY PTS/40 {_fmt(r.get('pts_per40_yoy_delta'))}"
    )

lines.extend(
    [
        "",
        "## Output Files",
        "",
        f"- {PROFILES_OUT}",
        f"- {LEADERBOARD_OUT}",
        f"- {FLAGS_OUT}",
    ]
)

SUMMARY_OUT.write_text("\n".join(lines))
print(f"Saved summary markdown: {SUMMARY_OUT}")

print("\n✅ Script 11 complete.")
