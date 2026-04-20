"""
Script 10 — Role Archetypes + Roster Construction
===================================================
Builds the first decision-layer deliverables for the BIS sprint:
  1) Player archetype assignments that preserve canonical C0-C5 taxonomy labels
  2) Role-level cluster summary for QA and reporting
  3) Team role balance report
  4) Role-gap recommendations with replacement candidates

Design principles
-----------------
- Preserve original role labels from clustering outputs whenever available.
- Keep cluster IDs stable (C0-C5) and map names from dashboard taxonomy metadata.
- Infer missing player roles via centroid distance only when a player has no prior
  assignment in the archived clustering outputs.

Outputs
-------
- analysis/role_archetypes/role_archetype_assignments_2026.csv
- analysis/role_archetypes/role_cluster_summary_2026.csv
- analysis/roster_construction/team_role_balance_report_2026.csv
- analysis/roster_construction/role_gap_recommendations_2026.csv
- analysis/role_archetypes/role_archetypes_summary_2026.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PLAYER_FEATURE_TABLE, REPO_ROOT, SEASON, validate_inputs


# =============================================================================
# PATHS
# =============================================================================

ROLE_SOURCE_DIR = REPO_ROOT / "analysis" / "player_archetypes" / "Player Archetypes"
ROLE_TAGS_FILE = ROLE_SOURCE_DIR / "player_role_usage_tags.parquet"
DASHBOARD_SLIM_FILE = ROLE_SOURCE_DIR / "dashboard_slim_k6_enriched.json"

ROLE_OUT_DIR = REPO_ROOT / "analysis" / "role_archetypes"
ROSTER_OUT_DIR = REPO_ROOT / "analysis" / "roster_construction"

ASSIGNMENTS_OUT = ROLE_OUT_DIR / "role_archetype_assignments_2026.csv"
SUMMARY_OUT = ROLE_OUT_DIR / "role_cluster_summary_2026.csv"
TEAM_BALANCE_OUT = ROSTER_OUT_DIR / "team_role_balance_report_2026.csv"
GAP_RECS_OUT = ROSTER_OUT_DIR / "role_gap_recommendations_2026.csv"
NARRATIVE_OUT = ROLE_OUT_DIR / "role_archetypes_summary_2026.md"


# =============================================================================
# CANONICAL TAXONOMY (locked labels + one-liners from role taxonomy docs)
# =============================================================================

CANONICAL_ROLE_NAMES = {
    0: "Stretch Defender",
    1: "Complete Performer",
    2: "Perimeter Contributor",
    3: "Shot Creator",
    4: "Two-Way Elite Wing",
    5: "Interior Anchor",
}

CANONICAL_ROLE_ONELINERS = {
    0: "Floor-spacer who guards across positions and lives behind the arc.",
    1: "Transcendent all-around elite: dominant scorer, playmaker, and defender.",
    2: "Low-usage support player who spaces the floor without demanding possessions.",
    3: "Primary offensive engine who gets own shot, draws fouls, and initiates.",
    4: "High-usage offensive weapon who also leads in defensive disruption.",
    5: "Post-up and roll scorer with elite rebounding dominance.",
}


# =============================================================================
# HELPERS
# =============================================================================

@dataclass
class RoleReference:
    role_id: int
    role_code: str
    role_name: str
    role_one_liner: str


def _to_int_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def _choose_team_col(df: pd.DataFrame) -> str:
    for c in ["team_location", "team_name", "team_short_display_name", "team_id"]:
        if c in df.columns:
            return c
    raise ValueError("No team column found in feature table.")


def _choose_player_name_col(df: pd.DataFrame) -> str:
    for c in ["athlete_display_name", "athlete_short_name", "player_name"]:
        if c in df.columns:
            return c
    raise ValueError("No player display-name column found in feature table.")


def _choose_minutes_col(df: pd.DataFrame) -> str:
    for c in ["on_minutes", "minutes_total", "minutes_per_game"]:
        if c in df.columns:
            return c
    return "minutes_total"


def _load_role_reference() -> Dict[int, RoleReference]:
    names = dict(CANONICAL_ROLE_NAMES)

    if DASHBOARD_SLIM_FILE.exists():
        try:
            meta = json.loads(DASHBOARD_SLIM_FILE.read_text())
            cl = meta.get("cl", {})
            for k, v in cl.items():
                rid = int(k)
                if isinstance(v, dict) and v.get("name"):
                    names[rid] = v["name"]
        except Exception as exc:
            print(f"  ⚠️  Could not parse dashboard cluster metadata: {exc}")

    ref = {}
    for rid in sorted(CANONICAL_ROLE_NAMES.keys()):
        ref[rid] = RoleReference(
            role_id=rid,
            role_code=f"C{rid}",
            role_name=names.get(rid, CANONICAL_ROLE_NAMES[rid]),
            role_one_liner=CANONICAL_ROLE_ONELINERS.get(rid, ""),
        )
    return ref


def _infer_missing_roles(
    df: pd.DataFrame,
    role_ref: Dict[int, RoleReference],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Infer roles for players without cluster assignments.

    Returns:
      inferred_role_id, inferred_role_name, inferred_confidence
    """
    feature_candidates = [
        "usage_proxy",
        "ts_pct",
        "pts_per40",
        "reb_per40",
        "ast_per40",
        "stocks_per40",
        "stl_per40",
        "blk_per40",
        "tov_per40",
        "paint_share",
        "three_share",
        "on_net_rtg",
        "net_rtg_diff",
    ]
    feat_cols = [c for c in feature_candidates if c in df.columns]

    # Need enough signal for centroid matching.
    if len(feat_cols) < 5:
        print("  ⚠️  Not enough numeric features for centroid inference; defaulting missing to C2")
        rid = pd.Series(np.where(df["cluster_id"].isna(), 2, pd.NA), index=df.index, dtype="Int64")
        rname = rid.map(lambda x: role_ref[int(x)].role_name if pd.notna(x) else pd.NA)
        conf = pd.Series(np.where(df["cluster_id"].isna(), 0.55, pd.NA), index=df.index)
        return rid, rname, conf

    train = df[df["cluster_id"].notna()].copy()
    train = train[train[feat_cols].notna().sum(axis=1) >= max(4, int(0.6 * len(feat_cols)))]

    if train.empty:
        print("  ⚠️  No canonical assignments available for centroid training; defaulting missing to C2")
        rid = pd.Series(np.where(df["cluster_id"].isna(), 2, pd.NA), index=df.index, dtype="Int64")
        rname = rid.map(lambda x: role_ref[int(x)].role_name if pd.notna(x) else pd.NA)
        conf = pd.Series(np.where(df["cluster_id"].isna(), 0.55, pd.NA), index=df.index)
        return rid, rname, conf

    # Standardize using training statistics.
    train_feat = train[feat_cols].copy()
    means = train_feat.mean(numeric_only=True)
    stds = train_feat.std(ddof=0, numeric_only=True).replace(0, 1)

    all_feat = df[feat_cols].copy().fillna(means)
    z_all = (all_feat - means) / stds

    z_train = z_all.loc[train.index]
    centroids: Dict[int, np.ndarray] = {}
    for rid, g in train.groupby("cluster_id"):
        rid_int = int(rid)
        centroids[rid_int] = z_train.loc[g.index, feat_cols].mean().to_numpy(dtype=float)

    missing_mask = df["cluster_id"].isna()
    inferred_role_id = pd.Series(pd.NA, index=df.index, dtype="Int64")
    inferred_role_name = pd.Series(pd.NA, index=df.index, dtype="object")
    inferred_confidence = pd.Series(np.nan, index=df.index, dtype="float64")

    if not missing_mask.any():
        return inferred_role_id, inferred_role_name, inferred_confidence

    centroid_ids = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[rid] for rid in centroid_ids])

    z_missing = z_all.loc[missing_mask, feat_cols].to_numpy(dtype=float)
    # Euclidean distance to each centroid.
    diffs = z_missing[:, None, :] - centroid_matrix[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))

    # Convert distances to pseudo-probabilities via softmax(-distance).
    logits = -dists
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    argmax = probs.argmax(axis=1)
    best_ids = [centroid_ids[i] for i in argmax]
    best_probs = probs[np.arange(len(argmax)), argmax]

    idx = df.index[missing_mask]
    inferred_role_id.loc[idx] = pd.Series(best_ids, index=idx, dtype="Int64")
    inferred_role_name.loc[idx] = inferred_role_id.loc[idx].map(lambda x: role_ref[int(x)].role_name)
    inferred_confidence.loc[idx] = best_probs

    return inferred_role_id, inferred_role_name, inferred_confidence


def _usage_tier(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x >= 24:
        return "High Usage"
    if x >= 18:
        return "Moderate Usage"
    return "Low Usage"


def _efficiency_tier(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x >= 0.58:
        return "Elite Efficiency"
    if x >= 0.52:
        return "Average Efficiency"
    return "Below Average Efficiency"


def _ball_security_tier(tov40: float) -> str:
    if pd.isna(tov40):
        return "Unknown"
    if tov40 <= 2.1:
        return "Strong Ball Security"
    if tov40 <= 3.0:
        return "Average Ball Security"
    return "Turnover Risk"


def _role_usage_archetype(usage_tier: str, efficiency_tier: str) -> str:
    if usage_tier == "High Usage" and efficiency_tier == "Elite Efficiency":
        return "Prototypical Scorer"
    if usage_tier == "Low Usage" and efficiency_tier == "Elite Efficiency":
        return "Underutilized Creator"
    if usage_tier == "High Usage" and efficiency_tier == "Below Average Efficiency":
        return "Usage Overload"
    return "Role-Appropriate"


def _confidence_band(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x >= 0.85:
        return "High"
    if x >= 0.70:
        return "Medium"
    return "Low"


def _build_team_balance(assignments: pd.DataFrame, role_ref: Dict[int, RoleReference]) -> Tuple[pd.DataFrame, pd.Series]:
    team_col = _choose_team_col(assignments)
    minutes_col = _choose_minutes_col(assignments)

    work = assignments.copy()
    if minutes_col == "minutes_per_game" and "games_played" in work.columns:
        work["role_minutes"] = pd.to_numeric(work["minutes_per_game"], errors="coerce").fillna(0) * pd.to_numeric(
            work["games_played"], errors="coerce"
        ).fillna(0)
    else:
        work["role_minutes"] = pd.to_numeric(work[minutes_col], errors="coerce").fillna(0)

    work["role_minutes"] = work["role_minutes"].clip(lower=0)

    total = (
        work.groupby(team_col, dropna=False)
        .agg(
            team_id=("team_id", "first") if "team_id" in work.columns else ("role_minutes", "size"),
            seed=("seed", "first") if "seed" in work.columns else ("role_minutes", "size"),
            region=("region", "first") if "region" in work.columns else ("role_minutes", "size"),
            total_players=("athlete_id", "nunique"),
            total_role_minutes=("role_minutes", "sum"),
        )
        .reset_index()
    )

    role_counts = (
        work.pivot_table(
            index=team_col,
            columns="role_name",
            values="athlete_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    role_minutes = (
        work.pivot_table(
            index=team_col,
            columns="role_name",
            values="role_minutes",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    report = total.merge(role_counts, on=team_col, how="left", suffixes=("", "_count"))

    # Add minute shares per role.
    for ref in role_ref.values():
        rn = ref.role_name
        count_col = rn
        if count_col not in report.columns:
            report[count_col] = 0

    role_count_cols = [r.role_name for r in role_ref.values()]

    role_minutes_long = role_minutes.set_index(team_col)
    for rn in role_count_cols:
        minute_col = f"{rn}_minutes"
        share_col = f"{rn}_share"
        if rn in role_minutes_long.columns:
            report = report.merge(
                role_minutes_long[[rn]].rename(columns={rn: minute_col}).reset_index(),
                on=team_col,
                how="left",
            )
        else:
            report[minute_col] = 0.0

        report[minute_col] = pd.to_numeric(report[minute_col], errors="coerce").fillna(0.0)
        denom = report["total_role_minutes"].replace(0, np.nan)
        report[share_col] = (report[minute_col] / denom).fillna(0.0)

    # Dominant role by minute share.
    share_cols = [f"{r.role_name}_share" for r in role_ref.values()]
    dom_idx = report[share_cols].to_numpy().argmax(axis=1)
    dom_role = [share_cols[i].replace("_share", "") for i in dom_idx]
    report["dominant_role"] = dom_role

    # Shannon diversity index normalized to [0, 100].
    p = report[share_cols].to_numpy(dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1)
    entropy_max = np.log(len(share_cols)) if share_cols else 1.0
    report["role_diversity_score"] = (100.0 * entropy / entropy_max).round(2)

    # Target profile from top-seeded teams.
    seeded = report[pd.to_numeric(report["seed"], errors="coerce").notna()].copy()
    top_seed = seeded[pd.to_numeric(seeded["seed"], errors="coerce") <= 4]
    baseline_pool = top_seed if not top_seed.empty else seeded
    if baseline_pool.empty:
        baseline_pool = report

    target_profile = baseline_pool[share_cols].mean()

    # Role gaps: positive means shortage relative to baseline profile.
    for sc in share_cols:
        gap_col = sc.replace("_share", "_gap")
        report[gap_col] = (target_profile[sc] - report[sc]).round(4)

    gap_cols = [c for c in report.columns if c.endswith("_gap")]

    def _top_gaps(row: pd.Series) -> Tuple[str, float, str, float]:
        vals = sorted([(c.replace("_gap", ""), row[c]) for c in gap_cols], key=lambda x: x[1], reverse=True)
        g1_role, g1_val = vals[0]
        g2_role, g2_val = vals[1] if len(vals) > 1 else ("", 0.0)
        return g1_role, float(g1_val), g2_role, float(g2_val)

    tops = report.apply(_top_gaps, axis=1, result_type="expand")
    tops.columns = ["top_gap_role_1", "top_gap_size_1", "top_gap_role_2", "top_gap_size_2"]
    report = pd.concat([report, tops], axis=1)

    return report, target_profile


def _build_gap_recommendations(
    assignments: pd.DataFrame,
    team_report: pd.DataFrame,
) -> pd.DataFrame:
    team_col = _choose_team_col(assignments)
    name_col = _choose_player_name_col(assignments)

    pool = assignments.copy()
    for c in ["on_net_rtg", "ts_pct", "pts_per40"]:
        if c not in pool.columns:
            pool[c] = np.nan

    pool["perf_score"] = (
        0.4 * _zscore(pool["on_net_rtg"]).fillna(0)
        + 0.3 * _zscore(pool["ts_pct"]).fillna(0)
        + 0.3 * _zscore(pool["pts_per40"]).fillna(0)
    )

    pool["role_fit_score"] = (
        0.60 * pd.to_numeric(pool["role_confidence"], errors="coerce").fillna(0.0)
        + 0.40 * pool["perf_score"].fillna(0.0)
    )

    rows: List[dict] = []

    for _, tr in team_report.iterrows():
        team_name = tr[team_col]

        for role_col, gap_col in [("top_gap_role_1", "top_gap_size_1"), ("top_gap_role_2", "top_gap_size_2")]:
            role_name = tr.get(role_col)
            gap_size = float(tr.get(gap_col, 0.0) or 0.0)
            if not isinstance(role_name, str) or role_name == "":
                continue
            if gap_size <= 0:
                continue

            candidates = pool[(pool["role_name"] == role_name) & (pool[team_col] != team_name)].copy()
            if candidates.empty:
                continue

            candidates["team_need_adjusted_score"] = candidates["role_fit_score"] * (1.0 + gap_size)
            top = candidates.sort_values("team_need_adjusted_score", ascending=False).head(5)

            for rank, (_, c) in enumerate(top.iterrows(), start=1):
                rows.append(
                    {
                        "target_team": team_name,
                        "needed_role": role_name,
                        "gap_size": round(gap_size, 4),
                        "recommendation_rank": rank,
                        "candidate_athlete_id": c.get("athlete_id"),
                        "candidate_name": c.get(name_col),
                        "candidate_team": c.get(team_col),
                        "candidate_role_confidence": round(float(c.get("role_confidence", np.nan)), 4)
                        if pd.notna(c.get("role_confidence", np.nan))
                        else np.nan,
                        "candidate_fit_score": round(float(c.get("team_need_adjusted_score", np.nan)), 4)
                        if pd.notna(c.get("team_need_adjusted_score", np.nan))
                        else np.nan,
                        "candidate_on_net_rtg": c.get("on_net_rtg"),
                        "candidate_ts_pct": c.get("ts_pct"),
                        "candidate_pts_per40": c.get("pts_per40"),
                    }
                )

    recs = pd.DataFrame(rows)
    if not recs.empty:
        recs = recs.sort_values(["target_team", "needed_role", "recommendation_rank"]).reset_index(drop=True)
    return recs


# =============================================================================
# MAIN
# =============================================================================

print("\n== Script 10: Role Archetypes + Roster Construction ==")

if not validate_inputs(required=[PLAYER_FEATURE_TABLE], optional=[ROLE_TAGS_FILE, DASHBOARD_SLIM_FILE]):
    raise SystemExit(1)

ROLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
ROSTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

feature = pd.read_csv(PLAYER_FEATURE_TABLE, low_memory=False)
feature["athlete_id"] = _to_int_id(feature["athlete_id"])
print(f"Loaded feature table: {len(feature):,} rows x {len(feature.columns)} columns")

role_ref = _load_role_reference()

# Optional archival role tags (canonical historical clustering output).
role_tags = None
if ROLE_TAGS_FILE.exists():
    role_tags = pd.read_parquet(ROLE_TAGS_FILE)
    role_tags["athlete_id"] = _to_int_id(role_tags["athlete_id"])
    keep = [
        "athlete_id",
        "cluster_id",
        "role_label",
        "role_confidence",
        "role_usage_archetype",
        "usage_tier",
        "efficiency_tier",
        "ball_security_tier",
    ]
    keep = [c for c in keep if c in role_tags.columns]
    role_tags = role_tags[keep].drop_duplicates(subset=["athlete_id"])
    print(f"Loaded archived role tags: {len(role_tags):,} player rows")
else:
    print("  ⚠️  Archived role tags missing; full assignment will use centroid inference.")

work = feature.copy()
if role_tags is not None:
    work = work.merge(role_tags, on="athlete_id", how="left", suffixes=("", "_arch"))

# Canonical label preservation first.
work["cluster_id"] = pd.to_numeric(work.get("cluster_id"), errors="coerce").astype("Int64")
work["role_name"] = work.get("role_label")

for rid, ref in role_ref.items():
    mask = work["cluster_id"] == rid
    work.loc[mask & work["role_name"].isna(), "role_name"] = ref.role_name

# Infer only missing roles.
infer_id, infer_name, infer_conf = _infer_missing_roles(work, role_ref)
missing_mask = work["cluster_id"].isna()
work.loc[missing_mask, "cluster_id"] = infer_id.loc[missing_mask]
work.loc[missing_mask, "role_name"] = infer_name.loc[missing_mask]

# Assignment confidence/source.
work["assignment_source"] = np.where(missing_mask, "centroid_inference", "existing_cluster_model")
work["role_confidence"] = pd.to_numeric(work.get("role_confidence"), errors="coerce")
work.loc[missing_mask, "role_confidence"] = infer_conf.loc[missing_mask]
work["role_confidence"] = work["role_confidence"].clip(lower=0.0, upper=1.0)

work["role_code"] = work["cluster_id"].map(lambda x: f"C{int(x)}" if pd.notna(x) else pd.NA)
work["role_one_liner"] = work["cluster_id"].map(
    lambda x: role_ref[int(x)].role_one_liner if pd.notna(x) and int(x) in role_ref else pd.NA
)

# Usage/efficiency tiers (preserve archival values where present).
if "usage_tier" not in work.columns:
    work["usage_tier"] = pd.NA
if "efficiency_tier" not in work.columns:
    work["efficiency_tier"] = pd.NA
if "ball_security_tier" not in work.columns:
    work["ball_security_tier"] = pd.NA
if "role_usage_archetype" not in work.columns:
    work["role_usage_archetype"] = pd.NA

work["usage_tier"] = work["usage_tier"].fillna(pd.to_numeric(work.get("usage_proxy"), errors="coerce").map(_usage_tier))
work["efficiency_tier"] = work["efficiency_tier"].fillna(pd.to_numeric(work.get("ts_pct"), errors="coerce").map(_efficiency_tier))
work["ball_security_tier"] = work["ball_security_tier"].fillna(
    pd.to_numeric(work.get("tov_per40"), errors="coerce").map(_ball_security_tier)
)
work["role_usage_archetype"] = work["role_usage_archetype"].fillna(
    work.apply(lambda r: _role_usage_archetype(r["usage_tier"], r["efficiency_tier"]), axis=1)
)

work["confidence_band"] = work["role_confidence"].map(_confidence_band)

# Keep core fields first for readability.
front_cols = [
    "athlete_id",
    _choose_player_name_col(work),
    "team_id" if "team_id" in work.columns else None,
    _choose_team_col(work),
    "athlete_position_abbreviation" if "athlete_position_abbreviation" in work.columns else None,
    "cluster_id",
    "role_code",
    "role_name",
    "role_one_liner",
    "role_confidence",
    "confidence_band",
    "assignment_source",
    "usage_tier",
    "efficiency_tier",
    "ball_security_tier",
    "role_usage_archetype",
]
front_cols = [c for c in front_cols if c and c in work.columns]
other_cols = [c for c in work.columns if c not in front_cols]
assignments = work[front_cols + other_cols].copy()

assignments.to_csv(ASSIGNMENTS_OUT, index=False)
print(f"Saved assignments: {ASSIGNMENTS_OUT}")

# Cluster summary.
summary = (
    assignments.groupby(["cluster_id", "role_code", "role_name"], dropna=False)
    .agg(
        players=("athlete_id", "nunique"),
        avg_confidence=("role_confidence", "mean"),
        high_confidence_rate=("confidence_band", lambda s: float((s == "High").mean() if len(s) else np.nan)),
        avg_usage_proxy=("usage_proxy", "mean") if "usage_proxy" in assignments.columns else ("athlete_id", "size"),
        avg_ts_pct=("ts_pct", "mean") if "ts_pct" in assignments.columns else ("athlete_id", "size"),
        avg_pts_per40=("pts_per40", "mean") if "pts_per40" in assignments.columns else ("athlete_id", "size"),
        avg_reb_per40=("reb_per40", "mean") if "reb_per40" in assignments.columns else ("athlete_id", "size"),
        avg_ast_per40=("ast_per40", "mean") if "ast_per40" in assignments.columns else ("athlete_id", "size"),
    )
    .reset_index()
)

summary["role_one_liner"] = summary["cluster_id"].map(
    lambda x: role_ref[int(x)].role_one_liner if pd.notna(x) and int(x) in role_ref else pd.NA
)
summary = summary.sort_values("cluster_id")
summary.to_csv(SUMMARY_OUT, index=False)
print(f"Saved role summary: {SUMMARY_OUT}")

# Team role-balance + role-gap recommendations.
team_report, target_profile = _build_team_balance(assignments, role_ref)
team_report = team_report.sort_values(_choose_team_col(team_report)).reset_index(drop=True)
team_report.to_csv(TEAM_BALANCE_OUT, index=False)
print(f"Saved team role-balance report: {TEAM_BALANCE_OUT}")

gap_recs = _build_gap_recommendations(assignments, team_report)
gap_recs.to_csv(GAP_RECS_OUT, index=False)
print(f"Saved role-gap recommendations: {GAP_RECS_OUT}")

# Markdown run summary.
now = datetime.now().strftime("%Y-%m-%d %H:%M")
seeded_n = int(pd.to_numeric(assignments.get("seed"), errors="coerce").notna().sum()) if "seed" in assignments.columns else 0
preserved_n = int((assignments["assignment_source"] == "existing_cluster_model").sum())
inferred_n = int((assignments["assignment_source"] == "centroid_inference").sum())

lines = [
    "# Role Archetypes + Roster Construction Summary",
    "",
    f"Run timestamp: {now}",
    f"Season: {SEASON}",
    "",
    "## Canonical Taxonomy Preservation",
    "",
    "The script preserves the original C0-C5 role IDs and role names from clustering metadata.",
    "Existing archived player assignments are kept as-is; only missing players are inferred.",
    "",
    f"- Players with preserved archival assignment: {preserved_n}",
    f"- Players inferred via centroid matching: {inferred_n}",
    f"- Total players in feature table: {len(assignments)}",
    f"- Players with tournament seed context: {seeded_n}",
    "",
    "## Role Distribution",
    "",
]

for _, r in summary.iterrows():
    role = r.get("role_name")
    players = int(r.get("players", 0)) if pd.notna(r.get("players")) else 0
    conf = r.get("avg_confidence")
    conf_txt = f"{conf:.3f}" if pd.notna(conf) else "NA"
    lines.append(f"- {r.get('role_code')}: {role} — {players} players, avg confidence {conf_txt}")

lines.extend(
    [
        "",
        "## Target Role Mix Baseline",
        "",
        "Baseline profile is computed from top-seeded teams (seed <= 4), with fallback to all seeded teams.",
        "",
    ]
)
for k, v in target_profile.items():
    lines.append(f"- {k.replace('_share', '')}: {v:.3f} share")

lines.extend(
    [
        "",
        "## Output Files",
        "",
        f"- {ASSIGNMENTS_OUT}",
        f"- {SUMMARY_OUT}",
        f"- {TEAM_BALANCE_OUT}",
        f"- {GAP_RECS_OUT}",
    ]
)

NARRATIVE_OUT.write_text("\n".join(lines))
print(f"Saved markdown summary: {NARRATIVE_OUT}")

print("\n✅ Script 10 complete.")
