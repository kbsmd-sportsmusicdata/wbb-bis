"""
enrich_advanced_metrics.py
==========================
Computes eFG%, TS%, TOV%, USG% for all 393 dashboard players
from regular-season (season_type==2) game logs, then writes
.rp (role-peer) and .gp (global) percentiles back into
dashboard_slim_k6_enriched.json under p.dp and p.d.

JOIN KEY: p["aid"] (ESPN athlete_id) — this is the permanent join between
the dashboard JSON and the ESPN parquet files. Every player in the JSON has
an "aid" field set during enrichment (see add_jersey.py / integrate_bio.py).
The parquet's "athlete_id" column maps directly to p["aid"].

USG% formula (per-game weighted):
  USG% = 100 × Σ_g[(p_FGA + 0.44·p_FTA + p_TOV) × (team_MP_g / 5)]
               / Σ_g[p_MP_g × (t_FGA_g + 0.44·t_FTA_g + t_TOV_g)]

eFG%  = (FGM + 0.5 × 3PM) / FGA
TS%   = PTS / (2 × (FGA + 0.44 × FTA))
TOV%  = 100 × TOV / (FGA + 0.44 × FTA + TOV)

Season scope filter scaffold:
  SEASON_TYPES = [2]         → regular season only
  SEASON_TYPES = [2, 3]      → RS + tournament
  SEASON_TYPES = [2]         + CONF_ONLY = True → conference games only
                               (requires player_box_with_game_type.json)
"""

import json, sys
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# ── Paths ────────────────────────────────────────────────────────────────────
PLAYER_BOX       = "/sessions/happy-sharp-wozniak/mnt/uploads/player_box_2026_0322.parquet"
TEAM_BOX         = "/sessions/happy-sharp-wozniak/mnt/uploads/team_box_2026_0322.parquet"
# Tagged JSON (from build_player_box_json.py) — only needed when CONF_ONLY=True
PLAYER_BOX_TYPED = "/sessions/happy-sharp-wozniak/mnt/outputs/player_box_with_game_type.json"
JSON_IN          = "/sessions/happy-sharp-wozniak/dashboard_slim_k6_enriched.json"
JSON_OUT         = "/sessions/happy-sharp-wozniak/dashboard_slim_k6_enriched.json"

# ── Season type config ───────────────────────────────────────────────────────
# season_type values in ESPN WBB data:
#   1 = preseason
#   2 = regular season  (includes conference + non-conference games)
#   3 = postseason / tournament
#
# SEASON_TYPES — controls which season_type values are included in aggregation.
#   [2]       → regular season only (default)
#   [2, 3]    → regular season + tournament
#   [3]       → tournament games only
#
# CONF_ONLY — if True, further filter to conference games only.
#   Requires player_box_with_game_type.json to be present (see build_player_box_json.py).
#   When True, set PLAYER_BOX_TYPED to the path of that JSON.
SEASON_TYPES = [2]    # regular season only — expand to [2,3] to include tournament
CONF_ONLY    = False  # set True + set PLAYER_BOX_TYPED to enable conference-only scope

MIN_GAMES = 5   # minimum games to compute reliable metrics

# ── Load dashboard JSON (needed before parquet to build aid lookup) ──────────
print("Loading dashboard JSON...")
with open(JSON_IN) as f:
    data = json.load(f)

players = data["p"]
print(f"  JSON players: {len(players)}")

# ── Build permanent join key: p["aid"] = ESPN athlete_id ────────────────────
# Every player in the JSON has an "aid" field set by the jersey enrichment step.
# The parquet's "athlete_id" column maps directly to p["aid"].
# Fallback to p["id"] for any player where "aid" was not yet set.
json_ids = {}
no_aid   = []
for p in players:
    key = p.get("aid") or p.get("id")
    if key is not None:
        json_ids[str(int(key))] = p
    if not p.get("aid"):
        no_aid.append(p["n"])

if no_aid:
    print(f"  WARNING: {len(no_aid)} players missing 'aid' — fell back to 'id' key:")
    for nm in no_aid[:5]:
        print(f"    {nm}")

# Confirm clusters present (needed for role percentiles)
clusters = {}
for p in players:
    key = p.get("aid") or p.get("id")
    if key is not None:
        clusters[str(int(key))] = p.get("cl", None)

# ── Load & filter parquet ────────────────────────────────────────────────────
print("\nLoading parquet files...")
pb = pd.read_parquet(PLAYER_BOX)
tb = pd.read_parquet(TEAM_BOX)

# ── Optional: CONF_ONLY scope ────────────────────────────────────────────────
# If CONF_ONLY=True, load the pre-tagged JSON and restrict to conference rows.
# The tagged JSON was produced by build_player_box_json.py and has game_type field.
conf_game_ids = None
if CONF_ONLY:
    import os
    if not os.path.exists(PLAYER_BOX_TYPED):
        raise FileNotFoundError(
            f"CONF_ONLY=True but {PLAYER_BOX_TYPED} not found. "
            "Run build_player_box_json.py first."
        )
    print(f"  CONF_ONLY=True — loading game_type tags from {PLAYER_BOX_TYPED}")
    with open(PLAYER_BOX_TYPED) as f_typed:
        typed = json.load(f_typed)
    conf_game_ids = {
        str(r["game_id"])
        for r in typed["games"]
        if r["game_type"] == "conference"
    }
    print(f"  Conference game_ids loaded: {len(conf_game_ids):,}")

rs_pb = pb[pb["season_type"].isin(SEASON_TYPES)].copy()
rs_tb = tb[tb["season_type"].isin(SEASON_TYPES)].copy()

if CONF_ONLY and conf_game_ids:
    rs_pb = rs_pb[rs_pb["game_id"].astype(str).isin(conf_game_ids)].copy()
    rs_tb = rs_tb[rs_tb["game_id"].astype(str).isin(conf_game_ids)].copy()
    print(f"  After CONF_ONLY filter: {len(rs_pb):,} player rows, {len(rs_tb):,} team rows")
else:
    print(f"  Player box: {len(rs_pb):,} rows")
    print(f"  Team box:   {len(rs_tb):,} rows")

# ── Drop DNP / no athlete_id ─────────────────────────────────────────────────
# NOTE: ESPN's `active` flag is unreliable — it can be False even for players
# who logged real minutes. Use did_not_play==False + minutes>0 as the true
# "participated" filter instead.
rs_pb = rs_pb[
    (rs_pb["did_not_play"] == False) &
    (rs_pb["athlete_id"].notna()) &
    (rs_pb["minutes"].notna()) &
    (rs_pb["minutes"] > 0)
].copy()

rs_pb["athlete_id"] = rs_pb["athlete_id"].astype(int).astype(str)
print(f"  After DNP/minutes filter: {len(rs_pb):,} rows")

# ── Team minutes per game (for USG%) ────────────────────────────────────────
# team_MP = sum of all player minutes per team per game
team_mp = (
    rs_pb.groupby(["game_id", "team_id"])["minutes"]
    .sum()
    .reset_index()
    .rename(columns={"minutes": "team_mp"})
)

# ── Team-level counting stats per game (for USG%) ───────────────────────────
team_counts = rs_tb[[
    "game_id", "team_id",
    "field_goals_attempted",
    "free_throws_attempted",
    "turnovers"
]].copy().rename(columns={
    "field_goals_attempted": "t_fga",
    "free_throws_attempted": "t_fta",
    "turnovers":             "t_tov"
})

# ── Merge team context into player rows ─────────────────────────────────────
rs_pb["team_id_int"] = rs_pb["team_id"].astype(int)
rs_pb = rs_pb.merge(
    team_mp.rename(columns={"team_id": "team_id_int"}),
    on=["game_id", "team_id_int"], how="left"
)
rs_pb = rs_pb.merge(
    team_counts.rename(columns={"team_id": "team_id_int"}),
    on=["game_id", "team_id_int"], how="left"
)

missing_team = rs_pb["team_mp"].isna().sum()
if missing_team > 0:
    print(f"  WARNING: {missing_team} rows missing team context — filling with 200")
    rs_pb["team_mp"]  = rs_pb["team_mp"].fillna(200)
    rs_pb["t_fga"]    = rs_pb["t_fga"].fillna(rs_pb["field_goals_attempted"].median())
    rs_pb["t_fta"]    = rs_pb["t_fta"].fillna(rs_pb["free_throws_attempted"].median())
    rs_pb["t_tov"]    = rs_pb["t_tov"].fillna(rs_pb["turnovers"].median())

# ── Coerce counting columns to numeric ───────────────────────────────────────
count_cols = [
    "minutes", "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "points", "turnovers",
    "team_mp", "t_fga", "t_fta", "t_tov"
]
for col in count_cols:
    rs_pb[col] = pd.to_numeric(rs_pb[col], errors="coerce").fillna(0)

# ── USG% per-game components ─────────────────────────────────────────────────
# numerator per game: (p_FGA + 0.44·p_FTA + p_TOV) × (team_MP / 5)
rs_pb["usg_num"] = (
    (rs_pb["field_goals_attempted"] + 0.44 * rs_pb["free_throws_attempted"] + rs_pb["turnovers"])
    * (rs_pb["team_mp"] / 5)
)
# denominator per game: p_MP × (t_FGA + 0.44·t_FTA + t_TOV)
rs_pb["usg_den"] = (
    rs_pb["minutes"]
    * (rs_pb["t_fga"] + 0.44 * rs_pb["t_fta"] + rs_pb["t_tov"])
)

# ── Aggregate per player ─────────────────────────────────────────────────────
agg = rs_pb.groupby("athlete_id").agg(
    games            = ("game_id",                             "nunique"),
    min_total        = ("minutes",                              "sum"),
    fgm              = ("field_goals_made",                     "sum"),
    fga              = ("field_goals_attempted",                "sum"),
    three_pm         = ("three_point_field_goals_made",         "sum"),
    three_pa         = ("three_point_field_goals_attempted",    "sum"),
    ftm              = ("free_throws_made",                     "sum"),
    fta              = ("free_throws_attempted",                "sum"),
    pts              = ("points",                               "sum"),
    tov              = ("turnovers",                            "sum"),
    usg_num          = ("usg_num",                              "sum"),
    usg_den          = ("usg_den",                              "sum"),
).reset_index()

scope_label = "conference" if CONF_ONLY else ("RS+" + "+".join(str(s) for s in SEASON_TYPES))
print(f"\nAggregated {len(agg)} unique players ({scope_label} games)")

# ── Compute advanced metrics ─────────────────────────────────────────────────
def safe_div(num, den, default=np.nan):
    """Divide arrays safely, returning default where den==0."""
    result = np.where(den > 0, num / den, default)
    return result

agg["efg_pct"] = safe_div(
    agg["fgm"] + 0.5 * agg["three_pm"],
    agg["fga"]
)
agg["ts_pct"] = safe_div(
    agg["pts"],
    2 * (agg["fga"] + 0.44 * agg["fta"])
)
agg["tov_pct"] = 100 * safe_div(
    agg["tov"],
    agg["fga"] + 0.44 * agg["fta"] + agg["tov"]
)
agg["usg_pct"] = 100 * safe_div(
    agg["usg_num"],
    agg["usg_den"]
)

# Clamp to reasonable ranges
agg["efg_pct"] = agg["efg_pct"].clip(0, 1)
agg["ts_pct"]  = agg["ts_pct"].clip(0, 1)
agg["tov_pct"] = agg["tov_pct"].clip(0, 100)
agg["usg_pct"] = agg["usg_pct"].clip(0, 100)

# Filter to players with enough games
agg_valid = agg[agg["games"] >= MIN_GAMES].copy()
print(f"Players with ≥{MIN_GAMES} games: {len(agg_valid)}")

# ── Filter to only our 393 JSON players (join on aid = ESPN athlete_id) ──────
agg_valid = agg_valid[agg_valid["athlete_id"].isin(json_ids.keys())].copy()
print(f"JSON players with metrics (joined on aid): {len(agg_valid)}")

# ── Build cluster map for role percentiles ───────────────────────────────────
# Map ESPN athlete_id (aid) → cluster number
agg_valid["cluster"] = agg_valid["athlete_id"].map(clusters)

NEW_METRICS = ["efg_pct", "ts_pct", "tov_pct", "usg_pct"]
METRIC_DISPLAY = {
    "efg_pct": "eFG%",
    "ts_pct":  "TS%",
    "tov_pct": "TOV%",
    "usg_pct": "USG%",
}

# ── Compute global percentiles ───────────────────────────────────────────────
# Higher is better for eFG%, TS%, USG%; LOWER is better for TOV%
# Standard: percentileofscore gives pct of scores < x (kind='rank' = proportion ≤ x)
# For TOV%: invert (100 - pct) so higher percentile = better ball security

for metric in NEW_METRICS:
    col_gp = f"{metric}_gp"
    values = agg_valid[metric].dropna().values

    def gp_score(x, vals=values, metric=metric):
        if np.isnan(x):
            return np.nan
        pct = percentileofscore(vals, x, kind='rank')
        # For TOV%: lower is better → invert
        if metric == "tov_pct":
            pct = 100 - pct
        return round(pct, 1)

    agg_valid[col_gp] = agg_valid[metric].apply(gp_score)

# ── Compute role (cluster-peer) percentiles ──────────────────────────────────
for metric in NEW_METRICS:
    col_rp = f"{metric}_rp"
    agg_valid[col_rp] = np.nan

    for cluster_id in agg_valid["cluster"].unique():
        if cluster_id is None:
            continue
        mask = agg_valid["cluster"] == cluster_id
        group_vals = agg_valid.loc[mask, metric].dropna().values

        def rp_score(x, vals=group_vals, metric=metric):
            if np.isnan(x):
                return np.nan
            pct = percentileofscore(vals, x, kind='rank')
            if metric == "tov_pct":
                pct = 100 - pct
            return round(pct, 1)

        agg_valid.loc[mask, col_rp] = agg_valid.loc[mask, metric].apply(rp_score)

# ── Write metrics back into JSON ─────────────────────────────────────────────
# Join key: row["athlete_id"] == p["aid"] (ESPN athlete_id)
print("\nWriting metrics back to JSON (join key: aid = ESPN athlete_id)...")

matched = 0
unmatched = 0

for _, row in agg_valid.iterrows():
    pid = row["athlete_id"]   # this is the ESPN athlete_id (= p["aid"])
    if pid not in json_ids:
        unmatched += 1
        continue

    p = json_ids[pid]
    matched += 1

    # Ensure p.d and p.dp exist
    if "d" not in p:
        p["d"] = {}
    if "dp" not in p:
        p["dp"] = {}

    # Raw values → p.d
    p["d"]["eFG_pct"]  = round(float(row["efg_pct"]), 4) if not np.isnan(row["efg_pct"]) else None
    p["d"]["TS_pct"]   = round(float(row["ts_pct"]),  4) if not np.isnan(row["ts_pct"])  else None
    p["d"]["TOV_pct"]  = round(float(row["tov_pct"]), 2) if not np.isnan(row["tov_pct"]) else None
    p["d"]["USG_pct"]  = round(float(row["usg_pct"]), 2) if not np.isnan(row["usg_pct"]) else None

    # Percentiles → p.dp
    for metric, dp_key in [
        ("efg_pct", "eFG_pct"),
        ("ts_pct",  "TS_pct"),
        ("tov_pct", "TOV_pct"),
        ("usg_pct", "USG_pct"),
    ]:
        rp = row.get(f"{metric}_rp", np.nan)
        gp = row.get(f"{metric}_gp", np.nan)
        p["dp"][dp_key] = {
            "rp": round(float(rp), 1) if not np.isnan(rp) else None,
            "gp": round(float(gp), 1) if not np.isnan(gp) else None,
        }

print(f"  Matched: {matched} | Unmatched: {unmatched}")

# ── Players with no parquet data (too few games or missing) ─────────────────
# Note: json_ids keys are ESPN athlete_ids (= p["aid"] for all players)
all_json_ids = set(json_ids.keys())
enriched_ids = set(agg_valid["athlete_id"].values)
not_enriched = all_json_ids - enriched_ids
if not_enriched:
    print(f"\n  {len(not_enriched)} JSON players had <{MIN_GAMES} RS games — metrics set to None:")
    for pid in sorted(not_enriched)[:10]:
        p = json_ids[pid]
        print(f"    {p['n']} ({p.get('tm','?')})")
        # Still set null entries so JS doesn't break
        if "d" not in p:
            p["d"] = {}
        if "dp" not in p:
            p["dp"] = {}
        for key in ["eFG_pct", "TS_pct", "TOV_pct", "USG_pct"]:
            p["d"][key] = None
            p["dp"][key] = {"rp": None, "gp": None}

# ── Save updated JSON ─────────────────────────────────────────────────────────
print(f"\nSaving updated JSON to {JSON_OUT}...")
with open(JSON_OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print("Done!")

# ── Spot-check report ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SPOT-CHECK: 8 sample players")
print("="*60)

sample_ids = list(enriched_ids)[:8]
for pid in sample_ids:
    p = json_ids[pid]
    row = agg_valid[agg_valid["athlete_id"]==pid].iloc[0]
    dp = p["dp"]
    print(f"\n{p['n']} ({p.get('tm','?')}) | Cluster {p.get('cl','?')} | Games: {int(row['games'])}")
    print(f"  eFG%={p['d']['eFG_pct']:.3f}  rp={dp.get('eFG_pct',{}).get('rp')}  gp={dp.get('eFG_pct',{}).get('gp')}")
    print(f"  TS% ={p['d']['TS_pct']:.3f}  rp={dp.get('TS_pct',{}).get('rp')}  gp={dp.get('TS_pct',{}).get('gp')}")
    print(f"  TOV%={p['d']['TOV_pct']:.2f}  rp={dp.get('TOV_pct',{}).get('rp')}  gp={dp.get('TOV_pct',{}).get('gp')}")
    print(f"  USG%={p['d']['USG_pct']:.2f}  rp={dp.get('USG_pct',{}).get('rp')}  gp={dp.get('USG_pct',{}).get('gp')}")

print("\n" + "="*60)
print("Summary stats for new metrics (across 393 players):")
for m in NEW_METRICS:
    valid = agg_valid[m].dropna()
    print(f"  {m}: min={valid.min():.3f}  median={valid.median():.3f}  max={valid.max():.3f}  n={len(valid)}")
