"""
gen_k6_enriched.py — Enrich dashboard_slim_k6.json with:
  Phase 1 : Core derived metrics + percentiles
  Phase 2 : Strength flags  (boolean, role-peer percentiles)
  Phase 3 : Weakness flags  (boolean, role-peer percentiles)
  Tags    : Player impact tags (Groups 1–3)

Outputs: dashboard_slim_k6_enriched.json
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── 1. Load model & data ──────────────────────────────────────────────────────
print("Loading model + data...")
with open('model_artifacts_k6c.pkl', 'rb') as f:
    arts6      = pickle.load(f)
scaler6    = arts6['scaler6']
gmm6c      = arts6['gmm6c']
FEATURES_6 = arts6['FEATURES_6']

with open('dashboard_slim_k6.json') as f:
    slim_data    = json.load(f)
slim_players = slim_data['p']
player_by_id = {p['id']: p for p in slim_players}
PLAYER_IDS   = set(player_by_id.keys())
print(f"  Slim players: {len(slim_players)}")

df       = pd.read_parquet('df_2026_clustered.parquet')
df_qual  = df[df['cluster'] != 5].copy()                      # exclude zero-FGM games
df_qual  = df_qual[df_qual['athlete_id'].isin(PLAYER_IDS)]
df_qual  = df_qual.dropna(subset=FEATURES_6)
print(f"  Qualifying game rows: {len(df_qual):,}")

# ── 2. Pre-compute per-game derived columns ───────────────────────────────────
print("Computing per-game derived columns...")
df_qual = df_qual.copy()
df_qual['twoPM_per40'] = df_qual['field_goals_made_per40'] - df_qual['three_point_field_goals_made_per40']
df_qual['twoPA_per40'] = df_qual['field_goals_attempted_per40'] - df_qual['three_point_field_goals_attempted_per40']
df_qual['stocks_per40'] = df_qual['steals_per40'] + df_qual['blocks_per40']

# ── 3. Player-level aggregation ───────────────────────────────────────────────
print("Aggregating player-level stats...")

# Season totals (for ratio/pct metrics calculated from totals, not means)
totals = df_qual.groupby('athlete_id').agg(
    n_qual_games  = ('field_goals_made', 'count'),
    FGM_sum       = ('field_goals_made', 'sum'),
    FGA_sum       = ('field_goals_attempted', 'sum'),
    thrPM_sum     = ('three_point_field_goals_made', 'sum'),
    thrPA_sum     = ('three_point_field_goals_attempted', 'sum'),
    FTM_sum       = ('free_throws_made', 'sum'),
    FTA_sum       = ('free_throws_attempted', 'sum'),
).reset_index()

totals['twoPM_sum'] = totals['FGM_sum'] - totals['thrPM_sum']
totals['twoPA_sum'] = totals['FGA_sum'] - totals['thrPA_sum']
totals['FT_pct']    = totals['FTM_sum'] / totals['FTA_sum'].clip(lower=1)
totals['twoPT_pct'] = totals['twoPM_sum'] / totals['twoPA_sum'].clip(lower=1)
totals['fg_pct_tot'] = totals['FGM_sum'] / totals['FGA_sum'].clip(lower=1)

# Per-40 means
per40 = df_qual.groupby('athlete_id').agg(
    FGM_per40      = ('field_goals_made_per40', 'mean'),
    FGA_per40      = ('field_goals_attempted_per40', 'mean'),
    FTM_per40      = ('free_throws_made_per40', 'mean'),
    FTA_per40      = ('free_throws_attempted_per40', 'mean'),
    twoPM_per40    = ('twoPM_per40', 'mean'),
    twoPA_per40    = ('twoPA_per40', 'mean'),
    thrPA_per40    = ('three_point_field_goals_attempted_per40', 'mean'),
    stocks_per40   = ('stocks_per40', 'mean'),
    ast_mean       = ('assists_per40', 'mean'),
    to_mean        = ('turnovers_per40', 'mean'),
    pts_mean       = ('points_per40', 'mean'),
    pts_std        = ('points_per40', 'std'),
).reset_index()

# Shot-profile shares (from totals)
totals['threePA_share'] = totals['thrPA_sum'] / totals['FGA_sum'].clip(lower=1)
totals['twoPA_share']   = totals['twoPA_sum'] / totals['FGA_sum'].clip(lower=1)

# Merge
agg = per40.merge(
    totals[['athlete_id','FT_pct','twoPT_pct','fg_pct_tot','threePA_share','twoPA_share',
            'FTM_sum','FTA_sum','thrPA_sum','FGA_sum']],
    on='athlete_id', how='left'
)

# Derived: ast/to ratio, scoring CV
agg['ast_tov_ratio']       = agg['ast_mean'] / agg['to_mean'].clip(lower=0.1)
agg['scoring_volatility']  = agg['pts_std'] / agg['pts_mean'].clip(lower=0.1)

# ── 4. Role-confidence from k=6 GMM ──────────────────────────────────────────
print("Computing k=6 role confidence (GMM max probability)...")
# Map FEATURES_6 to column names in agg — these were aggregated as means from df_qual
# FEATURES_6 columns present in agg via explicit per40 aggregation
FEAT6_TO_AGG = {
    'points_per40':                           'pts_mean',
    'rebounds_per40':                         None,       # not in agg directly
    'assists_per40':                          'ast_mean',
    'steals_per40':                           None,
    'blocks_per40':                           None,
    'turnovers_per40':                        'to_mean',
    'field_goals_made_per40':                 'FGM_per40',
    'field_goals_attempted_per40':            'FGA_per40',
    'three_point_field_goals_made_per40':     None,       # twoPM derived but not 3PM directly
    'three_point_field_goals_attempted_per40':'thrPA_per40',
    'free_throws_made_per40':                 'FTM_per40',
    'free_throws_attempted_per40':            'FTA_per40',
    'offensive_rebounds_per40':               None,
    'defensive_rebounds_per40':               None,
    'fouls_per40':                            None,
}
# For missing ones, re-aggregate from df_qual
extra_cols = ['rebounds_per40','steals_per40','blocks_per40',
              'three_point_field_goals_made_per40','offensive_rebounds_per40',
              'defensive_rebounds_per40','fouls_per40']
extra_agg = df_qual.groupby('athlete_id')[extra_cols].mean().reset_index()
agg = agg.merge(extra_agg, on='athlete_id', how='left')

# Now build feature matrix in FEATURES_6 order from agg
F6_col_map = {
    'points_per40':                           'pts_mean',
    'rebounds_per40':                         'rebounds_per40',
    'assists_per40':                          'ast_mean',
    'steals_per40':                           'steals_per40',
    'blocks_per40':                           'blocks_per40',
    'turnovers_per40':                        'to_mean',
    'field_goals_made_per40':                 'FGM_per40',
    'field_goals_attempted_per40':            'FGA_per40',
    'three_point_field_goals_made_per40':     'three_point_field_goals_made_per40',
    'three_point_field_goals_attempted_per40':'thrPA_per40',
    'free_throws_made_per40':                 'FTM_per40',
    'free_throws_attempted_per40':            'FTA_per40',
    'offensive_rebounds_per40':               'offensive_rebounds_per40',
    'defensive_rebounds_per40':               'defensive_rebounds_per40',
    'fouls_per40':                            'fouls_per40',
}

X_f6 = agg[[F6_col_map[f] for f in FEATURES_6]].values.astype(float)
valid_mask = ~np.any(np.isnan(X_f6), axis=1)
print(f"  Valid rows for GMM scoring: {valid_mask.sum()} / {len(X_f6)}")

role_conf_arr = np.full(len(X_f6), np.nan)
if valid_mask.sum() > 0:
    X_scaled  = scaler6.transform(X_f6[valid_mask])
    proba_mat = gmm6c.predict_proba(X_scaled)
    max_proba = proba_mat.max(axis=1)
    role_conf_arr[valid_mask] = max_proba

agg['role_confidence'] = role_conf_arr
valid_rc = agg['role_confidence'].dropna()
print(f"  Mean role confidence: {valid_rc.mean():.3f}  "
      f"(min {valid_rc.min():.3f}, max {valid_rc.max():.3f})")

# ── 5. Attach dominant cluster from slim ─────────────────────────────────────
agg['cl6'] = agg['athlete_id'].map(lambda x: player_by_id.get(int(x), {}).get('cl', -1))
agg['c5_pct'] = agg['athlete_id'].map(lambda x: player_by_id.get(int(x), {}).get('c5', 0.0))
print(f"  Cluster distribution: {agg['cl6'].value_counts().sort_index().to_dict()}")

# ── 6. Percentile helpers ─────────────────────────────────────────────────────
def compute_pctiles(df_in, stat_cols):
    """Return dict of {stat: Series} for global percentile ranks (0-100)."""
    global_p = {}
    for s in stat_cols:
        if s not in df_in.columns:
            continue
        vals = df_in[s]
        global_p[s] = (vals.rank(method='average') - 1) / max(len(vals) - 1, 1) * 100
    return global_p

def compute_role_pctiles(df_in, stat_cols):
    """Return dict of {stat: Series} for role-peer percentile ranks, per cl6 cluster."""
    role_p = {s: pd.Series(index=df_in.index, dtype=float) for s in stat_cols if s in df_in.columns}
    for cl in df_in['cl6'].unique():
        mask = df_in['cl6'] == cl
        if mask.sum() < 2:
            for s in role_p:
                role_p[s][mask] = np.nan
            continue
        peer = df_in[mask]
        for s in role_p:
            ranked = (peer[s].rank(method='average') - 1) / max(len(peer) - 1, 1) * 100
            role_p[s][mask] = ranked
    return role_p

DERIVED_STATS = [
    'ast_tov_ratio', 'stocks_per40', 'role_confidence',
    'FGM_per40', 'FGA_per40', 'twoPM_per40', 'twoPA_per40', 'twoPT_pct',
    'FTM_per40', 'FTA_per40', 'FT_pct',
    'threePA_share', 'twoPA_share', 'scoring_volatility'
]

print("Computing derived metric percentiles...")
g_pctile = compute_pctiles(agg, DERIVED_STATS)
r_pctile = compute_role_pctiles(agg, DERIVED_STATS)

# Also need existing stats for flag/tag logic (pull from slim's p dict — rp/gp already there)
# We build a lookup: {athlete_id: {stat: {'rp': val, 'gp': val}}}
# For NEW derived stats only — existing stats come from slim['p']

# ── 7. Build per-player derived percentile dict ───────────────────────────────
def gp_val(stat, idx):
    return round(float(g_pctile[stat].iloc[idx]), 1) if stat in g_pctile else None

def rp_val(stat, idx):
    return round(float(r_pctile[stat].iloc[idx]), 1) if stat in r_pctile else None

# ── 8. Flag / tag helpers using pre-built lookups ────────────────────────────
def get_rp(p_dict, new_rp, stat):
    """Get role-peer percentile: prefer existing slim p_dict, fallback to new_rp."""
    if stat in p_dict:
        return p_dict[stat]['rp']
    return new_rp.get(stat, 50.0)  # fallback 50

def get_gp(p_dict, new_gp, stat):
    if stat in p_dict:
        return p_dict[stat]['gp']
    return new_gp.get(stat, 50.0)

# ── 9. Main enrichment loop ───────────────────────────────────────────────────
print("Building enriched player records...")

agg_by_id = {int(r['athlete_id']): (i, r) for i, (_, r) in enumerate(agg.iterrows())}

enriched_players = []

for p in slim_players:
    aid  = p['id']
    p_dict = p['p']         # existing role-peer + global percentiles
    s_dict = p['s']         # existing raw stat means

    if aid not in agg_by_id:
        # Player missing from agg — attach empty enrichment
        p_new = dict(p)
        p_new['d'] = {}
        p_new['fl'] = {}
        p_new['tg'] = {}
        enriched_players.append(p_new)
        continue

    idx, row = agg_by_id[aid]

    # ── Phase 1: derived metric values ────────────────────────────────────────
    d = {
        'ast_tov_ratio':      round(float(row['ast_tov_ratio']),   3),
        'stocks_per40':       round(float(row['stocks_per40']),     3),
        'role_confidence':    round(float(row['role_confidence']),  4) if pd.notna(row['role_confidence']) else None,
        'FGM_per40':          round(float(row['FGM_per40']),        3),
        'FGA_per40':          round(float(row['FGA_per40']),        3),
        'twoPM_per40':        round(float(row['twoPM_per40']),      3),
        'twoPA_per40':        round(float(row['twoPA_per40']),      3),
        'twoPT_pct':          round(float(row['twoPT_pct']),        4),
        'FTM_per40':          round(float(row['FTM_per40']),        3),
        'FTA_per40':          round(float(row['FTA_per40']),        3),
        'FT_pct':             round(float(row['FT_pct']),           4),
        'threePA_share':      round(float(row['threePA_share']),    4),
        'twoPA_share':        round(float(row['twoPA_share']),      4),
        'scoring_volatility': round(float(row['scoring_volatility']), 4),
    }

    # Derived percentiles for new stats
    new_rp = {s: rp_val(s, idx) for s in DERIVED_STATS}
    new_gp = {s: gp_val(s, idx) for s in DERIVED_STATS}

    # Attach derived percentiles to d dict  (rp/gp sub-dict like existing p)
    d_pctile = {s: {'rp': new_rp[s], 'gp': new_gp[s]} for s in DERIVED_STATS
                if new_rp[s] is not None}

    # ── Phase 2: Strength flags ────────────────────────────────────────────────
    # c5_pct: lower is better → 100-rp for consistency
    c5_rp_raw = r_pctile['scoring_volatility'].iloc[idx] if 'scoring_volatility' in r_pctile else 50.0
    # c5_pct role-peer — compute separately (it's on p not new derived)
    # We use existing p_dict which has turnovers_per40 (inverted) and others
    # For c5_pct we build it from the slim c5 value and the cluster

    # Quick c5 role-peer pctile: rank within same cluster
    # (We'll compute this outside loop once; for now approximate from global rank of c5_pct)
    # We pre-compute c5 role percentiles below and inject them — placeholder for now

    fl = {
        # Strength flags (1=strength present)
        'strength_scoring':          int(get_rp(p_dict, new_rp, 'points_per40') >= 85),
        'strength_shooting':         int(get_rp(p_dict, new_rp, 'fg_pct') >= 75 and
                                         get_rp(p_dict, new_rp, 'three_pt_pct') >= 75),
        'strength_spacing':          int(get_rp(p_dict, new_rp, 'three_point_field_goals_attempted_per40') >= 85 and
                                         get_rp(p_dict, new_rp, 'three_pt_pct') >= 75),
        'strength_playmaking':       int(get_rp(p_dict, new_rp, 'assists_per40') >= 85),
        'strength_ball_security':    int(get_rp(p_dict, new_rp, 'ast_tov_ratio') >= 70 and
                                         # turnovers already inverted in slim p_dict (lower TO = higher rp)
                                         get_rp(p_dict, new_rp, 'turnovers_per40') >= 60),
        'strength_defense_events':   int(get_rp(p_dict, new_rp, 'stocks_per40') >= 85),
        'strength_perimeter_defense':int(get_rp(p_dict, new_rp, 'steals_per40') >= 85),
        'strength_rim_protection':   int(get_rp(p_dict, new_rp, 'blocks_per40') >= 85),
        'strength_efficiency':       int(get_rp(p_dict, new_rp, 'fg_pct') >= 80),
        # consistency: low c5_pct = good; use scoring_volatility rp (high rp = high volatility = bad)
        # strength = low volatility (rp ≤ 30th means low volatility relative to peers = more consistent)
        'strength_consistency':      0,   # filled in below after c5 role pctile computation

        # Weakness flags (1=weakness present)
        'weak_scoring':              int(get_rp(p_dict, new_rp, 'points_per40') <= 20),
        'weak_efficiency':           int(get_rp(p_dict, new_rp, 'fg_pct') <= 25),
        'weak_shooting':             int(get_rp(p_dict, new_rp, 'three_pt_pct') <= 25),
        'weak_playmaking':           int(get_rp(p_dict, new_rp, 'assists_per40') <= 20),
        'weak_ball_security':        int(get_rp(p_dict, new_rp, 'ast_tov_ratio') <= 25 and
                                         # turnovers inverted in slim: low rp = high TO
                                         get_rp(p_dict, new_rp, 'turnovers_per40') <= 25),
        'weak_rebounding':           int(get_rp(p_dict, new_rp, 'rebounds_per40') <= 20),
        'weak_defense_events':       int(get_rp(p_dict, new_rp, 'stocks_per40') <= 20),
        'weak_rim_protection':       int(get_rp(p_dict, new_rp, 'blocks_per40') <= 20),
        'weak_perimeter_defense':    int(get_rp(p_dict, new_rp, 'steals_per40') <= 20),
        'weak_free_throw_pressure':  int(get_rp(p_dict, new_rp, 'FTA_per40') <= 20),
        'weak_consistency':          0,   # filled in below
        'weak_high_variance':        int(new_rp.get('scoring_volatility', 50) >= 75),
    }

    # ── Player impact tags ─────────────────────────────────────────────────────
    rp = lambda stat: get_rp(p_dict, new_rp, stat)
    gp = lambda stat: get_gp(p_dict, new_gp, stat)

    tg = {
        # ── Group 1: role peer % ──
        'tag_high_volume_scorer':      int(rp('points_per40') >= 80),
        'tag_elite_shooter':           int(rp('fg_pct') >= 80 and rp('three_pt_pct') >= 80),
        'tag_floor_spacer':            int(rp('three_point_field_goals_attempted_per40') >= 80 and
                                           rp('three_pt_pct') >= 60),
        'tag_three_level_scorer':      int(rp('points_per40') >= 70 and
                                           rp('assists_per40') >= 70 and
                                           rp('rebounds_per40') >= 70),
        # FGA role-peer via FGA_per40 derived (field_goals_attempted_per40 excluded from slim p)
        'tag_low_efficiency_volume':   int(new_rp.get('FGA_per40', 50) >= 75 and
                                           rp('fg_pct') <= 40),
        'tag_primary_initiator':       int(rp('assists_per40') >= 80 and
                                           # turnovers inverted → high rp means low TO
                                           rp('turnovers_per40') >= 50),
        'tag_secondary_creator':       int(rp('assists_per40') >= 55 and rp('assists_per40') < 80),
        'tag_low_turnover_connector':  int(rp('turnovers_per40') >= 65 and rp('assists_per40') >= 50),
        'tag_high_turnover_risk':      int(rp('turnovers_per40') <= 20),   # inverted → low rp = high TO
        'tag_event_creator':           int(rp('stocks_per40') >= 75),
        'tag_point_of_attack_defender':int(rp('steals_per40') >= 75),
        'tag_rim_protector':           int(rp('blocks_per40') >= 80),
        'tag_low_event_defender':      int(rp('stocks_per40') <= 25),

        # ── Group 2: global % ──
        # NOTE: field_goals_attempted_per40 is excluded from slim p percentiles,
        # so use FGA_per40 derived global percentile instead (same value, separate compute)
        'tag_high_usage_engine':       int(new_gp.get('FGA_per40', 50) >= 85),
        'tag_low_usage_efficient':     int(new_gp.get('FGA_per40', 50) <= 20 and
                                           gp('fg_pct') >= 60),
        'tag_microwave_scorer':        int(gp('points_per40') >= 75 and
                                           new_gp.get('scoring_volatility', 50) >= 65),
        'tag_stable_producer':         int(new_gp.get('scoring_volatility', 50) <= 30 and
                                           gp('points_per40') >= 50),

        # ── Group 3: raw/derived thresholds (no %ile) ──
        'tag_floor_spacer_raw':        int(d['threePA_share'] >= 0.38),
        'tag_secondary_creator_raw':   int(d['ast_tov_ratio'] >= 1.5 and
                                           p['s']['assists_per40'] >= 3.0 and
                                           p['s']['assists_per40'] < 6.5),
        'tag_rim_finisher_raw':        int(d['twoPM_per40'] >= 5.5 and d['twoPA_share'] >= 0.62),
        'tag_microwave_scorer_raw':    int(d['scoring_volatility'] >= 0.52 and
                                           p['s']['points_per40'] >= 14.0),
        'tag_low_usage_connector_raw': int(d['FGA_per40'] <= 8.0 and
                                           p['s']['assists_per40'] >= 2.5),
    }

    # ── Assemble enriched player ───────────────────────────────────────────────
    p_new = dict(p)
    p_new['d'] = d
    p_new['dp'] = d_pctile     # derived percentiles (rp/gp)
    p_new['fl'] = fl
    p_new['tg'] = tg
    enriched_players.append(p_new)

# ── 10. Post-loop: fill in consistency flags using c5_pct role-peer pctile ────
print("Computing c5 and scoring_volatility role-peer percentiles for consistency flags...")

# Build mapping: athlete_id → cluster + c5_pct + scoring_volatility
c5_map   = {p['id']: p['c5'] for p in slim_players}
cl_map   = {p['id']: p['cl'] for p in slim_players}
sv_map   = {int(r['athlete_id']): float(r['scoring_volatility'])
            for _, r in agg.iterrows()}

# Compute c5 role-peer percentiles
c5_series = pd.Series(c5_map)
cl_series = pd.Series(cl_map)
c5_role_pctile = {}
for cl in range(6):
    peers = c5_series[cl_series == cl]
    if len(peers) < 2:
        for aid in peers.index:
            c5_role_pctile[aid] = 50.0
        continue
    ranked = (peers.rank(method='average') - 1) / max(len(peers) - 1, 1) * 100
    for aid, pct in ranked.items():
        c5_role_pctile[aid] = float(pct)

# Compute scoring_volatility role-peer percentiles
sv_series = pd.Series(sv_map)
sv_role_pctile = {}
for cl in range(6):
    peer_ids = [aid for aid, c in cl_map.items() if c == cl and aid in sv_map]
    if len(peer_ids) < 2:
        for aid in peer_ids:
            sv_role_pctile[aid] = 50.0
        continue
    peer_sv = sv_series[peer_ids]
    ranked  = (peer_sv.rank(method='average') - 1) / max(len(peer_sv) - 1, 1) * 100
    for aid, pct in ranked.items():
        sv_role_pctile[aid] = float(pct)

# Now patch consistency flags and weak_high_variance in enriched players
for p_new in enriched_players:
    aid = p_new['id']
    if 'fl' not in p_new:
        continue
    c5_rp  = c5_role_pctile.get(aid, 50.0)
    sv_rp  = sv_role_pctile.get(aid, 50.0)

    # strength_consistency: low c5_pct AND low volatility relative to peers
    p_new['fl']['strength_consistency'] = int(c5_rp <= 30 and sv_rp <= 30)
    # weak_consistency: high c5_pct OR high volatility
    p_new['fl']['weak_consistency']     = int(c5_rp >= 70 or sv_rp >= 75)

    # Update weak_high_variance with proper sv_rp
    p_new['fl']['weak_high_variance']   = int(sv_rp >= 75)

    # Also patch tag_stable_producer: needs low sv_rp
    if 'tg' in p_new:
        existing_sv_gp = 0  # fallback — we used gp above; sv_rp is role-peer here
        # recompute stable_producer with sv_rp
        sv_gp = 100 - sv_rp   # invert: low volatility → high "stability" rank
        gp_pts = p_new['p'].get('points_per40', {}).get('gp', 50)
        p_new['tg']['tag_stable_producer'] = int(sv_gp >= 70 and gp_pts >= 50)

# ── 11. Write output ──────────────────────────────────────────────────────────
print("\nWriting enriched output...")
slim_data_out = dict(slim_data)
slim_data_out['p'] = enriched_players

with open('dashboard_slim_k6_enriched.json', 'w') as f:
    json.dump(slim_data_out, f, separators=(',', ':'))

size_kb = len(json.dumps(slim_data_out, separators=(',',':')).encode()) / 1024
print(f"  Wrote dashboard_slim_k6_enriched.json  ({len(enriched_players)} players, {size_kb:.0f} KB)")

# ── 12. Validation ────────────────────────────────────────────────────────────
print("\n── Validation ────────────────────────────────────────────────────────────")

# Count flags distribution
fl_sums = {}
tg_sums = {}
for p_new in enriched_players:
    for k, v in p_new.get('fl', {}).items():
        fl_sums[k] = fl_sums.get(k, 0) + v
    for k, v in p_new.get('tg', {}).items():
        tg_sums[k] = tg_sums.get(k, 0) + v

n = len(enriched_players)
print(f"\nStrength flags  (n={n} players):")
for k in sorted(fl_sums):
    if k.startswith('strength'):
        pct = 100 * fl_sums[k] / n
        print(f"  {k:<35}: {fl_sums[k]:>4}  ({pct:.1f}%)")

print(f"\nWeakness flags:")
for k in sorted(fl_sums):
    if k.startswith('weak'):
        pct = 100 * fl_sums[k] / n
        print(f"  {k:<35}: {fl_sums[k]:>4}  ({pct:.1f}%)")

print(f"\nImpact tags:")
for k in sorted(tg_sums):
    pct = 100 * tg_sums[k] / n
    print(f"  {k:<40}: {tg_sums[k]:>4}  ({pct:.1f}%)")

print("\nSpot-check (derived + flags):")
SPOT = ['Lauren Betts','Paige Bueckers','Hannah Hidalgo','Raegan Beers',
        'Madison Booker','Sarah Strong','Olivia Miles','Joyce Edwards']
for name in SPOT:
    last = name.split()[-1]
    matches = [p for p in enriched_players if last in p['n']]
    for p in matches[:1]:
        tags_on  = [k for k, v in p.get('tg', {}).items() if v]
        fl_str   = [k.replace('strength_','✓').replace('weak_','✗') for k, v in p.get('fl', {}).items() if v]
        sv       = p.get('d', {}).get('scoring_volatility', '?')
        rc       = p.get('d', {}).get('role_confidence', '?')
        ast_to   = p.get('d', {}).get('ast_tov_ratio', '?')
        print(f"\n  {p['n']:<26} {p['tm']:<18} C{p['cl']}·{p['r']}")
        print(f"    rc={rc}  sv={sv}  ast/to={ast_to}")
        print(f"    flags: {', '.join(fl_str) or '(none)'}")
        print(f"    tags:  {', '.join(tags_on) or '(none)'}")

print("\nDONE.")
