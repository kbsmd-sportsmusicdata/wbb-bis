"""
integrate_bio.py
─────────────────
Merges ESPN bio data (from espn_bio.json) into dashboard_slim_k6_enriched.json.
Run this after scrape_espn_bio.py has produced espn_bio.json.

    python3 integrate_bio.py

Adds the following fields to each player object:
    p['ht']  → height string,   e.g. "6'1\""      (None if not fetched)
    p['wt']  → weight string,   e.g. "165 lbs"    (None if not fetched)
    p['yr']  → class abbreviation, e.g. "JR"      (None if not fetched)
    p['yr_full'] → class full name, e.g. "Junior" (None if not fetched)

Does NOT overwrite p['jersey'] — the parquet-derived value is preferred.
"""

import json
from pathlib import Path

JSON_IN  = 'dashboard_slim_k6_enriched.json'
BIO_FILE = 'espn_bio.json'

if not Path(BIO_FILE).exists():
    print(f"ERROR: {BIO_FILE} not found. Run scrape_espn_bio.py first.")
    exit(1)

with open(JSON_IN) as f:
    slim = json.load(f)
with open(BIO_FILE) as f:
    bio_data = json.load(f)

bio_map = bio_data.get('bio', {})   # aid (str) → {ht, wt, yr, yr_full}

patched = 0
for p in slim['p']:
    aid = p.get('aid')
    bio = bio_map.get(str(aid), {}) if aid else {}

    p['ht']      = bio.get('ht',      None)
    p['wt']      = bio.get('wt',      None)
    p['yr']      = bio.get('yr',      None)   # "FR","SO","JR","SR","GR"
    p['yr_full'] = bio.get('yr_full', None)   # "Freshman", etc.

    if bio:
        patched += 1

print(f"Integrated bio for {patched}/{len(slim['p'])} players.")

# Spot-check
for p in slim['p'][:6]:
    print(f"  {p['n']}: #{p.get('jersey')} | {p.get('ht','—')} | {p.get('yr_full','—')}")

with open(JSON_IN, 'w') as f:
    json.dump(slim, f, separators=(',',':'))

print(f"\nSaved → {JSON_IN}")
print("Next: run  python3 build_v5.py  to rebuild the dashboard.")
