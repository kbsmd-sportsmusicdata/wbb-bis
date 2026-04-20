"""
scrape_espn_bio.py
──────────────────
Fetches height, displayHeight, and class/year from ESPN's athlete endpoint
for every player in dashboard_slim_k6_enriched.json, then writes a
companion file: espn_bio.json

Run this locally (not in the VM — ESPN API requires open internet access):
    python3 scrape_espn_bio.py

Then copy espn_bio.json back and run:
    python3 integrate_bio.py

ESPN endpoint used:
    https://sports.core.api.espn.com/v2/sports/basketball/college-womens/athletes/{aid}?lang=en&region=us

Response fields extracted:
    displayHeight   → e.g. "6'1\""
    displayWeight   → e.g. "165 lbs"
    experience.abbreviation → "FR" | "SO" | "JR" | "SR" | "GR"
    experience.displayValue → "Freshman" | "Sophomore" | etc.
    jersey          → cross-check against parquet-derived value
"""

import json, time, urllib.request, urllib.error
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
SLEEP_MS        = 120     # ms between requests (polite rate limit)
RETRY_WAIT_S    = 8       # seconds to wait on rate-limit / 5xx
MAX_RETRIES     = 2
INPUT_JSON      = 'dashboard_slim_k6_enriched.json'
OUTPUT_JSON     = 'espn_bio.json'
ESPN_TEMPLATE   = (
    'https://sports.core.api.espn.com/v2/sports/basketball/'
    'college-womens/athletes/{aid}?lang=en&region=us'
)

# ── Load player list ─────────────────────────────────────────────────────
with open(INPUT_JSON) as f:
    slim = json.load(f)

players = [p for p in slim['p'] if p.get('aid')]
print(f"Players to fetch: {len(players)}")

# ── Helpers ──────────────────────────────────────────────────────────────
def fetch_bio(aid):
    url = ESPN_TEMPLATE.format(aid=aid)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Accept': 'application/json'
    })
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())

def parse_bio(data):
    out = {}
    # Height
    if 'displayHeight' in data:
        out['ht'] = data['displayHeight']          # e.g. "6'1\""
    # Weight
    if 'displayWeight' in data:
        out['wt'] = data['displayWeight']          # e.g. "165 lbs"
    # Class / experience
    exp = data.get('experience', {})
    if exp:
        out['yr']      = exp.get('abbreviation', '')   # "FR","SO","JR","SR","GR"
        out['yr_full'] = exp.get('displayValue', '')   # "Junior"
    # Jersey (cross-check)
    if 'jersey' in data:
        out['espn_jersey'] = str(data['jersey'])
    return out

# ── Main loop ─────────────────────────────────────────────────────────────
results = {}   # aid → bio dict
errors  = {}

for i, p in enumerate(players):
    aid  = p['aid']
    name = p['n']

    for attempt in range(MAX_RETRIES + 1):
        try:
            data = fetch_bio(aid)
            bio  = parse_bio(data)
            results[str(aid)] = bio
            print(f"  [{i+1}/{len(players)}] {name}: {bio}")
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < MAX_RETRIES:
                print(f"  [{i+1}] {name}: HTTP {e.code} — retrying in {RETRY_WAIT_S}s")
                time.sleep(RETRY_WAIT_S)
            else:
                print(f"  [{i+1}] {name}: HTTP {e.code} — skipping")
                errors[str(aid)] = str(e)
                break
        except Exception as e:
            print(f"  [{i+1}] {name}: ERROR {e}")
            errors[str(aid)] = str(e)
            break

    time.sleep(SLEEP_MS / 1000)

# ── Write output ──────────────────────────────────────────────────────────
output = {'bio': results, 'errors': errors}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nDone. Fetched {len(results)}/{len(players)} players.")
print(f"Errors: {len(errors)}")
print(f"Output: {OUTPUT_JSON}")
