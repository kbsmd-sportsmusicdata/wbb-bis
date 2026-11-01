# Metric conventions — possession coefficient and published statistics

**Decided 13 October 2026.** Applies to WBB work in this repo, to the v5 transfer
evaluation cards, and to transfer translation methodology v2.

## The house coefficient stays 0.44

Every possession and usage estimate in this pipeline already uses a 0.44
free-throw coefficient:

| Script | Where it appears |
|---|---|
| `01_player_box_processing.py` | individual possessions, TSA, usage load, PPP |
| `03_lineup_stints_regular.py` | team possessions |
| `04_lineup_stints_postseason.py` | team possessions |
| `12_build_historical_player_box_metrics.py` | historical TSA |
| `14_build_player_profile_bundle.py` | possession estimate |
| `17_build_pbp_situational_splits.py` | shooter possessions |
| `18_build_player_scoped_pbp_situational_splits.py` | shooter possessions |

Sports-Reference uses **0.475** for women's basketball. This was confirmed on
three separate player pages: their published TOV% and TS% reconcile at 0.475 and
not at 0.44.

Moving the house coefficient to 0.475 would mean editing eight scripts and
regenerating every derived dataset — which would silently restate outputs that
have already been published. What it buys is roughly 0.2 percentage points.
Not worth it. **0.44 stays.**

## The rule

1. **Quoting one player's published season figure** — a transfer card, a player
   profile — use the source's published value and name the source. Don't
   recompute what the source already publishes.
2. **Computing across your own data** — pipelines, dashboards, multi-player
   comparisons, anything with no published counterpart — use the house 0.44 and
   state the formula.
3. **Never mix the two inside one comparison.** One table, one chart, one
   row-set comes from one pipeline.

Rule 1 is why the Shannon Dowell evaluation card carries Sports-Reference's
published USG% and TOV% rather than pipeline values.

## The one-line note for the cards

Add to the methodology block of the v5 transfer evaluation cards and to
translation methodology v2:

> Possession-based rates computed here use a 0.44 free-throw coefficient.
> Sports-Reference uses 0.475 for women's basketball, so figures quoted directly
> from Sports-Reference — as on the Dowell evaluation card — sit about 0.2
> percentage points away from pipeline values. The difference is a convention,
> not a correction.

## What the coefficient does not touch

eFG%, FT%, FTA/FGA, AST/TO and per-40 rates contain no free-throw coefficient and
are identical under either convention. The coefficient only affects possession
estimates: USG%, TOV%, TS%, PPP, and anything built on team or individual
possession counts.
