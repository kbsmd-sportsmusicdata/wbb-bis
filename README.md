# WBB Basketball Intelligence System

NCAA Women's Basketball analytics pipeline — 2025-26 season.

## Pipeline Overview

| Step | Script | Input → Output |
|---|---|---|
| 1 | `scripts/01_player_box_processing.py` | player_box_2026.parquet → player_box_advanced_metrics.csv, player_game_log_enriched.csv |
| 2 | `scripts/02_pbp_player_processing.py` | play_by_play.parquet → pbp_player_metrics.csv |
| 3 | `scripts/03_lineup_stints_regular.py` | play_by_play.parquet → lineup_stints_raw.csv, player_onoff_metrics.csv |
| 4 | `scripts/04_lineup_stints_postseason.py` | march_pbp_2026.parquet → postseason_stints_raw.csv, postseason_onoff_metrics.csv |
| 5 | `scripts/05_merge_scouting.py` | box_advanced + pbp_metrics + bracket → player_scouting_tournament68.csv, player_scouting_top50.csv |
| 6 | `scripts/06_merge_feature_table.py` | box_advanced + pbp_metrics + onoff + rosters → player_feature_table_2026.csv |
| 7 | `scripts/07_build_dashboard.py` | scouting files + context → HTML dashboards |
| 8 | `scripts/08_build_visualization.py` | Kaggle historical → wbb_conference_funnel.html |

## Quick Start

```bash
pip install -r requirements.txt
python config.py          # verify all input files are present
python pipeline.py        # run full pipeline (steps 1, 2, 5, 7)
```

Run steps 3 and 4 separately (lineup stints are slow; run independently after raw PBP is updated):
```bash
python scripts/03_lineup_stints_regular.py
python scripts/04_lineup_stints_postseason.py
```

## Updating for New Data

Edit only `config.py` — update `SCHEDULE_FILE` and `NET_FILE` to the new dated filenames, then re-run `pipeline.py`.

## Data Sources

See `data/raw/SOURCES.md` for scraping instructions and wehoop release URLs.
