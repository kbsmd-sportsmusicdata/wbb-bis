# Raw Data Sources

Large data files are excluded from git. Download or regenerate them using the
sources below before running the pipeline.

## Player Box Scores — `player_box_2026.parquet`

Source: [wehoop-wbb-data GitHub releases](https://github.com/sportsdataverse/wehoop-wbb-data)

```python
import wehoop
player_box = wehoop.load_wbb_player_boxscore(seasons=2026)
player_box.to_parquet("data/raw/player_box_2026.parquet", index=False)
```

## Play-by-Play — `play_by_play_YYYYMMDD.parquet`

Source: wehoop-wbb-data GitHub releases (same repo above)

```python
pbp = wehoop.load_wbb_pbp(seasons=2026)
pbp.to_parquet("data/raw/play_by_play_YYYYMMDD.parquet", index=False)
```

## Schedule — `schedule_filtered_YYYYMMDD.csv`

Source: wehoop schedule scrape, filtered to D1 games, enriched with opponent NET rank.
Pull via: `wehoop.load_wbb_schedule(seasons=2026)` then filter and export.

## NET Rankings — `net_rankings_YYYYMMDD.csv`

Source: Manual pull from NCAA.com/sports/basketball/rankings or HerHoopStats.
Columns required: `team_id`, `team_name`, `net_rank`, `run_date`

## Postseason PBP — `postseason/march_pbp_2026.parquet`

Source: Same wehoop release, filtered to `season_type == 3` (postseason).

## Kaggle Historical — `data/static/kaggle/`

Source: [NCAA WBB Kaggle dataset](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data)
Download and place all CSV files in `data/static/kaggle/`. Files are static and already committed.
