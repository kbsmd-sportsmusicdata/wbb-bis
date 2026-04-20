###############################################
# WBB AGGREGATION & PERCENTILES PIPELINE
###############################################

# 0. Install packages (run once if needed)
# install.packages("wehoop")
# install.packages("dplyr")
# install.packages("readr")

# 0. Load packages (dplyr is needed for the pipe operator %>%)
library(dplyr)
library(readr)

###############################################
# 1. Data Load from wehoop (2024-2026)
###############################################

# Ensure libraries are loaded (since you are in R)
library(wehoop)
library(dplyr)
library(readr)

cat("Starting full data load from wehoop (2024-2026). This will take a moment...\n")

# Use load_wbb_player_box to fetch all available seasons directly.
raw_source_data <- wehoop::load_wbb_player_box(seasons = 2024:2026)
  wbb_pbp <- wehoop::load_wbb_pbp(seasons = 2024:2026)

cat(sprintf("✅ Total rows downloaded from source: %d\n", nrow(raw_source_data)))
cat(sprintf("✅ Seasons included: %s\n", paste(sort(unique(raw_source_data$season)), collapse=", ")))

###############################################
# Helper: weighted percentile function
###############################################
weighted_percentile <- function(x, w) {
  # Returns values in [0,1] representing the
  # weighted cumulative distribution position of each x
  if (length(x) == 0) return(numeric(0))

  w[is.na(w)] <- 0
  x_na <- is.na(x)

  if (all(x_na) || sum(w[!x_na]) <= 0) {
    return(rep(NA_real_, length(x)))
  }

  x2 <- x[!x_na]
  w2 <- w[!x_na]

  ord <- order(x2)
  x_sorted <- x2[ord]
  w_sorted <- w2[ord]

  cw <- cumsum(w_sorted)
  total_w <- sum(w_sorted)

  p_sorted <- cw / total_w

  p2 <- numeric(length(x))
  p2[x_na] <- NA
  p2[!x_na][ord] <- p_sorted

  return(p2)
}

###############################################
# 2. Aggregation / Cleaning
###############################################

# We aggregate game-log level data into season totals for each player
player_season_stats <- raw_box %>%
  # Ensure minutes are numeric (sometimes they come as strings)
  mutate(minutes = as.numeric(minutes)) %>%
  group_by(athlete_id, athlete_display_name, team_short_display_name, athlete_position_abbreviation, season) %>% # Added 'season' here
  summarise(
    games_played  = n(),
    games_started = sum(starter, na.rm = TRUE),
    minutes_total = sum(minutes, na.rm = TRUE),

    # Counting stats
    pts_total     = sum(points, na.rm = TRUE),
    reb_total     = sum(rebounds, na.rm = TRUE),
    oreb_total    = sum(offensive_rebounds, na.rm = TRUE),
    dreb_total    = sum(defensive_rebounds, na.rm = TRUE),
    ast_total     = sum(assists, na.rm = TRUE),
    stl_total     = sum(steals, na.rm = TRUE),
    blk_total     = sum(blocks, na.rm = TRUE),
    tov_total     = sum(turnovers, na.rm = TRUE),

    # Shooting totals
    fgm  = sum(field_goals_made, na.rm = TRUE),
    fga  = sum(field_goals_attempted, na.rm = TRUE),
    fg3m = sum(three_point_field_goals_made, na.rm = TRUE),
    fg3a = sum(three_point_goals_attempted, na.rm = TRUE),
    ftm  = sum(free_throws_made, na.rm = TRUE),
    fta  = sum(free_throws_attempted, na.rm = TRUE),

    .groups = "drop"
  ) %>%
# Basic per-game and per-40 calculations
  mutate(
    mpg = minutes_total / games_played,
    ppg = pts_total / games_played,
    rpg = reb_total / games_played,
    apg = ast_total / games_played,
    spg = stl_total / games_played,
    bpg = blk_total / games_played,
    tovpg = tov_total / games_played,

    # Per 40 Mins (Normalize for playing time)
    # Avoid division by zero with pmax
    pts_per_40  = (pts_total / pmax(minutes_total, 1)) * 40,
    reb_per_40  = (reb_total / pmax(minutes_total, 1)) * 40,
    oreb_per_40 = (oreb_total / pmax(minutes_total, 1)) * 40,
    dreb_per_40 = (dreb_total / pmax(minutes_total, 1)) * 40,
    ast_per_40  = (ast_total / pmax(minutes_total, 1)) * 40,
    stl_per_40  = (stl_total / pmax(minutes_total, 1)) * 40,
    blk_per_40  = (blk_total / pmax(minutes_total, 1)) * 40,
    tov_per_40  = (tov_total / pmax(minutes_total, 1)) * 40
  ) %>%n  # Efficiency & Advanced Metrics
  mutate(
    # Shooting Percentages
    fg_pct  = ifelse(fga > 0, fgm / fga, 0),
    fg3_pct = ifelse(fg3a > 0, fg3m / fg3a, 0),
    ft_pct  = ifelse(fta > 0, ftm / fta, 0),

    # Three Point Attempt Rate
    threepar = ifelse(fga > 0, fg3a / fga, 0),

    # Free Throw Attempt Rate
    fta_rate = ifelse(fga > 0, fta / fga, 0),

    # Effective Field Goal %
    efg_pct = ifelse(fga > 0, (fgm + 0.5 * fg3m) / fga, 0),

    # True Shooting %
    # Approximation: TSA = FGA + 0.44 * FTA
    ts_pct = ifelse((fga + 0.44 * fta) > 0,
                    pts_total / (2 * (fga + 0.44 * fta)), 0),

    # Usage Rate (Approximate Version)
    # Basic Formula: (FGA + 0.44*FTA + TOV) / (Minutes) * (Team Minutes / 5)
    # Since we don't have team totals here, we calculate "Usage Load"
    # and will treat it as a proxy or raw usage volume.
    usage_load = (fga + 0.44 * fta + tov_total),
    usage      = usage_load / pmax(minutes_total, 1), # possessions used per minute

    # Ratio Stats
    ast_to_tov = ifelse(tov_total > 0, ast_total / tov_total, ast_total),

    # Estimated percentages (Simplified without team totals)
    # e.g., ast_pct ~ Ast / (FGA + 0.44*FTA + Ast + TOV)
    # This is a 'player-based' approximation often used when team totals aren't joined.
    possessions_estimated = fga + 0.44 * fta + tov_total + ast_total,
    ast_pct  = ifelse(possessions_estimated > 0, ast_total / possessions_estimated, 0),
    tov_pct  = ifelse(possessions_estimated > 0, tov_total / possessions_estimated, 0),

    # Rebounding Shares (Approximation using per-40/position baselines is common if team totals missing)
    # Here we will just stick to the per-40 or total counts unless we join team data.
    # For the lab, we'll create simple ratios:
    oreb_pct = oreb_total / pmax(reb_total, 1), # % of player's rebs that were offensive
    dreb_pct = dreb_total / pmax(reb_total, 1)  # % of player's rebs that were defensive
  )


###############################################
# 3. Percentiles by Position
###############################################

# We will calculate weighted percentiles (weighted by minutes played)
# so that bench warmers don't skew the distribution for starters.
# Groups: We will group by 'athlete_position_abbreviation' (G, F, C, etc.)

final_dataset <- player_season_stats %>%
  group_by(athlete_position_abbreviation) %>%
  mutate(
    usage_pctile_pos = weighted_percentile(usage, minutes_total),
    ts_pctile_pos    = weighted_percentile(ts_pct, minutes_total),
    efg_pctile_pos   = weighted_percentile(efg_pct, minutes_total),
    ast_pctile_pos   = weighted_percentile(ast_per_40, minutes_total),
    tov_pctile_pos   = weighted_percentile(tov_per_40, minutes_total),
    stl_pctile_pos   = weighted_percentile(stl_per_40, minutes_total),
    blk_pctile_pos   = weighted_percentile(blk_per_40, minutes_total),
    reb_pctile_pos   = weighted_percentile(reb_per_40, minutes_total)
  ) %>%
  ungroup() %>%
  # Filter out players with very low minutes to clean up the dataset
  # (e.g., must have played at least 50 minutes total)
  filter(minutes_total >= 50) %>%
  select(
    athlete_id,
    player    = athlete_display_name,
    team      = team_short_display_name,
    position  = athlete_position_abbreviation,
    season    = season, # Ensure season column exists for tracking

    games_played,
    games_started,
    minutes_total,
    mpg,

    pts_total,
    ppg,
    pts_per_40,

    reb_total,
    rpg,
    reb_per_40,

    oreb_total,
    oreb_pg = oreb_per_40, # Note: naming convention adjusted for simplicity if desired
    oreb_per_40,

    dreb_total,
    dreb_pg = dreb_per_40,
    dreb_per_40,

    ast_total,
    apg,
    ast_per_40,

    stl_total,
    spg,
    stl_per_40,

    blk_total,
    bpg,
    blk_per_40,

    tov_total,
    tovpg,
    tov_per_40,

    fg_pct,
    fg3_pct,
    threepar,
    ft_pct,
    fta_rate,
    efg_pct,
    ts_pct,
    usage,
    ast_pct,
    tov_pct,
    oreb_pct,
    dreb_pct,

    usage_pctile_pos,
    ts_pctile_pos,
    efg_pctile_pos,
    ast_pctile_pos,
    tov_pctile_pos,
    stl_pctile_pos,
    blk_pctile_pos,
    reb_pctile_pos
  )

###############################################
# 4. Export to CSV (Backup)
###############################################
write_csv(final_dataset, "wbb_player_data_2026.csv")

