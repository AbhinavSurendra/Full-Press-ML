# Feature Engineering Plan

## Context

The project predicts NBA possession outcomes (5 classes: `made_2`, `made_3`, `missed_shot`, `turnover`, `free_throws`) from SportVU player-tracking data. The core research question is **how early in a possession the outcome can be predicted** (at 2s, 4s, 6s, 8s horizons). The pipeline produces two frame-level data sources (see schema below) which feed tabular baselines (logistic regression, XGBoost) and the LSTM.

---

## Data Sources

### Standard frames (`data/processed/standard/frames.csv`) — 31 columns
Pre-computed team aggregates, built by `build_possessions.py`.

Key columns: `ball_x/y/z`, `shot_clock`, `game_clock`, `quarter`, `possession_frame_idx`, `offense_centroid_x/y`, `defense_centroid_x/y`, `offense_mean_radius`, `defense_mean_radius`, `offense_mean_distance_to_ball`, `defense_mean_distance_to_ball`, `split`, `possession_id`, `terminal_label`

### Rich frames (`data/processed/rich/rich_frames.csv`) — 68 columns ✓ generated
Raw per-player slot coordinates, built by `build_rich_tracking.py`.

Key columns: same ball/clock/game fields as above, plus `player_0_team_id/x/y/z` … `player_9_team_id/x/y/z`, `player_slot_count`

**Missing from rich_frames vs standard frames** (derived by `add_rich_player_features`):
`offense_centroid_x/y`, `defense_centroid_x/y`, `offense_mean_radius`, `defense_mean_radius`, `offense_mean_distance_to_ball`, `defense_mean_distance_to_ball`, `player_count`, `offense_player_count`, `defense_player_count`, `missing_shot_clock`, `possession_frame_idx`

### Player frames (`data/processed/rich/player_frames.csv`) ✓ generated
Long format — one row per player per frame. Available but not yet used in feature engineering.

---

## Critical Files

| File | Role |
|---|---|
| `src/full_press_ml/features/engineer.py` | All feature engineering — standard and rich pipelines |
| `src/full_press_ml/training/train_baseline.py` | `--aggregate-frames` + `--rich` flags dispatch the right pipeline |
| `src/full_press_ml/data/build_rich_tracking.py` | Generates `rich_frames.csv` and `player_frames.csv` |
| `data/processed/standard/frames.csv` | Standard pipeline input |
| `data/processed/rich/rich_frames.csv` | Rich pipeline input |
| `data/raw/tiny/2015-16_pbp.csv` | Raw play-by-play |

---

## Pipeline Architecture

### Standard pipeline
```
frames.csv → build_frame_aggregate_table() → possession-level features
```

### Rich pipeline (new)
```
rich_frames.csv → add_rich_player_features()
                     ├── derives standard columns from player slots (vectorized numpy)
                     ├── adds Phase B per-frame features
                     └── build_rich_frame_aggregate_table()
                             ├── calls build_frame_aggregate_table() for all Phase A aggregations
                             └── merges Phase B aggregations on top
```

### Training command
```bash
# Standard pipeline
python -m full_press_ml.training.train_baseline \
  --data data/processed/standard/frames.csv \
  --aggregate-frames --model xgboost --eval-split val

# Rich pipeline (Phase A + B features)
python -m full_press_ml.training.train_baseline \
  --data data/processed/rich/rich_frames.csv \
  --aggregate-frames --rich --model xgboost --eval-split val
```

---

## Phase A — ✓ Implemented

All features computed in `add_basic_tracking_features()` and `_add_motion_features()` in `engineer.py`. Work with both standard and rich frames (the shim provides the required columns for rich frames).

### Spatial

| Feature | Status | Output columns |
|---|---|---|
| `ball_dist_to_hoop` | ✓ done | `ball_dist_to_hoop_mean/std/min/max`, `ball_dist_to_hoop_start/end`, `ball_dist_to_hoop_delta` |
| `ball_in_paint` | ✓ done | `ball_in_paint_fraction`, `ball_entered_paint` |
| `ball_distance_from_center` | ✓ done | `ball_distance_from_center_mean/std/min/max` |
| offense/defense spacing | ✓ done | `offense_mean_radius_mean/std/min/max/start/end/delta`, same for defense |

### Movement / Motion

| Feature | Status | Output columns |
|---|---|---|
| `ball_speed` | ✓ done | `ball_speed_mean/std/min/max` |
| `ball_dist_traveled` | ✓ done | `ball_dist_traveled` |
| `ball_toward_basket` | ✓ done | `ball_toward_basket_mean/std/min/max` |
| `pass_count_proxy` | ✓ done | `pass_count_proxy` |

### Context / Timing

| Feature | Status | Output columns |
|---|---|---|
| `shot_clock_bucket` | ✓ done | `shot_clock_bucket_mean`, `shot_clock_bucket_start` |
| `shot_clock_low` | ✓ done | `shot_clock_low_mean` |
| `shot_clock_consumed` | ✓ done | `shot_clock_consumed` |
| game clock start/end | ✓ done | `game_clock_start/end` |

---

## Phase B — ✓ Implemented (rich pipeline only)

Computed in `add_rich_player_features()` using numpy vectorization over the 10 player slot columns. Aggregated in `build_rich_frame_aggregate_table()`.

### How `add_rich_player_features` works

1. Sorts by `(game_id, possession_id, event_id, frame_idx)`, derives `possession_frame_idx` via `cumcount`
2. Builds `(N, 10)` numpy arrays for `team_ids`, `xs`, `ys` from player slots
3. Creates boolean `is_offense` / `is_defense` masks by comparing `player_i_team_id` against `offense_team_id` (NaN-safe)
4. Computes all standard-frames columns (centroids, radii, distances) via masked `np.nanmean`
5. Adds Phase B per-frame features (see below)

### Phase B features

| Feature | Per-frame logic | Aggregated output columns |
|---|---|---|
| `nearest_defender_to_ball` | `np.nanmin` of defense player distances to ball | `nearest_defender_to_ball_mean`, `nearest_defender_to_ball_min` |
| `offense_convex_hull_area` | `scipy.spatial.ConvexHull(offense_x/y_pts).volume` per row | `offense_convex_hull_area_mean/std` |
| `defense_convex_hull_area` | Same for defense | `defense_convex_hull_area_mean/std` |
| `paint_occupancy_offense` | Count offense players with x/y in paint zone | `paint_occupancy_offense_mean/max` |
| `paint_occupancy_defense` | Count defense players in paint | `paint_occupancy_defense_mean/max` |

**Note on convex hull:** `_convex_hull_area(pts)` returns `NaN` for < 3 valid players or collinear points. These fill to `0.0` via `fillna` in the training script.

### Phase B features not yet implemented

| Feature | Reason deferred |
|---|---|
| Per-player speeds (`off_player_avg_speed`, `off_player_max_speed`) | Requires per-player tracking through possession; complex identity matching — use `player_frames.csv` long format |
| `ball_handler_speed` / `ball_handler_acceleration` | Ball-handler identity flips frame-to-frame; needs smoothing — deferred |

---

## Phase C — Needs External Data

### Context features

| Feature | What's needed | Notes |
|---|---|---|
| `score_margin` | Score at possession start | Check if `SCORE`/`SCOREMARGIN` in `2015-16_pbp.csv`. Use value at possession *start* only (leakage risk). |
| `home_away_indicator` | Which team is home | From game schedule or game_id metadata |
| `rest_days` / back-to-back | Game schedule | Basketball Reference 2015-16 schedule |
| `star_player_on_court` | Player quality ratings | Join 2015-16 PER/VORP by `player_id` |

### Team season stats (join on `offense_team_id`)

| Feature | Source |
|---|---|
| `offense_true_shooting_pct` | Basketball Reference team stats 2015-16 |
| `offense_turnover_rate` | Same |
| `offense_net_rating`, `defense_net_rating` | Basketball Reference / NBA stats API |
| `offense_reb_pct`, `offense_foul_rate`, `offense_ft_pct` | Same |

**How to implement:** Download per-stat CSVs. Map SportVU numeric `team_id` to Basketball Reference team name. Static join at possession level.

---

## Verification

```bash
# 1. Confirm Phase A still works on standard frames
python -m full_press_ml.training.train_baseline \
  --data data/processed/standard/frames.csv \
  --aggregate-frames --model xgboost --eval-split val

# 2. Run rich pipeline — compare accuracy against 53.5% standard baseline
python -m full_press_ml.training.train_baseline \
  --data data/processed/rich/rich_frames.csv \
  --aggregate-frames --rich --model xgboost --eval-split val

# 3. Spot-check Phase B columns are present
python -c "
import pandas as pd
from full_press_ml.features.engineer import build_rich_frame_aggregate_table
df = pd.read_csv('data/processed/rich/rich_frames.csv')
out = build_rich_frame_aggregate_table(df)
phase_b = [c for c in out.columns if any(k in c for k in ['hull', 'paint', 'defender'])]
print(phase_b)
print(out[phase_b].describe())
"
```

Expected outcomes:
- Phase A standard: matches or exceeds existing 53.5% benchmark
- Phase A + B rich: same or better than standard (Phase B adds new signal)
- Phase B columns should have no all-NaN possessions for `nearest_defender_to_ball`; hull areas may have NaN for very short possessions (< 3 players tracked)
