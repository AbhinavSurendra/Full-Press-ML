# Feature Engineering Plan — Phase C (External Context & Season Stats)

## Context

Phases A and B are implemented and passing — 62.26% XGBoost accuracy on the medium 100-game slice after the April cleanup (drops + re-aggregations in [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py)). The remaining feature-importance headroom is in **context that lives outside the tracking frames**: which team is on offense, how good they are, where the score stands, whether anyone on the court is tired or elite, and whether the team plays better at home. The weak class is `made_3` (f1 = 0.38) — a class that should respond specifically to team shooting profile and pace.

This plan covers Phase C only. Phases A and B are documented in git history and in the pipeline overview.

---

## Data audit — what we have vs. what we need

### Already present in the repo (just needs to be propagated)

| Signal | Source in raw data | Currently in processed tables? |
|---|---|---|
| `game_date` | JSON `gamedate` per game | ❌ not propagated |
| `home_team_id`, `away_team_id` | JSON `home.teamid` / `visitor.teamid`, or PBP | ❌ not propagated |
| `score`, `score_margin` at any event | PBP `SCORE`, `SCOREMARGIN` | ❌ not merged |
| `offense_team_id` (join key) | inferred from PBP | ✅ in frames.csv & possessions.csv |
| Player IDs (for future player joins) | raw JSON moments / rich_frames player slots | ✅ in rich_frames.csv |

### Newly uploaded — [data/external/](data/external/)

All four files are **team-level, 2015-16 season, Basketball Reference format**.

| File | Columns we'll use | Notes |
|---|---|---|
| [advanced_stats.csv](data/external/advanced_stats.csv) | `ORtg`, `DRtg`, `NRtg`, `Pace`, `TS%`, `FTr`, `3PAr`, Off. Four Factors (`eFG%`, `TOV%`, `ORB%`, `FT/FGA`), Def. Four Factors (`eFG%`, `TOV%`, `DRB%`, `FT/FGA`) | Two-row header (group label + column name). Read with `header=[0,1]` or `skiprows=1`. |
| [per_game_stats.csv](data/external/per_game_stats.csv) | `FG%`, `3P%`, `FT%`, `AST`, `TOV`, `TRB`, `PTS` | Single-row header. Most already covered by per-100 version — likely drop in favor of the per-100 file. |
| [per_100_possessions.csv](data/external/per_100_possessions.csv) | Per-100 versions of per-game — pace-adjusted, better for offense/defense comparison | Single-row header. Preferred over per_game for pace-adjusted comparison. |
| [shooting_stats.csv](data/external/shooting_stats.csv) | `%FGA` by distance (0-3, 3-10, 10-16, 16-3P, 3P), `FG%` by distance, `%FG assisted` (2P, 3P), corner 3 `%3PA` + `3P%` | Two-row header. Drives a team-specific shot-type prior for `made_3` class. |

All four files have:
- A "League Average" row at the bottom — filter on `Rk.notna()` or `Team != "League Average"`
- Playoff teams marked with `*` suffix (e.g. `"Golden State Warriors*"`) — strip before matching
- Team names as full strings, not IDs — needs mapping to numeric SportVU `team_id`

### Missing — would unlock additional features if sourced later

| Signal | Would enable feature | Alternative if not sourced |
|---|---|---|
| Player-level PER / BPM / usage% for 2015-16 | `star_player_on_court`, `offense_mean_usage_on_court`, `offense_mean_per_on_court` | Drop player-quality features from Phase C; rely on team-level stats only |
| NBA 2015-16 game schedule | `rest_days`, `is_back_to_back` | Compute from `game_date` ordering across our own games; same team + gap → compute on the fly (works for the subset of games in our dataset) |

---

## Sub-phases (three independent tracks)

### C-1 — Propagate already-available context (no external file work)

Three features, all from files already in the repo.

| Feature | Implementation |
|---|---|
| `game_date` | In [src/full_press_ml/data/raw_loader.py](src/full_press_ml/data/raw_loader.py) near line 137, add `"game_date": game.get("gamedate")` to the event row dict; carry it onto frames via the event→frame merge. Same addition in [src/full_press_ml/data/build_rich_tracking.py](src/full_press_ml/data/build_rich_tracking.py) |
| `is_offense_home` | In the same raw_loader pass, pull `home.teamid` / `visitor.teamid` from the JSON. Emit per-event columns `home_team_id`, `away_team_id`. Compute `is_offense_home = int(offense_team_id == home_team_id)` — either in raw_loader or in a new enrichment step. |
| `score_margin_at_start`, `offense_score_diff_at_start` | Merge from PBP on `(game_id, event_id)` using the **first event of each possession only** to avoid outcome leakage. Parse `SCOREMARGIN` format — `"+5"` / `"-3"` / `"TIE"` / numeric-as-string. `offense_score_diff = margin` if home team has ball else `-margin` (sign depends on which side PBP reports). Implement in [src/full_press_ml/data/possession_rules.py](src/full_press_ml/data/possession_rules.py) inside `segment_possessions()` or as a post-step. |

**Leakage guard:** verify on a made-shot possession that the merged `score_margin_at_start` does NOT reflect the made basket. Easy check: find any `made_2` possession, compare `score_margin_at_start` against the PBP `SCORE` row at the *previous* event.

**Aggregation into features:** these three are possession-level constants → merge onto the possession table. All three land as single columns (no mean/std/min/max).

### C-2 — Derived from C-1 (still no external files)

Requires C-1's `game_date` to be in the processed tables.

| Feature | Formula |
|---|---|
| `rest_days` | For each possession, look up `offense_team_id`'s most recent game date *before* this one in the dataset. `rest_days = (this_game_date - prior_game_date).days`. First game for each team → fill with median. |
| `is_back_to_back` | `(rest_days <= 1).astype(int)` |
| `days_into_season` | `game_date - min(game_date in dataset)`, in days. Proxy for midseason fatigue / role clarity. |

**Implementation:** new module `src/full_press_ml/data/enrich_context.py`, called by `build_possessions.py` after possession tables are built. Input: possessions dataframe with `game_id, game_date, offense_team_id`. Output: same frame with three extra columns.

**Caveat:** `rest_days` computed from our dataset only — if a team's actual prior NBA game was outside our 100-game slice, we'll overestimate rest. Acceptable for a first pass; revisit if we source the full schedule.

### C-3 — External CSV joins (team-level)

New module: `src/full_press_ml/data/enrich_season_stats.py`. Called after aggregation in [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py) — specifically at the end of `build_rich_frame_aggregate_table()` and `build_frame_aggregate_table()`, OR as a post-step inside [train_baseline.py](src/full_press_ml/training/train_baseline.py) before feature selection (simpler, same effect since aggregation is deterministic).

#### Step 1 — Team ID mapping table

New file: `data/external/team_id_map.csv` (30 rows, hand-built once).

```
team_id,br_team_name,br_abbr
1610612737,Atlanta Hawks,ATL
1610612738,Boston Celtics,BOS
…
```

SportVU team IDs come from raw JSON; Basketball Reference team names come from the CSVs. One-time manual join; stable across seasons.

#### Step 2 — Loader for each CSV

```python
# enrich_season_stats.py
def load_advanced_stats(path):
    df = pd.read_csv(path, skiprows=1)     # skip the group-label row
    df = df[df["Rk"].notna()]               # drop "League Average"
    df["Team"] = df["Team"].str.rstrip("*") # strip playoff marker
    return df

def load_per_100(path): ...
def load_shooting(path): ...
```

Each loader returns a DataFrame keyed by `Team` (full name), ready to merge onto the ID map.

#### Step 3 — Features from [advanced_stats.csv](data/external/advanced_stats.csv)

All joined on `offense_team_id` (and mirrored on `defense_team_id` where it exists — defense team is the non-offense team in the matchup, and for our data this means the home/away team complement).

Offense side (the team with the ball):
- `offense_team_ortg` — offensive rating
- `offense_team_pace`
- `offense_team_ts_pct`
- `offense_team_ftr`, `offense_team_3par`
- `offense_team_efg_pct`, `offense_team_tov_pct`, `offense_team_orb_pct`, `offense_team_ft_per_fga`

Defense side (the opponent's defense):
- `defense_team_drtg`
- `defense_team_opp_efg_pct`, `defense_team_opp_tov_pct`, `defense_team_drb_pct`, `defense_team_opp_ft_per_fga`
- `matchup_net_rating_diff = offense_team_ortg - defense_team_drtg`

#### Step 4 — Features from [per_100_possessions.csv](data/external/per_100_possessions.csv)

Most overlap with advanced_stats. Pull the two that don't:
- `offense_team_ast_per_100`
- `offense_team_stl_per_100`, `defense_team_blk_per_100`

Skip [per_game_stats.csv](data/external/per_game_stats.csv) — it's the same data pace-confounded. Keep it as a fallback reference but don't join it in.

#### Step 5 — Features from [shooting_stats.csv](data/external/shooting_stats.csv)

These are the most interesting for `made_3` class recovery.
- `offense_team_pct_fga_3p` — how three-happy is the offense
- `offense_team_fg_pct_3p` — how good are they at 3s
- `offense_team_corner3_rate`, `offense_team_corner3_pct`
- `offense_team_pct_fga_0_3` — rim rate
- `offense_team_fg_pct_0_3` — finishing at rim
- `offense_team_pct_fg_3p_assisted` — team playstyle (catch-and-shoot vs. pull-up)

Mirror the distance breakdowns on defense for opponent allowed shots → `defense_team_pct_fga_3p_allowed` etc. Basketball Reference reports these under the same CSV for the defending team.

### C-4 — Player-level features (DEFERRED)

Would add: `star_player_on_court`, `offense_mean_per_on_court`, `offense_mean_usage_on_court`.

Requires a player-level 2015-16 stats CSV (PER, BPM, usage%) from Basketball Reference's player season-totals page — NOT in the uploaded files. Flagged as the single highest-value missing data source.

When sourced, join logic lives in the same `enrich_season_stats.py`:
1. Load `player_stats.csv` keyed on BR player ID
2. Build map from SportVU `player_id` (from rich_frames player slots) to BR ID — requires either a manual name-based map or downloading a mapping table
3. For each possession, look up the 5 `is_offense` player slots in rich_frames at the possession's first frame, join their PER / usage%, and aggregate: `max(PER)`, `mean(usage%)`, `count(PER >= 20)`

---

## Critical Files

| Path | Change |
|---|---|
| [src/full_press_ml/data/raw_loader.py](src/full_press_ml/data/raw_loader.py) | Propagate `gamedate`, `home_team_id`, `away_team_id` (C-1) |
| [src/full_press_ml/data/build_rich_tracking.py](src/full_press_ml/data/build_rich_tracking.py) | Same — rich pipeline has a parallel loader path |
| [src/full_press_ml/data/possession_rules.py](src/full_press_ml/data/possession_rules.py) | Merge PBP `SCOREMARGIN` at possession start (C-1) |
| `src/full_press_ml/data/enrich_context.py` (new) | Rest days, back-to-back, days into season (C-2) |
| `src/full_press_ml/data/enrich_season_stats.py` (new) | Loaders + joins for the four external CSVs (C-3) |
| `data/external/team_id_map.csv` (new) | 30-row mapping, SportVU numeric ↔ Basketball Reference name ↔ abbr |
| [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py) | Call `enrich_season_stats` at the end of `build_rich_frame_aggregate_table` and `build_frame_aggregate_table` |

---

## Verification

```bash
# Baseline (Phases A+B after April cleanup)
python -m full_press_ml.training.train_baseline \
  --data data/processed/rich_medium/rich_frames.csv \
  --aggregate-frames --rich --model xgboost --eval-split test

# After C-1 + C-2 (context from files we already have):
# Expect small but consistent lift; watch is_offense_home, rest_days, score_margin land
# in the importance table.

# After C-3 (external CSVs joined):
# Expect the biggest lift to be on made_3 f1. Team pace and 3PA rate should land
# in the top half of the importance table.
```

Spot checks:
- `score_margin_at_start` should NOT equal the post-possession margin (leakage sniff test)
- Pick a known back-to-back game from the schedule and verify `rest_days == 1` for the right team
- Every possession's `offense_team_ortg` should match the team's row in advanced_stats.csv
- No NaNs introduced on the 30 teams in the dataset (unmapped team IDs would silently fill NaN)

## Recommended ordering

1. **C-1** — half day; uses only files we already have; confirms the plumbing
2. **C-2** — hour or two; depends on C-1's game_date; minimal external risk
3. **C-3** — half day; needs the team ID map (the manual one-time piece); largest expected accuracy gain
4. **C-4** — deferred until player-level CSV is sourced

After each stage, re-run medium and diff the feature importance table — the point isn't just overall accuracy, it's identifying *which* context signals land.
