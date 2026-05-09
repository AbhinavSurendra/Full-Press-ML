# Feature Engineering Plan — Phase C (External Context & Season Stats)

## Context

Phases A and B are implemented and passing — 62.66% XGBoost accuracy on the medium 100-game slice after the April cleanup (drops + re-aggregations in [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py)). The remaining feature-importance headroom is in **context that lives outside the tracking frames**: which team is on offense, how good they are, where the score stands, and whether anyone on the court is elite. The weak class is `made_3` (f1 = 0.37) — a class that should respond specifically to team shooting profile, pace, and star-shooter presence.

This plan covers Phase C only. Phases A and B are documented in git history and in the pipeline overview.

## Strategy decision (May 2026)

After reviewing available data and weighing signal-vs-noise on the 100-game medium slice:

- **Reduced C-1** is in scope for the first iteration (team_id plumbing, score margin, `is_offense_home`).
- **C-2 is in scope** (NEW) — shot-release detection + 8 derived features (`release_dist_to_hoop`, `is_behind_three_point_arc`, etc.) added to [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py). Replaces former C-2 (rest/B2B), which is now C-5. Highest-leverage gap on `made_3` per the May 2026 results audit. Pure frame-math, no external data.
- **C-3 is in scope** — team season stats from [data/external/team_stats/](data/external/team_stats/).
- **C-4 is in scope** (promoted from deferred) — player season stats are available at [data/external/player_stats/](data/external/player_stats/).
- **C-5 is deferred** to the full-dataset run (former C-2 content: `rest_days`, `is_back_to_back`, `days_into_season`). On medium, `rest_days` is dominated by gaps to games NOT in the dataset → noise.
- All Phase C features will be injected in [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py) at the end of `build_rich_frame_aggregate_table()` and `build_frame_aggregate_table()`.

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

### Uploaded — team-level [data/external/team_stats/](data/external/team_stats/)

All four files are **team-level, 2015-16 season, Basketball Reference format**.

| File | Columns we'll use | Notes |
|---|---|---|
| [team_stats/advanced_stats.csv](data/external/team_stats/advanced_stats.csv) | `ORtg`, `DRtg`, `NRtg`, `Pace`, `TS%`, `FTr`, `3PAr`, Off. Four Factors (`eFG%`, `TOV%`, `ORB%`, `FT/FGA`), Def. Four Factors (`eFG%`, `TOV%`, `DRB%`, `FT/FGA`) | Two-row header (group label + column name). Read with `header=[0,1]` or `skiprows=1`. |
| [team_stats/per_game_stats.csv](data/external/team_stats/per_game_stats.csv) | `FG%`, `3P%`, `FT%`, `AST`, `TOV`, `TRB`, `PTS` | Single-row header. Most already covered by per-100 version — drop in favor of the per-100 file. |
| [team_stats/per_100_possessions.csv](data/external/team_stats/per_100_possessions.csv) | Per-100 versions of per-game — pace-adjusted, better for offense/defense comparison | Single-row header. Preferred over per_game for pace-adjusted comparison. |
| [team_stats/shooting_stats.csv](data/external/team_stats/shooting_stats.csv) | `%FGA` by distance (0-3, 3-10, 10-16, 16-3P, 3P), `FG%` by distance, `%FG assisted` (2P, 3P), corner 3 `%3PA` + `3P%` | Two-row header. Drives a team-specific shot-type prior for `made_3` class. |

All four files have:
- A "League Average" row at the bottom — filter on `Rk.notna()` or `Team != "League Average"`
- Playoff teams marked with `*` suffix (e.g. `"Golden State Warriors*"`) — strip before matching
- Team names as full strings, not IDs — needs mapping to numeric SportVU `team_id`

### Uploaded — player-level [data/external/player_stats/](data/external/player_stats/)

Per-player, 2015-16 season, Basketball Reference format. Promotes C-4 from deferred to active.

| File | Columns we'll use | Notes |
|---|---|---|
| [player_stats/advanced_stats.csv](data/external/player_stats/advanced_stats.csv) | `PER`, `TS%`, `USG%`, `BPM`, `OBPM`, `DBPM`, `WS/48`, `VORP` | Single-row header. Key file for star-player features. |
| [player_stats/per_100_possessions.csv](data/external/player_stats/per_100_possessions.csv) | per-100 box stats per player | Optional — pace-adjusted volume. |
| [player_stats/shooting.csv](data/external/player_stats/shooting.csv) | per-player shot distribution + accuracy by distance | Two-row header. Drives `mean_offense_3p_pct_on_court` etc. |
| [player_stats/adjusted_shooting.csv](data/external/player_stats/adjusted_shooting.csv) | shooting normalized to league context | Two-row header. Optional. |

All player files use BR slug `Player-additional` (e.g. `curryst01`) as canonical key. SportVU rich_frames use NBA stats numeric `playerid`. **No direct numeric link in our files** — must bridge via name.

### Missing — would unlock additional features if sourced later

| Signal | Would enable feature | Alternative if not sourced |
|---|---|---|
| NBA 2015-16 game schedule | `rest_days`, `is_back_to_back` (C-5) | Compute from `game_date` ordering across our own games; works only on full-dataset run, not 100-game medium |
| NBA-stats-id ↔ BR-slug mapping CSV | Direct numeric player ID join (skip fuzzy match) | Fuzzy name match — see C-4 below |

---

## Sub-phases

### C-1 (reduced) — Propagate already-available context (in scope for medium)

Score margin merge + team_id plumbing. `game_date` propagation is also done so C-5 unlocks for the full-dataset run later.

| Feature | Status on medium | Implementation |
|---|---|---|
| `home_team_id`, `away_team_id` | **Required plumbing**, not a feature itself | In [src/full_press_ml/data/raw_loader.py](src/full_press_ml/data/raw_loader.py) near line 137, pull `home.teamid` / `visitor.teamid` from the JSON. Emit per-event columns. Same addition in [src/full_press_ml/data/build_rich_tracking.py](src/full_press_ml/data/build_rich_tracking.py). Needed for C-3 / C-4 defense-side joins (`defense_team_id` = the non-offense team). |
| `is_offense_home` | **Keep** — cheap byproduct, weak but real signal | `int(offense_team_id == home_team_id)`, computed in the same enrichment step. |
| `score_margin_at_start`, `offense_score_diff_at_start` | **Keep** — strongest C-1 signal (garbage-time + end-of-quarter shot selection shifts) | Merge from PBP on `(game_id, event_id)` using the **first event of each possession only** to avoid outcome leakage. Parse `SCOREMARGIN` format — `"+5"` / `"-3"` / `"TIE"` / numeric-as-string. `offense_score_diff = margin` if home team has ball else `-margin`. Implement in [src/full_press_ml/data/possession_rules.py](src/full_press_ml/data/possession_rules.py) inside `segment_possessions()` or as a post-step. |
| `game_date` | **Plumb but don't use as a feature on medium** — feeds C-5 once full dataset is available | Add `"game_date": game.get("gamedate")` to the event row dict in raw_loader; carry to frames via the event→frame merge. |

**Leakage guard:** verify on a made-shot possession that the merged `score_margin_at_start` does NOT reflect the made basket. Spot-check: find any `made_2` possession, compare `score_margin_at_start` against the PBP `SCORE` row at the *previous* event.

**Aggregation into features:** all listed C-1 features are possession-level constants → merge onto the possession table. Single columns (no mean/std/min/max).

### C-2 — Shot-release detection + derived features (IN SCOPE, medium)

Highest-leverage gap on `made_3`. Model currently uses `ball_z_max` as a brittle 3pt proxy because no release-point feature exists; `ball_dist_to_hoop_min` reflects post-release ball trajectory through the rim, identical for makes/misses from the same spot. Adds release-frame detection (frame-level math, no external data) plus eight derived features. Also fixes `pass_count_proxy` / `ball_dist_traveled` which currently count post-shot rebound motion across the whole possession.

| Feature | Definition |
|---|---|
| `release_x`, `release_y`, `release_z` | Ball coords (ft) at detected release frame |
| `release_dist_to_hoop` | xy distance from release point to attacking hoop |
| `is_behind_three_point_arc` | NBA geometry: corner zone (`abs(release_y - 25) > 11`) → `abs(release_x - hoop_x) >= 22.0`; else euclidean `release_dist_to_hoop >= 23.75` |
| `is_corner_three` | Corner zone AND behind arc |
| `release_z_above_rim` | `release_z - 10` |
| `release_ball_speed_xy` | Ball xy speed at release frame (jumper vs floater proxy) |
| `frames_after_release` | `max(possession_frame_idx) - release_frame_idx` (rebound / outlet time) |
| `nearest_defender_at_release` | Closest defender at release frame (rich frames only — closeout / contest proxy) |

**Detection algorithm** (per possession, no label peek):

1. Compute `ball_z_range = ball_z.max() - ball_z.min()`. If `< 4 ft` → no release (e.g. dribble-only turnover).
2. `apex_idx = argmax(ball_z)`. If `ball_z[apex_idx] < 10 ft` → no release.
3. Walk back from `apex_idx` along the rising edge: release frame = first frame above 7 ft on the way up to apex.

**Free-throw gating** (structural — no `terminal_label` peek): `shot_clock` null across the whole possession OR `ball_x` std < 1 ft for 25+ frames before apex with `ball_x_start` within ±2 ft of FT line (`x ∈ {19, 75}`) → all release features NaN.

**`pass_count_proxy` / `ball_dist_traveled` fix**: aggregate `ball_step_dist` only over `possession_frame_idx <= release_frame_idx`. Possessions with NaN release fall back to whole possession (current behaviour). Same column names; semantics narrow to pre-release for shooting possessions.

**Leakage guard**: release detection consumes only ball x/y/z and `possession_frame_idx`. No PBP, no `terminal_label`. Verify `release_dist_to_hoop` for known made-3 events ≥ 22 ft.

**Implementation location**: new private helper `_detect_shot_release()` in [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py), alongside existing `_add_motion_features` ([engineer.py:105-146](src/full_press_ml/features/engineer.py#L105-L146)). Extend `_add_motion_features` with z-velocity (`ball_step_dist_z`, `ball_speed_z`) — required for the walk-back. Add NBA arc constants (`_THREE_POINT_RADIUS = 23.75`, `_THREE_POINT_CORNER_X = 22.0`, `_RIM_HEIGHT = 10.0`) near existing court constants ([engineer.py:11-26](src/full_press_ml/features/engineer.py#L11-L26)). Call from both `build_frame_aggregate_table` ([engineer.py:337](src/full_press_ml/features/engineer.py#L337)) and `build_rich_frame_aggregate_table` ([engineer.py:296](src/full_press_ml/features/engineer.py#L296)) after motion aggregation, before metadata merge.

### C-3 — External CSV joins (team-level) — IN SCOPE

New module: `src/full_press_ml/data/enrich_season_stats.py`. Called at the end of `build_rich_frame_aggregate_table()` and `build_frame_aggregate_table()` in [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py).

#### Step 1 — Team ID mapping table

New file: [data/external/team_id_map.csv](data/external/team_id_map.csv) (30 rows, scaffolded then verified).

```
team_id,br_team_name,br_abbr
1610612737,Atlanta Hawks,ATL
1610612738,Boston Celtics,BOS
…
```

**Scaffold approach (reduces hand work):**
1. Iterate raw JSON game headers and collect distinct `(team_id, team_name, team_abbr)` triples — produces 30 rows in canonical SportVU form.
2. Read the `Team` column out of [team_stats/advanced_stats.csv](data/external/team_stats/advanced_stats.csv), strip `*` playoff marker.
3. Auto-pair on abbreviation where SportVU and BR agree (most do); pair on Levenshtein/normalized full name where abbreviations differ (`BKN`/`BRK`, `CHA`/`CHO`, `PHX`/`PHO`).
4. Write `data/external/team_id_map.csv`. User verifies 30 rows once.

Stable across seasons — build it once and commit.

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

### C-4 — Player-level features — IN SCOPE (promoted from deferred)

Player CSVs are present at [data/external/player_stats/](data/external/player_stats/). Highest expected lift on the weak `made_3` class.

**Target features (offense side):**
- `offense_max_per_on_court` — best player on the floor (proxy for star_player presence)
- `offense_mean_per_on_court`
- `offense_mean_usg_pct_on_court`
- `offense_count_elite_on_court` — `sum(PER >= 20)`
- `offense_mean_3p_pct_on_court` — drives `made_3` directly
- `offense_mean_ts_pct_on_court`

**Target features (defense side, mirror):**
- `defense_max_bpm_on_court`, `defense_mean_dbpm_on_court`
- `defense_mean_blk_pct_on_court`

#### Player ID bridge — fuzzy name match (Path C: hybrid)

SportVU rich_frames carry NBA-stats numeric `playerid` plus `firstname` / `lastname`. BR player CSVs carry full `Player` name + slug. No direct numeric link, so we bridge by name.

**Build [data/external/player_id_map.csv](data/external/player_id_map.csv) once:**

1. Collect all distinct `(playerid, firstname, lastname)` from raw rich_frames player slots → ~500 rows.
2. Normalize names on both sides:
    - `unicodedata.normalize("NFKD", name).encode("ascii", "ignore")` — strip diacritics (`Jokić` → `Jokic`)
    - lowercase, strip punctuation, strip `Jr.` / `Sr.` / `II` / `III` suffixes
3. Auto-match on `(first_norm, last_norm)` exact equality → expected ~95-98% hit rate for a single season.
4. Dump unmatched + ambiguous (multiple BR rows match one SportVU row, or vice versa) to `data/external/player_id_map_review.csv`. Hand-fix ~10-25 rows. Common culprits:
    - Senior/Junior collisions: e.g. Tim Hardaway / Tim Hardaway Jr.
    - Initial punctuation: `J.R. Smith` vs `JR Smith`
    - Apostrophe edge cases: `D'Angelo Russell` (covered by punctuation strip but verify)
    - Hyphenated last names
5. Merge resolved rows into `player_id_map.csv`. Commit.

**Error budget:** with 500 players and ~3% miss rate post-normalize, expect ~15 manual rows. ~15 minutes of human work after the auto-match script runs. If a public NBA-stats↔BR mapping CSV is sourced later, the whole step collapses to a numeric merge.

#### Aggregation logic

In `enrich_season_stats.py`, after team joins:
1. For each possession, get the 5 offense `playerid`s at the **possession's first frame** (avoid frame-by-frame variation; substitutions inside a possession are rare and aggregating across frames would dilute).
2. Inner-join to `player_id_map.csv` → BR slug.
3. Inner-join to player advanced/shooting tables.
4. Aggregate to possession row: `max`, `mean`, `count(>= threshold)` per stat.
5. Mirror for the 5 defense players.

**Missing-player handling:** if any of the 5 slots fails to map, fall back to `mean(team-level stat)` for that slot rather than NaN-poisoning the aggregate. Log unmapped fraction — if > 2%, fix the map before training.

### C-5 — Rest / fatigue features (DEFERRED to full-dataset run)

Moved verbatim from former C-2. Depends on `game_date` propagation (still in scope under C-1).

| Feature | Formula | Re-enable when |
|---|---|---|
| `rest_days` | For each possession, look up `offense_team_id`'s most recent game date *before* this one. `rest_days = (this_game_date - prior_game_date).days`. First game for each team → fill with median. | Full-dataset run. |
| `is_back_to_back` | `(rest_days <= 1).astype(int)` | Full-dataset run. |
| `days_into_season` | `game_date - min(game_date in dataset)`, in days. Proxy for midseason fatigue / role clarity. | Full-dataset run, OR if medium slice spans many months. |

Skipped on medium because the 100-game slice means most teams' true prior NBA game is NOT in the dataset → `rest_days` dominated by inter-game gaps that don't exist in reality.

**Future implementation:** new module `src/full_press_ml/data/enrich_context.py`, called by `build_possessions.py` after possession tables are built. Input: possessions dataframe with `game_id, game_date, offense_team_id`. Output: same frame with three extra columns. C-1's `game_date` plumbing is the prerequisite (already in scope).

---

## Critical Files

| Path | Change | Phase |
|---|---|---|
| [src/full_press_ml/data/raw_loader.py](src/full_press_ml/data/raw_loader.py) | Propagate `gamedate`, `home_team_id`, `away_team_id` | C-1 |
| [src/full_press_ml/data/build_rich_tracking.py](src/full_press_ml/data/build_rich_tracking.py) | Same — rich pipeline has a parallel loader path | C-1 |
| [src/full_press_ml/data/possession_rules.py](src/full_press_ml/data/possession_rules.py) | Merge PBP `SCOREMARGIN` at possession start, derive `is_offense_home` | C-1 |
| [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py) | Add 3pt-arc constants, extend `_add_motion_features` with z-velocity, add `_detect_shot_release` helper, 8 release-derived features, pre-release truncation of `pass_count_proxy` / `ball_dist_traveled` | C-2 |
| `src/full_press_ml/data/enrich_context.py` (new, deferred) | Rest days, back-to-back, days into season | C-5 |
| `src/full_press_ml/data/enrich_season_stats.py` (new) | Loaders + joins for team and player CSVs | C-3 + C-4 |
| `scripts/build_team_id_map.py` (new) | Auto-scaffold `team_id_map.csv` from raw JSON + BR CSV | C-3 |
| `scripts/build_player_id_map.py` (new) | Auto-scaffold `player_id_map.csv` via fuzzy name match, dump review CSV | C-4 |
| [data/external/team_id_map.csv](data/external/team_id_map.csv) (new) | 30-row mapping, SportVU numeric ↔ BR name ↔ abbr | C-3 |
| [data/external/player_id_map.csv](data/external/player_id_map.csv) (new) | ~500-row mapping, NBA-stats numeric ↔ BR slug | C-4 |
| [src/full_press_ml/features/engineer.py](src/full_press_ml/features/engineer.py) | Call `enrich_season_stats` at the end of `build_rich_frame_aggregate_table` and `build_frame_aggregate_table` | C-3 + C-4 |

---

## Verification

```bash
# Baseline (Phases A+B after April cleanup)
python -m full_press_ml.training.train_baseline \
  --data data/processed/rich_medium/rich_frames.csv \
  --aggregate-frames --rich --model xgboost --eval-split test

# After reduced C-1:
# Expect small but consistent lift. Watch score_margin_at_start, is_offense_home
# land in the importance table.

# After C-2 (shot release):
# Expect biggest lift on made_3 f1 (currently 0.37) and reduction of made_3→made_2
# confusion (currently 0.34). release_dist_to_hoop, is_behind_three_point_arc,
# release_z should land in top-15 importance. pass_count_proxy gain should drop
# (no longer counting post-shot frames). made_3 calibration curve should extend
# further right (currently dies at ~0.55 mean prob).

# After C-3 (team CSVs joined):
# Expect further lift on made_3 f1. Team pace and 3PA rate should land
# in the top half of the importance table.

# After C-4 (player CSVs joined):
# Expect further made_3 lift (star shooters → made 3 prior). offense_max_per_on_court
# and offense_mean_3p_pct_on_court should land high.

# After full-dataset run, re-enable C-5:
# rest_days, is_back_to_back, days_into_season come into play.
```

Spot checks:
- `score_margin_at_start` should NOT equal the post-possession margin (leakage sniff test)
- C-2: `release_dist_to_hoop` for any `made_3` possession should be ≥ 22 ft (corner) or ≥ 23.75 ft (above break)
- C-2: `release_z_above_rim` distribution should center positive (most NBA jumpers release above rim)
- C-2: free-throw possessions should have NaN release features (structural mask hit)
- C-2: `is_behind_three_point_arc.mean()` for `made_3` should be ≥ 0.95
- C-2: `pass_count_proxy` after fix should not change for non-shooting possessions; should drop modestly for shooting possessions (rebound frames excluded)
- C-2: Steph Curry made-3 spot check — `release_dist_to_hoop >= 23.75`, `is_behind_three_point_arc == 1`
- Every possession's `offense_team_ortg` should match the team's row in [team_stats/advanced_stats.csv](data/external/team_stats/advanced_stats.csv)
- No NaNs introduced on the 30 teams in the dataset (unmapped team IDs would silently fill NaN — assert in loader)
- C-4: log unmapped player fraction. If > 2%, fix `player_id_map.csv` before training.
- Sanity: a possession with Steph Curry on the floor should have `offense_max_per_on_court >= 28` (his 2015-16 PER).
- C-5 leakage guard (when re-enabled on full dataset): pick a known back-to-back from the schedule and verify `rest_days == 1` for the right team.

## Recommended ordering (current iteration)

1. **Reduced C-1** — half day; team_id plumbing + score margin + `is_offense_home`. `game_date` plumbed but unused on medium (feeds C-5 later).
2. **C-2 (shot release)** — half day; pure frame-math, no external data. Targets weak `made_3` directly, also fixes suspected `pass_count_proxy` post-shot-frame leakage. No ID maps required → cheapest big-impact step.
3. **C-3** — half day; needs `team_id_map.csv` (auto-scaffold + verify).
4. **C-4** — day; needs `player_id_map.csv` via fuzzy match + ~15 min hand verification. Highest expected made_3 lift among season-stat joins.
5. **C-5** — re-enable on full-dataset run only. Skipped on medium.

After each stage, re-run medium and diff the feature importance table — the point isn't just overall accuracy, it's identifying *which* context signals land.
