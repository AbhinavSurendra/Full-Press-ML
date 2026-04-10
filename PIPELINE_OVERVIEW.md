# Pipeline Overview

This document describes the current data cleaning and preparation pipeline in the repository, what still needs to be built or tightened, and how the outputs can be fed into a model.

## Current Pipeline

The implemented pipeline starts from raw SportVU-style game JSON and a play-by-play CSV, then produces cleaned event-, frame-, and possession-level tables.

### 1. Raw Inputs

The pipeline expects:

- Game JSON files under a `games/` directory, one file per game.
- A play-by-play CSV with NBA event metadata.

The main loader entrypoint is [`load_normalized_tracking_data`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/raw_loader.py#L158).

### 2. Game-Level Split Assignment

[`assign_game_splits`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/raw_loader.py#L14) assigns each game to `train`, `val`, or `test` using a deterministic shuffle seed. This happens before any frame or possession rows are emitted, so every derived row for a game inherits the same split.

This is important because it prevents leakage across splits when multiple possessions come from the same game.

### 3. Event and Frame Normalization

[`load_normalized_tracking_data`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/raw_loader.py#L158) does the first real cleaning pass:

- Reads all game JSON files.
- Joins each tracking event to the play-by-play row using `EVENTNUM`.
- Marks unmatched events with `pbp_join_status = "missing"`.
- Flattens each tracking moment into a frame row.
- Extracts ball coordinates, shot clock, game clock, and simple team-shape summaries.
- Counts valid vs invalid frames per event.

[`_flatten_frame_features`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/raw_loader.py#L94) currently emits:

- Ball position: `ball_x`, `ball_y`, `ball_z`
- Time context: `quarter`, `game_clock`, `shot_clock`
- Player counts
- Offense and defense centroids
- Offense and defense mean spatial spread
- Offense and defense mean distance to the ball

Frame validity is currently defined conservatively:

- A moment must have a coordinates list.
- A fully valid tracking frame has exactly 11 tracked objects: 1 ball + 10 players.

### 4. Offense Inference and Possession Labeling

Possessions are built from matched event rows only. The main logic lives in [`segment_possessions`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/possession_rules.py#L158).

Supporting rules:

- [`infer_offense_team_id`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/possession_rules.py#L28) infers which team has the ball from play-by-play metadata.
- [`classify_terminal_event`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/possession_rules.py#L49) maps play-by-play events into the project’s 5 labels:
  - `made_2`
  - `made_3`
  - `missed_shot`
  - `turnover`
  - `free_throws`

Possessions are segmented by:

- Period changes
- Offense team changes
- Hard terminal events such as made shots and turnovers

The possession table also records quality metadata:

- Number of events in the possession
- Number of valid and invalid frames
- Missing shot-clock counts
- End reason
- `is_usable`

Right now a possession is marked usable if:

- Its label is one of the supported outcomes
- It has at least 25 valid frames
- Its invalid-frame ratio is at most 20%

### 5. Frame-to-Possession Attachment

[`build_processed_datasets`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/build_possessions.py#L72) produces four outputs:

- `events`
- `frames`
- `possessions`
- `audit_summary`

[`attach_possessions_to_frames`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/build_possessions.py#L14) joins possession metadata onto frame rows using `game_id + event_id`.

It also creates `possession_frame_idx`, which is the running frame index across the full possession. This matters because the raw `frame_idx` resets at the start of each event.

### 6. Rich Tracking Export

There is also a richer export path in [`build_rich_processed_datasets`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/build_rich_tracking.py#L167).

This path preserves:

- Wide per-frame player slots in `rich_frames`
- Long-form per-player coordinates in `player_frames`

The player slots are sorted with offense-first ordering, based on the inferred offense team for the event.

## Expected Outputs

The current build scripts are designed to emit CSV tables and a JSON audit summary.

### Standard processed dataset

The standard path writes:

- `events.csv`
- `frames.csv`
- `possessions.csv`
- `audit_summary.json`

See [`build_possessions.py`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/build_possessions.py#L86).

### Rich processed dataset

The rich path writes:

- `events.csv`
- `possessions.csv`
- `rich_frames.csv`
- `player_frames.csv`
- `audit_summary.json`

See [`build_rich_tracking.py`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/build_rich_tracking.py#L220).

## How To Feed This Into a Model

There are two practical modeling paths with the current outputs.

### 1. Sequence model from `frames.csv`

[`PossessionSequenceDataset`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/data/tracking_dataset.py#L19) groups the frame table by `game_id` and `possession_id`, sorts rows in possession order, truncates to `max_len`, and pads sequences for batching.

Current sequence-model flow:

1. Build processed data with `frames.csv` containing possession metadata and labels.
2. Filter to usable possessions if desired.
3. Convert the label column into numeric ids such as `0..4`.
4. Select only numeric feature columns.
5. Train with [`train_lstm.py`](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/src/full_press_ml/training/train_lstm.py#L38).

Important details:

- `possession_frame_idx` should be used for ordering, not as a model feature.
- Rows should be filtered to one split at a time before dataset construction.
- Labels should be constant within a possession because they describe the terminal outcome.

Minimal example:

```bash
python scripts/build_possessions.py --games-dir data/raw/tiny/games --pbp data/raw/tiny/2015-16_pbp.csv --output-dir data/processed/standard
python scripts/train_lstm.py --data data/processed/standard/frames.csv --label-column label_id --max-len 100
```

The training script currently assumes the input CSV is already model-ready. That means label ids and any final feature filtering or imputation should happen before calling it.

### 2. Tabular model from `possessions.csv`

The possession table can be turned into one row per possession for classical models such as logistic regression, random forests, or XGBoost.

Useful starting features already present:

- `valid_frame_count`
- `invalid_frame_count`
- `missing_shot_clock_count`
- `num_events`
- `period`
- `offense_team_id`

This is not yet a strong baseline by itself. A proper tabular baseline should add engineered aggregates from the frame table such as:

- Early-possession ball location summaries
- Team spacing summaries over the first `N` frames
- Motion features such as velocity and acceleration
- Frame-window statistics at different horizons

## What Still Needs To Be Done

The pipeline is substantially further along than the original README suggests, but it is not end-to-end production-ready yet.

### Data and labeling work

- Validate the possession segmentation rules on a larger sample of real games.
- Review edge cases around fouls, free-throw sequences, jump balls, end-of-period possessions, and ambiguous offense inference.
- Decide whether unmatched play-by-play events should be excluded entirely or surfaced in downstream quality reports more explicitly.
- Reconcile the rich-tracking audit summary with the canonical invalid-frame accounting so malformed moments are not silently undercounted.

### Feature work

- Add explicit feature-generation code for both sequence and tabular pipelines.
- Create a stable label-id mapping from string labels to class ids.
- Decide which frame columns are safe numeric features and which are metadata only.
- Add missing-value handling and normalization for model inputs.

### Training pipeline work

- Add split-aware training and evaluation entrypoints rather than reading a single mixed CSV.
- Train on possession prefixes at multiple horizons to answer the project’s main question: how early can the terminal outcome be predicted?
- Add validation metrics and saved artifacts such as checkpoints, confusion matrices, and per-class results.
- Add a baseline tabular training script that consumes engineered possession-level features.

### Testing and documentation work

- Add loader tests covering malformed moments, missing play-by-play matches, and split assignment.
- Add tests covering possession labeling edge cases beyond rebounds and turnovers.
- Update the main README so its commands and dataset descriptions match the current code.

## Practical Next Step

The fastest path to a usable experiment loop is:

1. Build `frames.csv` and `possessions.csv` from a small set of games.
2. Add a preprocessing step that maps `terminal_label -> label_id` and filters to `is_usable == 1`.
3. Train the current LSTM on standardized numeric frame features.
4. Add prefix-based truncation during training and evaluation.
5. Compare prediction quality at multiple possession horizons.

That would give the project its first complete modeling loop without requiring a full redesign of the current pipeline.
