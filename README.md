# Full Press ML

NBA possession outcome prediction using Python and PyTorch.

## Goal

Predict how an NBA possession ends from partial tracking data. The project is organized around a 5-class classification task:

1. made 2
2. made 3
3. missed shot
4. turnover
5. free throws

The main analytical question is: **how early in a possession can the final outcome be predicted?**

## Tech Stack

- Python 3.9+
- PyTorch
- pandas / numpy
- scikit-learn
- XGBoost

## Project Structure

```text
src/full_press_ml/
  data/         data loading, schemas, possession extraction
  features/     feature engineering for tabular models
  models/       baseline models and PyTorch sequence models
  training/     train loops and experiment entrypoints
  evaluation/   metrics, reports, and analysis helpers
scripts/        thin wrappers for common tasks
tasks/          task breakdown for the team
configs/        experiment configuration files
data/           raw and processed datasets
tests/          unit tests
```

## Data Format

The raw dataset is game-level tracking data from the 2015-16 NBA season. Each game is stored as a JSON object with three top-level keys:

- `gameid`: NBA game identifier
- `gamedate`: game date
- `events`: list of play-by-play events with tracking data attached

Each event contains metadata plus a sequence of tracking `moments`. The important event-level fields are:

- `eventId`: event identifier within the game
- `home` / `visitor`: team metadata and active players
- `moments`: frame-by-frame tracking data sampled at `25 Hz`

Each tracking moment contains:

- `quarter`
- `game_clock`
- `shot_clock`
- `ball_coordinates`: `x`, `y`, `z`
- `player_coordinates`: list of 10 player positions with `teamid`, `playerid`, `x`, `y`, `z`

This means the raw data is naturally hierarchical:

1. game
2. event
3. moment
4. player / ball coordinates

For modeling, we will flatten this into two main derived tables:

- a frame-level table for sequence models
- a possession-level table for baseline classifiers

## Local Raw Data Layout

The repository uses `data/raw/` for downloaded source files.

For the tiny setup, the expected structure is:

```text
data/raw/
  nba_tracking_data_15_16/   dataset metadata and loader reference
  tiny/
    archives/                downloaded .7z game archives
    games/                   extracted per-game JSON files
    2015-16_pbp.csv          play-by-play table
    manifest.json            record of downloaded games
```

The tiny split contains `5` games, which is enough to validate the pipeline before downloading larger subsets.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Pipeline Entry Points

The repository uses a split between:

- `src/full_press_ml/...` for implementation code
- `scripts/...` for thin CLI wrappers that call into the package

For example, `scripts/build_possessions.py` is just a command-line entrypoint for `src/full_press_ml/data/build_possessions.py`. Both are intentional and should stay aligned.

## Current Pipeline Commands

Standard processed dataset:

```bash
python scripts/build_possessions.py \
  --games-dir data/raw/tiny/games \
  --pbp data/raw/tiny/2015-16_pbp.csv \
  --output-dir data/processed/standard
```

Rich tracking export:

```bash
python scripts/build_rich_tracking.py \
  --games-dir data/raw/tiny/games \
  --pbp data/raw/tiny/2015-16_pbp.csv \
  --output-dir data/processed/rich
```

Sequence model:

```bash
python scripts/train_lstm.py --data data/processed/standard/frames.csv --label-column label_id
```

## Recommended Execution Order

1. Build normalized event, frame, and possession tables from raw games plus play-by-play.
2. Validate the possession segmentation and 5-class labeling scheme.
3. Add label ids and feature preprocessing for model-ready datasets.
4. Train baseline tabular models and the PyTorch LSTM on possession prefixes.
5. Compare horizon-based results and write up the analysis.

See [PIPELINE_OVERVIEW.md](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/PIPELINE_OVERVIEW.md) for the current pipeline design, remaining gaps, and model-ingestion notes. See [PROJECT_PLAN.md](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/PROJECT_PLAN.md) and the files in [tasks](/Users/abhinavsurendra/Documents/Important/Work/GitHub%20Projects/Full%20Press%20ML/tasks) for the work breakdown.
