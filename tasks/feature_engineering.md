# Feature Engineering Tasks

## Objective

Create possession-prefix features for baseline models.

## Tasks

1. Add spatial summaries such as ball location and distance from center.
2. Add timing features such as shot clock buckets.
3. Add possession-level aggregates for the first `k` seconds.
4. Test several horizons: `2`, `4`, `6`, `8`.
5. Export a clean baseline table for training.

## Main File

- `src/full_press_ml/features/engineer.py`

