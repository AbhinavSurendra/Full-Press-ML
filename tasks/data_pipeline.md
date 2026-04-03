# Data Pipeline Tasks

## Objective

Turn raw NBA tracking data into clean possession-level and frame-level training datasets.

## Tasks

1. Validate the raw tracking schema and column availability.
2. Define how possessions begin and end.
3. Build the possession extraction script.
4. Map terminal events into the 5-class label space.
5. Manually inspect sampled possessions for label quality.
6. Save clean outputs to `data/processed/`.

## Output Files

- `data/processed/possessions.csv`
- `data/processed/frames.csv`

