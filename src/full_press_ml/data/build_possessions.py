"""Build possession-level examples from raw tracking data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def validate_input_frame(df: pd.DataFrame) -> None:
    required = {"game_id", "possession_id", "frame_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def build_possessions(df: pd.DataFrame) -> pd.DataFrame:
    validate_input_frame(df)
    grouped = df.groupby(["game_id", "possession_id"], as_index=False)
    possession_df = grouped.agg(
        start_frame=("frame_idx", "min"),
        end_frame=("frame_idx", "max"),
        num_frames=("frame_idx", "count"),
        mean_ball_x=("ball_x", "mean"),
        mean_ball_y=("ball_y", "mean"),
        mean_shot_clock=("shot_clock", "mean"),
    )
    if "possession_outcome" in df.columns:
        labels = (
            df.sort_values("frame_idx")
            .groupby(["game_id", "possession_id"], as_index=False)["possession_outcome"]
            .last()
        )
        possession_df = possession_df.merge(labels, on=["game_id", "possession_id"], how="left")
    return possession_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    possession_df = build_possessions(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    possession_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

