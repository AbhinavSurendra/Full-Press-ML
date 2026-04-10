"""Build processed event/frame/possession tables from raw tracking data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from full_press_ml.data.possession_rules import segment_possessions
from full_press_ml.data.raw_loader import load_normalized_tracking_data


def attach_possessions_to_frames(
    frames: pd.DataFrame,
    possessions: pd.DataFrame,
) -> pd.DataFrame:
    """Assign possession IDs and labels to frame rows via event membership."""

    if frames.empty or possessions.empty:
        return frames.copy()

    event_map_rows = []
    for _, possession in possessions.iterrows():
        event_ids = str(possession["event_ids"]).split(",")
        for event_id in event_ids:
            if not event_id:
                continue
            event_map_rows.append(
                {
                    "game_id": int(possession["game_id"]),
                    "event_id": int(event_id),
                    "possession_id": possession["possession_id"],
                    "possession_number": int(possession["possession_number"]),
                    "terminal_label": possession["terminal_label"],
                    "terminal_event_id": possession["terminal_event_id"],
                    "possession_is_usable": int(possession["is_usable"]),
                }
            )

    event_map = pd.DataFrame(event_map_rows)
    merged = frames.merge(event_map, on=["game_id", "event_id"], how="left")
    merged = merged.sort_values(["game_id", "possession_number", "event_id", "frame_idx"]).reset_index(
        drop=True
    )
    merged["possession_frame_idx"] = merged.groupby(["game_id", "possession_id"]).cumcount()
    return merged


def summarize_outputs(events: pd.DataFrame, frames: pd.DataFrame, possessions: pd.DataFrame) -> dict[str, object]:
    """Build a compact audit summary for the processed dataset."""

    usable_possessions = possessions[possessions["is_usable"] == 1]
    label_counts = usable_possessions["terminal_label"].value_counts().to_dict()
    split_counts = usable_possessions["split"].value_counts().to_dict()
    return {
        "games": int(events["game_id"].nunique()) if not events.empty else 0,
        "events": int(len(events)),
        "matched_events": int((events["pbp_join_status"] == "matched").sum()) if not events.empty else 0,
        "unmatched_events": int((events["pbp_join_status"] == "missing").sum()) if not events.empty else 0,
        "frames": int(len(frames)),
        "valid_frames": int((frames["is_valid_frame"] == 1).sum()) if not frames.empty else 0,
        "invalid_frames": int((frames["is_valid_frame"] == 0).sum()) if not frames.empty else 0,
        "possessions": int(len(possessions)),
        "usable_possessions": int(len(usable_possessions)),
        "usable_label_counts": label_counts,
        "usable_split_counts": split_counts,
    }


def build_processed_datasets(
    games_dir: Path,
    pbp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build normalized events, frames, possessions, and an audit summary."""

    events, frames = load_normalized_tracking_data(games_dir=games_dir, pbp_path=pbp_path)
    matched_events = events[events["pbp_join_status"] == "matched"].copy()
    possessions = segment_possessions(matched_events)
    frames = attach_possessions_to_frames(frames=frames, possessions=possessions)
    summary = summarize_outputs(events=events, frames=frames, possessions=possessions)
    return events, frames, possessions, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", required=True, type=Path)
    parser.add_argument("--pbp", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    events, frames, possessions, summary = build_processed_datasets(
        games_dir=args.games_dir,
        pbp_path=args.pbp,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "events.csv", index=False)
    frames.to_csv(args.output_dir / "frames.csv", index=False)
    possessions.to_csv(args.output_dir / "possessions.csv", index=False)
    with (args.output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
