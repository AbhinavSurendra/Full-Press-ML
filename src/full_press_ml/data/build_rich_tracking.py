"""Build richer tracking outputs with full player coordinates preserved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from full_press_ml.data.build_possessions import summarize_outputs
from full_press_ml.data.possession_rules import infer_offense_team_id, segment_possessions
from full_press_ml.data.raw_loader import assign_game_splits


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slot_players(raw_players: list[list[object]], offense_team_id: float | None) -> list[dict[str, float | None]]:
    players: list[dict[str, float | None]] = []
    for raw_player in raw_players:
        if len(raw_player) < 5:
            continue
        player = {
            "team_id": _safe_float(raw_player[0]),
            "player_id": _safe_float(raw_player[1]),
            "x": _safe_float(raw_player[2]),
            "y": _safe_float(raw_player[3]),
            "z": _safe_float(raw_player[4]),
        }
        players.append(player)

    def _sort_key(player: dict[str, float | None]) -> tuple[int, float, float]:
        is_offense = (
            0
            if offense_team_id is not None
            and player["team_id"] is not None
            and int(player["team_id"]) == int(offense_team_id)
            else 1
        )
        team_id = player["team_id"] if player["team_id"] is not None else -1.0
        player_id = player["player_id"] if player["player_id"] is not None else -1.0
        return (is_offense, team_id, player_id)

    return sorted(players, key=_sort_key)


def _build_rich_frames_for_game(
    game: dict[str, object],
    pbp_game: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp_lookup = {int(row["EVENTNUM"]): row for _, row in pbp_game.iterrows()}
    rich_frame_rows: list[dict[str, object]] = []
    player_frame_rows: list[dict[str, object]] = []

    for event in game["events"]:
        event_id = int(event["eventId"])
        pbp_row = pbp_lookup.get(event_id)
        offense_team_id = infer_offense_team_id(pbp_row) if pbp_row is not None else None

        for frame_idx, moment in enumerate(event.get("moments", [])):
            if len(moment) < 6 or not isinstance(moment[5], list) or not moment[5]:
                continue

            coordinates = moment[5]
            ball = coordinates[0]
            players = _slot_players(coordinates[1:], offense_team_id)
            rich_row = {
                "game_id": int(game["gameid"]),
                "event_id": event_id,
                "frame_idx": frame_idx,
                "split": split,
                "offense_team_id": offense_team_id,
                "quarter": int(moment[0]),
                "game_clock": _safe_float(moment[2]),
                "shot_clock": _safe_float(moment[3]),
                "ball_x": _safe_float(ball[2]) if len(ball) > 2 else None,
                "ball_y": _safe_float(ball[3]) if len(ball) > 3 else None,
                "ball_z": _safe_float(ball[4]) if len(ball) > 4 else None,
                "player_slot_count": len(players),
                "is_valid_frame": int(len(coordinates) == 11),
            }

            for slot_idx in range(10):
                prefix = f"player_{slot_idx}"
                if slot_idx < len(players):
                    player = players[slot_idx]
                    rich_row[f"{prefix}_team_id"] = player["team_id"]
                    rich_row[f"{prefix}_player_id"] = player["player_id"]
                    rich_row[f"{prefix}_x"] = player["x"]
                    rich_row[f"{prefix}_y"] = player["y"]
                    rich_row[f"{prefix}_z"] = player["z"]
                else:
                    rich_row[f"{prefix}_team_id"] = None
                    rich_row[f"{prefix}_player_id"] = None
                    rich_row[f"{prefix}_x"] = None
                    rich_row[f"{prefix}_y"] = None
                    rich_row[f"{prefix}_z"] = None

            rich_frame_rows.append(rich_row)

            for slot_idx, player in enumerate(players):
                player_frame_rows.append(
                    {
                        "game_id": int(game["gameid"]),
                        "event_id": event_id,
                        "frame_idx": frame_idx,
                        "split": split,
                        "offense_team_id": offense_team_id,
                        "quarter": int(moment[0]),
                        "game_clock": _safe_float(moment[2]),
                        "shot_clock": _safe_float(moment[3]),
                        "slot_idx": slot_idx,
                        "team_id": player["team_id"],
                        "player_id": player["player_id"],
                        "x": player["x"],
                        "y": player["y"],
                        "z": player["z"],
                        "is_offense_player": int(
                            offense_team_id is not None
                            and player["team_id"] is not None
                            and int(player["team_id"]) == int(offense_team_id)
                        ),
                    }
                )

    rich_frames = pd.DataFrame(rich_frame_rows)
    player_frames = pd.DataFrame(player_frame_rows)
    return rich_frames, player_frames


def attach_possessions_by_event(
    table: pd.DataFrame,
    possessions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach possession metadata to any frame-like table keyed by event."""

    if table.empty or possessions.empty:
        return table.copy()

    mapping_rows = []
    for _, possession in possessions.iterrows():
        for event_id in str(possession["event_ids"]).split(","):
            if event_id:
                mapping_rows.append(
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
    mapping = pd.DataFrame(mapping_rows)
    return table.merge(mapping, on=["game_id", "event_id"], how="left")


def build_rich_processed_datasets(
    games_dir: Path,
    pbp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build rich tracking outputs with wide and long player-coordinate tables."""

    pbp = pd.read_csv(pbp_path)
    json_paths = sorted(games_dir.glob("*/*.json"))
    game_ids = []
    games = []
    for json_path in json_paths:
        with json_path.open(encoding="utf-8") as handle:
            game = json.load(handle)
        games.append(game)
        game_ids.append(int(game["gameid"]))

    split_map = assign_game_splits(game_ids)

    from full_press_ml.data.raw_loader import load_normalized_tracking_data

    events, _ = load_normalized_tracking_data(games_dir=games_dir, pbp_path=pbp_path)
    matched_events = events[events["pbp_join_status"] == "matched"].copy()
    possessions = segment_possessions(matched_events)

    rich_frames_by_game = []
    player_frames_by_game = []
    for game in games:
        game_id = int(game["gameid"])
        pbp_game = pbp[pbp["GAME_ID"] == game_id]
        split = split_map[game_id]
        rich_frames, player_frames = _build_rich_frames_for_game(game=game, pbp_game=pbp_game, split=split)
        rich_frames_by_game.append(rich_frames)
        player_frames_by_game.append(player_frames)

    rich_frames = pd.concat(rich_frames_by_game, ignore_index=True) if rich_frames_by_game else pd.DataFrame()
    player_frames = (
        pd.concat(player_frames_by_game, ignore_index=True) if player_frames_by_game else pd.DataFrame()
    )

    rich_frames = attach_possessions_by_event(rich_frames, possessions)
    player_frames = attach_possessions_by_event(player_frames, possessions)
    if not rich_frames.empty:
        rich_frames = rich_frames.sort_values(["game_id", "event_id", "frame_idx"]).reset_index(drop=True)
    if not player_frames.empty:
        player_frames = player_frames.sort_values(
            ["game_id", "event_id", "frame_idx", "slot_idx"]
        ).reset_index(drop=True)

    summary = summarize_outputs(events=events, frames=rich_frames, possessions=possessions)
    summary["player_frames"] = int(len(player_frames))
    return events, possessions, rich_frames, player_frames, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", required=True, type=Path)
    parser.add_argument("--pbp", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    events, possessions, rich_frames, player_frames, summary = build_rich_processed_datasets(
        games_dir=args.games_dir,
        pbp_path=args.pbp,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "events.csv", index=False)
    possessions.to_csv(args.output_dir / "possessions.csv", index=False)
    rich_frames.to_csv(args.output_dir / "rich_frames.csv", index=False)
    player_frames.to_csv(args.output_dir / "player_frames.csv", index=False)
    with (args.output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
