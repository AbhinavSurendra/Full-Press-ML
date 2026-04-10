"""Load and normalize raw tracking JSON plus play-by-play data."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

from full_press_ml.data.possession_rules import infer_offense_team_id, parse_period_clock


def assign_game_splits(game_ids: list[int], seed: int = 9) -> dict[int, str]:
    """Assign deterministic train/val/test splits at the game level."""

    shuffled = sorted(game_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return {}

    if total == 1:
        return {shuffled[0]: "train"}
    if total == 2:
        return {shuffled[0]: "train", shuffled[1]: "test"}

    train_count = max(1, round(total * 0.6))
    val_count = max(1, round(total * 0.2))
    if train_count + val_count >= total:
        val_count = 1
        train_count = total - 2
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        train_count = max(1, train_count - 1)

    split_map: dict[int, str] = {}
    for index, game_id in enumerate(shuffled):
        if index < train_count:
            split_map[game_id] = "train"
        elif index < train_count + val_count:
            split_map[game_id] = "val"
        else:
            split_map[game_id] = "test"
    return split_map


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _player_summary(players: list[dict[str, float]], ball_xyz: tuple[float | None, float | None, float | None]) -> dict[str, float | None]:
    if not players:
        return {
            "centroid_x": None,
            "centroid_y": None,
            "mean_radius": None,
            "mean_distance_to_ball": None,
        }

    xs = [player["x"] for player in players]
    ys = [player["y"] for player in players]
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    mean_radius = (
        sum(math.hypot(player["x"] - centroid_x, player["y"] - centroid_y) for player in players)
        / len(players)
    )

    ball_x, ball_y, _ = ball_xyz
    if ball_x is None or ball_y is None:
        mean_distance_to_ball = None
    else:
        mean_distance_to_ball = (
            sum(math.hypot(player["x"] - ball_x, player["y"] - ball_y) for player in players)
            / len(players)
        )

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "mean_radius": mean_radius,
        "mean_distance_to_ball": mean_distance_to_ball,
    }


def _flatten_frame_features(moment: list[object], offense_team_id: float | None) -> tuple[dict[str, object], bool]:
    if len(moment) < 6 or not isinstance(moment[5], list):
        return {}, False

    coordinates = moment[5]
    is_valid = len(coordinates) == 11
    ball = coordinates[0] if coordinates else [None, None, None, None, None]
    ball_x = _safe_float(ball[2]) if len(ball) > 2 else None
    ball_y = _safe_float(ball[3]) if len(ball) > 3 else None
    ball_z = _safe_float(ball[4]) if len(ball) > 4 else None

    players = []
    for raw_player in coordinates[1:]:
        if len(raw_player) < 5:
            continue
        team_id = _safe_float(raw_player[0])
        player_id = _safe_float(raw_player[1])
        x = _safe_float(raw_player[2])
        y = _safe_float(raw_player[3])
        z = _safe_float(raw_player[4])
        if team_id is None or player_id is None or x is None or y is None or z is None:
            continue
        players.append(
            {
                "team_id": team_id,
                "player_id": player_id,
                "x": x,
                "y": y,
                "z": z,
            }
        )

    offense_players = []
    defense_players = []
    for player in players:
        if offense_team_id is not None and int(player["team_id"]) == int(offense_team_id):
            offense_players.append(player)
        else:
            defense_players.append(player)

    offense_summary = _player_summary(offense_players, (ball_x, ball_y, ball_z))
    defense_summary = _player_summary(defense_players, (ball_x, ball_y, ball_z))

    return {
        "quarter": int(moment[0]),
        "game_clock": _safe_float(moment[2]),
        "shot_clock": _safe_float(moment[3]),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_z": ball_z,
        "player_count": len(players),
        "offense_player_count": len(offense_players),
        "defense_player_count": len(defense_players),
        "offense_centroid_x": offense_summary["centroid_x"],
        "offense_centroid_y": offense_summary["centroid_y"],
        "offense_mean_radius": offense_summary["mean_radius"],
        "offense_mean_distance_to_ball": offense_summary["mean_distance_to_ball"],
        "defense_centroid_x": defense_summary["centroid_x"],
        "defense_centroid_y": defense_summary["centroid_y"],
        "defense_mean_radius": defense_summary["mean_radius"],
        "defense_mean_distance_to_ball": defense_summary["mean_distance_to_ball"],
    }, is_valid


def load_normalized_tracking_data(
    games_dir: Path,
    pbp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize raw game JSON files into event-level and frame-level tables."""

    pbp = pd.read_csv(pbp_path)
    json_paths = sorted(games_dir.glob("*/*.json"))
    game_ids = []
    for json_path in json_paths:
        with json_path.open(encoding="utf-8") as handle:
            game = json.load(handle)
        game_ids.append(int(game["gameid"]))
    split_map = assign_game_splits(game_ids)

    event_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []

    for json_path in json_paths:
        with json_path.open(encoding="utf-8") as handle:
            game = json.load(handle)

        game_id = int(game["gameid"])
        split = split_map[game_id]
        pbp_game = pbp[pbp["GAME_ID"] == game_id]
        pbp_lookup = {
            int(row["EVENTNUM"]): row
            for _, row in pbp_game.iterrows()
        }

        for event in game["events"]:
            event_id = int(event["eventId"])
            event_row = pbp_lookup.get(event_id)
            moments = event.get("moments", [])

            if event_row is None:
                event_rows.append(
                    {
                        "game_id": game_id,
                        "event_id": event_id,
                        "period": None,
                        "clock_seconds_remaining": None,
                        "event_msg_type": None,
                        "event_msg_action_type": None,
                        "home_description": None,
                        "visitor_description": None,
                        "offense_team_id": None,
                        "num_moments": len(moments),
                        "valid_frame_count": 0,
                        "invalid_frame_count": len(moments),
                        "missing_shot_clock_count": 0,
                        "has_tracking": int(bool(moments)),
                        "pbp_join_status": "missing",
                        "split": split,
                    }
                )
                continue

            offense_team_id = infer_offense_team_id(event_row)
            valid_frame_count = 0
            invalid_frame_count = 0
            missing_shot_clock_count = 0

            for frame_idx, moment in enumerate(moments):
                frame_features, is_valid = _flatten_frame_features(moment, offense_team_id)
                if not frame_features:
                    invalid_frame_count += 1
                    continue

                if frame_features["shot_clock"] is None:
                    missing_shot_clock_count += 1
                if is_valid:
                    valid_frame_count += 1
                else:
                    invalid_frame_count += 1

                frame_rows.append(
                    {
                        "game_id": game_id,
                        "event_id": event_id,
                        "frame_idx": frame_idx,
                        "split": split,
                        "offense_team_id": offense_team_id,
                        "pbp_join_status": "matched",
                        "is_valid_frame": int(is_valid),
                        "missing_shot_clock": int(frame_features["shot_clock"] is None),
                        **frame_features,
                    }
                )

            event_rows.append(
                {
                    "game_id": game_id,
                    "event_id": event_id,
                    "period": int(event_row["PERIOD"]),
                    "clock_seconds_remaining": parse_period_clock(event_row["PCTIMESTRING"]),
                    "event_msg_type": int(event_row["EVENTMSGTYPE"]),
                    "event_msg_action_type": int(event_row["EVENTMSGACTIONTYPE"]),
                    "home_description": event_row["HOMEDESCRIPTION"],
                    "visitor_description": event_row["VISITORDESCRIPTION"],
                    "offense_team_id": offense_team_id,
                    "num_moments": len(moments),
                    "valid_frame_count": valid_frame_count,
                    "invalid_frame_count": invalid_frame_count,
                    "missing_shot_clock_count": missing_shot_clock_count,
                    "has_tracking": int(bool(moments)),
                    "pbp_join_status": "matched",
                    "split": split,
                }
            )

    events = pd.DataFrame(event_rows).sort_values(["game_id", "event_id"]).reset_index(drop=True)
    frames = pd.DataFrame(frame_rows).sort_values(["game_id", "event_id", "frame_idx"]).reset_index(drop=True)
    return events, frames
