"""Build richer tracking outputs with full player coordinates preserved."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import orjson
import pandas as pd

from full_press_ml.data.possession_rules import (
    infer_offense_team_id,
    parse_period_clock,
    segment_possessions,
)
from full_press_ml.data.raw_loader import assign_game_splits

_GAME_ID_PATTERN = re.compile(r'"gameid"\s*:\s*"?(?P<game_id>\d+)"?')

# PBP columns read by _build_rows_for_game and infer_offense_team_id. Keeping
# the lookup narrow avoids carrying ~20 extra columns per PBP row in memory.
_PBP_LOOKUP_COLUMNS = [
    "GAME_ID",
    "EVENTNUM",
    "PERIOD",
    "PCTIMESTRING",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "HOMEDESCRIPTION",
    "VISITORDESCRIPTION",
    "PLAYER1_TEAM_ID",
    "PLAYER2_TEAM_ID",
]


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


def _extract_game_id(json_path: Path) -> int:
    """Fast-path extract of game ID from the JSON header with JSON fallback."""
    with json_path.open(encoding="utf-8") as handle:
        head = handle.read(4096)
    match = _GAME_ID_PATTERN.search(head)
    if match:
        return int(match.group("game_id"))
    with json_path.open("rb") as handle:
        game = orjson.loads(handle.read())
    return int(game["gameid"])


def _build_pbp_lookup_by_game(pbp: pd.DataFrame) -> dict[int, dict[int, dict[str, object]]]:
    """Index play-by-play rows for O(1) game/event lookup as plain dicts.

    Plain dicts (not pd.Series) avoid the per-row Series construction overhead
    of iterrows. Downstream helpers only need __getitem__ / get on these rows.
    """
    available = [c for c in _PBP_LOOKUP_COLUMNS if c in pbp.columns]
    subset = pbp[available].dropna(subset=["GAME_ID", "EVENTNUM"])
    lookups: dict[int, dict[int, dict[str, object]]] = {}
    for record in subset.to_dict(orient="records"):
        game_key = int(record["GAME_ID"])
        event_key = int(record["EVENTNUM"])
        lookups.setdefault(game_key, {})[event_key] = record
    return lookups


def _build_rows_for_game(
    game: dict[str, object],
    pbp_lookup: dict[int, dict[str, object]],
    split: str,
    collect_player_rows: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Build event rows, rich frame rows, and optionally player frame rows for one game."""
    event_rows: list[dict[str, object]] = []
    rich_frame_rows: list[dict[str, object]] = []
    player_frame_rows: list[dict[str, object]] = []
    game_id = int(game["gameid"])

    for event in game["events"]:
        event_id = int(event["eventId"])
        pbp_row = pbp_lookup.get(event_id)
        moments = event.get("moments", [])
        offense_team_id = infer_offense_team_id(pbp_row) if pbp_row is not None else None
        valid_frame_count = 0
        invalid_frame_count = 0
        missing_shot_clock_count = 0

        if pbp_row is None:
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

        for frame_idx, moment in enumerate(moments):
            if len(moment) < 6 or not isinstance(moment[5], list):
                if pbp_row is not None:
                    invalid_frame_count += 1
                continue

            shot_clock = _safe_float(moment[3])
            is_valid = int(len(moment[5]) == 11)
            if pbp_row is not None:
                if shot_clock is None:
                    missing_shot_clock_count += 1
                if is_valid:
                    valid_frame_count += 1
                else:
                    invalid_frame_count += 1

            coordinates = moment[5]
            if not coordinates:
                continue

            ball = coordinates[0]
            players = _slot_players(coordinates[1:], offense_team_id)
            rich_row = {
                "game_id": game_id,
                "event_id": event_id,
                "frame_idx": frame_idx,
                "split": split,
                "offense_team_id": offense_team_id,
                "quarter": int(moment[0]),
                "game_clock": _safe_float(moment[2]),
                "shot_clock": shot_clock,
                "ball_x": _safe_float(ball[2]) if len(ball) > 2 else None,
                "ball_y": _safe_float(ball[3]) if len(ball) > 3 else None,
                "ball_z": _safe_float(ball[4]) if len(ball) > 4 else None,
                "player_slot_count": len(players),
                "is_valid_frame": is_valid,
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

            if collect_player_rows:
                for slot_idx, player in enumerate(players):
                    player_frame_rows.append(
                        {
                            "game_id": game_id,
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

        if pbp_row is not None:
            event_rows.append(
                {
                    "game_id": game_id,
                    "event_id": event_id,
                    "period": int(pbp_row["PERIOD"]),
                    "clock_seconds_remaining": parse_period_clock(pbp_row["PCTIMESTRING"]),
                    "event_msg_type": int(pbp_row["EVENTMSGTYPE"]),
                    "event_msg_action_type": int(pbp_row["EVENTMSGACTIONTYPE"]),
                    "home_description": pbp_row["HOMEDESCRIPTION"],
                    "visitor_description": pbp_row["VISITORDESCRIPTION"],
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

    return event_rows, rich_frame_rows, player_frame_rows


def attach_possessions_by_event(
    table: pd.DataFrame,
    possessions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach possession metadata to any frame-like table keyed by event."""

    if table.empty or possessions.empty:
        return table.copy()

    mapping = possessions[
        [
            "game_id",
            "event_ids",
            "possession_id",
            "possession_number",
            "terminal_label",
            "terminal_event_id",
            "is_usable",
        ]
    ].copy()
    mapping["event_id"] = mapping["event_ids"].astype(str).str.split(",")
    mapping = mapping.explode("event_id")
    mapping = mapping[mapping["event_id"].astype(str).str.len() > 0].copy()
    mapping["event_id"] = mapping["event_id"].astype(int)
    mapping["game_id"] = mapping["game_id"].astype(int)
    mapping["possession_number"] = mapping["possession_number"].astype(int)
    mapping["possession_is_usable"] = mapping["is_usable"].astype(int)
    mapping = mapping.drop(columns=["event_ids", "is_usable"])

    return table.merge(mapping, on=["game_id", "event_id"], how="left")


def _stream_attach_and_write(
    tmp_dir: Path,
    output_path: Path,
    possessions: pd.DataFrame,
    sort_cols: list[str],
) -> None:
    """Second pass: read each per-game temp CSV, attach possessions, append to final CSV."""
    if output_path.exists():
        output_path.unlink()

    first_write = True
    for tmp_csv in sorted(tmp_dir.glob("*.csv")):
        game_df = pd.read_csv(tmp_csv)
        merged = attach_possessions_by_event(game_df, possessions)
        if not merged.empty:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
        merged.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False


def build_rich_processed_datasets(
    games_dir: Path,
    pbp_path: Path,
    output_dir: Path,
    write_player_frames: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build rich tracking outputs, streaming per-game CSV writes to bound memory.

    First pass writes each game's rich_frames (and optionally player_frames) to
    a per-game temp CSV, keeping peak memory at one game's worth of rows. After
    possessions are segmented, a second pass re-reads each temp CSV, joins the
    possession metadata, and appends to the final output CSV.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_rich_dir = output_dir / "_tmp_rich"
    if tmp_rich_dir.exists():
        shutil.rmtree(tmp_rich_dir)
    tmp_rich_dir.mkdir()

    tmp_player_dir: Path | None = None
    if write_player_frames:
        tmp_player_dir = output_dir / "_tmp_player"
        if tmp_player_dir.exists():
            shutil.rmtree(tmp_player_dir)
        tmp_player_dir.mkdir()

    pbp = pd.read_csv(pbp_path)
    json_paths = sorted(games_dir.glob("*/*.json"))
    game_ids = [_extract_game_id(path) for path in json_paths]
    split_map = assign_game_splits(game_ids)
    pbp_lookups = _build_pbp_lookup_by_game(pbp)
    del pbp

    event_rows: list[dict[str, object]] = []
    total_rich_rows = 0
    total_valid_frames = 0
    total_invalid_frames = 0
    total_player_rows = 0

    for json_path in json_paths:
        with json_path.open("rb") as handle:
            game = orjson.loads(handle.read())
        game_id = int(game["gameid"])
        game_event_rows, game_rich_rows, game_player_rows = _build_rows_for_game(
            game=game,
            pbp_lookup=pbp_lookups.get(game_id, {}),
            split=split_map[game_id],
            collect_player_rows=write_player_frames,
        )
        event_rows.extend(game_event_rows)

        for row in game_rich_rows:
            if row["is_valid_frame"] == 1:
                total_valid_frames += 1
            else:
                total_invalid_frames += 1
        total_rich_rows += len(game_rich_rows)

        if game_rich_rows:
            pd.DataFrame(game_rich_rows).to_csv(
                tmp_rich_dir / f"{game_id}.csv", index=False
            )

        if write_player_frames and tmp_player_dir is not None:
            total_player_rows += len(game_player_rows)
            if game_player_rows:
                pd.DataFrame(game_player_rows).to_csv(
                    tmp_player_dir / f"{game_id}.csv", index=False
                )

        del game, game_rich_rows, game_player_rows

    events = pd.DataFrame(event_rows)
    if not events.empty:
        events = events.sort_values(["game_id", "event_id"]).reset_index(drop=True)

    matched_events = events[events["pbp_join_status"] == "matched"].copy() if not events.empty else pd.DataFrame()
    possessions = segment_possessions(matched_events)

    _stream_attach_and_write(
        tmp_dir=tmp_rich_dir,
        output_path=output_dir / "rich_frames.csv",
        possessions=possessions,
        sort_cols=["game_id", "event_id", "frame_idx"],
    )
    shutil.rmtree(tmp_rich_dir)

    if write_player_frames and tmp_player_dir is not None:
        _stream_attach_and_write(
            tmp_dir=tmp_player_dir,
            output_path=output_dir / "player_frames.csv",
            possessions=possessions,
            sort_cols=["game_id", "event_id", "frame_idx", "slot_idx"],
        )
        shutil.rmtree(tmp_player_dir)

    usable = possessions[possessions["is_usable"] == 1] if not possessions.empty else possessions
    summary: dict[str, object] = {
        "games": int(events["game_id"].nunique()) if not events.empty else 0,
        "events": int(len(events)),
        "matched_events": int((events["pbp_join_status"] == "matched").sum()) if not events.empty else 0,
        "unmatched_events": int((events["pbp_join_status"] == "missing").sum()) if not events.empty else 0,
        "frames": total_rich_rows,
        "valid_frames": total_valid_frames,
        "invalid_frames": total_invalid_frames,
        "possessions": int(len(possessions)),
        "usable_possessions": int(len(usable)),
        "usable_label_counts": usable["terminal_label"].value_counts().to_dict() if not usable.empty else {},
        "usable_split_counts": usable["split"].value_counts().to_dict() if not usable.empty else {},
    }
    if write_player_frames:
        summary["player_frames"] = total_player_rows

    return events, possessions, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", required=True, type=Path)
    parser.add_argument("--pbp", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--write-player-frames",
        action="store_true",
        help="Also write player_frames.csv (long-form per-player rows).",
    )
    args = parser.parse_args()

    events, possessions, summary = build_rich_processed_datasets(
        games_dir=args.games_dir,
        pbp_path=args.pbp,
        output_dir=args.output_dir,
        write_player_frames=args.write_player_frames,
    )

    events.to_csv(args.output_dir / "events.csv", index=False)
    possessions.to_csv(args.output_dir / "possessions.csv", index=False)
    with (args.output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
