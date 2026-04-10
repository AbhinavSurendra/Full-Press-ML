"""Rules for inferring offense and possession outcomes from play-by-play."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from full_press_ml.data.schema import POSSESSION_OUTCOMES


def _contains(text: object, needle: str) -> bool:
    return needle in str(text).upper()


def parse_period_clock(clock_text: object) -> float | None:
    """Convert a play-by-play period clock like '11:46' to seconds remaining."""

    if pd.isna(clock_text):
        return None
    text = str(clock_text).strip()
    if not text or ":" not in text:
        return None
    minutes, seconds = text.split(":", maxsplit=1)
    try:
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def infer_offense_team_id(row: pd.Series) -> float | None:
    """Infer which team has possession during the event."""

    event_type = int(row["EVENTMSGTYPE"])

    if event_type in {1, 2, 3, 4, 5}:
        team_id = row.get("PLAYER1_TEAM_ID")
        return None if pd.isna(team_id) else float(team_id)

    home_desc = row.get("HOMEDESCRIPTION")
    visitor_desc = row.get("VISITORDESCRIPTION")
    if _contains(home_desc, "OFF.FOUL") or _contains(visitor_desc, "OFF.FOUL"):
        team_id = row.get("PLAYER1_TEAM_ID")
        return None if pd.isna(team_id) else float(team_id)

    if event_type == 6:
        team_id = row.get("PLAYER2_TEAM_ID")
        return None if pd.isna(team_id) else float(team_id)

    return None


def classify_terminal_event(row: pd.Series) -> str | None:
    """Map a play-by-play row into the project 5-class outcome space."""

    event_type = int(row["EVENTMSGTYPE"])
    action_type = int(row.get("EVENTMSGACTIONTYPE", 0))
    home_desc = str(row.get("HOMEDESCRIPTION") or "")
    visitor_desc = str(row.get("VISITORDESCRIPTION") or "")
    merged_desc = f"{home_desc} {visitor_desc}".upper()

    if event_type == 1:
        if "3PT" in merged_desc:
            return "made_3"
        return "made_2"

    if event_type == 2:
        return "missed_shot"

    if event_type == 5:
        return "turnover"

    if event_type == 3:
        # Action types in the 10s are substitutions/technical variants and not
        # the core possession terminal we want to label.
        if action_type >= 10 and "FREE THROW" not in merged_desc:
            return None
        return "free_throws"

    return None


def is_hard_terminal_label(label: str | None) -> bool:
    return label in {"made_2", "made_3", "turnover"}


def is_same_team(team_a: float | None, team_b: float | None) -> bool:
    if team_a is None or team_b is None:
        return False
    return int(team_a) == int(team_b)


@dataclass
class PossessionAccumulator:
    game_id: int
    possession_number: int
    start_event_id: int
    period: int
    offense_team_id: float | None
    split: str
    event_ids: list[int]
    label_candidate: str | None = None
    label_event_id: int | None = None
    free_throw_events: int = 0
    valid_frame_count: int = 0
    invalid_frame_count: int = 0
    missing_shot_clock_count: int = 0


def _finalize_possession(
    current: PossessionAccumulator | None,
    end_event_id: int | None,
    reason: str,
) -> dict[str, object] | None:
    if current is None:
        return None

    total_frames = current.valid_frame_count + current.invalid_frame_count
    invalid_ratio = (
        current.invalid_frame_count / total_frames if total_frames else 1.0
    )
    missing_shot_clock_ratio = (
        current.missing_shot_clock_count / current.valid_frame_count
        if current.valid_frame_count
        else 1.0
    )
    label = current.label_candidate
    is_usable = (
        label in POSSESSION_OUTCOMES
        and current.valid_frame_count >= 25
        and invalid_ratio <= 0.20
    )

    return {
        "game_id": current.game_id,
        "possession_id": f"{current.game_id}_{current.possession_number:04d}",
        "possession_number": current.possession_number,
        "period": current.period,
        "offense_team_id": current.offense_team_id,
        "split": current.split,
        "start_event_id": current.start_event_id,
        "end_event_id": end_event_id if end_event_id is not None else current.event_ids[-1],
        "num_events": len(current.event_ids),
        "event_ids": ",".join(str(event_id) for event_id in current.event_ids),
        "terminal_label": label,
        "terminal_event_id": current.label_event_id,
        "free_throw_events": current.free_throw_events,
        "valid_frame_count": current.valid_frame_count,
        "invalid_frame_count": current.invalid_frame_count,
        "missing_shot_clock_count": current.missing_shot_clock_count,
        "invalid_frame_ratio": invalid_ratio,
        "missing_shot_clock_ratio": missing_shot_clock_ratio,
        "end_reason": reason,
        "is_usable": int(is_usable),
    }


def segment_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """Create conservative possession segments from normalized event rows."""

    if events.empty:
        return pd.DataFrame()

    required = {
        "game_id",
        "event_id",
        "period",
        "event_msg_type",
        "offense_team_id",
        "split",
        "valid_frame_count",
        "invalid_frame_count",
        "missing_shot_clock_count",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required event columns: {sorted(missing)}")

    rows = []

    for game_id, game_events in events.groupby("game_id", sort=True):
        game_events = game_events.sort_values(
            by=["period", "clock_seconds_remaining", "event_id"],
            ascending=[True, False, True],
        )

        current: PossessionAccumulator | None = None
        possession_number = 0
        current_period: int | None = None

        for _, row in game_events.iterrows():
            row_period = int(row["period"])
            offense_team_id = row["offense_team_id"]
            offense_team_id = None if pd.isna(offense_team_id) else float(offense_team_id)
            event_id = int(row["event_id"])
            split = str(row["split"])

            if current is not None and current_period != row_period:
                finalized = _finalize_possession(current, current.event_ids[-1], "period_change")
                if finalized is not None:
                    rows.append(finalized)
                current = None

            current_period = row_period

            offense_changed = (
                current is not None
                and current.offense_team_id is not None
                and offense_team_id is not None
                and not is_same_team(current.offense_team_id, offense_team_id)
            )
            if offense_changed:
                finalized = _finalize_possession(current, current.event_ids[-1], "offense_change")
                if finalized is not None:
                    rows.append(finalized)
                current = None

            if current is None:
                possession_number += 1
                current = PossessionAccumulator(
                    game_id=int(game_id),
                    possession_number=possession_number,
                    start_event_id=event_id,
                    period=row_period,
                    offense_team_id=offense_team_id,
                    split=split,
                    event_ids=[],
                )

            current.event_ids.append(event_id)
            if current.offense_team_id is None and offense_team_id is not None:
                current.offense_team_id = offense_team_id

            current.valid_frame_count += int(row["valid_frame_count"])
            current.invalid_frame_count += int(row["invalid_frame_count"])
            current.missing_shot_clock_count += int(row["missing_shot_clock_count"])

            terminal_label = classify_terminal_event(
                pd.Series(
                    {
                        "EVENTMSGTYPE": row["event_msg_type"],
                        "EVENTMSGACTIONTYPE": row["event_msg_action_type"],
                        "HOMEDESCRIPTION": row["home_description"],
                        "VISITORDESCRIPTION": row["visitor_description"],
                    }
                )
            )
            if terminal_label == "free_throws":
                current.free_throw_events += 1
            if terminal_label is not None:
                current.label_candidate = terminal_label
                current.label_event_id = event_id

            if is_hard_terminal_label(terminal_label):
                finalized = _finalize_possession(current, event_id, "terminal_event")
                if finalized is not None:
                    rows.append(finalized)
                current = None

        finalized = _finalize_possession(current, None, "end_of_game")
        if finalized is not None:
            rows.append(finalized)

    return pd.DataFrame(rows)
