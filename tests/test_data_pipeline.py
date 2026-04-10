import pandas as pd

from full_press_ml.data.build_possessions import attach_possessions_to_frames
from full_press_ml.data.build_rich_tracking import attach_possessions_by_event
from full_press_ml.data.possession_rules import classify_terminal_event, segment_possessions
from full_press_ml.data.tracking_dataset import PossessionSequenceDataset


def test_classify_terminal_event_maps_project_labels() -> None:
    made_three = pd.Series(
        {
            "EVENTMSGTYPE": 1,
            "EVENTMSGACTIONTYPE": 0,
            "HOMEDESCRIPTION": "Player makes 24' 3PT Jump Shot",
            "VISITORDESCRIPTION": None,
        }
    )
    turnover = pd.Series(
        {
            "EVENTMSGTYPE": 5,
            "EVENTMSGACTIONTYPE": 0,
            "HOMEDESCRIPTION": "Lost Ball Turnover",
            "VISITORDESCRIPTION": None,
        }
    )
    free_throw = pd.Series(
        {
            "EVENTMSGTYPE": 3,
            "EVENTMSGACTIONTYPE": 11,
            "HOMEDESCRIPTION": "Free Throw 2 of 2",
            "VISITORDESCRIPTION": None,
        }
    )

    assert classify_terminal_event(made_three) == "made_3"
    assert classify_terminal_event(turnover) == "turnover"
    assert classify_terminal_event(free_throw) == "free_throws"


def test_segment_possessions_keeps_offensive_rebound_in_same_possession() -> None:
    events = pd.DataFrame(
        [
            {
                "game_id": 1,
                "event_id": 1,
                "period": 1,
                "clock_seconds_remaining": 720.0,
                "event_msg_type": 2,
                "event_msg_action_type": 0,
                "home_description": "MISS Player Layup",
                "visitor_description": None,
                "offense_team_id": 10.0,
                "valid_frame_count": 30,
                "invalid_frame_count": 0,
                "missing_shot_clock_count": 0,
                "split": "train",
            },
            {
                "game_id": 1,
                "event_id": 2,
                "period": 1,
                "clock_seconds_remaining": 719.0,
                "event_msg_type": 4,
                "event_msg_action_type": 0,
                "home_description": "OFF REBOUND",
                "visitor_description": None,
                "offense_team_id": 10.0,
                "valid_frame_count": 20,
                "invalid_frame_count": 0,
                "missing_shot_clock_count": 0,
                "split": "train",
            },
            {
                "game_id": 1,
                "event_id": 3,
                "period": 1,
                "clock_seconds_remaining": 715.0,
                "event_msg_type": 1,
                "event_msg_action_type": 0,
                "home_description": "Player 2' Layup",
                "visitor_description": None,
                "offense_team_id": 10.0,
                "valid_frame_count": 25,
                "invalid_frame_count": 0,
                "missing_shot_clock_count": 0,
                "split": "train",
            },
        ]
    )

    possessions = segment_possessions(events)

    assert len(possessions) == 1
    assert possessions.iloc[0]["terminal_label"] == "made_2"
    assert possessions.iloc[0]["num_events"] == 3


def test_segment_possessions_ends_missed_shot_on_defensive_rebound_change() -> None:
    events = pd.DataFrame(
        [
            {
                "game_id": 7,
                "event_id": 10,
                "period": 1,
                "clock_seconds_remaining": 600.0,
                "event_msg_type": 2,
                "event_msg_action_type": 0,
                "home_description": "MISS Player Jumper",
                "visitor_description": None,
                "offense_team_id": 10.0,
                "valid_frame_count": 35,
                "invalid_frame_count": 0,
                "missing_shot_clock_count": 0,
                "split": "train",
            },
            {
                "game_id": 7,
                "event_id": 11,
                "period": 1,
                "clock_seconds_remaining": 599.0,
                "event_msg_type": 4,
                "event_msg_action_type": 0,
                "home_description": "DEF REBOUND",
                "visitor_description": None,
                "offense_team_id": 20.0,
                "valid_frame_count": 10,
                "invalid_frame_count": 0,
                "missing_shot_clock_count": 0,
                "split": "train",
            },
            {
                "game_id": 7,
                "event_id": 12,
                "period": 1,
                "clock_seconds_remaining": 594.0,
                "event_msg_type": 5,
                "event_msg_action_type": 0,
                "home_description": None,
                "visitor_description": "Turnover",
                "offense_team_id": 20.0,
                "valid_frame_count": 28,
                "invalid_frame_count": 0,
                "missing_shot_clock_count": 0,
                "split": "train",
            },
        ]
    )

    possessions = segment_possessions(events)

    assert len(possessions) == 2
    assert possessions.iloc[0]["terminal_label"] == "missed_shot"
    assert possessions.iloc[1]["terminal_label"] == "turnover"


def test_attach_possessions_to_frames_assigns_stable_possession_frame_index() -> None:
    frames = pd.DataFrame(
        [
            {"game_id": 1, "event_id": 1, "frame_idx": 0},
            {"game_id": 1, "event_id": 1, "frame_idx": 1},
            {"game_id": 1, "event_id": 2, "frame_idx": 0},
        ]
    )
    possessions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "possession_id": "1_0001",
                "possession_number": 1,
                "event_ids": "1,2",
                "terminal_label": "made_2",
                "terminal_event_id": 2,
                "is_usable": 1,
            }
        ]
    )

    attached = attach_possessions_to_frames(frames=frames, possessions=possessions)

    assert attached["possession_id"].nunique() == 1
    assert attached["possession_frame_idx"].tolist() == [0, 1, 2]


def test_attach_possessions_by_event_preserves_frame_like_rows() -> None:
    table = pd.DataFrame(
        [
            {"game_id": 1, "event_id": 4, "frame_idx": 0, "player_0_x": 12.0},
            {"game_id": 1, "event_id": 5, "frame_idx": 0, "player_0_x": 18.0},
        ]
    )
    possessions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "possession_id": "1_0002",
                "possession_number": 2,
                "event_ids": "4,5",
                "terminal_label": "made_3",
                "terminal_event_id": 5,
                "is_usable": 1,
            }
        ]
    )

    attached = attach_possessions_by_event(table=table, possessions=possessions)

    assert list(attached["possession_id"]) == ["1_0002", "1_0002"]
    assert list(attached["terminal_label"]) == ["made_3", "made_3"]


def test_possession_sequence_dataset_prefers_possession_frame_order() -> None:
    frame_table = pd.DataFrame(
        [
            {"game_id": 1, "possession_id": "1_0001", "frame_idx": 0, "possession_frame_idx": 2, "feat": 2.0, "label_id": 4},
            {"game_id": 1, "possession_id": "1_0001", "frame_idx": 1, "possession_frame_idx": 1, "feat": 1.0, "label_id": 4},
            {"game_id": 1, "possession_id": "1_0001", "frame_idx": 0, "possession_frame_idx": 0, "feat": 0.0, "label_id": 4},
        ]
    )

    dataset = PossessionSequenceDataset(
        frame_table=frame_table,
        feature_columns=["feat"],
        label_column="label_id",
        max_len=3,
    )

    sequence, label = dataset[0]

    assert sequence[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert int(label) == 4
