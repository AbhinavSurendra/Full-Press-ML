"""Shared dataset schema definitions."""

from __future__ import annotations

POSSESSION_OUTCOMES = [
    "made_2",
    "made_3",
    "missed_shot",
    "turnover",
    "free_throws",
]

REQUIRED_TRACKING_COLUMNS = [
    "game_id",
    "possession_id",
    "frame_idx",
    "game_clock",
    "shot_clock",
    "ball_x",
    "ball_y",
    "ball_z",
]

