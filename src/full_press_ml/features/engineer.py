"""Feature engineering for possession-level baselines."""

from __future__ import annotations

import pandas as pd


def add_basic_tracking_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    if {"ball_x", "ball_y"}.issubset(feature_df.columns):
        feature_df["ball_distance_from_center"] = (
            (feature_df["ball_x"] ** 2 + feature_df["ball_y"] ** 2) ** 0.5
        )
    if "shot_clock" in feature_df.columns:
        feature_df["shot_clock_low"] = (feature_df["shot_clock"] <= 8).astype(int)
    return feature_df


def build_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = add_basic_tracking_features(df)
    id_columns = [col for col in ["game_id", "possession_id", "possession_outcome"] if col in df.columns]
    numeric_columns = [col for col in feature_df.select_dtypes(include="number").columns if col not in id_columns]
    grouped = feature_df.groupby(["game_id", "possession_id"], as_index=False)[numeric_columns].mean()
    if "possession_outcome" in feature_df.columns:
        labels = feature_df.groupby(["game_id", "possession_id"], as_index=False)["possession_outcome"].last()
        grouped = grouped.merge(labels, on=["game_id", "possession_id"], how="left")
    return grouped

