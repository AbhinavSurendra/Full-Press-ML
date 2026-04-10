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
        #Changed to 6 seconds for low shot clock
        feature_df["shot_clock_low"] = (feature_df["shot_clock"] <= 6).astype(int)
    return feature_df


def build_possession_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = add_basic_tracking_features(df)
    id_columns = [
        col
        for col in ["game_id", "possession_id", "possession_number", "terminal_label", "split", "possession_is_usable"]
        if col in feature_df.columns
    ]
    numeric_columns = [col for col in feature_df.select_dtypes(include="number").columns if col not in id_columns]
    grouped = feature_df.groupby(["game_id", "possession_id"], as_index=False)[numeric_columns].mean()
    meta_columns = [
        col
        for col in ["possession_number", "terminal_label", "split", "possession_is_usable"]
        if col in feature_df.columns
    ]
    if meta_columns:
        labels = feature_df.groupby(["game_id", "possession_id"], as_index=False)[meta_columns].last()
        grouped = grouped.merge(labels, on=["game_id", "possession_id"], how="left")
    return grouped


def build_frame_aggregate_table(frame_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = add_basic_tracking_features(frame_df)
    feature_df = feature_df.sort_values(["game_id", "possession_id", "possession_frame_idx"])

    base_group = ["game_id", "possession_id"]
    aggregations: dict[str, list[str]] = {}
    for col in [
        "game_clock",
        "shot_clock",
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_distance_from_center",
        "offense_centroid_x",
        "offense_centroid_y",
        "offense_mean_radius",
        "offense_mean_distance_to_ball",
        "defense_centroid_x",
        "defense_centroid_y",
        "defense_mean_radius",
        "defense_mean_distance_to_ball",
        "missing_shot_clock",
        "is_valid_frame",
    ]:
        if col in feature_df.columns:
            aggregations[col] = ["mean", "std", "min", "max"]

    if "shot_clock_low" in feature_df.columns:
        aggregations["shot_clock_low"] = ["mean"]

    grouped = feature_df.groupby(base_group).agg(aggregations)
    grouped.columns = ["_".join(part for part in col if part).rstrip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    first_last_features = []
    for suffix, frame_slice in [
        ("start", feature_df.groupby(base_group, as_index=False).head(1)),
        ("end", feature_df.groupby(base_group, as_index=False).tail(1)),
    ]:
        keep_cols = [
            col
            for col in [
                "game_id",
                "possession_id",
                "game_clock",
                "shot_clock",
                "ball_x",
                "ball_y",
                "ball_z",
                "offense_mean_radius",
                "offense_mean_distance_to_ball",
                "defense_mean_radius",
                "defense_mean_distance_to_ball",
            ]
            if col in frame_slice.columns
        ]
        slice_df = frame_slice[keep_cols].copy()
        rename_map = {col: f"{col}_{suffix}" for col in keep_cols if col not in base_group}
        first_last_features.append(slice_df.rename(columns=rename_map))

    for slice_df in first_last_features:
        grouped = grouped.merge(slice_df, on=base_group, how="left")

    for source in ["ball_x", "ball_y", "shot_clock", "game_clock"]:
        start_col = f"{source}_start"
        end_col = f"{source}_end"
        if start_col in grouped.columns and end_col in grouped.columns:
            grouped[f"{source}_delta"] = grouped[end_col] - grouped[start_col]

    meta_columns = [
        col
        for col in ["possession_number", "terminal_label", "split", "possession_is_usable"]
        if col in feature_df.columns
    ]
    if meta_columns:
        meta = feature_df.groupby(base_group, as_index=False)[meta_columns].last()
        grouped = grouped.merge(meta, on=base_group, how="left")

    return grouped
