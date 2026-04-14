"""Feature engineering for possession-level baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# SportVU court dimensions (feet): 94 long x 50 wide
# Hoops at x ≈ 5.25 and x ≈ 88.75, mid-court y ≈ 25
_HOOP_LEFT_X = 5.25
_HOOP_RIGHT_X = 88.75
_HOOP_Y = 25.0
_HALF_COURT_X = 47.0
_FRAME_RATE = 25.0  # Hz

# NBA paint region (both ends): x ∈ [0,19] or [75,94], y ∈ [17,33]
_PAINT_DEPTH = 19.0
_PAINT_Y_LOW = 17.0
_PAINT_Y_HIGH = 33.0

# Ball step distance threshold for pass proxy (ft per frame at 25 Hz).
# 5 ft/frame = 125 ft/s — physically impossible via dribbling; consistent with pass or shot.
_PASS_STEP_THRESHOLD = 5.0


def _convex_hull_area(pts: np.ndarray) -> float:
    """Convex hull area (ft²) for a (K, 2) array; NaN if < 3 valid points or collinear."""
    valid = pts[~np.isnan(pts).any(axis=1)]
    if len(valid) < 3:
        return float("nan")
    try:
        return ConvexHull(valid).volume  # .volume == area in 2D
    except Exception:
        return float("nan")


def add_basic_tracking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-frame derived features to a frame-level DataFrame.

    All features here depend only on single-row values (no consecutive-frame
    lookups). Sequence-based features (ball speed, direction) are added by
    _add_motion_features(), which requires the DataFrame to be sorted within
    possessions first.
    """
    feature_df = df.copy()

    # --- existing features ------------------------------------------------
    if {"ball_x", "ball_y"}.issubset(feature_df.columns):
        feature_df["ball_distance_from_center"] = (
            (feature_df["ball_x"] ** 2 + feature_df["ball_y"] ** 2) ** 0.5
        )

    if "shot_clock" in feature_df.columns:
        feature_df["shot_clock_low"] = (feature_df["shot_clock"] <= 6).astype(int)

    # --- NEW: shot clock bucket -------------------------------------------
    # 0 = desperation (0–5 s), 1 = normal clock (5–16 s), 2 = early (>16 s)
    if "shot_clock" in feature_df.columns:
        feature_df["shot_clock_bucket"] = pd.cut(
            feature_df["shot_clock"],
            bins=[-0.001, 5.0, 16.0, float("inf")],
            labels=[0, 1, 2],
        ).astype(float)

    # --- NEW: ball distance to attacking hoop ----------------------------
    # Determine attacking end per frame: if offense centroid is on the left
    # half of court (x < 47) they are attacking the right basket and vice versa.
    if {"ball_x", "ball_y", "offense_centroid_x"}.issubset(feature_df.columns):
        attacking_right = (feature_df["offense_centroid_x"] < _HALF_COURT_X).astype(float)
        hoop_x = attacking_right * _HOOP_RIGHT_X + (1.0 - attacking_right) * _HOOP_LEFT_X
        feature_df["ball_dist_to_hoop"] = (
            (feature_df["ball_x"] - hoop_x) ** 2 + (feature_df["ball_y"] - _HOOP_Y) ** 2
        ) ** 0.5

    # --- NEW: ball in paint (binary) -------------------------------------
    if {"ball_x", "ball_y"}.issubset(feature_df.columns):
        in_left_paint = (
            (feature_df["ball_x"] >= 0.0)
            & (feature_df["ball_x"] <= _PAINT_DEPTH)
            & (feature_df["ball_y"] >= _PAINT_Y_LOW)
            & (feature_df["ball_y"] <= _PAINT_Y_HIGH)
        )
        in_right_paint = (
            (feature_df["ball_x"] >= 94.0 - _PAINT_DEPTH)
            & (feature_df["ball_x"] <= 94.0)
            & (feature_df["ball_y"] >= _PAINT_Y_LOW)
            & (feature_df["ball_y"] <= _PAINT_Y_HIGH)
        )
        feature_df["ball_in_paint"] = (in_left_paint | in_right_paint).astype(int)

    return feature_df


def _add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-frame motion features that require consecutive-frame context.

    Expects *df* to be sorted by (game_id, possession_id, possession_frame_idx)
    before calling. Frame diffs are computed within possession groups so
    event-boundary artifacts don't bleed across possessions.

    Adds:
      ball_step_dist  — Euclidean ball displacement (ft) from previous frame
      ball_speed      — ball_step_dist * FRAME_RATE (ft/s)
      ball_toward_basket — dot product of ball step direction with unit vector
                          toward the attacking basket (positive = toward hoop)
    """
    if not {"ball_x", "ball_y"}.issubset(df.columns):
        return df

    feature_df = df.copy()
    group_keys = ["game_id", "possession_id"]
    valid_groups = [k for k in group_keys if k in feature_df.columns]
    if not valid_groups:
        return feature_df

    dx = feature_df.groupby(valid_groups)["ball_x"].diff()
    dy = feature_df.groupby(valid_groups)["ball_y"].diff()
    feature_df["ball_step_dist"] = (dx**2 + dy**2) ** 0.5
    feature_df["ball_speed"] = feature_df["ball_step_dist"] * _FRAME_RATE

    # Direction toward basket: dot product of (dx, dy) with hoop unit vector
    if "offense_centroid_x" in feature_df.columns:
        attacking_right = (feature_df["offense_centroid_x"] < _HALF_COURT_X).astype(float)
        hoop_x = attacking_right * _HOOP_RIGHT_X + (1.0 - attacking_right) * _HOOP_LEFT_X
        to_hoop_x = hoop_x - feature_df["ball_x"]
        to_hoop_y = _HOOP_Y - feature_df["ball_y"]
        hoop_dist = (to_hoop_x**2 + to_hoop_y**2) ** 0.5
        # Avoid division by zero when ball is exactly on the hoop
        hoop_dist = hoop_dist.replace(0.0, float("nan"))
        unit_x = to_hoop_x / hoop_dist
        unit_y = to_hoop_y / hoop_dist
        step_dist = feature_df["ball_step_dist"].replace(0.0, float("nan"))
        feature_df["ball_toward_basket"] = (dx * unit_x + dy * unit_y) / step_dist

    return feature_df


def add_rich_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive team-aggregate and Phase B features from rich_frames player slots.

    Takes a rich_frames DataFrame (player_0 through player_9 slot columns) and
    returns a copy with columns added so that add_basic_tracking_features() and
    _add_motion_features() work unchanged.

    Derived columns matching the standard frames schema:
      possession_frame_idx, player_count, offense_player_count,
      defense_player_count, offense_centroid_x/y, defense_centroid_x/y,
      offense_mean_radius, defense_mean_radius,
      offense_mean_distance_to_ball, defense_mean_distance_to_ball,
      missing_shot_clock

    Phase B columns (not in standard frames):
      nearest_defender_to_ball, offense_convex_hull_area,
      defense_convex_hull_area, paint_occupancy_offense,
      paint_occupancy_defense
    """
    if "player_0_x" not in df.columns:
        raise ValueError(
            "add_rich_player_features requires rich_frames columns (player_0_x … player_9_x). "
            "Pass rich_frames.csv, not standard frames.csv."
        )

    feature_df = df.copy()
    feature_df = feature_df.sort_values(["game_id", "possession_id", "event_id", "frame_idx"]).reset_index(drop=True)

    # possession_frame_idx — needed by _add_motion_features sort
    if "possession_frame_idx" not in feature_df.columns:
        feature_df["possession_frame_idx"] = feature_df.groupby(
            ["game_id", "possession_id"]
        ).cumcount()

    # missing_shot_clock flag
    feature_df["missing_shot_clock"] = feature_df["shot_clock"].isna().astype(int)

    # --- build (N, 10) numpy arrays from player slot columns ---
    N = len(feature_df)
    team_ids = np.column_stack(
        [feature_df[f"player_{i}_team_id"].to_numpy(dtype=float) for i in range(10)]
    )
    xs = np.column_stack(
        [feature_df[f"player_{i}_x"].to_numpy(dtype=float) for i in range(10)]
    )
    ys = np.column_stack(
        [feature_df[f"player_{i}_y"].to_numpy(dtype=float) for i in range(10)]
    )

    # offense/defense masks — rows where offense_team_id is NaN produce all-False masks
    off_team = feature_df["offense_team_id"].to_numpy(dtype=float).reshape(N, 1)
    is_offense = (team_ids == off_team) & ~np.isnan(team_ids)
    is_defense = (~is_offense) & ~np.isnan(team_ids)

    # masked position arrays (NaN for slots that don't belong to each side)
    off_x = np.where(is_offense, xs, np.nan)
    off_y = np.where(is_offense, ys, np.nan)
    def_x = np.where(is_defense, xs, np.nan)
    def_y = np.where(is_defense, ys, np.nan)

    # --- player counts ---
    feature_df["offense_player_count"] = is_offense.sum(axis=1).astype(float)
    feature_df["defense_player_count"] = is_defense.sum(axis=1).astype(float)
    feature_df["player_count"] = feature_df.get("player_slot_count", feature_df["offense_player_count"] + feature_df["defense_player_count"])

    # --- centroids, radii, distances — suppress expected all-NaN warnings ---
    # Rows where offense_team_id is NaN produce all-NaN slices; result is NaN,
    # which is correct. numpy raises RuntimeWarning via warnings.warn (not via
    # the float error machinery), so warnings.catch_warnings is needed here,
    # not np.errstate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        off_cx = np.nanmean(off_x, axis=1)
        off_cy = np.nanmean(off_y, axis=1)
        def_cx = np.nanmean(def_x, axis=1)
        def_cy = np.nanmean(def_y, axis=1)

        feature_df["offense_centroid_x"] = off_cx
        feature_df["offense_centroid_y"] = off_cy
        feature_df["defense_centroid_x"] = def_cx
        feature_df["defense_centroid_y"] = def_cy

        off_dist_centroid = np.sqrt(
            (off_x - off_cx[:, None]) ** 2 + (off_y - off_cy[:, None]) ** 2
        )
        def_dist_centroid = np.sqrt(
            (def_x - def_cx[:, None]) ** 2 + (def_y - def_cy[:, None]) ** 2
        )
        feature_df["offense_mean_radius"] = np.nanmean(off_dist_centroid, axis=1)
        feature_df["defense_mean_radius"] = np.nanmean(def_dist_centroid, axis=1)

        bx = feature_df["ball_x"].to_numpy(dtype=float).reshape(N, 1)
        by = feature_df["ball_y"].to_numpy(dtype=float).reshape(N, 1)
        off_dist_ball = np.sqrt((off_x - bx) ** 2 + (off_y - by) ** 2)
        def_dist_ball = np.sqrt((def_x - bx) ** 2 + (def_y - by) ** 2)
        feature_df["offense_mean_distance_to_ball"] = np.nanmean(off_dist_ball, axis=1)
        feature_df["defense_mean_distance_to_ball"] = np.nanmean(def_dist_ball, axis=1)

        feature_df["nearest_defender_to_ball"] = np.nanmin(def_dist_ball, axis=1)

    # --- Phase B: paint occupancy ---
    in_left_paint = (
        (xs >= 0.0) & (xs <= _PAINT_DEPTH)
        & (ys >= _PAINT_Y_LOW) & (ys <= _PAINT_Y_HIGH)
    )
    in_right_paint = (
        (xs >= 94.0 - _PAINT_DEPTH) & (xs <= 94.0)
        & (ys >= _PAINT_Y_LOW) & (ys <= _PAINT_Y_HIGH)
    )
    in_paint = in_left_paint | in_right_paint
    feature_df["paint_occupancy_offense"] = (in_paint & is_offense).sum(axis=1).astype(float)
    feature_df["paint_occupancy_defense"] = (in_paint & is_defense).sum(axis=1).astype(float)

    # --- Phase B: convex hull areas (Python loop — variable point count per row) ---
    off_pts = np.stack([off_x, off_y], axis=2)  # (N, 10, 2)
    def_pts = np.stack([def_x, def_y], axis=2)
    feature_df["offense_convex_hull_area"] = np.array(
        [_convex_hull_area(off_pts[i]) for i in range(N)]
    )
    feature_df["defense_convex_hull_area"] = np.array(
        [_convex_hull_area(def_pts[i]) for i in range(N)]
    )

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


def build_rich_frame_aggregate_table(rich_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a rich_frames DataFrame to possession-level features.

    Calls add_rich_player_features() to derive the standard column contract,
    then delegates to build_frame_aggregate_table() for all Phase A logic.
    Phase B aggregations (nearest defender, convex hull, paint occupancy) are
    merged in afterward.
    """
    feature_df = add_rich_player_features(rich_df)
    grouped = build_frame_aggregate_table(feature_df)

    # --- Phase B possession-level aggregations ---
    phase_b: dict[str, list[str]] = {}
    for col in ["nearest_defender_to_ball"]:
        if col in feature_df.columns:
            phase_b[col] = ["mean", "min"]
    for col in ["offense_convex_hull_area", "defense_convex_hull_area"]:
        if col in feature_df.columns:
            phase_b[col] = ["mean", "std"]
    for col in ["paint_occupancy_offense", "paint_occupancy_defense"]:
        if col in feature_df.columns:
            phase_b[col] = ["mean", "max"]

    if phase_b:
        base_group = ["game_id", "possession_id"]
        b_grouped = feature_df.groupby(base_group).agg(phase_b)
        b_grouped.columns = [
            "_".join(c for c in col if c).rstrip("_")
            for col in b_grouped.columns.to_flat_index()
        ]
        grouped = grouped.merge(b_grouped.reset_index(), on=base_group, how="left")

    return grouped


def build_frame_aggregate_table(frame_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = add_basic_tracking_features(frame_df)
    feature_df = feature_df.sort_values(["game_id", "possession_id", "possession_frame_idx"])

    # Add sequence-based motion features (requires sorted order within possessions)
    feature_df = _add_motion_features(feature_df)

    base_group = ["game_id", "possession_id"]
    aggregations: dict[str, list[str]] = {}

    # --- existing aggregate columns --------------------------------------
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

    # --- NEW: per-frame feature aggregations -----------------------------
    for col in ["ball_dist_to_hoop", "ball_speed", "ball_toward_basket"]:
        if col in feature_df.columns:
            aggregations[col] = ["mean", "std", "min", "max"]

    if "shot_clock_bucket" in feature_df.columns:
        aggregations["shot_clock_bucket"] = ["mean"]

    if "ball_in_paint" in feature_df.columns:
        aggregations["ball_in_paint"] = ["mean", "max"]

    grouped = feature_df.groupby(base_group).agg(aggregations)
    grouped.columns = ["_".join(part for part in col if part).rstrip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    # Rename ball_in_paint_mean → ball_in_paint_fraction, ball_in_paint_max → ball_entered_paint
    rename_map: dict[str, str] = {}
    if "ball_in_paint_mean" in grouped.columns:
        rename_map["ball_in_paint_mean"] = "ball_in_paint_fraction"
    if "ball_in_paint_max" in grouped.columns:
        rename_map["ball_in_paint_max"] = "ball_entered_paint"
    if rename_map:
        grouped = grouped.rename(columns=rename_map)

    # --- start / end features (first and last frame of each possession) --
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
                "ball_dist_to_hoop",
            ]
            if col in frame_slice.columns
        ]
        slice_df = frame_slice[keep_cols].copy()
        rename = {col: f"{col}_{suffix}" for col in keep_cols if col not in base_group}
        first_last_features.append(slice_df.rename(columns=rename))

    for slice_df in first_last_features:
        grouped = grouped.merge(slice_df, on=base_group, how="left")

    # --- delta features (end - start) ------------------------------------
    delta_sources = [
        "ball_x",
        "ball_y",
        "shot_clock",
        "game_clock",
        "offense_mean_radius",
        "defense_mean_radius",
        "ball_dist_to_hoop",
    ]
    for source in delta_sources:
        start_col = f"{source}_start"
        end_col = f"{source}_end"
        if start_col in grouped.columns and end_col in grouped.columns:
            grouped[f"{source}_delta"] = grouped[end_col] - grouped[start_col]

    # --- NEW: possession-level aggregations requiring sum / custom logic --

    # ball_dist_traveled: total distance the ball moved over the possession
    if "ball_step_dist" in feature_df.columns:
        motion_agg = (
            feature_df.groupby(base_group)
            .agg(
                ball_dist_traveled=("ball_step_dist", "sum"),
                pass_count_proxy=("ball_step_dist", lambda x: int((x > _PASS_STEP_THRESHOLD).sum())),
            )
            .reset_index()
        )
        grouped = grouped.merge(motion_agg, on=base_group, how="left")

    # --- NEW: shot_clock_consumed (positive = clock was used) ------------
    if "shot_clock_start" in grouped.columns and "shot_clock_end" in grouped.columns:
        grouped["shot_clock_consumed"] = grouped["shot_clock_start"] - grouped["shot_clock_end"]

    # --- NEW: shot_clock_bucket at possession start ----------------------
    if "shot_clock_bucket" in feature_df.columns:
        bucket_start = (
            feature_df.groupby(base_group, as_index=False)
            .head(1)[base_group + ["shot_clock_bucket"]]
            .rename(columns={"shot_clock_bucket": "shot_clock_bucket_start"})
        )
        grouped = grouped.merge(bucket_start, on=base_group, how="left")

    # --- metadata columns ------------------------------------------------
    meta_columns = [
        col
        for col in ["possession_number", "terminal_label", "split", "possession_is_usable"]
        if col in feature_df.columns
    ]
    if meta_columns:
        meta = feature_df.groupby(base_group, as_index=False)[meta_columns].last()
        grouped = grouped.merge(meta, on=base_group, how="left")

    return grouped
