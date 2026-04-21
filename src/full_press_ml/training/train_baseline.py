"""Train baseline possession outcome classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from full_press_ml.features.engineer import build_frame_aggregate_table, build_rich_frame_aggregate_table
from full_press_ml.models.baselines import build_logistic_regression, build_xgboost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", default="terminal_label")
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="xgboost")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--use-all-rows", action="store_true")
    parser.add_argument("--aggregate-frames", action="store_true")
    parser.add_argument("--rich", action="store_true", help="Input is rich_frames.csv; use rich feature pipeline.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.aggregate_frames:
        if args.rich:
            df = build_rich_frame_aggregate_table(df)
        else:
            df = build_frame_aggregate_table(df)
    if not args.use_all_rows and "is_usable" in df.columns:
        df = df[df["is_usable"] == 1].copy()
    if not args.use_all_rows and "possession_is_usable" in df.columns:
        df = df[df["possession_is_usable"] == 1].copy()
    df = df[df[args.label_column].notna()].copy()

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        eval_df = df[df["split"] == args.eval_split].copy()
    else:
        raise ValueError("Expected a split column in the input data.")

    if train_df.empty or eval_df.empty:
        raise ValueError("Training or evaluation split is empty after filtering.")

    drop_columns = {
        "game_id",
        "possession_id",
        "possession_number",
        "split",
        "event_ids",
        "end_reason",
        "is_usable",
        "possession_is_usable",
        args.label_column,
    }
    feature_columns = [col for col in train_df.columns if col not in drop_columns]
    x_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_eval = eval_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[args.label_column])
    y_eval = label_encoder.transform(eval_df[args.label_column])

    if args.model == "logreg":
        model = build_logistic_regression()
    else:
        model = build_xgboost(num_classes=len(label_encoder.classes_))

    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)

    print(f"train_rows={len(train_df)} eval_rows={len(eval_df)} eval_split={args.eval_split}")
    print(f"accuracy={accuracy_score(y_eval, predictions):.4f}")
    print(
        classification_report(
            y_eval,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
