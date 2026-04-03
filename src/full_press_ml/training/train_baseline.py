"""Train baseline possession outcome classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from full_press_ml.models.baselines import build_logistic_regression, build_xgboost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--label-column", default="possession_outcome")
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="xgboost")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    feature_columns = [col for col in df.columns if col not in {"game_id", "possession_id", args.label_column}]
    x = df[feature_columns]
    y = df[args.label_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if args.model == "logreg":
        model = build_logistic_regression()
    else:
        model = build_xgboost(num_classes=y.nunique())

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()

