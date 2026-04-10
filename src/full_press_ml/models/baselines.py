"""Baseline model factories."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )


def build_xgboost(num_classes: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
    )
