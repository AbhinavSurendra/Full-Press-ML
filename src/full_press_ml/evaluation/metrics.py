"""Common metrics for multiclass evaluation."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report, confusion_matrix, f1_score


def summarize_predictions(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

