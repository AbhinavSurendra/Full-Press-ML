from full_press_ml.evaluation.metrics import summarize_predictions


def test_summarize_predictions_contains_expected_keys() -> None:
    summary = summarize_predictions([0, 1, 1], [0, 1, 0])
    assert "macro_f1" in summary
    assert "weighted_f1" in summary
    assert "confusion_matrix" in summary
    assert "classification_report" in summary
