"""Tests for src/evaluation/metrics.py — verify metrics are computed correctly
against known-by-hand values (never trust metrics code without checking it
against a hand-computable case)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.metrics import compute_classification_metrics  # noqa: E402


def test_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_known_accuracy():
    # 4 correct out of 5 -> accuracy exactly 0.8
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 1]  # index 2 wrong
    m = compute_classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert abs(m["accuracy"] - 0.8) < 1e-9


def test_confusion_matrix_shape_and_diagonal():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    m = compute_classification_metrics(y_true, y_pred, ["a", "b", "c"])
    cm = m["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    assert cm[0][0] == 2 and cm[1][1] == 2 and cm[2][2] == 2


def test_per_class_keys_match_class_names():
    y_true = [0, 1]
    y_pred = [0, 0]
    class_names = ["yes", "no"]
    m = compute_classification_metrics(y_true, y_pred, class_names)
    assert set(m["per_class"].keys()) == set(class_names)
    assert m["per_class"]["yes"]["recall"] == 1.0
    assert m["per_class"]["no"]["recall"] == 0.0
