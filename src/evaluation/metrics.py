"""Evaluation metrics computed from real model predictions — never fabricated."""
from __future__ import annotations

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[list[int], list[int], list[list[float]]]:
    """Run the model over a DataLoader and collect (true labels, predicted labels, probabilities)."""
    model.eval()
    y_true, y_pred, probs_all = [], [], []

    for log_mel, labels in loader:
        log_mel = log_mel.to(device)
        logits = model(log_mel)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    return y_true, y_pred, probs_all


def compute_classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    """Real precision/recall/F1/accuracy/confusion-matrix computed from actual predictions."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0,
    )
    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "n_samples": len(y_true),
    }
