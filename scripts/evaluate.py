#!/usr/bin/env python3
"""STEP 7 — Evaluate a trained checkpoint on the held-out test set.

Loads a checkpoint saved by train_mamba.py / train_gru.py, runs it on
data/processed/test.csv, and writes real computed metrics + a confusion
matrix figure + training-curve figures to experiments/results/<run_name>/.

Usage:
    python scripts/evaluate.py --run-name mamba_1234567890
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from evaluation.metrics import collect_predictions, compute_classification_metrics  # noqa: E402
from evaluation.plots import plot_confusion_matrix, plot_training_curves  # noqa: E402
from models.classifier import build_model  # noqa: E402
from training.dataset import SpeechCommandDataset  # noqa: E402
from training.device import select_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Run directory name under experiments/results/")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "experiments" / "results" / args.run_name
    if not run_dir.exists():
        print(f"ERROR: run directory {run_dir} not found.")
        return 1

    with open(run_dir / "summary.json") as f:
        run_summary = json.load(f)

    backbone = run_summary["backbone"]
    checkpoint_path = Path(run_summary["checkpoint_path"])
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint {checkpoint_path} not found.")
        return 1

    cfg = load_config(args.config)
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    test_manifest = processed_dir / "test.csv"
    if not test_manifest.exists():
        print(f"ERROR: {test_manifest} not found. Run scripts/prepare_dataset.py first.")
        return 1

    test_ds = SpeechCommandDataset(test_manifest, cfg, PROJECT_ROOT)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = select_device(cfg["device"]["prefer"], cfg["device"]["fallback"])
    model = build_model(cfg, backbone=backbone)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    print(f"Loaded {backbone} checkpoint from {checkpoint_path} onto {device}")

    t0 = time.time()
    y_true, y_pred, _ = collect_predictions(model, test_loader, device)
    eval_time = time.time() - t0

    metrics = compute_classification_metrics(y_true, y_pred, test_ds.class_names)
    metrics["eval_time_sec"] = eval_time
    metrics["n_test_samples"] = len(y_true)

    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix figure
    fig_dir = PROJECT_ROOT / "experiments" / "figures"
    plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["class_names"],
        title=f"{backbone.upper()} — Confusion Matrix (test set)",
        out_path=fig_dir / f"{args.run_name}_confusion_matrix.png",
    )

    # Training curves figure (from the CSV saved during training)
    curve_rows = []
    with open(run_dir / "training_curves.csv") as f:
        curve_rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in curve_rows]
    train_loss = [float(r["train_loss"]) for r in curve_rows]
    val_loss = [float(r["val_loss"]) if r["val_loss"] != "" else None for r in curve_rows]
    train_acc = [float(r["train_acc"]) for r in curve_rows]
    val_acc = [float(r["val_acc"]) if r["val_acc"] != "" else None for r in curve_rows]

    plot_training_curves(
        epochs, train_loss, val_loss, train_acc, val_acc,
        title=f"{backbone.upper()} training", out_path=fig_dir / f"{args.run_name}_training_curves.png",
    )

    # Also append a row to the shared cross-run metrics CSV (paper artifact).
    metrics_csv_path = PROJECT_ROOT / "experiments" / "results" / "all_runs_metrics.csv"
    write_header = not metrics_csv_path.exists()
    with open(metrics_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_name", "backbone", "n_params", "device", "test_accuracy",
                              "macro_precision", "macro_recall", "macro_f1", "n_test_samples", "eval_time_sec"])
        writer.writerow([args.run_name, backbone, run_summary["n_params"], run_summary["device"],
                          metrics["accuracy"], metrics["macro_precision"], metrics["macro_recall"],
                          metrics["macro_f1"], metrics["n_test_samples"], metrics["eval_time_sec"]])

    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision/recall/F1: {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"Metrics written to {metrics_path}")
    print(f"Figures written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
