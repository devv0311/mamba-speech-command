#!/usr/bin/env python3
"""STEP 9 — Noise robustness experiment (Experiment 3).

Evaluates a trained checkpoint on the test set under each noise condition
defined in configs/default.yaml: noise.conditions (clean / mild / moderate /
strong, additive white Gaussian noise at fixed SNRs), using a reproducible,
index-derived seed so the exact same noisy audio is generated on every run.

Usage:
    python scripts/noise_experiment.py --run-name mamba_run1
    python scripts/noise_experiment.py --run-name gru_run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from evaluation.metrics import collect_predictions, compute_classification_metrics  # noqa: E402
from models.classifier import build_model  # noqa: E402
from training.dataset import SpeechCommandDataset  # noqa: E402
from training.device import select_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "experiments" / "results" / args.run_name
    if not run_dir.exists():
        print(f"ERROR: run directory {run_dir} not found. Train a model first.")
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

    device = select_device(cfg["device"]["prefer"], cfg["device"]["fallback"])
    model = build_model(cfg, backbone=backbone)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    print(f"Loaded {backbone} checkpoint from {checkpoint_path} onto {device}")

    conditions = cfg["noise"]["conditions"]
    noise_seed = cfg["noise"]["seed"]

    results = {}
    for condition_name, condition_cfg in conditions.items():
        snr_db = condition_cfg["snr_db"]
        test_ds = SpeechCommandDataset(test_manifest, cfg, PROJECT_ROOT, snr_db=snr_db, noise_seed=noise_seed)
        test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

        y_true, y_pred, _ = collect_predictions(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, test_ds.class_names)

        results[condition_name] = {
            "snr_db": snr_db,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "n_samples": metrics["n_samples"],
        }
        snr_str = f"{snr_db} dB" if snr_db is not None else "clean"
        print(f"{condition_name:10s} ({snr_str:8s}): accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    out_path = run_dir / "noise_robustness.json"
    with open(out_path, "w") as f:
        json.dump({"run_name": args.run_name, "backbone": backbone, "results": results}, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Also write/append a shared CSV across runs, useful for the comparison plot.
    import csv  # noqa: PLC0415
    csv_path = PROJECT_ROOT / "experiments" / "results" / "noise_robustness_all_runs.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_name", "backbone", "condition", "snr_db", "accuracy", "macro_f1"])
        for condition_name, r in results.items():
            writer.writerow([args.run_name, backbone, condition_name, r["snr_db"], r["accuracy"], r["macro_f1"]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
