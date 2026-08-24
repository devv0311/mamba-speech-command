#!/usr/bin/env python3
"""STEP 5 — Tiny overfit sanity check.

Trains on a deliberately tiny subset (a handful of samples per class) for
many epochs. If the model cannot drive training accuracy to ~100% on this
trivial task, something in the pipeline (data, features, model, loss,
optimizer) is broken — full training should NOT proceed until this passes.

Usage:
    python scripts/tiny_overfit_test.py --backbone mamba
    python scripts/tiny_overfit_test.py --backbone gru
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from models.classifier import build_model, count_parameters  # noqa: E402
from training.dataset import SpeechCommandDataset  # noqa: E402
from training.device import select_device  # noqa: E402
from training.trainer import train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--backbone", type=str, required=True, choices=["mamba", "gru"])
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument("--lr-multiplier", type=float, default=20.0,
                         help="Multiplier applied to configs/default.yaml training.learning_rate for this sanity check only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["project"]["seed"])

    train_manifest = PROJECT_ROOT / cfg["dataset"]["processed_dir"] / "train.csv"
    if not train_manifest.exists():
        print(f"ERROR: {train_manifest} not found. Run scripts/prepare_dataset.py first.")
        return 1

    full_dataset = SpeechCommandDataset(train_manifest, cfg, PROJECT_ROOT)

    # Build a tiny, class-balanced subset.
    by_class: dict[int, list[int]] = {}
    for i, row in enumerate(full_dataset.rows):
        label_idx = full_dataset.label_to_idx[row["label"]]
        by_class.setdefault(label_idx, []).append(i)

    tiny_indices = []
    for label_idx, indices in by_class.items():
        tiny_indices.extend(indices[: args.samples_per_class])

    tiny_dataset = Subset(full_dataset, tiny_indices)
    print(f"Tiny overfit set: {len(tiny_dataset)} samples ({args.samples_per_class} per class x {len(by_class)} classes)")

    loader = DataLoader(tiny_dataset, batch_size=len(tiny_dataset), shuffle=True)

    device = select_device(cfg["device"]["prefer"], cfg["device"]["fallback"])
    print(f"Device: {device}")

    model = build_model(cfg, backbone=args.backbone)
    print(f"Backbone: {args.backbone}, parameters: {count_parameters(model)}")

    history = train_model(
        model, loader, val_loader=None, device=device, epochs=args.epochs,
        learning_rate=cfg["training"]["learning_rate"] * args.lr_multiplier,  # faster overfit for the sanity check
        weight_decay=0.0,  # no regularization — we WANT it to overfit here
        grad_clip_norm=cfg["training"]["grad_clip_norm"],
        early_stopping_patience=None,
        checkpoint_path=None,
        log_every=10,
    )

    final_acc = history.epochs[-1].train_acc
    print(f"\nFinal train accuracy on tiny set: {final_acc:.4f}")

    if final_acc >= args.target_acc:
        print(f"PASS — model can overfit a tiny dataset (>= {args.target_acc:.0%}). Pipeline is sound.")
        return 0
    else:
        print(f"FAIL — could not reach {args.target_acc:.0%} train accuracy on a trivial tiny set.")
        print("Do NOT proceed to full training. Debug the pipeline (data/features/model/loss).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
