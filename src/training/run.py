"""Shared "run a full training job and save artifacts" logic, used by both
scripts/train_mamba.py and scripts/train_gru.py so the two experiments follow
an identical, comparable protocol (same manifests, same loader settings, same
checkpointing/logging behavior) — only the backbone differs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import PROJECT_ROOT
from models.classifier import build_model, count_parameters
from training.dataset import SpeechCommandDataset
from training.device import select_device
from training.trainer import train_model


def run_training(cfg: dict, backbone: str, run_name: str | None = None) -> dict:
    torch.manual_seed(cfg["project"]["seed"])

    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    train_manifest = processed_dir / "train.csv"
    val_manifest = processed_dir / "val.csv"

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(f"{train_manifest} / {val_manifest} not found. Run scripts/prepare_dataset.py first.")

    train_ds = SpeechCommandDataset(train_manifest, cfg, PROJECT_ROOT)
    val_ds = SpeechCommandDataset(val_manifest, cfg, PROJECT_ROOT)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                               num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                             num_workers=cfg["training"]["num_workers"])

    device = select_device(cfg["device"]["prefer"], cfg["device"]["fallback"])
    print(f"Device: {device}")

    model = build_model(cfg, backbone=backbone)
    n_params = count_parameters(model)
    print(f"{backbone} model parameters: {n_params}")

    run_name = run_name or f"{backbone}_{int(time.time())}"
    run_dir = PROJECT_ROOT / "experiments" / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    history = train_model(
        model, train_loader, val_loader, device=device,
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip_norm=cfg["training"]["grad_clip_norm"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        checkpoint_path=PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / f"{run_name}.pt",
        log_every=1,
    )
    total_train_time = time.time() - t0

    history.save_csv(run_dir / "training_curves.csv")
    history.save_json(run_dir / "training_curves.json")

    val_accs = [e.val_acc for e in history.epochs if e.val_acc is not None]
    summary = {
        "run_name": run_name,
        "backbone": backbone,
        "n_params": n_params,
        "device": str(device),
        "total_training_time_sec": total_train_time,
        "epochs_run": len(history.epochs),
        "final_train_acc": history.epochs[-1].train_acc,
        "final_val_acc": history.epochs[-1].val_acc,
        "best_val_acc": max(val_accs) if val_accs else None,
        "checkpoint_path": str(PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / f"{run_name}.pt"),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete in {total_train_time:.1f}s. Run saved to {run_dir}")
    print(json.dumps(summary, indent=2))
    return summary
