"""Shared training loop for both the Mamba and GRU backbones."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float | None
    val_acc: float | None
    epoch_time_sec: float


@dataclass
class TrainingHistory:
    epochs: list[EpochResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"epochs": [vars(e) for e in self.epochs]}

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: str | Path) -> None:
        import csv  # noqa: PLC0415
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"])
            for e in self.epochs:
                writer.writerow([e.epoch, e.train_loss, e.train_acc, e.val_loss, e.val_acc, e.epoch_time_sec])


def run_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device,
                   grad_clip_norm: float | None, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss, total_correct, total_count = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for log_mel, labels in loader:
            log_mel = log_mel.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(log_mel)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None = 1.0,
    early_stopping_patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    log_every: int = 1,
) -> TrainingHistory:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    history = TrainingHistory()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip_norm, train=True)

        val_loss, val_acc = (None, None)
        if val_loader is not None:
            val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, grad_clip_norm, train=False)

        scheduler.step()
        epoch_time = time.time() - t0

        result = EpochResult(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                              val_loss=val_loss, val_acc=val_acc, epoch_time_sec=epoch_time)
        history.epochs.append(result)

        if epoch % log_every == 0 or epoch == epochs:
            val_str = f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}" if val_loss is not None else ""
            print(f"[epoch {epoch:3d}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}{val_str} ({epoch_time:.1f}s)")

        if val_loader is not None and early_stopping_patience is not None:
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {early_stopping_patience} epochs).")
                    break
        elif checkpoint_path is not None and epoch == epochs:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history
