"""Figure generation for research artifacts — confusion matrix, training curves,
Mamba-vs-GRU comparisons, noise robustness. All plots are generated from real
computed data passed in by the caller; this module does not invent numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str, out_path: str | Path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(epochs: list[int], train_loss: list[float], val_loss: list[float],
                          train_acc: list[float], val_acc: list[float], title: str, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, train_loss, label="train")
    if any(v is not None for v in val_loss):
        axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    if any(v is not None for v in val_acc):
        axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_model_comparison_bar(labels: list[str], values_by_model: dict[str, list[float]],
                               ylabel: str, title: str, out_path: str | Path) -> None:
    """values_by_model: {"Mamba": [...], "GRU": [...]}, aligned with `labels`."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_models = len(values_by_model)
    width = 0.8 / n_models
    x = np.arange(len(labels))

    for i, (model_name, values) in enumerate(values_by_model.items()):
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_noise_robustness(conditions: list[str], accuracy_by_model: dict[str, list[float]],
                           out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_name, accs in accuracy_by_model.items():
        ax.plot(conditions, accs, marker="o", label=model_name)
    ax.set_xlabel("Noise condition")
    ax.set_ylabel("Accuracy")
    ax.set_title("Noise robustness")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
