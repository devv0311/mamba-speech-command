"""Visualization helpers for waveforms and spectrograms (used by scripts and the real-time demo)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for scripts; realtime demo uses its own live canvas
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_waveform_and_spectrogram(
    waveform: torch.Tensor,
    log_mel: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    title: str,
    out_path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    t = np.arange(waveform.shape[-1]) / sample_rate
    axes[0].plot(t, waveform.numpy(), linewidth=0.7)
    axes[0].set_title(f"{title} — waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    im = axes[1].imshow(
        log_mel.numpy(),
        aspect="auto",
        origin="lower",
        extent=[0, waveform.shape[-1] / sample_rate, 0, log_mel.shape[0]],
        cmap="magma",
    )
    axes[1].set_title(f"{title} — log-Mel spectrogram")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[1], format="%+2.0f dB-like")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
