"""Audio loading and normalization utilities.

NOTE on backend choice: torchaudio's own load()/save() functions changed
their I/O backend across recent releases to require the optional
`torchcodec` package (which wraps FFmpeg). To keep this project's
dependency footprint small and avoid an extra FFmpeg-coupled dependency
whose behavior can vary by platform, WAV file I/O here goes through
`soundfile` (libsndfile) directly, which is lightweight and has no such
requirement. torchaudio is still used for its tensor-level transforms
(resampling, MelSpectrogram in src/features/logmel.py), which don't depend
on its file I/O backend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_wav(path: str | Path, target_sample_rate: int = 16000) -> torch.Tensor:
    """Load a WAV file as mono, resampled to target_sample_rate.

    Returns a 1-D float32 tensor of shape (num_samples,), values in [-1, 1].
    Raises FileNotFoundError if the path doesn't exist, and ValueError if the
    loaded audio is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    waveform = torch.from_numpy(data).T  # -> (channels, samples)

    if waveform.numel() == 0:
        raise ValueError(f"Loaded empty audio tensor from {path}")

    # Convert to mono by averaging channels if needed.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)  # -> (samples,)
    return waveform.to(torch.float32)


def save_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a 1-D (or (1, N)) float32 waveform tensor as a WAV file via soundfile."""
    wf = waveform.detach().cpu()
    if wf.dim() == 2:
        wf = wf.squeeze(0)
    sf.write(str(path), wf.numpy().astype(np.float32), sample_rate)


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Peak-normalize a waveform to [-1, 1]. No-op on (near-)silent audio."""
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform / peak


def fix_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    """Pad with zeros or center-trim a 1-D waveform to exactly target_num_samples."""
    n = waveform.shape[-1]
    if n == target_num_samples:
        return waveform
    if n < target_num_samples:
        pad_total = target_num_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.nn.functional.pad(waveform, (pad_left, pad_right))
    # trim (center crop)
    start = (n - target_num_samples) // 2
    return waveform[start:start + target_num_samples]
