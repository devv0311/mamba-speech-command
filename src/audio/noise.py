"""Reproducible additive-noise generation for the noise-robustness experiment."""
from __future__ import annotations

import torch


def add_white_noise_at_snr(waveform: torch.Tensor, snr_db: float, generator: torch.Generator) -> torch.Tensor:
    """Add zero-mean Gaussian white noise to `waveform` at the requested SNR (dB).

    SNR is computed relative to the signal's own power, so the same snr_db
    always produces a comparably audible noise level regardless of input
    amplitude. `generator` must be a seeded torch.Generator for reproducibility.
    """
    signal_power = waveform.pow(2).mean()
    if signal_power <= 0:
        # Silent input: adding noise at a "relative" SNR is undefined; return unchanged.
        return waveform

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = noise_power.sqrt()

    noise = torch.randn(waveform.shape, generator=generator) * noise_std
    return waveform + noise
