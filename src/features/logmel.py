"""Log-Mel spectrogram feature extraction.

All parameters are read from configs/default.yaml (features: section) — none
are hard-coded here beyond torchaudio API defaults that the config overrides.
"""
from __future__ import annotations

import torch
import torchaudio


class LogMelExtractor:
    """Wraps torchaudio's MelSpectrogram + log compression into one callable.

    Input:  1-D waveform tensor, shape (num_samples,), sample rate = cfg sample_rate.
    Output: 2-D log-Mel spectrogram, shape (n_mels, num_frames).
    """

    def __init__(self, cfg: dict):
        audio_cfg = cfg["audio"]
        feat_cfg = cfg["features"]

        self.sample_rate = audio_cfg["sample_rate"]
        self.log_offset = feat_cfg["log_offset"]

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=feat_cfg["n_fft"],
            win_length=feat_cfg["win_length"],
            hop_length=feat_cfg["hop_length"],
            n_mels=feat_cfg["n_mels"],
            f_min=feat_cfg["f_min"],
            f_max=feat_cfg["f_max"],
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 1:
            raise ValueError(f"Expected 1-D waveform, got shape {tuple(waveform.shape)}")

        mel = self.mel_spec(waveform)                       # (n_mels, num_frames), power spectrogram
        log_mel = torch.log(mel + self.log_offset)           # numerically stable log
        return log_mel

    def output_shape(self, num_samples: int) -> tuple[int, int]:
        """Compute (n_mels, num_frames) for a given input length without running a forward pass."""
        n_mels = self.mel_spec.n_mels
        hop = self.mel_spec.hop_length
        num_frames = num_samples // hop + 1
        return (n_mels, num_frames)
