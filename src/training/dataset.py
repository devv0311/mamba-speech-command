"""PyTorch Dataset wrapping the prepared manifest CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio.io import fix_length, load_wav, peak_normalize
from audio.noise import add_white_noise_at_snr
from features.logmel import LogMelExtractor


class SpeechCommandDataset(Dataset):
    def __init__(self, manifest_path: str | Path, cfg: dict, project_root: Path,
                 snr_db: float | None = None, noise_seed: int | None = None):
        """snr_db: if set, apply reproducible additive white noise at this SNR
        (used by the noise-robustness experiment). None = clean audio."""
        self.project_root = Path(project_root)
        with open(manifest_path) as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError(f"Manifest at {manifest_path} is empty")

        self.commands = sorted(cfg["dataset"]["commands"])
        self.label_to_idx = {label: i for i, label in enumerate(self.commands)}

        self.sample_rate = cfg["audio"]["sample_rate"]
        self.target_len = int(cfg["dataset"]["target_duration_sec"] * self.sample_rate)
        self.extractor = LogMelExtractor(cfg)

        self.snr_db = snr_db
        self.noise_seed = noise_seed if noise_seed is not None else cfg["noise"]["seed"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wav_path = self.project_root / row["filepath"]

        waveform = load_wav(wav_path, target_sample_rate=self.sample_rate)
        waveform = peak_normalize(waveform)
        waveform = fix_length(waveform, self.target_len)

        if self.snr_db is not None:
            # Deterministic per-sample noise: seed derived from the global
            # noise seed and the sample index, so re-running the same
            # experiment reproduces identical noisy audio.
            gen = torch.Generator().manual_seed(self.noise_seed * 100003 + idx)
            waveform = add_white_noise_at_snr(waveform, self.snr_db, generator=gen)

        log_mel = self.extractor(waveform)
        label_idx = self.label_to_idx[row["label"]]

        return log_mel, label_idx

    @property
    def class_names(self) -> list[str]:
        return self.commands
