"""Shared pytest fixtures: synthetic audio generation (no dataset dependency)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config  # noqa: E402


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def synthetic_waveform():
    """A 1-second 16kHz sine-wave tone standing in for real speech audio.

    Used for tests that only need *some* audio signal (shape/finiteness/
    pipeline plumbing), not the Speech Commands dataset itself.
    """
    sample_rate = 16000
    duration_sec = 1.0
    freq = 440.0
    t = torch.arange(int(sample_rate * duration_sec)) / sample_rate
    waveform = 0.5 * torch.sin(2 * torch.pi * freq * t)
    return waveform.to(torch.float32)


@pytest.fixture()
def tmp_wav_file(tmp_path, synthetic_waveform):
    from audio.io import save_wav  # noqa: PLC0415

    path = tmp_path / "synthetic.wav"
    save_wav(path, synthetic_waveform, sample_rate=16000)
    return path
