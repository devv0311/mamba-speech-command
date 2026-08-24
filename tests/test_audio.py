"""Tests for src/audio/io.py — audio loading."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from audio.io import fix_length, load_wav, peak_normalize, save_wav  # noqa: E402


def test_load_wav_valid_file(tmp_wav_file):
    waveform = load_wav(tmp_wav_file, target_sample_rate=16000)
    assert waveform.dim() == 1
    assert waveform.shape[0] == 16000
    assert waveform.dtype == torch.float32


def test_load_wav_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_wav("/nonexistent/path/does_not_exist.wav")


def test_load_wav_resamples(tmp_path, synthetic_waveform):
    import soundfile as sf  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    # Resample the synthetic tone to 48kHz first so the file is genuinely at
    # that rate, then verify load_wav brings it back to 16kHz.
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
    wave_48k = resampler(synthetic_waveform.unsqueeze(0)).squeeze(0)

    path = tmp_path / "highrate.wav"
    sf.write(str(path), wave_48k.numpy(), 48000)

    waveform = load_wav(path, target_sample_rate=16000)
    # 1 second at 48kHz resampled to 16kHz -> ~16000 samples (allow resampler edge slop)
    assert abs(waveform.shape[0] - 16000) < 10


def test_load_wav_stereo_to_mono(tmp_path, synthetic_waveform):
    import soundfile as sf  # noqa: PLC0415

    stereo = torch.stack([synthetic_waveform, synthetic_waveform * 0.5], dim=-1)  # (samples, channels)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo.numpy(), 16000)

    waveform = load_wav(path, target_sample_rate=16000)
    assert waveform.dim() == 1  # collapsed to mono


def test_save_wav_roundtrip(tmp_path, synthetic_waveform):
    path = tmp_path / "roundtrip.wav"
    save_wav(path, synthetic_waveform, sample_rate=16000)
    reloaded = load_wav(path, target_sample_rate=16000)
    assert reloaded.shape == synthetic_waveform.shape
    assert torch.allclose(reloaded, synthetic_waveform, atol=1e-3)


def test_peak_normalize():
    x = torch.tensor([0.1, -0.5, 0.25])
    normed = peak_normalize(x)
    assert torch.isclose(normed.abs().max(), torch.tensor(1.0))


def test_peak_normalize_silence_is_noop():
    x = torch.zeros(100)
    normed = peak_normalize(x)
    assert torch.equal(normed, x)


def test_fix_length_pads_short():
    x = torch.ones(100)
    fixed = fix_length(x, 200)
    assert fixed.shape[0] == 200


def test_fix_length_trims_long():
    x = torch.ones(300)
    fixed = fix_length(x, 200)
    assert fixed.shape[0] == 200


def test_fix_length_noop_when_exact():
    x = torch.ones(160)
    fixed = fix_length(x, 160)
    assert torch.equal(fixed, x)
