"""Tests for src/features/logmel.py — log-Mel spectrogram extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features.logmel import LogMelExtractor  # noqa: E402


def test_output_shape_matches_config(cfg, synthetic_waveform):
    extractor = LogMelExtractor(cfg)
    log_mel = extractor(synthetic_waveform)

    n_mels_expected = cfg["features"]["n_mels"]
    assert log_mel.shape[0] == n_mels_expected
    assert log_mel.dim() == 2


def test_output_is_finite(cfg, synthetic_waveform):
    extractor = LogMelExtractor(cfg)
    log_mel = extractor(synthetic_waveform)
    assert torch.isfinite(log_mel).all()


def test_output_no_nan_on_silence(cfg):
    extractor = LogMelExtractor(cfg)
    silence = torch.zeros(16000)
    log_mel = extractor(silence)
    assert torch.isfinite(log_mel).all()
    assert not torch.isnan(log_mel).any()


def test_rejects_non_1d_input(cfg):
    extractor = LogMelExtractor(cfg)
    bad_input = torch.zeros(2, 16000)
    try:
        extractor(bad_input)
        assert False, "expected ValueError for non-1D input"
    except ValueError:
        pass


def test_output_shape_helper_matches_actual(cfg, synthetic_waveform):
    extractor = LogMelExtractor(cfg)
    actual = extractor(synthetic_waveform).shape
    predicted = extractor.output_shape(synthetic_waveform.shape[0])
    assert predicted[0] == actual[0]
    # num_frames can differ by at most 1 depending on centering/padding conventions
    assert abs(predicted[1] - actual[1]) <= 1
