"""Tests for src/models/classifier.py — full pipeline (both backbones)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.classifier import build_model, count_parameters  # noqa: E402


@pytest.mark.parametrize("backbone", ["mamba", "gru"])
def test_output_shape(cfg, backbone):
    model = build_model(cfg, backbone=backbone)
    n_mels = cfg["features"]["n_mels"]
    n_classes = cfg["classifier"]["n_classes"]
    batch, n_frames = 3, 50

    log_mel = torch.randn(batch, n_mels, n_frames)
    logits = model(log_mel)
    assert logits.shape == (batch, n_classes)


@pytest.mark.parametrize("backbone", ["mamba", "gru"])
def test_predict_proba_normalized(cfg, backbone):
    model = build_model(cfg, backbone=backbone)
    n_mels = cfg["features"]["n_mels"]
    log_mel = torch.randn(2, n_mels, 40)
    probs = model.predict_proba(log_mel)

    assert torch.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1).all()
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.parametrize("backbone", ["mamba", "gru"])
def test_parameter_count_is_reasonable(cfg, backbone):
    model = build_model(cfg, backbone=backbone)
    n_params = count_parameters(model)
    # Sanity bound only — not a claimed/reported research figure, just a
    # guard against a config regression producing a wildly oversized model.
    assert 0 < n_params < 5_000_000


def test_mamba_state_trace_available(cfg):
    model = build_model(cfg, backbone="mamba")
    n_mels = cfg["features"]["n_mels"]
    log_mel = torch.randn(1, n_mels, 40)
    logits, traces = model(log_mel, return_state_trace=True)
    assert logits.shape == (1, cfg["classifier"]["n_classes"])
    assert traces is not None
    assert len(traces) == cfg["model_mamba"]["n_layers"]
