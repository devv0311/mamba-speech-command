"""Tests for src/models/gru.py — the GRU baseline encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.gru import GRUEncoder  # noqa: E402


def test_gru_encoder_shape():
    batch, length, d_model = 2, 20, 16
    encoder = GRUEncoder(d_model=d_model, hidden_size=24, n_layers=2)
    x = torch.randn(batch, length, d_model)
    out = encoder(x)
    assert out.shape == (batch, length, d_model)


def test_gru_encoder_finite():
    encoder = GRUEncoder(d_model=16, hidden_size=24, n_layers=2)
    x = torch.randn(2, 15, 16)
    out = encoder(x)
    assert torch.isfinite(out).all()


def test_gru_gradient_propagation():
    encoder = GRUEncoder(d_model=16, hidden_size=24, n_layers=2)
    x = torch.randn(2, 15, 16, requires_grad=True)
    out = encoder(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
