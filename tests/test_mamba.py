"""Tests for src/models/mamba.py — the Selective SSM / Mamba implementation.

Covers: input/output shape, forward pass correctness (finite outputs),
gradient propagation, CPU execution, and MPS execution when available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.mamba import MambaBlock, MambaEncoder, SelectiveSSM  # noqa: E402


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


@pytest.fixture()
def small_input():
    batch, length, d_model = 2, 20, 16
    return torch.randn(batch, length, d_model)


def test_selective_ssm_shape():
    batch, length, d_inner, d_state = 2, 15, 32, 8
    ssm = SelectiveSSM(d_inner=d_inner, d_state=d_state)
    x = torch.randn(batch, length, d_inner)
    y = ssm(x)
    assert y.shape == (batch, length, d_inner)


def test_selective_ssm_finite_output():
    ssm = SelectiveSSM(d_inner=16, d_state=8)
    x = torch.randn(3, 10, 16)
    y = ssm(x)
    assert torch.isfinite(y).all()


def test_selective_ssm_state_trace_shape():
    batch, length, d_inner, d_state = 1, 12, 16, 8
    ssm = SelectiveSSM(d_inner=d_inner, d_state=d_state)
    x = torch.randn(batch, length, d_inner)
    y, trace = ssm(x, return_state_trace=True)
    assert y.shape == (batch, length, d_inner)
    assert trace["hidden_state_trajectory"].shape == (batch, length, d_inner, d_state)
    assert torch.isfinite(trace["hidden_state_trajectory"]).all()


def test_mamba_block_preserves_shape(small_input):
    block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
    out = block(small_input)
    assert out.shape == small_input.shape


def test_mamba_block_is_causal():
    """Changing a later timestep must not change an earlier timestep's output."""
    torch.manual_seed(0)
    block = MambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    block.eval()

    x = torch.randn(1, 10, 8)
    x_modified = x.clone()
    x_modified[:, 7:, :] = torch.randn(1, 3, 8)  # perturb only the tail

    with torch.no_grad():
        y1 = block(x)
        y2 = block(x_modified)

    # Outputs at timesteps before the perturbation must be unaffected.
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-5)


def test_mamba_encoder_shape(small_input):
    encoder = MambaEncoder(d_model=16, d_state=8, d_conv=4, expand=2, n_layers=3)
    out = encoder(small_input)
    assert out.shape == small_input.shape


def test_mamba_encoder_state_trace(small_input):
    encoder = MambaEncoder(d_model=16, d_state=8, d_conv=4, expand=2, n_layers=2)
    out, traces = encoder(small_input, return_state_trace=True)
    assert out.shape == small_input.shape
    assert len(traces) == 2  # one trace dict per layer


def test_gradient_propagation(small_input):
    block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
    x = small_input.clone().requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0  # gradient actually flowed, not all zeros

    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient reached parameter: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in: {name}"


def test_cpu_execution(small_input):
    block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2)
    block = block.to("cpu")
    out = block(small_input.to("cpu"))
    assert out.device.type == "cpu"
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _mps_available(), reason="MPS not available on this machine")
def test_mps_execution(small_input):
    block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2).to("mps")
    x = small_input.to("mps")
    out = block(x)
    assert out.device.type == "mps"
    assert torch.isfinite(out.cpu()).all()


@pytest.mark.skipif(not _mps_available(), reason="MPS not available on this machine")
def test_mps_gradient_propagation(small_input):
    block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2).to("mps")
    x = small_input.to("mps").requires_grad_(True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad.cpu()).all()
