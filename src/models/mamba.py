"""A compact, Apple-Silicon-compatible (MPS/CPU) PyTorch implementation of the
Selective State Space Model (S6) mechanism from:

    Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces." arXiv:2312.00752.

IMPORTANT — SCOPE AND HONESTY NOTE (see paper/methodology_notes.md for the
full discussion):

The official reference implementation (`mamba-ssm` / `causal-conv1d`,
https://github.com/state-spaces/mamba) ships a fused, hardware-aware CUDA
kernel that performs the selective scan with a parallel work-efficient scan
algorithm and recomputation-based gradient checkpointing. That package
requires an NVIDIA GPU with CUDA/nvcc and CANNOT install or run on Apple
Silicon.

This module reimplements the SAME mathematical recurrence — zero-order-hold
discretization of a continuous linear state-space model, with the
transition/input/output parameters (Delta, B, C) made input-dependent
("selective") via learned linear projections — using a plain sequential
`for`-loop scan in PyTorch. It is:

  * Mathematically consistent with the Selective SSM formulation in the
    paper (same discretization, same selection mechanism, same recurrence).
  * NOT a copy of, and NOT claimed to be, the original CUDA kernel.
  * Slower than the optimized kernel (O(L) sequential Python-level steps
    per forward pass rather than a fused parallel scan), which is an
    expected, documented trade-off — not a hidden limitation.

Where practical, this implementation exposes the intermediate objects named
in the project brief: the input sequence, the discretized/selective
parameters, the hidden-state sequence, and the output sequence, so the
real-time demo can visualize actual state evolution rather than a fabricated
placeholder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """The S6 selective-scan mechanism (the core of a Mamba block).

    Input:  x of shape (batch, length, d_inner)  — already conv+activated
    Output: y of shape (batch, length, d_inner)

    Also returns (in forward, via `return_state_trace=True`) the full hidden
    state trajectory of shape (batch, length, d_inner, d_state) for
    visualization purposes. This is memory-heavier and only meant to be used
    for a single short sequence at a time (e.g. the real-time demo), not
    during batched training.
    """

    def __init__(self, d_inner: int, d_state: int, dt_rank: int | None = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)

        # Selection projections: input-dependent Delta, B, C (Section on the
        # "selection mechanism" in Gu & Dao 2023).
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A is parameterized in log-space and kept negative (via -exp(A_log))
        # so the continuous system is stable, following the reference design.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))       # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(d_inner))     # skip/passthrough term

    def forward(self, x: torch.Tensor, return_state_trace: bool = False):
        """x: (batch, length, d_inner)"""
        batch, length, d_inner = x.shape
        device = x.device

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), continuous A, kept negative-definite

        x_dbl = self.x_proj(x)  # (batch, length, dt_rank + 2*d_state)
        delta_raw, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))  # (batch, length, d_inner) — input-dependent step size

        # Zero-order-hold discretization:
        #   A_bar = exp(delta * A)
        #   B_bar = (delta * A)^-1 * (exp(delta*A) - I) * delta * B
        # We use the common first-order (Euler) approximation for B_bar,
        # B_bar ~= delta * B, which the reference implementation itself uses
        # by default (exact ZOH is available there as an option but the
        # simplified update is the standard, documented approximation) —
        # documented here explicitly rather than silently substituted.
        delta_A = torch.einsum("bld,dn->bldn", delta, A)     # (batch, length, d_inner, d_state)
        A_bar = torch.exp(delta_A)
        B_bar = torch.einsum("bld,bln->bldn", delta, B)      # (batch, length, d_inner, d_state)

        h = torch.zeros(batch, d_inner, self.d_state, device=device, dtype=x.dtype)
        ys = []
        state_trace = [] if return_state_trace else None

        for t in range(length):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)   # (batch, d_inner, d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])                  # (batch, d_inner)
            ys.append(y_t)
            if return_state_trace:
                state_trace.append(h.detach().clone())

        y = torch.stack(ys, dim=1)          # (batch, length, d_inner)
        y = y + x * self.D                  # skip/passthrough term (D)

        if return_state_trace:
            state_trace = torch.stack(state_trace, dim=1)  # (batch, length, d_inner, d_state)
            return y, {
                "delta": delta.detach(),
                "A": A.detach(),
                "B": B.detach(),
                "C": C.detach(),
                "hidden_state_trajectory": state_trace,
            }
        return y


class MambaBlock(nn.Module):
    """One Mamba block: expand -> (conv1d -> SiLU -> SSM) gated by a parallel
    SiLU branch -> project back down, wrapped in a residual connection.

    This ordering (dual-branch expansion, causal depthwise conv + SiLU
    feeding the SSM, multiplicative gating by a second SiLU branch, then a
    down-projection, all inside a residual) matches the block design
    described in the Mamba paper and its reference implementation.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,          # depthwise
            padding=d_conv - 1,             # causal: trim the right side after conv
            bias=True,
        )

        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_state_trace: bool = False):
        """x: (batch, length, d_model) -> (batch, length, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                                   # (batch, length, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)                # each (batch, length, d_inner)

        # Causal depthwise conv over the time axis.
        x_conv = x_branch.transpose(1, 2)                       # (batch, d_inner, length)
        x_conv = self.conv1d(x_conv)[:, :, : x_branch.shape[1]]  # trim to causal length
        x_conv = x_conv.transpose(1, 2)                          # (batch, length, d_inner)
        x_conv = F.silu(x_conv)

        if return_state_trace:
            y, trace = self.ssm(x_conv, return_state_trace=True)
        else:
            y = self.ssm(x_conv)
            trace = None

        y = y * F.silu(z_branch)   # multiplicative gate
        out = self.out_proj(y)
        out = self.dropout(out)
        out = out + residual

        if return_state_trace:
            return out, trace
        return out


class MambaEncoder(nn.Module):
    """Stack of MambaBlocks used as the sequence-modeling backbone."""

    supports_state_trace = True

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_state_trace: bool = False):
        """x: (batch, length, d_model)"""
        traces = [] if return_state_trace else None
        for layer in self.layers:
            if return_state_trace:
                x, trace = layer(x, return_state_trace=True)
                traces.append(trace)
            else:
                x = layer(x)
        x = self.final_norm(x)
        if return_state_trace:
            return x, traces
        return x
