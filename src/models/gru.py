"""GRU baseline encoder — matched-capacity recurrent comparison for Mamba.

Uses the same input feature projection convention as the Mamba encoder so
the two backbones are as close to drop-in-comparable as a fundamentally
different recurrence allows (see configs/default.yaml: model_gru is tuned
to land in the same parameter-count order of magnitude as model_mamba).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, n_layers: int, bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.out_proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, d_model) -> (batch, length, d_model)"""
        y, _ = self.gru(x)
        y = self.out_proj(y)
        return self.norm(y)
