"""Shared feature-projection + pooling + classification head, used by both
the Mamba and GRU speech-command models so the only difference between the
two experimental conditions is the sequence-modeling backbone itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPooling(nn.Module):
    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "last", "max"):
            raise ValueError(f"Unknown pooling mode: {mode}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, d_model) -> (batch, d_model)"""
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "last":
            return x[:, -1, :]
        if self.mode == "max":
            return x.max(dim=1).values
        raise AssertionError("unreachable")


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_model) -> (batch, n_classes) logits"""
        return self.net(x)


class SpeechCommandModel(nn.Module):
    """Full pipeline: (batch, n_mels, n_frames) log-Mel -> class logits.

    `encoder` is either a MambaEncoder or a GRUEncoder — anything that maps
    (batch, length, d_model) -> (batch, length, d_model).
    """

    def __init__(self, encoder: nn.Module, n_mels: int, d_model: int, pooling: str,
                 head_hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.encoder = encoder
        self.pooling = TemporalPooling(pooling)
        self.head = ClassificationHead(d_model, head_hidden, n_classes, dropout=dropout)

    def forward(self, log_mel: torch.Tensor, return_state_trace: bool = False):
        """log_mel: (batch, n_mels, n_frames) -> logits (batch, n_classes)

        If return_state_trace=True and the encoder supports it (MambaEncoder),
        also returns the per-layer SSM state trace for visualization.
        """
        x = log_mel.transpose(1, 2)          # (batch, n_frames, n_mels)
        x = self.input_proj(x)                # (batch, n_frames, d_model)

        supports_trace = getattr(self.encoder, "supports_state_trace", False)
        if return_state_trace and supports_trace:
            x, trace = self.encoder(x, return_state_trace=True)
        else:
            x = self.encoder(x)
            trace = None

        pooled = self.pooling(x)              # (batch, d_model)
        logits = self.head(pooled)            # (batch, n_classes)

        if return_state_trace:
            return logits, trace
        return logits

    def predict_proba(self, log_mel: torch.Tensor) -> torch.Tensor:
        logits = self.forward(log_mel)
        return torch.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, backbone: str) -> SpeechCommandModel:
    """backbone: 'mamba' or 'gru'"""
    n_mels = cfg["features"]["n_mels"]
    n_classes = cfg["classifier"]["n_classes"]
    head_hidden = cfg["classifier"]["head_hidden"]

    if backbone == "mamba":
        from models.mamba import MambaEncoder  # noqa: PLC0415
        m_cfg = cfg["model_mamba"]
        encoder = MambaEncoder(
            d_model=m_cfg["d_model"], d_state=m_cfg["d_state"], d_conv=m_cfg["d_conv"],
            expand=m_cfg["expand"], n_layers=m_cfg["n_layers"], dropout=m_cfg["dropout"],
        )
        return SpeechCommandModel(
            encoder=encoder, n_mels=n_mels, d_model=m_cfg["d_model"], pooling=m_cfg["pooling"],
            head_hidden=head_hidden, n_classes=n_classes, dropout=m_cfg["dropout"],
        )
    elif backbone == "gru":
        from models.gru import GRUEncoder  # noqa: PLC0415
        g_cfg = cfg["model_gru"]
        encoder = GRUEncoder(
            d_model=g_cfg["d_model"], hidden_size=g_cfg["hidden_size"], n_layers=g_cfg["n_layers"],
            bidirectional=g_cfg["bidirectional"], dropout=g_cfg["dropout"],
        )
        return SpeechCommandModel(
            encoder=encoder, n_mels=n_mels, d_model=g_cfg["d_model"], pooling=g_cfg["pooling"],
            head_hidden=head_hidden, n_classes=n_classes, dropout=g_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}, expected 'mamba' or 'gru'")
