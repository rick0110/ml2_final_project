from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ResidualGatedConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.filter_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.gate_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.output_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        groups = max(1, channels // 8)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        gated = self.output_proj(self.dropout(gated))
        return self.norm(x + gated)


class PosteriorEncoder(nn.Module):
    """Posterior encoder that maps mel spectrograms into latent z."""

    def __init__(
        self,
        n_mels: int = 80,
        hidden_channels: int = 192,
        latent_channels: int = 192,
        n_layers: int = 16,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels

        self.input_proj = nn.Conv1d(n_mels, hidden_channels, kernel_size=1)
        dilations = [2 ** (idx % 4) for idx in range(n_layers)]
        self.blocks = nn.ModuleList(
            [
                ResidualGatedConv1d(hidden_channels, dilation=dilation, kernel_size=kernel_size, dropout=dropout)
                for dilation in dilations
            ]
        )
        self.output_proj = nn.Conv1d(hidden_channels, latent_channels * 2, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mel.dim() != 3:
            raise ValueError(f"mel must be 3D (batch, n_mels, time), got {tuple(mel.shape)}")
        if mel.size(1) != self.n_mels:
            raise ValueError(f"Expected mel with {self.n_mels} bins, got {mel.size(1)}")

        x = self.input_proj(mel)
        for block in self.blocks:
            x = block(x)

        stats = self.output_proj(x)
        mean, log_std = stats.chunk(2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(log_std)
        return z, mean, log_std
