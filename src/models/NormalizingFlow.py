from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        hidden_channels: int = 192,
        kernel_size: int = 3,
        scale_limit: float = 1.5,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even for coupling split")
        padding = (kernel_size - 1) // 2
        self.scale_limit = float(scale_limit)
        self.net = nn.Sequential(
            nn.Conv1d(channels // 2 + cond_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x.chunk(2, dim=1)
        h = torch.cat([x_a, cond], dim=1)
        shift, log_scale = self.net(h).chunk(2, dim=1)
        log_scale = self.scale_limit * torch.tanh(log_scale)
        y_b = x_b * torch.exp(log_scale) + shift
        y = torch.cat([x_a, y_b], dim=1)
        log_det = log_scale.sum(dim=(1, 2))
        return y, log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y_a, y_b = y.chunk(2, dim=1)
        h = torch.cat([y_a, cond], dim=1)
        shift, log_scale = self.net(h).chunk(2, dim=1)
        log_scale = self.scale_limit * torch.tanh(log_scale)
        x_b = (y_b - shift) * torch.exp(-log_scale)
        return torch.cat([y_a, x_b], dim=1)

    def zero_initialize(self) -> None:
        last_layer = self.net[-1]
        if hasattr(last_layer, "weight"):
            nn.init.normal_(last_layer.weight, mean=0.0, std=1e-3)
        if hasattr(last_layer, "bias") and last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)


class ChannelFlip(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(1,))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(1,))


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        channels: int = 192,
        cond_channels: int = 192,
        n_flows: int = 4,
        hidden_channels: int = 192,
    ):
        super().__init__()
        self.flows = nn.ModuleList([])
        for _ in range(n_flows):
            self.flows.append(AffineCoupling(channels, cond_channels, hidden_channels=hidden_channels))
            self.flows.append(ChannelFlip())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        for flow in self.flows:
            if isinstance(flow, AffineCoupling):
                z, log_det = flow(z, cond)
                log_det_total = log_det_total + log_det
            else:
                z = flow(z)
        return z, log_det_total

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = z
        for flow in reversed(self.flows):
            if isinstance(flow, AffineCoupling):
                x = flow.inverse(x, cond)
            else:
                x = flow.inverse(x)
        return x

    def initialize_identity(self) -> None:
        for flow in self.flows:
            if isinstance(flow, AffineCoupling):
                flow.zero_initialize()
