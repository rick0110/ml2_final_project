from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size - 1) * dilation // 2
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            h = F.leaky_relu(x, 0.1)
            h = conv1(h)
            h = F.leaky_relu(h, 0.1)
            h = conv2(h)
            x = h + x
        return x


class HiFiGenerator(nn.Module):
    def __init__(
        self,
        input_channels: int = 192,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_kernel_sizes: List[int] | None = None,
        resblock_dilations: List[List[int]] | None = None,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates or [8, 8, 2, 2]
        self.upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 4, 4]
        self.resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        self.resblock_dilations = resblock_dilations or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.conv_pre = weight_norm(nn.Conv1d(input_channels, 512, kernel_size=7, padding=3))

        self.upsamplers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        in_channels = 512
        for rate, kernel in zip(self.upsample_rates, self.upsample_kernel_sizes):
            out_channels = in_channels // 2
            padding = (kernel - rate) // 2
            self.upsamplers.append(
                weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel, stride=rate, padding=padding))
            )
            for res_k, res_d in zip(self.resblock_kernel_sizes, self.resblock_dilations):
                self.resblocks.append(ResidualBlock(out_channels, res_k, res_d))
            in_channels = out_channels

        self.conv_post = weight_norm(nn.Conv1d(in_channels, 1, kernel_size=7, padding=3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"z must be 3D (batch, channels, time), got {tuple(z.shape)}")

        x = self.conv_pre(z)
        resblock_idx = 0
        for upsampler in self.upsamplers:
            x = F.leaky_relu(x, 0.1)
            x = upsampler(x)
            res_outputs = []
            for _ in range(len(self.resblock_kernel_sizes)):
                res_outputs.append(self.resblocks[resblock_idx](x))
                resblock_idx += 1
            x = sum(res_outputs) / len(res_outputs)

        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.conv_post(x))
        return x


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"audio must be 3D (batch, 1, time), got {tuple(x.shape)}")

        batch, channels, time = x.size()
        if time % self.period != 0:
            pad = self.period - (time % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            time = time + pad

        x = x.view(batch, channels, time // self.period, self.period)
        features: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int] | None = None):
        super().__init__()
        self.periods = periods or [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([DiscriminatorP(period) for period in self.periods])

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        return [discriminator(x) for discriminator in self.discriminators]


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
                weight_norm(nn.Conv1d(128, 128, kernel_size=41, stride=2, padding=20, groups=4)),
                weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, padding=20, groups=4)),
                weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=4)),
                weight_norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, padding=20, groups=4)),
                weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, padding=20, groups=4)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS() for _ in range(3)])
        self.pooling = nn.ModuleList([nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        results = []
        for index, discriminator in enumerate(self.discriminators):
            results.append(discriminator(x))
            if index < len(self.pooling):
                x = self.pooling[index](x)
        return results


class HiFiGanDiscriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, List[torch.Tensor]]], List[Tuple[torch.Tensor, List[torch.Tensor]]]]:
        return self.mpd(x), self.msd(x)
