"""
HiFi-GAN generator (V1), used as a frozen vocoder: mel-spectrogram -> waveform.

Adapted from https://github.com/jik876/hifi-gan (MIT License, Jungil Kong et al.,
"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech
Synthesis", NeurIPS 2020). Only the inference path is kept.

The UNIVERSAL_V1 checkpoint was trained on many speakers and languages with the
same mel settings as this project (22.05 kHz, 80 mels, FFT 1024, hop 256,
f_max 8 kHz, log-compressed with a 1e-5 floor), so the Tacotron output can be
fed to it directly.

Tensor conventions:
    mel: (B, 80, T)   ->   audio: (B, 1, T * 256)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1
UPSAMPLE_RATES = [8, 8, 2, 2]
UPSAMPLE_KERNELS = [16, 16, 4, 4]
UPSAMPLE_INITIAL_CHANNEL = 512
RESBLOCK_KERNELS = [3, 7, 11]
RESBLOCK_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=_padding(kernel_size, d)))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_padding(kernel_size)))
            for _ in dilations
        ])

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x


class HifiGanGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(80, UPSAMPLE_INITIAL_CHANNEL, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = UPSAMPLE_INITIAL_CHANNEL
        for rate, kernel in zip(UPSAMPLE_RATES, UPSAMPLE_KERNELS):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, ch // 2, kernel, rate, padding=(kernel - rate) // 2)))
            ch //= 2
            for k, d in zip(RESBLOCK_KERNELS, RESBLOCK_DILATIONS):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, mel: Tensor) -> Tensor:
        x = self.conv_pre(mel)
        n = len(RESBLOCK_KERNELS)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            x = sum(self.resblocks[i * n + j](x) for j in range(n)) / n
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def remove_weight_norm(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass


def load_hifigan(checkpoint_path: str | Path, device: torch.device | str = "cpu") -> HifiGanGenerator:
    """Load a jik876/hifi-gan generator checkpoint (e.g. UNIVERSAL_V1 `g_02500000`)."""
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model = HifiGanGenerator()
    model.load_state_dict(state["generator"])
    model.remove_weight_norm()
    return model.eval().to(device)


@torch.inference_mode()
def vocode(model: HifiGanGenerator, mel: Tensor) -> Tensor:
    """mel (80, T) or (B, 80, T) in log scale -> waveform (T * 256,) or (B, T * 256) in [-1, 1]."""
    single = mel.dim() == 2
    audio = model(mel.unsqueeze(0) if single else mel).squeeze(1)
    return audio[0] if single else audio
