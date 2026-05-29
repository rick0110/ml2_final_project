"""Loss functions for the direct TTS experiment."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MaskedMelReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predicted_mel: torch.Tensor, target_mel: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        if predicted_mel.dim() != 3 or target_mel.dim() != 3:
            raise ValueError(
                f"predicted_mel and target_mel must be 3D (batch, n_mels, time), got predicted={tuple(predicted_mel.shape)}, target={tuple(target_mel.shape)}"
            )

        max_time = min(predicted_mel.size(-1), target_mel.size(-1))
        predicted_mel = predicted_mel[..., :max_time]
        target_mel = target_mel[..., :max_time]

        mask = torch.arange(max_time, device=predicted_mel.device).unsqueeze(0) < target_lengths.to(device=predicted_mel.device).unsqueeze(1)
        mask = mask.unsqueeze(1).to(dtype=predicted_mel.dtype)

        absolute_error = (predicted_mel - target_mel).abs() * mask
        if self.reduction == "sum":
            return absolute_error.sum()
        normalizer = mask.sum().clamp_min(1.0)
        return absolute_error.sum() / normalizer


class LengthPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, predicted_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        predicted_log = torch.log1p(predicted_lengths.clamp_min(1.0))
        target_log = torch.log1p(target_lengths.to(device=predicted_lengths.device, dtype=predicted_lengths.dtype).clamp_min(1.0))
        return self.loss_fn(predicted_log, target_log)


class DirectTTSLoss(nn.Module):
    def __init__(self, weight_reconstruction: float = 1.0, weight_length: float = 0.1):
        super().__init__()
        self.weight_reconstruction = weight_reconstruction
        self.weight_length = weight_length
        self.reconstruction_loss = MaskedMelReconstructionLoss(reduction="mean")
        self.length_loss = LengthPredictionLoss()

    def forward(
        self,
        predicted_mel: torch.Tensor,
        target_mel: torch.Tensor,
        predicted_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.reconstruction_loss(predicted_mel, target_mel, target_lengths)
        length_loss = self.length_loss(predicted_lengths, target_lengths)
        total_loss = self.weight_reconstruction * recon_loss + self.weight_length * length_loss
        return total_loss, recon_loss, length_loss