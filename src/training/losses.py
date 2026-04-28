"""
Loss functions for prosody and style transfer training.

The total training objective combines:

1. **Mel-spectrogram L1 loss** — primary reconstruction loss between the
   predicted and ground-truth log-mel spectrograms.
2. **Duration loss** — mean-squared error between log-predicted and
   log-ground-truth durations (following FastSpeech 2).
3. **Pitch loss** — MSE on the predicted vs. ground-truth F0 contour.
4. **Energy loss** — MSE on the predicted vs. ground-truth energy contour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelLoss(nn.Module):
    """L1 reconstruction loss on log-mel spectrograms.

    Supports variable-length sequences via a binary mask.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked L1 mel loss.

        Args:
            pred_mel: Predicted mel-spectrogram ``(B, n_mels, T)``.
            target_mel: Ground-truth mel-spectrogram ``(B, n_mels, T)``.
            mask: Frame mask ``(B, T)`` — 1 for valid frames, 0 for padding.

        Returns:
            Scalar loss tensor.
        """
        loss = F.l1_loss(pred_mel, target_mel, reduction="none")  # (B, n_mels, T)
        if mask is not None:
            loss = loss * mask.unsqueeze(1)  # broadcast over mel dim
        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / (mask.sum() * pred_mel.size(1) + 1e-8)
            return loss.mean()
        return loss.sum()


class DurationLoss(nn.Module):
    """MSE loss on log-scale predicted durations.

    Ground-truth durations are transformed to log-scale before computing the
    loss so that the predictor operates in the same space.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_log_durations: torch.Tensor,
        target_durations: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute duration loss.

        Args:
            pred_log_durations: Log-scale predictions ``(B, T, 1)`` or ``(B, T)``.
            target_durations: Integer ground-truth durations ``(B, T)``.
            mask: Encoder frame mask ``(B, T)``.

        Returns:
            Scalar loss tensor.
        """
        pred = pred_log_durations.squeeze(-1)  # (B, T)
        target_log = torch.log(target_durations.float().clamp(min=1.0))
        loss = F.mse_loss(pred, target_log, reduction="none")  # (B, T)
        if mask is not None:
            loss = loss * mask
        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            return loss.mean()
        return loss.sum()


class PitchLoss(nn.Module):
    """MSE loss on pitch (F0) predictions.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pitch loss.

        Args:
            pred_pitch: Predicted F0 ``(B, T, 1)`` or ``(B, T)``.
            target_pitch: Ground-truth F0 ``(B, T, 1)`` or ``(B, T)``.
            mask: Encoder frame mask ``(B, T)``.

        Returns:
            Scalar loss tensor.
        """
        pred = pred_pitch.squeeze(-1)
        tgt = target_pitch.squeeze(-1)
        loss = F.mse_loss(pred, tgt, reduction="none")
        if mask is not None:
            loss = loss * mask
        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            return loss.mean()
        return loss.sum()


class EnergyLoss(nn.Module):
    """MSE loss on energy predictions.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute energy loss.

        Args:
            pred_energy: Predicted energy ``(B, T, 1)`` or ``(B, T)``.
            target_energy: Ground-truth energy ``(B, T, 1)`` or ``(B, T)``.
            mask: Encoder frame mask ``(B, T)``.

        Returns:
            Scalar loss tensor.
        """
        pred = pred_energy.squeeze(-1)
        tgt = target_energy.squeeze(-1)
        loss = F.mse_loss(pred, tgt, reduction="none")
        if mask is not None:
            loss = loss * mask
        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            return loss.mean()
        return loss.sum()


class TotalLoss(nn.Module):
    """Weighted combination of all individual losses.

    Args:
        mel_weight: Weight for the mel-spectrogram loss.
        duration_weight: Weight for the duration loss.
        pitch_weight: Weight for the pitch loss.
        energy_weight: Weight for the energy loss.
    """

    def __init__(
        self,
        mel_weight: float = 1.0,
        duration_weight: float = 1.0,
        pitch_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mel_weight = mel_weight
        self.duration_weight = duration_weight
        self.pitch_weight = pitch_weight
        self.energy_weight = energy_weight

        self.mel_loss = MelLoss()
        self.duration_loss = DurationLoss()
        self.pitch_loss = PitchLoss()
        self.energy_loss = EnergyLoss()

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the total weighted loss.

        Args:
            model_output: Dictionary returned by the model's forward pass.
                Expected keys: ``"mel"``, ``"pred_durations"``,
                ``"pred_pitch"``, ``"pred_energy"``.
            targets: Dictionary of ground-truth values.
                Expected keys: ``"mel"``, ``"durations"``,
                ``"pitch"``, ``"energy"``.
                Optional key: ``"mel_mask"`` / ``"enc_mask"`` for masking.

        Returns:
            Dictionary with keys ``"total"``, ``"mel"``, ``"duration"``,
            ``"pitch"``, ``"energy"`` (all scalar tensors).
        """
        mel_mask = targets.get("mel_mask")
        enc_mask = targets.get("enc_mask")

        l_mel = self.mel_loss(model_output["mel"], targets["mel"], mel_mask)
        l_dur = self.duration_loss(
            model_output["pred_durations"], targets["durations"], enc_mask
        )
        l_pitch = self.pitch_loss(
            model_output["pred_pitch"], targets["pitch"], enc_mask
        )
        l_energy = self.energy_loss(
            model_output["pred_energy"], targets["energy"], enc_mask
        )

        total = (
            self.mel_weight * l_mel
            + self.duration_weight * l_dur
            + self.pitch_weight * l_pitch
            + self.energy_weight * l_energy
        )

        return {
            "total": total,
            "mel": l_mel,
            "duration": l_dur,
            "pitch": l_pitch,
            "energy": l_energy,
        }
