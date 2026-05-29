"""Loss functions for the cross-attention TTS experiment."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class L1ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predicted_mel, target_mel)


class StyleDiversityLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, style_embeddings: torch.Tensor) -> torch.Tensor:
        norm_embeddings = torch.nn.functional.normalize(style_embeddings, p=2, dim=-1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())

        batch_size = style_embeddings.shape[0]
        mask = torch.eye(batch_size, device=style_embeddings.device, dtype=torch.bool)
        off_diagonal_similarities = similarity_matrix[~mask]

        diversity_loss = torch.relu(off_diagonal_similarities - (1 - self.margin))
        return diversity_loss.mean()


class CombinedTTSLoss(nn.Module):
    def __init__(
        self,
        weight_reconstruction: float = 1.0,
        weight_diversity: float = 0.5,
        diversity_margin: float = 0.1,
    ):
        super().__init__()
        self.weight_reconstruction = weight_reconstruction
        self.weight_diversity = weight_diversity
        self.l1_loss = L1ReconstructionLoss(reduction="mean")
        self.diversity_loss = StyleDiversityLoss(margin=diversity_margin)

    def forward(
        self,
        predicted_mel: torch.Tensor,
        target_mel: torch.Tensor,
        style_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.l1_loss(predicted_mel, target_mel)
        div_loss = self.diversity_loss(style_embeddings)
        total_loss = self.weight_reconstruction * recon_loss + self.weight_diversity * div_loss
        return total_loss, recon_loss, div_loss
