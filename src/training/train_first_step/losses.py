"""Loss functions for first-step TTS training.

Includes:
- L1 Reconstruction Loss: MSE between predicted and target mel spectrograms
- Style Diversity Loss: Penalizes style tokens from collapsing into the same space
"""

import torch
import torch.nn as nn
from typing import Tuple


class L1ReconstructionLoss(nn.Module):
    """L1 reconstruction loss between predicted and target mel spectrograms.
    
    Args:
        reduction: 'mean' or 'sum' reduction over batch and time dimensions
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
    
    def forward(self, predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        """Compute L1 reconstruction loss.
        
        Args:
            predicted_mel: Shape (batch_size, n_mels, time_steps) - predicted mel spectrogram
            target_mel: Shape (batch_size, n_mels, time_steps) - target mel spectrogram
        
        Returns:
            Scalar loss value
        """
        return self.loss_fn(predicted_mel, target_mel)


class StyleDiversityLoss(nn.Module):
    """Style diversity loss that penalizes style tokens from collapsing.
    
    Encourages diversity by minimizing pairwise similarities between style embeddings.
    Uses cosine similarity as the metric for distance.
    
    Args:
        margin: Minimum distance margin to maintain between embeddings (default: 0.1)
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, style_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute style diversity loss.
        
        Penalizes when style embeddings become too similar (collapse into same space).
        
        Args:
            style_embeddings: Shape (batch_size, embedding_dim) - style embeddings from GST
        
        Returns:
            Scalar loss value (mean over batch)
        """
        # Normalize embeddings for cosine similarity
        norm_embeddings = torch.nn.functional.normalize(style_embeddings, p=2, dim=-1)
        
        # Compute pairwise cosine similarities
        # Shape: (batch_size, batch_size)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # Create mask for off-diagonal elements (don't compare with self)
        batch_size = style_embeddings.shape[0]
        mask = torch.eye(batch_size, device=style_embeddings.device, dtype=torch.bool)
        
        # Extract off-diagonal similarities
        off_diagonal_similarities = similarity_matrix[~mask]
        
        # Penalize high similarities (low distances)
        # We want distances > margin, so we penalize when similarity > (1 - margin)
        # Using ReLU to only penalize positive violations
        diversity_loss = torch.relu(off_diagonal_similarities - (1 - self.margin))
        
        return diversity_loss.mean()


class CombinedTTSLoss(nn.Module):
    """Combined loss for TTS training.
    
    Total Loss = w_recon * L1_recon + w_diversity * style_diversity
    
    Args:
        weight_reconstruction: Weight for L1 reconstruction loss (default: 1.0)
        weight_diversity: Weight for style diversity loss (default: 0.5)
        diversity_margin: Margin for style diversity loss (default: 0.1)
    """
    
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
        """Compute combined TTS loss.
        
        Args:
            predicted_mel: Shape (batch_size, n_mels, time_steps)
            target_mel: Shape (batch_size, n_mels, time_steps)
            style_embeddings: Shape (batch_size, embedding_dim)
        
        Returns:
            Tuple of (total_loss, reconstruction_loss, diversity_loss)
        """
        recon_loss = self.l1_loss(predicted_mel, target_mel)
        div_loss = self.diversity_loss(style_embeddings)
        
        total_loss = (
            self.weight_reconstruction * recon_loss +
            self.weight_diversity * div_loss
        )
        
        return total_loss, recon_loss, div_loss
