import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def mel_reconstruction_loss(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss between predicted and target mel spectrograms."""
    return F.l1_loss(predicted_mel, target_mel)

def style_consistency_loss(style_gen: torch.Tensor, style_ref: torch.Tensor) -> torch.Tensor:
    """L2 loss between generated and reference style embeddings."""
    return F.mse_loss(style_gen, style_ref)

def style_separation_loss(style_embeddings: torch.Tensor) -> torch.Tensor:
    """Style separation loss to encourage distinct style representations."""
    # Compute pairwise cosine similarity between all style embeddings
    normed = F.normalize(style_embeddings, dim=-1)
    similarities = torch.matmul(normed, normed.transpose(-1, -2))
    
    # Mask out self-similarity and take max
    mask = torch.eye(style_embeddings.size(0), device=style_embeddings.device)
    max_similarities = similarities.masked_fill(mask.bool(), -float('inf')).max(dim=-1).values
    
    # Encourage maximum similarity to be less than margin
    margin = 0.1
    return torch.clamp(max_similarities - margin, min=0.0).mean()

def contrastive_style_loss(style_gen: torch.Tensor, style_ref: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Contrastive loss to encourage discriminative style representations."""
    ref_norm = F.normalize(style_ref, dim=-1)
    gen_norm = F.normalize(style_gen, dim=-1)
    
    # Compute logits and apply temperature
    logits = torch.matmul(gen_norm, ref_norm.transpose(-1, -2)) / temperature
    
    # Compute cross-entropy loss
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def total_loss(
    predicted_mel: torch.Tensor,
    target_mel: torch.Tensor,
    style_gen: torch.Tensor,
    style_ref: torch.Tensor,
    style_embeddings: torch.Tensor,
    loss_weights: Dict[str, float]
) -> torch.Tensor:
    """Combine all loss components with configurable weights."""
    losses = {}
    
    # Compute individual losses
    losses["mel_recon"] = mel_reconstruction_loss(predicted_mel, target_mel)
    losses["style_consistency"] = style_consistency_loss(style_gen, style_ref)
    losses["style_separation"] = style_separation_loss(style_embeddings)
    losses["contrastive_style"] = contrastive_style_loss(style_gen, style_ref)
    
    # Weighted sum of losses
    total = sum(weight * loss for loss, weight in losses.items() if weight > 0)
    
    return total, losses
