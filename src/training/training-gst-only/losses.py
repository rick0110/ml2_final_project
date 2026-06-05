"""Loss functions for first-step TTS training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple



def mel_reconstruction_loss(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss between predicted and target mel spectrograms."""
    return F.l1_loss(predicted_mel, target_mel)

def style_separation_loss(global_style_tokens: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Style separation loss to encourage distinct style representations.
    ATENÇÃO: Deve receber a matriz [num_tokens, dim] e NÃO o output do batch.
    """
    normed = F.normalize(global_style_tokens, dim=-1)
    similarities = torch.matmul(normed, normed.transpose(-1, -2))
    
    mask = torch.eye(global_style_tokens.size(0), device=global_style_tokens.device)
    max_similarities = similarities.masked_fill(mask.bool(), -float('inf')).max(dim=-1).values
    
    return torch.clamp(max_similarities - margin, min=0.0).mean()

def style_consistency_loss(style_gen: torch.Tensor, style_ref: torch.Tensor) -> torch.Tensor:
    """L2 loss between generated and reference style embeddings. (Uso no Step 2)"""
    return F.mse_loss(style_gen, style_ref)

def contrastive_style_loss(style_gen: torch.Tensor, style_ref: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Contrastive loss to encourage discriminative style representations. (Uso no Step 2)"""
    ref_norm = F.normalize(style_ref, dim=-1)
    gen_norm = F.normalize(style_gen, dim=-1)
    
    logits = torch.matmul(gen_norm, ref_norm.transpose(-1, -2)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)



class CombinedTTSLoss(nn.Module):
    """Combined loss for First-Step TTS training.
    Total Loss = w_recon * mel_recon + w_diversity * style_separation
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
        self.margin = diversity_margin
    
    def forward(
        self,
        predicted_mel: torch.Tensor,
        target_mel: torch.Tensor,
        global_style_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        recon_loss = mel_reconstruction_loss(predicted_mel, target_mel)
        
        div_loss = style_separation_loss(global_style_tokens, margin=self.margin)
        
        total_loss = (self.weight_reconstruction * recon_loss) + (self.weight_diversity * div_loss)
        
        return total_loss, recon_loss, div_loss
    

class Loss_audio(nn.Module):
    """Loss para o modelo de áudio (Step 2), combinando consistência de estilo e contraste."""
    
    def __init__(
        self,
        weight_consistency: float = 1.0,
    ):
        super().__init__()
        self.weight_consistency = weight_consistency
    
    def forward(
        self,
        audio_gen: torch.Tensor,
        audio_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return self.weight_consistency * F.mse_loss(audio_gen, audio_ref)