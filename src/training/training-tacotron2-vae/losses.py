"""
Loss functions for Tacotron 2 VAE.

Responsibilities:
    - Implement Tacotron2LossVAE: Combined loss for mel reconstruction, gate prediction, and VAE KL divergence.
    - Provide KL weight annealing utilities to stabilize training.
    - Prevent posterior collapse via free bits and cyclical annealing.

Main Classes:
    - Tacotron2LossVAE: Top-level loss module for the VAE-based model.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    n_mels = mel channels
    L = latent dimension
"""
import numpy as np
import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional

try:
    from models.tacotron2_vae.hparams import Tacotron2VAEHparams
    from models.tacotron2_vae.utils import get_mask_from_lengths
except ImportError:
    # Fallback for local imports
    from hparams import Tacotron2VAEHparams
    from utils import get_mask_from_lengths


class Tacotron2LossVAE(nn.Module):
    """
    Combined loss function for Tacotron 2 VAE.

    Mathematical Intuition:
        Total Loss = MSE(Mel) + MSE(Post-Mel) + BCE(Gate) + KL_Weight * KL_Divergence
        KL uses free bits to prevent posterior collapse.

    Architecture:
        Sum of reconstruction errors and weighted latent divergence.

    Inputs:
        model_output:
            [mel_out, mel_out_postnet, gate_out, alignment, mu, logvar, z]
        targets:
            (mel_target, gate_target)
        step:
            Current training step (for KL annealing).

    Outputs:
        total_loss: Scalar
        recon_loss: Scalar (Mel + Gate)
        kl_loss: Scalar
        kl_weight: Scalar
    """

    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize the loss function.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        self.anneal_function: str = hparams.anneal_function
        self.lag: int = hparams.anneal_lag
        self.k: float = hparams.anneal_k
        self.x0: int = hparams.anneal_x0
        self.upper: float = hparams.anneal_upper
        self.free_bits: float = getattr(hparams, 'free_bits', 0.25)

        # Pre-instantiate loss functions to avoid re-creation every forward pass
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def kl_anneal_function(self, anneal_function: str, lag: int, step: int, k: float, x0: int, upper: float) -> float:
        """
        Calculate the annealed weight for the KL divergence term.

        All annealing functions respect the lag parameter: returns 0.0 when step < lag,
        and uses effective_step = step - lag for the actual computation.

        Args:
            anneal_function (str): Type of annealing ('logistic', 'linear', 'cyclical', or 'constant').
            lag (int): Number of steps before annealing starts.
            step (int): Current global step.
            k (float): Growth rate for logistic annealing.
            x0 (int): Midpoint for logistic/linear annealing.
            upper (float): Maximum value for the weight.

        Returns:
            float: Computed KL weight.
        """
        # All functions respect the lag parameter
        if step < lag:
            return 0.0

        effective_step = step - lag

        if anneal_function == "logistic":
            return float(upper / (1.0 + np.exp(-k * (effective_step - x0))))
        if anneal_function == "linear":
            return float(min(upper, effective_step / max(x0, 1)))
        if anneal_function == "cyclical":
            # Cyclical annealing: repeats a linear ramp every x0 steps
            cycle_length = max(x0, 1)
            cycle_pos = effective_step % (cycle_length * 2)
            if cycle_pos < cycle_length:
                # Ramp up phase
                return float(upper * cycle_pos / cycle_length)
            else:
                # Hold at upper
                return float(upper)
        if anneal_function == "constant":
            return float(upper)
        raise ValueError(f"Unknown anneal function: {anneal_function}")

    def forward(
        self, model_output: Tuple[Tensor, ...], targets: Tuple[Tensor, Tensor], step: int, output_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Compute the total Tacotron 2 VAE loss.

        Args:
            model_output (Tuple[Tensor, ...]): Predicted values from the model.
                Includes: 
                - mel_out: (B, n_mels, T)
                - mel_out_postnet: (B, n_mels, T)
                - gate_out: (B, T)
                - alignment: (B, T, T_text)
                - mu: (B, L)
                - logvar: (B, L)
            targets (Tuple[Tensor, Tensor]): Ground truth values.
                Includes:
                - mel_target: (B, n_mels, T)
                - gate_target: (B, T)
            step (int): Current training step.
            output_lengths (Tensor, optional): Lengths of the sequences for masking.

        Returns:
            Tuple[Tensor, Tensor, Tensor, float]: total_loss, reconstruction_loss, kl_loss, kl_weight.
        """
        mel_target, gate_target = targets[0], targets[1]
        
        mel_out, mel_out_postnet, gate_out, _, mu, logvar, _ = model_output

        # Raw losses
        mel_loss_raw: Tensor = self.mse_loss(mel_out, mel_target) + self.mse_loss(mel_out_postnet, mel_target) # (B, n_mels, T)
        gate_loss_raw: Tensor = self.bce_loss(gate_out, gate_target) # (B, T)
        
        if output_lengths is not None:
            # Generate boolean mask from lengths: (B, T)
            mask = get_mask_from_lengths(output_lengths).to(mel_out.device)
            
            # Compute masked means
            mel_loss = (mel_loss_raw * mask.unsqueeze(1)).sum() / (mask.unsqueeze(1).sum() * mel_out.size(1))
            gate_loss = (gate_loss_raw * mask).sum() / mask.sum()
        else:
            mel_loss = mel_loss_raw.mean()
            gate_loss = gate_loss_raw.mean()
        
        # KL Divergence Loss with Free Bits
        # Compute KL per dimension: (B, L)
        kl_per_dim: Tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Compute mean KL per dimension over the batch: shape (L,)
        kl_per_dim_mean: Tensor = kl_per_dim.mean(dim=0)
        
        # Clamp each dimension's expectation at the free_bits threshold: shape (L,)
        kl_per_dim_clamped: Tensor = torch.clamp(kl_per_dim_mean, min=self.free_bits)
        
        # Average over latent dimensions: scalar
        kl_loss: Tensor = kl_per_dim_clamped.mean() # scalar
        
        # Weight Annealing
        kl_weight: float = self.kl_anneal_function(
            self.anneal_function, self.lag, step, self.k, self.x0, self.upper
        )

        recon_loss: Tensor = mel_loss + gate_loss # scalar
        total_loss: Tensor = recon_loss + kl_weight * kl_loss # scalar
        
        return total_loss, recon_loss, kl_loss, kl_weight
