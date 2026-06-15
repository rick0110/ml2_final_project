"""
Loss functions for Tacotron 2 VAE.

Responsibilities:
    - Implement Tacotron2LossVAE: Combined loss for mel reconstruction, gate prediction, and VAE KL divergence.
    - Provide KL weight annealing utilities to stabilize training.

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
except ImportError:
    # Fallback for local imports
    from hparams import Tacotron2VAEHparams


class Tacotron2LossVAE(nn.Module):
    """
    Combined loss function for Tacotron 2 VAE.

    Mathematical Intuition:
        Total Loss = MSE(Mel) + MSE(Post-Mel) + BCE(Gate) + KL_Weight * KL_Divergence

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

    def kl_anneal_function(self, anneal_function: str, lag: int, step: int, k: float, x0: int, upper: float) -> float:
        """
        Calculate the annealed weight for the KL divergence term.

        Args:
            anneal_function (str): Type of annealing ('logistic', 'linear', or 'constant').
            lag (int): Step offset before annealing starts.
            step (int): Current global step.
            k (float): Growth rate for logistic annealing.
            x0 (int): Midpoint for logistic/linear annealing.
            upper (float): Maximum value for the weight.

        Returns:
            float: Computed KL weight.
        """
        if anneal_function == "logistic":
            return float(upper / (1.0 + np.exp(-k * (step - x0))))
        if anneal_function == "linear":
            if step > lag:
                return min(upper, step / x0)
            return 0.0
        if anneal_function == "constant":
            return 0.001
        raise ValueError(f"Unknown anneal function: {anneal_function}")

    def forward(
        self, model_output: Tuple[Tensor, ...], targets: Tuple[Tensor, Tensor], step: int
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

        Returns:
            Tuple[Tensor, Tensor, Tensor, float]: total_loss, reconstruction_loss, kl_loss, kl_weight.
        """
        mel_target, gate_target = targets[0], targets[1]
        gate_target = gate_target.view(-1, 1) # (B*T, 1)

        mel_out, mel_out_postnet, gate_out, _, mu, logvar, _ = model_output
        gate_out = gate_out.view(-1, 1) # (B*T, 1)

        # Mel Reconstruction Losses
        mel_loss: Tensor = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target) # scalar
        
        # Gate Prediction Loss
        gate_loss: Tensor = nn.BCEWithLogitsLoss()(gate_out, gate_target) # scalar
        
        # KL Divergence Loss
        kl_loss: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # scalar
        
        # Weight Annealing
        kl_weight: float = self.kl_anneal_function(
            self.anneal_function, self.lag, step, self.k, self.x0, self.upper
        )

        # Apply KL loss only once every K steps to prevent collapse, as described in the paper
        k_steps = 100 if step < 15000 else 400
        if step % k_steps != 0:
            kl_weight = 0.0

        recon_loss: Tensor = mel_loss + gate_loss # scalar
        total_loss: Tensor = recon_loss + kl_weight * kl_loss # scalar
        
        return total_loss, recon_loss, kl_loss, kl_weight
