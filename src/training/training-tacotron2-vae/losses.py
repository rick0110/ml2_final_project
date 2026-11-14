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
        Total Loss = MSE(Mel) + MSE(Post-Mel) + BCE(Gate)
                   + KL_Weight * KL_Divergence
                   + guided_attention_weight * GuidedAttentionLoss
                   + contrastive_weight * SpeakerContrastiveLoss (optional)
        KL uses free bits to prevent posterior collapse.
        Guided attention penalises non-monotonic attention diagonally.

    Architecture:
        Sum of reconstruction errors, weighted latent divergence, and
        diagonal attention prior.

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
        guided_attn_loss: Scalar
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
        self.free_bits: float = getattr(hparams, 'free_bits', 0.5)
        self.guided_attention_weight: float = getattr(hparams, 'guided_attention_weight', 2.0)
        self.guided_attention_sigma: float = getattr(hparams, 'guided_attention_sigma', 0.4)
        self.gate_positive_weight: float = getattr(hparams, 'gate_positive_weight', 1.0)
        self.contrastive_weight: float = getattr(hparams, 'contrastive_weight', 0.0)
        self.contrastive_temperature: float = getattr(hparams, 'contrastive_temperature', 0.1)
        self.contrastive_memory: int = getattr(hparams, 'contrastive_memory', 512)
        # Latents of recent batches, used as extra negatives (most of a batch is one speaker)
        self.memory_latents: Optional[Tensor] = None
        self.memory_speakers: Optional[Tensor] = None
        self.last_contrastive: float = 0.0

        # Pre-instantiate loss functions to avoid re-creation every forward pass
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    @staticmethod
    def compute_guided_attention_loss(
        alignment: Tensor,
        input_lengths: Tensor,
        output_lengths: Tensor,
        sigma: float,
    ) -> Tensor:
        """
        Diagonal guided attention loss.

        Penalises attention weights that deviate from the expected diagonal
        (monotonic text→audio alignment).

        W[b,t,n] = 1 - exp(-(n/N_b - t/T_b)^2 / (2*sigma^2))

        Args:
            alignment (Tensor): Attention weights (B, T_mel, T_text).
            input_lengths (Tensor): Text lengths (B,).
            output_lengths (Tensor): Mel lengths (B,).
            sigma (float): Gaussian width controlling tolerance.

        Returns:
            Tensor: Scalar guided attention loss.
        """
        B, T_mel, T_text = alignment.size()
        device = alignment.device

        t_idx = torch.arange(T_mel, device=device, dtype=torch.float32)
        n_idx = torch.arange(T_text, device=device, dtype=torch.float32)

        # Normalise positions to [0,1] per sample: (B, T_mel, 1) and (B, 1, T_text)
        N = input_lengths.float().clamp(min=1).view(B, 1, 1)
        T = output_lengths.float().clamp(min=1).view(B, 1, 1)
        t_norm = t_idx.view(1, T_mel, 1) / T  # (B, T_mel, 1) via broadcasting
        n_norm = n_idx.view(1, 1, T_text) / N  # (B, 1, T_text)

        # Penalty weight: high off-diagonal, zero on diagonal
        W = 1.0 - torch.exp(-((n_norm - t_norm) ** 2) / (2.0 * sigma ** 2))  # (B, T_mel, T_text)

        # Mask padding positions so they don't contribute to the loss
        mel_mask = (torch.arange(T_mel, device=device).view(1, T_mel, 1)
                    < output_lengths.view(B, 1, 1))  # (B, T_mel, 1)
        text_mask = (torch.arange(T_text, device=device).view(1, 1, T_text)
                     < input_lengths.view(B, 1, 1))  # (B, 1, T_text)
        valid = (mel_mask & text_mask).float()

        denom = valid.sum().clamp(min=1.0)
        return (W * alignment * valid).sum() / denom

    def compute_contrastive_loss(self, mu_reference: Tensor, mu_target: Tensor, speakers: Tensor) -> Tensor:
        """
        Speaker-contrastive loss on the style latent (InfoNCE, both directions).

        The reference and the target of an item are two utterances of one speaker, so they
        are the positive pair. Negatives are latents of other speakers only, from this batch
        and from a memory of recent batches; same-speaker items are never pushed apart.

        Args:
            mu_reference (Tensor): Latent mean of the style reference (B, L).
            mu_target (Tensor): Latent mean of the target utterance (B, L).
            speakers (Tensor): Speaker index of each item (B,).

        Returns:
            Tensor: Scalar loss, averaged over the items that have at least one negative.
        """
        reference = nn.functional.normalize(mu_reference.float(), dim=1)
        target = nn.functional.normalize(mu_target.float(), dim=1)
        bank, bank_speakers = torch.cat([reference, target]).detach(), torch.cat([speakers, speakers])
        if self.memory_latents is not None:
            bank = torch.cat([bank, self.memory_latents.to(bank.device)])
            bank_speakers = torch.cat([bank_speakers, self.memory_speakers.to(bank.device)])
        negative = bank_speakers.unsqueeze(0) != speakers.unsqueeze(1)  # (B, M)
        has_negative = negative.any(dim=1)

        loss = mu_reference.new_zeros(())
        if has_negative.any():
            positive = (reference * target).sum(dim=1, keepdim=True) / self.contrastive_temperature  # (B, 1)
            for anchor in (reference, target):
                logits = (anchor @ bank.t() / self.contrastive_temperature).masked_fill(~negative, float('-inf'))
                per_item = torch.logsumexp(torch.cat([positive, logits], dim=1), dim=1) - positive.squeeze(1)
                loss = loss + 0.5 * per_item[has_negative].mean()

        keep = self.contrastive_memory
        self.memory_latents, self.memory_speakers = bank[:keep], bank_speakers[:keep]
        return loss

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
        self,
        model_output: Tuple[Tensor, ...],
        targets: Tuple[Tensor, Tensor],
        step: int,
        output_lengths: Optional[Tensor] = None,
        input_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, float, Tensor]:
        """
        Compute the total Tacotron 2 VAE loss.

        Args:
            model_output (Tuple[Tensor, ...]): Predicted values from the model.
                Includes:
                - mel_out: (B, n_mels, T)
                - mel_out_postnet: (B, n_mels, T)
                - gate_out: (B, T)
                - alignment: (B, T_mel, T_text)
                - mu: (B, L)
                - logvar: (B, L)
                - z: (B, L)
            targets (Tuple[Tensor, Tensor]): Ground truth values.
                Includes:
                - mel_target: (B, n_mels, T)
                - gate_target: (B, T)
            step (int): Current training step.
            output_lengths (Tensor, optional): Mel lengths for masking (B,).
            input_lengths (Tensor, optional): Text lengths for guided attention (B,).

        Returns:
            Tuple: total_loss, reconstruction_loss, kl_loss, kl_weight, guided_attn_loss.
        """
        mel_target, gate_target = targets[0], targets[1]

        mel_out, mel_out_postnet, gate_out, alignment, mu, logvar, _ = model_output[:7]

        # --- Reconstruction losses (masked over valid frames only) ---
        mel_loss_raw: Tensor = self.mse_loss(mel_out, mel_target) + self.mse_loss(mel_out_postnet, mel_target)  # (B, n_mels, T)
        gate_loss_raw: Tensor = self.bce_loss(gate_out, gate_target)  # (B, T)
        if self.gate_positive_weight != 1.0:
            # Equivalent to BCEWithLogitsLoss(pos_weight=w) for binary targets:
            # scales only the stop-frame terms, countering the stop/continue imbalance
            gate_loss_raw = gate_loss_raw * (1.0 + (self.gate_positive_weight - 1.0) * gate_target)

        if output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths).to(mel_out.device)  # (B, T) — True for valid positions
            mel_loss = (mel_loss_raw * mask.unsqueeze(1)).sum() / (mask.unsqueeze(1).sum() * mel_out.size(1))
            gate_loss = (gate_loss_raw * mask).sum() / mask.sum()
        else:
            mel_loss = mel_loss_raw.mean()
            gate_loss = gate_loss_raw.mean()

        # --- KL Divergence with Free Bits ---
        kl_per_dim: Tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, L)
        kl_per_dim_mean: Tensor = kl_per_dim.mean(dim=0)                      # (L,)
        kl_per_dim_clamped: Tensor = torch.clamp(kl_per_dim_mean, min=self.free_bits)
        kl_loss: Tensor = kl_per_dim_clamped.mean()  # scalar

        kl_weight: float = self.kl_anneal_function(
            self.anneal_function, self.lag, step, self.k, self.x0, self.upper
        )

        # --- Guided Attention Loss ---
        guided_attn_loss: Tensor = torch.tensor(0.0, device=mel_out.device)
        if (self.guided_attention_weight > 0.0
                and alignment is not None
                and input_lengths is not None
                and output_lengths is not None):
            guided_attn_loss = self.compute_guided_attention_loss(
                alignment, input_lengths, output_lengths, self.guided_attention_sigma
            )

        recon_loss: Tensor = mel_loss + gate_loss
        total_loss: Tensor = (
            recon_loss
            + kl_weight * kl_loss
            + self.guided_attention_weight * guided_attn_loss
        )
        if self.contrastive_weight > 0.0 and len(model_output) > 8:
            contrastive: Tensor = self.compute_contrastive_loss(mu, model_output[7], model_output[8])
            self.last_contrastive = float(contrastive.detach())
            total_loss = total_loss + self.contrastive_weight * contrastive

        return total_loss, recon_loss, kl_loss, kl_weight, guided_attn_loss
