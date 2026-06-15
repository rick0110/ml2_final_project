"""
VAE Global Style Token (GST) module.

Responsibilities:
    - Implement ReferenceEncoder: Extract prosodic features from mel-spectrograms using CNNs and GRU.
    - Implement VAE_GST: Variational Autoencoder wrapper around ReferenceEncoder for latent prosody modeling.
    - Provide style embeddings and latent distribution parameters (mu, logvar).

Main Classes:
    - ReferenceEncoder: Neural network to encode spectral temporal features.
    - VAE_GST: Module implementing the reparameterization trick and latent mapping.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    n_mels = mel channels
    H = hidden dimension
    L = latent dimension (z_latent_dim)
    E = style embedding dimension
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from models.tacotron2_vae.coord_conv import CoordConv2d
from models.tacotron2_vae.hparams import Tacotron2VAEHparams


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder for VAE-GST.

    Architecture:
        CoordConv2d -> [Conv2d -> BN -> ReLU] * (k-1) -> GRU (Last Hidden State)

    Inputs:
        inputs:
            Shape (B, n_mels, T)

    Outputs:
        encoded:
            Shape (B, E // 2)

    Example:
        >>> encoder = ReferenceEncoder(hparams)
        >>> x = torch.randn(16, 80, 100)
        >>> out = encoder(x)
    """

    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize the Reference Encoder.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        k: int = len(hparams.ref_enc_filters)
        filters: List[int] = [1] + hparams.ref_enc_filters

        # Initialize convolutional layers
        convs: List[nn.Module] = [
            CoordConv2d(
                in_channels=filters[0],
                out_channels=filters[1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                with_r=True,
            )
        ]
        # Add subsequent convolutional layers
        convs.extend(
            [
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
                for i in range(1, k)
            ]
        )
        self.convs: nn.ModuleList = nn.ModuleList(convs)
        self.bns: nn.ModuleList = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i]) for i in range(k)]
        )

        # Calculate GRU input size
        out_channels: int = self.calculate_channels(
            length=hparams.n_mel_channels,
            kernel_size=3,
            stride=2,
            pad=1,
            n_convs=k
        )
        self.gru: nn.GRU = nn.GRU(
            input_size=hparams.ref_enc_filters[-1] * out_channels,
            hidden_size=hparams.E // 2,
            batch_first=True,
        )
        self.n_mels: int = hparams.n_mel_channels

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass of the Reference Encoder.

        Args:
            inputs (Tensor): Mel-spectrogram input.
                Shape: (B, n_mels, T)

        Returns:
            Tensor: Last hidden state of GRU.
                Shape: (B, E // 2)
        """
        batch_size: int = inputs.size(0)
        # Reshape input: (B, 1, T, n_mels)
        out: Tensor = inputs.contiguous().view(batch_size, 1, -1, self.n_mels)

        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)   # (B, C_i, T_i, n_mels_i)
            out = bn(out)
            out = F.relu(out)

        # Prepare for GRU: (B, T_last, C_last * n_mels_last)
        out = out.transpose(1, 2) # (B, T_last, C_last, n_mels_last)
        T_prime: int = out.size(1)
        out = out.contiguous().view(batch_size, T_prime, -1) # (B, T_last, input_size)

        # Process with GRU
        _, hidden = self.gru(out) # (1, B, E // 2)
        return hidden.squeeze(0)  # (B, E // 2)

    @staticmethod
    def calculate_channels(length: int, kernel_size: int, stride: int, pad: int, n_convs: int) -> int:
        """
        Calculates the output dimension after multiple convolutions.

        Args:
            length (int): Input length.
            kernel_size (int): Kernel size.
            stride (int): Stride.
            pad (int): Padding.
            n_convs (int): Number of layers.

        Returns:
            int: Final output dimension.
        """
        for _ in range(n_convs):
            length = (length - kernel_size + 2 * pad) // stride + 1
        return length


class VAE_GST(nn.Module):
    """
    Variational Autoencoder for Global Style Token (GST) extraction.

    Architecture:
        ReferenceEncoder -> [FC_mu, FC_logvar] -> Reparameterize (z) -> FC_style

    Inputs:
        inputs:
            Shape (B, n_mels, T)

    Outputs:
        style_embed:
            Shape (B, E)
        mu:
            Shape (B, L)
        logvar:
            Shape (B, L)
        z:
            Shape (B, L)

    Example:
        >>> vae_gst = VAE_GST(hparams)
        >>> style, mu, logvar, z = vae_gst(mel_input)
    """

    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize the VAE_GST module.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        self.ref_encoder: ReferenceEncoder = ReferenceEncoder(hparams)
        # Fully connected layers for mean and log-variance
        self.fc1: nn.Linear = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        self.fc2: nn.Linear = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        # Project latent to style embedding
        self.fc3: nn.Linear = nn.Linear(hparams.z_latent_dim, hparams.E)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick for sampling from the latent distribution.

        Args:
            mu (Tensor): Mean. Shape: (B, L)
            logvar (Tensor): Log variance. Shape: (B, L)

        Returns:
            Tensor: Sampled latent variable z. Shape: (B, L)
        """
        if self.training:
            std: Tensor = torch.exp(0.5 * logvar) # (B, L)
            eps: Tensor = torch.randn_like(std)   # (B, L)
            return eps.mul(std).add_(mu)          # (B, L)
        return mu  # Return mean during inference

    def forward(
        self, inputs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE_GST module.

        Args:
            inputs (Tensor): Mel-spectrogram input.
                Shape: (B, n_mels, T)

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: style_embed, mu, logvar, z.
        """
        # Encode
        enc_out: Tensor = self.ref_encoder(inputs) # (B, E // 2)
        
        # Latent distribution parameters
        mu: Tensor = self.fc1(enc_out)     # (B, L)
        logvar: Tensor = self.fc2(enc_out) # (B, L)
        
        # Sample z
        z: Tensor = self.reparameterize(mu, logvar) # (B, L)
        
        # Project to style embedding
        style_embed: Tensor = self.fc3(z) # (B, E)
        
        return style_embed, mu, logvar, z
