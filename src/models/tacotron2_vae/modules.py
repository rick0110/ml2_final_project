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
        self.attn: nn.Linear = nn.Linear(hparams.E // 2, 1)
        self.n_mels: int = hparams.n_mel_channels

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass of the Reference Encoder.

        Args:
            inputs (Tensor): Mel-spectrogram input.
                Shape: (B, n_mels, T)

        Returns:
            Tensor: Attention-pooled GRU encoding.
                Shape: (B, E // 2)
        """
        encoded, _ = self._encode(self._cnn_forward(inputs))
        return encoded

    def _cnn_forward(self, inputs: Tensor) -> Tensor:
        batch_size: int = inputs.size(0)
        out: Tensor = inputs.contiguous().view(batch_size, 1, -1, self.n_mels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
        out = out.transpose(1, 2)
        T_prime: int = out.size(1)
        return out.contiguous().view(batch_size, T_prime, -1)

    def _encode(self, gru_input: Tensor) -> Tuple[Tensor, Tensor]:
        out, _ = self.gru(gru_input)           # (B, T', E // 2)
        scores = self.attn(out)                # (B, T', 1)
        weights = torch.softmax(scores, dim=1) # (B, T', 1)
        encoded = (out * weights).sum(dim=1)   # (B, E // 2)
        return encoded, weights.squeeze(-1)    # (B, T')

    def encode_with_attention(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass retornando o encoding e os pesos de atenção temporal.

        Args:
            inputs (Tensor): Mel-spectrogram. Shape: (B, n_mels, T)

        Returns:
            Tuple[Tensor, Tensor]:
                encoded: Shape (B, E // 2)
                weights: Pesos de atenção por frame da GRU. Shape (B, T')
        """
        return self._encode(self._cnn_forward(inputs))

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
            return eps * std + mu                  # (B, L)
        return mu  # Return mean during inference

    def _vae_from_enc(self, enc_out: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu: Tensor = self.fc1(enc_out)
        logvar: Tensor = torch.clamp(self.fc2(enc_out), min=-10.0, max=2.0)
        z: Tensor = self.reparameterize(mu, logvar)
        style_embed: Tensor = self.fc3(z)
        return style_embed, mu, logvar, z

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
        enc_out: Tensor = self.ref_encoder(inputs)
        return self._vae_from_enc(enc_out)

    def forward_with_attention(
        self, inputs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass retornando as saídas do VAE e os pesos de atenção temporal.

        Args:
            inputs (Tensor): Mel-spectrogram input.
                Shape: (B, n_mels, T)

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                style_embed: Shape (B, E)
                mu: Shape (B, L)
                logvar: Shape (B, L)
                z: Shape (B, L)
                temporal_weights: Pesos de atenção por frame da GRU. Shape (B, T')
        """
        enc_out, temporal_weights = self.ref_encoder.encode_with_attention(inputs)
        style_embed, mu, logvar, z = self._vae_from_enc(enc_out)
        return style_embed, mu, logvar, z, temporal_weights
