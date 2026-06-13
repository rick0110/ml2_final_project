"""
VAE Global Style Token (GST) module.

This module implements the Variational Autoencoder (VAE) component
used for Global Style Token (GST) extraction, specifically designed
to capture prosodic features from mel-spectrograms. It consists of
a Reference Encoder and fully connected layers to produce style embeddings,
mean (mu), and log-variance (logvar) for the latent space.

Dependencies:
    - torch: PyTorch for neural network operations.
    - torch.nn: PyTorch neural network modules.
    - torch.nn.functional: PyTorch functional API.
    - models.tacotron2_vae.coord_conv: Custom Coordinate Convolution layer.
    - models.tacotron2_vae.hparams: Hyperparameters for the VAE_GST.

Typical Usage:
    >>> from src.models.tacotron2_vae.hparams import create_hparams
    >>> hparams = create_hparams()
    >>> vae_gst = VAE_GST(hparams)
    >>> mel_spectrogram = torch.randn(16, 80, 100) # Batch, n_mel_channels, time_steps
    >>> style_embedding, mu, logvar, z = vae_gst(mel_spectrogram)
    >>> print(style_embedding.shape, mu.shape, logvar.shape, z.shape)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.tacotron2_vae.coord_conv import CoordConv2d
from models.tacotron2_vae.hparams import Tacotron2VAEHparams


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder for VAE-GST.

    This module encodes mel-spectrograms into a fixed-size representation
    using a series of convolutional layers followed by a GRU. It's designed
    to capture style information from the input audio features.

    Args:
        hparams (Tacotron2VAEHparams): Hyperparameters object containing configurations
                                       for the reference encoder.

    Attributes:
        convs (nn.ModuleList): List of convolutional layers. Uses CoordConv2d for the first layer.
        bns (nn.ModuleList): List of BatchNorm2d layers corresponding to the convolutional layers.
        gru (nn.GRU): Gated Recurrent Unit layer for sequence encoding.
        n_mels (int): Number of mel-spectrogram channels.
    """

    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        k: int = len(hparams.ref_enc_filters)
        filters: list[int] = [1] + hparams.ref_enc_filters  # Add input channel dimension

        # Initialize convolutional layers
        convs: list[nn.Module] = [
            CoordConv2d(
                in_channels=filters[0],
                out_channels=filters[1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                with_r=True,  # Use coordinate features with radius
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

        # Calculate the output number of channels after convolutions to determine GRU input size
        out_channels: int = self.calculate_channels(
            length=hparams.n_mel_channels,
            kernel_size=3,
            stride=2,
            pad=1,
            n_convs=k
        )
        self.gru: nn.GRU = nn.GRU(
            input_size=hparams.ref_enc_filters[-1] * out_channels,
            hidden_size=hparams.E // 2,  # Use half of the embedding dimension for GRU hidden size
            batch_first=True,
        )
        self.n_mels: int = hparams.n_mel_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Reference Encoder.

        Args:
            inputs (torch.Tensor): Mel-spectrogram input.
                                   Shape: (batch_size, n_mel_channels, time_steps)

        Returns:
            torch.Tensor: Encoded representation from the GRU's last hidden state.
                          Shape: (batch_size, E // 2)
        """
        batch_size: int = inputs.size(0)
        # Reshape input for convolutional layers: (batch_size, 1, time_steps, n_mels)
        out: torch.Tensor = inputs.contiguous().view(batch_size, 1, -1, self.n_mels)

        # Apply convolutional layers, batch normalization, and ReLU activation
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        # Transpose to prepare for GRU: (batch_size, time_steps', channels)
        out = out.transpose(1, 2)
        time_steps_prime: int = out.size(1)
        # Reshape for GRU: (batch_size, time_steps', feature_dim)
        out = out.contiguous().view(batch_size, time_steps_prime, -1)

        # Process with GRU
        _, out = self.gru(out)  # out contains the last hidden state: (1, batch_size, hidden_size)
        return out.squeeze(0)  # Return (batch_size, hidden_size)

    @staticmethod
    def calculate_channels(length: int, kernel_size: int, stride: int, pad: int, n_convs: int) -> int:
        """
        Calculates the output length after a series of convolutions.

        Args:
            length (int): Initial length (e.g., number of mel channels or time steps).
            kernel_size (int): Kernel size of the convolution.
            stride (int): Stride of the convolution.
            pad (int): Padding of the convolution.
            n_convs (int): Number of convolutional layers.

        Returns:
            int: The output length after applying the convolutions.
        """
        for _ in range(n_convs):
            length = (length - kernel_size + 2 * pad) // stride + 1
        return length


class VAE_GST(nn.Module):
    """
    Variational Autoencoder for Global Style Token (GST) extraction.

    Encodes mel-spectrograms into a latent space, producing style embeddings,
    mean (mu), and log-variance (logvar) for the latent distribution.

    Args:
        hparams (Tacotron2VAEHparams): Hyperparameters object.

    Attributes:
        ref_encoder (ReferenceEncoder): The reference encoder network.
        fc1 (nn.Linear): Linear layer to compute mu from the encoder output.
        fc2 (nn.Linear): Linear layer to compute logvar from the encoder output.
        fc3 (nn.Linear): Linear layer to project the sampled latent variable (z) to the style embedding dimension.
    """

    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        self.ref_encoder: ReferenceEncoder = ReferenceEncoder(hparams)
        # Fully connected layers for mean and log-variance
        self.fc1: nn.Linear = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        self.fc2: nn.Linear = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        # Fully connected layer to map latent variable to style embedding
        self.fc3: nn.Linear = nn.Linear(hparams.z_latent_dim, hparams.E)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Logarithm of the variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent variable (z).
        """
        if self.training:
            std: torch.Tensor = torch.exp(0.5 * logvar)
            eps: torch.Tensor = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu  # Return mean during inference

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE_GST module.

        Args:
            inputs (torch.Tensor): Mel-spectrogram input.
                                   Shape: (batch_size, n_mel_channels, time_steps)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - style_embed (torch.Tensor): The extracted style embedding. Shape: (batch_size, E)
                - mu (torch.Tensor): Mean of the latent distribution. Shape: (batch_size, z_latent_dim)
                - logvar (torch.Tensor): Log-variance of the latent distribution. Shape: (batch_size, z_latent_dim)
                - z (torch.Tensor): Sampled latent variable. Shape: (batch_size, z_latent_dim)
        """
        # Encode the input mel-spectrogram
        enc_out: torch.Tensor = self.ref_encoder(inputs)
        # Compute mu and logvar
        mu: torch.Tensor = self.fc1(enc_out)
        logvar: torch.Tensor = self.fc2(enc_out)
        # Reparameterize to get the latent variable z
        z: torch.Tensor = self.reparameterize(mu, logvar)
        # Project z to the style embedding dimension
        style_embed: torch.Tensor = self.fc3(z)
        return style_embed, mu, logvar, z
