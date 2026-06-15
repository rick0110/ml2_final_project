"""
Custom layers for Tacotron 2 VAE model.

Responsibilities:
    - Implement LinearNorm: A linear layer with specialized initialization.
    - Implement ConvNorm: A 1D convolutional layer with specialized initialization.
    - Implement TacotronSTFT: A module for computing mel-spectrograms using STFT.

Main Classes:
    - LinearNorm: Linear layer with Xavier uniform initialization.
    - ConvNorm: 1D convolutional layer with Xavier uniform initialization.
    - TacotronSTFT: Mel-spectrogram computation module.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    n_mels = mel channels
    S = number of audio samples
    D = feature dimension
"""
import torch
from torch import Tensor
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np

# Ensure the correct path is set for local imports
ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

try:
    from audio_processing import dynamic_range_compression, dynamic_range_decompression
    from stft import STFT
except ImportError:
    # Fallback for different directory structures if needed
    from models.tacotron2_vae.audio_processing import dynamic_range_compression, dynamic_range_decompression
    from models.tacotron2_vae.stft import STFT


class LinearNorm(torch.nn.Module):
    """
    Linear layer with Xavier uniform initialization.

    Architecture:
        Linear -> Weight Init

    Inputs:
        x:
            Shape (..., in_dim)

    Outputs:
        output:
            Shape (..., out_dim)

    Example:
        >>> layer = LinearNorm(128, 256)
        >>> x = torch.randn(16, 128)
        >>> y = layer(x)
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear") -> None:
        """
        Initialize the LinearNorm layer.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            bias (bool): Whether to include a bias term.
            w_init_gain (str): Gain for Xavier uniform initialization.
        """
        super().__init__()
        self.linear_layer: torch.nn.Linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        # Initialize weights using Xavier uniform initialization
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor. Shape: (..., in_dim)

        Returns:
            Tensor: Output tensor. Shape: (..., out_dim)
        """
        output: Tensor = self.linear_layer(x)  # (..., out_dim)
        return output


class ConvNorm(torch.nn.Module):
    """
    1D Convolutional layer with Xavier uniform initialization.

    Architecture:
        Conv1d -> Weight Init

    Inputs:
        signal:
            Shape (B, in_channels, T)

    Outputs:
        output:
            Shape (B, out_channels, T_new)

    Example:
        >>> layer = ConvNorm(80, 512, kernel_size=5, padding=2)
        >>> x = torch.randn(16, 80, 100)
        >>> y = layer(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        """
        Initialize the ConvNorm layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (Optional[int]): Padding added to both sides.
            dilation (int): Spacing between kernel elements.
            bias (bool): Whether to include a bias term.
            w_init_gain (str): Gain for Xavier uniform initialization.
        """
        super().__init__()
        # Calculate padding automatically if not provided, assuming kernel_size is odd for 'same' padding
        if padding is None:
            assert kernel_size % 2 == 1, "kernel_size must be odd for automatic padding calculation"
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv: torch.nn.Conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        # Initialize weights using Xavier uniform initialization
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal: Tensor) -> Tensor:
        """
        Forward pass through the 1D convolutional layer.

        Args:
            signal (Tensor): Input tensor. Shape: (B, in_channels, T)

        Returns:
            Tensor: Output tensor. Shape: (B, out_channels, T_new)
        """
        output: Tensor = self.conv(signal)  # (B, out_channels, T_new)
        return output


class TacotronSTFT(torch.nn.Module):
    """
    Mel-Spectrogram computation module using Short-Time Fourier Transform (STFT).

    Architecture:
        STFT -> Magnitude -> Mel Basis Projection -> Spectral Normalization.

    Inputs:
        y:
            Shape (B, S)

    Outputs:
        mel_output:
            Shape (B, n_mels, T)

    Example:
        >>> stft = TacotronSTFT()
        >>> x = torch.randn(1, 22050)
        >>> mel = stft.mel_spectrogram(x)
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ) -> None:
        """
        Initialize the TacotronSTFT module.

        Args:
            filter_length (int): FFT filter length.
            hop_length (int): Hop length for STFT.
            win_length (int): Window length for STFT.
            n_mel_channels (int): Number of mel frequency bins.
            sampling_rate (int): Audio sampling rate.
            mel_fmin (float): Minimum frequency for mel filter bank.
            mel_fmax (float): Maximum frequency for mel filter bank.
        """
        super().__init__()
        self.n_mel_channels: int = n_mel_channels
        self.sampling_rate: int = sampling_rate
        # Initialize the STFT module
        self.stft_fn: STFT = STFT(filter_length, hop_length, win_length)

        # Compute the mel-filter bank using librosa
        mel_basis_np: np.ndarray = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis: Tensor = torch.from_numpy(mel_basis_np).float()
        # Register mel_basis as a buffer
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes: Tensor) -> Tensor:
        """
        Applies dynamic range compression to magnitudes (log-mel scaling).

        Args:
            magnitudes (Tensor): Magnitude spectrum.

        Returns:
            Tensor: Compressed magnitude spectrum. Shape matches input.
        """
        return dynamic_range_compression(magnitudes)

    def spectral_de_normalize(self, magnitudes: Tensor) -> Tensor:
        """
        Applies inverse dynamic range compression to magnitudes.

        Args:
            magnitudes (Tensor): Compressed magnitude spectrum.

        Returns:
            Tensor: Decompressed magnitude spectrum. Shape matches input.
        """
        return dynamic_range_decompression(magnitudes)

    def mel_spectrogram(
        self, y: Tensor, ref_level_db: float = 20, magnitude_power: float = 1.5
    ) -> Tensor:
        """
        Computes the mel-spectrogram from an audio waveform.

        Args:
            y (Tensor): Input audio waveform. Expected range [-1, 1].
                              Shape: (B, S)
            ref_level_db (float): Reference level in dB for normalization.
            magnitude_power (float): Power to raise magnitudes to.

        Returns:
            Tensor: Computed mel-spectrogram. Shape: (B, n_mels, T)
        """
        # Assertions for input waveform range
        assert torch.min(y.data) >= -1, "audio must be on the range [-1, 1]"
        assert torch.max(y.data) <= 1, "audio must be on the range [-1, 1]"

        # Compute STFT magnitudes and phases
        magnitudes: Tensor
        magnitudes, _ = self.stft_fn.transform(y)  # (B, filter_length/2 + 1, T)
        magnitudes = magnitudes.data

        # Apply mel filter bank
        mel_output: Tensor = torch.matmul(self.mel_basis, magnitudes)  # (B, n_mels, T)

        # Normalize the mel-spectrogram
        mel_output = self.spectral_normalize(mel_output)  # (B, n_mels, T)
        return mel_output