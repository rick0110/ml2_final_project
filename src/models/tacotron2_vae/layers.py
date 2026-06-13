"""
Custom layers for Tacotron 2 VAE model.

This module defines reusable neural network layers:
- LinearNorm: A linear layer with Xavier uniform initialization.
- ConvNorm: A 1D convolutional layer with Xavier uniform initialization.
- TacotronSTFT: A module for computing mel-spectrograms using STFT.

Dependencies:
    - torch: PyTorch for neural network operations.
    - librosa: For mel-filter bank calculation.
    - pathlib: For path manipulation.
    - sys: For system path manipulation.
    - stft: Custom STFT module.
    - audio_processing: Utilities for dynamic range compression/decompression.

Typical Usage:
    >>> from src.models.tacotron2_vae.layers import LinearNorm, ConvNorm, TacotronSTFT
    >>> linear_layer = LinearNorm(100, 50)
    >>> conv_layer = ConvNorm(80, 512, kernel_size=5, padding=2)
    >>> stft_module = TacotronSTFT(filter_length=1024, hop_length=256)
"""
import torch
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
import sys
from typing import Optional, Tuple

# Ensure the correct path is set for local imports
ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
# This is a workaround to allow importing modules from the root 'src' directory
# It should ideally be handled by a proper package structure or installation
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT


class LinearNorm(torch.nn.Module):
    """
    Linear layer with Xavier uniform initialization.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        w_init_gain (str, optional): Gain for Xavier uniform initialization based on activation function.
                                     Defaults to "linear".

    Attributes:
        linear_layer (torch.nn.Linear): The underlying linear layer.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        self.linear_layer: torch.nn.Linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        # Initialize weights using Xavier uniform initialization
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.

        Args:
            x (torch.Tensor): Input tensor. Shape: (..., in_dim)

        Returns:
            torch.Tensor: Output tensor. Shape: (..., out_dim)
        """
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """
    1D Convolutional layer with Xavier uniform initialization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int | None, optional): Padding added to both sides of the input.
                                        If None, calculated automatically for same padding. Defaults to None.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        w_init_gain (str, optional): Gain for Xavier uniform initialization based on activation function.
                                     Defaults to "linear".

    Attributes:
        conv (torch.nn.Conv1d): The underlying 1D convolutional layer.
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
    ):
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

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 1D convolutional layer.

        Args:
            signal (torch.Tensor): Input tensor. Shape: (batch_size, in_channels, length)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, out_channels, new_length)
        """
        return self.conv(signal)


class TacotronSTFT(torch.nn.Module):
    """
    Mel-Spectrogram computation module using Short-Time Fourier Transform (STFT).

    This module computes the mel-spectrogram from raw audio waveforms. It utilizes
    a STFT implementation and a pre-computed mel-filter bank. It also includes
    methods for spectral normalization and de-normalization.

    Args:
        filter_length (int, optional): FFT filter length. Defaults to 1024.
        hop_length (int, optional): Hop length for STFT. Defaults to 256.
        win_length (int, optional): Window length for STFT. Defaults to 1024.
        n_mel_channels (int, optional): Number of mel frequency bins. Defaults to 80.
        sampling_rate (int, optional): Audio sampling rate. Defaults to 22050.
        mel_fmin (float, optional): Minimum frequency for mel filter bank. Defaults to 0.0.
        mel_fmax (float, optional): Maximum frequency for mel filter bank. Defaults to 8000.0.

    Attributes:
        n_mel_channels (int): Number of mel channels.
        sampling_rate (int): Sampling rate of the audio.
        stft_fn (STFT): STFT module instance.
        mel_basis (torch.Tensor): Pre-computed mel-filter bank weights.
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
    ):
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
        mel_basis: torch.Tensor = torch.from_numpy(mel_basis_np).float()
        # Register mel_basis as a buffer, so it's part of the model state but not trainable
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Applies dynamic range compression to magnitudes (log-mel scaling).

        Args:
            magnitudes (torch.Tensor): Magnitude spectrum.

        Returns:
            torch.Tensor: Compressed magnitude spectrum.
        """
        return dynamic_range_compression(magnitudes)

    def spectral_de_normalize(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Applies inverse dynamic range compression to magnitudes.

        Args:
            magnitudes (torch.Tensor): Compressed magnitude spectrum.

        Returns:
            torch.Tensor: Decompressed magnitude spectrum.
        """
        return dynamic_range_decompression(magnitudes)

    def mel_spectrogram(
        self, y: torch.Tensor, ref_level_db: float = 20, magnitude_power: float = 1.5
    ) -> torch.Tensor:
        """
        Computes the mel-spectrogram from an audio waveform.

        Args:
            y (torch.Tensor): Input audio waveform. Expected range [-1, 1].
                              Shape: (batch_size, num_samples)
            ref_level_db (float, optional): Reference level in dB for normalization. Defaults to 20.
            magnitude_power (float, optional): Power to raise magnitudes to before normalization. Defaults to 1.5.

        Returns:
            torch.Tensor: Computed mel-spectrogram. Shape: (batch_size, n_mel_channels, num_frames)
        """
        # Assertions for input waveform range
        assert torch.min(y.data) >= -1, "audio must be on the range [-1, 1]"
        assert torch.max(y.data) <= 1, "audio must be on the range [-1, 1]"

        # Compute STFT magnitudes and phases
        magnitudes, _ = self.stft_fn.transform(y)
        magnitudes = magnitudes.data  # Detach from computation graph if needed, though usually not necessary here.

        # Apply mel filter bank
        mel_output: torch.Tensor = torch.matmul(self.mel_basis, magnitudes)

        # Normalize the mel-spectrogram
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
