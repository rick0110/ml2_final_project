"""
Short-Time Fourier Transform (STFT) module for PyTorch.

Responsibilities:
    - Compute the forward STFT of an audio signal.
    - Compute the inverse STFT (iSTFT) to reconstruct an audio signal.
    - Handle windowing and padding for perfect reconstruction (when applicable).

Main Classes:
    - STFT: Convolution-based STFT/iSTFT layer.

Tensor Conventions:
    B = batch size
    T = number of frames
    F = frequency bins (filter_length // 2 + 1)
    S = number of audio samples
"""
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from librosa.util import pad_center
from scipy.signal import get_window
from typing import Tuple, Optional

from pathlib import Path
import sys
ROOT_DIR: Path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

try:
    from audio_processing import window_sumsquare
except ImportError:
    # Fallback for different directory structures
    from models.tacotron2_vae.audio_processing import window_sumsquare


class STFT(torch.nn.Module):
    """
    Short-Time Fourier Transform (STFT) layer.
    
    This module computes the STFT and its inverse using PyTorch's 1D convolution
    and transpose convolution with a pre-computed Fourier basis.
    
    Architecture:
        Forward: Conv1d with Fourier kernels.
        Inverse: ConvTranspose1d with Pseudo-inverse Fourier kernels.
    
    Inputs:
        input_data:
            Shape (B, S)
            
    Outputs:
        magnitude:
            Shape (B, F, T)
        phase:
            Shape (B, F, T)
            
    Example:
        >>> stft = STFT(filter_length=1024, hop_length=256)
        >>> audio = torch.randn(2, 22050)
        >>> magnitude, phase = stft.transform(audio)
    """
    def __init__(
        self, 
        filter_length: int = 800, 
        hop_length: int = 200, 
        win_length: int = 800, 
        window: Optional[str] = "hann"
    ) -> None:
        """
        Initialize the STFT module.

        Args:
            filter_length (int): FFT size.
            hop_length (int): Number of samples between frames.
            win_length (int): Window size.
            window (Optional[str]): Window function name.
        """
        super().__init__()
        self.filter_length: int = filter_length
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.window: Optional[str] = window
        scale: float = self.filter_length / self.hop_length
        fourier_basis: np.ndarray = np.fft.fft(np.eye(self.filter_length))

        cutoff: int = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis: Tensor = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis: Tensor = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert filter_length >= win_length
            fft_window: np.ndarray = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window_t: Tensor = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window_t
            inverse_basis *= fft_window_t

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the STFT magnitude and phase of an input audio signal.
        
        Args:
            input_data (Tensor): Input audio waveform.
                Shape: (B, S)
            
        Returns:
            Tuple[Tensor, Tensor]: Magnitude and phase spectra.
                Each Shape: (B, F, T)
        """
        num_batches: int = input_data.size(0)
        num_samples: int = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)  # (B, 1, S)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )  # (B, 1, 1, S + filter_length)
        input_data = input_data.squeeze(1)  # (B, 1, S + filter_length)

        forward_transform: Tensor = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0,
        )  # (B, filter_length + 2, T)

        cutoff: int = int((self.filter_length / 2) + 1)
        real_part: Tensor = forward_transform[:, :cutoff, :]  # (B, F, T)
        imag_part: Tensor = forward_transform[:, cutoff:, :]  # (B, F, T)

        magnitude: Tensor = torch.sqrt(real_part**2 + imag_part**2)  # (B, F, T)
        phase: Tensor = torch.atan2(imag_part, real_part)             # (B, F, T)
        
        return magnitude, phase

    def inverse(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        """
        Computes the inverse STFT to reconstruct the audio signal.
        
        Args:
            magnitude (Tensor): Magnitude spectrum.
                Shape: (B, F, T)
            phase (Tensor): Phase spectrum.
                Shape: (B, F, T)
            
        Returns:
            Tensor: Reconstructed audio signal.
                Shape: (B, 1, S)
        """
        recombine_magnitude_phase: Tensor = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )  # (B, filter_length + 2, T)

        inverse_transform: Tensor = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )  # (B, 1, S_padded)

        if self.window is not None:
            window_sum: np.ndarray = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            approx_nonzero_indices: Tensor = torch.from_numpy(np.where(window_sum > np.finfo(np.float32).tiny)[0])
            window_sum_t: Tensor = torch.from_numpy(window_sum)
            
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum_t[approx_nonzero_indices] # (B, 1, S_padded)
            inverse_transform *= float(self.filter_length) / self.hop_length # (B, 1, S_padded)

        # Trim padding
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :] # (B, 1, S_trimmed)
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2)] # (B, 1, S)
        
        return inverse_transform
