import numpy as np
import torch
import torch.nn.functional as F
from librosa.util import pad_center
from scipy.signal import get_window
from typing import Tuple, Optional

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

from audio_processing import window_sumsquare


class STFT(torch.nn.Module):
    """
    Short-Time Fourier Transform (STFT) layer.
    
    This module computes the STFT and its inverse using PyTorch's 1D convolution.
    
    Args:
        filter_length (int, optional): The length of the FFT window. Defaults to 800.
        hop_length (int, optional): The number of samples between successive frames. Defaults to 200.
        win_length (int, optional): The length of the window function. Defaults to 800.
        window (str, optional): The name of the window function to apply. Defaults to "hann".
        
    Example:
        >>> stft = STFT(filter_length=1024, hop_length=256, win_length=1024, window="hann")
        >>> audio = torch.randn(2, 22050)
        >>> magnitude, phase = stft.transform(audio)
        >>> magnitude.shape
        torch.Size([2, 513, 87])
        >>> reconstructed_audio = stft.inverse(magnitude, phase)
        >>> reconstructed_audio.shape
        torch.Size([2, 1, 22016])
    """
    def __init__(self, filter_length: int = 800, hop_length: int = 200, win_length: int = 800, window: Optional[str] = "hann"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the STFT magnitude and phase of an input audio signal.
        
        Args:
            input_data (torch.Tensor): The input audio signal. Shape: (batch_size, time)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the magnitude and phase spectra.
                Shapes are both (batch_size, filter_length // 2 + 1, frames)
        """
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)  # [batch_size, 1, time]
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )  # [batch_size, 1, 1, padded_time]
        input_data = input_data.squeeze(1)  # [batch_size, 1, padded_time]

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0,
        )  # [batch_size, filter_length + 2, frames]

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]  # [batch_size, filter_length // 2 + 1, frames]
        imag_part = forward_transform[:, cutoff:, :]  # [batch_size, filter_length // 2 + 1, frames]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)  # [batch_size, filter_length // 2 + 1, frames]
        phase = torch.atan2(imag_part, real_part)  # [batch_size, filter_length // 2 + 1, frames]
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse STFT to reconstruct the audio signal.
        
        Args:
            magnitude (torch.Tensor): The magnitude spectrum. Shape: (batch_size, filter_length // 2 + 1, frames)
            phase (torch.Tensor): The phase spectrum. Shape: (batch_size, filter_length // 2 + 1, frames)
            
        Returns:
            torch.Tensor: The reconstructed audio signal. Shape: (batch_size, 1, time)
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )  # [batch_size, filter_length + 2, frames]

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )  # [batch_size, 1, padded_time]

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > np.finfo(np.float32).tiny)[0])  # [num_nonzero_indices]
            window_sum = torch.from_numpy(window_sum)  # [time_frames]
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]  # [batch_size, 1, padded_time]
            inverse_transform *= float(self.filter_length) / self.hop_length  # [batch_size, 1, padded_time]

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]  # [batch_size, 1, trimmed_time]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2)]  # [batch_size, 1, time]
        return inverse_transform
