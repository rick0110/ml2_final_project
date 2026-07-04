"""
Audio processing utilities for Tacotron 2 VAE.

Responsibilities:
    - Compute window sum-square envelopes for STFT overlap-add.
    - Implement dynamic range compression (log-mel scaling).
    - Implement dynamic range decompression (inverse log-mel scaling).

Main Functions:
    - window_sumsquare: Compute energy envelope of a sliding window.
    - dynamic_range_compression: Apply log scaling for mel-spectrograms.
    - dynamic_range_decompression: Invert log scaling.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    S = number of audio samples
"""
import numpy as np
import torch
from torch import Tensor
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from typing import Optional, Union, Any


def window_sumsquare(
    window: str,
    n_frames: int,
    hop_length: int = 200,
    win_length: Optional[int] = 800,
    n_fft: int = 800,
    dtype: type = np.float32,
    norm: Optional[Any] = None,
) -> np.ndarray:
    """
    Computes the sum-square envelope of a window function at a given hop length.
    
    This is used to estimate the energy or amplitude envelope of an STFT using
    the overlap-add method.
    
    Args:
        window (str): The name of the window function to use (e.g., 'hann', 'hamming').
        n_frames (int): Number of frames.
        hop_length (int): Hop length between frames.
        win_length (Optional[int]): Window length. If None, set to n_fft.
        n_fft (int): Size of the FFT.
        dtype (type): The data type of the output.
        norm (Optional[Any]): Normalization parameter for librosa's `normalize`.
        
    Returns:
        np.ndarray: The sum-square envelope. 
            Shape: (n_fft + hop_length * (n_frames - 1),)
        
    Example:
        >>> envelope = window_sumsquare('hann', n_frames=10, hop_length=100, win_length=400, n_fft=400)
    """
    if win_length is None:
        win_length = n_fft

    n: int = n_fft + hop_length * (n_frames - 1)
    x: np.ndarray = np.zeros(n, dtype=dtype)  # Shape: (n,)

    win_sq: np.ndarray = get_window(window, win_length, fftbins=True)  # Shape: (win_length,)
    win_sq = normalize(win_sq, norm=norm) ** 2  # Shape: (win_length,)
    win_sq = pad_center(win_sq, n_fft)  # Shape: (n_fft,)

    for i in range(n_frames):
        sample: int = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    
    return x  # Shape: (n,)


def dynamic_range_compression(x: Tensor, c: float = 1.0, clip_val: float = 1e-5) -> Tensor:
    """
    Applies dynamic range compression (log scaling) to a tensor.
    
    Mathematical Intuition:
        compressed = log(max(x, clip_val) * c)
    
    Args:
        x (Tensor): The input tensor (e.g., mel magnitudes).
        c (float): A scaling factor inside the logarithm.
        clip_val (float): The minimum value to clamp the input to.
        
    Returns:
        Tensor: The compressed tensor. 
            Shape: (...,) matches input.
        
    Example:
        >>> x = torch.tensor([0.0, 0.5, 1.0])
        >>> compressed = dynamic_range_compression(x)
    """
    compressed: Tensor = torch.log(torch.clamp(x, min=clip_val) * c)  # Shape: (...,) matches input
    return compressed


def dynamic_range_decompression(x: Tensor, c: float = 1.0) -> Tensor:
    """
    Applies dynamic range decompression (exponential scaling) to a tensor.
    
    Mathematical Intuition:
        decompressed = exp(x) / c
    
    Args:
        x (Tensor): The input compressed tensor.
        c (float): A scaling factor to divide the exponential by.
        
    Returns:
        Tensor: The decompressed tensor.
            Shape: (...,) matches input.
        
    Example:
        >>> x = torch.tensor([-11.51, -0.69, 0.0])
        >>> decompressed = dynamic_range_decompression(x)
    """
    decompressed: Tensor = torch.exp(x) / c  # Shape: (...,) matches input
    return decompressed
