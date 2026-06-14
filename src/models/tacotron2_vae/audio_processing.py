import numpy as np
import torch
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
        hop_length (int, optional): Hop length between frames. Defaults to 200.
        win_length (int, optional): Window length. Defaults to 800. If None, set to n_fft.
        n_fft (int, optional): Size of the FFT. Defaults to 800.
        dtype (type, optional): The data type of the output. Defaults to np.float32.
        norm (Optional[Any], optional): Normalization parameter for librosa's `normalize`. Defaults to None.
        
    Returns:
        np.ndarray: The sum-square envelope. Shape: (n_fft + hop_length * (n_frames - 1),)
        
    Example:
        >>> envelope = window_sumsquare('hann', n_frames=10, hop_length=100, win_length=400, n_fft=400)
        >>> envelope.shape
        (1300,)
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)  # [n]

    win_sq = get_window(window, win_length, fftbins=True)  # [win_length]
    win_sq = normalize(win_sq, norm=norm) ** 2  # [win_length]
    win_sq = pad_center(win_sq, n_fft)  # [n_fft]

    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x  # [n]


def dynamic_range_compression(x: torch.Tensor, c: float = 1, clip_val: float = 1e-5) -> torch.Tensor:
    """
    Applies dynamic range compression (log scaling) to a tensor.
    
    Args:
        x (torch.Tensor): The input tensor.
        c (float, optional): A scaling factor inside the logarithm. Defaults to 1.
        clip_val (float, optional): The minimum value to clamp the input to before applying log. Defaults to 1e-5.
        
    Returns:
        torch.Tensor: The compressed tensor. Shape matches the input shape.
        
    Example:
        >>> x = torch.tensor([0.0, 0.5, 1.0])
        >>> compressed = dynamic_range_compression(x)
    """
    return torch.log(torch.clamp(x, min=clip_val) * c)  # [shape matches input x]


def dynamic_range_decompression(x: torch.Tensor, c: float = 1) -> torch.Tensor:
    """
    Applies dynamic range decompression (exponential scaling) to a tensor.
    
    Args:
        x (torch.Tensor): The input tensor.
        c (float, optional): A scaling factor to divide the exponential by. Defaults to 1.
        
    Returns:
        torch.Tensor: The decompressed tensor. Shape matches the input shape.
        
    Example:
        >>> x = torch.tensor([-11.51, -0.69, 0.0])
        >>> decompressed = dynamic_range_decompression(x)
    """
    return torch.exp(x) / c  # [shape matches input x]
