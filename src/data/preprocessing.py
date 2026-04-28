"""
Audio preprocessing utilities.

Provides functions and a class for loading audio files and computing
acoustic features (mel-spectrograms, pitch, energy) used throughout the
training and inference pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa


def load_audio(
    path: str,
    target_sr: int = 22050,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample to *target_sr*.

    Args:
        path: Path to the audio file (wav, mp3, flac, …).
        target_sr: Target sample rate in Hz.
        mono: If ``True`` the audio is converted to mono.

    Returns:
        A tuple ``(waveform, sample_rate)`` where *waveform* has shape
        ``(1, T)`` and *sample_rate* equals *target_sr*.
    """
    waveform, sr = torchaudio.load(path)
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    norm: str = "slaney",
    mel_scale: str = "slaney",
) -> torch.Tensor:
    """Compute a log-mel spectrogram.

    Args:
        waveform: Audio waveform, shape ``(1, T)`` or ``(T,)``.
        sample_rate: Sample rate of the waveform.
        n_fft: FFT size.
        hop_length: STFT hop length.
        win_length: STFT window length.
        n_mels: Number of mel bins.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        norm: Mel filter normalisation (``"slaney"`` or ``None``).
        mel_scale: Mel scale (``"slaney"`` or ``"htk"``).

    Returns:
        Log-mel spectrogram, shape ``(n_mels, T_mel)``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=norm,
        mel_scale=mel_scale,
    )
    mel = mel_transform(waveform.squeeze(0))  # (n_mels, T_mel)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel


def extract_pitch(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int = 22050,
    hop_length: int = 256,
    f_min: float = 50.0,
    f_max: float = 600.0,
) -> torch.Tensor:
    """Extract fundamental frequency (F0) using librosa's pyin algorithm.

    Unvoiced frames are set to 0.0.

    Args:
        waveform: Audio waveform ``(T,)`` or ``(1, T)``.
        sample_rate: Sample rate in Hz.
        hop_length: Hop length in samples.
        f_min: Minimum F0 frequency.
        f_max: Maximum F0 frequency.

    Returns:
        Pitch contour tensor of shape ``(T_frames,)``.
    """
    if isinstance(waveform, torch.Tensor):
        wav_np = waveform.squeeze().cpu().numpy()
    else:
        wav_np = np.squeeze(waveform)

    f0, voiced_flag, _ = librosa.pyin(
        wav_np,
        fmin=f_min,
        fmax=f_max,
        sr=sample_rate,
        hop_length=hop_length,
        fill_na=0.0,
    )
    f0 = np.where(voiced_flag, f0, 0.0).astype(np.float32)
    return torch.from_numpy(f0)


def extract_energy(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """Compute frame-level energy as the L2 norm of the STFT magnitude.

    Args:
        waveform: Audio waveform ``(1, T)`` or ``(T,)``.
        n_fft: FFT size.
        hop_length: STFT hop length.
        win_length: STFT window length.

    Returns:
        Energy tensor of shape ``(T_frames,)``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=True,
    )
    magnitude = stft.abs()           # (n_fft//2+1, T_frames)
    energy = magnitude.norm(dim=0)   # (T_frames,)
    return energy


class AudioPreprocessor:
    """Convenience wrapper for common audio feature extraction.

    Args:
        sample_rate: Target sample rate.
        n_mels: Mel-spectrogram bins.
        n_fft: FFT size.
        hop_length: STFT hop length.
        win_length: STFT window length.
        f_min: Minimum frequency for mel / pitch.
        f_max_mel: Maximum frequency for mel filterbank.
        f_min_pitch: Minimum F0 for pitch extraction.
        f_max_pitch: Maximum F0 for pitch extraction.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max_mel: float = 8000.0,
        f_min_pitch: float = 50.0,
        f_max_pitch: float = 600.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max_mel = f_max_mel
        self.f_min_pitch = f_min_pitch
        self.f_max_pitch = f_max_pitch

    def load(self, path: str) -> tuple[torch.Tensor, int]:
        """Load and resample an audio file."""
        return load_audio(path, target_sr=self.sample_rate)

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram from waveform."""
        return extract_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max_mel,
        )

    def pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract pitch (F0) contour from waveform."""
        return extract_pitch(
            waveform,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            f_min=self.f_min_pitch,
            f_max=self.f_max_pitch,
        )

    def energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract frame-level energy from waveform."""
        return extract_energy(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def process(self, path: str) -> dict[str, torch.Tensor]:
        """Load an audio file and compute all acoustic features.

        Args:
            path: Path to an audio file.

        Returns:
            Dictionary with keys ``"waveform"``, ``"mel"``, ``"pitch"``,
            ``"energy"``.
        """
        waveform, _ = self.load(path)
        return {
            "waveform": waveform.squeeze(0),
            "mel": self.mel(waveform),
            "pitch": self.pitch(waveform),
            "energy": self.energy(waveform),
        }
