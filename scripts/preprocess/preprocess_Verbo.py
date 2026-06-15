"""
Preprocess Verbo raw data into Tacotron2-VAE compatible mel-spectrogram tensors.

This script scans the VERBO dataset, computes mel-spectrograms for each audio file,
and saves them as PyTorch tensors (.pt).

Responsibilities:
    - Scan raw audio files in the VERBO dataset.
    - Load audio and compute mel-spectrograms using TacotronSTFT.
    - Save processed mel-spectrograms to the specified output directory.

Main Classes:
    - MelSpectrogramProcessor: Wrapper for TacotronSTFT to compute mel-spectrograms.

Main Functions:
    - get_args: Parse command-line arguments and initialize the mel processor.
    - find_audio_files: Recursively find all .wav files in a directory.
    - process_audio_file: Load a single audio file, compute its mel-spectrogram, and save it.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    n_mels = mel channels
    S = number of audio samples
"""

from pathlib import Path
import argparse
import sys
import torch
from torch import Tensor
import csv
import torchaudio
from typing import List, Tuple

# Ensure the correct path is set for local imports
ROOT_DIR: Path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

try:
    from layers import TacotronSTFT
except ImportError:
    # Fallback for different directory structures if needed
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from models.tacotron2_vae.layers import TacotronSTFT


class MelSpectrogramProcessor:
    """
    Wrapper for TacotronSTFT to compute mel-spectrograms from raw audio.

    Architecture:
        Uses TacotronSTFT (STFT -> Mel Filterbank -> Log Scaling).

    Inputs:
        audio:
            Shape (B, S) or (S,)

    Outputs:
        mel_spec:
            Shape (B, n_mels, T)

    Example:
        >>> processor = MelSpectrogramProcessor()
        >>> audio = torch.randn(1, 22050)
        >>> mel = processor(audio)
    """

    def __init__(
        self,
        sampling_rate: int = 22050,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ) -> None:
        """
        Initialize the MelSpectrogramProcessor.

        Args:
            sampling_rate (int): Audio sampling rate. Defaults to 22050.
            filter_length (int): FFT filter length. Defaults to 1024.
            hop_length (int): Hop length for STFT. Defaults to 256.
            win_length (int): Window length for STFT. Defaults to 1024.
            n_mel_channels (int): Number of mel frequency bins. Defaults to 80.
            mel_fmin (float): Minimum frequency for mel filter bank. Defaults to 0.0.
            mel_fmax (float): Maximum frequency for mel filter bank. Defaults to 8000.0.
        """
        self.stft: TacotronSTFT = TacotronSTFT(
            sampling_rate=sampling_rate,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )

    def __call__(self, audio: Tensor) -> Tensor:
        """
        Compute mel-spectrogram from audio.

        Args:
            audio (Tensor): Audio waveform.
                Shape: (B, S) or (S,)

        Returns:
            Tensor: Mel-spectrogram.
                Shape: (B, n_mels, T)
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, S)
        
        mel_spec: Tensor = self.stft.mel_spectrogram(audio)  # (B, n_mels, T)
        return mel_spec


def get_args() -> Tuple[argparse.Namespace, MelSpectrogramProcessor]:
    """
    Parse command-line arguments and initialize the MelSpectrogramProcessor.

    Returns:
        Tuple[argparse.Namespace, MelSpectrogramProcessor]: Parsed arguments and initialized processor.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess Verbo raw data into FastPitch-compatible mel-spectrogram tensors."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        help="Root directory of the raw dataset",
        default=Path("./data/raw/VERBO-Dataset"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        help="Root directory to save the preprocessed dataset",
        default=Path("./data/preprocessed/VERBO-Dataset"),
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=8000.0)

    args: argparse.Namespace = parser.parse_args()

    mel_processor: MelSpectrogramProcessor = MelSpectrogramProcessor(
        sampling_rate=args.sample_rate,
        filter_length=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.n_mels,
        mel_fmin=args.fmin,
        mel_fmax=args.fmax,
    )

    return args, mel_processor


def find_audio_files(root: Path) -> List[Path]:
    """
    Recursively find all .wav files in a directory.

    Args:
        root (Path): Root directory to search.

    Returns:
        List[Path]: List of paths to audio files.
    """
    audio_files: List[Path] = list(Path(root / "Audios").rglob("*.wav"))
    return audio_files


def process_audio_file(
    audio_path: Path, out_root: Path, mel_processor: MelSpectrogramProcessor
) -> None:
    """
    Load a single audio file, compute its mel-spectrogram, and save it.

    Args:
        audio_path (Path): Path to the audio file.
        out_root (Path): Root directory to save the output tensor.
        mel_processor (MelSpectrogramProcessor): Initialized mel processor.
    """
    audio_path = audio_path.resolve()
    audio_id: str = str(audio_path)
    out_path: Path = (out_root / "mels").resolve()

    audio: Tensor
    sr: int
    audio, sr = torchaudio.load(audio_path)  # audio: (channels, S)

    # Ensure audio is mono by averaging channels if necessary
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # (1, S)

    mel: Tensor = mel_processor(audio)  # (1, n_mels, T)

    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(mel, out_path / f"{Path(audio_id).stem}.pt")


if __name__ == "__main__":
    args, mel_processor = get_args()
    # Find audio files in the specified input root
    audio_files: List[Path] = find_audio_files(args.input_root)
    
    if audio_files:
        # For demonstration/Verbo specific script, processing the first file
        # or it could be a loop over all files.
        process_audio_file(audio_files[0], args.out_root, mel_processor)
    else:
        print(f"No audio files found in {args.input_root}")
