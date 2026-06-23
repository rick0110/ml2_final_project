"""
Data loading utilities for Tacotron 2 VAE.

Responsibilities:
    - Implement TextNormalizerEN: Normalize English text for LibriSpeech.
    - Implement DatasetLibriSpeechTacotronVAE: Custom Dataset for loading processed LibriSpeech tensors.
    - Handle on-the-fly mel-spectrogram computation from waveforms.
    - Manage dataset splitting into train, test, and validation sets.

Main Classes:
    - TextNormalizerEN: Normalizer for English transcripts.
    - DatasetLibriSpeechTacotronVAE: PyTorch Dataset for Tacotron2-VAE.

Main Functions:
    - load_data: Factory function to create and split datasets.

Tensor Conventions:
    B = batch size
    T = sequence length (frames/tokens)
    n_mels = mel channels
    S = number of audio samples
"""
import csv
import re
import sys
import torch
import torchaudio
from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset, random_split, Subset
from typing import List, Tuple, Dict, Any, Optional

from num2words import num2words

ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

try:
    from layers import TacotronSTFT
except ImportError:
    # Handle local absolute paths
    from src.models.tacotron2_vae.layers import TacotronSTFT

MAX_WAV_VALUE: float = 32768.0





class DatasetLibriSpeechTacotronVAE(Dataset):
    """
    Dataset for LibriSpeech processed for Tacotron 2 VAE.

    Architecture:
        Loads .pt files -> Computes Mel-spectrogram -> Returns (text_id, mel, emotion).

    Inputs:
        text_processor: Utility to convert text to IDs.
        data_dir: Path to processed files.

    Returns:
        tuple: (text_sequence, mel_tensor, emotion_vector)
            text_sequence: (T_text,)
            mel_tensor: (n_mels, T_mel)
            emotion_vector: (4,)
    """
    def __init__(
        self,
        text_processor: Any,
        data_dir: Path = Path("data/processed/tts-portuguese-Corpora"),
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            text_processor (Any): Text to sequence converter.
            data_dir (Path): Root directory for data.
            cache_dir (Optional[Path]): Cache directory path. Defaults to data_dir / "_cache".
        """
        self.data_dir: Path = Path(data_dir)
        if cache_dir is None:
            self.cache_dir: Path = self.data_dir / "_cache"
        else:
            self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path: Path = self.data_dir / "mels_metadata.csv"
        self.text_processor: Any = text_processor 
        self.files: List[Dict[str, str]] = self._load_files_list()
        self.stft: TacotronSTFT = TacotronSTFT(
            filter_length=1024,
            hop_length=256,
            win_length=1024, 
            sampling_rate=22050, 
            mel_fmin=0.0, 
            mel_fmax=8000.0
        )

    def _load_files_list(self) -> List[Dict[str, str]]:
        """Load metadata manifest."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        """int: Number of examples."""
        return len(self.files)

    def get_mel(self, audio: Tensor, orig_freq: Optional[int] = None) -> Tensor:
        """
        Compute mel-spectrogram from audio tensor.

        Args:
            audio (Tensor): Waveform (S,).
            orig_freq (Optional[int]): Original sampling rate. If provided and different
                                      from self.stft.sampling_rate, the audio is resampled.

        Returns:
            Tensor: Mel-spectrogram (n_mels, T).
        """
        if orig_freq is not None and orig_freq != self.stft.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=self.stft.sampling_rate)
        
        # Clamp audio to [-1.0, 1.0] to prevent STFT AssertionErrors due to resampling overshoots
        audio = torch.clamp(audio, -1.0, 1.0)

        audio_norm: Tensor = audio.unsqueeze(0) # (1, S)
        melspec: Tensor = self.stft.mel_spectrogram(audio_norm) # (1, n_mels, T)
        return melspec.squeeze(0) # (n_mels, T)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get an example.

        Args:
            idx (int): Index.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: text, mel, emotion.
        """
        row: Dict[str, str] = self.files[idx]
        utt_id: str = row.get("utt_id", Path(row["mel_path"]).stem)
        cache_file: Path = self.cache_dir / f"{utt_id}.pt"

        sequence_list: List[int] = self.text_processor.text_to_sequence(row["text"])

        text_sequence: Tensor = torch.LongTensor(sequence_list) # (T_text,)

        emotion: Tensor = torch.zeros(4, dtype=torch.float32) # (4,)
        emotion[0] = 1.0  # Neutral

        if cache_file.exists():
            mel_tensor: Tensor = torch.load(cache_file, map_location="cpu", weights_only=False)
        else:
            sample: Dict[str, Any] = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
            audio: Tensor = sample["waveform"].squeeze(0) # (S,)
            sr: int = sample.get("sr", 16000)
            mel_tensor = self.get_mel(audio, orig_freq=sr)
            torch.save(mel_tensor, cache_file)

        return text_sequence, mel_tensor, emotion

    def get_audio_mel(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Utility for inspection."""
        row: Dict[str, str] = self.files[idx]
        utt_id: str = row.get("utt_id", Path(row["mel_path"]).stem)
        cache_file: Path = self.cache_dir / f"{utt_id}.pt"

        sample: Dict[str, Any] = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
        audio: Tensor = sample["waveform"].squeeze(0)
        sr: int = sample.get("sr", 16000)
        if sr != self.stft.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.stft.sampling_rate)

        if cache_file.exists():
            mel_tensor: Tensor = torch.load(cache_file, map_location="cpu", weights_only=False)
        else:
            mel_tensor = self.get_mel(audio)
            torch.save(mel_tensor, cache_file)

        return audio, mel_tensor


def load_data(
    text_processor: Any,
    data_dir: Path = Path("data/processed/tts-portuguese-Corpora"),
    val_split: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Subset, Subset, Subset]:
    """
    Load and split the dataset.

    Args:
        text_processor (Any): Text processor.
        data_dir (Path): Data directory.
        val_split (float): Split percentage.
        generator (Optional[Generator]): Random generator.

    Returns:
        Tuple: train, test, val subsets.
    """
    dataset: DatasetLibriSpeechTacotronVAE = DatasetLibriSpeechTacotronVAE(text_processor=text_processor, data_dir=data_dir)
    n_val: int = int(len(dataset) * val_split // 2)
    n_test: int = n_val
    n_train: int = len(dataset) - n_val - n_test

    data_train, data_test, data_val = random_split(dataset, [n_train, n_test, n_val], generator=generator)

    return data_train, data_test, data_val