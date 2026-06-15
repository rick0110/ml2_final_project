"""
Data loading utilities for WaveGlow vocoder.

Responsibilities:
    - Implement Mel2Samp: Custom Dataset for loading audio and computing mel-spectrograms.
    - Provide utilities to load WAV files into PyTorch tensors.
    - Support random segment extraction from audio files for fixed-length batching.
    - Implement standalone script for batch mel-spectrogram generation.

Main Classes:
    - Mel2Samp: Dataset that returns (mel, audio) pairs.

Main Functions:
    - load_wav_to_torch: Load a WAV file into a float tensor.
    - files_to_list: Read filenames from a text file.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    n_mels = mel channels
    S = number of audio samples (segment_length)
"""
import os
import random
import argparse
import json
import torch
import torch.utils.data
from torch import Tensor
import sys
from scipy.io.wavfile import read
from pathlib import Path
import csv
from typing import List, Tuple, Dict, Any

# We're using the audio processing from TacoTron2 to make sure it matches
ROOT_DIR: Path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"/ "models" / "tacotron2_vae"))

try:
    from layers import TacotronSTFT
except ImportError:
    from src.models.tacotron2_vae.layers import TacotronSTFT

MAX_WAV_VALUE: float = 32768.0

def files_to_list(filename: str) -> List[str]:
    """
    Takes a text file of filenames and makes a list of filenames.

    Args:
        filename (str): Path to the text file.

    Returns:
        List[str]: List of stripped filenames.
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path: str) -> Tuple[Tensor, int]:
    """
    Loads wavdata into torch array.

    Args:
        full_path (str): Path to WAV file.

    Returns:
        Tuple[Tensor, int]: Audio tensor and sampling rate.
    """
    sampling_rate: int
    data: Any
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    Dataset class that returns mel-spectrogram and corresponding audio segments.

    Architecture:
        Loads audio -> Extracts random segment -> Computes Mel-spectrogram.

    Inputs:
        training_files: CSV manifest path.
        segment_length: Audio samples per segment.
        filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax: STFT params.

    Returns:
        tuple: (mel, audio)
            mel: (n_mels, T_mel)
            audio: (segment_length,)
    """
    def __init__(
        self, 
        training_files: str, 
        segment_length: int, 
        filter_length: int,
        hop_length: int, 
        win_length: int, 
        sampling_rate: int, 
        mel_fmin: float, 
        mel_fmax: float
    ) -> None:
        """
        Initialize Mel2Samp.

        Args:
            training_files (str): Path to manifest.
            segment_length (int): Fixed length of audio to return.
            filter_length (int): FFT size.
            hop_length (int): Hop size.
            win_length (int): Window size.
            sampling_rate (int): SR.
            mel_fmin (float): Min freq.
            mel_fmax (float): Max freq.
        """
        self.files: List[Dict[str, str]] = self._load_files_list(str(ROOT_DIR / str(training_files)))
        random.seed(1234)
        random.shuffle(self.files)
        self.stft: TacotronSTFT = TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin, 
            mel_fmax=mel_fmax
        )
        self.segment_length: int = segment_length
        self.sampling_rate: int = sampling_rate

    def get_mel(self, audio: Tensor) -> Tensor:
        """
        Compute mel-spectrogram from audio.

        Args:
            audio (Tensor): Waveform.

        Returns:
            Tensor: Mel-spectrogram.
        """
        audio_norm: Tensor = audio.unsqueeze(0) # (1, S)
        melspec: Tensor = self.stft.mel_spectrogram(audio_norm) # (1, n_mels, T)
        return melspec.squeeze(0) # (n_mels, T)

    def _load_files_list(self, files_path: str) -> List[Dict[str, str]]:
        """Load manifest."""
        if not os.path.isfile(files_path):
            raise FileNotFoundError(f"File list not found: {files_path}")
        with open(files_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a (mel, audio) segment.

        Args:
            index (int): Index.

        Returns:
            Tuple[Tensor, Tensor]: Mel-spectrogram and audio.
        """
        row: Dict[str, str] = self.files[index]
        path: str = row["mel_path"]
        data: Dict[str, Any] = torch.load(path, map_location="cpu")
        audio: Tensor = data["waveform"].squeeze() # (S_full,)

        if audio.size(0) >= self.segment_length:
            max_audio_start: int = audio.size(0) - self.segment_length
            audio_start: int = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length] # (S,)
        else: 
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant') # (S,)

        mel: Tensor = self.get_mel(audio) # (n_mels, T)
        return (mel, audio)

    def __len__(self) -> int:
        """int: Number of files."""
        return len(self.files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    args: argparse.Namespace = parser.parse_args()

    with open(args.config) as f:
        data_str: str = f.read()
    data_config: Dict[str, Any] = json.loads(data_str)["data_config"]
    mel2samp: Mel2Samp = Mel2Samp(**data_config)

    filepaths: List[str] = files_to_list(args.filelist_path)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio_t, sr_val = load_wav_to_torch(filepath)
        melspectrogram: Tensor = mel2samp.get_mel(audio_t)
        filename: str = os.path.basename(filepath)
        new_filepath: str = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)