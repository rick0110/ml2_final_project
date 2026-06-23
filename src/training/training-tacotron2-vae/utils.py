"""
Training utility functions for Tacotron 2 VAE.

Responsibilities:
    - Implement TextMelCollate: Collate function for batching text, mel, and emotion labels.
    - Provide factory function for creating DataLoaders.
    - Manage experiment directory structure and timestamping.

Main Classes:
    - TextMelCollate: Handles variable-length sequence padding for training batches.

Main Functions:
    - create_dataloader: Standardized DataLoader initialization.
    - create_experiment_dir: Setup folder structure for a new training run.

Tensor Conventions:
    B = batch size
    T = sequence length (frames/tokens)
    n_mels = mel channels
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR: Path = PROJECT_ROOT / "data" / "processed" / "tts-portuguese-Corpora"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments" / "tacotron2-vae"


class TextMelCollate:
    """
    Collate function for Tacotron 2 VAE training.

    Architecture:
        Sorts by text length -> Pads text -> Pads mel (multiple of frames_per_step) -> Sets gate targets.

    Inputs:
        batch: List of tuples from Dataset.

    Outputs:
        text_padded: (B, max_T_text)
        input_lengths: (B,)
        mel_padded: (B, n_mels, max_T_mel)
        gate_padded: (B, max_T_mel)
        output_lengths: (B,)
        emotions: (B, 4)

    Example:
        >>> collate_fn = TextMelCollate(n_frames_per_step=1)
        >>> loader = DataLoader(dataset, collate_fn=collate_fn)
    """
    def __init__(self, n_frames_per_step: int = 1) -> None:
        """
        Initialize the collate function.

        Args:
            n_frames_per_step (int): Multiple for mel padding.
        """
        self.n_frames_per_step: int = n_frames_per_step

    def __call__(self, batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        """
        Process a list of samples into a padded batch.

        Args:
            batch: list of (text, mel, emotion).

        Returns:
            Tuple of padded tensors.
        """
        # Sort by text length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(item[0]) for item in batch]),
            dim=0,
            descending=True,
        )
        max_input_len: int = input_lengths[0].item()

        # Pad text
        text_padded: Tensor = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text: Tensor = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Collect emotion vectors from each sample in the batch
        emotions: Tensor = torch.stack([batch[ids_sorted_decreasing[i]][2] for i in range(len(ids_sorted_decreasing))])  # (B, 4)

        # Mel padding calculations
        num_mels: int = batch[0][1].size(0)
        max_target_len: int = max(x[1].size(1) for x in batch)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        mel_padded: Tensor = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.fill_(-11.5129)  # Silence in log-mel domain (log(1e-5))
        gate_padded: Tensor = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths: Tensor = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel: Tensor = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            # Gate target is 1 at the end of the sequence
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            emotions,
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: TextMelCollate,
    shuffle: bool,
) -> DataLoader:
    """
    Initialize a DataLoader.

    Args:
        dataset (Dataset): Source data.
        batch_size (int): Samples per batch.
        num_workers (int): Parallel workers.
        collate_fn (TextMelCollate): Collate logic.
        shuffle (bool): Whether to shuffle.

    Returns:
        DataLoader: Initialized loader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=shuffle,
    )


def create_experiment_dir(experiment_name: Optional[str] = None) -> Path:
    """
    Setup directory structure for a new experiment run.

    Args:
        experiment_name (Optional[str]): Custom name or None for timestamped folder.

    Returns:
        Path: Experiment root directory.
    """
    experiments_root: Path = PROJECT_ROOT / "experiments" / "tacotron2-vae"
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir: Path = experiments_root / (experiment_name or f"attempt_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "tensorboard").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    return experiment_dir
