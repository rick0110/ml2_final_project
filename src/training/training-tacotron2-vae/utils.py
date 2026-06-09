from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.loader_TTS_GST.DataSet import DatasetTTSPortuguese
from text_processing import TextProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "tacotron2-vae"


class Tacotron2VAEDataset(Dataset):
    """Adapter over loader_TTS_GST for Tacotron2-VAE training."""

    def __init__(
        self,
        filelist_path: Path,
        text_processor: TextProcessor,
        n_speakers: int = 1,
        n_emotions: int = 4,
    ):
        self.text_processor = text_processor
        self.n_speakers = n_speakers
        self.n_emotions = n_emotions
        self.samples = self._load_filelist(filelist_path)

    @staticmethod
    def _load_filelist(path: Path) -> List[Dict[str, str]]:
        rows = []
        with open(path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        sample = torch.load(row["mel_path"], weights_only=False)
        mel = sample["mel"].squeeze(0)
        text = torch.LongTensor(self.text_processor.text_to_sequence(row["text"]))
        speaker = torch.zeros(self.n_speakers, dtype=torch.float32)
        speaker[0] = 1.0
        emotion = torch.zeros(self.n_emotions, dtype=torch.float32)
        emotion[0] = 1.0
        return text, mel, speaker, emotion


class TextMelCollate:
    def __init__(self, n_frames_per_step: int = 1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(item[0]) for item in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        speakers = torch.FloatTensor(len(batch), len(batch[0][2]))
        emotions = torch.FloatTensor(len(batch), len(batch[0][3]))
        for i in range(len(ids_sorted_decreasing)):
            speakers[i, :] = batch[ids_sorted_decreasing[i]][2]
            emotions[i, :] = batch[ids_sorted_decreasing[i]][3]

        num_mels = batch[0][1].size(0)
        max_target_len = max(x[1].size(1) for x in batch)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            speakers,
            emotions,
        )


def create_dataset(
    filelist_path: Path,
    text_processor: TextProcessor,
    n_speakers: int = 1,
    n_emotions: int = 4,
) -> Tacotron2VAEDataset:
    return Tacotron2VAEDataset(
        filelist_path=filelist_path,
        text_processor=text_processor,
        n_speakers=n_speakers,
        n_emotions=n_emotions,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: TextMelCollate,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=shuffle,
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: TextMelCollate,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = create_dataloader(
        train_dataset, batch_size, num_workers, collate_fn, shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, batch_size, num_workers, collate_fn, shuffle=False
    )
    return train_loader, val_loader


def create_experiment_dir(experiment_name: Optional[str] = None) -> Path:
    experiments_root = PROJECT_ROOT / "experiments" / "tacotron2-vae"
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = experiments_root / (experiment_name or f"attempt_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "tensorboard").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    return experiment_dir
