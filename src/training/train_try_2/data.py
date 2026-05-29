"""Dataset and dataloader helpers for train_try_2."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.first_step_data_loaders.datasets import build_first_step_dataset


def collate_first_step_batch(batch):
    mels = []
    waveforms = []
    mel_lengths = []
    texts = []
    durations = []
    utt_ids = []
    mel_paths = []
    sources = []
    sample_rates = []

    for sample in batch:
        mel = sample["mel"]
        if not isinstance(mel, torch.Tensor):
            mel = torch.as_tensor(mel)
        mel = mel.detach().clone().to(dtype=torch.float32).contiguous()

        waveform = sample.get("waveform")
        if waveform is not None:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.as_tensor(waveform)
            waveform = waveform.detach().clone().to(dtype=torch.float32).contiguous()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = torch.zeros(1, 1, dtype=torch.float32)

        if mel.dim() == 3 and mel.size(0) == 1:
            mel = mel.squeeze(0)
        if mel.dim() != 2:
            raise ValueError(f"Expected mel with 2 dims, got shape {tuple(mel.shape)}")

        mels.append(mel)
        waveforms.append(waveform)
        mel_lengths.append(mel.size(1))
        texts.append(sample.get("text", ""))
        durations.append(float(sample.get("duration", 0.0)))
        utt_ids.append(str(sample.get("utt_id", "")))
        mel_paths.append(str(sample.get("mel_path", "")))
        sources.append(str(sample.get("source", "")))
        sample_rates.append(int(sample.get("sr") or 22050))

    max_time = max(mel_lengths)
    padded_mels = []
    for mel in mels:
        pad_time = max_time - mel.size(1)
        if pad_time > 0:
            mel = F.pad(mel, (0, pad_time), mode="constant", value=0.0)
        padded_mels.append(mel)

    max_wave_time = max(waveform.size(-1) for waveform in waveforms)
    padded_waveforms = []
    for waveform in waveforms:
        pad_time = max_wave_time - waveform.size(-1)
        if pad_time > 0:
            waveform = F.pad(waveform, (0, pad_time), mode="constant", value=0.0)
        padded_waveforms.append(waveform)

    return {
        "mel": torch.stack(padded_mels, dim=0),
        "waveform": torch.stack(padded_waveforms, dim=0),
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
        "sr": torch.tensor(sample_rates, dtype=torch.long),
        "text": texts,
        "duration": torch.tensor(durations, dtype=torch.float32),
        "utt_id": utt_ids,
        "mel_path": mel_paths,
        "source": sources,
    }


def create_dataloaders(batch_size: int, num_workers: int, val_split: float, seed: int = 42):
    dataset = build_first_step_dataset()
    if len(dataset) < 2:
        raise ValueError(f"Dataset must contain at least 2 samples, got {len(dataset)}")

    val_size = int(len(dataset) * val_split)
    if val_size <= 0 or val_size >= len(dataset):
        raise ValueError(
            f"Invalid val_split={val_split} for dataset of size {len(dataset)}. "
            "The split must produce at least one validation sample and one training sample."
        )

    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_first_step_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_first_step_batch,
    )
    return train_loader, val_loader
