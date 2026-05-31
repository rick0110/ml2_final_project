"""Shared LibriSpeech-EN data loaders for the last-model training phases."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "libriSpeech-en"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "libriSpeech-en"
DEFAULT_MANIFEST_NAME = "librispeech_mels_metadata.csv"


class LibriSpeechEnDataset(Dataset):
    """Dataset backed by the LibriSpeech-EN processed manifest."""

    def __init__(self, processed_root: Path = DEFAULT_PROCESSED_ROOT, manifest_name: str = DEFAULT_MANIFEST_NAME) -> None:
        self.processed_root = processed_root
        self.manifest_path = self._resolve_manifest_path(processed_root, manifest_name)
        self.samples = self._load_metadata(self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.samples[index]
        payload = torch.load(row["mel_path"], map_location="cpu")
        mel = payload.get("mel")
        if mel is None:
            raise KeyError(f"Missing 'mel' tensor in {row['mel_path']}")

        waveform = payload.get("waveform")
        sr = payload.get("sr")

        return {
            "mel": mel,
            "waveform": waveform,
            "sr": int(sr) if sr is not None else None,
            "duration": float(row.get("duration", 0.0)),
            "text": str(row.get("text", "")),
            "utt_id": str(row.get("utt_id", Path(row["mel_path"]).stem)),
            "mel_path": str(row["mel_path"]),
            "audio_path": str(payload.get("audio_path", row.get("audio_path", ""))),
            "source": "libriSpeech-en",
        }

    @staticmethod
    def _resolve_manifest_path(processed_root: Path, manifest_name: str) -> Path:
        candidates = [
            processed_root / manifest_name,
            processed_root / "mels" / manifest_name,
            processed_root.parent / manifest_name,
            processed_root.parent / "mels" / manifest_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _resolve_mel_path(csv_path: Path, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path

        candidates = [
            csv_path.parent / path,
            PROJECT_ROOT / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[1]

    @classmethod
    def _load_metadata(cls, csv_path: Path) -> List[Dict[str, object]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        samples: List[Dict[str, object]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                mel_path = row.get("mel_path")
                if not mel_path:
                    continue

                resolved = cls._resolve_mel_path(csv_path, mel_path)
                if not resolved.exists():
                    continue

                sample = dict(row)
                sample["mel_path"] = str(resolved)
                samples.append(sample)

        return samples


def build_librispeech_en_dataset(processed_root: Path = DEFAULT_PROCESSED_ROOT) -> LibriSpeechEnDataset:
    return LibriSpeechEnDataset(processed_root=processed_root)


def collate_last_model_batch(batch):
    mels: List[torch.Tensor] = []
    waveforms: List[torch.Tensor] = []
    mel_lengths: List[int] = []
    waveform_lengths: List[int] = []
    texts: List[str] = []
    durations: List[float] = []
    utt_ids: List[str] = []
    mel_paths: List[str] = []
    audio_paths: List[str] = []
    sources: List[str] = []
    sample_rates: List[int] = []

    for sample in batch:
        mel = sample["mel"]
        if not isinstance(mel, torch.Tensor):
            mel = torch.as_tensor(mel)
        mel = mel.detach().clone().to(dtype=torch.float32).contiguous()
        if mel.dim() == 3 and mel.size(0) == 1:
            mel = mel.squeeze(0)
        if mel.dim() != 2:
            raise ValueError(f"Expected mel with 2 dims, got shape {tuple(mel.shape)}")

        waveform = sample.get("waveform")
        if waveform is not None:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.as_tensor(waveform)
            waveform = waveform.detach().clone().to(dtype=torch.float32).contiguous()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = torch.zeros(1, 1, dtype=torch.float32)

        mels.append(mel)
        waveforms.append(waveform)
        mel_lengths.append(mel.size(1))
        waveform_lengths.append(waveform.size(-1))
        texts.append(str(sample.get("text", "")))
        durations.append(float(sample.get("duration", 0.0)))
        utt_ids.append(str(sample.get("utt_id", "")))
        mel_paths.append(str(sample.get("mel_path", "")))
        audio_paths.append(str(sample.get("audio_path", "")))
        sources.append(str(sample.get("source", "libriSpeech-en")))
        sample_rates.append(int(sample.get("sr") or 22050))

    max_mel_len = max(mel_lengths)
    padded_mels = []
    for mel in mels:
        pad_time = max_mel_len - mel.size(1)
        if pad_time > 0:
            mel = F.pad(mel, (0, pad_time), mode="constant", value=0.0)
        padded_mels.append(mel)

    max_wave_len = max(waveform.size(-1) for waveform in waveforms)
    padded_waveforms = []
    for waveform in waveforms:
        pad_time = max_wave_len - waveform.size(-1)
        if pad_time > 0:
            waveform = F.pad(waveform, (0, pad_time), mode="constant", value=0.0)
        padded_waveforms.append(waveform)

    return {
        "mel": torch.stack(padded_mels, dim=0),
        "waveform": torch.stack(padded_waveforms, dim=0),
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
        "waveform_lengths": torch.tensor(waveform_lengths, dtype=torch.long),
        "sr": torch.tensor(sample_rates, dtype=torch.long),
        "text": texts,
        "duration": torch.tensor(durations, dtype=torch.float32),
        "utt_id": utt_ids,
        "mel_path": mel_paths,
        "audio_path": audio_paths,
        "source": sources,
    }


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int = 42,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> Tuple[DataLoader, DataLoader]:
    dataset = build_librispeech_en_dataset(processed_root=processed_root)
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
        collate_fn=collate_last_model_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_last_model_batch,
    )
    return train_loader, val_loader