"""
Dataset loaders for Portuguese prosody and style transfer.

Three datasets are supported:

1. **TTS-Portuguese Corpus** — Neutral read speech for base mapping training.
   Expected directory layout::

       <root>/
           metadata.csv          # columns: file_id, text
           wavs/
               <file_id>.wav

2. **LibriVox PT-BR** — Portuguese audiobook recordings for unsupervised
   prosody learning.  Expected layout::

       <root>/
           <chapter_id>/
               <utterance_id>.flac   (or .wav)

3. **VERBO** (Voice Emotion Recognition in Brazilian Portuguese) — Acted
   emotional speech for style extraction during inference.
   Expected layout::

       <root>/
           <speaker_id>/
               <emotion>/
                   <file>.wav

All datasets return a common sample dictionary so that they can be used
interchangeably with the :class:`ProsodyTransferDataset` wrapper.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessing import AudioPreprocessor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def _find_audio_files(root: str | Path) -> list[Path]:
    """Recursively find all audio files under *root*."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _AUDIO_EXTS)


# ---------------------------------------------------------------------------
# TTS-Portuguese Corpus
# ---------------------------------------------------------------------------

class TTSPortugueseDataset(Dataset):
    """Dataset wrapping the TTS-Portuguese Corpus.

    Args:
        root: Root directory of the corpus (contains ``metadata.csv`` and
            ``wavs/``).
        preprocessor: :class:`~preprocessing.AudioPreprocessor` instance.
        max_duration_s: Discard utterances longer than this many seconds.
        transform: Optional callable applied to each sample dict.
    """

    def __init__(
        self,
        root: str | Path,
        preprocessor: AudioPreprocessor | None = None,
        max_duration_s: float = 10.0,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.max_duration_s = max_duration_s
        self.transform = transform
        self.samples = self._load_metadata()

    def _load_metadata(self) -> list[dict]:
        metadata_path = self.root / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {self.root}")
        samples = []
        with open(metadata_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("file_id") or row.get("id") or list(row.values())[0]
                text = row.get("text") or row.get("transcript") or ""
                wav_path = self.root / "wavs" / f"{file_id}.wav"
                if wav_path.exists():
                    samples.append({"path": str(wav_path), "text": text, "id": file_id})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]
        features = self.preprocessor.process(meta["path"])
        sample = {
            "id": meta["id"],
            "text": meta["text"],
            "waveform": features["waveform"],
            "mel": features["mel"],
            "pitch": features["pitch"],
            "energy": features["energy"],
            "dataset": "tts_portuguese",
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# LibriVox PT-BR
# ---------------------------------------------------------------------------

class LibriVoxPTBRDataset(Dataset):
    """Dataset wrapping Portuguese LibriVox audiobook recordings.

    Recursively searches *root* for all audio files.  No metadata file is
    required — the file hierarchy itself provides the structure.

    Args:
        root: Root directory of the LibriVox PT-BR corpus.
        preprocessor: :class:`~preprocessing.AudioPreprocessor` instance.
        max_duration_s: Discard utterances longer than this many seconds.
        transform: Optional callable applied to each sample dict.
    """

    def __init__(
        self,
        root: str | Path,
        preprocessor: AudioPreprocessor | None = None,
        max_duration_s: float = 30.0,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.max_duration_s = max_duration_s
        self.transform = transform
        self.files = _find_audio_files(root)
        if not self.files:
            raise FileNotFoundError(f"No audio files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        features = self.preprocessor.process(str(path))
        sample = {
            "id": path.stem,
            "text": "",
            "waveform": features["waveform"],
            "mel": features["mel"],
            "pitch": features["pitch"],
            "energy": features["energy"],
            "dataset": "librivox_ptbr",
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# VERBO
# ---------------------------------------------------------------------------

# Standard emotion labels in the VERBO dataset
VERBO_EMOTIONS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
)


class VERBODataset(Dataset):
    """Dataset wrapping the VERBO (Voice Emotion Recognition in Brazilian
    Portuguese) corpus.

    Expected directory layout::

        <root>/
            <speaker_id>/
                <emotion>/
                    <file>.wav

    Args:
        root: Root directory of the VERBO corpus.
        emotions: Subset of emotions to include (``None`` means all).
        preprocessor: :class:`~preprocessing.AudioPreprocessor` instance.
        transform: Optional callable applied to each sample dict.
    """

    def __init__(
        self,
        root: str | Path,
        emotions: list[str] | None = None,
        preprocessor: AudioPreprocessor | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.transform = transform
        self.allowed_emotions = set(emotions) if emotions else set(VERBO_EMOTIONS)
        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No matching samples found under {root} for emotions {self.allowed_emotions}"
            )

    def _discover_samples(self) -> list[dict]:
        samples = []
        for speaker_dir in sorted(self.root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            for emotion_dir in sorted(speaker_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                emotion = emotion_dir.name.lower()
                if emotion not in self.allowed_emotions:
                    continue
                for audio_file in sorted(emotion_dir.iterdir()):
                    if audio_file.suffix.lower() in _AUDIO_EXTS:
                        samples.append(
                            {
                                "path": str(audio_file),
                                "speaker_id": speaker_id,
                                "emotion": emotion,
                                "id": audio_file.stem,
                            }
                        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]
        features = self.preprocessor.process(meta["path"])
        sample = {
            "id": meta["id"],
            "text": "",
            "speaker_id": meta["speaker_id"],
            "emotion": meta["emotion"],
            "waveform": features["waveform"],
            "mel": features["mel"],
            "pitch": features["pitch"],
            "energy": features["energy"],
            "dataset": "verbo",
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Collate a list of samples into a batched dictionary.

    Sequences are padded along the time dimension with zeros.

    Args:
        batch: List of sample dictionaries as returned by the dataset classes.

    Returns:
        Batched dictionary.
    """
    result: dict = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            max_len = max(v.shape[-1] for v in values)
            padded = []
            for v in values:
                pad_amount = max_len - v.shape[-1]
                if pad_amount > 0:
                    pad_shape = list(v.shape)
                    pad_shape[-1] = pad_amount
                    v = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype)], dim=-1)
                padded.append(v)
            result[key] = torch.stack(padded, dim=0)
        else:
            result[key] = values
    return result


# ---------------------------------------------------------------------------
# Combined dataset for prosody transfer training
# ---------------------------------------------------------------------------

class ProsodyTransferDataset(Dataset):
    """Wraps multiple source datasets and returns (source, reference) pairs.

    During training the model receives:
    - A *source* sample (content to be synthesised).
    - A *reference* sample (style / prosody target) drawn from the VERBO
      dataset or any other style-rich corpus.

    Args:
        content_dataset: Dataset providing content utterances.
        reference_dataset: Dataset providing reference (style) utterances.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        content_dataset: Dataset,
        reference_dataset: Dataset,
        seed: int = 42,
    ) -> None:
        self.content = content_dataset
        self.reference = reference_dataset
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.content)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        source = self.content[idx]
        ref_idx = self.rng.randrange(len(self.reference))  # type: ignore[arg-type]
        reference = self.reference[ref_idx]

        return {
            "source_waveform": source["waveform"],
            "source_mel": source["mel"],
            "source_pitch": source["pitch"],
            "source_energy": source["energy"],
            "source_id": source["id"],
            "ref_mel": reference["mel"],
            "ref_id": reference["id"],
            "ref_emotion": reference.get("emotion", ""),
        }
