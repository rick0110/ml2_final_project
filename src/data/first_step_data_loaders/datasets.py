"""PyTorch datasets for the project's first-step processed data.

The processed data lives in two separate roots:
- data/processed/libriSpeech-pt/<split>/librispeech_mels_metadata.csv
- data/processed/tts_portuguese/mels_metadata.csv

This module exposes a dataset per source and a combined dataset that
concatenates them for joint training.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROCESSED_ROOT = _PROJECT_ROOT / "data" / "processed"

class _ProcessedMelDataset(Dataset):
	"""Base class for datasets backed by a metadata CSV and .pt files."""

	def __init__(self, csv_path: Path, source_name: str) -> None:
		self.csv_path = csv_path
		self.source_name = source_name
		self.samples = self._load_metadata(csv_path)

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
			"source": self.source_name,
		}

	@staticmethod
	def _resolve_mel_path(csv_path: Path, raw_path: str) -> Path:
		path = Path(raw_path)
		if path.is_absolute():
			return path

		# Try relative to CSV parent first
		candidates = [
			csv_path.parent / path,
			_PROJECT_ROOT / path,
		]
		for candidate in candidates:
			if candidate.exists():
				return candidate

		return candidates[1]  # Default to PROJECT_ROOT / path

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
				row = dict(row)
				row["mel_path"] = str(resolved)
				samples.append(row)

		return samples


class LibriSpeechPTDataset(_ProcessedMelDataset):
	"""Processed LibriSpeech-pt mel dataset for a specific split."""

	def __init__(self, split: str = "train", processed_root: Path = _DEFAULT_PROCESSED_ROOT) -> None:
		csv_path = processed_root / "libriSpeech-pt" / split / "librispeech_mels_metadata.csv"
		super().__init__(csv_path=csv_path, source_name=f"libriSpeech-pt:{split}")


class TTSPortugueseDataset(_ProcessedMelDataset):
	"""Processed TTS Portuguese mel dataset."""

	def __init__(self, processed_root: Path = _DEFAULT_PROCESSED_ROOT) -> None:
		csv_path = processed_root / "tts_portuguese" / "mels_metadata.csv"
		super().__init__(csv_path=csv_path, source_name="tts_portuguese")


class CombinedFirstStepDataset(Dataset):
	"""Concatenation of all first-step datasets."""

	def __init__(self, datasets: Sequence[Dataset]) -> None:
		self.datasets = list(datasets)
		if not self.datasets:
			raise ValueError("CombinedFirstStepDataset requires at least one dataset")
		self._concat = ConcatDataset(self.datasets)

	def __len__(self) -> int:
		return len(self._concat)

	def __getitem__(self, index: int):
		return self._concat[index]


def build_librispeech_dataset(
	split: str = "train",
	processed_root: Path = _DEFAULT_PROCESSED_ROOT,
) -> LibriSpeechPTDataset:
	return LibriSpeechPTDataset(split=split, processed_root=processed_root)


def build_tts_portuguese_dataset(processed_root: Path = _DEFAULT_PROCESSED_ROOT) -> TTSPortugueseDataset:
	return TTSPortugueseDataset(processed_root=processed_root)


def build_first_step_dataset(
	include_librispeech_splits: Sequence[str] = ("train", "test"),
	include_tts_portuguese: bool = True,
	processed_root: Path = _DEFAULT_PROCESSED_ROOT,
) -> CombinedFirstStepDataset:
	datasets: List[Dataset] = []
	for split in include_librispeech_splits:
		datasets.append(build_librispeech_dataset(split=split, processed_root=processed_root))
	if include_tts_portuguese:
		datasets.append(build_tts_portuguese_dataset(processed_root=processed_root))
	return CombinedFirstStepDataset(datasets)