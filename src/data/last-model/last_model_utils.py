"""Utility helpers for the shared last-model LibriSpeech-EN data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "libriSpeech-en"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "libriSpeech-en"


def resolve_processed_root(processed_root: Path = DEFAULT_PROCESSED_ROOT) -> Path:
    if processed_root.name == "mels":
        return processed_root.parent
    return processed_root


def resolve_raw_root(raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    candidate = raw_root / "LibriSpeech"
    if candidate.exists():
        return candidate
    return raw_root


def count_files(root: Path, suffixes: Sequence[str] | None = None) -> int:
    if not root.exists():
        return 0

    suffix_set = tuple(suffixes) if suffixes is not None else None
    count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if suffix_set is not None and not path.name.endswith(suffix_set):
            continue
        count += 1
    return count


def count_manifest_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def summarize_processed_data(processed_root: Path = DEFAULT_PROCESSED_ROOT) -> Dict[str, int]:
    root = resolve_processed_root(processed_root)
    manifest_candidates = [
        root / "librispeech_mels_metadata.csv",
        root / "mels_metadata.csv",
    ]
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), manifest_candidates[0])
    mels_dir = root / "mels"

    return {
        "manifest_rows": count_manifest_rows(manifest_path),
        "mel_tensors": count_files(mels_dir, (".pt",)),
        "figures": count_files(root / "figures", (".png", ".jpg", ".jpeg")),
    }


def summarize_raw_data(raw_root: Path = DEFAULT_RAW_ROOT) -> Dict[str, int]:
    root = resolve_raw_root(raw_root)
    return {
        "audio_files": count_files(root, (".flac", ".wav")),
        "transcripts": count_files(root, (".trans.txt",)),
    }