#!/usr/bin/env python3

"""Sanity-check the shared LibriSpeech-EN last-model data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from last_model_data import DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT, build_librispeech_en_dataset, create_dataloaders
from last_model_utils import summarize_processed_data, summarize_raw_data


def main() -> None:
    dataset = build_librispeech_en_dataset()
    train_loader, val_loader = create_dataloaders(batch_size=4, num_workers=0, val_split=0.1, seed=42)

    processed = summarize_processed_data(DEFAULT_PROCESSED_ROOT)
    raw = summarize_raw_data(DEFAULT_RAW_ROOT)

    print("LibriSpeech-EN last-model sanity check")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Train loader samples: {len(train_loader.dataset)}")
    print(f"Val loader samples: {len(val_loader.dataset)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    print(f"Processed manifest rows: {processed['manifest_rows']}")
    print(f"Processed mel tensors: {processed['mel_tensors']}")
    print(f"Processed figures: {processed['figures']}")
    print(f"Raw audio files: {raw['audio_files']}")
    print(f"Raw transcript files: {raw['transcripts']}")

    durations = [float(sample.get("duration", 0.0)) for sample in dataset.samples]
    if durations:
        total_duration = sum(durations)
        print(f"Total duration (sec): {total_duration:.2f}")
        print(f"Average duration (sec): {total_duration / len(durations):.2f}")
        print(f"Min duration (sec): {min(durations):.2f}")
        print(f"Max duration (sec): {max(durations):.2f}")

    first_batch = next(iter(train_loader), None)
    if first_batch is not None:
        print(f"First batch mel shape: {tuple(first_batch['mel'].shape)}")
        print(f"First batch waveform shape: {tuple(first_batch['waveform'].shape)}")


if __name__ == "__main__":
    main()