#!/usr/bin/env python3
"""Prepare Tacotron2-VAE filelists and vocabulary from loader_TTS_GST corpus."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))

from data.loader_TTS_GST.DataSet import DatasetTTSPortuguese
from text_processing import TextProcessor, build_symbols_from_texts
from utils import ARTIFACTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tacotron2-VAE filelists and symbol table from loader_TTS_GST data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tts-portuguese-Corpora",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def write_filelist(rows: List[dict], path: Path) -> None:
    fieldnames = ["mel_path", "text", "duration", "speaker", "emotion"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset = DatasetTTSPortuguese(data_dir=args.data_dir)
    rows = []
    texts = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata_row = dataset.files[idx]
        text = sample["text"]
        texts.append(text)
        rows.append(
            {
                "mel_path": metadata_row["mel_path"],
                "text": text,
                "duration": metadata_row["duration"],
                "speaker": "0",
                "emotion": "0",
            }
        )

    random.shuffle(rows)
    val_size = int(len(rows) * args.val_split)
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.csv"
    val_path = args.output_dir / "val.csv"
    write_filelist(train_rows, train_path)
    write_filelist(val_rows, val_path)

    symbols = build_symbols_from_texts(texts)
    processor = TextProcessor(symbols=symbols)
    processor.save(args.output_dir / "symbols.json")

    summary = {
        "data_dir": str(args.data_dir),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "n_symbols": len(symbols),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "symbols_file": str(args.output_dir / "symbols.json"),
    }
    (args.output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
