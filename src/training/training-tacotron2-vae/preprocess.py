"""
Preprocessing utility for Tacotron 2 VAE dataset preparation.

Responsibilities:
    - Load processed mel-spectrogram data and metadata.
    - Split dataset into training and validation sets.
    - Generate unique symbol vocabulary from text transcripts.
    - Save filelists (CSV) and symbol tables (JSON) for training.

Main Functions:
    - write_filelist: Serialize dataset rows to a CSV file.
    - main: Primary orchestration for filelist and vocabulary creation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))

try:
    from data.loader_TTS_GST.DataSet import DatasetTTSPortuguese
    from text_processing import TextProcessor, build_symbols_from_texts
    from utils import ARTIFACTS_DIR
except ImportError:
    # Handle absolute paths if needed
    from src.training.training_tacotron2_vae.text_processing import TextProcessor, build_symbols_from_texts
    from src.training.training_tacotron2_vae.utils import ARTIFACTS_DIR


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset preprocessing.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
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


def write_filelist(rows: List[Dict[str, Any]], path: Path) -> None:
    """
    Write a list of dataset rows to a CSV file.

    Args:
        rows (List[Dict[str, Any]]): Metadata rows.
        path (Path): Destination CSV path.
    """
    fieldnames: List[str] = ["mel_path", "text", "duration", "emotion"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """
    Main preprocessing routine.
    """
    args: argparse.Namespace = parse_args()
    random.seed(args.seed)

    # Note: Assuming DatasetTTSPortuguese exists in the specified path
    try:
        from data.loader_TTS_GST.DataSet import DatasetTTSPortuguese
        dataset: DatasetTTSPortuguese = DatasetTTSPortuguese(data_dir=args.data_dir)
    except ImportError:
        print("Warning: DatasetTTSPortuguese import failed. Logic requires proper src setup.")
        return

    rows: List[Dict[str, Any]] = []
    texts: List[str] = []

    for idx in range(len(dataset)):
        sample: Dict[str, Any] = dataset[idx]
        metadata_row: Dict[str, str] = dataset.files[idx]
        text: str = sample["text"]
        texts.append(text)
        rows.append(
            {
                "mel_path": metadata_row["mel_path"],
                "text": text,
                "duration": metadata_row["duration"],
                "emotion": "0",
            }
        )

    random.shuffle(rows)
    val_size: int = int(len(rows) * args.val_split)
    val_rows: List[Dict[str, Any]] = rows[:val_size]
    train_rows: List[Dict[str, Any]] = rows[val_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path: Path = args.output_dir / "train.csv"
    val_path: Path = args.output_dir / "val.csv"
    write_filelist(train_rows, train_path)
    write_filelist(val_rows, val_path)

    symbols: List[str] = build_symbols_from_texts(texts)
    processor: TextProcessor = TextProcessor(symbols=symbols)
    processor.save(args.output_dir / "symbols.json")

    summary: Dict[str, Any] = {
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
