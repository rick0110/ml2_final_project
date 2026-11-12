#!/usr/bin/env python3
"""
Compute and cache the mel-spectrogram of every utterance in a manifest.

The data loader caches each mel the first time it is read, which makes the
first training epoch CPU-bound. This script does that pass up front with a
pool of single-threaded workers.

Example:
    python scripts/preprocess/cache_mels.py --data-dir data/processed/LJSpeech --text-cleaner english_cleaners
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "training" / "training-tacotron2-vae"))
sys.path.insert(0, str(ROOT / "src" / "data" / "loader_vae_tacotron"))
from loader_tacotron import DatasetLibriSpeechTacotronVAE  # noqa: E402
from text_processing import TextProcessor  # noqa: E402


def _single_thread(_: int) -> None:
    torch.set_num_threads(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--text-cleaner", default="english_cleaners")
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()
    dataset = DatasetLibriSpeechTacotronVAE(TextProcessor(cleaner_names=[a.text_cleaner]), data_dir=a.data_dir)
    loader = DataLoader(dataset, batch_size=None, num_workers=a.num_workers, worker_init_fn=_single_thread)
    frames = sum(mel.size(1) for _, mel, _ in loader)
    print(f"cached {len(dataset)} mels, {frames * 256 / 22050 / 3600:.2f} h of audio")


if __name__ == "__main__":
    main()
