#!/usr/bin/env python3

"""Preprocess TTS-Portuguese-Corpus into mel-spectrogram tensors and PyTorch Dataset.

What this script does:
- Reads cleaned wavs and corpus metadata
- Resamples audio to 22050 Hz, converts to mono
- Computes 80-band mel-spectrograms (log-scaled)
- Filters by duration (default 1s to 15s)
- Saves per-utterance tensors as `.pt` files and writes a `metadata.csv`
- Optionally plots a few mel spectrograms for validation

Usage example:
    python scripts/preprocess/prepare_mels_tts_portuguese.py --input-dir data/processed/tts-portuguese-Corpora/wavs --out-dir data/processed/tts-portuguese-Corpora/pt_tensors --plot-samples 8
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import tqdm
import matplotlib.pyplot as plt



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 80-band mel spectrograms and dataset for TTS-PT corpus")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw/tts-portuguese-Corpora/TTS-Portuguese-Corpus/"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/tts-portuguese-Corpora/"))
    p.add_argument("--target-sr", type=int, default=22050)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--win-length", type=int, default=1024)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=15.0)
    p.add_argument("--plot-samples", type=int, default=5, help="Number of mel images to save for validation (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def make_mel_transform(sr: int, n_fft: int, hop_length: int, win_length: int, n_mels: int) -> torch.nn.Module:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )
    db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
    return torch.nn.Sequential(mel, db)


def process_example(
    ex: List[Path],
    out_root: Path,
    mel_transform: torch.nn.Module,
    target_sr: int,
    hash_paths: Dict[Path, str],
) -> Optional[Dict[str, Any]]:
    waveform, sr = torchaudio.load(ex)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    log_mel = mel_transform(waveform)
    duration = waveform.shape[1] / sr
    text = text_from_path(ex, hash_paths)
    out_path_wav = out_root / "mels"
    out_path_wav.mkdir(parents=True, exist_ok=True)
    out_path = out_path_wav / f"{ex.stem}.pt"
    torch.save({"waveform": waveform, "mel": log_mel, "sr": sr, "duration": duration, "text": text}, str(out_path))
    return {"mel_path": str(out_path), "duration": duration, "text": text}


def plot_mels(manifest, out_dir, n=5, seed=42):
    rnd = random.Random(seed)
    samples = rnd.sample(manifest, min(n, len(manifest)))

    for i, row in enumerate(samples, start=1):
        data = torch.load(row["mel_path"])

        mel = data["mel"].squeeze(0).numpy()

        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(
            mel,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(f"{Path(row['mel_path']).stem}")
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Frame")

        fig.colorbar(im, ax=ax)

        out_path = out_dir / f"mel_{i:02d}_{Path(row['mel_path']).stem}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def find_examples(input_dir: Path) -> List[Path]:
    wav_paths = list(input_dir.rglob("*.wav"))
    return wav_paths

def text_from_path(wav_path: Path, hash_paths: Dict[Path, str]) -> str:
    return hash_paths.get(wav_path, "")

def create_hash_paths(input_dir: Path) -> Dict[Path, str]:
    with open(input_dir / "texts.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    lines = map(lambda x: list(x.strip().split("==", maxsplit=1)), lines)
    hash_paths = {}
    for (path, text) in lines:
        wav_path = input_dir / path
        text = text.strip()
        hash_paths[wav_path] = text
    return hash_paths

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.input_dir = args.input_dir.resolve()


    examples = find_examples(Path(args.input_dir))
    hash_paths = create_hash_paths(Path(args.input_dir))

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    mel_transform = make_mel_transform(args.target_sr, args.n_fft, args.hop_length, args.win_length, args.n_mels)

    manifest: List[Dict[str, Any]] = []
    for ex in tqdm.tqdm(examples, desc="Processing examples"):
        res = process_example(ex, out_root, mel_transform, args.target_sr, hash_paths)
        if res is None:
            continue
        if res["duration"] < args.min_duration or res["duration"] > args.max_duration:
            continue
        manifest.append(res)

    manifest_csv = out_root / "mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text"])
        for row in manifest:
            writer.writerow([row["mel_path"], f"{row['duration']:.6f}", row.get("text", "")])

    if args.plot_samples > 0:
        figs_dir = out_root / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")


if __name__ == "__main__":
    main()
