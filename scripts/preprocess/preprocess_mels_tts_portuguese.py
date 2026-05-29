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

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class Example:
    audio_path: str
    text: str
    duration: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 80-band mel spectrograms and dataset for TTS-PT corpus")
    p.add_argument("--input-dir", type=Path, default=Path("data/processed/tts-portuguese-Corpora/wavs"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/tts-portuguese-Corpora/pt_tensors"))
    p.add_argument("--target-sr", type=int, default=22050)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--win-length", type=int, default=1024)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=15.0)
    p.add_argument("--plot-samples", type=int, default=5, help="Number of mel images to save for validation (0 disables)")
    p.add_argument("--seed", type=int, default=42)''
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()



def make_mel_transform(sr: int, n_fft: int, hop_length: int, win_length: int, n_mels: int) -> torch.nn.Module:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=1.0,
    )
    db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
    return torch.nn.Sequential(mel, db)


def process_example(
    ex: Example,
    input_root: Path,
    out_root: Path,
    mel_transform: torch.nn.Module,
    target_sr: int,
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    audio_p = Path(ex.audio_path)
    if audio_p.is_absolute():
        src = audio_p
    else:
        candidate = input_root / audio_p
        if candidate.exists():
            src = candidate
        else:
            candidate2 = input_root / audio_p.name
            if candidate2.exists():
                src = candidate2
            else:
                matches = list(input_root.rglob(audio_p.name))
                if matches:
                    src = matches[0]
                else:
                    return None
    stem = Path(ex.audio_path).stem
    out_path = out_root / f"{stem}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        data = torch.load(out_path)
        return {"mel_path": str(out_path), "duration": data.get("duration", 0.0), "text": data.get("text", "")}

    waveform, sr = torchaudio.load(str(src))
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    duration = waveform.shape[-1] / sr
    mel = mel_transform(waveform)  
    mel = mel.squeeze(0)

    torch.save({"waveform": waveform, "mel": mel, "sr": sr, "duration": duration, "text": ex.text}, str(out_path))
    return {"mel_path": str(out_path), "duration": duration, "text": ex.text}


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    rnd = random.Random(seed)
    samples = rnd.sample(manifest, min(n, len(manifest)))
    for i, row in enumerate(samples, start=1):
        data = torch.load(row["mel_path"]) if isinstance(row["mel_path"], str) else row["mel_path"]
        mel = data["mel"].numpy()
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(f"{Path(row['mel_path']).stem} | {row.get('text','')[:60]}")
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Frame")
        fig.colorbar(im, ax=ax)
        out_path = out_dir / f"mel_{i:02d}_{Path(row['mel_path']).stem}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    examples: List[Example]
    wav_paths = sorted(Path(args.input_dir).rglob("*.wav"))
    examples = [Example(audio_path=str(p.relative_to(args.input_dir)), text="", duration=0.0) for p in wav_paths]

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    mel_transform = make_mel_transform(args.target_sr, args.n_fft, args.hop_length, args.win_length, args.n_mels)

    manifest: List[Dict[str, Any]] = []
    for ex in tqdm.tqdm(examples, desc="Processing examples"):
        res = process_example(ex, Path(args.input_dir), out_root, mel_transform, args.target_sr, overwrite=args.overwrite)
        if res is None:
            continue
        if res["duration"] < args.min_duration or res["duration"] > args.max_duration:
            continue
        manifest.append(res)

    manifest_csv = out_root.parent / "mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text"])
        for row in manifest:
            writer.writerow([row["mel_path"], f"{row['duration']:.6f}", row.get("text", "")])

    if args.plot_samples > 0:
        figs_dir = out_root.parent / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")


if __name__ == "__main__":
    main()
