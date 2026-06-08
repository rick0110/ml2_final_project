#!/usr/bin/env python3

"""Preprocess LibriSpeech raw data into FastPitch-compatible mel-spectrogram tensors.

What this script does:
- Scans a LibriSpeech raw root for transcript files and matching audio
- Reads utterance text from `*.trans.txt`
- Loads audio, converts to mono, and resamples to 22050 Hz
- Computes FastPitch-compatible 80-bin log-mel spectrograms
- Filters examples by duration
- Saves per-utterance `.pt` tensors
- Computes global mean/std over the full dataset
- Saves normalized mels ("mel_normalized") back into the `.pt` files
- Writes a metadata CSV and global stats (`mel_stats.pt`)
- Optionally plots a few mel spectrograms for validation
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


_WORKER_INPUT_ROOT: Optional[Path] = None
_WORKER_OUT_ROOT: Optional[Path] = None
_WORKER_TARGET_SR: Optional[int] = None
_WORKER_OVERWRITE: bool = False
_WORKER_MEL_TRANSFORM: Optional[torch.nn.Module] = None


@dataclass
class Example:
    audio_path: str
    text: str
    duration: float
    utt_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FastPitch-compatible mel spectrograms for LibriSpeech",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/libriSpeech-pt/mls_portuguese_opus/train"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/LibriSpeech/mels"))
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Number of worker processes for preprocessing (1 disables multiprocessing)")
    
    # FastPitch Standard Parameters
    parser.add_argument("--target-sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--log-zero-guard-value", type=float, default=1e-5)

    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--plot-samples", type=int, default=5, help="Number of mel images to save for validation (0 disables)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class FastPitchMelTransform(torch.nn.Module):
    """
    FastPitch-compatible log-mel extraction.
    Matches the NeMo preprocessor settings exactly.
    """
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        log_zero_guard_value: float = 1e-5,
    ) -> None:
        super().__init__()

        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
            window_fn=torch.hann_window,
        )

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )

        self.log_zero_guard_value = float(log_zero_guard_value)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        spec = self.spec(waveform)          # [C, F, T]
        mel = self.mel_scale(spec)          # [C, n_mels, T]
        log_mel = torch.log(mel + self.log_zero_guard_value)
        return log_mel


def find_audio_file(transcript_path: Path, utt_id: str) -> Optional[Path]:
    candidate_stems = [
        transcript_path.parent / f"{utt_id}.wav"
    ]
    for candidate in candidate_stems:
        if candidate.exists():
            return candidate

    warning(f"Audio file not found for {utt_id} in {transcript_path.parent}")   
    return None


def discover_examples(input_root: Path) -> List[Example]:
    examples: List[Example] = []
    transcript_file = (input_root ).glob("trans*")
    for trans in transcript_file: 
        transcript_file = trans
    
    with transcript_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line:
                continue
            line = line.strip()
            match = re.match(r"^(\S+)\s+(.*)$", line)
            utt_id = match.group(1).strip() if match else None
            text = match.group(2).strip() if match else None
            
            if not utt_id or not text:
                continue
            split_utt_id = utt_id.split("_")
            audio_file_root = input_root / "audio" / "/".join(split_utt_id[:-1])
            audio_file = audio_file_root / f"{'_'.join(split_utt_id)}.wav"

            if audio_file is None:
                continue
            examples.append(
                Example(
                    audio_path=str(audio_file.relative_to(input_root)),
                    text=text.strip(),
                    duration=0.0,
                    utt_id=utt_id,
                )
            )
    print(f"Discovered {len(examples)} examples from {transcript_file}")

    return examples


def process_example(
    ex: Example,
    input_root: Path,
    out_root: Path,
    mel_transform: torch.nn.Module,
    target_sr: int,
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    audio_path = input_root / Path(ex.audio_path)
    if not audio_path.exists():
        return None

    out_path = out_root / f"{ex.utt_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        data = torch.load(out_path, map_location="cpu")
        return {
            "mel_path": str(out_path),
            "duration": float(data.get("duration", 0.0)),
            "text": data.get("text", ""),
            "utt_id": ex.utt_id,
        }

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    duration = waveform.shape[-1] / sr
    
    with torch.no_grad():
        log_mel = mel_transform(waveform)

    torch.save(
        {
            "waveform": waveform.cpu(),
            "mel": log_mel.cpu(),
            "sr": sr,
            "duration": float(duration),
            "text": ex.text,
            "utt_id": ex.utt_id,
            "audio_path": ex.audio_path,
        },
        str(out_path),
    )
    return {
        "mel_path": str(out_path),
        "duration": float(duration),
        "text": ex.text,
        "utt_id": ex.utt_id,
    }


def _init_worker(
    input_root: Path,
    out_root: Path,
    target_sr: int,
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    log_zero_guard_value: float,
    overwrite: bool,
) -> None:
    global _WORKER_INPUT_ROOT, _WORKER_OUT_ROOT, _WORKER_TARGET_SR, _WORKER_OVERWRITE, _WORKER_MEL_TRANSFORM
    _WORKER_INPUT_ROOT = input_root
    _WORKER_OUT_ROOT = out_root
    _WORKER_TARGET_SR = target_sr
    _WORKER_OVERWRITE = overwrite
    _WORKER_MEL_TRANSFORM = FastPitchMelTransform(
        sample_rate=target_sr,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        log_zero_guard_value=log_zero_guard_value,
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _process_example_worker(ex: Example) -> Optional[Dict[str, Any]]:
    if _WORKER_INPUT_ROOT is None or _WORKER_OUT_ROOT is None or _WORKER_TARGET_SR is None or _WORKER_MEL_TRANSFORM is None:
        raise RuntimeError("Worker was not initialized correctly")
    return process_example(
        ex,
        _WORKER_INPUT_ROOT,
        _WORKER_OUT_ROOT,
        _WORKER_MEL_TRANSFORM,
        _WORKER_TARGET_SR,
        overwrite=_WORKER_OVERWRITE,
    )


def compute_dataset_stats(manifest: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Compute global mean/std over ALL values in ALL mel tensors.
    """
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    for row in tqdm.tqdm(manifest, desc="Computing global mel stats"):
        data = torch.load(row["mel_path"], map_location="cpu")
        mel = data["mel"].to(torch.float64)

        total_sum += mel.sum().item()
        total_sum_sq += (mel * mel).sum().item()
        total_count += mel.numel()

    if total_count == 0:
        raise RuntimeError("No mel values found to compute statistics.")

    mean = total_sum / total_count
    var = total_sum_sq / total_count - mean * mean

    if var <= 0.0:
        raise RuntimeError(f"Computed non-positive variance: {var}")

    std = var ** 0.5
    return mean, std


def add_normalized_mels(
    manifest: List[Dict[str, Any]],
    mean: float,
    std: float,
) -> None:
    """
    Reopen every .pt file and add z-score normalized mel.
    """
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    for row in tqdm.tqdm(manifest, desc="Writing normalized mels"):
        path = Path(row["mel_path"])
        data = torch.load(path, map_location="cpu")

        mel = data["mel"].to(torch.float32)
        mel_normalizado = (mel - mean_t) / std_t

        data["mel"] = mel
        data["mel_normalized"] = mel_normalizado

        torch.save(data, path)


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    if not manifest:
        print("No examples available for plotting")
        return

    rnd = random.Random(seed)
    samples = rnd.sample(manifest, min(n, len(manifest)))
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i, row in enumerate(samples, start=1):
        data = torch.load(row["mel_path"], map_location="cpu")
        mel = data["mel"].squeeze(0).numpy()
        mel_norm = data["mel_normalized"].squeeze(0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        im0 = axes[0].imshow(mel, origin="lower", aspect="auto", interpolation="nearest")
        axes[0].set_title(f"{row.get('utt_id', Path(row['mel_path']).stem)} - mel")
        axes[0].set_ylabel("Mel bin")
        axes[0].set_xlabel("Frame")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(mel_norm, origin="lower", aspect="auto", interpolation="nearest")
        axes[1].set_title(f"{row.get('utt_id', Path(row['mel_path']).stem)} - mel_normalized")
        axes[1].set_ylabel("Mel bin")
        axes[1].set_xlabel("Frame")
        fig.colorbar(im1, ax=axes[1])

        out_path = out_dir / f"mel_{i:02d}_{row.get('utt_id', Path(row['mel_path']).stem)}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_root = args.input_dir
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    examples = discover_examples(input_root)

    manifest: List[Dict[str, Any]] = []
    if args.num_workers <= 1:
        mel_transform = FastPitchMelTransform(
            sample_rate=args.target_sr,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            f_min=args.f_min,
            f_max=args.f_max,
            log_zero_guard_value=args.log_zero_guard_value,
        )
        for ex in tqdm.tqdm(examples, desc="Processing examples"):
            res = process_example(ex, input_root, out_root, mel_transform, args.target_sr, overwrite=args.overwrite)
            if res is None:
                warning(f"Failed to process example {ex.utt_id}; skipping")
                continue
            
            if res["duration"] < args.min_duration or res["duration"] > args.max_duration:
                warning(f"Example {ex.utt_id} duration {res['duration']:.2f}s out of bounds; skipping")
                continue
            manifest.append(res)
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(
                input_root,
                out_root,
                args.target_sr,
                args.n_mels,
                args.n_fft,
                args.win_length,
                args.hop_length,
                args.f_min,
                args.f_max,
                args.log_zero_guard_value,
                args.overwrite,
            ),
        ) as executor:
            chunksize = max(1, len(examples) // max(args.num_workers * 8, 1))
            for ex, res in zip(examples, tqdm.tqdm(executor.map(_process_example_worker, examples, chunksize=chunksize), total=len(examples), desc="Processing examples")):
                if res is None:
                    warning(f"Failed to process example {ex.utt_id}; skipping")
                    continue
                if res["duration"] < args.min_duration or res["duration"] > args.max_duration:
                    warning(f"Example {ex.utt_id} duration {res['duration']:.2f}s out of bounds; skipping")
                    continue
                manifest.append(res)

    if not manifest:
        raise RuntimeError("No examples were kept after duration filtering.")

    # Compute Global Dataset Stats
    mean, std = compute_dataset_stats(manifest)

    # Save Stats File
    stats_path = out_root.parent / "mel_stats.pt"
    torch.save(
        {
            "mean": float(mean),
            "std": float(std),
            "var": float(std * std),
            "n_mels": args.n_mels,
            "sample_rate": args.target_sr,
            "n_fft": args.n_fft,
            "win_length": args.win_length,
            "hop_length": args.hop_length,
            "f_min": args.f_min,
            "f_max": args.f_max,
            "log_zero_guard_value": args.log_zero_guard_value,
        },
        stats_path,
    )

    # Re-open .pt files to inject mel_normalized
    add_normalized_mels(manifest, mean, std)

    manifest_csv = out_root.parent / "librispeech_mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text", "utt_id"])
        for row in manifest:
            writer.writerow([row["mel_path"], f"{row['duration']:.6f}", row.get("text", ""), row.get("utt_id", "")])

    if args.plot_samples > 0:
        figs_dir = out_root.parent / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")
    print(f"Metadata written to {manifest_csv}")
    print(f"Global mel stats written to {stats_path}")
    print(f"Global mean: {mean:.6f}")
    print(f"Global std : {std:.6f}")


if __name__ == "__main__":
    main()