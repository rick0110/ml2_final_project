#!/usr/bin/env python3

## Warning: these transformed mels should not be used for inference or training the model
# All the transform should be applied by our engine to ensure compatibility

"""
Preprocess LibriSpeech PT raw data into FastPitch-compatible mel-spectrogram tensors.

Responsibilities:
    - Scan a LibriSpeech raw root for transcript files and matching audio.
    - Read utterance text from transcripts.
    - Load audio, convert to mono, and resample to 22050 Hz.
    - Compute FastPitch-compatible 80-bin log-mel spectrograms.
    - Filter examples by duration.
    - Save per-utterance `.pt` tensors.
    - Compute global mean/std over the full dataset for normalization.
    - Save normalized mels ("mel_normalized") back into the `.pt` files.
    - Write a metadata CSV and global stats (`mel_stats.pt`).
    - Optionally plot a few mel spectrograms for validation.

Main Classes:
    - FastPitchMelTransform: Mel extraction module matching NeMo/FastPitch settings.
    - Example: Dataclass for example metadata.

Main Functions:
    - discover_examples: Scan directory for audio and text pairs.
    - process_example: Load, transform, and save single example.
    - compute_dataset_stats: Calculate global mean and standard deviation.
    - add_normalized_mels: Update saved tensors with normalized versions.

Tensor Conventions:
    B = batch size
    S = number of audio samples
    T = number of frames
    n_mels = mel frequency bins (standard 80)
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
from torch import Tensor
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
    """
    Metadata for a discovered LibriSpeech-PT example.

    Attributes:
        audio_path (str): Relative path to the audio file.
        text (str): Utterance transcript.
        duration (float): Utterance duration in seconds.
        utt_id (str): Unique identifier.
    """
    audio_path: str
    text: str
    duration: float
    utt_id: str


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
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

    Architecture:
        Spectrogram -> Mel Scale -> Log Guard -> Log.

    Inputs:
        waveform:
            Shape (B, S) or (S,)

    Outputs:
        log_mel:
            Shape (B, n_mels, T)

    Example:
        >>> transform = FastPitchMelTransform()
        >>> audio = torch.randn(1, 22050)
        >>> mel = transform(audio)
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
        """
        Initialize FastPitch mel transform.

        Args:
            sample_rate (int): Sampling rate. Defaults to 22050.
            n_mels (int): Number of mel bins. Defaults to 80.
            n_fft (int): FFT size. Defaults to 1024.
            win_length (int): Window size. Defaults to 1024.
            hop_length (int): Hop size. Defaults to 256.
            f_min (float): Min frequency. Defaults to 0.0.
            f_max (float): Max frequency. Defaults to 8000.0.
            log_zero_guard_value (float): Log guard. Defaults to 1e-5.
        """
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

        self.log_zero_guard_value: float = float(log_zero_guard_value)

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Transform waveform to log-mel spectrogram.

        Args:
            waveform (Tensor): Input waveform.
                Shape: (B, S) or (S,)

        Returns:
            Tensor: Log-mel spectrogram.
                Shape: (B, n_mels, T)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, S)

        spec: Tensor = self.spec(waveform)          # (B, n_fft/2 + 1, T)
        mel: Tensor = self.mel_scale(spec)          # (B, n_mels, T)
        log_mel: Tensor = torch.log(mel + self.log_zero_guard_value) # (B, n_mels, T)
        return log_mel


def find_audio_file(transcript_path: Path, utt_id: str) -> Optional[Path]:
    """
    Find audio file for a given utterance ID.

    Args:
        transcript_path (Path): Path to transcript.
        utt_id (str): Utterance ID.

    Returns:
        Optional[Path]: Path to audio if exists.
    """
    candidate_stems = [
        transcript_path.parent / f"{utt_id}.wav"
    ]
    for candidate in candidate_stems:
        if candidate.exists():
            return candidate

    warning(f"Audio file not found for {utt_id} in {transcript_path.parent}")   
    return None


def discover_examples(input_root: Path) -> List[Example]:
    """
    Scan root directory for transcript and audio.

    Args:
        input_root (Path): Dataset root.

    Returns:
        List[Example]: Discovered metadata.
    """
    examples: List[Example] = []
    transcript_files: List[Path] = list((input_root).glob("trans*"))
    
    if not transcript_files:
        warning(f"No transcript files starting with 'trans' found in {input_root}")
        return []

    # Assuming we use the first one if multiple exist, or iterate
    for transcript_file in transcript_files:
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

                examples.append(
                    Example(
                        audio_path=str(audio_file.relative_to(input_root)),
                        text=text.strip(),
                        duration=0.0,
                        utt_id=utt_id,
                    )
                )
    print(f"Discovered {len(examples)} examples from {input_root}")

    return examples


def process_example(
    ex: Example,
    input_root: Path,
    out_root: Path,
    mel_transform: torch.nn.Module,
    target_sr: int,
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Process a single LibriSpeech-PT example.

    Args:
        ex (Example): Example metadata.
        input_root (Path): Input root.
        out_root (Path): Output root.
        mel_transform (torch.nn.Module): Mel transform.
        target_sr (int): Target SR.
        overwrite (bool): Overwrite flag.

    Returns:
        Optional[Dict[str, Any]]: Processed metadata.
    """
    audio_path: Path = input_root / Path(ex.audio_path)
    if not audio_path.exists():
        return None

    out_path: Path = out_root / f"{ex.utt_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        data: Dict[str, Any] = torch.load(out_path, map_location="cpu")
        return {
            "mel_path": str(out_path),
            "duration": float(data.get("duration", 0.0)),
            "text": data.get("text", ""),
            "utt_id": ex.utt_id,
        }

    waveform: Tensor
    sr: int
    waveform, sr = torchaudio.load(str(audio_path)) # (channels, S)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True) # (1, S)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr) # (1, S_new)
        sr = target_sr

    duration: float = waveform.shape[-1] / sr
    
    with torch.no_grad():
        log_mel: Tensor = mel_transform(waveform) # (1, n_mels, T)

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
    """
    Initialize worker global state.

    Args:
        input_root (Path): Input root.
        out_root (Path): Output root.
        target_sr (int): Target SR.
        n_mels (int): Mel bins.
        n_fft (int): FFT size.
        win_length (int): Window size.
        hop_length (int): Hop size.
        f_min (float): Min freq.
        f_max (float): Max freq.
        log_zero_guard_value (float): Guard value.
        overwrite (bool): Overwrite flag.
    """
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
    """
    Worker wrapper for processing.

    Args:
        ex (Example): Example metadata.

    Returns:
        Optional[Dict[str, Any]]: Processed metadata.
    """
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

    Args:
        manifest (List[Dict[str, Any]]): List of processed example metadata.

    Returns:
        Tuple[float, float]: Global mean and standard deviation.
    """
    total_sum: float = 0.0
    total_sum_sq: float = 0.0
    total_count: int = 0

    for row in tqdm.tqdm(manifest, desc="Computing global mel stats"):
        data: Dict[str, Any] = torch.load(row["mel_path"], map_location="cpu")
        mel: Tensor = data["mel"].to(torch.float64) # (1, n_mels, T)

        total_sum += mel.sum().item()
        total_sum_sq += (mel * mel).sum().item()
        total_count += mel.numel()

    if total_count == 0:
        raise RuntimeError("No mel values found to compute statistics.")

    mean: float = total_sum / total_count
    var: float = total_sum_sq / total_count - mean * mean

    if var <= 0.0:
        raise RuntimeError(f"Computed non-positive variance: {var}")

    std: float = var ** 0.5
    return mean, std


def add_normalized_mels(
    manifest: List[Dict[str, Any]],
    mean: float,
    std: float,
) -> None:
    """
    Reopen every .pt file and add z-score normalized mel.

    Args:
        manifest (List[Dict[str, Any]]): Manifest of examples.
        mean (float): Global mean.
        std (float): Global standard deviation.
    """
    mean_t: Tensor = torch.tensor(mean, dtype=torch.float32)
    std_t: Tensor = torch.tensor(std, dtype=torch.float32)

    for row in tqdm.tqdm(manifest, desc="Writing normalized mels"):
        path: Path = Path(row["mel_path"])
        data: Dict[str, Any] = torch.load(path, map_location="cpu")

        mel: Tensor = data["mel"].to(torch.float32) # (1, n_mels, T)
        mel_normalizado: Tensor = (mel - mean_t) / std_t # (1, n_mels, T)

        data["mel"] = mel
        data["mel_normalized"] = mel_normalizado

        torch.save(data, path)


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> None:
    """
    Visualize mel spectrograms.

    Args:
        manifest (List[Dict[str, Any]]): Manifest of examples.
        out_dir (Path): Output directory.
        n (int): Number of samples. Defaults to 5.
        seed (int): Random seed. Defaults to 42.
    """
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
        data: Dict[str, Any] = torch.load(row["mel_path"], map_location="cpu")
        mel: Tensor = data["mel"].squeeze(0) # (n_mels, T)
        mel_norm: Tensor = data["mel_normalized"].squeeze(0) # (n_mels, T)
        
        mel_np = mel.numpy()
        mel_norm_np = mel_norm.numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        im0 = axes[0].imshow(mel_np, origin="lower", aspect="auto", interpolation="nearest")
        axes[0].set_title(f"{row.get('utt_id', Path(row['mel_path']).stem)} - mel")
        axes[0].set_ylabel("Mel bin")
        axes[0].set_xlabel("Frame")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(mel_norm_np, origin="lower", aspect="auto", interpolation="nearest")
        axes[1].set_title(f"{row.get('utt_id', Path(row['mel_path']).stem)} - mel_normalized")
        axes[1].set_ylabel("Mel bin")
        axes[1].set_xlabel("Frame")
        fig.colorbar(im1, ax=axes[1])

        out_path = out_dir / f"mel_{i:02d}_{row.get('utt_id', Path(row['mel_path']).stem)}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    """
    Main entry point for LibriSpeech PT preprocessing.
    """
    args: argparse.Namespace = parse_args()
    random.seed(args.seed)

    input_root: Path = args.input_dir
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    examples: List[Example] = discover_examples(input_root)

    manifest: List[Dict[str, Any]] = []
    if args.num_workers <= 1:
        mel_transform: FastPitchMelTransform = FastPitchMelTransform(
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
            res: Optional[Dict[str, Any]] = process_example(
                ex, input_root, out_root, mel_transform, args.target_sr, overwrite=args.overwrite
            )
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
            chunksize: int = max(1, len(examples) // max(args.num_workers * 8, 1))
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
    stats_path: Path = out_root.parent / "mel_stats.pt"
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

    manifest_csv: Path = out_root.parent / "librispeech_mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text", "utt_id"])
        for row in manifest:
            writer.writerow([row["mel_path"], f"{row['duration']:.6f}", row.get("text", ""), row.get("utt_id", "")])

    if args.plot_samples > 0:
        figs_dir: Path = out_root.parent / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")
    print(f"Metadata written to {manifest_csv}")
    print(f"Global mel stats written to {stats_path}")
    print(f"Global mean: {mean:.6f}")
    print(f"Global std : {std:.6f}")


if __name__ == "__main__":
    main()
