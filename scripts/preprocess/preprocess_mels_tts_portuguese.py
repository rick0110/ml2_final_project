#!/usr/bin/env python3
"""
Preprocess TTS-Portuguese-Corpus into FastPitch-compatible mel-spectrogram tensors.

Responsibilities:
    - Read wavs and corpus metadata from the TTS-Portuguese-Corpus.
    - Resample audio to 22050 Hz and convert to mono.
    - Compute FastPitch-compatible 80-bin log-mel spectrograms.
    - Filter by duration bounds.
    - Save per-utterance tensors as .pt files containing waveform, mel, and metadata.
    - Write a metadata CSV manifest.
    - Compute global mean/std over the full dataset for z-score normalization.
    - Inject "mel_normalized" into saved tensors.
    - Optionally plot mel spectrograms for validation.

Main Classes:
    - FastPitchMelTransform: Mel extraction module matching NeMo/FastPitch settings.

Main Functions:
    - find_examples: Find all .wav files recursively.
    - create_text_lookup: Map wav paths to transcripts using texts.csv.
    - process_example: Single example processing logic.
    - compute_dataset_stats: Global statistics calculation.
    - add_normalized_mels: Apply normalization to saved tensors.

Tensor Conventions:
    B = batch size
    S = number of audio samples
    T = number of frames
    n_mels = mel frequency bins (standard 80)
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torchaudio
import tqdm


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare FastPitch-compatible mel spectrograms for TTS-Portuguese corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/tts-portuguese-Corpora/TTS-Portuguese-Corpus/"),
        help="Root directory containing wav files and texts.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/tts-portuguese-Corpora/"),
        help="Output directory for mel tensors and metadata",
    )

    parser.add_argument("--target-sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--log-zero-guard-value", type=float, default=1e-5)

    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=5,
        help="Number of mel images to save for validation (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class FastPitchMelTransform(torch.nn.Module):
    """
    FastPitch-compatible log-mel extraction.

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
            log_zero_guard_value (float): Guard value for log scaling. Defaults to 1e-5.
        """
        super().__init__()

        self.spec: torchaudio.transforms.Spectrogram = torchaudio.transforms.Spectrogram(
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

        self.mel_scale: torchaudio.transforms.MelScale = torchaudio.transforms.MelScale(
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
            waveform = waveform.unsqueeze(0) # (1, S)

        spec: Tensor = self.spec(waveform)          # (B, n_fft/2 + 1, T)
        mel: Tensor = self.mel_scale(spec)          # (B, n_mels, T)
        log_mel: Tensor = torch.log(mel + self.log_zero_guard_value) # (B, n_mels, T)
        return log_mel


def find_examples(input_dir: Path) -> List[Path]:
    """
    Find all .wav files recursively in a directory.

    Args:
        input_dir (Path): Directory to search.

    Returns:
        List[Path]: Sorted list of .wav file paths.
    """
    return sorted(input_dir.rglob("*.wav"))


def create_text_lookup(input_dir: Path) -> Dict[Path, str]:
    """
    Create a mapping from wav path to transcript using texts.csv.

    Args:
        input_dir (Path): Directory containing texts.csv.

    Returns:
        Dict[Path, str]: Lookup dictionary.
    """
    texts_file: Path = input_dir / "texts.csv"
    if not texts_file.exists():
        raise FileNotFoundError(f"texts.csv not found: {texts_file}")

    lookup: Dict[Path, str] = {}

    with texts_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line: str = raw_line.strip()
            if not line:
                continue

            if "==" not in line:
                continue

            rel_path, text = line.split("==", maxsplit=1)
            wav_path: Path = (input_dir / rel_path).resolve()
            lookup[wav_path] = text.strip()

    return lookup


def text_from_path(wav_path: Path, lookup: Dict[Path, str]) -> str:
    """
    Get transcript text for a wav path from the lookup.

    Args:
        wav_path (Path): Path to wav file.
        lookup (Dict[Path, str]): Lookup dictionary.

    Returns:
        str: Transcript text.
    """
    return lookup.get(wav_path.resolve(), "")


def process_example(
    wav_path: Path,
    out_root: Path,
    mel_transform: FastPitchMelTransform,
    target_sr: int,
    text_lookup: Dict[Path, str],
    min_duration: float,
    max_duration: float,
) -> Optional[Dict[str, Any]]:
    """
    Process a single audio example.

    Args:
        wav_path (Path): Path to wav file.
        out_root (Path): Output directory for mels.
        mel_transform (FastPitchMelTransform): Mel transform module.
        target_sr (int): Target sampling rate.
        text_lookup (Dict[Path, str]): Mapping from path to text.
        min_duration (float): Minimum duration filter.
        max_duration (float): Maximum duration filter.

    Returns:
        Optional[Dict[str, Any]]: Metadata if processed, else None.
    """
    waveform: Tensor
    sr: int
    waveform, sr = torchaudio.load(wav_path) # (channels, S)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
        )(waveform) # (channels, S_new)
        sr = target_sr

    duration: float = waveform.shape[1] / sr
    if duration < min_duration or duration > max_duration:
        return None

    with torch.no_grad():
        log_mel: Tensor = mel_transform(waveform) # (channels, n_mels, T)

    text: str = text_from_path(wav_path, text_lookup)

    out_mels_dir: Path = out_root / "mels"
    out_mels_dir.mkdir(parents=True, exist_ok=True)

    out_path: Path = out_mels_dir / f"{wav_path.stem}.pt"

    payload: Dict[str, Any] = {
        "waveform": waveform.cpu(),
        "mel": log_mel.cpu(),  # raw log-mel
        "sr": sr,
        "duration": float(duration),
        "text": text,
        "source_wav": str(wav_path.resolve()),
    }
    torch.save(payload, out_path)

    return {
        "mel_path": str(out_path.resolve()),
        "duration": float(duration),
        "text": text,
        "source_wav": str(wav_path.resolve()),
    }


def compute_dataset_stats(manifest: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Compute global mean/std over ALL values in ALL mel tensors.
    This is a dataset-wide z-score normalization.

    Args:
        manifest (List[Dict[str, Any]]): Manifest of processed examples.

    Returns:
        Tuple[float, float]: Global mean and standard deviation.
    """
    total_sum: float = 0.0
    total_sum_sq: float = 0.0
    total_count: int = 0

    for row in tqdm.tqdm(manifest, desc="Computing global mel stats"):
        data: Dict[str, Any] = torch.load(row["mel_path"], map_location="cpu")
        mel: Tensor = data["mel"].to(torch.float64) # (channels, n_mels, T)

        total_sum += mel.sum().item()
        total_sum_sq += (mel * mel).sum().item()
        total_count += mel.numel()

    if total_count == 0:
        raise RuntimeError("No mel values found to compute statistics.")

    mean: float = total_sum / total_count
    var: float = total_sum_sq / total_count - mean * mean

    # Numerical safety
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

        mel: Tensor = data["mel"].to(torch.float32) # (channels, n_mels, T)
        mel_normalizado: Tensor = (mel - mean_t) / std_t # (channels, n_mels, T)

        data["mel"] = mel
        data["mel_normalized"] = mel_normalizado

        torch.save(data, path)


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> None:
    """
    Visualize mel spectrograms for a sample of examples.

    Args:
        manifest (List[Dict[str, Any]]): Manifest of processed examples.
        out_dir (Path): Output directory for figures.
        n (int): Number of samples to plot. Defaults to 5.
        seed (int): Random seed for sampling. Defaults to 42.
    """
    if not manifest:
        return

    rnd = random.Random(seed)
    samples: List[Dict[str, Any]] = rnd.sample(manifest, min(n, len(manifest)))

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(samples, start=1):
        data: Dict[str, Any] = torch.load(row["mel_path"], map_location="cpu")
        mel: Tensor = data["mel"].squeeze(0) # (n_mels, T)
        mel_norm: Tensor = data["mel_normalized"].squeeze(0) # (n_mels, T)
        
        mel_np = mel.numpy()
        mel_norm_np = mel_norm.numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        im0 = axes[0].imshow(
            mel_np,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[0].set_title(f"{Path(row['mel_path']).stem} - mel")
        axes[0].set_ylabel("Mel bin")
        axes[0].set_xlabel("Frame")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            mel_norm_np,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[1].set_title(f"{Path(row['mel_path']).stem} - mel_normalized")
        axes[1].set_ylabel("Mel bin")
        axes[1].set_xlabel("Frame")
        fig.colorbar(im1, ax=axes[1])

        out_path: Path = out_dir / f"mel_{i:02d}_{Path(row['mel_path']).stem}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    """
    Main entry point for TTS-Portuguese preprocessing.
    """
    args: argparse.Namespace = parse_args()
    random.seed(args.seed)

    input_dir: Path = args.input_dir.resolve()
    out_root: Path = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mels").mkdir(parents=True, exist_ok=True)

    examples: List[Path] = find_examples(input_dir)
    if not examples:
        raise FileNotFoundError(f"No .wav files found under: {input_dir}")

    text_lookup: Dict[Path, str] = create_text_lookup(input_dir)

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

    manifest: List[Dict[str, Any]] = []

    for wav_path in tqdm.tqdm(examples, desc="Processing examples"):
        result = process_example(
            wav_path=wav_path,
            out_root=out_root,
            mel_transform=mel_transform,
            target_sr=args.target_sr,
            text_lookup=text_lookup,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        if result is not None:
            manifest.append(result)

    if not manifest:
        raise RuntimeError("No examples were kept after duration filtering.")

    mean, std = compute_dataset_stats(manifest)

    stats_path: Path = out_root / "mel_stats.pt"
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

    add_normalized_mels(manifest, mean, std)

    manifest_csv: Path = out_root / "mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text", "utt_id", "source_wav"])
        for row in manifest:
            utt_id = Path(row["mel_path"]).stem
            writer.writerow(
                [
                    row["mel_path"],
                    f"{row['duration']:.6f}",
                    row.get("text", ""),
                    utt_id,
                    row.get("source_wav", ""),
                ]
            )

    if args.plot_samples > 0:
        figs_dir: Path = out_root / "figures"
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")
    print(f"Metadata written to {manifest_csv}")
    print(f"Global mel stats written to {stats_path}")
    print(f"Global mean: {mean:.6f}")
    print(f"Global std : {std:.6f}")


if __name__ == "__main__":
    main()
