#!/usr/bin/env python3
"""
Preprocess TTS-Portuguese-Corpus into FastPitch-compatible mel-spectrogram tensors.

What this script does:
- Reads wavs and corpus metadata
- Resamples audio to 22050 Hz
- Converts to mono
- Computes FastPitch-compatible 80-bin log-mel spectrograms
- Filters by duration
- Saves per-utterance tensors as .pt files
- Writes a metadata CSV
- Computes global mean/std over the full dataset
- Saves:
    - "mel"             -> raw log-mel
    - "mel_normalizado" -> z-score normalized mel using dataset-wide mean/std
- Optionally plots a few mel spectrograms for validation

FastPitch defaults used here:
- sample_rate = 22050
- n_mels = 80
- n_fft = 1024
- win_length = 1024
- hop_length = 256
- lowfreq = 0
- highfreq = 8000
- window = hann
- log = True
- log_zero_guard_value = 1e-5
- mag_power = 1.0
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torchaudio
import tqdm


def parse_args() -> argparse.Namespace:
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

    Matches the NeMo preprocessor settings:
    - features=80
    - lowfreq=0
    - highfreq=8000
    - n_fft=1024
    - n_window_size=1024
    - n_window_stride=256
    - sample_rate=22050
    - window='hann'
    - log=True
    - log_zero_guard_value=1e-5
    - mag_power=1.0
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
        """
        waveform: [channels, time] or [time]
        returns:   [channels, n_mels, frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        spec = self.spec(waveform)          # [C, F, T]
        mel = self.mel_scale(spec)          # [C, n_mels, T]
        log_mel = torch.log(mel + self.log_zero_guard_value)
        return log_mel


def find_examples(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("*.wav"))


def create_text_lookup(input_dir: Path) -> Dict[Path, str]:
    texts_file = input_dir / "texts.csv"
    if not texts_file.exists():
        raise FileNotFoundError(f"texts.csv not found: {texts_file}")

    lookup: Dict[Path, str] = {}

    with texts_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "==" not in line:
                continue

            rel_path, text = line.split("==", maxsplit=1)
            wav_path = (input_dir / rel_path).resolve()
            lookup[wav_path] = text.strip()

    return lookup


def text_from_path(wav_path: Path, lookup: Dict[Path, str]) -> str:
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
    waveform, sr = torchaudio.load(wav_path)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
        )(waveform)
        sr = target_sr

    duration = waveform.shape[1] / sr
    if duration < min_duration or duration > max_duration:
        return None

    with torch.no_grad():
        log_mel = mel_transform(waveform)

    text = text_from_path(wav_path, text_lookup)

    out_mels_dir = out_root / "mels"
    out_mels_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_mels_dir / f"{wav_path.stem}.pt"

    payload = {
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

    # Numerical safety
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
    Reopen every .pt file and add:
        mel_normalizado = (mel - mean) / std
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
    if not manifest:
        return

    rnd = random.Random(seed)
    samples = rnd.sample(manifest, min(n, len(manifest)))

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(samples, start=1):
        data = torch.load(row["mel_path"], map_location="cpu")
        mel = data["mel"].squeeze(0).numpy()
        mel_norm = data["mel_normalized"].squeeze(0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        im0 = axes[0].imshow(
            mel,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[0].set_title(f"{Path(row['mel_path']).stem} - mel")
        axes[0].set_ylabel("Mel bin")
        axes[0].set_xlabel("Frame")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            mel_norm,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axes[1].set_title(f"{Path(row['mel_path']).stem} - mel_normalized")
        axes[1].set_ylabel("Mel bin")
        axes[1].set_xlabel("Frame")
        fig.colorbar(im1, ax=axes[1])

        out_path = out_dir / f"mel_{i:02d}_{Path(row['mel_path']).stem}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_dir = args.input_dir.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "mels").mkdir(parents=True, exist_ok=True)

    examples = find_examples(input_dir)
    if not examples:
        raise FileNotFoundError(f"No .wav files found under: {input_dir}")

    text_lookup = create_text_lookup(input_dir)

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

    manifest: List[Dict[str, Any]] = []

    for wav_path in tqdm.tqdm(examples, desc="Processing examples"):
        res = process_example(
            wav_path=wav_path,
            out_root=out_root / "mels",
            mel_transform=mel_transform,
            target_sr=args.target_sr,
            text_lookup=text_lookup,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        if res is not None:
            manifest.append(res)

    if not manifest:
        raise RuntimeError("No examples were kept after duration filtering.")

    mean, std = compute_dataset_stats(manifest)

    stats_path = out_root / "mel_stats.pt"
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

    manifest_csv = out_root / "mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text", "source_wav"])
        for row in manifest:
            writer.writerow(
                [
                    row["mel_path"],
                    f"{row['duration']:.6f}",
                    row.get("text", ""),
                    row.get("source_wav", ""),
                ]
            )

    if args.plot_samples > 0:
        figs_dir = out_root / "figures"
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")
    print(f"Metadata written to {manifest_csv}")
    print(f"Global mel stats written to {stats_path}")
    print(f"Global mean: {mean:.6f}")
    print(f"Global std : {std:.6f}")


if __name__ == "__main__":
    main()