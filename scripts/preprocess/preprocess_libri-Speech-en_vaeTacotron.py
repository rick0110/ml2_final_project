#!/usr/bin/env python3
"""
Preprocess LibriSpeech EN raw data into Tacotron2-VAE compatible mel-spectrogram tensors.

What this script does:
- Scans the LibriSpeech raw root for transcript files and matching audio
- Reads utterance text from `*.trans.txt`
- Loads audio, converts to mono, and resamples to target SR
- Computes Tacotron2-compatible 80-bin log-mel spectrograms (Dynamic Range Compression)
- Filters examples by duration
- Saves per-utterance `.pt` tensors
- Writes a metadata CSV
- Optionally plots a few mel spectrograms for validation
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
import librosa.filters
import tqdm

try:
    import matplotlib.subplots
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
        description="Prepare Tacotron2-compatible mel spectrograms for LibriSpeech EN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/libriSpeech-en"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/libriSpeech-en-tacotron-vae/mels"))
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Number of worker processes")
    
    # Tacotron 2 VAE Standard Parameters
    parser.add_argument("--target-sr", type=int, default=16000) # 16000 used in original repo
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--filter-length", type=int, default=1024) # n_fft
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--log-zero-guard-value", type=float, default=1e-5)

    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20) # Reduzido para 10s para evitar OOM no Tacotron
    parser.add_argument("--plot-samples", type=int, default=5, help="Number of mel images to save for validation (0 disables)")
    parser.add_argument("--seed", type=int, default=1234) # Padrão original
    parser.add_argument("--overwrite", action="store_true")
    
    return parser.parse_args()


class Tacotron2MelTransform(torch.nn.Module):
    """
    Exact replica of TacotronSTFT from the original tacotron2-vae-master project.
    Uses librosa for the mel basis to guarantee 100% parity.
    """
    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 16000,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
        clip_val: float = 1e-5,
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.clip_val = clip_val

        # Using librosa explicitly to match Tacotron2 exactly
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Explicit Padding to match original TacotronSTFT
        p = int((self.filter_length - self.hop_length) / 2)
        waveform = F.pad(waveform.unsqueeze(1), (p, p), mode='reflect').squeeze(1)

        stft_out = torch.stft(
            waveform,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        magnitudes = torch.abs(stft_out)

        mel_output = torch.matmul(self.mel_basis, magnitudes)
        
        # Dynamic Range Compression (Tacotron standard)
        mel_output = torch.log(torch.clamp(mel_output, min=self.clip_val))
        return mel_output


def resolve_input_root(input_dir: Path) -> Path:
    if input_dir.name == "LibriSpeech" and input_dir.exists():
        return input_dir
    candidate = input_dir / "LibriSpeech"
    if candidate.exists():
        return candidate
    return input_dir


def find_audio_file(transcript_path: Path, utt_id: str) -> Optional[Path]:
    for extension in (".flac", ".wav"):
        candidate = transcript_path.parent / f"{utt_id}{extension}"
        if candidate.exists():
            return candidate

    warning(f"Audio file not found for {utt_id} in {transcript_path.parent}")
    return None


def discover_examples(input_root: Path) -> List[Example]:
    examples: List[Example] = []
    transcript_files = sorted(input_root.rglob("*.trans.txt"))

    for transcript_file in transcript_files:
        with transcript_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue

                utt_id, text = parts[0].strip(), parts[1].strip()
                audio_file = find_audio_file(transcript_file, utt_id)
                if audio_file is None:
                    continue

                examples.append(
                    Example(
                        audio_path=str(audio_file.relative_to(input_root)),
                        text=text,
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
    audio_path = input_root / Path(ex.audio_path)
    if not audio_path.exists():
        return None

    out_path = out_root / f"{ex.utt_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists() and not overwrite:
        data = torch.load(out_path, map_location="cpu", weights_only=False)
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
    filter_length: int,
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
    _WORKER_MEL_TRANSFORM = Tacotron2MelTransform(
        sampling_rate=target_sr,
        n_mel_channels=n_mels,
        filter_length=filter_length,
        win_length=win_length,
        hop_length=hop_length,
        mel_fmin=f_min,
        mel_fmax=f_max,
        clip_val=log_zero_guard_value,
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


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 1234) -> None:
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
        data = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
        mel = data["mel"].squeeze(0).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(f"{row.get('utt_id', Path(row['mel_path']).stem)} - Tacotron2 Mel")
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Frame")
        fig.colorbar(im, ax=ax)

        out_path = out_dir / f"mel_{i:02d}_{row.get('utt_id', Path(row['mel_path']).stem)}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_root = resolve_input_root(args.input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    examples = discover_examples(input_root)
    manifest: List[Dict[str, Any]] = []

    if args.num_workers <= 1:
        mel_transform = Tacotron2MelTransform(
            sampling_rate=args.target_sr,
            n_mel_channels=args.n_mels,
            filter_length=args.filter_length,
            win_length=args.win_length,
            hop_length=args.hop_length,
            mel_fmin=args.f_min,
            mel_fmax=args.f_max,
            clip_val=args.log_zero_guard_value,
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
                args.filter_length,
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

    manifest_csv = out_root.parent / "mels_metadata.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mel_path", "duration", "text", "utt_id"])
        for row in manifest:
            writer.writerow([row["mel_path"], f"{row['duration']:.6f}", row.get("text", ""), row.get("utt_id", "")])

    if args.plot_samples > 0:
        figs_dir = out_root.parent / "figures"
        plot_mels(manifest, figs_dir, n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")
    print(f"Metadata written to {manifest_csv}")


if __name__ == "__main__":
    main()