#!/usr/bin/env python3

"""Preprocess LibriSpeech EN raw data into mel-spectrogram tensors.

What this script does:
- Scans the LibriSpeech raw root for transcript files and matching audio
- Reads utterance text from `*.trans.txt`
- Loads audio, converts to mono, and resamples to 22050 Hz
- Computes 80-band mel-spectrograms (log-scaled)
- Filters examples by duration
- Saves per-utterance `.pt` tensors and a manifest CSV
- Optionally plots a few mel spectrograms for validation

Expected LibriSpeech layout:
	<input-dir>/LibriSpeech/<split>/<speaker>/<chapter>/<utt-id>.flac
	<input-dir>/LibriSpeech/<split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

The script also accepts an input directory that already points to the
`LibriSpeech/` folder.

Example:
	python scripts/preprocess/preprocess_libri-Speech-en.py \
		--input-dir data/raw/libriSpeech-en \
		--out-dir data/processed/libriSpeech-en/mels \
		--plot-samples 8
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

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
	parser = argparse.ArgumentParser(description="Prepare 80-band mel spectrograms and dataset for LibriSpeech EN")
	parser.add_argument("--input-dir", type=Path, default=Path("data/raw/libriSpeech-en"))
	parser.add_argument("--out-dir", type=Path, default=Path("data/processed/libriSpeech-en/mels"))
	parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Number of worker processes for preprocessing (1 disables multiprocessing)")
	parser.add_argument("--target-sr", type=int, default=22050)
	parser.add_argument("--n-mels", type=int, default=80)
	parser.add_argument("--n-fft", type=int, default=1024)
	parser.add_argument("--hop-length", type=int, default=256)
	parser.add_argument("--win-length", type=int, default=1024)
	parser.add_argument("--min-duration", type=float, default=0.3)
	parser.add_argument("--max-duration", type=float, default=20.0)
	parser.add_argument("--plot-samples", type=int, default=5, help="Number of mel images to save for validation (0 disables)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--overwrite", action="store_true")
	return parser.parse_args()


def resolve_input_root(input_dir: Path) -> Path:
	if input_dir.name == "LibriSpeech" and input_dir.exists():
		return input_dir
	candidate = input_dir / "LibriSpeech"
	if candidate.exists():
		return candidate
	return input_dir


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
		data = torch.load(out_path)
		return {
			"mel_path": str(out_path),
			"duration": data.get("duration", 0.0),
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
	mel = mel_transform(waveform).squeeze(0)

	torch.save(
		{
			"waveform": waveform,
			"mel": mel,
			"sr": sr,
			"duration": duration,
			"text": ex.text,
			"utt_id": ex.utt_id,
			"audio_path": ex.audio_path,
		},
		str(out_path),
	)
	return {
		"mel_path": str(out_path),
		"duration": duration,
		"text": ex.text,
		"utt_id": ex.utt_id,
	}


def _init_worker(
	input_root: Path,
	out_root: Path,
	target_sr: int,
	n_fft: int,
	hop_length: int,
	win_length: int,
	n_mels: int,
	overwrite: bool,
) -> None:
	global _WORKER_INPUT_ROOT, _WORKER_OUT_ROOT, _WORKER_TARGET_SR, _WORKER_OVERWRITE, _WORKER_MEL_TRANSFORM
	_WORKER_INPUT_ROOT = input_root
	_WORKER_OUT_ROOT = out_root
	_WORKER_TARGET_SR = target_sr
	_WORKER_OVERWRITE = overwrite
	_WORKER_MEL_TRANSFORM = make_mel_transform(target_sr, n_fft, hop_length, win_length, n_mels)
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


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 42) -> None:
	if plt is None:
		print("matplotlib not available; skipping plots")
		return
	if not manifest:
		print("No examples available for plotting")
		return

	rnd = random.Random(seed)
	samples = rnd.sample(manifest, min(n, len(manifest)))
	for i, row in enumerate(samples, start=1):
		data = torch.load(row["mel_path"]) if isinstance(row["mel_path"], str) else row["mel_path"]
		mel = data["mel"].numpy()
		fig, ax = plt.subplots(figsize=(8, 3))
		im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest")
		ax.set_title(f"{row.get('utt_id', Path(row['mel_path']).stem)} | {row.get('text', '')[:60]}")
		ax.set_ylabel("Mel bin")
		ax.set_xlabel("Frame")
		fig.colorbar(im, ax=ax)
		out_path = out_dir / f"mel_{i:02d}_{row.get('utt_id', Path(row['mel_path']).stem)}.png"
		fig.tight_layout()
		fig.savefig(out_path)
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
		mel_transform = make_mel_transform(args.target_sr, args.n_fft, args.hop_length, args.win_length, args.n_mels)
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
				args.n_fft,
				args.hop_length,
				args.win_length,
				args.n_mels,
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


if __name__ == "__main__":
	main()
