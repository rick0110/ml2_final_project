#!/usr/bin/env python3

"""Preprocess LibriSpeech raw data into mel-spectrogram tensors.

What this script does:
- Scans a LibriSpeech raw root for transcript files and matching audio
- Reads utterance text from `*.trans.txt`
- Loads audio, converts to mono, and resamples to 22050 Hz
- Computes 80-band mel-spectrograms (log-scaled)
- Filters examples by duration
- Saves per-utterance `.pt` tensors and a manifest CSV
- Provides a `MelDataset` and `collate_fn` for PyTorch DataLoader use
- Optionally plots a few mel spectrograms for validation

Expected LibriSpeech layout:
	<input-dir>/<split>/<speaker>/<chapter>/<utt-id>.flac
	<input-dir>/<split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

Example:
	python scripts/preprocess/preprocess_libriSpeech-pt.py \
		--input-dir data/raw/LibriSpeech \
		--out-dir data/processed/LibriSpeech/mels \
		--plot-samples 8
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

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
	utt_id: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Prepare 80-band mel spectrograms and dataset for LibriSpeech")
	parser.add_argument("--input-dir", type=Path, default=Path("data/raw/LibriSpeech"))
	parser.add_argument("--out-dir", type=Path, default=Path("data/processed/LibriSpeech/mels"))
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


def find_audio_file(transcript_path: Path, utt_id: str) -> Optional[Path]:
	candidate_stems = [
		transcript_path.parent / f"{utt_id}.wav"
	]
	for candidate in candidate_stems:
		if candidate.exists():
			return candidate

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
			match = re.match(r"^([^a-zA-Z]*)([a-zA-Z].*)", line)
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

	input_root = args.input_dir
	if not input_root.exists():
		raise FileNotFoundError(f"Input directory does not exist: {input_root}")

	out_root = args.out_dir
	out_root.mkdir(parents=True, exist_ok=True)

	mel_transform = make_mel_transform(args.target_sr, args.n_fft, args.hop_length, args.win_length, args.n_mels)
	examples = discover_examples(input_root)

	manifest: List[Dict[str, Any]] = []
	for ex in tqdm.tqdm(examples, desc="Processing examples"):
		res = process_example(ex, input_root, out_root, mel_transform, args.target_sr, overwrite=args.overwrite)
		if res is None:
			continue
		if res["duration"] < args.min_duration or res["duration"] > args.max_duration:
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
