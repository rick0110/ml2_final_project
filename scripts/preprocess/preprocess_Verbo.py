#!/usr/bin/env python3
"""
Preprocess VERBO raw data into Tacotron2-VAE compatible mel-spectrogram tensors.

VERBO (Voice Emotion Recognition dataBase in pOrtuguese language, Torres Neto
et al. 2018) stores audios as ``Audios/<speaker>/<emotion>-<speaker>-<phrase>.wav``.
The filename itself encodes the ground truth: the emotion the actor performed,
the speaker who recorded it, and the id of one of the 14 fixed phrases defined
by the paper's linguistic material (Table 1). This script parses that
convention to relate every audio file to its Portuguese transcription and its
emotion label, then computes the matching mel-spectrogram.

Responsibilities:
    - Scan the VERBO raw root (`Audios/<speaker>/*.wav`).
    - Parse each filename into (emotion, speaker, phrase) using the dataset's
      naming convention.
    - Look up the phrase transcription (PT/EN) and emotion label from the
      tables published in the VERBO paper.
    - Load audio, convert to mono, and resample to the target sampling rate.
    - Compute Tacotron2-compatible 80-bin log-mel spectrograms.
    - Filter examples by duration.
    - Save per-utterance `.pt` tensors.
    - Write a metadata CSV relating each mel tensor to its transcription,
      emotion and speaker.

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
import torch.nn.functional as F
import torchaudio
import librosa.filters
import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# Emotion code (as used in filenames / MFCC feature folders) -> (Portuguese
# label, English label, class id). Class ids follow "neutral first, then the
# six Ekman basic emotions" so they are stable across the rest of the pipeline.
EMOTION_MAP: Dict[str, Tuple[str, str, int]] = {
    "neu": ("neutro", "neutral", 0),
    "ale": ("alegria", "happiness", 1),
    "des": ("desgosto", "disgust", 2),
    "med": ("medo", "fear", 3),
    "rai": ("raiva", "anger", 4),
    "sur": ("surpresa", "surprise", 5),
    "tri": ("tristeza", "sadness", 6),
}

# Phrase id -> (category, Portuguese phrase, English gloss), transcribed from
# Table 1 of the VERBO paper (Torres Neto et al., 2018).
PHRASE_MAP: Dict[str, Tuple[str, str, str]] = {
    "l1": ("long", "Os bombeiros estão equipados com uma arma.", "Firefighters are equipped with a gun."),
    "l2": ("long", "No próximo outono, Antônio vai a Minas em quinze de outubro.", "The next fall Antônio goes to Minas on October fifteenth."),
    "l3": ("long", "Agora vou pôr a camiseta e sair para uma caminhada.", "Now I'm going to put on my T-shirt and go for a walk."),
    "l4": ("long", "Um momento depois, ele caminhou... e tropeçou.", "A moment later he walked... and stumbled."),
    "l5": ("long", "Eu queria o número de telefone do seu João.", "I would like the telephone number of Mr. João."),
    "ns1": ("nonsense", "A casa forte quer com pão.", "The strong house wants with bread."),
    "ns2": ("nonsense", "A força está para cima e alho vermelho.", "The force is up and red garlic."),
    "ns3": ("nonsense", "O gato está rolando na pêra.", "The cat is rolling in the pear."),
    "ns4": ("nonsense", "Salada de massa pata de carneiro amendoim.", "Pasta salad leg of lamb peanut."),
    "ns5": ("nonsense", "Um quarenta e três vinte e sete noventa e cinco mil.", "One forty-three twenty-seven ninety-five thousand."),
    "q1": ("question", "Sábado à noite, o que vai fazer?", "Saturday night, what are you going to do?"),
    "q2": ("question", "Você vai trazer aquela coisa com você?", "Will you bring that thing with you?"),
    "s1": ("short", "Os operários levantam cedo.", "Workers get up early."),
    "s2": ("short", "A cachoeira faz muito barulho.", "The waterfall makes a lot of noise."),
}

# ale-f1-s1.wav -> emotion="ale", speaker="f1", phrase="s1"
FILENAME_RE = re.compile(r"^([a-z]+)-([mf]\d+)-([a-z0-9]+)$")

_WORKER_INPUT_ROOT: Optional[Path] = None
_WORKER_OUT_ROOT: Optional[Path] = None
_WORKER_TARGET_SR: Optional[int] = None
_WORKER_OVERWRITE: bool = False
_WORKER_MEL_TRANSFORM: Optional[torch.nn.Module] = None


@dataclass
class Example:
    """
    Metadata for a single discovered VERBO utterance.

    Attributes:
        audio_path (str): Path to the audio file, relative to the input root.
        text (str): Portuguese transcription of the phrase spoken.
        text_en (str): English gloss of the phrase, for reference.
        duration (float): Utterance duration in seconds (filled after loading).
        utt_id (str): Unique identifier (the file stem, e.g. "ale-f1-s1").
        speaker (str): Speaker id (e.g. "f1", "m4").
        gender (str): "female" or "male", derived from the speaker id.
        phrase_id (str): Phrase id (e.g. "s1", "ns3", "l2").
        phrase_category (str): One of "short", "long", "question", "nonsense".
        emotion (str): English emotion label (e.g. "happiness").
        emotion_pt (str): Portuguese emotion label (e.g. "alegria").
        emotion_id (int): Stable integer class id for the emotion.
    """
    audio_path: str
    text: str
    text_en: str
    duration: float
    utt_id: str
    speaker: str
    gender: str
    phrase_id: str
    phrase_category: str
    emotion: str
    emotion_pt: str
    emotion_id: int


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Relate VERBO audios to their transcription/emotion and prepare Tacotron2-compatible mel spectrograms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/verbo"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/VERBO/mels"))
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))

    # Tacotron 2 standard params
    parser.add_argument("--target-sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--filter-length", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--log-zero-guard-value", type=float, default=1e-5)

    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


class Tacotron2MelTransform(torch.nn.Module):
    """
    Tacotron2-compatible log-mel extraction (STFT -> Mel filterbank -> log).

    Inputs:
        waveform:
            Shape (B, S) or (S,)

    Outputs:
        log_mel:
            Shape (B, n_mels, T)
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
        clip_val: float = 1e-5,
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.clip_val = clip_val

        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, S)

        p = int((self.filter_length - self.hop_length) / 2)
        waveform = F.pad(waveform.unsqueeze(1), (p, p), mode="reflect").squeeze(1)

        stft_out = torch.stft(
            waveform, n_fft=self.filter_length, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, center=False,
            pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
        )
        magnitudes = torch.abs(stft_out)
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=self.clip_val))
        return mel_output


def parse_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a VERBO filename stem into (emotion_code, speaker, phrase_id).

    Args:
        stem (str): File stem, e.g. "ale-f1-s1".

    Returns:
        Optional[Tuple[str, str, str]]: (emotion_code, speaker, phrase_id), or
            None if the stem does not match the VERBO naming convention.
    """
    match = FILENAME_RE.match(stem)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def discover_examples(input_root: Path) -> List[Example]:
    """
    Scan the VERBO `Audios/` tree and relate every file to its transcription
    and emotion using the dataset's filename convention.

    Args:
        input_root (Path): VERBO dataset root (contains an `Audios/` folder).

    Returns:
        List[Example]: Discovered metadata, one entry per parsable audio file.
    """
    audios_dir = input_root / "Audios"
    if not audios_dir.exists():
        warning(f"'Audios' directory not found in {input_root}")
        return []

    examples: List[Example] = []
    for audio_path in sorted(audios_dir.rglob("*.wav")):
        stem = audio_path.stem
        parsed = parse_filename(stem)
        if parsed is None:
            warning(f"Filename does not match VERBO convention, skipping: {audio_path}")
            continue

        emotion_code, speaker, phrase_id = parsed

        emotion_entry = EMOTION_MAP.get(emotion_code)
        if emotion_entry is None:
            warning(f"Unknown emotion code '{emotion_code}' in {audio_path}, skipping")
            continue

        phrase_entry = PHRASE_MAP.get(phrase_id)
        if phrase_entry is None:
            warning(f"Unknown phrase id '{phrase_id}' in {audio_path}, skipping")
            continue

        emotion_pt, emotion_en, emotion_id = emotion_entry
        phrase_category, text_pt, text_en = phrase_entry
        gender = "female" if speaker.startswith("f") else "male"

        examples.append(
            Example(
                audio_path=str(audio_path.relative_to(input_root)),
                text=text_pt,
                text_en=text_en,
                duration=0.0,
                utt_id=stem,
                speaker=speaker,
                gender=gender,
                phrase_id=phrase_id,
                phrase_category=phrase_category,
                emotion=emotion_en,
                emotion_pt=emotion_pt,
                emotion_id=emotion_id,
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
    Load a single VERBO audio file, compute its mel-spectrogram, and save it
    together with its transcription/emotion metadata.

    Args:
        ex (Example): Example metadata.
        input_root (Path): VERBO dataset root.
        out_root (Path): Output root for `.pt` tensors.
        mel_transform (torch.nn.Module): Mel transform.
        target_sr (int): Target sampling rate.
        overwrite (bool): Recompute even if the `.pt` file already exists.

    Returns:
        Optional[Dict[str, Any]]: Row for the metadata manifest, or None if
            the audio file could not be found.
    """
    audio_path = input_root / Path(ex.audio_path)
    if not audio_path.exists():
        return None

    out_path = out_root / f"{ex.utt_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        data: Dict[str, Any] = torch.load(out_path, map_location="cpu", weights_only=False)
        return {**ex.__dict__, "mel_path": str(out_path), "duration": float(data.get("duration", 0.0))}

    waveform: Tensor
    sr: int
    waveform, sr = torchaudio.load(str(audio_path))  # (channels, S)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, S)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    duration = waveform.shape[-1] / sr

    with torch.no_grad():
        log_mel: Tensor = mel_transform(waveform)  # (1, n_mels, T)

    torch.save(
        {
            "waveform": waveform.cpu(),
            "mel": log_mel.cpu(),
            "sr": sr,
            **ex.__dict__,
            "duration": float(duration),
        },
        str(out_path),
    )

    return {**ex.__dict__, "mel_path": str(out_path), "duration": float(duration)}


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
        sampling_rate=target_sr, n_mel_channels=n_mels, filter_length=filter_length,
        win_length=win_length, hop_length=hop_length, mel_fmin=f_min,
        mel_fmax=f_max, clip_val=log_zero_guard_value,
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
        ex, _WORKER_INPUT_ROOT, _WORKER_OUT_ROOT, _WORKER_MEL_TRANSFORM, _WORKER_TARGET_SR,
        overwrite=_WORKER_OVERWRITE,
    )


def plot_mels(manifest: List[Dict[str, Any]], out_dir: Path, n: int = 5, seed: int = 1234) -> None:
    """
    Visualize a random sample of mel spectrograms, annotated with their
    emotion and transcription, for a quick sanity check.

    Args:
        manifest (List[Dict[str, Any]]): Processed example rows.
        out_dir (Path): Output directory for figures.
        n (int): Number of samples to plot.
        seed (int): Random seed.
    """
    if plt is None or not manifest:
        return
    rnd = random.Random(seed)
    samples = rnd.sample(manifest, min(n, len(manifest)))
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(samples, start=1):
        data = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
        mel = data["mel"].squeeze(0).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(f"{row['utt_id']} - {row['emotion']} ({row['emotion_pt']}) - \"{row['text']}\"")
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Frame")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"mel_{i:02d}_{row['utt_id']}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_root: Path = args.input_dir
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    examples = discover_examples(input_root)
    if not examples:
        raise RuntimeError(f"No VERBO examples discovered under {input_root}")

    manifest: List[Dict[str, Any]] = []

    if args.num_workers <= 1:
        mel_transform = Tacotron2MelTransform(
            sampling_rate=args.target_sr, n_mel_channels=args.n_mels, filter_length=args.filter_length,
            win_length=args.win_length, hop_length=args.hop_length, mel_fmin=args.f_min,
            mel_fmax=args.f_max, clip_val=args.log_zero_guard_value,
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
                input_root, out_root, args.target_sr, args.n_mels, args.filter_length,
                args.win_length, args.hop_length, args.f_min, args.f_max,
                args.log_zero_guard_value, args.overwrite,
            ),
        ) as executor:
            chunksize = max(1, len(examples) // max(args.num_workers * 8, 1))
            for ex, res in zip(examples, tqdm.tqdm(
                executor.map(_process_example_worker, examples, chunksize=chunksize),
                total=len(examples), desc="Processing examples",
            )):
                if res is None:
                    warning(f"Failed to process example {ex.utt_id}; skipping")
                    continue
                if res["duration"] < args.min_duration or res["duration"] > args.max_duration:
                    warning(f"Example {ex.utt_id} duration {res['duration']:.2f}s out of bounds; skipping")
                    continue
                manifest.append(res)

    if not manifest:
        raise RuntimeError("No examples were kept after duration filtering.")

    manifest_csv = out_root.parent / "verbo_metadata.csv"
    fieldnames = [
        "mel_path", "duration", "text", "text_en", "utt_id", "speaker", "gender",
        "phrase_id", "phrase_category", "emotion", "emotion_pt", "emotion_id",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    if args.plot_samples > 0:
        plot_mels(manifest, out_root.parent / "figures", n=args.plot_samples, seed=args.seed)

    print(f"Prepared {len(manifest)} mel tensors in {out_root}")
    print(f"Metadata written to {manifest_csv}")


if __name__ == "__main__":
    main()
