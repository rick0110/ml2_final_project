#!/usr/bin/env python3
"""
Build the training manifest for one corpus (LJSpeech, TTS-Portuguese Corpus, VERBO or LibriTTS).

For every utterance it writes a .pt file with the mono waveform resampled to
22.05 kHz, and one row in <out-dir>/mels_metadata.csv. The mel-spectrogram is
computed later by the data loader, so there is a single mel implementation.

It only needs soundfile and librosa (no torchaudio).

Examples:
    python scripts/preprocess/make_manifest.py --corpus ljspeech \
        --input-dir data/raw/LJSpeech-1.1 --out-dir data/processed/LJSpeech
    python scripts/preprocess/make_manifest.py --corpus tts_portuguese \
        --input-dir data/raw/tts-portuguese-Corpora/TTS-Portuguese-Corpus --out-dir data/processed/TTS-Portuguese
    python scripts/preprocess/make_manifest.py --corpus verbo \
        --input-dir data/raw/VERBO-Dataset --out-dir data/processed/VERBO
"""
from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

TARGET_SR = 22050

VERBO_EMOTIONS: Dict[str, str] = {
    "neu": "neutro", "ale": "alegria", "des": "desgosto", "med": "medo",
    "rai": "raiva", "sur": "surpresa", "tri": "tristeza",
}
# Phrases from Table 1 of the VERBO paper (Torres Neto et al., 2018)
VERBO_PHRASES: Dict[str, str] = {
    "l1": "Os bombeiros estão equipados com uma arma.",
    "l2": "No próximo outono, Antônio vai a Minas em quinze de outubro.",
    "l3": "Agora vou pôr a camiseta e sair para uma caminhada.",
    "l4": "Um momento depois, ele caminhou... e tropeçou.",
    "l5": "Eu queria o número de telefone do seu João.",
    "ns1": "A casa forte quer com pão.",
    "ns2": "A força está para cima e alho vermelho.",
    "ns3": "O gato está rolando na pêra.",
    "ns4": "Salada de massa pata de carneiro amendoim.",
    "ns5": "Um quarenta e três vinte e sete noventa e cinco mil.",
    "q1": "Sábado à noite, o que vai fazer?",
    "q2": "Você vai trazer aquela coisa com você?",
    "s1": "Os operários levantam cedo.",
    "s2": "A cachoeira faz muito barulho.",
}
VERBO_NAME = re.compile(r"^([a-z]+)-([mf]\d+)-([a-z0-9]+)$")

Item = Tuple[str, Path, str, str, str]  # utt_id, wav path, text, speaker, emotion


def ljspeech_items(root: Path) -> Iterator[Item]:
    with open(root / "metadata.csv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 3:
                yield parts[0], root / "wavs" / f"{parts[0]}.wav", parts[2], "lj", ""


def tts_portuguese_items(root: Path) -> Iterator[Item]:
    with open(root / "texts.csv", encoding="utf-8") as f:
        for line in f:
            if "==" in line:
                rel, text = line.rstrip("\n").split("==", 1)
                yield Path(rel).stem, root / rel, text.strip(), "edresson", ""


def verbo_items(root: Path) -> Iterator[Item]:
    for wav in sorted(root.rglob("*.wav")):
        m = VERBO_NAME.match(wav.stem)
        if m and m.group(1) in VERBO_EMOTIONS and m.group(3) in VERBO_PHRASES:
            yield wav.stem, wav, VERBO_PHRASES[m.group(3)], m.group(2), VERBO_EMOTIONS[m.group(1)]


def libritts_items(root: Path) -> Iterator[Item]:
    """LibriTTS layout: <root>/<speaker>/<chapter>/<utt>.wav with <utt>.normalized.txt beside it."""
    for wav in sorted(root.rglob("*.wav")):
        txt = wav.with_suffix(".normalized.txt")
        if txt.exists():
            yield wav.stem, wav, txt.read_text(encoding="utf-8").strip(), wav.parts[-3], ""


CORPORA = {"ljspeech": ljspeech_items, "tts_portuguese": tts_portuguese_items, "verbo": verbo_items, "libritts": libritts_items}


def process(args: Tuple[Item, Path, float, float, bool]) -> Optional[Dict[str, str]]:
    (utt_id, wav_path, text, speaker, emotion), out_dir, min_dur, max_dur, trim = args
    out_path = out_dir / "mels" / f"{utt_id}.pt"
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:  # unreadable file: skip it, keep going
        print(f"skip {wav_path}: {exc}")
        return None
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=40)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    duration = len(audio) / TARGET_SR
    if not (min_dur <= duration <= max_dur) or not text:
        return None
    torch.save({"waveform": torch.from_numpy(audio).unsqueeze(0), "sr": TARGET_SR}, out_path)
    return {"utt_id": utt_id, "mel_path": str(out_path), "text": text, "speaker": speaker,
            "emotion": emotion, "duration": f"{duration:.3f}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=11.0)
    ap.add_argument("--speakers", nargs="*", default=None, help="Keep only these speakers (VERBO: f1 m2 ...)")
    ap.add_argument("--no-trim", action="store_true", help="Do not trim leading/trailing silence")
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()

    (a.out_dir / "mels").mkdir(parents=True, exist_ok=True)
    items: List[Item] = [it for it in CORPORA[a.corpus](a.input_dir) if not a.speakers or it[3] in a.speakers]
    jobs = [(it, a.out_dir, a.min_duration, a.max_duration, not a.no_trim) for it in items]
    with ProcessPoolExecutor(a.num_workers) as pool:
        rows = [r for r in pool.map(process, jobs, chunksize=64) if r is not None]
    with open(a.out_dir / "mels_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "mel_path", "text", "speaker", "emotion", "duration"])
        w.writeheader()
        w.writerows(rows)
    hours = sum(float(r["duration"]) for r in rows) / 3600
    print(f"{a.corpus}: kept {len(rows)} of {len(items)} utterances, {hours:.2f} h -> {a.out_dir}")


if __name__ == "__main__":
    main()
