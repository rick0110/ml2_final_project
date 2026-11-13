#!/usr/bin/env python3
"""
Build the training set of one run of the coverage experiment.

Question: how many distinct sentences per voice ("style") does the model need before it
can copy a voice from a reference AND still read new text in it?

Each run trains on a fixed subset of LJSpeech (one voice, wide text coverage) plus N
LibriTTS speakers with K sentences each. N and K are the two factors. The selections are
nested (the speakers of N=2 are inside N=8, the sentences of K=8 are inside K=32), so
runs differ only in what was added.

Outputs in --out:
    mels_metadata.csv        training manifest (LJ subset + N speakers x K sentences)
    refs/mels_metadata.csv   style references, never used in training: held-out sentences
                             of trained speakers ("seen") and of speakers left out of
                             every run ("unseen"); the column `emotion` holds seen/unseen

Example:
    python scripts/coverage/coverage_manifest.py --lj data/processed/LJSpeech \
        --libritts data/processed/LibriTTS --n 8 --k 32 --out data/processed/cov_n8_k32
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch

MAX_N = 32          # largest number of training speakers in any run
N_UNSEEN = 4        # speakers kept out of every run
HELDOUT = 4         # sentences per speaker kept as references
FIELDS = ["utt_id", "mel_path", "text", "speaker", "emotion", "duration"]


def read(path: Path) -> List[dict]:
    with open(path / "mels_metadata.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: List[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "mels_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def median_f0(rows: List[dict]) -> float:
    vals = []
    for r in rows[:2]:
        wav = torch.load(r["mel_path"], map_location="cpu", weights_only=True)["waveform"][0].numpy()
        f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=450, sr=22050, frame_length=1024, hop_length=256)
        if voiced.any():
            vals.append(float(np.nanmedian(f0[voiced])))
    return float(np.mean(vals)) if vals else 0.0


def speaker_order(libritts: Path, by_speaker: Dict[str, List[dict]]) -> List[str]:
    """The MAX_N + N_UNSEEN speakers with most sentences, ordered so that every prefix mixes low and high voices."""
    cache = libritts / "coverage_speakers.json"
    if cache.exists():
        return json.loads(cache.read_text())
    top = sorted(by_speaker, key=lambda s: (-len(by_speaker[s]), s))[: MAX_N + N_UNSEEN]
    by_pitch = sorted(top, key=lambda s: median_f0(by_speaker[s]))
    order: List[str] = []
    while by_pitch:  # lowest, highest, 2nd lowest, 2nd highest, ...
        order.append(by_pitch.pop(0))
        if by_pitch:
            order.append(by_pitch.pop(-1))
    cache.write_text(json.dumps(order))
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lj", type=Path, required=True)
    ap.add_argument("--libritts", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True, help="Number of LibriTTS speakers in training")
    ap.add_argument("--k", type=int, required=True, help="Sentences per LibriTTS speaker in training")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lj-size", type=int, default=2000)
    a = ap.parse_args()
    assert a.n <= MAX_N

    rng = random.Random(0)
    lj = sorted(read(a.lj), key=lambda r: r["utt_id"])
    rng.shuffle(lj)
    lj = lj[: a.lj_size]

    by_speaker: Dict[str, List[dict]] = defaultdict(list)
    for r in sorted(read(a.libritts), key=lambda r: r["utt_id"]):
        by_speaker[r["speaker"]].append(r)
    for rows in by_speaker.values():
        random.Random(1).shuffle(rows)
    order = speaker_order(a.libritts, by_speaker)
    seen, unseen = order[: a.n], order[MAX_N:]

    train = list(lj)
    for s in seen:
        chosen = by_speaker[s][HELDOUT: HELDOUT + a.k]
        if len(chosen) < a.k:
            print(f"warning: speaker {s} has only {len(chosen)} sentences for K={a.k}")
        train += chosen
    write(a.out, train)

    refs = []
    per_seen = max(1, 8 // a.n)  # 8 seen references in every run
    for s in seen[:8]:
        refs += [dict(r, emotion="seen") for r in by_speaker[s][:per_seen]]
    for s in unseen:
        refs += [dict(r, emotion="unseen") for r in by_speaker[s][:1]]
    write(a.out / "refs", refs)
    print(f"N={a.n} K={a.k}: {len(train)} training utterances ({len(train) - len(lj)} LibriTTS), {len(refs)} references")


if __name__ == "__main__":
    main()
