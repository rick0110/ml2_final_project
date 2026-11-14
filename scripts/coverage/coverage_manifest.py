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
    test_sentences.txt       LJSpeech sentences outside the training subset, the same in every
                             run: the text the model must read in each reference voice
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
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


MAX_N = 32          # largest number of training speakers in any run
HELDOUT = 4         # sentences per speaker kept as references
N_TEST = 25         # test sentences
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


# Fixed so that every run, on any machine, uses the same voices. The first eight were picked
# to alternate low and high voices (2 speakers = one of each); the last four are never trained on.
FIRST_SEEN = ["3486", "4137", "8088", "7178", "2952", "5163", "6209", "6147"]
UNSEEN = ["4297", "696", "6415", "7800"]


def speaker_order(by_speaker: Dict[str, List[dict]]) -> List[str]:
    """FIRST_SEEN, then the speakers with most sentences (up to MAX_N in total), then UNSEEN."""
    pinned = set(FIRST_SEEN + UNSEEN)
    rest = sorted((s for s in by_speaker if s not in pinned), key=lambda s: (-len(by_speaker[s]), s))
    return FIRST_SEEN + sorted(rest[: MAX_N - len(FIRST_SEEN)]) + UNSEEN


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
    lj, lj_rest = lj[: a.lj_size], lj[a.lj_size:]
    # Plain sentences only: digits and abbreviations make the word error rate depend on the
    # recognizer's spelling, not on what was said
    test = [r["text"].strip() for r in lj_rest
            if 6 <= len(r["text"].split()) <= 14 and re.fullmatch(r"[A-Za-z ,']+[.?]", r["text"].strip())
            and not re.search(r"\b[A-Z]{2,}\b|\b(Mr|Mrs|Dr|St|No)\b", r["text"])][:N_TEST]

    by_speaker: Dict[str, List[dict]] = defaultdict(list)
    for r in sorted(read(a.libritts), key=lambda r: r["utt_id"]):
        by_speaker[r["speaker"]].append(r)
    for rows in by_speaker.values():
        random.Random(1).shuffle(rows)
    order = speaker_order(by_speaker)
    seen, unseen = order[: a.n], order[MAX_N:]

    train = list(lj)
    for s in seen:
        chosen = by_speaker[s][HELDOUT: HELDOUT + a.k]
        if len(chosen) < a.k:
            print(f"warning: speaker {s} has only {len(chosen)} sentences for K={a.k}")
        train += chosen
    write(a.out, train)
    (a.out / "test_sentences.txt").write_text("\n".join(test) + "\n", encoding="utf-8")

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
