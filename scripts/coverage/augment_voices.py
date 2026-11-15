#!/usr/bin/env python3
"""
Add pseudo-speakers to a coverage training set by changing the playback speed of each utterance.

Playing a recording faster by a factor r raises the pitch and the formants by r (and shortens
it by r): the same sentence in what sounds like another voice. Each utterance of the manifest
gets one such copy, with r drawn at random, so the decoder hears every training sentence in
two voices and many more (text, voice) pairs than the recordings alone give.

The output directory holds the original rows plus the copies (speaker = <speaker>_v<step>,
step = the shift in semitones), the same test sentences and the same references.

Example:
    python scripts/coverage/augment_voices.py --data-dir data/processed/cov_n8_k128 \
        --out data/processed/cov_n8_k128_aug
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import librosa
import torch

MAX_DURATION = 10.5  # seconds; a slowed-down copy longer than this is sped up instead


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-shift", type=float, default=1.5, help="Smallest shift, in semitones")
    ap.add_argument("--max-shift", type=float, default=4.0, help="Largest shift, in semitones")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.data_dir / "mels_metadata.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    audio_dir = a.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    copies = []
    for row in rows:
        shift = rng.uniform(a.min_shift, a.max_shift) * rng.choice([-1, 1])
        if shift < 0 and float(row["duration"]) / 2 ** (shift / 12) > MAX_DURATION:
            shift = -shift
        rate = 2 ** (shift / 12)
        utt_id = f"{row['utt_id']}_v{shift:+.1f}"
        path = audio_dir / f"{utt_id}.pt"
        if not path.exists():
            sample = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
            wave = sample["waveform"].squeeze(0).numpy()
            sr = sample.get("sr", 16000)
            # Resampling to sr / rate and keeping the label sr plays the audio `rate` times faster
            warped = librosa.resample(wave, orig_sr=sr, target_sr=int(round(sr / rate)))
            torch.save({"waveform": torch.from_numpy(warped).unsqueeze(0), "sr": sr}, path)
        copies.append(dict(row, utt_id=utt_id, mel_path=str(path.resolve()),
                           speaker=f"{row['speaker']}_v{round(shift):+d}",
                           duration=f"{float(row['duration']) / rate:.3f}"))

    with open(a.out / "mels_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows + copies)
    shutil.copy(a.data_dir / "test_sentences.txt", a.out / "test_sentences.txt")
    if not (a.out / "refs").exists():
        shutil.copytree(a.data_dir / "refs", a.out / "refs")
    print(f"{len(rows)} utterances + {len(copies)} copies in {len({c['speaker'] for c in copies})} pseudo-speakers")


if __name__ == "__main__":
    main()
