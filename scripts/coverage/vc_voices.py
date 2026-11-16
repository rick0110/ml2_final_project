#!/usr/bin/env python3
"""
Add real voices to a coverage training set with voice conversion.

Every utterance of the main speaker of the training set (the long, single-voice part) gets one
copy converted to the voice of a LibriTTS speaker with a pretrained voice-conversion model
(kNN-VC, see scripts/coverage/vc_cascade.py). The target speakers are outside every coverage
run: none of them is a trained speaker or a test reference, and only their audio is used, not
their transcriptions. With 2,000 utterances and 200 voices the decoder hears 200 more voices
with 10 sentences each.

Unlike a change of playback speed, the copies differ from each other the way real voices do.

Example:
    python scripts/coverage/vc_voices.py --data-dir data/processed/cov_n8_k8 \
        --libritts data/processed/LibriTTS --out data/processed/cov_n8_k8_vc
"""
import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_manifest import MAX_N, speaker_order  # noqa: E402
from vc_cascade import VC_SR, load_knn_vc, stored_16k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--libritts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="Where the converted audio is kept (default: <out>/audio); runs that share it convert once")
    ap.add_argument("--n-voices", type=int, default=200)
    ap.add_argument("--utts-per-voice", type=int, default=20, help="Utterances that describe each target voice")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.data_dir / "mels_metadata.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    main_speaker = Counter(r["speaker"] for r in rows).most_common(1)[0][0]
    source = [r for r in rows if r["speaker"] == main_speaker]

    by_speaker = defaultdict(list)
    with open(a.libritts / "mels_metadata.csv", encoding="utf-8") as f:
        for r in sorted(csv.DictReader(f), key=lambda r: r["utt_id"]):
            by_speaker[r["speaker"]].append(r)
    order, unseen = speaker_order(by_speaker)
    used = set(order[:MAX_N]) | set(unseen)  # trained speakers of the grid and the unseen test speakers
    voices = sorted(s for s in by_speaker if s not in used and len(by_speaker[s]) >= a.utts_per_voice)
    random.Random(a.seed).shuffle(voices)
    voices = voices[: a.n_voices]
    assert voices, "no LibriTTS speaker left for conversion"

    audio_dir = a.audio_dir or a.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    vc, matching = None, {}
    copies = []
    for i, row in enumerate(source):
        voice = voices[i % len(voices)]
        utt_id = f"{row['utt_id']}_vc{voice}"
        path = audio_dir / f"{utt_id}.pt"
        if not path.exists():
            vc = vc or load_knn_vc("cuda" if torch.cuda.is_available() else "cpu")
            if voice not in matching:
                matching[voice] = vc.get_matching_set([stored_16k(r["mel_path"]) for r in by_speaker[voice][: a.utts_per_voice]])
            sr = torch.load(row["mel_path"], map_location="cpu", weights_only=True).get("sr", 22050)
            with torch.inference_mode():
                converted = vc.match(vc.get_features(stored_16k(row["mel_path"])), matching[voice], topk=4).cpu().numpy()
            wave = librosa.resample(converted, orig_sr=VC_SR, target_sr=sr)
            torch.save({"waveform": torch.from_numpy(wave).unsqueeze(0), "sr": sr}, path)
        copies.append(dict(row, utt_id=utt_id, mel_path=str(path.resolve()), speaker=f"vc{voice}"))

    with open(a.out / "mels_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows + copies)
    shutil.copy(a.data_dir / "test_sentences.txt", a.out / "test_sentences.txt")
    if not (a.out / "refs").exists():
        shutil.copytree(a.data_dir / "refs", a.out / "refs")
    print(f"{len(rows)} utterances + {len(copies)} converted copies in {len(voices)} voices")


if __name__ == "__main__":
    main()
