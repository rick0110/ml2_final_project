#!/usr/bin/env python3
"""
Second stage of a cascade: convert already generated audio to the voice of its reference.

Takes the folder written by scripts/eval/synth_check.py (one .wav per sample and results.csv)
and passes every sample through a pretrained voice-conversion model (kNN-VC: WavLM features of
the sample are replaced by their nearest neighbours among the features of the target speaker,
then vocoded). The target is described by --n-refs utterances of the reference speaker: the
reference itself plus other utterances of the same speaker taken from --pool. The converted
audio is transcribed again, so the output folder has the same layout as the input one and
scripts/coverage/speaker_similarity.py runs on it unchanged.

The voice-conversion model is never trained here and needs no transcription.

Example:
    python scripts/coverage/vc_cascade.py --eval-dir out/run/step_2500/strength_0.0 \
        --out-dir out/run/step_2500/strength_0.0_vc8 --refs data/processed/cov_n8_k8/refs \
        --pool data/processed/LibriTTS --n-refs 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from synth_check import prosody, wer  # noqa: E402

VC_SR = 16000


def load_knn_vc(device: str):
    return torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=device)


def stored_16k(path: str) -> torch.Tensor:
    sample = torch.load(path, map_location="cpu", weights_only=True)
    wave = sample["waveform"][0].numpy()
    return torch.from_numpy(librosa.resample(wave, orig_sr=sample.get("sr", 22050), target_sr=VC_SR))[None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True, help="Processed folder of the references")
    ap.add_argument("--pool", type=Path, action="append", default=[],
                    help="Processed corpora with more utterances of the reference speakers (repeatable)")
    ap.add_argument("--n-refs", type=int, default=1, help="Utterances of the reference speaker given to the converter")
    ap.add_argument("--language", default="en")
    ap.add_argument("--whisper", default="small")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    vc = load_knn_vc(device)

    with open(a.refs / "mels_metadata.csv", encoding="utf-8") as f:
        refs = {r["utt_id"]: r for r in csv.DictReader(f)}
    pool = defaultdict(list)
    for p in a.pool:
        if not (p / "mels_metadata.csv").exists():
            continue
        with open(p / "mels_metadata.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pool[r["speaker"]].append(r)
    targets, n_used = {}, {}
    for utt_id, ref in refs.items():
        others = sorted((r for r in pool[ref["speaker"]] if r["utt_id"] != utt_id), key=lambda r: r["utt_id"])
        chosen = [ref] + others[: a.n_refs - 1]
        targets[utt_id] = vc.get_matching_set([stored_16k(r["mel_path"]) for r in chosen])
        n_used[utt_id] = len(chosen)

    from faster_whisper import WhisperModel
    try:
        asr = WhisperModel(a.whisper, device=device, compute_type="float16" if device == "cuda" else "int8")
        list(asr.transcribe(np.zeros(16000, dtype=np.float32), language=a.language)[0])
    except Exception as e:
        print(f"whisper on {device} failed ({e}); using the CPU")
        asr = WhisperModel(a.whisper, device="cpu", compute_type="int8")

    with open(a.eval_dir / "results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_rows = []
    for r in rows:
        audio, _ = librosa.load(a.eval_dir / f"{r['sample']}.wav", sr=VC_SR)
        with torch.inference_mode():
            query = vc.get_features(torch.from_numpy(audio)[None])
            converted = vc.match(query, targets[r["reference"]], topk=4).cpu().numpy()
        heard = " ".join(s.text.strip() for s in asr.transcribe(converted, language=a.language, beam_size=5)[0])
        audio22 = librosa.resample(converted, orig_sr=VC_SR, target_sr=22050)
        sf.write(a.out_dir / f"{r['sample']}.wav", audio22, 22050)
        pros = prosody(audio22, len(r["text"]))
        r = {k: v for k, v in r.items() if k not in ("sim_ref", "sim_default")}
        r.update(heard=heard, wer=round(wer(r["text"], heard), 3), duration_s=round(len(audio22) / 22050, 2),
                 f0_hz=round(pros["f0_hz"], 1), f0_std_hz=round(pros["f0_std_hz"], 1),
                 chars_per_s=round(pros["chars_per_s"], 2), vc_refs=n_used[r["reference"]])
        out_rows.append(r)

    with open(a.out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"MEAN WER {np.mean([r['wer'] for r in out_rows]):.3f} over {len(out_rows)} converted samples")


if __name__ == "__main__":
    main()
