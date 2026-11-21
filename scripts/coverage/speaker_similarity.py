#!/usr/bin/env python3
"""
Was the reference voice copied? Speaker-embedding similarity of each generated audio.

Reads the results.csv written by scripts/eval/synth_check.py and adds two columns:
    sim_ref      cosine similarity between the generated audio and its reference audio
    sim_default  cosine similarity between the generated audio and the default voice
                 (mean embedding of utterances of the main speaker of the training set)
With --pool it also adds sim_spk, the similarity to the mean embedding of up to 20 utterances
of the reference speaker (the reference included), comparable to sim_default.
The voice was copied when sim_ref is high and above sim_default. Embeddings come from
Resemblyzer (a speaker-verification encoder trained on other data). All embeddings are also
saved to embeddings.npz in the eval folder, for scripts/coverage/voice_bits.py.

Example:
    python scripts/coverage/speaker_similarity.py --eval-dir exports/coverage/run/strength_1.0 \
        --refs data/processed/cov_n8_k32/refs --train data/processed/cov_n8_k32
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav


def embed(enc: VoiceEncoder, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    return enc.embed_utterance(preprocess_wav(audio.astype(np.float32), source_sr=sr))


def stored(path: str) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=True)["waveform"][0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True, help="Folder with results.csv and the generated .wav")
    ap.add_argument("--refs", type=Path, required=True, help="Processed folder of the references")
    ap.add_argument("--train", type=Path, required=True, help="Processed folder of the training set")
    ap.add_argument("--default-speaker", default=None, help="Default: the speaker with most training utterances")
    ap.add_argument("--n-default", type=int, default=20)
    ap.add_argument("--pool", type=Path, action="append", default=[],
                    help="Processed corpora with more utterances of the reference speakers (adds sim_spk)")
    a = ap.parse_args()

    enc = VoiceEncoder(verbose=False)
    with open(a.refs / "mels_metadata.csv", encoding="utf-8") as f:
        refs = list(csv.DictReader(f))
    ref_emb = {r["utt_id"]: embed(enc, stored(r["mel_path"])) for r in refs}
    spk_emb = {}
    if a.pool:
        by_speaker = {}
        for p in a.pool:
            if (p / "mels_metadata.csv").exists():
                with open(p / "mels_metadata.csv", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        by_speaker.setdefault(r["speaker"], []).append(r)
        for r in refs:
            others = sorted((x for x in by_speaker.get(r["speaker"], []) if x["utt_id"] != r["utt_id"]),
                            key=lambda x: x["utt_id"])[: a.n_default - 1]
            e = np.mean([ref_emb[r["utt_id"]]] + [embed(enc, stored(x["mel_path"])) for x in others], axis=0)
            spk_emb[r["utt_id"]] = e / np.linalg.norm(e)
    with open(a.train / "mels_metadata.csv", encoding="utf-8") as f:
        train = list(csv.DictReader(f))
    name = a.default_speaker or Counter(r["speaker"] for r in train).most_common(1)[0][0]
    default = [r for r in train if r["speaker"] == name][: a.n_default]
    assert default, f"no utterance of speaker {name} in {a.train}"
    default_emb = np.mean([embed(enc, stored(r["mel_path"])) for r in default], axis=0)
    default_emb /= np.linalg.norm(default_emb)

    path = a.eval_dir / "results.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_emb = []
    for r in rows:
        audio, _ = librosa.load(a.eval_dir / f"{r['sample']}.wav", sr=22050)
        e = embed(enc, audio)
        out_emb.append(e)
        r["sim_ref"] = round(float(e @ ref_emb[r["reference"]]), 3)
        r["sim_default"] = round(float(e @ default_emb), 3)
        if spk_emb:
            r["sim_spk"] = round(float(e @ spk_emb[r["reference"]]), 3)
    group = {r["utt_id"]: r.get("emotion", "") for r in refs}
    ref_ids = sorted(ref_emb)
    np.savez_compressed(
        a.eval_dir / "embeddings.npz",
        sample=np.array([r["sample"] for r in rows]), reference=np.array([r["reference"] for r in rows]),
        output=np.array(out_emb, dtype=np.float32),
        ref_id=np.array(ref_ids), ref_group=np.array([group[i] for i in ref_ids]),
        ref_utt=np.array([ref_emb[i] for i in ref_ids], dtype=np.float32),
        ref_spk=np.array([spk_emb.get(i, ref_emb[i]) for i in ref_ids], dtype=np.float32),
        default=default_emb.astype(np.float32),
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"sim_ref {np.mean([r['sim_ref'] for r in rows]):.3f} | sim_default {np.mean([r['sim_default'] for r in rows]):.3f} "
          f"over {len(rows)} samples")


if __name__ == "__main__":
    main()
