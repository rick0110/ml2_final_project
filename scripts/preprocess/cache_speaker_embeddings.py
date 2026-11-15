#!/usr/bin/env python3
"""
Compute the pretrained speaker embedding of every utterance in a manifest.

Writes <data-dir>/speaker_embeddings.pt, a dictionary from utterance id to a unit-norm
vector, read by train.py --speaker-embeddings.

Example:
    python scripts/preprocess/cache_speaker_embeddings.py --data-dir data/processed/cov_n8_k8
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.speaker_encoder import load_speaker_encoder, speaker_embedding  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    a = ap.parse_args()
    out = a.data_dir / "speaker_embeddings.pt"
    with open(a.data_dir / "mels_metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done = torch.load(out, map_location="cpu") if out.exists() else {}
    encoder = load_speaker_encoder(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for row in rows:
        utt_id = row.get("utt_id", Path(row["mel_path"]).stem)
        if utt_id in done:
            continue
        sample = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
        done[utt_id] = speaker_embedding(encoder, sample["waveform"], sample.get("sr", 16000))
    torch.save(done, out)
    print(f"{len(done)} speaker embeddings in {out}")


if __name__ == "__main__":
    main()
