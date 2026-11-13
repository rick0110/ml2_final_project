#!/usr/bin/env python3
"""
Encode utterances with the reference encoder of a checkpoint and save the latent means.

Output: one .npz with `mu` (N, L), `logvar` (N, L) and the string arrays `utt_id`,
`speaker`, `emotion`, `text`, `source` (the manifest each row came from). The analysis
of the latent space (probes, distances, projections) is done offline on this file.

Example:
    python scripts/eval/latent_dump.py --checkpoint ckpt/final --out latents.npz \
        --manifest data/processed/VERBO --manifest data/processed/TTS-Portuguese --max-per-manifest 400
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.tacotron2_vae.hparams import Tacotron2VAEHparams  # noqa: E402
from models.tacotron2_vae.layers import TacotronSTFT  # noqa: E402
from models.tacotron2_vae.model import Tacotron2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True, help="Processed corpus dir (repeatable)")
    ap.add_argument("--max-per-manifest", type=int, default=0, help="Random sample per manifest (0 = all)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = Tacotron2(Tacotron2VAEHparams.from_dict(ckpt["hparams"]))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    stft = TacotronSTFT(1024, 256, 1024, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0)

    rows = []
    for m in a.manifest:
        with open(m / "mels_metadata.csv", encoding="utf-8") as f:
            part = [dict(r, source=m.name) for r in csv.DictReader(f)]
        if a.max_per_manifest and len(part) > a.max_per_manifest:
            part = random.Random(0).sample(part, a.max_per_manifest)
        rows += part

    mus, logvars = [], []
    with torch.inference_mode():
        for r in rows:
            wav = torch.load(r["mel_path"], map_location="cpu", weights_only=True)["waveform"]
            mel = stft.mel_spectrogram(wav.clamp(-1, 1)).to(device)  # (1, 80, T)
            _, mu, logvar, _ = model.vae_gst(mel)
            mus.append(mu[0].float().cpu().numpy())
            logvars.append(logvar[0].float().cpu().numpy())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, mu=np.stack(mus), logvar=np.stack(logvars),
             **{k: np.array([r.get(k, "") for r in rows]) for k in ["utt_id", "speaker", "emotion", "text", "source"]})
    print(f"saved {len(rows)} latents to {a.out}")


if __name__ == "__main__":
    main()
