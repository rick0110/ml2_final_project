"""Reconstruct validation-loss curves by evaluating saved checkpoints on a fixed
LJSpeech utterance set — identical protocol for baseline_lj and attn_lj."""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tts_toolkit as tk
from tts_toolkit import PROJECT_ROOT, DEVICE

sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))
from losses import Tacotron2LossVAE

OUT = Path(__file__).parent / "ckpt_eval.json"

N_UTTS = 32
BATCH = 16
random.seed(42)

rows = list(csv.DictReader(open(PROJECT_ROOT / "data/processed/LJSpeech-tacotron-vae/mels_metadata.csv")))
rows = [r for r in rows if 3.0 <= float(r["duration"]) <= 7.0]
sample = random.sample(rows, N_UTTS)


def build_batches(tp):
    items = []
    for r in sample:
        d = torch.load(PROJECT_ROOT / r["mel_path"], map_location="cpu", weights_only=False)
        mel = d["mel"]
        while mel.dim() > 2:
            mel = mel.squeeze(0)
        seq = torch.LongTensor(tp.text_to_sequence(r["text"]))
        items.append((seq, mel))
    batches = []
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        chunk.sort(key=lambda x: -len(x[0]))
        input_lengths = torch.LongTensor([len(t) for t, _ in chunk])
        max_t = input_lengths.max().item()
        text_padded = torch.zeros(len(chunk), max_t, dtype=torch.long)
        for j, (t, _) in enumerate(chunk):
            text_padded[j, :len(t)] = t
        max_m = max(m.size(1) for _, m in chunk)
        mel_padded = torch.full((len(chunk), 80, max_m), -11.5129)
        gate_padded = torch.zeros(len(chunk), max_m)
        output_lengths = torch.LongTensor(len(chunk))
        for j, (_, m) in enumerate(chunk):
            mel_padded[j, :, :m.size(1)] = m
            gate_padded[j, m.size(1) - 1:] = 1
            output_lengths[j] = m.size(1)
        batches.append((text_padded, input_lengths, mel_padded, gate_padded, output_lengths))
    return batches


@torch.inference_mode()
def eval_ckpt(model, loss_fn, batches, iteration):
    tot = {"total": 0.0, "recon": 0.0, "kl": 0.0, "gattn": 0.0}
    n = 0
    for text_padded, input_lengths, mel_padded, gate_padded, output_lengths in batches:
        x = (text_padded.to(DEVICE), input_lengths.to(DEVICE),
             mel_padded.to(DEVICE), output_lengths.to(DEVICE))
        y = (mel_padded.to(DEVICE), gate_padded.to(DEVICE))
        out = model.forward(x)
        total, recon, kl, klw, gattn = loss_fn(
            out, y, iteration,
            output_lengths=output_lengths.to(DEVICE),
            input_lengths=input_lengths.to(DEVICE),
        )
        b = text_padded.size(0)
        tot["total"] += total.item() * b
        tot["recon"] += recon.item() * b
        tot["kl"] += kl.item() * b
        tot["gattn"] += gattn.item() * b
        n += b
    return {k: v / n for k, v in tot.items()}


def main():
    results = {}
    for name, stride in [("baseline_lj", 2), ("attn_lj", 2)]:
        exp_dir = tk.EXPS[name]
        ckpts = sorted(exp_dir.glob("checkpoints/epoch_*"),
                       key=lambda p: int(p.name.split("_")[1]))
        ckpts = [c for i, c in enumerate(ckpts) if i % stride == 0 or c == ckpts[-1]]
        print(f"{name}: {len(ckpts)} checkpoints", flush=True)
        model, tp, hp, has_attn, _ = tk.load_model(name)
        loss_fn = Tacotron2LossVAE(hp)
        batches = build_batches(tp)
        curve = []
        for c in ckpts:
            it = int(c.name.split("_")[1])
            sd = torch.load(c, map_location=DEVICE, weights_only=False)["state_dict"]
            model.load_state_dict(sd, strict=False)
            model.eval()
            m = eval_ckpt(model, loss_fn, batches, it)
            m["iteration"] = it
            curve.append(m)
            print(f"  {name} it={it} recon={m['recon']:.4f} kl={m['kl']:.4f}", flush=True)
        results[name] = curve
        del model
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=1))
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
