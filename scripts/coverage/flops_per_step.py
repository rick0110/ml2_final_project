#!/usr/bin/env python3
"""
FLOPs of one training step (forward + backward) of a coverage run, counted on the CPU.

GPU hardware counters (DCGM profiling, Nsight Compute) need admin rights on the cluster
(RmProfilingAdminOnly=1), so the achieved throughput is estimated instead:

    achieved FLOP/s = FLOPs per step (this script) x steps per second (training log)

FLOPs are counted with the PyTorch profiler (with_flops=True), which counts matrix
multiplications and convolutions (the bulk of the work; elementwise ops are not counted),
on real batches of the run's training set, with the run's own hyperparameters and
vocabulary (hparams.json and symbols.json of the experiment folder). Weights do not
change the count, so the model is freshly initialized. Runs on the CPU: the GPU is not used.

Writes <out> as JSON: mean, sd and per-batch FLOPs, batch size, and mel frames per batch.

Example:
    python scripts/coverage/flops_per_step.py --experiment experiments/tacotron2-vae/cov_n128_k32_s1 \
        --data-dir data/processed/cov_n128_k32 --batches 4 --out flops_n128.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
# The training folder goes first: the data-loader folder has its own utils.py with another TextMelCollate
for p in (ROOT / "src", ROOT / "src" / "data" / "loader_vae_tacotron", ROOT / "src" / "training" / "training-tacotron2-vae"):
    sys.path.insert(0, str(p))
from losses import Tacotron2LossVAE  # noqa: E402
from loader_tacotron import load_data  # noqa: E402
from models.tacotron2_vae.hparams import Tacotron2VAEHparams  # noqa: E402
from models.tacotron2_vae.model import load_tacotron2_vae_model  # noqa: E402
from text_processing import TextProcessor  # noqa: E402
import importlib.util  # noqa: E402

# The training TextMelCollate (gate targets per frame); several folders have a utils.py, so load it by path
_spec = importlib.util.spec_from_file_location("train_utils_collate", ROOT / "src" / "training" / "training-tacotron2-vae" / "utils.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TextMelCollate = _mod.TextMelCollate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=Path, required=True, help="Experiment folder with hparams.json and symbols.json")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    hp = Tacotron2VAEHparams.from_dict(json.loads((a.experiment / "hparams.json").read_text()))
    tp = TextProcessor.load(a.experiment / "symbols.json")
    train, _, _ = load_data(text_processor=tp, data_dir=a.data_dir, val_split=0.03,
                            generator=torch.Generator().manual_seed(1))
    loader = torch.utils.data.DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0,
                                         collate_fn=TextMelCollate(hp.n_frames_per_step),
                                         generator=torch.Generator().manual_seed(2))
    device = torch.device("cpu")
    model = load_tacotron2_vae_model(hp, device=device)
    model.train()
    criterion = Tacotron2LossVAE(hp)

    per_batch, frames = [], []
    for i, batch in enumerate(loader):
        if i >= a.batches:
            break
        x, y = model.parse_batch(batch, device)
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            out = model(x)
            loss = criterion(out, y, 2500, output_lengths=x[3], input_lengths=x[1])[0]
            loss.backward()
        model.zero_grad(set_to_none=True)
        per_batch.append(int(sum(e.flops for e in prof.events() if e.flops)))
        frames.append(int(x[2].shape[-1]))
        print(f"batch {i}: {per_batch[-1] / 1e12:.3f} TFLOP, {frames[-1]} mel frames (padded)", flush=True)

    res = dict(experiment=str(a.experiment), data_dir=str(a.data_dir), batch_size=a.batch_size,
               flops_per_step_mean=float(np.mean(per_batch)), flops_per_step_sd=float(np.std(per_batch)),
               flops_per_batch=per_batch, padded_mel_frames=frames,
               counted="matmul and convolution FLOPs, forward + backward (torch.profiler with_flops)")
    a.out.write_text(json.dumps(res, indent=1))
    print(f"mean {res['flops_per_step_mean'] / 1e12:.3f} TFLOP per step (sd {res['flops_per_step_sd'] / 1e12:.3f})")


if __name__ == "__main__":
    main()
