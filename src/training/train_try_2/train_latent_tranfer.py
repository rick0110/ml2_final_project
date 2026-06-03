#!/usr/bin/env python3
"""Train a mel frontend transfer model for HiFi_GAN compatibility.

This script learns a lightweight adapter that maps the project's current
stored mel frontend to the mel frontend expected by the bundled HiFi_GAN.

The model is trained on paired data:
- input: mel spectrogram already stored in the dataset
- target: mel spectrogram computed from the paired waveform using the
  HiFi_GAN frontend (trg_melspec_fn)

The resulting adapter can be inserted before the vocoder to remove the
frontend mismatch that causes noisy reconstructions.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.train_try_2.data import create_dataloaders


class MetricsTracker:
    def __init__(self):
        self.values: Dict[str, list[float]] = {}

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.values.setdefault(key, []).append(float(value))

    def averages(self) -> Dict[str, float]:
        return {key: sum(vals) / len(vals) for key, vals in self.values.items()}


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{key}" if prefix else key, value, step)

    def _normalize_image_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().float().cpu()
        min_value = tensor.min()
        max_value = tensor.max()
        if torch.isclose(max_value, min_value):
            return torch.zeros_like(tensor)
        return (tensor - min_value) / (max_value - min_value)

    def log_mel_examples(
        self,
        step: int,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        adapted_mel: torch.Tensor,
        prefix: str = "examples/",
    ):
        import matplotlib.pyplot as plt

        source_mel = source_mel.detach().float().cpu().squeeze(0)
        target_mel = target_mel.detach().float().cpu().squeeze(0)
        adapted_mel = adapted_mel.detach().float().cpu().squeeze(0)

        panels = [
            (source_mel, "Source Mel"),
            (target_mel, "Target Mel"),
            (adapted_mel, "Adapted Mel"),
        ]

        figure, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
        for axis, (panel, title) in zip(axes, panels):
            image = axis.imshow(panel, aspect="auto", origin="lower", interpolation="nearest")
            axis.set_title(title)
            axis.set_xlabel("Time")
            axis.set_ylabel("Mel")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        self.writer.add_figure(f"{prefix}mel_comparison", figure, step)
        plt.close(figure)

    def log_audio_examples(
        self,
        step: int,
        sample_rate: int,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
        adapted_audio: torch.Tensor,
        prefix: str = "examples/",
    ):
        self.writer.add_audio(f"{prefix}source_audio", source_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}target_audio", target_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}adapted_audio", adapted_audio.detach().cpu(), step, sample_rate=sample_rate)

    def log_hyperparameters(self, hparams: Dict[str, float], metrics: Dict[str, float]):
        sanitized_hparams = {}
        frontend_config = None

        for key, value in hparams.items():
            if isinstance(value, dict):
                if key == "vocoder_frontend":
                    frontend_config = value
                continue
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                sanitized_hparams[key] = value
            else:
                sanitized_hparams[key] = str(value)

        self.writer.add_hparams(sanitized_hparams, metrics)
        if frontend_config is not None:
            self.writer.add_text("hparams/vocoder_frontend", json.dumps(frontend_config, indent=2, ensure_ascii=False))

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        num_groups = max(1, math.gcd(channels, 8))
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class MelFrontendTransfer(nn.Module):
    def __init__(self, n_mels: int = 80, hidden_channels: int = 128, num_blocks: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(n_mels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden_channels, dropout=dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, n_mels, kernel_size=1),
        )

    def forward(self, source_mel: torch.Tensor) -> torch.Tensor:
        if source_mel.dim() != 3:
            raise ValueError(f"source_mel must be 3D (batch, n_mels, time), got {tuple(source_mel.shape)}")

        x = self.input_proj(source_mel)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a mel frontend transfer model for HiFi_GAN compatibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--analysis-interval", type=int, default=5)
    parser.add_argument("--analysis-examples", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)
    return parser.parse_args()


def create_experiment_dir(experiment_name: str | None = None) -> Path:
    experiments_root = PROJECT_ROOT / "experiments" / "train_try_2" / "latent_transfer"
    experiments_root.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        experiment_name = f"latent_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dir = experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "tensorboard").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    return experiment_dir


def resolve_experiment_dir(experiment_ref: str) -> Path:
    candidate = Path(experiment_ref)
    if candidate.exists():
        return candidate

    named_candidate = PROJECT_ROOT / "experiments" / "train_try_2" / "latent_transfer" / experiment_ref
    if named_candidate.exists():
        return named_candidate

    raise FileNotFoundError(
        f"Experiment directory not found: {experiment_ref}. Tried {candidate} and {named_candidate}"
    )


def load_experiment_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_candidates = []
    for checkpoint_path in checkpoint_dir.glob("epoch_*.pt"):
        match = re.search(r"epoch_(\d+)\.pt$", checkpoint_path.name)
        if match:
            checkpoint_candidates.append((int(match.group(1)), checkpoint_path))

    if checkpoint_candidates:
        return max(checkpoint_candidates, key=lambda item: item[0])[1]

    best_checkpoint = checkpoint_dir / "best.pt"
    if best_checkpoint.exists():
        return best_checkpoint

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")


def save_checkpoint(model, optimizer, scheduler, epoch: int, metrics: Dict[str, float], checkpoint_dir: Path, filename: str):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return checkpoint["epoch"], checkpoint.get("metrics", {})


def load_hifigan_vocoder(device: torch.device):
    hifigan_path = PROJECT_ROOT / "src" / "models" / "HiFi_GAN.py"
    if not hifigan_path.exists():
        raise FileNotFoundError(f"HiFi_GAN loader not found at {hifigan_path}")

    spec = importlib.util.spec_from_file_location("hifigan_module", hifigan_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import HiFi_GAN module from {hifigan_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_hifigan_model"):
        raise ImportError("load_hifigan_model not found in HiFi_GAN module")

    _, vocoder = module.load_hifigan_model(freeze=True)
    return vocoder.to(device).eval()


def _to_audio_1d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu().squeeze().clamp(-1.0, 1.0)


def _as_batch_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 3 and waveform.size(1) == 1:
        return waveform.squeeze(1)
    if waveform.dim() == 2:
        return waveform
    raise ValueError(f"Expected waveform with shape (batch, time) or (batch, 1, time), got {tuple(waveform.shape)}")


def _align_time(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    if predicted_mel.dim() != 3 or target_mel.dim() != 3:
        raise ValueError(f"Expected 3D mel tensors, got predicted={predicted_mel.shape}, target={target_mel.shape}")

    if predicted_mel.size(-1) == target_mel.size(-1):
        return predicted_mel

    return F.interpolate(predicted_mel, size=target_mel.size(-1), mode="linear", align_corners=False)


def _compute_target_mel(vocoder, waveform: torch.Tensor) -> torch.Tensor:
    waveform = _as_batch_waveform(waveform)
    seq_len = torch.full((waveform.size(0),), waveform.size(-1), device=waveform.device, dtype=torch.long)
    target = vocoder.trg_melspec_fn(waveform, seq_len)
    if isinstance(target, tuple):
        target = target[0]
    if target.dim() != 3:
        raise ValueError(f"Expected target mel to be 3D, got {tuple(target.shape)}")
    return target


def _log_example(
    model,
    vocoder,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    logger: TensorBoardLogger,
    step: int,
):
    source_mel = batch["mel"].to(device)
    waveform = batch["waveform"].to(device)
    target_mel = _compute_target_mel(vocoder, waveform)

    with torch.no_grad():
        adapted_mel = model(source_mel)
        adapted_mel = _align_time(adapted_mel, target_mel)

        source_audio = vocoder(spec=source_mel[0:1])
        target_audio = vocoder(spec=target_mel[0:1])
        adapted_audio = vocoder(spec=adapted_mel[0:1])

    sr_value = batch.get("sr", 22050)
    if isinstance(sr_value, torch.Tensor):
        sample_rate = int(sr_value[0].item())
    elif isinstance(sr_value, (list, tuple)):
        sample_rate = int(sr_value[0])
    else:
        sample_rate = int(sr_value)

    logger.log_mel_examples(
        step=step,
        source_mel=source_mel[0],
        target_mel=target_mel[0],
        adapted_mel=adapted_mel[0],
    )
    logger.log_audio_examples(
        step=step,
        sample_rate=sample_rate,
        source_audio=_to_audio_1d(source_audio),
        target_audio=_to_audio_1d(target_audio),
        adapted_audio=_to_audio_1d(adapted_audio),
    )


def train_epoch(model, vocoder, train_loader, optimizer, device, epoch: int, max_epochs: int):
    model.train()
    metrics = MetricsTracker()

    from tqdm import tqdm

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [TRAIN]", leave=True, total=len(train_loader))
    for batch in pbar:
        source_mel = batch["mel"].to(device)
        waveform = batch["waveform"].to(device)
        target_mel = _compute_target_mel(vocoder, waveform)

        optimizer.zero_grad()
        adapted_mel = model(source_mel)
        adapted_mel = _align_time(adapted_mel, target_mel)

        l1_loss = F.l1_loss(adapted_mel, target_mel)
        mse_loss = F.mse_loss(adapted_mel, target_mel)
        smooth_loss = F.l1_loss(
            adapted_mel[..., 1:] - adapted_mel[..., :-1],
            target_mel[..., 1:] - target_mel[..., :-1],
        ) if adapted_mel.size(-1) > 1 else torch.tensor(0.0, device=device)

        loss = l1_loss + 0.25 * mse_loss + 0.1 * smooth_loss
        loss.backward()
        optimizer.step()

        metrics.add(loss=loss.item(), l1=l1_loss.item(), mse=mse_loss.item(), smooth=smooth_loss.item())
        averages = metrics.averages()
        pbar.set_postfix({"loss": f"{averages['loss']:.4f}"}, refresh=True)

    if "loss" not in metrics.values:
        raise RuntimeError("No training batches produced valid losses.")
    return metrics.averages()


def validate_epoch(model, vocoder, val_loader, device, epoch: int, max_epochs: int):
    model.eval()
    metrics = MetricsTracker()

    from tqdm import tqdm

    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [VAL]", leave=True, total=len(val_loader))
    with torch.no_grad():
        for batch in pbar:
            source_mel = batch["mel"].to(device)
            waveform = batch["waveform"].to(device)
            target_mel = _compute_target_mel(vocoder, waveform)
            adapted_mel = _align_time(model(source_mel), target_mel)

            loss = F.l1_loss(adapted_mel, target_mel)
            mse_loss = F.mse_loss(adapted_mel, target_mel)
            metrics.add(loss=loss.item(), l1=loss.item(), mse=mse_loss.item())

            averages = metrics.averages()
            pbar.set_postfix({"loss": f"{averages['loss']:.4f}"}, refresh=True)

    if "loss" not in metrics.values:
        raise RuntimeError("No validation batches produced valid losses.")
    return metrics.averages()


def _frontend_config(vocoder) -> Dict[str, object]:
    frontend = vocoder.trg_melspec_fn
    keys = [
        "sample_rate",
        "n_fft",
        "hop_length",
        "win_length",
        "nfilt",
        "log",
        "log_zero_guard_type",
        "log_zero_guard_value",
        "normalize",
        "dither",
        "preemph",
        "pad_to",
        "exact_pad",
    ]
    config = {}
    for key in keys:
        if hasattr(frontend, key):
            value = getattr(frontend, key)
            try:
                json.dumps(value)
                config[key] = value
            except Exception:
                config[key] = str(value)
    return config


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resume_experiment_dir = Path(args.resume_experiment) if args.resume_experiment else None
    if resume_experiment_dir is not None:
        experiment_dir = resolve_experiment_dir(args.resume_experiment)
    else:
        experiment_dir = create_experiment_dir(args.experiment_name)

    print(f"\nExperiment directory: {experiment_dir}")
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"

    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    vocoder = load_hifigan_vocoder(device)

    model = MelFrontendTransfer(
        n_mels=80,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    hparams = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_min_lr": args.scheduler_min_lr,
        "num_workers": args.num_workers,
        "hidden_channels": args.hidden_channels,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "val_split": args.val_split,
        "seed": args.seed,
        "vocoder_frontend": _frontend_config(vocoder),
    }

    config_path = experiment_dir / "config.json"
    if resume_experiment_dir is not None:
        saved_hparams = load_experiment_config(experiment_dir)
        for key, value in saved_hparams.items():
            if hasattr(args, key):
                setattr(args, key, value)
        hparams.update(saved_hparams)
        print(f"✓ Loaded config from {config_path}")
    else:
        config_path.write_text(json.dumps(hparams, indent=2), encoding="utf-8")
        print(f"✓ Config saved to {config_path}")

    tb_logger = TensorBoardLogger(tensorboard_dir)
    tb_logger.log_hyperparameters(hparams, {})

    start_epoch = 0
    if args.resume or args.resume_experiment:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir) if args.resume_experiment else Path(args.resume)
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)
        print(f"  Loaded from epoch {start_epoch}")

    best_val_loss = float("inf")
    print("\nStarting latent transfer training...")

    for epoch in range(start_epoch, args.num_epochs):
        train_metrics = train_epoch(model, vocoder, train_loader, optimizer, device, epoch, args.num_epochs)
        val_metrics = validate_epoch(model, vocoder, val_loader, device, epoch, args.num_epochs)

        scheduler.step(val_metrics["loss"])

        tb_logger.log_metrics(train_metrics, epoch, prefix="train/")
        tb_logger.log_metrics(val_metrics, epoch, prefix="val/")
        if args.analysis_interval > 0 and epoch % args.analysis_interval == 0:
            example_batch = next(iter(val_loader))
            _log_example(model, vocoder, example_batch, device, tb_logger, epoch)
        tb_logger.flush()

        print(f"\nEpoch {epoch + 1}/{args.num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"    ├─ L1: {train_metrics['l1']:.6f}")
        print(f"    ├─ MSE: {train_metrics['mse']:.6f}")
        print(f"    └─ Smooth: {train_metrics['smooth']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"    ├─ L1: {val_metrics['l1']:.6f}")
        print(f"    └─ MSE: {val_metrics['mse']:.6f}")

        metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, checkpoint_dir, f"epoch_{epoch + 1:04d}.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, checkpoint_dir, "best.pt")
            print(f"  ✓ New best validation loss: {best_val_loss:.6f}")

    print("\nTraining completed!")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")

    tb_logger.close()


if __name__ == "__main__":
    main()