from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses import Tacotron2LossVAE
from models.tacotron2_vae.hparams import Tacotron2VAEHparams
from models.tacotron2_vae.model import Tacotron2, get_model_size_info


class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_averages(self) -> Dict[str, float]:
        return {key: sum(values) / len(values) for key, values in self.metrics.items()}

    def reset(self):
        self.metrics = defaultdict(list)


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_model_info(self, model: Tacotron2):
        info = get_model_size_info(model)
        for key, value in info.items():
            self.writer.add_text("model_info", f"{key}: {value}", 0)

    def log_training(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        duration: float,
        recon_loss: float,
        kl_loss: float,
        kl_weight: float,
        iteration: int,
    ):
        self.writer.add_scalar("train/loss", loss, iteration)
        self.writer.add_scalar("train/recon_loss", recon_loss, iteration)
        self.writer.add_scalar("train/kl_loss", kl_loss, iteration)
        self.writer.add_scalar("train/kl_weight", kl_weight, iteration)
        self.writer.add_scalar("train/grad_norm", grad_norm, iteration)
        self.writer.add_scalar("train/learning_rate", learning_rate, iteration)
        self.writer.add_scalar("train/duration", duration, iteration)

    def log_validation(self, val_loss: float, iteration: int):
        self.writer.add_scalar("val/loss", val_loss, iteration)

    def close(self):
        self.writer.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    filepath: Path,
    hparams: Tacotron2VAEHparams,
):
    payload = {
        "iteration": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
        "hparams": hparams.to_dict(),
    }
    torch.save(payload, filepath)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    learning_rate = checkpoint.get("learning_rate", 1e-3)
    iteration = checkpoint.get("iteration", 0)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, learning_rate, iteration


def validate_epoch(
    model: Tacotron2,
    criterion: Tacotron2LossVAE,
    val_loader: DataLoader,
    device: torch.device,
    iteration: int,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = model.parse_batch(batch, device)
            y_pred = model((x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
            loss, _, _, _ = criterion(y_pred, y, iteration)
            total_loss += loss.item()
    model.train()
    return total_loss / max(len(val_loader), 1)


def train_epoch(
    model: Tacotron2,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Tacotron2LossVAE,
    device: torch.device,
    iteration: int,
    grad_clip_thresh: float,
    learning_rate: float,
) -> Dict[str, Any]:
    model.train()
    metrics = MetricsTracker()

    for batch in tqdm(train_loader, desc=f"Iteration {iteration}", leave=False):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        optimizer.zero_grad()
        x, y = model.parse_batch(batch, device)
        y_pred = model((x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
        loss, recon_loss, kl_loss, kl_weight = criterion(y_pred, y, iteration)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        optimizer.step()

        metrics.add(
            loss=loss.item(),
            recon_loss=recon_loss.item(),
            kl_loss=kl_loss.item(),
            kl_weight=float(kl_weight),
            grad_norm=float(grad_norm),
        )
        iteration += 1

    return {"iteration": iteration, **metrics.get_averages()}


def save_hparams(hparams: Tacotron2VAEHparams, path: Path):
    path.write_text(json.dumps(hparams.to_dict(), indent=2), encoding="utf-8")
