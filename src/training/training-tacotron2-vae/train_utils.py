from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys
import time
from time import time
from typing import Any, Dict, Optional
import math
import sys
import numpy as np
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "tacotron2_vae"))


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
    **kwargs,
):
    payload = {
        "iteration": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": kwargs.get("scheduler_state_dict", None),
        "learning_rate": learning_rate,
        "hparams": hparams.to_dict(),
    }
    torch.save(payload, filepath)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    learning_rate = checkpoint.get("learning_rate", 1e-3)
    iteration = checkpoint.get("iteration", 0)
    optimizer = checkpoint.get("optimizer", None)
    scheduler = checkpoint.get("scheduler", None)

    
    return model, optimizer, learning_rate, iteration

def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoint_files = list(checkpoint_dir.glob("epoch_*"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.stem.split("_")[1]))
    return latest_checkpoint



def get_singular_values_of_latent_covariance(
    model,
    val_loader: DataLoader,
    device: torch.device,
):
    """
    Coleta todos os vetores latentes z do validation set,
    calcula PCA e retorna estatísticas do espaço latente.
    """

    model.eval()

    latent_vectors = []

    with torch.no_grad():
        for batch in val_loader:

            x, _ = model.parse_batch(batch, device)

            outputs = model(x)

            z = outputs[6]

            latent_vectors.append(
                z.detach().cpu().numpy()
            )

    z_numpy = np.concatenate(latent_vectors, axis=0)

    z_centered = z_numpy - np.mean(z_numpy, axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(
        z_centered,
        full_matrices=False
    )

    explained_variance = (
        S**2 / (z_centered.shape[0] - 1)
    )

    explained_variance_ratio = (
        explained_variance /
        explained_variance.sum()
    )

    return {
        "latent_vectors": z_numpy,
        "singular_values": S,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "components": Vt,
    }

def train_epoch(
    model: Tacotron2,
    hparams: Tacotron2VAEHparams,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Tacotron2LossVAE,
    device: torch.device,
    iteration: int,
    learning_rate: float,
    training_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    model.train()
    # metrics = MetricsTracker()

    for batch in tqdm(train_loader, desc=f"training", leave=False):
        model.train()
        start = time.perf_counter()
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        optimizer.zero_grad()
        x, y = model.parse_batch(batch, device)
        y_pred = model((x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
        loss, recon_loss, kl_loss, kl_weight = criterion(y_pred, y, iteration)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), hparams.grad_clip_thresh
        )
        optimizer.step()
        reduced_loss = loss.item()
        if not math.isnan(reduced_loss):
            duration = time.perf_counter() - start
            print(
                f"Train loss {iteration} {reduced_loss:.6f} "
                f"Grad Norm {float(grad_norm):.6f} {duration:.2f}s/it"
            )
            # tb_logger.log_training(
            #     reduced_loss,
            #     float(grad_norm),
            #     learning_rate,
            #     duration,
            #     recon_loss.item(),
            #    kl_loss.item(),
            #    float(kl_weight),
            #    iteration,
            #)

        if iteration % hparams.iters_per_checkpoint == 0:
            checkpoint_path = hparams.checkpoint_dir / f"epoch_{iteration}"
            save_checkpoint(
                model, optimizer, learning_rate, iteration, checkpoint_path, hparams
            )
        iteration += 1

    # tb_logger.close()
        
        if training_metadata is not None:
            singular_values_info = get_singular_values_of_latent_covariance(model, test_loader, device)
            model.eval()
            batch = next(iter(test_loader))
            with torch.no_grad():
                x, y = model.parse_batch(batch, device)
                y_pred = model(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            


            mel_pred = y_pred[0][1].cpu().numpy()
            mel_target = y[0][0].cpu().numpy()

            losses = training_metadata.get("training_loss", [])
            grad_norms = training_metadata.get("grad_norm", [])
            learning_rates = training_metadata.get("learning_rate", [])
            durations = training_metadata.get("duration", [])
            recon_losses = training_metadata.get("recon_loss", [])
            kl_losses = training_metadata.get("kl_loss", [])
            kl_weights = training_metadata.get("kl_weight", [])
            singular_values = training_metadata.get("singular_values_of_latent_covariance", [])
            target_predicts = training_metadata.get("target_predict_example", [])
            losses.append(reduced_loss)
            grad_norms.append(float(grad_norm))
            learning_rates.append(learning_rate)
            durations.append(duration)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            kl_weights.append(float(kl_weight))
            singular_values.append(singular_values_info)
            target_predicts.append((mel_pred, mel_target))

    return training_metadata



def save_hparams(hparams: Tacotron2VAEHparams, path: Path):
    path.write_text(json.dumps(hparams.to_dict(), indent=2), encoding="utf-8")
