"""
Training utilities for Tacotron 2 VAE.

Responsibilities:
    - Track and average training metrics using MetricsTracker.
    - Log training progress to TensorBoard.
    - Handle checkpoint saving and loading.
    - Perform latent space analysis using Singular Value Decomposition (SVD).
    - Manage the training loop for an epoch, including diagnostic plotting.

Main Classes:
    - MetricsTracker: Simple utility to accumulate and average scalars.
    - TensorBoardLogger: Wrapper for PyTorch SummaryWriter.

Main Functions:
    - save_checkpoint: Serialize model and optimizer state.
    - load_checkpoint: Restore model and optimizer state.
    - get_singular_values_of_latent_covariance: Perform PCA on latent space z.
    - train_epoch: Run a single epoch of training with monitoring.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, List, Tuple
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "tacotron2_vae"))

try:
    from losses import Tacotron2LossVAE
    from models.tacotron2_vae.hparams import Tacotron2VAEHparams
    from models.tacotron2_vae.model import Tacotron2, get_model_size_info
except ImportError:
    # Handle local relative imports
    from losses import Tacotron2LossVAE
    from hparams import Tacotron2VAEHparams
    from model import Tacotron2, get_model_size_info


class MetricsTracker:
    """
    Utility for tracking multiple scalar metrics.
    """
    def __init__(self) -> None:
        """Initialize the tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    def add(self, **kwargs: float) -> None:
        """
        Add new metric values.

        Args:
            **kwargs: Metric name and value pairs.
        """
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_averages(self) -> Dict[str, float]:
        """
        Compute mean of all tracked metrics.

        Returns:
            Dict[str, float]: Averaged metrics.
        """
        return {key: sum(values) / len(values) for key, values in self.metrics.items()}

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self.metrics = defaultdict(list)


class TensorBoardLogger:
    """
    Logger for TensorBoard events.
    """
    def __init__(self, log_dir: Path) -> None:
        """
        Initialize the logger.

        Args:
            log_dir (Path): Directory for logs.
        """
        self.writer: SummaryWriter = SummaryWriter(log_dir=str(log_dir))

    def log_model_info(self, model: Tacotron2) -> None:
        """
        Log model parameters to TensorBoard.

        Args:
            model (Tacotron2): The model.
        """
        info: Dict[str, int] = get_model_size_info(model)
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
    ) -> None:
        """
        Log training step metrics.

        Args:
            loss (float): Total loss.
            grad_norm (float): Gradient norm.
            learning_rate (float): Current LR.
            duration (float): Step duration.
            recon_loss (float): Reconstruction loss.
            kl_loss (float): KL divergence loss.
            kl_weight (float): Annealed KL weight.
            iteration (int): Global step.
        """
        self.writer.add_scalar("train/loss", loss, iteration)
        self.writer.add_scalar("train/recon_loss", recon_loss, iteration)
        self.writer.add_scalar("train/kl_loss", kl_loss, iteration)
        self.writer.add_scalar("train/kl_weight", kl_weight, iteration)
        self.writer.add_scalar("train/grad_norm", grad_norm, iteration)
        self.writer.add_scalar("train/learning_rate", learning_rate, iteration)
        self.writer.add_scalar("train/duration", duration, iteration)

    def log_validation(self, val_loss: float, iteration: int) -> None:
        """
        Log validation loss.

        Args:
            val_loss (float): Loss.
            iteration (int): Step.
        """
        self.writer.add_scalar("val/loss", val_loss, iteration)

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    filepath: Path,
    hparams: Tacotron2VAEHparams,
    **kwargs: Any,
) -> None:
    """
    Save model and training state.

    Args:
        model (nn.Module): The model.
        optimizer (Optimizer): The optimizer.
        learning_rate (float): Current LR.
        iteration (int): Global step.
        filepath (Path): Save path.
        hparams (Tacotron2VAEHparams): Hyperparameters.
    """
    payload: Dict[str, Any] = {
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
) -> Tuple[nn.Module, Optional[Dict[str, Any]], float, int]:
    """
    Load model and training state.

    Args:
        checkpoint_path (Path): Path to .pt file.
        model (nn.Module): Model to load into.

    Returns:
        Tuple: model, optimizer_state, learning_rate, iteration.
    """
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    learning_rate: float = checkpoint.get("learning_rate", 1e-3)
    iteration: int = checkpoint.get("iteration", 0)
    optimizer: Optional[Dict[str, Any]] = checkpoint.get("optimizer", None)

    return model, optimizer, learning_rate, iteration


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Find the checkpoint with the highest iteration count.

    Args:
        checkpoint_dir (Path): Directory containing checkpoints.

    Returns:
        Path: Latest checkpoint path.
    """
    checkpoint_files: List[Path] = list(checkpoint_dir.glob("epoch_*"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    latest_checkpoint: Path = max(checkpoint_files, key=lambda f: int(f.stem.split("_")[1]))
    return latest_checkpoint


def get_singular_values_of_latent_covariance(
    model: Tacotron2,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Collect latent vectors z from validation set and perform PCA.

    Args:
        model (Tacotron2): The model.
        val_loader (DataLoader): Validation data.
        device (torch.device): Compute device.

    Returns:
        Dict[str, Any]: SVD statistics of latent space.
    """
    model.eval()
    latent_vectors: List[np.ndarray] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 32:  # Limit validation batches to keep PCA fast
                break
            x, _ = model.parse_batch(batch, device)
            outputs: List[torch.Tensor] = model(x)
            z: torch.Tensor = outputs[6] # (B, L)
            latent_vectors.append(z.detach().cpu().numpy())

    z_numpy: np.ndarray = np.concatenate(latent_vectors, axis=0) # (N_samples, L)
    z_centered: np.ndarray = z_numpy - np.mean(z_numpy, axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)

    explained_variance: np.ndarray = S**2 / (z_centered.shape[0] - 1)
    explained_variance_ratio: np.ndarray = explained_variance / explained_variance.sum()

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
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Tacotron2LossVAE,
    device: torch.device,
    iteration: int,
    learning_rate: float,
    training_metadata: Optional[Dict[str, Any]] = None,
    tensorboard_logger: Optional[TensorBoardLogger] = None,
) -> Dict[str, Any]:
    """
    Run one training epoch.

    Args:
        model (Tacotron2): The model.
        hparams (Tacotron2VAEHparams): Hyperparameters.
        train_loader (DataLoader): Training data.
        test_loader (DataLoader): Test/Validation data.
        optimizer (Optimizer): Optimizer.
        criterion (Tacotron2LossVAE): Loss module.
        device (torch.device): Device.
        iteration (int): Current global iteration.
        learning_rate (float): Current LR.
        training_metadata (Optional[Dict]): Tracking dict for history.

    Returns:
        Dict[str, Any]: Updated training metadata.
    """
    model.train()

    for batch in tqdm(train_loader, desc=f"training", leave=False):
        model.train()
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        optimizer.zero_grad()
        x, y = model.parse_batch(batch, device)
        y_pred: List[torch.Tensor] = model(x) # [mel, mel_post, gate, align, mu, logvar, z]

        loss, recon_loss, kl_loss, kl_weight = criterion(y_pred, y, iteration)
        loss.backward()

        # Capture gradient norms per layer (only when logging checkpoints/plots to save GPU-CPU sync time)
        layer_grads: List[float] = []
        if iteration % hparams.iters_per_checkpoint == 0 or iteration == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer_grads.append(param.grad.norm().item())

        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), hparams.grad_clip_thresh
        )
        optimizer.step()
        reduced_loss: float = loss.item()

        if iteration % hparams.iters_per_checkpoint == 0:
            checkpoint_path: Path = Path(hparams.checkpoint_dir) / f"epoch_{iteration}"
            save_checkpoint(
                model, optimizer, learning_rate, iteration, checkpoint_path, hparams
            )

        # Diagnostic analysis and plotting
        if training_metadata is not None:
            # 1. Update basic history every step
            training_metadata.setdefault("training_loss", []).append(reduced_loss)
            training_metadata.setdefault("grad_norm", []).append(float(grad_norm))
            training_metadata.setdefault("learning_rate", []).append(learning_rate)
            training_metadata.setdefault("recon_loss", []).append(recon_loss.item())
            training_metadata.setdefault("kl_loss", []).append(kl_loss.item())
            training_metadata.setdefault("kl_weight", []).append(float(kl_weight))

            # 2. TensorBoard logging & Heavy validation evaluation: run only every iters_per_checkpoint steps (or step 0)
            if iteration % hparams.iters_per_checkpoint == 0 or iteration == 0:
                singular_values_info: Dict[str, Any] = get_singular_values_of_latent_covariance(model, test_loader, device)

                model.eval()
                test_batch: Any = next(iter(test_loader))
                val_batch: Any = next(iter(val_loader))
                
                with torch.no_grad():
                    xt, yt = model.parse_batch(test_batch, device)
                    yt_pred: List[torch.Tensor] = model(xt)
                    test_loss, test_recon, test_kl, _ = criterion(yt_pred, yt, iteration)
                    
                    xv, yv = model.parse_batch(val_batch, device)
                    yv_pred: List[torch.Tensor] = model(xv)
                    val_loss, val_recon, val_kl, _ = criterion(yv_pred, yv, iteration)

                mel_pred: np.ndarray = yt_pred[0][0].cpu().numpy() # (n_mels, T)
                mel_target: np.ndarray = yt[0][0].cpu().numpy()   # (n_mels, T)

                training_metadata.setdefault("singular_values_of_latent_covariance", []).append(singular_values_info)
                training_metadata.setdefault("target_predict_example", []).append((mel_pred, mel_target))

                if tensorboard_logger:
                    writer = tensorboard_logger.writer
                    
                    # Log training scalars
                    writer.add_scalar("Loss/Train_Total", reduced_loss, iteration)
                    writer.add_scalar("Loss/Train_Recon", recon_loss.item(), iteration)
                    writer.add_scalar("Loss/Train_KL", kl_loss.item(), iteration)
                    writer.add_scalar("LearningRate", learning_rate, iteration)
                    writer.add_scalar("Loss/Train_KL_Weight", kl_weight, iteration)
                    writer.add_scalar("Gradients/Global_Norm", float(grad_norm), iteration)
                    
                    # Log individual layer gradient norms
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_scalar(f"GradientNorms_Layers/{name}", param.grad.norm().item(), iteration)
                    
                    # Log test/validation losses
                    writer.add_scalar("Loss/Test_Total", test_loss.item(), iteration)
                    writer.add_scalar("Loss/Test_Recon", test_recon.item(), iteration)
                    writer.add_scalar("Loss/Test_KL", test_kl.item(), iteration)

                    writer.add_scalar("Loss/Val_Total", val_loss.item(), iteration)
                    writer.add_scalar("Loss/Val_Recon", val_recon.item(), iteration)
                    writer.add_scalar("Loss/Val_KL", val_kl.item(), iteration)

                    # Pair: target and predicted spectrograms
                    fig_spec, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
                    ax_t.set_title("Target Spectrogram")
                    im_t = ax_t.imshow(mel_target, aspect="auto", origin="lower")
                    fig_spec.colorbar(im_t, ax=ax_t)
                    ax_p.set_title("Predicted Spectrogram")
                    im_p = ax_p.imshow(mel_pred, aspect="auto", origin="lower")
                    fig_spec.colorbar(im_p, ax=ax_p)
                    writer.add_figure("Spectrograms/Target_vs_Predicted", fig_spec, iteration)
                    plt.close(fig_spec)

                    # Bar graph of PCA singular values of latent variables
                    s_vals: List[float] = singular_values_info["singular_values"].tolist()
                    fig_pca, ax_pca = plt.subplots(figsize=(8, 4))
                    ax_pca.set_title("Latent Space Singular Values (Z PCA)")
                    ax_pca.bar(range(len(s_vals)), s_vals, color='purple')
                    writer.add_figure("PCA/Singular_Values", fig_pca, iteration)
                    plt.close(fig_pca)

                    # Histogram/Bar graph of gradient norms distribution of the model layers
                    fig_grad, ax_grad = plt.subplots(figsize=(10, 4))
                    ax_grad.set_title("Gradient Norms Distribution")
                    ax_grad.bar(range(len(layer_grads)), layer_grads, color='orange')
                    writer.add_figure("Gradients/Norms_Distribution", fig_grad, iteration)
                    plt.close(fig_grad)

                # Explicitly delete GPU tensors to free VRAM immediately and prevent scoping memory leak
                del xt, yt, yt_pred, xv, yv, yv_pred, test_loss, val_loss, test_batch, val_batch
                torch.cuda.empty_cache()
                model.train()

        iteration += 1

    return training_metadata


def save_hparams(hparams: Tacotron2VAEHparams, path: Path) -> None:
    """
    Save hyperparameters to a JSON file.

    Args:
        hparams (Tacotron2VAEHparams): The hparams object.
        path (Path): Destination path.
    """
    path.write_text(json.dumps(hparams.to_dict(), indent=2), encoding="utf-8")