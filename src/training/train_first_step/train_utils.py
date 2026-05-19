"""Training utilities for first-step TTS model.

Includes:
- Epoch training loop
- Checkpoint saving/loading
- TensorBoard logging
- Metrics computation
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.train_first_step.model_loader import FirstStepTTSModel
from training.train_first_step.losses import CombinedTTSLoss


class MetricsTracker:
    """Track training metrics across batches."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def add(self, **kwargs):
        """Add metric values."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average of all tracked metrics."""
        return {key: sum(values) / len(values) for key, values in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)


def _align_predicted_mel_to_target(
    predicted_mel: torch.Tensor,
    target_mel: torch.Tensor,
) -> torch.Tensor:
    """Align predicted mel time axis to the target mel time axis.

    Expects tensors in (batch, n_mels, time_steps).
    """
    if predicted_mel.dim() != 3 or target_mel.dim() != 3:
        raise ValueError(
            f"Expected 3D mel tensors, got predicted={predicted_mel.shape}, target={target_mel.shape}"
        )

    target_time = target_mel.size(2)
    pred_time = predicted_mel.size(2)
    if pred_time == target_time:
        return predicted_mel

    return F.interpolate(
        predicted_mel,
        size=target_time,
        mode="linear",
        align_corners=False,
    )


def train_epoch(
    model: FirstStepTTSModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedTTSLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    metrics = MetricsTracker()

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{max_epochs} [TRAIN]",
        leave=True,
        total=len(train_loader),
    )

    for batch_idx, batch in enumerate(pbar):
        mels = batch["mel"].to(device)
        batch_size = mels.shape[0]
        max_text_len = 256
        text_ids = torch.randint(0, 1000, (batch_size, max_text_len)).to(device)

        optimizer.zero_grad()

        try:
            if use_amp:
                with torch.cuda.amp.autocast():
                    predicted_mel, style_embeddings = model(
                        text_ids=text_ids,
                        target_mel=mels,
                        use_vocoder=False,
                    )

                    if predicted_mel.dim() == 3:
                        predicted_mel = predicted_mel.transpose(1, 2)
                    predicted_mel = _align_predicted_mel_to_target(predicted_mel, mels)

                    loss, recon_loss, div_loss = criterion(
                        predicted_mel=predicted_mel,
                        target_mel=mels,
                        style_embeddings=style_embeddings,
                    )

                if scaler is None:
                    raise RuntimeError("AMP enabled but GradScaler is None")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_mel, style_embeddings = model(
                    text_ids=text_ids,
                    target_mel=mels,
                    use_vocoder=False,
                )

                if predicted_mel.dim() == 3:
                    predicted_mel = predicted_mel.transpose(1, 2)
                predicted_mel = _align_predicted_mel_to_target(predicted_mel, mels)

                loss, recon_loss, div_loss = criterion(
                    predicted_mel=predicted_mel,
                    target_mel=mels,
                    style_embeddings=style_embeddings,
                )

                loss.backward()
                optimizer.step()

            metrics.add(
                loss=loss.item(),
                recon_loss=recon_loss.item(),
                div_loss=div_loss.item(),
            )

            pbar.set_postfix({"loss": f"{metrics.get_averages()['loss']:.4f}"}, refresh=True)

        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            continue

    pbar.close()
    if "loss" not in metrics.metrics:
        raise RuntimeError("No training batches produced valid losses. Check batch error logs above.")
    return metrics.get_averages()


def validate_epoch(
    model: FirstStepTTSModel,
    val_loader: DataLoader,
    criterion: CombinedTTSLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
    """Validate one epoch."""
    model.eval()
    metrics = MetricsTracker()

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1}/{max_epochs} [VAL]",
        leave=True,
        total=len(val_loader),
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            mels = batch["mel"].to(device)
            batch_size = mels.shape[0]
            max_text_len = 256
            text_ids = torch.randint(0, 1000, (batch_size, max_text_len)).to(device)

            try:
                predicted_mel, style_embeddings = model(
                    text_ids=text_ids,
                    target_mel=mels,
                    use_vocoder=False,
                )

                if predicted_mel.dim() == 3:
                    predicted_mel = predicted_mel.transpose(1, 2)
                predicted_mel = _align_predicted_mel_to_target(predicted_mel, mels)

                loss, recon_loss, div_loss = criterion(
                    predicted_mel=predicted_mel,
                    target_mel=mels,
                    style_embeddings=style_embeddings,
                )

                metrics.add(
                    loss=loss.item(),
                    recon_loss=recon_loss.item(),
                    div_loss=div_loss.item(),
                )

                pbar.set_postfix({"loss": f"{metrics.get_averages()['loss']:.4f}"}, refresh=True)

            except Exception as e:
                print(f"\nError in validation batch {batch_idx}: {e}")
                continue

    pbar.close()
    if "loss" not in metrics.metrics:
        raise RuntimeError("No validation batches produced valid losses. Check batch error logs above.")
    return metrics.get_averages()


def save_checkpoint(
    model: FirstStepTTSModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt",
) -> Path:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Filename for checkpoint
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    
    return checkpoint_path


def load_checkpoint(
    model: FirstStepTTSModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, Dict[str, float]]:
    """Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        Tuple of (starting_epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint.get("metrics", {})


class TensorBoardLogger:
    """TensorBoard logging utility."""
    
    def __init__(self, log_dir: Path):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step (epoch or iteration)
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        for key, value in metrics.items():
            tag = f"{prefix}{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

    def _normalize_image_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale a tensor to [0, 1] for TensorBoard image logging."""
        tensor = tensor.detach().float().cpu()
        min_value = tensor.min()
        max_value = tensor.max()
        if torch.isclose(max_value, min_value):
            return torch.zeros_like(tensor)
        return (tensor - min_value) / (max_value - min_value)

    def log_audio_examples(
        self,
        step: int,
        sample_rate: int,
        original_audio: torch.Tensor,
        original_mel: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        predicted_audio: torch.Tensor,
        predicted_mel: torch.Tensor,
        prefix: str = "examples/",
    ):
        """Log audio and mel comparisons for a single validation example."""
        self.writer.add_audio(f"{prefix}original_audio", original_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}reconstructed_from_original_mel", reconstructed_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}predicted_from_model_mel", predicted_audio.detach().cpu(), step, sample_rate=sample_rate)

        original_mel_img = self._normalize_image_tensor(original_mel)
        predicted_mel_img = self._normalize_image_tensor(predicted_mel)
        self.writer.add_image(f"{prefix}original_mel", original_mel_img.unsqueeze(0), step)
        self.writer.add_image(f"{prefix}predicted_mel", predicted_mel_img.unsqueeze(0), step)

    def log_model_info(self, model: FirstStepTTSModel):
        """Log model information.

        Args:
            model: Model to log info for
        """
        trainable = sum(p.numel() for p in model.get_trainable_parameters())
        total = sum(p.numel() for p in model.parameters())

        self.writer.add_text(
            "model/info",
            f"Trainable: {trainable:,} | Total: {total:,}",
        )

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters.

        Args:
            hparams: Hyperparameters dictionary
            metrics: Final metrics dictionary
        """
        self.writer.add_hparams(hparams, metrics)


def log_validation_audio_examples(
    model: FirstStepTTSModel,
    batch: Dict[str, Any],
    device: torch.device,
    logger: TensorBoardLogger,
    step: int,
):
    """Log a representative validation example to TensorBoard."""
    if "waveform" not in batch or batch["waveform"] is None:
        return

    mels = batch["mel"].to(device)
    waveforms = batch["waveform"].to(device)
    sr_value = batch.get("sr", 22050)
    if isinstance(sr_value, torch.Tensor):
        sample_rate = int(sr_value[0].item())
    elif isinstance(sr_value, (list, tuple)):
        sample_rate = int(sr_value[0])
    else:
        sample_rate = int(sr_value)

    original_audio = waveforms[0].squeeze().float().clamp(-1.0, 1.0)
    original_mel = mels[0:1]

    batch_size = mels.shape[0]
    text_ids = torch.randint(0, 1000, (batch_size, 256), device=device)

    with torch.no_grad():
        predicted_mel, _ = model(
            text_ids=text_ids,
            target_mel=mels,
            use_vocoder=False,
        )
        if predicted_mel.dim() == 3:
            predicted_mel = predicted_mel.transpose(1, 2)
        predicted_mel = _align_predicted_mel_to_target(predicted_mel, mels)

        reconstructed_audio = model.vocoder(spec=original_mel)
        predicted_audio = model.vocoder(spec=predicted_mel[0:1])

    def _to_audio_1d(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu().squeeze().clamp(-1.0, 1.0)

    logger.log_audio_examples(
        step=step,
        sample_rate=sample_rate,
        original_audio=_to_audio_1d(original_audio),
        original_mel=original_mel.detach().float().cpu().squeeze(0),
        reconstructed_audio=_to_audio_1d(reconstructed_audio),
        predicted_audio=_to_audio_1d(predicted_audio),
        predicted_mel=predicted_mel[0].detach().float().cpu(),
        prefix="examples/",
    )
    
    def flush(self):
        """Flush TensorBoard logs."""
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
