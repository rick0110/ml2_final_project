#!/usr/bin/env python3
"""Training script for the direct text-to-speech model.

Pipeline:
Text -> text encoder -> length predictor -> frame expansion -> mel decoder -> mel projection

This script mirrors the experiment management, validation logging, and
diagnostic style of train_try_2, but the model itself is text-only and does
not consume audio as an input feature.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "last-model"))

from models.DirectTTS import get_model_size_info, load_direct_tts_model
from training.directly_tts.losses import DirectTTSLoss
from training.directly_tts.tokenizer import BatchTextTokenizer
from last_model_data import create_dataloaders


class MetricsTracker:
    def __init__(self):
        self.values: Dict[str, list[float]] = {}

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.values.setdefault(key, []).append(float(value))

    def averages(self) -> Dict[str, float]:
        return {key: sum(values) / len(values) for key, values in self.values.items()}


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

    def log_spectrogram_examples(
        self,
        step: int,
        target_mel: torch.Tensor,
        predicted_mel: torch.Tensor,
        prefix: str = "examples/",
    ):
        import matplotlib.pyplot as plt

        target_mel = target_mel.detach().float().cpu().squeeze(0)
        predicted_mel = predicted_mel.detach().float().cpu().squeeze(0)
        abs_error = (predicted_mel - target_mel).abs()

        self.writer.add_image(f"{prefix}target_mel", self._normalize_image_tensor(target_mel).unsqueeze(0), step)
        self.writer.add_image(f"{prefix}predicted_mel", self._normalize_image_tensor(predicted_mel).unsqueeze(0), step)
        self.writer.add_image(f"{prefix}mel_abs_error", self._normalize_image_tensor(abs_error).unsqueeze(0), step)

        figure, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
        panels = [(target_mel, "Target Mel"), (predicted_mel, "Predicted Mel"), (abs_error, "Abs Error")]
        for axis, (panel, title) in zip(axes, panels):
            image = axis.imshow(panel, aspect="auto", origin="lower", interpolation="nearest")
            axis.set_title(title)
            axis.set_xlabel("Time")
            axis.set_ylabel("Mel")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        self.writer.add_figure(f"{prefix}spectrogram_comparison", figure, step)
        plt.close(figure)

    def log_audio_examples(
        self,
        step: int,
        sample_rate: int,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        predicted_audio: torch.Tensor,
        prefix: str = "examples/",
    ):
        self.writer.add_audio(f"{prefix}original_audio", original_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(
            f"{prefix}reconstructed_from_target_mel",
            reconstructed_audio.detach().cpu(),
            step,
            sample_rate=sample_rate,
        )
        self.writer.add_audio(
            f"{prefix}predicted_from_model_mel",
            predicted_audio.detach().cpu(),
            step,
            sample_rate=sample_rate,
        )

    def log_state_analysis(
        self,
        step: int,
        state_name: str,
        state_tensor: torch.Tensor,
        prefix: str = "analysis/",
        max_points: int = 1024,
        tsne_points: int = 256,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        tensor = state_tensor.detach().float().cpu()
        if tensor.dim() == 3:
            matrix = tensor.reshape(-1, tensor.size(-1))
        elif tensor.dim() == 2:
            matrix = tensor
        else:
            return

        if matrix.size(0) < 3 or matrix.size(1) < 2:
            return

        matrix_np = matrix.numpy()
        seed = step + sum(ord(character) for character in state_name)
        rng = np.random.default_rng(seed)
        if matrix_np.shape[0] > max_points:
            sample_indices = rng.choice(matrix_np.shape[0], size=max_points, replace=False)
            matrix_np = matrix_np[sample_indices]

        centered = matrix_np - matrix_np.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)

        n_components = min(matrix_np.shape[0], matrix_np.shape[1])
        pca = PCA(n_components=n_components, svd_solver="full")
        pca_scores = pca.fit_transform(matrix_np)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        abs_scores = np.abs(pca_scores)
        pca_cv = abs_scores.std(axis=0) / (abs_scores.mean(axis=0) + 1e-8)

        effective_rank = float(np.exp(-np.sum(explained_variance_ratio * np.log(explained_variance_ratio + 1e-12))))
        components_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
        components_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
        components_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)

        self.writer.add_histogram(f"{prefix}{state_name}/singular_values", singular_values, step)
        self.writer.add_histogram(f"{prefix}{state_name}/pca_cv", pca_cv, step)
        self.writer.add_scalar(f"{prefix}{state_name}/effective_rank", effective_rank, step)
        self.writer.add_scalar(f"{prefix}{state_name}/components_90", components_90, step)
        self.writer.add_scalar(f"{prefix}{state_name}/components_95", components_95, step)
        self.writer.add_scalar(f"{prefix}{state_name}/components_99", components_99, step)
        self.writer.add_scalar(f"{prefix}{state_name}/first_component_ratio", float(explained_variance_ratio[0]), step)

        tsne_samples = min(tsne_points, pca_scores.shape[0])
        tsne_embedding = None
        if tsne_samples >= 3:
            tsne_indices = np.arange(pca_scores.shape[0])
            if pca_scores.shape[0] > tsne_samples:
                tsne_indices = rng.choice(pca_scores.shape[0], size=tsne_samples, replace=False)
            tsne_input = pca_scores[tsne_indices, : min(50, pca_scores.shape[1])]
            perplexity = max(2, min(30, tsne_input.shape[0] - 1))
            if perplexity < tsne_input.shape[0]:
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    init="pca",
                    learning_rate="auto",
                    random_state=seed,
                )
                tsne_embedding = tsne.fit_transform(tsne_input)

        figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        axes = axes.ravel()

        bins = min(40, max(10, singular_values.shape[0] // 2))
        axes[0].hist(singular_values, bins=bins, color="#1f77b4", alpha=0.9)
        axes[0].set_title(f"{state_name} - Singular Values")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        axes[1].hist(pca_cv, bins=bins, color="#ff7f0e", alpha=0.9)
        axes[1].set_title(f"{state_name} - PCA Coefficient of Variation")
        axes[1].set_xlabel("CV")
        axes[1].set_ylabel("Count")

        component_axis = np.arange(1, cumulative_variance.shape[0] + 1)
        axes[2].plot(component_axis, explained_variance_ratio, marker="o", linewidth=1.5, label="Explained variance")
        axes[2].plot(component_axis, cumulative_variance, linewidth=2.0, label="Cumulative variance")
        axes[2].axhline(0.90, color="gray", linestyle="--", linewidth=1)
        axes[2].axhline(0.95, color="gray", linestyle=":", linewidth=1)
        axes[2].set_title(f"{state_name} - PCA Spectrum")
        axes[2].set_xlabel("Component")
        axes[2].set_ylabel("Variance ratio")
        axes[2].set_ylim(0.0, 1.05)
        axes[2].legend(loc="best")

        if tsne_embedding is not None:
            scatter = axes[3].scatter(
                tsne_embedding[:, 0],
                tsne_embedding[:, 1],
                c=np.arange(tsne_embedding.shape[0]),
                cmap="viridis",
                s=14,
                alpha=0.85,
            )
            axes[3].set_title(f"{state_name} - t-SNE")
            axes[3].set_xlabel("Dim 1")
            axes[3].set_ylabel("Dim 2")
            figure.colorbar(scatter, ax=axes[3], fraction=0.046, pad=0.04)
        else:
            axes[3].axis("off")
            axes[3].text(0.5, 0.5, "t-SNE skipped\n(not enough samples)", ha="center", va="center")

        self.writer.add_figure(f"{prefix}{state_name}/summary", figure, step)
        plt.close(figure)

    def log_state_analysis_bundle(
        self,
        step: int,
        state_tensors: Dict[str, torch.Tensor],
        prefix: str = "analysis/",
        enabled: bool = True,
        max_points: int = 1024,
        tsne_points: int = 256,
    ):
        if not enabled:
            return
        for state_name, state_tensor in state_tensors.items():
            self.log_state_analysis(
                step=step,
                state_name=state_name,
                state_tensor=state_tensor,
                prefix=prefix,
                max_points=max_points,
                tsne_points=tsne_points,
            )

    def log_model_info(self, model):
        trainable = sum(parameter.numel() for parameter in model.get_trainable_parameters())
        total = sum(parameter.numel() for parameter in model.parameters())
        self.writer.add_text("model/info", f"Trainable: {trainable:,} | Total: {total:,}")

    def log_hyperparameters(self, hparams: Dict[str, float], metrics: Dict[str, float]):
        self.writer.add_hparams(hparams, metrics)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


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


def _trim_to_length(mel: torch.Tensor, length: torch.Tensor | int) -> torch.Tensor:
    if isinstance(length, torch.Tensor):
        length = int(length.item())
    return mel[..., : max(1, int(length))]


def log_validation_examples(
    model,
    tokenizer,
    vocoder,
    val_loader,
    device: torch.device,
    logger: TensorBoardLogger,
    step: int,
    analysis_max_points: int = 1024,
    analysis_tsne_points: int = 256,
):
    batch = next(iter(val_loader))
    mels = batch["mel"].to(device)
    waveforms = batch["waveform"].to(device)
    mel_lengths = batch["mel_lengths"].to(device)
    sr_value = batch.get("sr", 22050)
    if isinstance(sr_value, torch.Tensor):
        sample_rate = int(sr_value[0].item())
    elif isinstance(sr_value, (list, tuple)):
        sample_rate = int(sr_value[0])
    else:
        sample_rate = int(sr_value)

    text_ids = tokenizer.encode_batch(batch["text"]).to(device)

    with torch.no_grad():
        teacher_predicted_mel, teacher_predicted_lengths, teacher_intermediates = model.forward_with_intermediates(
            text_ids=text_ids,
            target_mel=mels,
            target_lengths=mel_lengths,
        )
        inference_predicted_mel, inference_predicted_lengths, _ = model.forward_with_intermediates(text_ids=text_ids[:1])

        teacher_predicted_mel = teacher_predicted_mel[..., : mels.size(-1)]
        teacher_predicted_mel = F.interpolate(
            teacher_predicted_mel,
            size=mels.size(-1),
            mode="linear",
            align_corners=False,
        ) if teacher_predicted_mel.size(-1) != mels.size(-1) else teacher_predicted_mel

        target_mel_0 = _trim_to_length(mels[0:1], mel_lengths[0])
        teacher_predicted_mel_0 = _trim_to_length(teacher_predicted_mel[0:1], mel_lengths[0])
        inference_predicted_mel_0 = inference_predicted_mel[0:1]
        inference_predicted_length_0 = inference_predicted_lengths[0]
        inference_predicted_mel_0 = _trim_to_length(inference_predicted_mel_0, inference_predicted_length_0)

        reconstructed_audio = vocoder(spec=target_mel_0)
        predicted_audio = vocoder(spec=inference_predicted_mel_0)

    logger.log_spectrogram_examples(
        step=step,
        target_mel=target_mel_0,
        predicted_mel=teacher_predicted_mel_0,
    )
    logger.log_audio_examples(
        step=step,
        sample_rate=sample_rate,
        original_audio=_to_audio_1d(waveforms[0]),
        reconstructed_audio=_to_audio_1d(reconstructed_audio),
        predicted_audio=_to_audio_1d(predicted_audio),
    )

    logger.log_state_analysis_bundle(
        step=step,
        state_tensors={
            "text_states": teacher_intermediates["text_states"],
            "expanded_states": teacher_intermediates["expanded_states"],
            "decoder_states": teacher_intermediates["decoder_states"],
            "pooled_text_state": teacher_intermediates["pooled_text_state"],
            "predicted_mel": teacher_predicted_mel,
        },
        enabled=True,
        max_points=analysis_max_points,
        tsne_points=analysis_tsne_points,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a direct text-to-speech model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-reconstruction", type=float, default=1.0)
    parser.add_argument("--weight-length", type=float, default=0.1)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)
    parser.add_argument("--analysis-interval", type=int, default=5)
    parser.add_argument("--analysis-max-points", type=int, default=1024)
    parser.add_argument("--analysis-tsne-points", type=int, default=256)
    return parser.parse_args()


def create_experiment_dir(experiment_name: str | None = None) -> Path:
    experiments_root = PROJECT_ROOT / "experiments" / "directly_tts"
    experiments_root.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        experiment_name = f"direct_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    named_candidate = PROJECT_ROOT / "experiments" / "directly_tts" / experiment_ref
    if named_candidate.exists():
        return named_candidate

    raise FileNotFoundError(f"Experiment directory not found: {experiment_ref}. Tried {candidate} and {named_candidate}")


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


def _align_time(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    if predicted_mel.dim() != 3 or target_mel.dim() != 3:
        raise ValueError(f"Expected 3D mel tensors, got predicted={predicted_mel.shape}, target={target_mel.shape}")

    if predicted_mel.size(-1) == target_mel.size(-1):
        return predicted_mel

    return F.interpolate(predicted_mel, size=target_mel.size(-1), mode="linear", align_corners=False)


def train_epoch(model, tokenizer, train_loader, optimizer, criterion, device, epoch: int, max_epochs: int):
    model.train()
    metrics = MetricsTracker()

    from tqdm import tqdm

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [TRAIN]", leave=True, total=len(train_loader))
    for batch in pbar:
        mels = batch["mel"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        text_ids = tokenizer.encode_batch(batch["text"]).to(device)

        optimizer.zero_grad()
        predicted_mel, predicted_lengths = model(text_ids=text_ids, target_mel=mels, target_lengths=mel_lengths)
        predicted_mel = _align_time(predicted_mel, mels)

        loss, recon_loss, length_loss = criterion(
            predicted_mel=predicted_mel,
            target_mel=mels,
            predicted_lengths=predicted_lengths,
            target_lengths=mel_lengths,
        )
        loss.backward()
        optimizer.step()

        metrics.add(loss=loss.item(), recon_loss=recon_loss.item(), length_loss=length_loss.item())
        averages = metrics.averages()
        pbar.set_postfix({"loss": f"{averages['loss']:.4f}"}, refresh=True)

    if "loss" not in metrics.values:
        raise RuntimeError("No training batches produced valid losses.")
    return metrics.averages()


def validate_epoch(model, tokenizer, val_loader, criterion, device, epoch: int, max_epochs: int):
    model.eval()
    metrics = MetricsTracker()

    from tqdm import tqdm

    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [VAL]", leave=True, total=len(val_loader))
    with torch.no_grad():
        for batch in pbar:
            mels = batch["mel"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            text_ids = tokenizer.encode_batch(batch["text"]).to(device)

            predicted_mel, predicted_lengths = model(text_ids=text_ids, target_mel=mels, target_lengths=mel_lengths)
            predicted_mel = _align_time(predicted_mel, mels)

            loss, recon_loss, length_loss = criterion(
                predicted_mel=predicted_mel,
                target_mel=mels,
                predicted_lengths=predicted_lengths,
                target_lengths=mel_lengths,
            )
            metrics.add(loss=loss.item(), recon_loss=recon_loss.item(), length_loss=length_loss.item())

            averages = metrics.averages()
            pbar.set_postfix({"loss": f"{averages['loss']:.4f}"}, refresh=True)

    if "loss" not in metrics.values:
        raise RuntimeError("No validation batches produced valid losses.")
    return metrics.averages()


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.resume_experiment is not None:
        experiment_dir = resolve_experiment_dir(args.resume_experiment)
    else:
        experiment_dir = create_experiment_dir(args.experiment_name)

    print(f"\nExperiment directory: {experiment_dir}")
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"

    hparams = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_min_lr": args.scheduler_min_lr,
        "num_workers": args.num_workers,
        "weight_reconstruction": args.weight_reconstruction,
        "weight_length": args.weight_length,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "n_mels": args.n_mels,
        "dropout": args.dropout,
        "text_max_length": args.text_max_length,
        "val_split": args.val_split,
        "seed": args.seed,
    }

    config_path = experiment_dir / "config.json"
    if args.resume_experiment is not None:
        saved_hparams = load_experiment_config(experiment_dir)
        for key, value in saved_hparams.items():
            if hasattr(args, key):
                setattr(args, key, value)
        hparams.update(saved_hparams)
        print(f"✓ Loaded config from {config_path}")
    else:
        config_path.write_text(json.dumps(hparams, indent=2), encoding="utf-8")
        print(f"✓ Config saved to {config_path}")

    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    tokenizer = BatchTextTokenizer(max_length=args.text_max_length)

    model = load_direct_tts_model(
        vocab_size=len(tokenizer.tokenizer),
        model_dim=args.model_dim,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        ff_dim=args.ff_dim,
        n_mels=args.n_mels,
        pad_idx=tokenizer.pad_idx,
        dropout=args.dropout,
    ).to(device)

    model_info = get_model_size_info(model)
    print(f"  Trainable parameters: {model_info['trainable']:,}")
    print(f"  Total parameters: {model_info['total']:,}")

    optimizer = SGD(model.get_trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    criterion = DirectTTSLoss(
        weight_reconstruction=args.weight_reconstruction,
        weight_length=args.weight_length,
    ).to(device)

    tb_logger = TensorBoardLogger(tensorboard_dir)
    tb_logger.log_model_info(model)
    tb_logger.log_hyperparameters(hparams, {})

    print("Loading HiFi_GAN vocoder for TensorBoard sample logging...")
    vocoder = load_hifigan_vocoder(device)
    print("  ✓ HiFi_GAN vocoder loaded (frozen)")

    start_epoch = 0
    if args.resume or args.resume_experiment:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir) if args.resume_experiment else Path(args.resume)
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)
        print(f"  Loaded from epoch {start_epoch}")

    best_val_loss = float("inf")
    print("\nStarting training...")

    for epoch in range(start_epoch, args.num_epochs):
        train_metrics = train_epoch(model, tokenizer, train_loader, optimizer, criterion, device, epoch, args.num_epochs)
        val_metrics = validate_epoch(model, tokenizer, val_loader, criterion, device, epoch, args.num_epochs)

        scheduler.step(val_metrics["loss"])

        tb_logger.log_metrics(train_metrics, epoch, prefix="train/")
        tb_logger.log_metrics(val_metrics, epoch, prefix="val/")
        if args.analysis_interval > 0 and epoch % args.analysis_interval == 0:
            log_validation_examples(
                model,
                tokenizer,
                vocoder,
                val_loader,
                device,
                tb_logger,
                epoch,
                analysis_max_points=args.analysis_max_points,
                analysis_tsne_points=args.analysis_tsne_points,
            )
        tb_logger.flush()

        print(f"\nEpoch {epoch + 1}/{args.num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"    ├─ Reconstruction: {train_metrics['recon_loss']:.6f}")
        print(f"    └─ Length: {train_metrics['length_loss']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"    ├─ Reconstruction: {val_metrics['recon_loss']:.6f}")
        print(f"    └─ Length: {val_metrics['length_loss']:.6f}")

        metrics = {**train_metrics, **{f"val_{key}": value for key, value in val_metrics.items()}}
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