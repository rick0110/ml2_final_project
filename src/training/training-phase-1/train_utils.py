from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_experiment_dir(experiment_name: Optional[str] = None) -> Path:
    experiments_root = PROJECT_ROOT / "experiments" / "phase_1"
    experiments_root.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        experiment_name = f"phase_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    named_candidate = PROJECT_ROOT / "experiments" / "phase_1" / experiment_ref
    if named_candidate.exists():
        return named_candidate

    raise FileNotFoundError(f"Experiment directory not found: {experiment_ref}")


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
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


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model_state: dict,
    optimizer_state: dict,
    disc_state: Optional[dict],
    metrics: Dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "discriminator_state_dict": disc_state,
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device)


def save_config(config_path: Path, config: dict) -> None:
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding="utf-8")


def build_mel_transform(
    sample_rate: int = 22050,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=0,
        f_max=8000,
    )


def match_length(mel: torch.Tensor, target_len: int) -> torch.Tensor:
    if mel.size(-1) == target_len:
        return mel
    if mel.size(-1) > target_len:
        return mel[..., :target_len]
    pad = target_len - mel.size(-1)
    return torch.nn.functional.pad(mel, (0, pad), mode="constant", value=0.0)


@dataclass
class MetricsTracker:
    values: Dict[str, list]

    def __init__(self) -> None:
        self.values = {}

    def add(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.values.setdefault(key, []).append(float(value))

    def averages(self) -> Dict[str, float]:
        return {key: sum(values) / max(1, len(values)) for key, values in self.values.items()}


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)

    def log_histogram(self, tag: str, tensor: torch.Tensor, step: int) -> None:
        data = tensor.detach().float().cpu()
        if data.numel() > 0:
            self.writer.add_histogram(tag, data, step)

    def log_tensor_report(self, tag: str, tensors: Dict[str, torch.Tensor], step: int) -> None:
        lines = []
        for name, tensor in tensors.items():
            data = tensor.detach().float().cpu()
            if data.numel() == 0:
                lines.append(f"{name}: empty")
                continue
            mean = float(data.mean().item())
            std = float(data.std(unbiased=False).item()) if data.numel() > 1 else 0.0
            min_value = float(data.min().item())
            max_value = float(data.max().item())
            lines.append(
                f"{name}: shape={tuple(data.shape)} mean={mean:.6f} std={std:.6f} min={min_value:.6f} max={max_value:.6f}"
            )
            self.writer.add_histogram(f"{tag}/{name}", data, step)
        self.writer.add_text(tag, "\n".join(lines), step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        data = image.detach().float().cpu()
        if data.dim() == 2:
            data = data.unsqueeze(0)
        elif data.dim() == 3:
            if data.size(0) == 1:
                data = data.repeat(3, 1, 1)
            elif data.size(0) > 3:
                data = data[:3]
        else:
            raise ValueError(f"image must be 2D or 3D, got shape {tuple(data.shape)}")

        min_value = float(data.min().item())
        max_value = float(data.max().item())
        if abs(max_value - min_value) < 1e-12:
            normalized = torch.zeros_like(data)
        else:
            normalized = (data - min_value) / (max_value - min_value)
        self.writer.add_image(tag, normalized, step)

    def log_audio(self, tag: str, audio: torch.Tensor, step: int, sample_rate: int = 22050) -> None:
        data = audio.detach().float().cpu()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        elif data.dim() != 2:
            raise ValueError(f"audio must be 1D or 2D, got shape {tuple(data.shape)}")
        self.writer.add_audio(tag, data.clamp(-1.0, 1.0), step, sample_rate=sample_rate)

    def log_gradient_summary(self, model: torch.nn.Module, step: int, prefix: str) -> None:
        gradient_norms = []
        lines = []
        global_norm_sq = 0.0

        for name, parameter in model.named_parameters():
            param = parameter.detach().float().cpu()
            param_norm = float(param.norm().item()) if param.numel() > 0 else 0.0

            if parameter.grad is None:
                # Explicitly log missing gradients as zero so it's visible in TensorBoard/exports
                grad_norm = 0.0
                grad_mean = 0.0
                grad_std = 0.0
                self.writer.add_scalar(f"{prefix}/layer_grad_norm/{name}", grad_norm, step)
                self.writer.add_scalar(f"{prefix}/layer_param_norm/{name}", param_norm, step)
                lines.append(f"{name}: NO_GRAD param_norm={param_norm:.6f} shape={tuple(parameter.shape)}")
                continue

            grad = parameter.grad.detach().float().cpu()
            grad_norm = float(grad.norm().item())
            grad_mean = float(grad.mean().item())
            grad_std = float(grad.std(unbiased=False).item()) if grad.numel() > 1 else 0.0

            gradient_norms.append(grad_norm)
            global_norm_sq += grad_norm * grad_norm
            self.writer.add_scalar(f"{prefix}/layer_grad_norm/{name}", grad_norm, step)
            self.writer.add_scalar(f"{prefix}/layer_param_norm/{name}", param_norm, step)

            lines.append(
                f"{name}: grad_norm={grad_norm:.6f} param_norm={param_norm:.6f} grad_mean={grad_mean:.6f} grad_std={grad_std:.6f} shape={tuple(parameter.shape)}"
            )

        # Always write a text summary so NO_GRAD entries are visible even if no grads exist
        if gradient_norms:
            norms_tensor = torch.tensor(gradient_norms, dtype=torch.float32)
            self.writer.add_histogram(f"{prefix}/layer_grad_norm_distribution", norms_tensor, step)
            self.writer.add_scalar(f"{prefix}/global_grad_norm", global_norm_sq ** 0.5, step)
            self.writer.add_scalar(f"{prefix}/mean_layer_grad_norm", float(norms_tensor.mean().item()), step)
            self.writer.add_scalar(f"{prefix}/max_layer_grad_norm", float(norms_tensor.max().item()), step)

        # Summary text (always present)
        summary_text = "\n".join(lines) if lines else "(no parameters)"
        self.writer.add_text(f"{prefix}/summary", summary_text, step)

        # Also log a compact list of missing-gradient parameter names and counts
        missing = [ln.split(":", 1)[0] for ln in lines if "NO_GRAD" in ln]
        self.writer.add_text(f"{prefix}/no_grad_list", "\n".join(missing) if missing else "(none)", step)
        self.writer.add_scalar(f"{prefix}/num_no_grad", len(missing), step)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        for key, value in metrics.items():
            name = f"{prefix}{key}" if prefix else key
            self.writer.add_scalar(name, value, step)

    def log_hyperparameters(self, hparams: Dict[str, float], metrics: Dict[str, float]) -> None:
        sanitized = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        self.writer.add_hparams(sanitized, metrics)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
