import sys
import io
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from losses import CombinedTTSLoss

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


def _align_predicted_mel_to_target(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    target_time = target_mel.size(2)
    pred_time = predicted_mel.size(2)

    if pred_time == target_time:
        return predicted_mel

    if pred_time > target_time:
        return predicted_mel[:, :, :target_time]
    else:
        pad_amount = target_time - pred_time
        return F.pad(predicted_mel, (0, pad_amount), value=0.0)
    
def _align_predicted_audio_to_target(predicted_audio: torch.Tensor, target_audio: Optional[torch.Tensor] = None) -> torch.Tensor:
    if target_audio is None:
        return predicted_audio
    target_time = target_audio.size(-1)
    pred_time = predicted_audio.size(-1)

    if pred_time == target_time:
        return predicted_audio

    if pred_time > target_time:
        return predicted_audio[..., :target_time]
    else:
        pad_amount = target_time - pred_time
        return F.pad(predicted_audio, (0, pad_amount), value=0.0)


def pad_sequence(sequences, padding_value):
    sequences = [seq.squeeze(0) if seq.dim() > 1 else seq for seq in sequences]
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        pad_amount = max_len - seq.size(0)
        padded_seq = F.pad(seq, (0, pad_amount), value=padding_value)
        padded_seqs.append(padded_seq)
    return torch.stack(padded_seqs)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedTTSLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
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
        if mels.dim() == 4 and mels.size(1) == 1:
            mels = mels.squeeze(1)
            
        texts = batch["text"]
        
        was_training = model.spec_generator.training
        model.spec_generator.eval()
        with torch.no_grad():
            parsed_texts = [torch.tensor(model.spec_generator.parse(t)) for t in texts]
        if was_training:
            model.spec_generator.train()
            
        padding_value = model.spec_generator.fastpitch.encoder.padding_idx
        text_ids = pad_sequence(parsed_texts, padding_value=padding_value).to(device)

        optimizer.zero_grad()

        audio, mel = model(text_tokens=text_ids, audio_inputs=mels, return_att_weights=False)

        predicted_mel = _align_predicted_mel_to_target(mel, mels)
        global_tokens = model.gst.style_tokens if hasattr(model, 'gst') else None

        loss, recon_loss, div_loss = criterion(
            predicted_mel=predicted_mel,
            target_mel=mels,
            global_style_tokens=global_tokens,
        )
        
        loss.backward()
        optimizer.step()

        metrics.add(
            loss=loss.item(),
            recon_loss=recon_loss.item(),
            div_loss=div_loss.item(),
        )

        pbar.set_postfix({"loss": f"{metrics.get_averages()['loss']:.4f}"}, refresh=True)

    pbar.close()
    return metrics.get_averages()


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: CombinedTTSLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
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
            if mels.dim() == 4 and mels.size(1) == 1:
                mels = mels.squeeze(1)
                
            texts = batch["text"]
            parsed_texts = [torch.tensor(model.spec_generator.parse(t)) for t in texts]
            padding_value = model.spec_generator.fastpitch.encoder.padding_idx
            text_ids = pad_sequence(parsed_texts, padding_value=padding_value).to(device)

            try:
                outputs = model(text_tokens=text_ids, audio_inputs=mels, return_att_weights=False)
                audio, mel = outputs

                predicted_mel = _align_predicted_mel_to_target(mel, mels)
                global_tokens = model.gst.style_tokens if hasattr(model, 'gst') else None

                loss, recon_loss, div_loss = criterion(
                    predicted_mel=predicted_mel,
                    target_mel=mels,
                    global_style_tokens=global_tokens,
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
    return metrics.get_averages()


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], epoch: int, metrics: Dict[str, float], checkpoint_dir: Path, filename: str = "checkpoint.pt") -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], checkpoint_path: Path, device: torch.device) -> Tuple[int, Dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint.get("metrics", {})


def find_last_epoch(checkpoint_dir: Path) -> Optional[str]:
    checkpoint_files = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoint_files:
        return None
    num_epochs = -1
    file_path_name = None
    for file in checkpoint_files:
        try:
            epoch_num = int(file.stem.split("_")[1])
            if epoch_num > num_epochs:
                num_epochs = epoch_num
                file_path_name = file.name
        except (IndexError, ValueError):
            continue
    return file_path_name


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for key, value in metrics.items():
            tag = f"{prefix}{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

    def _normalize_image_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().float().cpu()
        min_value = tensor.min()
        max_value = tensor.max()
        if torch.isclose(max_value, min_value):
            return torch.zeros_like(tensor)
        return (tensor - min_value) / (max_value - min_value)

    def log_audio_examples(self, step: int, sample_rate: int, original_audio: torch.Tensor, original_mel: torch.Tensor, reconstructed_audio: torch.Tensor, predicted_audio: torch.Tensor, predicted_mel: torch.Tensor, prefix: str = "examples/"):
        self.writer.add_audio(f"{prefix}original_audio", original_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}reconstructed_from_original_mel", reconstructed_audio.detach().cpu(), step, sample_rate=sample_rate)
        self.writer.add_audio(f"{prefix}predicted_from_model_mel", predicted_audio.detach().cpu(), step, sample_rate=sample_rate)

        original_mel_img = self._normalize_image_tensor(original_mel)
        predicted_mel_img = self._normalize_image_tensor(predicted_mel)
        self.writer.add_image(f"{prefix}original_mel", original_mel_img.unsqueeze(0), step)
        self.writer.add_image(f"{prefix}predicted_mel", predicted_mel_img.unsqueeze(0), step)

    def log_gst_interpretability(self, step: int, style_tokens: Optional[torch.Tensor] = None, attention_weights: Optional[torch.Tensor] = None, prefix: str = "interpretability/"):
        if style_tokens is not None:
            tokens_np = style_tokens.detach().cpu().numpy()
            norm = np.linalg.norm(tokens_np, axis=1, keepdims=True)
            norm_tokens = tokens_np / (norm + 1e-8)
            sim_matrix = np.dot(norm_tokens, norm_tokens.T)

            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
            fig.colorbar(cax)
            ax.set_title("GST Tokens Cosine Similarity")
            ax.set_xlabel("Token ID")
            ax.set_ylabel("Token ID")
            self.writer.add_figure(f"{prefix}token_similarity", fig, step)
            plt.close(fig)

        if attention_weights is not None:
            attn_np = attention_weights.detach().cpu().numpy()
            while len(attn_np.shape) > 1:
                attn_np = attn_np.mean(axis=0)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(np.arange(len(attn_np)), attn_np, color='skyblue')
            ax.set_title("Average GST Attention Weights")
            ax.set_xlabel("Token ID")
            ax.set_ylabel("Attention Weight")
            ax.set_xticks(np.arange(len(attn_np)))
            self.writer.add_figure(f"{prefix}attention_distribution", fig, step)
            plt.close(fig)

    def log_model_info(self, model: nn.Module):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.writer.add_text("model/info", f"Trainable: {trainable:,} | Total: {total:,}")

    def flush(self): self.writer.flush()
    def close(self): self.writer.close()


def log_validation_audio_examples(model: nn.Module, vocoder: nn.Module, latent_transfer_model: Optional[nn.Module], batch: Dict[str, Any], device: torch.device, logger: TensorBoardLogger, step: int):
    if "waveform" not in batch or batch["waveform"] is None:
        return

    mels = batch["mel"].to(device)
    if mels.dim() == 4 and mels.size(1) == 1:
        mels = mels.squeeze(1)
        
    waveforms = batch["waveform"].to(device)
    sr_value = batch.get("sr", 22050)
    sample_rate = int(sr_value[0].item()) if isinstance(sr_value, torch.Tensor) else int(sr_value[0])

    original_audio = waveforms[0].squeeze().float().clamp(-1.0, 1.0)
    original_mel = mels[0:1]

    texts = batch["text"]
    parsed_texts = [torch.tensor(model.spec_generator.parse(t)) for t in texts]
    padding_value = model.spec_generator.fastpitch.encoder.padding_idx
    text_ids = pad_sequence(parsed_texts, padding_value=padding_value).to(device)

    with torch.no_grad():
        outputs = model(text_tokens=text_ids, audio_inputs=mels, return_att_weights=True)
        if len(outputs) == 3:
            _, predicted_mel, attention_weights = outputs
        else:
            _, predicted_mel = outputs
            attention_weights = None

        predicted_mel = _align_predicted_mel_to_target(predicted_mel, mels)
        
        # APLICAÇÃO DO TRANSFERIDOR LATENTE AQUI
        if latent_transfer_model is not None:
            mapped_mel = latent_transfer_model(predicted_mel)
        else:
            mapped_mel = predicted_mel
            
        # CORREÇÃO 1: Passar apenas o primeiro item do batch [0:1] para o vocoder.
        # Isso economiza muita memória e deixa a validação mais rápida!
        predicted_audio = vocoder(spec=mapped_mel[0:1])
        reconstructed_audio = vocoder(spec=original_mel)

    # CORREÇÃO 2: Garantir que o tensor seja sempre 1D, pegando o índice [0] se for um batch
    def _to_audio_1d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None: return torch.zeros(1)
        t = tensor.detach().float().cpu().squeeze()
        if t.dim() > 1:
            t = t[0] # Força pegar apenas o primeiro áudio
        return t.clamp(-1.0, 1.0)

    logger.log_audio_examples(
        step=step,
        sample_rate=sample_rate,
        original_audio=_to_audio_1d(original_audio),
        original_mel=original_mel.detach().float().cpu().squeeze(0),
        reconstructed_audio=_to_audio_1d(reconstructed_audio),
        predicted_audio=_to_audio_1d(predicted_audio), 
        predicted_mel=mapped_mel[0].detach().float().cpu(),
        prefix="examples/",
    )

    style_tokens = model.gst.style_tokens if hasattr(model, 'gst') else None
    logger.log_gst_interpretability(step=step, style_tokens=style_tokens, attention_weights=attention_weights)