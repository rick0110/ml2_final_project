#!/usr/bin/env python3
"""First-step TTS model training script with GST Interpretability.

Usage:
    python train.py --num-epochs 100 --batch-size 32 --learning-rate 1e-3
"""

import sys
import argparse
import json
import importlib.util
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.first_step_data_loaders.datasets import LibriSpeechPTDataset, TTSPortugueseDataset
from training.train_first_step.model_loader import load_tts_models
from training.train_first_step.text_processing import BatchTextTokenizer

# IMPORTANTE: Aponte para a pasta local onde você salvou os novos arquivos utilitários
from train_utils import train_epoch, validate_epoch, save_checkpoint, load_checkpoint, TensorBoardLogger, log_validation_audio_examples
from losses import CombinedTTSLoss


def load_hifigan_vocoder(device: torch.device) -> nn.Module:
    hifigan_path = PROJECT_ROOT / "src" / "models" / "HiFi-GAN.py"
    spec = importlib.util.spec_from_file_location("hifigan_module", hifigan_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, vocoder = module.load_hifigan_model(freeze=True)
    return vocoder.to(device).eval()

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-reconstruction", type=float, default=1.0)
    parser.add_argument("--weight-diversity", type=float, default=0.5)
    parser.add_argument("--diversity-margin", type=float, default=0.1)
    parser.add_argument("--acoustic-decoder-hidden-size", type=int, default=256)
    parser.add_argument("--acoustic-decoder-num-layers", type=int, default=3)
    parser.add_argument("--style-embedding-dim", type=int, default=128)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# (Mantenha as funções create_experiment_dir, load_experiment_config, find_latest_checkpoint, create_datasets e collate_fn idênticas ao original)
# Como o código era longo e você pediu foco na arquitetura principal, essas funções de parse de path são padrão.
def create_experiment_dir(experiment_name: Optional[str] = None) -> Path:
    experiments_root = PROJECT_ROOT / "experiments" / "step_1"
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = experiments_root / (experiment_name or f"attempt_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "tensorboard").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    return experiment_dir

def create_datasets(batch_size: int, num_workers: int, val_split: float = 0.1):
    combined_dataset = ConcatDataset([LibriSpeechPTDataset(split="train"), TTSPortugueseDataset()])
    val_size = int(len(combined_dataset) * val_split)
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [len(combined_dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Collate Simplificado (Copie a versão completa do seu arquivo original aqui)
    def collate_fn(batch):
        mels = [torch.as_tensor(s["mel"]).float().contiguous().squeeze(0) for s in batch]
        waveforms = [torch.as_tensor(s.get("waveform", torch.zeros(1))).float().contiguous() for s in batch]
        texts = [s.get("text", "") for s in batch]
        max_time = max(m.size(1) for m in mels)
        padded_mels = torch.stack([F.pad(m, (0, max_time - m.size(1))) for m in mels])
        return {"mel": padded_mels, "waveform": waveforms, "text": texts}
        
    return DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn), DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiment_dir = Path(args.resume_experiment) if args.resume_experiment else create_experiment_dir(args.experiment_name)
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"
    
    train_loader, val_loader = create_datasets(args.batch_size, args.num_workers, args.val_split)
    tokenizer = BatchTextTokenizer()
    
    model = load_tts_models(
        device=device,
        acoustic_decoder_hidden_size=args.acoustic_decoder_hidden_size,
        acoustic_decoder_num_layers=args.acoustic_decoder_num_layers,
        style_embedding_dim=args.style_embedding_dim,
        vocab_size=len(tokenizer.tokenizer)
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
    criterion = CombinedTTSLoss(weight_reconstruction=args.weight_reconstruction, weight_diversity=args.weight_diversity, diversity_margin=args.diversity_margin).to(device)
    
    tb_logger = TensorBoardLogger(tensorboard_dir)
    tb_logger.log_model_info(model)
    vocoder = load_hifigan_vocoder(device)
    
    start_epoch = 0
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.num_epochs):
        train_metrics = train_epoch(model, tokenizer, train_loader, optimizer, criterion, device, epoch, args.num_epochs)
        val_metrics = validate_epoch(model, tokenizer, val_loader, criterion, device, epoch, args.num_epochs)
        scheduler.step(val_metrics["loss"])

        if epoch == 0 or (epoch + 1) % 1 == 0:
            example_batch = next(iter(val_loader))
            log_validation_audio_examples(model, vocoder, example_batch, device, tb_logger, epoch)
        
        tb_logger.log_metrics(train_metrics, epoch, prefix="train/")
        tb_logger.log_metrics(val_metrics, epoch, prefix="val/")
        tb_logger.flush()
        
        save_checkpoint(model, optimizer, scheduler, epoch + 1, train_metrics, checkpoint_dir, f"epoch_{epoch+1:04d}.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics, checkpoint_dir, "best.pt")
            
    tb_logger.close()

if __name__ == "__main__":
    main()