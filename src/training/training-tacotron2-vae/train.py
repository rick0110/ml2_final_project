#!/usr/bin/env python3
"""Tacotron2-VAE training script using loader_TTS_GST preprocessed data."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "loader_vae_tacotron"))


from losses import Tacotron2LossVAE
from models.tacotron2_vae.hparams import Tacotron2VAEHparams, create_hparams
from models.tacotron2_vae.model import load_tacotron2_vae_model
from text_processing import TextProcessor
from train_utils import (
    TensorBoardLogger,
    load_checkpoint,
    save_checkpoint,
    save_hparams,
    train_epoch,
    validate_epoch,
)
from utils import ARTIFACTS_DIR, TextMelCollate, create_dataset, create_dataloaders, create_experiment_dir

from loader_tacotron import DatasetLibriSpeechTacotronVAE
from torch.utils.data import random_split

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--iters-per-checkpoint", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip-thresh", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--anneal-function", type=str, default="logistic")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory with train.csv, val.csv and symbols.json from preprocess.py",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts_dir = Path(args.artifacts_dir)
    train_file = artifacts_dir / "train.csv"
    val_file = artifacts_dir / "val.csv"
    symbols_file = artifacts_dir / "symbols.json"

    if not train_file.exists() or not symbols_file.exists():
        raise FileNotFoundError(
            f"Missing preprocess artifacts in {artifacts_dir}. Run preprocess.py first."
        )

    text_processor = TextProcessor.load(symbols_file)
    hparams = create_hparams(
        {
            "epochs": args.epochs,
            "iters_per_checkpoint": args.iters_per_checkpoint,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip_thresh": args.grad_clip_thresh,
            "seed": args.seed,
            "anneal_function": args.anneal_function,
            "n_symbols": text_processor.n_symbols,
            "training_files": str(train_file),
            "validation_files": str(val_file),
        }
    )

    experiment_dir = (
        Path(args.resume_experiment)
        if args.resume_experiment
        else create_experiment_dir(args.experiment_name)
    )
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"
    save_hparams(hparams, experiment_dir / "hparams.json")
    text_processor.save(experiment_dir / "symbols.json")

    # 1. Instancia o dataset completo passando o text_processor
    full_dataset = DatasetLibriSpeechTacotronVAE(
        text_processor=text_processor, 
        data_dir=Path("data/processed/libriSpeech-en-tacotron-vae")
    )

    # 2. Divide os dados (ex: 90% para treino, 10% para validação usando o val_split dos argumentos)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=hparams.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = load_tacotron2_vae_model(hparams, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    criterion = Tacotron2LossVAE(hparams)
    tb_logger = TensorBoardLogger(tensorboard_dir)
    tb_logger.log_model_info(model)

    iteration = 0
    learning_rate = hparams.learning_rate
    if args.checkpoint_path:
        model, optimizer, learning_rate, iteration = load_checkpoint(
            Path(args.checkpoint_path), model, optimizer
        )
        iteration += 1

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    model.train()
    for epoch in range(hparams.epochs):
        print(f"Epoch: {epoch}")
        for batch in train_loader:
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
                tb_logger.log_training(
                    reduced_loss,
                    float(grad_norm),
                    learning_rate,
                    duration,
                    recon_loss.item(),
                    kl_loss.item(),
                    float(kl_weight),
                    iteration,
                )

            if iteration % hparams.iters_per_checkpoint == 0:
                val_loss = validate_epoch(model, criterion, val_loader, device, iteration)
                print(f"Validation loss {iteration}: {val_loss:9f}")
                tb_logger.log_validation(val_loss, iteration)
                checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}"
                save_checkpoint(
                    model, optimizer, learning_rate, iteration, checkpoint_path, hparams
                )

            iteration += 1

    tb_logger.close()
    print(f"Training finished. Experiment dir: {experiment_dir}")


if __name__ == "__main__":
    main()
