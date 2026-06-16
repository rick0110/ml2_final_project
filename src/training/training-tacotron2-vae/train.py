#!/usr/bin/env python3
"""Tacotron2-VAE training script using loader_TTS_GST preprocessed data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv

import torch
from tqdm import tqdm

from utils import ARTIFACTS_DIR, TextMelCollate, create_dataloader, create_experiment_dir


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "loader_vae_tacotron"))


from losses import Tacotron2LossVAE
from models.tacotron2_vae.hparams import Tacotron2VAEHparams, create_hparams
from models.tacotron2_vae.model import load_tacotron2_vae_model
from text_processing import TextProcessor, build_symbols_from_texts
from train_utils import (
    TensorBoardLogger,
    load_checkpoint,
    save_checkpoint,
    save_hparams,
    train_epoch,
    find_latest_checkpoint
)

from loader_tacotron import load_data

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--iters-per-checkpoint", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
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
        help="Path with the data, metadata.csv",
    )
    return parser.parse_args()


def main():
    training_metadata = {
        "training_loss": [],
        "test_loss": [],
        "grad_norm": [],
        "learning_rate": [],
        "duration": [],
        "recon_loss": [],
        "kl_loss": [],
        "kl_weight": [],
        "singular_values_of_latent_covariance": [],
        "target_predict_example": [],

    }
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts_dir = Path(args.artifacts_dir)
    train_file = artifacts_dir / "mels_metadata.csv"
    
    with open(train_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        texts = [row["text"] for row in reader]

    symbols = build_symbols_from_texts(texts)

    text_processor = TextProcessor(symbols=symbols)

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
            "training_data": str(train_file),
        }
    )

    experiment_dir = (
        Path(args.resume_experiment)
        if args.resume_experiment
        else create_experiment_dir(args.experiment_name)
    )

    hparams.experiment_dir = str(experiment_dir)
    hparams.checkpoint_dir = str(experiment_dir / "checkpoints")
    tensorboard_dir = experiment_dir / "tensorboard"
    save_hparams(hparams, experiment_dir / "hparams.json")
    text_processor.save(experiment_dir / "symbols.json")

    if "VERBO" in str(artifacts_dir):
        data_dir_to_load = artifacts_dir
    else:
        data_dir_to_load = Path("data/processed/libriSpeech-en-tacotron-vae")

    train_dataset, test_dataset, val_dataset = load_data(
        text_processor=text_processor,
        data_dir=data_dir_to_load,
        # data_dir=Path("data/processed/libriSpeech-en-tacotron-vae"), HERI e BIA
        val_split=args.val_split,
        generator=torch.Generator().manual_seed(args.seed)
    )

    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    
    train_loader = create_dataloader(train_dataset, args.batch_size, args.num_workers, collate_fn, True)
    test_loader = create_dataloader(test_dataset, args.batch_size, args.num_workers, collate_fn, False)
    val_loader = create_dataloader(val_dataset, args.batch_size, args.num_workers, collate_fn, False)


    model = load_tacotron2_vae_model(hparams, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    criterion = Tacotron2LossVAE(hparams)
    # tb_logger = TensorBoardLogger(tensorboard_dir)
    # tb_logger.log_model_info(model)

    iteration = 0
    learning_rate = hparams.learning_rate

    if args.resume_experiment:
        checkpoint_path = find_latest_checkpoint(Path(hparams.checkpoint_dir))
        if checkpoint_path:
            model, optimizer, learning_rate, iteration = load_checkpoint(checkpoint_path, model)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    model.train()
    for epoch in range(hparams.epochs):
        training_metadata = train_epoch(
            model=model,
            hparams=hparams,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            iteration=iteration,
            learning_rate=learning_rate,
            training_metadata=training_metadata
        )
        #print(f"Epoch: {epoch}")
        #for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        #    start = time.perf_counter()
        #    for param_group in optimizer.param_groups:
        #        param_group["lr"] = learning_rate
#
        #    optimizer.zero_grad()
        #    x, y = model.parse_batch(batch, device)
        #    y_pred = model((x[0], x[1], x[2], x[3]))
        #    loss, recon_loss, kl_loss, kl_weight = criterion(y_pred, y, iteration)
        #    loss.backward()
        #    grad_norm = torch.nn.utils.clip_grad_norm_(
        #        model.parameters(), hparams.grad_clip_thresh
        #    )
        #    optimizer.step()
#
        #    reduced_loss = loss.item()
        #    if not math.isnan(reduced_loss):
        #        duration = time.perf_counter() - start
        #        print(
        #            f"Train loss {iteration} {reduced_loss:.6f} "
        #            f"Grad Norm {float(grad_norm):.6f} {duration:.2f}s/it"
        #        )
        #        # tb_logger.log_training(
        #        #     reduced_loss,
        #        #     float(grad_norm),
        #        #     learning_rate,
        #        #     duration,
        #        #     recon_loss.item(),
        #        #    kl_loss.item(),
        #        #    float(kl_weight),
        #        #    iteration,
        #        #)
#
        #    if iteration % hparams.iters_per_checkpoint == 0:
        #        # val_loss = validate_epoch(model, criterion, val_loader, device, iteration)
        #        # print(f"Validation loss {iteration}: {val_loss:9f}")
        #        # tb_logger.log_validation(val_loss, iteration)
        #        checkpoint_path = Path(hparams.checkpoint_dir) / f"epoch_{iteration}"
        #        save_checkpoint(
        #            model, optimizer, learning_rate, iteration, checkpoint_path, hparams
        #        )
#
        #    iteration += 1

    # tb_logger.close()
    print(f"Training finished. Experiment dir: {experiment_dir}")


if __name__ == "__main__":
    main()
