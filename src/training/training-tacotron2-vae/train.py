#!/usr/bin/env python3
"""
Tacotron 2 VAE training script.

Responsibilities:
    - Load hyperparameters and initialize text processing.
    - Set up training, validation, and test datasets/loaders.
    - Initialize the Tacotron 2 VAE model, optimizer, and loss criterion.
    - Orchestrate the training process across multiple epochs.
    - Handle checkpoint saving and experiment management.

Main Functions:
    - main: Primary entry point for the training workflow.
    - parse_arguments: Handle command-line configuration.

Tensor Conventions:
    B = batch size
    T = sequence length (frames/tokens)
    n_mels = mel frequency bins
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple

# Import training utilities
try:
    from utils import ARTIFACTS_DIR, TextMelCollate, create_dataloader, create_experiment_dir
except ImportError:
    # Fallback to absolute paths if running from root
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from utils import ARTIFACTS_DIR, TextMelCollate, create_dataloader, create_experiment_dir


PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "loader_vae_tacotron"))


try:
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
except ImportError:
    # Handle cases where path insertion didn't cover all imports
    from src.training.training_tacotron2_vae.losses import Tacotron2LossVAE
    from src.models.tacotron2_vae.hparams import Tacotron2VAEHparams, create_hparams
    from src.models.tacotron2_vae.model import load_tacotron2_vae_model


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
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


def main() -> None:
    """
    Primary training workflow.
    """
    # Metadata for tracking training progress
    training_metadata: Dict[str, List[Any]] = {
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
    args: argparse.Namespace = parse_arguments()
    torch.manual_seed(args.seed)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts_dir: Path = Path(args.artifacts_dir)
    train_file: Path = artifacts_dir / "mels_metadata.csv"
    
    # Load texts to build symbols vocabulary
    with open(train_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        texts: List[str] = [row["text"] for row in reader]

    symbols: List[str] = build_symbols_from_texts(texts)
    text_processor: TextProcessor = TextProcessor(symbols=symbols)

    hparams: Tacotron2VAEHparams = create_hparams(
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

    experiment_dir: Path = (
        Path(args.resume_experiment)
        if args.resume_experiment
        else create_experiment_dir(args.experiment_name)
    )

    hparams.experiment_dir = str(experiment_dir)
    hparams.checkpoint_dir = str(experiment_dir / "checkpoints")
    save_hparams(hparams, experiment_dir / "hparams.json")
    text_processor.save(experiment_dir / "symbols.json")

    # Load and split datasets
    train_dataset: Any
    test_dataset: Any
    val_dataset: Any
    train_dataset, test_dataset, val_dataset = load_data(
        text_processor=text_processor,
        data_dir=args.artifacts_dir,
        val_split=args.val_split,
        generator=torch.Generator().manual_seed(args.seed)
    )

    collate_fn: TextMelCollate = TextMelCollate(hparams.n_frames_per_step)
    
    # Initialize DataLoaders
    train_loader: DataLoader = create_dataloader(train_dataset, args.batch_size, args.num_workers, collate_fn, True)
    test_loader: DataLoader = create_dataloader(test_dataset, args.batch_size, args.num_workers, collate_fn, False)
    val_loader: DataLoader = create_dataloader(val_dataset, args.batch_size, args.num_workers, collate_fn, False)

    # Initialize Model, Optimizer, and Loss
    model = load_tacotron2_vae_model(hparams, device=device)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    criterion: Tacotron2LossVAE = Tacotron2LossVAE(hparams)

    iteration: int = 0
    learning_rate: float = hparams.learning_rate

    # Resume from checkpoint if specified
    if args.resume_experiment:
        checkpoint_path: Optional[Path] = find_latest_checkpoint(Path(hparams.checkpoint_dir))
        if checkpoint_path:
            model, optimizer, learning_rate, iteration = load_checkpoint(checkpoint_path, model)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    # Training Loop
    model.train()
    tensorboard_logger = TensorBoardLogger(experiment_dir / "logs")
    
    for epoch in range(hparams.epochs):
        training_metadata = train_epoch(
            model=model,
            hparams=hparams,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            iteration=iteration,
            learning_rate=learning_rate,
            training_metadata=training_metadata,
            tensorboard_logger=tensorboard_logger
        )
        
        # Update global iteration count
        iteration += len(train_loader)

    print(f"Training finished. Experiment dir: {experiment_dir}")
    tensorboard_logger.close()


if __name__ == "__main__":
    main()
