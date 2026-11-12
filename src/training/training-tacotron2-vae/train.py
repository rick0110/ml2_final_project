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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--iters-per-checkpoint", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many iterations (0 = run all epochs)")
    parser.add_argument("--keep-last-checkpoints", type=int, default=0,
                        help="Delete all but the newest N checkpoints (0 = keep all)")
    parser.add_argument("--anneal-upper", type=float, default=None, help="Maximum KL weight")
    parser.add_argument("--anneal-lag", type=int, default=None, help="Steps before the KL weight starts ramping")
    parser.add_argument("--free-bits", type=float, default=None, help="KL floor per latent dimension (nats)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Default matches hparams.py; 1e-5 was found to cause a training plateau")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip-thresh", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--anneal-function", type=str, default="cyclical")
    parser.add_argument("--guided-attention-weight", type=float, default=2.0)
    parser.add_argument("--guided-attention-sigma", type=float, default=0.4)
    parser.add_argument("--gate-positive-weight", type=float, default=1.0,
                        help="BCE pos_weight for the stop frame (>1 counters the ~1:250 stop/continue imbalance)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear LR warmup over this many steps (0=disabled)")
    parser.add_argument("--lr-scheduler-patience", type=int, default=20,
                        help="ReduceLROnPlateau patience in epochs (default 20; set high to effectively disable)")
    parser.add_argument("--text-cleaner", type=str, default="portuguese_phonetic_cleaners",
                        help="Text cleaner: 'english_cleaners' for LJSpeech, 'portuguese_phonetic_cleaners' for PT")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)
    parser.add_argument("--pretrained-tacotron2", type=str, default=None,
                        help="Path to pretrained NVIDIA Tacotron2 checkpoint for finetuning backbone")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Path with the data, metadata.csv",
    )

    # Model capacity overrides (defaults match the full-size hparams.py, tuned for
    # multi-hour corpora; pass smaller values for small datasets like VERBO to
    # reduce overfitting risk and speed up convergence).
    parser.add_argument("--symbols-embedding-dim", type=int, default=None)
    parser.add_argument("--encoder-embedding-dim", type=int, default=None)
    parser.add_argument("--style-dim", type=int, default=None, help="VAE-GST style embedding dim (hparams.E)")
    parser.add_argument("--z-latent-dim", type=int, default=None)
    parser.add_argument("--decoder-rnn-dim", type=int, default=None)
    parser.add_argument("--attention-rnn-dim", type=int, default=None)
    parser.add_argument("--prenet-dim", type=int, default=None)
    parser.add_argument("--postnet-embedding-dim", type=int, default=None)

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
    text_processor: TextProcessor = TextProcessor(symbols=symbols, cleaner_names=[args.text_cleaner])

    hparams_overrides: Dict[str, Any] = {
        "epochs": args.epochs,
        "iters_per_checkpoint": args.iters_per_checkpoint,
        "max_steps": args.max_steps,
        "keep_last_checkpoints": args.keep_last_checkpoints,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip_thresh": args.grad_clip_thresh,
        "seed": args.seed,
        "anneal_function": args.anneal_function,
        "guided_attention_weight": args.guided_attention_weight,
        "guided_attention_sigma": args.guided_attention_sigma,
        "gate_positive_weight": args.gate_positive_weight,
        "warmup_steps": args.warmup_steps,
        "n_symbols": text_processor.n_symbols,
        "training_data": str(train_file),
    }
    for name in ("anneal_upper", "anneal_lag", "free_bits"):
        if getattr(args, name) is not None:
            hparams_overrides[name] = getattr(args, name)
    if args.symbols_embedding_dim is not None:
        hparams_overrides["symbols_embedding_dim"] = args.symbols_embedding_dim
    if args.encoder_embedding_dim is not None:
        hparams_overrides["encoder_embedding_dim"] = args.encoder_embedding_dim
    if args.z_latent_dim is not None:
        hparams_overrides["z_latent_dim"] = args.z_latent_dim
    if args.decoder_rnn_dim is not None:
        hparams_overrides["decoder_rnn_dim"] = args.decoder_rnn_dim
    if args.attention_rnn_dim is not None:
        hparams_overrides["attention_rnn_dim"] = args.attention_rnn_dim
    if args.prenet_dim is not None:
        hparams_overrides["prenet_dim"] = args.prenet_dim
    if args.postnet_embedding_dim is not None:
        hparams_overrides["postnet_embedding_dim"] = args.postnet_embedding_dim
    if args.style_dim is not None:
        hparams_overrides["E"] = args.style_dim
        # ReferenceEncoder's GRU hidden size is hardcoded to E // 2 (modules.py);
        # ref_enc_gru_size must match it exactly or VAE_GST.fc1/fc2 raise a shape
        # mismatch, so derive it here instead of exposing it as an independent flag.
        hparams_overrides["ref_enc_gru_size"] = args.style_dim // 2

    hparams: Tacotron2VAEHparams = create_hparams(hparams_overrides)

    # Tacotron2.forward() adds the VAE-GST style embedding directly to the encoder
    # output (encoder_outputs = transcript_outputs + prosody_outputs), so the two
    # must share a dimension or the addition raises a shape-mismatch RuntimeError.
    if hparams.E != hparams.encoder_embedding_dim:
        raise ValueError(
            f"hparams.E (style dim, {hparams.E}) must equal hparams.encoder_embedding_dim "
            f"({hparams.encoder_embedding_dim}): the VAE-GST style embedding is added "
            f"directly to the encoder output. Pass matching --style-dim and "
            f"--encoder-embedding-dim."
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

    # Load pretrained Tacotron2 backbone weights if provided
    if args.pretrained_tacotron2:
        from models.tacotron2_vae.model import load_pretrained_tacotron2_backbone
        model = load_pretrained_tacotron2_backbone(model, args.pretrained_tacotron2, device)
        print(f"Loaded pretrained Tacotron2 backbone from {args.pretrained_tacotron2}")

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_scheduler_patience, min_lr=1e-6
    )
    criterion: Tacotron2LossVAE = Tacotron2LossVAE(hparams)

    iteration: int = 0
    learning_rate: float = hparams.learning_rate

    # Resume from latest checkpoint in an existing experiment directory
    if args.resume_experiment:
        checkpoint_path: Optional[Path] = find_latest_checkpoint(Path(hparams.checkpoint_dir))
        if checkpoint_path:
            model, optimizer_state, learning_rate, iteration = load_checkpoint(checkpoint_path, model)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)

    # Fine-tune from a specific checkpoint (resets optimizer and iteration for fresh start)
    elif args.checkpoint_path:
        model, _, _, _ = load_checkpoint(Path(args.checkpoint_path), model)
        print(f"Fine-tuning from checkpoint: {args.checkpoint_path}")

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Training Loop — bf16 autocast (fp16 AMP overflows with random VAE weights
    # producing large mel values; bf16 keeps the fp32 exponent range, so no
    # GradScaler is needed — the disabled scaler below is a pass-through that
    # just satisfies train_epoch's use_amp gate)
    model.train()
    tensorboard_logger = TensorBoardLogger(experiment_dir / "logs")
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    for epoch in range(hparams.epochs):
        training_metadata, epoch_val_loss, iteration = train_epoch(
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
            tensorboard_logger=tensorboard_logger,
            scaler=scaler,
        )
        
        # Update learning rate based on validation loss
        if epoch_val_loss is not None:
            scheduler.step(epoch_val_loss)
            learning_rate = optimizer.param_groups[0]['lr']

        if hparams.max_steps > 0 and iteration >= hparams.max_steps:
            save_checkpoint(model, optimizer, learning_rate, iteration,
                            Path(hparams.checkpoint_dir) / f"epoch_{iteration}", hparams)
            break
        


    print(f"Training finished. Experiment dir: {experiment_dir}")
    tensorboard_logger.close()


if __name__ == "__main__":
    main()
