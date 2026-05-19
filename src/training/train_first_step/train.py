#!/usr/bin/env python3
"""First-step TTS model training script.

Trains a text-to-speech model with:
- Text Encoder (FastPitch)
- Acoustic Decoder (LSTM)
- Style Extractor (GST)
- Vocoder (HiFi-GAN)

Pipeline: Text → Text Encoder → h_text
          Mel → GST → z_style
          [h_text, z_style] → Acoustic Decoder → M_hat
          M_hat → HiFi-GAN → x_hat

Loss functions:
- L1 Reconstruction Loss
- Style Diversity Loss

Usage:
    python train.py --num-epochs 100 --batch-size 32 --learning-rate 1e-3
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.first_step_data_loaders.datasets import LibriSpeechPTDataset, TTSPortugueseDataset
from training.train_first_step.model_loader import load_tts_models
from training.train_first_step.losses import CombinedTTSLoss
from training.train_first_step.train_utils import (
    train_epoch,
    validate_epoch,
    save_checkpoint,
    load_checkpoint,
    TensorBoardLogger,
    log_validation_audio_examples,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train first-step TTS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    
    # Loss function weights
    parser.add_argument(
        "--weight-reconstruction",
        type=float,
        default=1.0,
        help="Weight for L1 reconstruction loss",
    )
    parser.add_argument(
        "--weight-diversity",
        type=float,
        default=0.5,
        help="Weight for style diversity loss",
    )
    parser.add_argument(
        "--diversity-margin",
        type=float,
        default=0.1,
        help="Margin for style diversity loss",
    )
    
    # Model architecture
    parser.add_argument(
        "--acoustic-decoder-hidden-size",
        type=int,
        default=256,
        help="Hidden size for acoustic decoder LSTM",
    )
    parser.add_argument(
        "--acoustic-decoder-num-layers",
        type=int,
        default=3,
        help="Number of layers in acoustic decoder LSTM",
    )
    parser.add_argument(
        "--style-embedding-dim",
        type=int,
        default=128,
        help="Dimension of style embeddings",
    )
    
    # Training configuration
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    
    # Experiment configuration
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name (default: attempt_<timestamp>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def create_experiment_dir(experiment_name: Optional[str] = None) -> Path:
    """Create experiment directory structure.
    
    Args:
        experiment_name: Custom experiment name, or None for timestamp-based
    
    Returns:
        Path to experiment directory
    """
    experiments_root = PROJECT_ROOT / "experiments" / "step_1"
    experiments_root.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"attempt_{timestamp}"
    
    experiment_dir = experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "tensorboard").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    return experiment_dir


def create_datasets(
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
) -> tuple:
    """Create training and validation datasets.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("\nLoading datasets...")
    
    try:
        librispeech_train = LibriSpeechPTDataset(split="train")
        print(f"  ✓ LibriSpeech-PT train: {len(librispeech_train)} samples")
    except Exception as e:
        print(f"  ✗ LibriSpeech-PT train: {e}")
        librispeech_train = None
    
    try:
        librispeech_test = LibriSpeechPTDataset(split="test")
        print(f"  ✓ LibriSpeech-PT test: {len(librispeech_test)} samples")
    except Exception as e:
        print(f"  ✗ LibriSpeech-PT test: {e}")
        librispeech_test = None
    
    try:
        tts_portuguese = TTSPortugueseDataset()
        print(f"  ✓ TTS Portuguese: {len(tts_portuguese)} samples")
    except Exception as e:
        print(f"  ✗ TTS Portuguese: {e}")
        tts_portuguese = None
    
    # Combine datasets
    datasets_to_combine = [d for d in [librispeech_train, librispeech_test, tts_portuguese] if d is not None]
    
    if not datasets_to_combine:
        raise RuntimeError("No datasets could be loaded!")
    
    combined_dataset = ConcatDataset(datasets_to_combine)
    print(f"\nCombined dataset: {len(combined_dataset)} samples")
    
    # Split into train/val
    val_size = int(len(combined_dataset) * val_split)
    train_size = len(combined_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    def collate_first_step_batch(batch):
        """Collate variable-length mel tensors with right padding.

        Avoids failures from default_collate when sample tensors differ in
        time length or use non-resizable storages.
        """
        mels = []
        waveforms = []
        mel_lengths = []
        texts = []
        durations = []
        utt_ids = []
        mel_paths = []
        sources = []
        sample_rates = []

        for sample in batch:
            mel = sample["mel"]
            if not isinstance(mel, torch.Tensor):
                mel = torch.as_tensor(mel)

            mel = mel.detach().clone().to(dtype=torch.float32).contiguous()

            waveform = sample.get("waveform")
            if waveform is not None:
                if not isinstance(waveform, torch.Tensor):
                    waveform = torch.as_tensor(waveform)
                waveform = waveform.detach().clone().to(dtype=torch.float32).contiguous()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                waveform = torch.zeros(1, 1, dtype=torch.float32)

            # Normalize common mel shapes to (n_mels, time_steps)
            if mel.dim() == 3 and mel.size(0) == 1:
                mel = mel.squeeze(0)
            if mel.dim() != 2:
                raise ValueError(f"Expected mel with 2 dims, got shape {tuple(mel.shape)}")

            mels.append(mel)
            waveforms.append(waveform)
            mel_lengths.append(mel.size(1))
            texts.append(sample.get("text", ""))
            durations.append(float(sample.get("duration", 0.0)))
            utt_ids.append(str(sample.get("utt_id", "")))
            mel_paths.append(str(sample.get("mel_path", "")))
            sources.append(str(sample.get("source", "")))
            sample_rates.append(int(sample.get("sr") or 22050))

        max_time = max(mel_lengths)
        padded_mels = []
        for mel in mels:
            pad_time = max_time - mel.size(1)
            if pad_time > 0:
                mel = F.pad(mel, (0, pad_time), mode="constant", value=0.0)
            padded_mels.append(mel)

        max_wave_time = max(waveform.size(-1) for waveform in waveforms)
        padded_waveforms = []
        for waveform in waveforms:
            pad_time = max_wave_time - waveform.size(-1)
            if pad_time > 0:
                waveform = F.pad(waveform, (0, pad_time), mode="constant", value=0.0)
            padded_waveforms.append(waveform)

        return {
            "mel": torch.stack(padded_mels, dim=0),
            "waveform": torch.stack(padded_waveforms, dim=0),
            "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
            "sr": torch.tensor(sample_rates, dtype=torch.long),
            "text": texts,
            "duration": torch.tensor(durations, dtype=torch.float32),
            "utt_id": utt_ids,
            "mel_path": mel_paths,
            "source": sources,
        }

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_first_step_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_first_step_batch,
    )
    
    return train_loader, val_loader


def main():
    """Main training loop."""
    args = parse_arguments()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(args.experiment_name)
    print(f"\nExperiment directory: {experiment_dir}")
    
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"
    
    # Save hyperparameters
    hparams = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "weight_reconstruction": args.weight_reconstruction,
        "weight_diversity": args.weight_diversity,
        "diversity_margin": args.diversity_margin,
        "acoustic_decoder_hidden_size": args.acoustic_decoder_hidden_size,
        "acoustic_decoder_num_layers": args.acoustic_decoder_num_layers,
        "style_embedding_dim": args.style_embedding_dim,
        "use_amp": args.use_amp,
        "seed": args.seed,
        "val_split": args.val_split,
    }
    
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"✓ Config saved to {experiment_dir / 'config.json'}")
    
    # Load datasets
    train_loader, val_loader = create_datasets(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )
    
    # Load model
    print("\nLoading TTS model...")
    model = load_tts_models(
        device=device,
        acoustic_decoder_hidden_size=args.acoustic_decoder_hidden_size,
        acoustic_decoder_num_layers=args.acoustic_decoder_num_layers,
        style_embedding_dim=args.style_embedding_dim,
    )
    
    # Setup optimizer
    optimizer = Adam(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(f"\nOptimizer: Adam(lr={args.learning_rate}, weight_decay={args.weight_decay})")
    
    # Setup loss function
    criterion = CombinedTTSLoss(
        weight_reconstruction=args.weight_reconstruction,
        weight_diversity=args.weight_diversity,
        diversity_margin=args.diversity_margin,
    ).to(device)
    print(f"Loss: Combined TTS Loss")
    print(f"  L1 Reconstruction weight: {args.weight_reconstruction}")
    print(f"  Style Diversity weight: {args.weight_diversity}")
    
    # Setup GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Setup TensorBoard
    tb_logger = TensorBoardLogger(tensorboard_dir)
    tb_logger.log_model_info(model)
    tb_logger.log_hyperparameters(hparams, {})
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(model, optimizer, Path(args.resume), device)
        print(f"  Loaded from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_epochs=args.num_epochs,
            scaler=scaler,
            use_amp=args.use_amp,
        )
        
        # Validate
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_epochs=args.num_epochs,
        )

        if epoch == 0 or (epoch + 1) % 1 == 0:
            example_batch = next(iter(val_loader))
            log_validation_audio_examples(
                model=model,
                batch=example_batch,
                device=device,
                logger=tb_logger,
                step=epoch,
            )
        
        # Log metrics
        tb_logger.log_metrics(train_metrics, epoch, prefix="train/")
        tb_logger.log_metrics(val_metrics, epoch, prefix="val/")
        tb_logger.flush()
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"    ├─ Reconstruction: {train_metrics['recon_loss']:.6f}")
        print(f"    └─ Diversity: {train_metrics['div_loss']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"    ├─ Reconstruction: {val_metrics['recon_loss']:.6f}")
        print(f"    └─ Diversity: {val_metrics['div_loss']:.6f}")
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
            checkpoint_dir=checkpoint_dir,
            filename=f"epoch_{epoch+1:04d}.pt",
        )
        
        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                checkpoint_dir=checkpoint_dir,
                filename="best.pt",
            )
            print(f"  ✓ New best validation loss: {best_val_loss:.6f}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"To view TensorBoard: tensorboard --logdir={tensorboard_dir}")
    
    tb_logger.close()


if __name__ == "__main__":
    main()
