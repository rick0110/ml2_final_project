#!/usr/bin/env python3
"""Convenience script to run training with pre-configured settings.

Usage:
    python run_training.py quick_test      # Quick test
    python run_training.py balanced        # Balanced training
    python run_training.py production      # Production training
    python run_training.py --list          # List all configs
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.train_first_step.configs import get_config, list_configs, CONFIGS


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run TTS training with pre-configured settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "config",
        nargs="?",
        help="Configuration name (use --list to see options)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_configs()
        return
    
    # Check config provided
    if not args.config:
        parser.print_help()
        print("\nUse --list to see available configurations")
        return
    
    # Get configuration
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        list_configs()
        return
    
    # Print configuration
    config.print_summary()
    
    # Build command
    train_script = PROJECT_ROOT / "src" / "training" / "train_first_step" / "train.py"
    
    cmd = [
        "python",
        str(train_script),
        "--experiment-name", config.experiment_name,
        "--num-epochs", str(config.num_epochs),
        "--batch-size", str(config.batch_size),
        "--learning-rate", str(config.learning_rate),
        "--weight-decay", str(config.weight_decay),
        "--num-workers", str(config.num_workers),
        "--weight-reconstruction", str(config.weight_reconstruction),
        "--weight-diversity", str(config.weight_diversity),
        "--diversity-margin", str(config.diversity_margin),
        "--acoustic-decoder-hidden-size", str(config.acoustic_decoder_hidden_size),
        "--acoustic-decoder-num-layers", str(config.acoustic_decoder_num_layers),
        "--style-embedding-dim", str(config.style_embedding_dim),
        "--val-split", str(config.val_split),
        "--seed", str(config.seed),
    ]
    
    if config.use_amp:
        cmd.append("--use-amp")
    
    # Print command
    print("\nCommand:")
    print(" ".join(cmd))
    print()
    
    # Execute or dry-run
    if args.dry_run:
        print("[DRY RUN] Command not executed")
        return
    
    print("Starting training...\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
