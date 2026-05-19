#!/usr/bin/env python3
"""Example training configurations for different scenarios.

This file provides pre-configured training setups for various use cases:
- Quick test (small model, few epochs)
- Balanced (medium model, normal training)
- Production (large model, long training)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    
    # Experiment
    experiment_name: str
    description: str
    
    # Training
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    
    # Loss weights
    weight_reconstruction: float
    weight_diversity: float
    diversity_margin: float
    
    # Model architecture
    acoustic_decoder_hidden_size: int
    acoustic_decoder_num_layers: int
    style_embedding_dim: int
    
    # Training options
    use_amp: bool
    val_split: float
    seed: int
    
    def to_command_line_args(self) -> str:
        """Convert config to command-line arguments string."""
        args = [
            f"--experiment-name {self.experiment_name}",
            f"--num-epochs {self.num_epochs}",
            f"--batch-size {self.batch_size}",
            f"--learning-rate {self.learning_rate}",
            f"--weight-decay {self.weight_decay}",
            f"--num-workers {self.num_workers}",
            f"--weight-reconstruction {self.weight_reconstruction}",
            f"--weight-diversity {self.weight_diversity}",
            f"--diversity-margin {self.diversity_margin}",
            f"--acoustic-decoder-hidden-size {self.acoustic_decoder_hidden_size}",
            f"--acoustic-decoder-num-layers {self.acoustic_decoder_num_layers}",
            f"--style-embedding-dim {self.style_embedding_dim}",
            f"--val-split {self.val_split}",
            f"--seed {self.seed}",
        ]
        
        if self.use_amp:
            args.append("--use-amp")
        
        return " \\\n    ".join(args)
    
    def print_summary(self):
        """Print configuration summary."""
        print(f"\n{'='*80}")
        print(f"Configuration: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"\nDescription:")
        print(f"  {self.description}")
        print(f"\nHyperparameters:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Weight Decay: {self.weight_decay}")
        print(f"\nLoss Weights:")
        print(f"  Reconstruction: {self.weight_reconstruction}")
        print(f"  Diversity: {self.weight_diversity}")
        print(f"  Diversity Margin: {self.diversity_margin}")
        print(f"\nModel Architecture:")
        print(f"  Decoder Hidden Size: {self.acoustic_decoder_hidden_size}")
        print(f"  Decoder Layers: {self.acoustic_decoder_num_layers}")
        print(f"  Style Embedding Dim: {self.style_embedding_dim}")
        print(f"\nOptions:")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Validation Split: {self.val_split}")
        print(f"  Seed: {self.seed}")
        print()


# Pre-configured training configurations

QUICK_TEST = TrainingConfig(
    experiment_name="quick_test",
    description="Quick test configuration for development. Small model, 5 epochs, 8 batch size.",
    num_epochs=5,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_workers=2,
    weight_reconstruction=1.0,
    weight_diversity=0.5,
    diversity_margin=0.1,
    acoustic_decoder_hidden_size=128,
    acoustic_decoder_num_layers=2,
    style_embedding_dim=64,
    use_amp=True,
    val_split=0.1,
    seed=42,
)

BALANCED = TrainingConfig(
    experiment_name="balanced_training",
    description="Balanced configuration for normal training. Medium model, 100 epochs, 32 batch size.",
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_workers=4,
    weight_reconstruction=1.0,
    weight_diversity=0.5,
    diversity_margin=0.1,
    acoustic_decoder_hidden_size=256,
    acoustic_decoder_num_layers=3,
    style_embedding_dim=128,
    use_amp=True,
    val_split=0.1,
    seed=42,
)

PRODUCTION = TrainingConfig(
    experiment_name="production_training",
    description="Production configuration. Large model, 200 epochs, 64 batch size. Requires high memory GPU.",
    num_epochs=200,
    batch_size=64,
    learning_rate=5e-4,
    weight_decay=1e-5,
    num_workers=8,
    weight_reconstruction=1.0,
    weight_diversity=0.5,
    diversity_margin=0.1,
    acoustic_decoder_hidden_size=512,
    acoustic_decoder_num_layers=4,
    style_embedding_dim=256,
    use_amp=True,
    val_split=0.1,
    seed=42,
)

HIGH_DIVERSITY = TrainingConfig(
    experiment_name="high_diversity",
    description="Configuration emphasizing style diversity. Higher diversity loss weight.",
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_workers=4,
    weight_reconstruction=1.0,
    weight_diversity=2.0,  # Higher diversity weight
    diversity_margin=0.15,  # Larger margin
    acoustic_decoder_hidden_size=256,
    acoustic_decoder_num_layers=3,
    style_embedding_dim=128,
    use_amp=True,
    val_split=0.1,
    seed=42,
)

LIGHTWEIGHT = TrainingConfig(
    experiment_name="lightweight",
    description="Lightweight configuration for limited resources. Small model, low batch size.",
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-5,
    num_workers=2,
    weight_reconstruction=1.0,
    weight_diversity=0.3,
    diversity_margin=0.1,
    acoustic_decoder_hidden_size=64,
    acoustic_decoder_num_layers=2,
    style_embedding_dim=64,
    use_amp=False,  # Disable AMP for compatibility
    val_split=0.15,  # Larger validation split for small model
    seed=42,
)

# Configuration registry
CONFIGS = {
    "quick_test": QUICK_TEST,
    "balanced": BALANCED,
    "production": PRODUCTION,
    "high_diversity": HIGH_DIVERSITY,
    "lightweight": LIGHTWEIGHT,
}


def get_config(name: str) -> TrainingConfig:
    """Get a configuration by name.
    
    Args:
        name: Configuration name
    
    Returns:
        TrainingConfig instance
    
    Raises:
        ValueError: If configuration name not found
    """
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(
            f"Configuration '{name}' not found. "
            f"Available: {available}"
        )
    
    return CONFIGS[name]


def list_configs():
    """List all available configurations."""
    print("\nAvailable Training Configurations:")
    print("=" * 80)
    
    for name, config in CONFIGS.items():
        print(f"\n{name.upper()}")
        print(f"  {config.description}")
        print(f"  Epochs: {config.num_epochs} | Batch: {config.batch_size} | "
              f"Decoder Hidden: {config.acoustic_decoder_hidden_size}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        try:
            config = get_config(config_name)
            config.print_summary()
            
            print("Command to run:")
            print(f"python src/training/train_first_step/train.py \\")
            print(f"    {config.to_command_line_args()}")
        except ValueError as e:
            print(f"Error: {e}")
            list_configs()
    else:
        list_configs()
