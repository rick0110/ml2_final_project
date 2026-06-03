from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    """Training configuration dataclass for GST-only training."""
    
    # Model configuration
    model: Dict[str, Any] = (
        {"fastpitch": {"hidden_dim": 256, "n_layers": 4, "n_heads": 4, "ff_dim": 1024},
         "gst": {"n_conv_layers": 6, "hidden_size": 128, "n_style_tokens": 30, "n_mels": 80, "n_heads": 4},
         "mel_decoder": {"model_dim": 256, "n_heads": 4, "n_layers": 4, "ff_dim": 1024},
         "hifigan": {"n_mels": 80}}
    )
    
    # Training hyperparameters
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 42
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights
    loss_weights: Dict[str, float] = (
        {"mel_recon": 1.0, "style_consistency": 0.5, "style_separation": 0.3, "contrastive_style": 0.2}
    )
    
    # Logging and checkpointing
    log_dir: str = "logs/training-gst-only"
    checkpoint_dir: str = "checkpoints/training-gst-only"
    
    def to_command_line_args(self) -> str:
        """Convert configuration to command line arguments."""
        return " ".join([f"--{k}={v}" for k, v in self.__dict__.items()])
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("Training Configuration Summary:")
        for k, v in self.__dict__.items():
            print(f"  {k}: {v}")
