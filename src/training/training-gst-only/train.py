import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.models import FastPitchModel, GST, MelDecoder, HifiGanModel
from src.data import build_dataset
from src.training.train_utils import TensorBoardLogger, train_epoch, validate_epoch
from src.training.training-gst-only.losses import total_loss
from src.training.training-gst-only.configs import TrainingConfig

def parse_arguments():
    """Parse command line arguments for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GST-only TTS model")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--log_dir", type=str, default="logs/training-gst-only", help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/training-gst-only", help="Model checkpoint directory")
    return parser.parse_args()

def load_experiment_config(experiment_dir: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    config_path = os.path.join(experiment_dir, "config.yaml")
    return OmegaConf.load(config_path)

def train_model(config: TrainingConfig):
    """Main training function."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    fastpitch = FastPitchModel(**config.model["fastpitch"]).to(device)
    gst = GST(**config.model["gst"]).to(device)
    mel_decoder = MelDecoder(**config.model["mel_decoder"]).to(device)
    hifigan = HifiGanModel(**config.model["hifigan"]).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        list(fastpitch.parameters()) + list(gst.parameters()) + list(mel_decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize data loaders
    dataset = build_dataset(config)
    train_loader, val_loader = dataset.get_data_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        seed=config.seed
    )
    
    # Initialize logger
    logger = TensorBoardLogger(config.log_dir)
    
    # Training loop
    for epoch in range(config.max_epochs):
        # Training phase
        train_loss, train_metrics = train_epoch(
            fastpitch, gst, mel_decoder, hif仁
