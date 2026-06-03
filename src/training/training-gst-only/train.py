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
            fastpitch, gst, mel_decoder, hifigan,
            train_loader, optimizer, config.loss_weights,
            device, epoch, config.max_epochs
        )
        
        # Validation phase
        val_loss, val_metrics = validate_epoch(
            fastpitch, gst, mel_decoder, hifigan,
            val_loader, config.loss_weights,
            device, epoch, config.max_epochs
        )
        
        # Log metrics
        logger.log_metrics(train_metrics, epoch, "train")
        logger.log_metrics(val_metrics, epoch, "val")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                fastpitch, gst, mel_decoder, optimizer,
                config.checkpoint_dir, epoch
            )
    
    # Final save
    save_checkpoint(
        fastpitch, gst, mel_decoder, optimizer,
        config.checkpoint_dir, config.max_epochs - 1
    )
    
    # Close logger
    logger.close()

def train_epoch(
    fastpitch: FastPitchModel,
    gst: GST,
    mel_decoder: MelDecoder,
    hifigan: HifiGanModel,
    train_loader: DataLoader,
    optimizer: optim.AdamW,
    loss_weights: Dict[str, float],
    device: torch.device,
    epoch: int,
    max_epochs: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train for one epoch."""
    fastpitch.train()
    gst.train()
    mel_decoder.train()
    
    total_loss = 0.0
    metrics = {}
    
    for batch in train_loader:
        # Move data to device
        text_ids = batch["text_ids"].to(device)
        target_mel = batch["mel"].to(device)
        reference_audio = batch["reference_audio"].to(device)
        
        # Forward pass
        text_states = fastpitch(text_ids)
        style_ref = gst(reference_audio)
        style_gen = gst(hifigan(target_mel))
        
        # Generate mel spectrogram
        mel = mel_decoder(text_states, style_gen)
        
        # Compute loss
        loss, loss_components = total_loss(
            mel, target_mel, style_gen, style_ref, style_gen,
            loss_weights
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in loss_components.items():
            metrics[k] = metrics.get(k, 0.0) + v.item()
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(train_loader)
    
    return total_loss / len(train_loader), metrics

def validate_epoch(
    fastpitch: FastPitchModel,
    gst: GST,
    mel_decoder: MelDecoder,
    hifigan: HifiGanModel,
    val_loader: DataLoader,
    loss_weights: Dict[str, float],
    device: torch.device,
    epoch: int,
    max_epochs: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Validate for one epoch."""
    fastpitch.eval()
    gst.eval()
    mel_decoder.eval()
    
    total_loss = 0.0
    metrics = {}
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            text_ids = batch["text_ids"].to(device)
            target_mel = batch["mel"].to(device)
            reference_audio = batch["reference_audio"].to(device)
            
            # Forward pass
            text_states = fastpitch(text_ids)
            style_ref = gst(reference_audio)
            style_gen = gst(hifigan(target_mel))
            
            # Generate mel spectrogram
            mel = mel_decoder(text_states, style_gen)
            
            # Compute loss
            loss, loss_components = total_loss(
                mel, target_mel, style_gen, style_ref, style_gen,
                loss_weights
            )
            
            # Update metrics
            total_loss += loss.item()
            for k, v in loss_components.items():
                metrics[k] = metrics.get(k, 0.0) + v.item()
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(val_loader)
    
    return total_loss / len(val_loader), metrics

def save_checkpoint(
    fastpitch: FastPitchModel,
    gst: GST,
    mel_decoder: MelDecoder,
    optimizer: optim.AdamW,
    checkpoint_dir: str,
    epoch: int
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    
    torch.save({
        "fastpitch": fastpitch.state_dict(),
        "gst": gst.state_dict(),
        "mel_decoder": mel_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, checkpoint_path)
