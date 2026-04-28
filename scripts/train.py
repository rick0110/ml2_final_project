#!/usr/bin/env python
"""
Training entry-point for the Prosody and Style Transfer model.

Usage::

    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch0010.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, random_split

# Allow importing from the src package directly when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.full_model import ProsodyStyleTransferModel
from data.dataset import TTSPortugueseDataset, LibriVoxPTBRDataset, ProsodyTransferDataset, collate_fn
from data.preprocessing import AudioPreprocessor
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prosody style transfer model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu)")
    return parser.parse_args()


def build_datasets(cfg: dict, preprocessor: AudioPreprocessor):
    data_cfg = cfg["data"]

    tts_root = data_cfg.get("tts_portuguese_root", "data/tts_portuguese")
    librivox_root = data_cfg.get("librivox_ptbr_root", "data/librivox_ptbr")
    max_dur = float(data_cfg.get("max_duration_s", 10.0))

    content_dataset = TTSPortugueseDataset(
        root=tts_root, preprocessor=preprocessor, max_duration_s=max_dur
    )
    reference_dataset = LibriVoxPTBRDataset(
        root=librivox_root, preprocessor=preprocessor, max_duration_s=30.0
    )

    combined = ProsodyTransferDataset(content_dataset, reference_dataset)

    val_size = max(1, int(0.05 * len(combined)))
    train_size = len(combined) - val_size
    train_ds, val_ds = random_split(combined, [train_size, val_size])
    logger.info("Train samples: %d | Val samples: %d", train_size, val_size)
    return train_ds, val_ds


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    audio_cfg = cfg.get("audio", {})
    preprocessor = AudioPreprocessor(
        sample_rate=audio_cfg.get("sample_rate", 22050),
        n_mels=audio_cfg.get("n_mels", 80),
        n_fft=audio_cfg.get("n_fft", 1024),
        hop_length=audio_cfg.get("hop_length", 256),
        win_length=audio_cfg.get("win_length", 1024),
        f_min=audio_cfg.get("f_min", 0.0),
        f_max_mel=audio_cfg.get("f_max_mel", 8000.0),
        f_min_pitch=audio_cfg.get("f_min_pitch", 50.0),
        f_max_pitch=audio_cfg.get("f_max_pitch", 600.0),
    )

    train_ds, val_ds = build_datasets(cfg, preprocessor)

    data_cfg = cfg.get("data", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model_cfg = cfg.get("model", {})
    model = ProsodyStyleTransferModel(
        hubert_model_name=model_cfg.get("hubert_model_name", "facebook/hubert-base-ls960"),
        freeze_hubert=bool(model_cfg.get("freeze_hubert", True)),
        n_mels=audio_cfg.get("n_mels", 80),
        d_model=int(model_cfg.get("d_model", 256)),
        style_dim=int(model_cfg.get("style_dim", 128)),
        mapping_hidden_dim=int(model_cfg.get("mapping_hidden_dim", 512)),
        mapping_num_layers=int(model_cfg.get("mapping_num_layers", 4)),
        gst_num_tokens=int(model_cfg.get("gst_num_tokens", 10)),
        gst_token_dim=int(model_cfg.get("gst_token_dim", 256)),
        gst_num_heads=int(model_cfg.get("gst_num_heads", 8)),
        variance_num_conv=int(model_cfg.get("variance_num_conv", 2)),
        decoder_upsample_rates=tuple(model_cfg.get("decoder_upsample_rates", [8, 8, 2, 2])),
        decoder_initial_channels=int(model_cfg.get("decoder_initial_channels", 512)),
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{num_params:,}")

    train_cfg = cfg.get("training", {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
