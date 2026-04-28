"""
Training loop for the prosody and style transfer model.

Supports:
- Mixed-precision training via ``torch.amp``.
- Gradient clipping.
- Checkpoint saving and resuming.
- TensorBoard / stdout logging.
- Separate learning-rate scheduling for the frozen-HuBERT regime.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import TotalLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for :class:`~models.ProsodyStyleTransferModel`.

    Args:
        model: The full prosody style transfer model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (can be ``None``).
        config: Dictionary of hyperparameters.  Recognised keys:

            ``lr`` (float, default ``1e-4``)
                Peak learning rate.
            ``weight_decay`` (float, default ``1e-2``)
                AdamW weight decay.
            ``max_epochs`` (int, default ``100``)
                Total training epochs.
            ``grad_clip`` (float, default ``1.0``)
                Gradient norm clipping value.
            ``log_every`` (int, default ``50``)
                Log every N batches.
            ``save_every`` (int, default ``1``)
                Save checkpoint every N epochs.
            ``output_dir`` (str, default ``"checkpoints"``)
                Directory for saving checkpoints.
            ``mel_weight`` (float, default ``1.0``)
                Weight for mel loss.
            ``duration_weight`` (float, default ``1.0``)
                Weight for duration loss.
            ``pitch_weight`` (float, default ``1.0``)
                Weight for pitch loss.
            ``energy_weight`` (float, default ``1.0``)
                Weight for energy loss.
            ``use_amp`` (bool, default ``True``)
                Enable automatic mixed precision.
        device: Target device (``"cuda"``, ``"cpu"``, …).  Defaults to CUDA
            if available.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        cfg = config or {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.lr = float(cfg.get("lr", 1e-4))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))
        self.max_epochs = int(cfg.get("max_epochs", 100))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.log_every = int(cfg.get("log_every", 50))
        self.save_every = int(cfg.get("save_every", 1))
        self.output_dir = Path(cfg.get("output_dir", "checkpoints"))
        self.use_amp = bool(cfg.get("use_amp", True))

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.criterion = TotalLoss(
            mel_weight=float(cfg.get("mel_weight", 1.0)),
            duration_weight=float(cfg.get("duration_weight", 1.0)),
            pitch_weight=float(cfg.get("pitch_weight", 1.0)),
            energy_weight=float(cfg.get("energy_weight", 1.0)),
        )

        # Only optimise parameters that require gradients
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(train_loader),
            epochs=self.max_epochs,
        )
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            train_loss = self._train_epoch()
            logger.info("Epoch %d | train_loss=%.4f", epoch + 1, train_loss)

            if self.val_loader is not None:
                val_loss = self._val_epoch()
                logger.info("Epoch %d | val_loss=%.4f", epoch + 1, val_loss)

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch: int) -> Path:
        """Save model + optimiser state to disk.

        Args:
            epoch: Current epoch number used in the filename.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved to %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        """Restore model and optimiser state from a checkpoint.

        Args:
            ckpt_path: Path to the ``.pt`` checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        logger.info("Resumed from checkpoint %s (epoch %d)", ckpt_path, self.epoch)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [train]", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)
            loss_dict = self._forward_and_loss(batch)
            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                pbar.set_postfix(
                    {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [val]", leave=False):
            batch = self._to_device(batch)
            loss_dict = self._forward_and_loss(batch)
            total_loss += loss_dict["total"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _forward_and_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Run one forward pass and compute losses.

        Handles both flat key names (``mel``, ``pitch``, …) and the
        ``source_*``-prefixed names produced by :class:`~data.ProsodyTransferDataset`.
        """
        # Support both flat-key and source_*-prefix key naming from dataset
        def _get(key: str):
            return batch.get(key) or batch.get(f"source_{key}")

        ctx = (
            torch.amp.autocast(device_type=self.device.type)
            if self.scaler is not None
            else contextlib.nullcontext()
        )
        source_mel = _get("mel")
        with ctx:
            output = self.model(
                source_waveform=batch["source_waveform"],
                ref_mel=batch.get("ref_mel"),
                attention_mask=batch.get("attention_mask"),
                target_durations=_get("durations"),
                target_pitch=_get("pitch"),
                target_energy=_get("energy"),
                target_len=source_mel.size(-1) if source_mel is not None else None,
            )
            targets = {
                "mel": source_mel if source_mel is not None else batch["source_mel"],
                "durations": _get("durations") or torch.ones(
                    batch["source_waveform"].size(0), 1,
                    device=self.device, dtype=torch.long,
                ),
                "pitch": _get("pitch") or torch.zeros(
                    batch["source_waveform"].size(0), 1, 1, device=self.device
                ),
                "energy": _get("energy") or torch.zeros(
                    batch["source_waveform"].size(0), 1, 1, device=self.device
                ),
            }
            return self.criterion(output, targets)

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
