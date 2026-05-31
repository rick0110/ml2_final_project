#!/usr/bin/env python3
"""Training script for Phase 2: Acoustic Fine-Tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "last-model"))

from last_model_data import create_dataloaders
from losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
    info_nce_loss,
    reconstruction_loss,
)
from model_loader import E2EFlowModel, build_discriminators, get_model_size_info
from text_processing import BatchTextTokenizer
from train_utils import (
    MetricsTracker,
    TensorBoardLogger,
    build_mel_transform,
    create_experiment_dir,
    find_latest_checkpoint,
    load_checkpoint,
    match_length,
    resolve_experiment_dir,
    save_checkpoint,
    save_config,
    set_seed,
)


LAMBDA_RECON = 1.0
LAMBDA_INFONCE = 0.1
LAMBDA_ADV = 1.0
LAMBDA_FM = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 training (Acoustic Fine-Tuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)

    parser.add_argument("--text-model-name", type=str, default="xlm-roberta-base")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--style-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--flow-hidden", type=int, default=192)
    parser.add_argument("--gst-tokens", type=int, default=30)

    return parser.parse_args()


def compute_mel_from_audio(mel_transform, audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    return mel_transform(audio)


def train_one_epoch(
    model: E2EFlowModel,
    discriminators: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    loader,
    tokenizer: BatchTextTokenizer,
    mel_transform,
    device: torch.device,
    logger: TensorBoardLogger,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    discriminators.train()

    tracker = MetricsTracker()

    progress_bar = tqdm(loader, desc="[phase2][train]", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}{postfix}")
    for batch_index, batch in enumerate(progress_bar, start=1):
        texts = batch["text"]
        tokenized = tokenizer.encode_batch_with_attention_mask(texts)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        target_mel = batch["mel"].to(device)
        target_audio = batch["waveform"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, target_mel=target_mel)
        generated_audio = outputs["generated_audio"]
        generated_audio = match_length(generated_audio, target_audio.size(-1))
        target_audio = match_length(target_audio, generated_audio.size(-1))

        predicted_mel = compute_mel_from_audio(mel_transform, generated_audio)
        predicted_mel = match_length(predicted_mel, target_mel.size(-1))
        target_mel = match_length(target_mel, predicted_mel.size(-1))

        style_gen = model.gst(predicted_mel.unsqueeze(1))

        disc_optimizer.zero_grad(set_to_none=True)
        real_mpd, real_msd = discriminators(target_audio)
        fake_mpd, fake_msd = discriminators(generated_audio.detach())
        d_loss = discriminator_loss(real_mpd, fake_mpd) + discriminator_loss(real_msd, fake_msd)
        d_loss.backward()
        if batch_index == 1:
            logger.log_gradient_summary(discriminators, epoch, prefix="phase2/train/discriminator_gradients")
        disc_optimizer.step()

        fake_mpd, fake_msd = discriminators(generated_audio)
        adv_value = generator_adv_loss(fake_mpd) + generator_adv_loss(fake_msd)
        fm_value = feature_matching_loss(real_mpd, fake_mpd) + feature_matching_loss(real_msd, fake_msd)
        recon_value = reconstruction_loss(predicted_mel, target_mel)
        infonce_value = info_nce_loss(outputs["style_ref"], style_gen)

        total_loss = LAMBDA_RECON * recon_value + LAMBDA_INFONCE * infonce_value
        total_loss = total_loss + LAMBDA_ADV * adv_value + LAMBDA_FM * fm_value

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if batch_index == 1:
            batch_summary = [
                f"input_ids.shape={tuple(input_ids.shape)}",
                f"attention_mask.shape={tuple(attention_mask.shape)}",
                f"target_mel.shape={tuple(target_mel.shape)}",
                f"target_audio.shape={tuple(target_audio.shape)}",
                f"generated_audio.shape={tuple(generated_audio.shape)}",
                f"predicted_mel.shape={tuple(predicted_mel.shape)}",
                f"z_post.shape={tuple(outputs['z_post'].shape)}",
                f"z_prior.shape={tuple(outputs['z_prior'].shape)}",
                f"cond.shape={tuple(outputs['cond'].shape)}",
                f"mel_lengths={batch['mel_lengths'].tolist()}",
                f"waveform_lengths={batch['waveform_lengths'].tolist()}",
            ]
            logger.log_text("phase2/train/shapes", "\n".join(batch_summary), epoch)
            logger.log_tensor_report(
                "phase2/train/tensors",
                {
                    "target_mel": target_mel,
                    "target_audio": target_audio,
                    "generated_audio": generated_audio,
                    "predicted_mel": predicted_mel,
                    "style_ref": outputs["style_ref"],
                    "z_post": outputs["z_post"],
                    "z_prior": outputs["z_prior"],
                    "cond": outputs["cond"],
                },
                epoch,
            )
            logger.log_image("phase2/train/target_mel", target_mel[0], epoch)
            logger.log_image("phase2/train/predicted_mel", predicted_mel[0], epoch)
            logger.log_image("phase2/train/mel_abs_diff", (predicted_mel[0] - target_mel[0]).abs(), epoch)
            logger.log_audio("phase2/train/target_audio", target_audio[0], epoch, sample_rate=int(batch["sr"][0].item()))
            logger.log_audio("phase2/train/generated_audio", generated_audio[0], epoch, sample_rate=int(batch["sr"][0].item()))
            logger.log_gradient_summary(model, epoch, prefix="phase2/train/generator_gradients")
        optimizer.step()

        tracker.add(
            loss=total_loss.item(),
            recon=recon_value.item(),
            infonce=infonce_value.item(),
            adv=adv_value.item(),
            fm=fm_value.item(),
        )
        progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

    return tracker.averages()


def validate_one_epoch(
    model: E2EFlowModel,
    loader,
    tokenizer: BatchTextTokenizer,
    mel_transform,
    device: torch.device,
    logger: TensorBoardLogger,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="[phase2][val]", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}{postfix}")
        for batch_index, batch in enumerate(progress_bar, start=1):
            texts = batch["text"]
            tokenized = tokenizer.encode_batch_with_attention_mask(texts)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            target_mel = batch["mel"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, target_mel=target_mel)
            generated_audio = outputs["generated_audio"]
            predicted_mel = compute_mel_from_audio(mel_transform, generated_audio)
            predicted_mel = match_length(predicted_mel, target_mel.size(-1))
            target_mel = match_length(target_mel, predicted_mel.size(-1))

            style_gen = model.gst(predicted_mel.unsqueeze(1))

            recon_value = reconstruction_loss(predicted_mel, target_mel)
            infonce_value = info_nce_loss(outputs["style_ref"], style_gen)
            total_loss = LAMBDA_RECON * recon_value + LAMBDA_INFONCE * infonce_value

            tracker.add(
                loss=total_loss.item(),
                recon=recon_value.item(),
                infonce=infonce_value.item(),
            )
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

            if batch_index == 1:
                batch_summary = [
                    f"input_ids.shape={tuple(input_ids.shape)}",
                    f"attention_mask.shape={tuple(attention_mask.shape)}",
                    f"target_mel.shape={tuple(target_mel.shape)}",
                    f"target_audio.shape={tuple(batch['waveform'].shape)}",
                    f"generated_audio.shape={tuple(generated_audio.shape)}",
                    f"predicted_mel.shape={tuple(predicted_mel.shape)}",
                    f"z_post.shape={tuple(outputs['z_post'].shape)}",
                    f"z_prior.shape={tuple(outputs['z_prior'].shape)}",
                    f"cond.shape={tuple(outputs['cond'].shape)}",
                    f"mel_lengths={batch['mel_lengths'].tolist()}",
                    f"waveform_lengths={batch['waveform_lengths'].tolist()}",
                ]
                logger.log_text("phase2/val/shapes", "\n".join(batch_summary), epoch)
                logger.log_tensor_report(
                    "phase2/val/tensors",
                    {
                        "target_mel": target_mel,
                        "target_audio": batch["waveform"],
                        "generated_audio": generated_audio,
                        "predicted_mel": predicted_mel,
                        "style_ref": outputs["style_ref"],
                        "z_post": outputs["z_post"],
                        "z_prior": outputs["z_prior"],
                        "cond": outputs["cond"],
                    },
                    epoch,
                )
                logger.log_image("phase2/val/target_mel", target_mel[0], epoch)
                logger.log_image("phase2/val/predicted_mel", predicted_mel[0], epoch)
                logger.log_image("phase2/val/mel_abs_diff", (predicted_mel[0] - target_mel[0]).abs(), epoch)
                logger.log_audio("phase2/val/target_audio", batch["waveform"][0], epoch, sample_rate=int(batch["sr"][0].item()))
                logger.log_audio("phase2/val/generated_audio", generated_audio[0], epoch, sample_rate=int(batch["sr"][0].item()))

    return tracker.averages()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BatchTextTokenizer(model_name=args.text_model_name, max_length=args.text_max_length)
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = E2EFlowModel(
        n_mels=80,
        style_dim=args.style_dim,
        latent_dim=args.latent_dim,
        flow_layers=args.flow_layers,
        flow_hidden=args.flow_hidden,
        gst_tokens=args.gst_tokens,
        text_model_name=args.text_model_name,
    ).to(device)

    model.freeze_text_encoder()
    model.unfreeze_vocoder()

    discriminators = build_discriminators().to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        discriminators.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    experiment_dir = create_experiment_dir(args.experiment_name)
    save_config(experiment_dir / "config.json", vars(args))

    logger = TensorBoardLogger(experiment_dir / "tensorboard")
    model_info = get_model_size_info(model)
    logger.log_hyperparameters({**vars(args), **model_info, "train_batches": len(train_loader), "val_batches": len(val_loader)}, {})

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("discriminator_state_dict"):
            discriminators.load_state_dict(checkpoint["discriminator_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("metrics", {}).get("val_loss", best_val))
    elif args.resume_experiment:
        exp_dir = resolve_experiment_dir(args.resume_experiment)
        checkpoint = load_checkpoint(find_latest_checkpoint(exp_dir / "checkpoints"), device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("discriminator_state_dict"):
            discriminators.load_state_dict(checkpoint["discriminator_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("metrics", {}).get("val_loss", best_val))

    mel_transform = build_mel_transform().to(device)

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            discriminators,
            optimizer,
            disc_optimizer,
            train_loader,
            tokenizer,
            mel_transform,
            device,
            logger,
            epoch,
        )
        val_metrics = validate_one_epoch(model, val_loader, tokenizer, mel_transform, device, logger, epoch)

        logger.log_metrics(train_metrics, epoch, prefix="train/")
        logger.log_metrics(val_metrics, epoch, prefix="val/")
        logger.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, epoch, prefix="train/")
        logger.flush()

        checkpoint_path = experiment_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            checkpoint_path,
            epoch,
            model.state_dict(),
            optimizer.state_dict(),
            discriminators.state_dict(),
            {"train_loss": train_metrics.get("loss", 0.0), "val_loss": val_metrics.get("loss", 0.0)},
        )

        if val_metrics.get("loss", float("inf")) < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                experiment_dir / "checkpoints" / "best.pt",
                epoch,
                model.state_dict(),
                optimizer.state_dict(),
                discriminators.state_dict(),
                {"train_loss": train_metrics.get("loss", 0.0), "val_loss": best_val},
            )

    logger.close()


if __name__ == "__main__":
    main()
