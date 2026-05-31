#!/usr/bin/env python3
"""Training script for Phase 1: Flow and Alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "last-model"))

from last_model_data import create_dataloaders
from losses import kl_loss
from model_loader import E2EFlowModel, get_model_size_info
from text_processing import BatchTextTokenizer
from train_utils import (
    MetricsTracker,
    TensorBoardLogger,
    build_mel_transform,
    create_experiment_dir,
    find_latest_checkpoint,
    load_checkpoint,
    resolve_experiment_dir,
    save_checkpoint,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 training (Flow and Alignment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)

    return parser.parse_args()


def train_one_epoch(
    model: E2EFlowModel,
    optimizer: torch.optim.Optimizer,
    loader,
    tokenizer: BatchTextTokenizer,
    device: torch.device,
    logger: TensorBoardLogger,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    tracker = MetricsTracker()

    progress_bar = tqdm(loader, desc="[phase1][train]", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}{postfix}")
    for batch_index, batch in enumerate(progress_bar, start=1):
        texts = batch["text"]
        tokenized = tokenizer.encode_batch_with_attention_mask(texts)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        target_mel = batch["mel"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mel=target_mel,
            generate_audio=False,
        )
        loss = kl_loss(
            outputs["z_post"],
            outputs["post_mean"],
            outputs["post_log_std"],
            outputs["z_prior"],
            outputs["log_det"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if batch_index == 1:
            mel_lengths = batch.get("mel_lengths")
            waveform_lengths = batch.get("waveform_lengths")
            summary_lines = [
                f"input_ids.shape={tuple(input_ids.shape)}",
                f"attention_mask.shape={tuple(attention_mask.shape)}",
                f"target_mel.shape={tuple(target_mel.shape)}",
                f"z_post.shape={tuple(outputs['z_post'].shape)}",
                f"z_prior.shape={tuple(outputs['z_prior'].shape)}",
                f"cond.shape={tuple(outputs['cond'].shape)}",
            ]
            if mel_lengths is not None:
                summary_lines.append(f"mel_lengths={mel_lengths.tolist()}")
            if waveform_lengths is not None:
                summary_lines.append(f"waveform_lengths={waveform_lengths.tolist()}")
            logger.log_text("phase1/train/shapes", "\n".join(summary_lines), epoch)
            logger.log_tensor_report(
                "phase1/train/tensors",
                {
                    "target_mel": target_mel,
                    "style_ref": outputs["style_ref"],
                    "z_post": outputs["z_post"],
                    "post_mean": outputs["post_mean"],
                    "post_log_std": outputs["post_log_std"],
                    "z_prior": outputs["z_prior"],
                    "cond": outputs["cond"],
                },
                epoch,
            )
            logger.log_image("phase1/train/target_mel", target_mel[0], epoch)
            logger.log_gradient_summary(model, epoch, prefix="phase1/train/gradients")
        optimizer.step()

        tracker.add(loss=loss.item())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return tracker.averages()


def validate_one_epoch(
    model: E2EFlowModel,
    loader,
    tokenizer: BatchTextTokenizer,
    device: torch.device,
    logger: TensorBoardLogger,
    mel_transform,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="[phase1][val]", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}{postfix}")
        for batch_index, batch in enumerate(progress_bar, start=1):
            texts = batch["text"]
            tokenized = tokenizer.encode_batch_with_attention_mask(texts)
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            target_mel = batch["mel"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_mel=target_mel,
                generate_audio=False,
            )
            loss = kl_loss(
                outputs["z_post"],
                outputs["post_mean"],
                outputs["post_log_std"],
                outputs["z_prior"],
                outputs["log_det"],
            )
            tracker.add(loss=loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_index == 1:
                mel_lengths = batch.get("mel_lengths")
                waveform_lengths = batch.get("waveform_lengths")
                summary_lines = [
                    f"input_ids.shape={tuple(input_ids.shape)}",
                    f"attention_mask.shape={tuple(attention_mask.shape)}",
                    f"target_mel.shape={tuple(target_mel.shape)}",
                    f"z_post.shape={tuple(outputs['z_post'].shape)}",
                    f"z_prior.shape={tuple(outputs['z_prior'].shape)}",
                    f"cond.shape={tuple(outputs['cond'].shape)}",
                ]
                if mel_lengths is not None:
                    summary_lines.append(f"mel_lengths={mel_lengths.tolist()}")
                if waveform_lengths is not None:
                    summary_lines.append(f"waveform_lengths={waveform_lengths.tolist()}")
                logger.log_text("phase1/val/shapes", "\n".join(summary_lines), epoch)
                logger.log_tensor_report(
                    "phase1/val/tensors",
                    {
                        "target_mel": target_mel,
                        "style_ref": outputs["style_ref"],
                        "z_post": outputs["z_post"],
                        "post_mean": outputs["post_mean"],
                        "post_log_std": outputs["post_log_std"],
                        "z_prior": outputs["z_prior"],
                        "cond": outputs["cond"],
                    },
                    epoch,
                )

                example_output = model(
                    input_ids=input_ids[:1],
                    attention_mask=attention_mask[:1],
                    target_mel=target_mel[:1],
                    generate_audio=True,
                )
                generated_audio = example_output["generated_audio"][0]
                predicted_mel = mel_transform(generated_audio).squeeze(0)
                target_mel_example = target_mel[0]
                min_time = min(predicted_mel.size(-1), target_mel_example.size(-1))
                predicted_mel = predicted_mel[..., :min_time]
                target_mel_example = target_mel_example[..., :min_time]
                logger.log_audio("phase1/val/generated_audio", generated_audio, epoch, sample_rate=int(batch["sr"][0].item()))
                logger.log_image("phase1/val/target_mel", target_mel_example, epoch)
                logger.log_image("phase1/val/predicted_mel", predicted_mel, epoch)
                logger.log_image("phase1/val/mel_abs_diff", (predicted_mel - target_mel_example).abs(), epoch)

    return tracker.averages()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BatchTextTokenizer(model_name=args.text_model_name, max_length=args.text_max_length)
    print("[phase1] loading dataloaders", flush=True)
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    print(f"[phase1] dataloaders ready: train={len(train_loader)} batches val={len(val_loader)} batches", flush=True)

    print("[phase1] loading model", flush=True)
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
    model.freeze_vocoder()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    experiment_dir = create_experiment_dir(args.experiment_name)
    save_config(experiment_dir / "config.json", vars(args))

    logger = TensorBoardLogger(experiment_dir / "tensorboard")
    model_info = get_model_size_info(model)
    logger.log_hyperparameters({**vars(args), **model_info, "train_batches": len(train_loader), "val_batches": len(val_loader)}, {})

    mel_transform = build_mel_transform().to(device)

    print("[phase1] starting training", flush=True)

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("metrics", {}).get("val_loss", best_val))
    elif args.resume_experiment:
        exp_dir = resolve_experiment_dir(args.resume_experiment)
        checkpoint = load_checkpoint(find_latest_checkpoint(exp_dir / "checkpoints"), device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("metrics", {}).get("val_loss", best_val))

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"[phase1] epoch {epoch}/{args.num_epochs}", flush=True)
        train_metrics = train_one_epoch(model, optimizer, train_loader, tokenizer, device, logger, epoch)
        val_metrics = validate_one_epoch(model, val_loader, tokenizer, device, logger, mel_transform, epoch)

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
            None,
            {"train_loss": train_metrics.get("loss", 0.0), "val_loss": val_metrics.get("loss", 0.0)},
        )

        if val_metrics.get("loss", float("inf")) < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                experiment_dir / "checkpoints" / "best.pt",
                epoch,
                model.state_dict(),
                optimizer.state_dict(),
                None,
                {"train_loss": train_metrics.get("loss", 0.0), "val_loss": best_val},
            )

    logger.close()


if __name__ == "__main__":
    main()
