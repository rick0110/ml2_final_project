#!/usr/bin/env python3
"""Training script for Phase 1: Flow and Alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import json
import torch
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "last-model"))

from last_model_data import create_dataloaders
from losses import kl_loss, reconstruction_loss
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
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
    parser.add_argument("--gst-lr-multiplier", type=float, default=5.0, help="LR multiplier for GST/style/text projection parameters")
    parser.add_argument("--grad-amplify-factor", type=float, default=1.0, help="Multiply gradients for GST/style/text parameters by this factor (>=1.0)")
    parser.add_argument("--aux-style-loss-steps", type=int, default=0, help="Number of steps to apply auxiliary style loss (0 = disabled)")
    parser.add_argument("--aux-style-loss-weight", type=float, default=0.0, help="Weight for auxiliary style loss when enabled")

    return parser.parse_args()


def train_one_epoch(
    model: E2EFlowModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader,
    tokenizer: BatchTextTokenizer,
    device: torch.device,
    logger: TensorBoardLogger,
    epoch: int,
    grad_clip_norm: float,
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
        # Safety clipping for unstable gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        if batch_index == 1:
            mel_lengths = batch.get("mel_lengths")
            waveform_lengths = batch.get("waveform_lengths")
            waveform = batch["waveform"][0]
            if waveform.dim() == 2:
                waveform = waveform[0]
            if waveform_lengths is not None:
                waveform = waveform[..., : int(waveform_lengths[0].item())]
            summary_lines = [
                f"input_ids.shape={tuple(input_ids.shape)}",
                f"attention_mask.shape={tuple(attention_mask.shape)}",
                f"target_mel.shape={tuple(target_mel.shape)}",
                f"waveform.shape={tuple(waveform.shape)}",
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
                    "waveform": waveform,
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
            logger.log_metrics(
                {
                    "flow_loss": float(loss.item()),
                    "kl_loss": float(loss.item()),
                    "target_mel_mean": float(target_mel.mean().item()),
                    "target_mel_std": float(target_mel.std(unbiased=False).item()),
                    "waveform_mean": float(waveform.mean().item()),
                    "waveform_std": float(waveform.std(unbiased=False).item()) if waveform.numel() > 1 else 0.0,
                    "posterior_latent_mean": float(outputs["z_post"].mean().item()),
                    "posterior_latent_std": float(outputs["z_post"].std(unbiased=False).item()) if outputs["z_post"].numel() > 1 else 0.0,
                    "posterior_mean_mean": float(outputs["post_mean"].mean().item()),
                    "posterior_std_mean": float(outputs["post_log_std"].exp().mean().item()),
                    "prior_latent_mean": float(outputs["z_prior"].mean().item()),
                    "prior_latent_std": float(outputs["z_prior"].std(unbiased=False).item()) if outputs["z_prior"].numel() > 1 else 0.0,
                    "style_ref_mean": float(outputs["style_ref"].mean().item()),
                    "style_ref_std": float(outputs["style_ref"].std(unbiased=False).item()) if outputs["style_ref"].numel() > 1 else 0.0,
                },
                epoch,
                prefix="phase1/train/diagnostics/",
            )
            logger.log_gradient_summary(model, epoch, prefix="phase1/train/gradients")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        tracker.add(loss=loss.item())
        tracker.add(flow_loss=loss.item(), kl_loss=loss.item())

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
            tracker.add(flow_loss=loss.item(), kl_loss=loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_index == 1:
                mel_lengths = batch.get("mel_lengths")
                waveform_lengths = batch.get("waveform_lengths")
                waveform = batch["waveform"][0]
                if waveform.dim() == 2:
                    waveform = waveform[0]
                if waveform_lengths is not None:
                    waveform = waveform[..., : int(waveform_lengths[0].item())]
                summary_lines = [
                    f"input_ids.shape={tuple(input_ids.shape)}",
                    f"attention_mask.shape={tuple(attention_mask.shape)}",
                    f"target_mel.shape={tuple(target_mel.shape)}",
                    f"waveform.shape={tuple(waveform.shape)}",
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
                        "waveform": waveform,
                        "style_ref": outputs["style_ref"],
                        "z_post": outputs["z_post"],
                        "post_mean": outputs["post_mean"],
                        "post_log_std": outputs["post_log_std"],
                        "z_prior": outputs["z_prior"],
                        "cond": outputs["cond"],
                    },
                    epoch,
                )
                logger.log_metrics(
                    {
                        "flow_loss": float(loss.item()),
                        "kl_loss": float(loss.item()),
                        "target_mel_mean": float(target_mel.mean().item()),
                        "target_mel_std": float(target_mel.std(unbiased=False).item()),
                        "waveform_mean": float(waveform.mean().item()),
                        "waveform_std": float(waveform.std(unbiased=False).item()) if waveform.numel() > 1 else 0.0,
                        "posterior_latent_mean": float(outputs["z_post"].mean().item()),
                        "posterior_latent_std": float(outputs["z_post"].std(unbiased=False).item()) if outputs["z_post"].numel() > 1 else 0.0,
                        "posterior_mean_mean": float(outputs["post_mean"].mean().item()),
                        "posterior_std_mean": float(outputs["post_log_std"].exp().mean().item()),
                        "prior_latent_mean": float(outputs["z_prior"].mean().item()),
                        "prior_latent_std": float(outputs["z_prior"].std(unbiased=False).item()) if outputs["z_prior"].numel() > 1 else 0.0,
                        "style_ref_mean": float(outputs["style_ref"].mean().item()),
                        "style_ref_std": float(outputs["style_ref"].std(unbiased=False).item()) if outputs["style_ref"].numel() > 1 else 0.0,
                    },
                    epoch,
                    prefix="phase1/val/diagnostics/",
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
                logger.log_metrics(
                    {
                        "reconstruction_proxy": float(reconstruction_loss(predicted_mel, target_mel_example).item()),
                        "generated_audio_mean": float(generated_audio.mean().item()),
                        "generated_audio_std": float(generated_audio.std(unbiased=False).item()) if generated_audio.numel() > 1 else 0.0,
                    },
                    epoch,
                    prefix="phase1/val/loss/",
                )

    return tracker.averages()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable autograd anomaly detection to catch NaNs/invalid grads early
    torch.autograd.set_detect_anomaly(True)

    tokenizer = BatchTextTokenizer(model_name=args.text_model_name, max_length=args.text_max_length)
    print("[phase1] loading dataloaders", flush=True)
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
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

    # Initialize flows close to identity without making the conditioning path exactly dead.
    model.initialize_identity()
    

    model.freeze_text_encoder()
    model.freeze_vocoder()

    # Build optimizer with a higher LR for GST/style/text projection parameters
    base_params = []
    gst_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if ("gst." in lname) or ("style_proj" in lname) or ("text_proj" in lname) or ("style_tokens" in lname):
            gst_params.append(p)
        else:
            base_params.append(p)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params})
    if gst_params:
        param_groups.append({"params": gst_params, "lr": args.learning_rate * float(args.gst_lr_multiplier)})

    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Optionally amplify gradients for selected parameters by registering hooks
    def _register_grad_amplify_hooks(model_obj, factor: float):
        if factor is None or float(factor) <= 1.0:
            return []
        handles = []
        for name, p in model_obj.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if ("gst." in lname) or ("style_proj" in lname) or ("text_proj" in lname) or ("style_tokens" in lname):
                try:
                    # param.register_hook returns a removable handle function
                    handle = p.register_hook(lambda g, f=float(factor): g * f)
                    handles.append((name, handle))
                except Exception:
                    pass
        return handles

    grad_hook_handles = _register_grad_amplify_hooks(model, args.grad_amplify_factor)

    # Step-based warmup to the target LR over the first few thousand optimizer steps
    warmup_steps = max(0, args.warmup_steps)
    scheduler = None
    if warmup_steps > 0:
        start_factor = min(1.0, 1e-6 / max(args.learning_rate, 1e-12))
        scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)

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

    # Run a one-batch diagnostic pass to inspect gradient flow and log to TensorBoard.
    try:
        train_iter = iter(train_loader)
        diagnostic_batch = next(train_iter)
    except Exception:
        diagnostic_batch = None

    if diagnostic_batch is not None:
        logger.log_text("phase1/diagnostic/info", "running gradient diagnostic (no optimizer step)", 0)
        model.train()
        tokenizer_local = tokenizer
        batch = diagnostic_batch
        texts = batch["text"]
        tokenized = tokenizer_local.encode_batch_with_attention_mask(texts)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        target_mel = batch["mel"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mel=target_mel,
            generate_audio=False,
        )
        diag_loss = kl_loss(
            outputs["z_post"],
            outputs["post_mean"],
            outputs["post_log_std"],
            outputs["z_prior"],
            outputs["log_det"],
        )
        diag_loss.backward()

        # Write detailed gradient summary (will log zeros for missing grads)
        logger.log_gradient_summary(model, 0, prefix="phase1/diagnostic/gradients")

        none_names = []
        zero_names = []
        total_trainable = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            total_trainable += 1
            if p.grad is None:
                none_names.append(name)
            else:
                try:
                    if float(p.grad.detach().float().norm().item()) == 0.0:
                        zero_names.append(name)
                except Exception:
                    pass

        logger.log_text("phase1/diagnostic/none_grad_names", "\n".join(none_names[:500]) or "(none)", 0)
        logger.log_text("phase1/diagnostic/zero_grad_names", "\n".join(zero_names[:500]) or "(none)", 0)
        logger.log_metrics({"num_none_grads": len(none_names), "num_zero_grads": len(zero_names), "total_trainable": total_trainable}, 0, prefix="phase1/diagnostic/")

        # clear gradients from diagnostic run
        optimizer.zero_grad()

        # If not resuming and GST/style_proj have no grads, enable them and rebuild optimizer with a small lr multiplier
        if not (args.resume or args.resume_experiment):
            flagged = [n for n in none_names if ("gst" in n or "style_proj" in n or "style" in n)]
            if flagged:
                # enable requires_grad for GST/style_proj params and rebuild optimizer with higher lr for them
                base_params = []
                gst_params = []
                for name, param in model.named_parameters():
                    # Target GST and style-related parameters for enabling and separate LR
                    if ("gst" in name) or ("style_proj" in name) or ("style_tokens" in name) or ("text_proj" in name):
                        if not param.requires_grad:
                            param.requires_grad = True
                        gst_params.append(param)
                    elif param.requires_grad:
                        # include other trainable parameters in the base group
                        base_params.append(param)
                    else:
                        # leave explicitly frozen parameters (e.g., text encoder backbone, vocoder)
                        continue

                if gst_params:
                    optimizer = torch.optim.AdamW(
                        [
                            {"params": base_params},
                            {"params": gst_params, "lr": args.learning_rate * float(args.gst_lr_multiplier)},
                        ],
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                    )
                    # rebuild scheduler if warmup configured
                    if warmup_steps > 0:
                        start_factor = min(1.0, 1e-6 / max(args.learning_rate, 1e-12))
                        scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps)

                    # re-register amplification hooks if configured
                    try:
                        # remove previous hooks if any
                        if "grad_hook_handles" in locals():
                            for _n, h in grad_hook_handles:
                                try:
                                    h.remove()
                                except Exception:
                                    pass
                        grad_hook_handles = _register_grad_amplify_hooks(model, args.grad_amplify_factor)
                    except Exception:
                        grad_hook_handles = []

                    logger.log_text(
                        "phase1/diagnostic/optimizer_rebuilt",
                        json.dumps({"base_param_tensors": len(base_params), "gst_param_tensors": len(gst_params), "gst_lr": args.learning_rate * float(args.gst_lr_multiplier)}, indent=2),
                        0,
                    )
                else:
                    logger.log_text("phase1/diagnostic/optimizer_rebuilt", "no gst/style params found to adjust", 0)

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"[phase1] epoch {epoch}/{args.num_epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            optimizer,
            scheduler,
            train_loader,
            tokenizer,
            device,
            logger,
            epoch,
            args.grad_clip_norm,
        )
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
