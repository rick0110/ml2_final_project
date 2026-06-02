#!/usr/bin/env python3
"""Generate mel spectrograms from an audio file using a Phase 1 checkpoint.

The checkpointed Phase 1 model is text-conditioned, so this script always
exports the reference mel from the input audio and optionally runs the model
when a transcript is provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = PROJECT_ROOT / "src" / "training" / "training-phase-1"
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_loader import E2EFlowModel  # type: ignore
from text_processing import BatchTextTokenizer  # type: ignore
from train_utils import build_mel_transform, find_latest_checkpoint, load_checkpoint  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a mel spectrogram from an audio file and, optionally, run a Phase 1 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to the input audio file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file or experiment/checkpoints directory")
    parser.add_argument("--output-dir", type=Path, default=Path("exports/mels"), help="Directory where outputs will be written")
    parser.add_argument("--text", type=str, default=None, help="Transcript used by the checkpointed text encoder")
    parser.add_argument("--text-file", type=Path, default=None, help="Optional text file containing the transcript")

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)

    parser.add_argument("--text-model-name", type=str, default="xlm-roberta-base")
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--style-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--flow-hidden", type=int, default=192)
    parser.add_argument("--gst-tokens", type=int, default=30)
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path

    if path.is_dir():
        checkpoint_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        return find_latest_checkpoint(checkpoint_dir)

    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_audio(audio_path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def build_model(args: argparse.Namespace, device: torch.device) -> E2EFlowModel:
    model = E2EFlowModel(
        n_mels=args.n_mels,
        style_dim=args.style_dim,
        latent_dim=args.latent_dim,
        flow_layers=args.flow_layers,
        flow_hidden=args.flow_hidden,
        gst_tokens=args.gst_tokens,
        text_model_name=args.text_model_name,
    )
    return model.to(device).eval()


def load_transcript(args: argparse.Namespace) -> Optional[str]:
    if args.text is not None and args.text.strip():
        return args.text.strip()
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8").strip()
    return None


def save_mel_outputs(output_dir: Path, stem: str, payload: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.pt"
    torch.save(payload, out_path)
    return out_path


def save_mel_comparison_plot(
    output_dir: Path,
    stem: str,
    target_mel: torch.Tensor,
    generated_mel: torch.Tensor,
) -> Optional[Path]:
    if plt is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_target_vs_generated_mel.png"

    target_data = target_mel.detach().float().cpu()
    generated_data = generated_mel.detach().float().cpu()
    min_time = min(target_data.size(-1), generated_data.size(-1))
    target_data = target_data[..., :min_time]
    generated_data = generated_data[..., :min_time]

    # Use a common color scale for target and generated for fair comparison
    try:
        vmin = float(min(float(target_data.min().item()), float(generated_data.min().item())))
        vmax = float(max(float(target_data.max().item()), float(generated_data.max().item())))
    except Exception:
        vmin, vmax = None, None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    plots = [
        (axes[0], target_data, "Target mel"),
        (axes[1], generated_data, "Generated mel"),
    ]

    for axis, data, title in plots:
        if vmin is not None and vmax is not None:
            image = axis.imshow(data.numpy(), origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        else:
            image = axis.imshow(data.numpy(), origin="lower", aspect="auto", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Mel bin")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    # Absolute difference (generated - target)
    try:
        diff = (generated_data - target_data).abs()
        diff_image = axes[2].imshow(diff.numpy(), origin="lower", aspect="auto", interpolation="nearest")
        axes[2].set_title("Absolute difference")
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("Mel bin")
        fig.colorbar(diff_image, ax=axes[2], fraction=0.046, pad=0.04)
    except Exception:
        axes[2].set_visible(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    transcript = load_transcript(args)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)

    waveform, sample_rate = load_audio(args.audio, args.sample_rate)
    mel_transform = build_mel_transform(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    ).to(device)

    with torch.inference_mode():
        reference_mel = mel_transform(waveform.to(device)).squeeze(0).cpu()

        reference_payload: Dict[str, Any] = {
            "kind": "reference",
            "audio_path": str(args.audio),
            "sample_rate": sample_rate,
            "mel": reference_mel,
        }
        reference_path = save_mel_outputs(args.output_dir, f"{args.audio.stem}_reference_mel", reference_payload)

        summary: Dict[str, Any] = {
            "audio_path": str(args.audio),
            "checkpoint_path": str(checkpoint_path),
            "reference_mel_path": str(reference_path),
            "reference_mel_shape": list(reference_mel.shape),
            "sample_rate": sample_rate,
            "checkpoint_epoch": checkpoint.get("epoch"),
        }

        if transcript is not None:
            model = build_model(args, device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)

            tokenizer = BatchTextTokenizer(model_name=args.text_model_name, max_length=args.text_max_length)
            tokenized = tokenizer.encode_batch_with_attention_mask([transcript])
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_mel=reference_mel.unsqueeze(0).to(device),
                generate_audio=True,
            )

            generated_audio = outputs["generated_audio"][0].detach().cpu()
            predicted_mel = mel_transform(generated_audio.to(device)).squeeze(0).cpu()

            predicted_payload = {
                "kind": "predicted",
                "audio_path": str(args.audio),
                "sample_rate": sample_rate,
                "text": transcript,
                "mel": predicted_mel,
                "generated_audio": generated_audio,
            }
            predicted_path = save_mel_outputs(args.output_dir, f"{args.audio.stem}_predicted_mel", predicted_payload)
            comparison_plot_path = save_mel_comparison_plot(args.output_dir, args.audio.stem, reference_mel, predicted_mel)
            summary["predicted_mel_path"] = str(predicted_path)
            summary["predicted_mel_shape"] = list(predicted_mel.shape)
            if comparison_plot_path is not None:
                summary["mel_comparison_plot_path"] = str(comparison_plot_path)
            else:
                summary["warning"] = "matplotlib is not available, so no comparison plot was saved."
        else:
            summary["warning"] = "No transcript provided, so only the reference mel was exported."

        summary_path = args.output_dir / f"{args.audio.stem}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()